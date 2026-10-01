"""全量 / 增量更新编排。

线程池只负责网络抓取，所有存储写入回到主线程串行执行，
避免并发写同一后端（sqlite 连接不能跨线程，文件后端避免写冲突）。

增量更新一律带重叠区间抓取：既用于复权漂移检测，也顺带修正数据源的历史修订。
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import fetch, store, universe
from .market.snapshot import limit_up_streaks, parse_limit_up_history

log = logging.getLogger(__name__)


def _today_str() -> str:
    return date.today().strftime("%Y%m%d")


def _after_market_close() -> bool:
    """判断当前时间是否在A股收盘后（15:00之后）。"""
    return datetime.now().time() >= time(15, 0)


def _overlap_start(last: str, bars: int) -> str:
    """从库中最后日期往前回退若干个交易日（按 1.6 倍日历日近似）。"""
    last_dt = pd.Timestamp(last).date()
    return (last_dt - timedelta(days=int(bars * 1.6) + 5)).strftime("%Y%m%d")


def _fetch_batch(
    tasks: list[tuple[str, str]], cfg, desc: str
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """并发抓取 [(code, start_date), ...]，返回 {code: df} 与失败列表。"""
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    end = _today_str()

    with ThreadPoolExecutor(max_workers=cfg.data.max_workers) as pool:
        futures = {
            pool.submit(fetch.fetch_kline, code, start, end, cfg): code
            for code, start in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="只"):
            code = futures[future]
            try:
                df, channel = future.result()
                results[code] = df
                if df.empty:
                    log.info("[%s] 最终结果: 无数据（所有渠道确认）", code)
                else:
                    log.info("[%s] 最终结果: 成功 ✓ 渠道=%s, K线=%d根", code, channel, len(df))
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] 最终结果: 失败 ✗ %s", code, exc)
                failed.append(code)
    return results, failed


def _update_index_klines(cfg, st) -> dict[str, bool]:
    """增量更新 3 只指数的日K 线，存入同一 store。返回 {name: 是否成功}。"""
    index_codes = cfg.market.index_codes
    today = date.today().strftime("%Y-%m-%d")
    end = _today_str()
    status: dict[str, bool] = {}

    for name, idx_code in index_codes.items():
        last = st.last_date(idx_code)
        if last is not None and last >= today:
            log.info("指数 %s (%s) 已有当日数据，跳过", name, idx_code)
            status[name] = True
            continue
        start = cfg.data.history_start if last is None else _overlap_start(
            last, cfg.data.drift_check_bars
        )
        try:
            df, channel = fetch.fetch_index_kline(idx_code, start, end, cfg)
            if df.empty:
                log.warning("指数 %s (%s) 无数据", name, idx_code)
                status[name] = False
                continue
            n = st.upsert_kline(idx_code, df)
            log.info("指数 %s: 写入 %d 根K线 (渠道=%s)", name, n, channel)
            status[name] = True
        except Exception as exc:  # noqa: BLE001
            log.warning("指数 %s (%s) 更新失败: %s", name, idx_code, exc)
            status[name] = False
    return status


def _read_last_row_csv(path: Path) -> dict[str, str] | None:
    """快速读取 CSV 文件最后一行（不加载整个文件）。"""
    rows = store.read_tail_rows_csv(path, 1)
    return rows[-1] if rows else None


def _num(value) -> float:
    """转 float，缺失/坏值一律给 NaN——不要因为一个空字段丢掉整只股票。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, payload) -> None:
    """原子写：先写 .tmp 再 os.replace，避免并发读到半个文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


SUMMARY_MAX_DAYS = 60


def _write_daily_summary(cfg, st, pool_codes: list[str]) -> None:
    """从每只股票最新一行 K 线聚合为 daily_summary.csv，保留 60 个交易日。

    同时更新涨停跟踪文件（limit_up_history.json + limit_up_streak.json）。
    合并为单次遍历以提升性能。

    daily_summary 历史断档时（文件被删或某天没跑）用 `cli.py rebuild-market`
    从 K 线整段重建，这里只负责往前追加一天。
    """
    market_dir = store.market_dir(cfg)
    summary_path = market_dir / "daily_summary.csv"
    hist_path = market_dir / "limit_up_history.json"
    streak_path = market_dir / "limit_up_streak.json"

    is_csv_store = cfg.data.storage == "csv"
    rows: list[dict] = []
    today_codes: list[str] = []

    for code in pool_codes:
        try:
            if is_csv_store and hasattr(st, "kline_dir"):
                last = _read_last_row_csv(st.kline_dir / f"{code}.csv")
                if not last:
                    continue
                date_val = last.get("date", "")
            else:
                df = st.read_kline(code, limit_bars=1)
                if df.empty:
                    continue
                last = df.iloc[-1]
                # read_kline 把日期放在索引上（没有 date 列），这里取索引值
                date_val = df.index[-1]
            row = {
                "code": str(code),
                "date": "" if pd.isna(date_val) else str(date_val)[:10],
                "pct_chg": _num(last.get("pct_chg")),
                "high": _num(last.get("high")),
                "close": _num(last.get("close")),
                "amount": _num(last.get("amount")),
            }
        except Exception:
            continue
        # 没有收盘价 / 涨跌幅的行没法参与任何指标；high/amount 缺失给 NaN 即可
        if not row["date"] or pd.isna(row["close"]) or pd.isna(row["pct_chg"]):
            continue
        rows.append(row)
        if row["pct_chg"] >= 9.5:
            today_codes.append(str(code))

    if not rows:
        return

    new_df = pd.DataFrame(rows)
    today_date = str(new_df["date"].max())
    new_df = new_df[new_df["date"] == today_date]

    if summary_path.exists():
        old = pd.read_csv(summary_path, dtype={"code": str})
        old = old[old["date"] < today_date]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df

    dates = sorted(combined["date"].unique())
    if len(dates) > SUMMARY_MAX_DAYS:
        keep = set(dates[-SUMMARY_MAX_DAYS:])
        combined = combined[combined["date"].isin(keep)]

    combined.to_csv(summary_path, index=False)
    log.info("daily_summary: %d 只 (%s), 累计 %d 个交易日", len(new_df), today_date, combined["date"].nunique())

    # 涨停历史按【数据日】存多日：sentiment 的 limit_follow_through 要的是上一交易日的
    # 名单，只存当日等于没存（旧版当日覆盖当日 → 该项常年在拿自己跟自己比，恒给满分）。
    history = parse_limit_up_history(_read_json(hist_path, {}))
    history[today_date] = sorted(today_codes)
    for day in sorted(history)[:-SUMMARY_MAX_DAYS]:
        history.pop(day, None)
    _write_json(hist_path, history)

    streaks = limit_up_streaks(history)
    _write_json(streak_path, streaks)
    log.info("涨停跟踪: %s %d 只, 连板 %d 只", today_date, len(today_codes), len(streaks))


def update_index_klines(cfg) -> dict:
    """独立入口：拉取指数 K 线 + 从现有数据生成日度聚合/涨停跟踪。

    用于首次建仓或数据缺失时单独补全市场环境所需的辅助文件，
    不必重跑完整 update。返回统计字典。
    """
    st = store.open_store(cfg)
    try:
        status = _update_index_klines(cfg, st)
        ok = sum(1 for v in status.values() if v)
        fail = sum(1 for v in status.values() if not v)
        log.info("指数更新完成: 成功 %d, 失败 %d", ok, fail)

        if cfg.data.storage == "csv" and hasattr(st, "kline_dir"):
            pool_codes = [
                p.stem for p in st.kline_dir.glob("*.csv")
                if not p.stem.startswith("idx_")
            ]
        else:
            pool_codes = [c for c in st.bar_counts() if not c.startswith("idx_")]
        _write_daily_summary(cfg, st, pool_codes)
        return {"index_ok": ok, "index_fail": fail}
    finally:
        st.close()


def run_update(
    cfg,
    full: bool = False,
    limit: int | None = None,
    codes: list[str] | None = None,
    retry_failed: bool = False,
) -> dict:
    """执行一轮更新，返回统计字典。"""
    st = store.open_store(cfg)

    out_dir = Path(cfg.data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = out_dir / "failed_codes.txt"

    if retry_failed:
        if not failed_path.exists():
            log.info("没有失败清单，无需重跑")
            return {"total": 0}
        codes = [c.strip() for c in failed_path.read_text().split() if c.strip()]
        log.info("重跑失败清单 %d 只", len(codes))

    if codes:
        pool_codes = list(dict.fromkeys(codes))
    else:
        log.info("正在获取全市场快照...")
        spot = fetch.fetch_spot(cfg)
        log.info("获取快照完成，共 %d 只", len(spot))
        pool_df = universe.filter_spot(spot, cfg)
        st.upsert_stock_info(pool_df, update_time=False)
        pool_codes = pool_df["code"].tolist()

    if limit:
        pool_codes = pool_codes[:limit]

    today = date.today().strftime("%Y-%m-%d")
    market_closed = _after_market_close()
    log.info(
        "今日 %s，当前%s",
        today,
        "已收盘（15:00后）" if market_closed else "未收盘（盘中或盘前）",
    )
    tasks: list[tuple[str, str]] = []
    skipped = 0
    for code in pool_codes:
        last = st.last_date(code)
        if last is not None and last >= today and market_closed:
            skipped += 1
            log.debug("[%s] 本地已有当日数据 (%s) 且已收盘，跳过", code, last)
            continue
        if last is not None and last >= today and not market_closed:
            log.debug("[%s] 本地有当日数据 (%s) 但未收盘，仍拉取更新", code, last)
        if full or last is None:
            tasks.append((code, cfg.data.history_start))
        else:
            tasks.append((code, _overlap_start(last, cfg.data.drift_check_bars)))

    log.info(
        "待更新 %d 只（%s），跳过 %d 只",
        len(tasks),
        "全量" if full else "增量",
        skipped,
    )

    results, failed = _fetch_batch(tasks, cfg, "全量拉取" if full else "增量更新")

    empty: list[str] = []
    drifted: list[str] = []
    written = 0
    for i, (code, df) in enumerate(results.items(), 1):
        if df.empty:
            empty.append(code)
            log.info("[%s] 跳过写入：数据为空", code)
            continue
        if not full and st.detect_adjust_drift(code, df, cfg.data.drift_tolerance):
            drifted.append(code)
            log.info("[%s] 检测到复权漂移，跳过写入，等待全量重拉", code)
            continue
        try:
            n = st.upsert_kline(code, df)
            written += n
            st.update_stock_time(code)
            log.info("[%s] 写入成功，%d 根K线 -> %s", code, n, st.kline_path(code))
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] 写入失败: %s", code, exc)
            failed.append(code)
        if i % 200 == 0:
            st.commit()
    st.commit()

    # 发生除权的股票：前复权历史全变了，必须整段重拉
    if drifted:
        log.info("检测到 %d 只发生除权（前复权历史漂移），重新全量拉取", len(drifted))
        refetch, refetch_failed = _fetch_batch(
            [(c, cfg.data.history_start) for c in drifted], cfg, "除权重拉"
        )
        failed.extend(refetch_failed)
        for code, df in refetch.items():
            if df.empty:
                empty.append(code)
                log.info("[%s] 除权重拉完成，数据为空", code)
                continue
            try:
                st.delete_code(code)
                n = st.upsert_kline(code, df)
                written += n
                st.update_stock_time(code)
                log.info("[%s] 除权全量重写成功，%d 根K线 -> %s", code, n, st.kline_path(code))
            except Exception as exc:  # noqa: BLE001
                log.error("[%s] 除权全量重写失败: %s", code, exc)
                failed.append(code)
        st.commit()

    st.set_meta("last_full_update" if full else "last_inc_update", today)

    problem = sorted(set(failed) | set(empty))
    if problem:
        failed_path.write_text("\n".join(problem), encoding="utf-8")
    elif failed_path.exists():
        failed_path.unlink()

    _update_index_klines(cfg, st)
    _write_daily_summary(cfg, st, pool_codes)

    st.close()

    stats = {
        "total": len(pool_codes),
        "fetched": len(results),
        "written_rows": written,
        "skipped": skipped,
        "drifted": len(drifted),
        "failed": len(failed),
        "empty": len(empty),
    }
    log.info(
        "更新完成: 股票池 %(total)d, 抓取 %(fetched)d, 写入 %(written_rows)d 行, "
        "跳过 %(skipped)d, 除权重拉 %(drifted)d, 失败 %(failed)d, 空数据 %(empty)d",
        stats,
    )
    if problem:
        log.info("失败/空数据清单已写入 %s，可用 --retry-failed 重跑", failed_path)
    return stats
