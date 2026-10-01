"""从本地 K 线重建市场环境辅助文件。

`update` 只把每只股票"最新一行"追加进 daily_summary.csv，一旦文件被删或
某天没跑，历史就断档了（实测只剩 1 个交易日），而 sector.py 通篇按"近 20 日"
设计——量比、趋势、持续性三个子指标直接退化成常数。

这里改成"从 K 线反推"：把每只股票最后 N 根 K 线读出来，按日期拼回完整的
日度聚合，顺带重建涨停历史与行业热度历史。只读 K 线，只写自己拥有的
data/market/ 下的文件，绝不回写 K 线。
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .. import store
from .sector import compute_sector_heat
from .sector_fetch import code_to_sector_map
from .snapshot import (
    limit_up_streaks,
    load_heat_history,
    prev_heat_from_history,
    save_heat_history,
)

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ["code", "date", "pct_chg", "high", "close", "amount"]
LIMIT_UP_PCT = 9.5

_UPDATE_ACTIVE_SECONDS = 120  # K线文件这么近被改过，就怀疑有 update 在写


def _list_codes(cfg, codes: list[str] | None = None) -> list[str]:
    if codes:
        return [str(c).strip() for c in codes if str(c).strip()]
    return store.stock_codes(cfg)


def _warn_if_update_running(cfg) -> None:
    """CSV 后端是非原子写，回填时撞上 update 会读到混合日期快照。"""
    if cfg.data.storage != "csv":
        return
    st = store.open_store(cfg, readonly=True)
    try:
        kdir = getattr(st, "kline_dir", None)
        if kdir is None or not kdir.exists():
            return
        now = time.time()
        recent = sum(
            1 for p in kdir.glob("*.csv")
            if now - p.stat().st_mtime < _UPDATE_ACTIVE_SECONDS
        )
    except OSError:
        return
    finally:
        st.close()
    if recent:
        log.warning(
            "有 %d 个K线文件在 %d 秒内被写过，可能正有 update 在跑；"
            "回填会读到混合日期的快照，建议等它结束再跑",
            recent, _UPDATE_ACTIVE_SECONDS,
        )


def _row_from(raw: dict, code: str) -> dict | None:
    # 统一成 'YYYY-MM-DD'：sqlite/json 后端的 date 是 Timestamp，直接写进去会
    # 和别处的字符串日期对不上（daily_summary 全程按字符串比较/排序）
    date = str(raw.get("date") or "").strip()[:10]
    if not date:
        return None
    try:
        return {
            "code": str(code),
            "date": date,
            "pct_chg": float(raw.get("pct_chg")),
            "high": float(raw.get("high")),
            "close": float(raw.get("close")),
            "amount": float(raw.get("amount")),
        }
    except (TypeError, ValueError):
        return None


def _chunk_rows(args: tuple) -> list[dict]:
    """worker：读一批股票的最后 N 根 K 线。worker 内只读打开，绝不写。"""
    cfg, chunk, days = args
    st = store.open_store(cfg, readonly=True)
    kdir = getattr(st, "kline_dir", None)
    out: list[dict] = []
    try:
        for code in chunk:
            try:
                if kdir is not None:
                    rows = store.read_tail_rows_csv(kdir / f"{code}.csv", days)
                else:
                    df = st.read_kline(code, days)
                    # read_kline 的日期在索引上，reset_index 才能拿到 date 列
                    rows = df.reset_index().to_dict("records") if not df.empty else []
            except Exception as exc:  # noqa: BLE001 - 单只坏文件不该中断整体回填
                log.warning("回填读取 %s 失败: %s", code, exc)
                continue
            for raw in rows:
                rec = _row_from(raw, code)
                if rec:
                    out.append(rec)
    finally:
        st.close()
    return out


def _build_summary(rows: list[dict], days: int) -> pd.DataFrame:
    """去重 → 排序 → 只留最后 days 个交易日。"""
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset=["code", "date"], keep="last")
    keep = sorted(df["date"].unique())[-days:]
    df = df[df["date"].isin(keep)]
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def _limit_up_tables(summary: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, int]]:
    """按日期升序重建 ({日期: [涨停代码]}, 最后一天的 {代码: 连板数})。"""
    history: dict[str, list[str]] = {}
    for day, grp in summary.groupby("date", sort=True):
        history[str(day)] = sorted(grp.loc[grp["pct_chg"] >= LIMIT_UP_PCT, "code"].astype(str))
    return history, limit_up_streaks(history)


def _seed_heat_history(cfg, summary: pd.DataFrame, weights: dict) -> int:
    """逐日算行业热度并落盘，让阶段判定（需要分数轨迹）当场可用。

    注意 history 必须按天截断到"当日为止"：`_sector_metrics` 里的 5d/20d
    走的是 `tail(5)/tail(20)`，如果整天整段传进去，历史日的窗口会取到数据
    末尾（未来函数），60 天的分数会被抹平成一条直线，轨迹也就没意义了。
    """
    c2s = code_to_sector_map(cfg)
    if not c2s:
        log.warning("行业分类为空，跳过行业热度历史预热（可先跑 `python3 cli.py fetch-sectors`）")
        return 0
    history = load_heat_history(cfg)
    marked = 0
    for day, chunk in summary.groupby("date", sort=True):
        day = str(day)
        hist_upto = summary[summary["date"] <= day]
        result = compute_sector_heat(
            chunk, hist_upto, c2s, prev_heat_from_history(history, day), weights,
        )
        if result:
            # 记录走的是"当日之前"的历史，所以同一天重复回填结果一致
            history[day] = {k: v["score"] for k, v in result.items()}
            marked += 1
    save_heat_history(cfg, history)
    return marked


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def rebuild_market_data(
    cfg, days: int | None = None, workers: int | None = None,
    codes: list[str] | None = None,
) -> dict:
    """重建 daily_summary.csv / limit_up_*.json / sector_heat_history.json。

    以本地 K 线为准，不与现有 daily_summary 合并——那份文件可能本身就是残缺的。
    """
    t0 = time.time()
    days = max(2, int(days or cfg.market.history_days))
    mkt = cfg.market
    _warn_if_update_running(cfg)

    pool = _list_codes(cfg, codes)
    if not pool:
        log.warning("没有可回填的股票（本地库为空？先跑 update）")
        return {"codes": 0, "days": 0, "rows": 0, "heat_days": 0, "elapsed": 0.0}

    workers = max(1, int(workers or cfg.data.max_workers))
    chunks = [c for c in (pool[i::workers] for i in range(workers)) if c]
    tasks = [(cfg, chunk, days) for chunk in chunks]

    rows: list[dict] = []
    if len(tasks) == 1:
        for task in tqdm(tasks, desc="回填", unit="块"):
            rows.extend(_chunk_rows(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool_exec:
            for part in tqdm(
                pool_exec.map(_chunk_rows, tasks), total=len(tasks), desc="回填", unit="块"
            ):
                rows.extend(part)

    summary = _build_summary(rows, days)
    if summary.empty:
        log.warning("没有读到任何 K 线，data/market 下的文件未改动")
        return {"codes": 0, "days": 0, "rows": 0, "heat_days": 0, "elapsed": time.time() - t0}

    market_dir = store.market_dir(cfg)
    _atomic_write_csv(summary, market_dir / "daily_summary.csv")

    limit_history, streaks = _limit_up_tables(summary)
    _atomic_write_json(limit_history, market_dir / "limit_up_history.json")
    _atomic_write_json(streaks, market_dir / "limit_up_streak.json")

    weights = dict(mkt.sector_weights) if hasattr(mkt.sector_weights, "items") else mkt.sector_weights
    try:
        heat_days = _seed_heat_history(cfg, summary, weights)
    except Exception as exc:  # noqa: BLE001 - 热度历史预热失败不该丢掉已写好的日度聚合
        log.warning("行业热度历史预热失败: %s", exc)
        heat_days = 0

    n_days = int(summary["date"].nunique())
    n_codes = int(summary["code"].nunique())
    elapsed = time.time() - t0
    log.info(
        "rebuild-market: %d 只, %d 个交易日, %d 行, 热度历史 %d 天, 用时 %.1fs",
        n_codes, n_days, len(summary), heat_days, elapsed,
    )
    return {
        "codes": n_codes,
        "days": n_days,
        "rows": len(summary),
        "heat_days": heat_days,
        "elapsed": elapsed,
        "range": (str(summary["date"].min()), str(summary["date"].max())),
    }
