"""扫描编排：全市场多进程扫描 + 单股调试扫描。

每个 worker 自己打开只读存储后端（连接/句柄不能跨进程共享），
只读取形态需要的最近若干根K线，避免把全部历史搬进内存。
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from . import indicators, store, swings, universe
from .patterns import DETECTORS, PATTERN_KEYS, PatternHit, ScanContext
from .patterns.base import (
    INVALIDATION_RULES,
    STATUS_CN,
    STATUS_INVALID,
    _CONFLICT_MAP,
)

log = logging.getLogger(__name__)


def build_intraday_df(
    hist: pd.DataFrame, quote: dict, today: pd.Timestamp | None = None
) -> pd.DataFrame:
    """把最新报价合成今日 bar，替换或追加到历史 K 线末端。"""
    if hist is None or hist.empty:
        return hist

    today = pd.Timestamp(today or date.today()).normalize()
    price = float(quote["price"])
    prev_close = float(hist["close"].iloc[-1])
    open_ = float(quote.get("open") or 0) or price
    high = max(float(quote.get("high") or 0), price, open_)
    lows = [v for v in (quote.get("low"), price, open_) if v and v > 0]
    low = min(lows) if lows else price
    bar = {
        "open": open_,
        "close": price,
        "high": high,
        "low": low,
        "volume": float(quote.get("volume") or 0),
        "amount": float(quote.get("amount") or 0),
        "turnover": float("nan"),
        "pct_chg": (price - prev_close) / prev_close * 100 if prev_close else 0.0,
    }

    out = hist.copy()
    if out.index[-1].normalize() == today:
        prev_close = float(out["close"].iloc[-2]) if len(out) > 1 else prev_close
        bar["pct_chg"] = (price - prev_close) / prev_close * 100 if prev_close else 0.0
        for key, value in bar.items():
            if key in out.columns:
                out.iloc[-1, out.columns.get_loc(key)] = value
        return out

    row = pd.DataFrame([bar], index=pd.DatetimeIndex([today], name=out.index.name))
    return pd.concat([out, row[[c for c in out.columns if c in row.columns]]])


def required_bars(cfg) -> int:
    """形态回看所需的最大K线数，用于限制单股读取量。"""
    # 破底翻：破底新低回看窗口 + 前低到新低的最大间隔 + 衡量前置跌幅的回看
    bb = cfg.break_bottom
    bb_bars = bb.break_within + bb.max_span + bb.prior_lookback
    b2_bars = cfg.break_bottom_b2.break_within_b2 + bb.max_span + bb.prior_lookback
    # 破底翻v2：L0 回看 + 前置结构回看，以及 TP2 的 120 日高点 / MA120
    v2 = cfg.break_bottom_v2
    v2_bars = max(
        v2.l0_lookback + v2.prior_lookback,
        v2.tp2_lookback,
        cfg.break_bottom_v2_b2.break_within_v2 + v2.prior_lookback,
    )
    return (
        max(
            max(cfg.channel.windows),
            bb_bars,
            b2_bars,
            v2_bars,
            cfg.w_bottom.max_width,
            cfg.m_top.max_width,
            cfg.breakout.cup_window,
            cfg.breakout.rounding_window,
            cfg.breakout.consolidation_window,
            cfg.trend_pullback.lookback,
        )
        + 80
    )


def resolve_detectors(keys: Iterable[str] | None) -> dict:
    """把 CLI 传的 key 列表解析成检测器字典。也接受 pattern 名（如 rising_channel）。"""
    if not keys:
        return dict(DETECTORS)
    pattern_to_detector = {
        pat: det for det, pats in PATTERN_KEYS.items() for pat in pats
    }
    out = {}
    for key in keys:
        key = key.strip()
        if key in DETECTORS:
            out[key] = DETECTORS[key]
        elif key in pattern_to_detector:
            det = pattern_to_detector[key]
            out[det] = DETECTORS[det]
        else:
            raise ValueError(
                f"未知形态 '{key}'，可选: {', '.join(DETECTORS)} "
                f"或 {', '.join(pattern_to_detector)}"
            )
    return out


def build_context(
    code: str, name: str, df: pd.DataFrame, cfg, explain: bool = False,
    market_snapshot=None,
) -> ScanContext:
    enriched = indicators.compute_indicators(df, cfg)
    pivots = swings.zigzag(enriched, cfg)
    return ScanContext(
        code=code, name=name, df=enriched, pivots=pivots, cfg=cfg, explain=explain,
        market_snapshot=market_snapshot,
    )


def scan_context(ctx: ScanContext, detectors: dict) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for key, module in detectors.items():
        try:
            hits.extend(module.detect(ctx))
        except Exception as exc:  # noqa: BLE001 - 单股单形态异常不应中断全局
            log.warning("%s 检测 %s 失败: %s", ctx.code, key, exc)
    # 阶段折扣在这里统一盖上，而不是让每个检测器自己记得传：漏一处就会
    # 悄悄退回默认值，改 config 对那个形态无效。这里是所有 hit 的必经之路。
    discount = float(ctx.cfg.scoring.forming_discount)
    regime = ctx.market_regime
    scan_date = ctx.df.index[-1] if not ctx.df.empty else None
    half_life = int(ctx.cfg.scoring.time_decay_half_life_days)
    decay_floor = float(ctx.cfg.scoring.time_decay_floor)
    for hit in hits:
        hit.forming_discount = discount
        hit.market_regime = regime
        if ctx.market_snapshot is not None:
            snap = ctx.market_snapshot
            hit.market_snapshot = snap
            hit.sector_name = snap.code_to_sector.get(str(hit.code), "")
            hit.sector_stage = snap.sector_stages.get(hit.sector_name, "")
        if scan_date is not None and hit.status == "formed":
            try:
                age = (scan_date - hit.signal_date).days
                hit.time_decay = max(decay_floor, 0.5 ** (age / half_life)) if age > 0 else 1.0
            except Exception:  # noqa: BLE001
                hit.time_decay = 1.0
    if ctx.cfg.scoring.invalidation_enabled:
        hits = apply_invalidation(hits)
    if ctx.cfg.scoring.dedup_enabled:
        hits = dedup_hits(hits)
    return hits


def apply_invalidation(hits: list[PatternHit]) -> list[PatternHit]:
    """后处理：检查已形成/构筑中的形态是否已被价格行为证伪。

    v2 口径自己管理失效（emit_invalid），这里不重复判定。
    只标记不移除——报表/观察池靠 pattern_invalid 列过滤。
    """
    for hit in hits:
        if hit.pattern_invalid:
            continue
        rule = INVALIDATION_RULES.get(hit.pattern)
        if rule is None:
            continue
        key_level = hit.key_levels.get(rule["key"])
        if key_level is None or key_level <= 0 or hit.close_price <= 0:
            continue
        triggered = False
        if rule["side"] == "long" and hit.close_price < key_level:
            triggered = True
        elif rule["side"] == "short" and hit.close_price > key_level:
            triggered = True
        if triggered:
            hit.pattern_invalid = True
            hit.invalid_reason = f"close({hit.close_price:.2f}) {'<' if rule['side'] == 'long' else '>'} {rule['key']}({key_level:.2f})"
            hit.status = STATUS_INVALID
    return hits


def dedup_hits(hits: list[PatternHit]) -> list[PatternHit]:
    """同股形态去重：互斥组内只保留综合评分最高的一个。

    非冲突组的形态全部保留。评分相同时保留信号日期更近的。
    """
    from collections import defaultdict
    by_code: dict[str, list[PatternHit]] = defaultdict(list)
    for hit in hits:
        by_code[hit.code].append(hit)

    result: list[PatternHit] = []
    for code, code_hits in by_code.items():
        groups: dict[int, list[PatternHit]] = defaultdict(list)
        independent: list[PatternHit] = []
        for hit in code_hits:
            gid = _CONFLICT_MAP.get(hit.pattern)
            if gid is not None:
                groups[gid].append(hit)
            else:
                independent.append(hit)
        for gid, group_hits in groups.items():
            best = max(
                group_hits,
                key=lambda h: (h.composite_score(), h.signal_date),
            )
            result.append(best)
        result.extend(independent)
    return result


def _scan_chunk(args) -> tuple[list[dict], int]:
    """扫一块股票，返回 (命中行, 因K线不足被跳过的只数)。

    min_bars 在这里判定，而不是扫描前用全市场 bar_counts() 预筛：csv 后端的
    bar_counts() 要逐文件数行（实测 ~112s/次），是每次扫描最大的固定开销。
    只要 limit_bars >= min_bars，read_kline 返回的是 min(limit_bars, 历史根数)，
    因此 `len(df) < min_bars` 与"历史根数 < min_bars"完全等价。
    """
    market_snapshot = None
    if len(args) == 6:
        cfg, codes, detector_keys, names, limit_bars, market_snapshot = args
    else:
        cfg, codes, detector_keys, names, limit_bars = args
    min_bars = int(cfg.universe.min_bars)
    detectors = resolve_detectors(detector_keys)
    st = store.open_store(cfg, readonly=True)
    rows: list[dict] = []
    short = 0
    try:
        for code in codes:
            try:
                df = st.read_kline(code, limit_bars)
                if df.empty or len(df) < min_bars:
                    short += 1
                    continue
                ctx = build_context(
                    code, names.get(code, code), df, cfg,
                    market_snapshot=market_snapshot,
                )
                for hit in scan_context(ctx, detectors):
                    rows.append(hit.to_row())
            except Exception as exc:  # noqa: BLE001
                log.warning("扫描 %s 失败: %s", code, exc)
    finally:
        st.close()
    return rows, short


def scan_market(
    cfg,
    codes: list[str] | None = None,
    detector_keys: list[str] | None = None,
    statuses: list[str] | None = None,
    min_score: float | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """全市场扫描，返回命中表。

    排序逻辑：按综合评分降序。
    过滤逻辑：根据 universe 配置过滤股票。
    """
    with store.open_store(cfg, readonly=True) as st:
        # 用 all_codes()（列目录 / 索引扫描，实测 ~27ms）取代 bar_counts()
        # （csv 后端要逐文件数行，实测 ~112s）。min_bars 过滤随之下沉到
        # _scan_chunk 读完之后判定，全市场每轮省掉一百多秒的固定开销。
        all_codes = st.all_codes()
        names = st.get_stock_names()
        stock_info = st.get_stock_info()

    # 根据 universe 配置过滤股票
    u = cfg.universe
    eligible_codes = set(all_codes)

    # 如果有 stock_info，应用更多过滤
    if not stock_info.empty:
        info_map = dict(zip(stock_info["code"], stock_info.to_dict("records")))

        # ST 过滤
        if u.exclude_st:
            before = len(eligible_codes)
            eligible_codes = {
                c for c in eligible_codes
                if c not in info_map or "ST" not in str(info_map[c].get("name", "")).upper()
            }
            skipped_st = before - len(eligible_codes)
        else:
            skipped_st = 0

        # 科创板过滤
        if u.exclude_star_market:
            before = len(eligible_codes)
            eligible_codes = {c for c in eligible_codes if not c.startswith("688")}
            skipped_star = before - len(eligible_codes)
        else:
            skipped_star = 0

        # 北交所过滤
        if u.exclude_bj:
            before = len(eligible_codes)
            eligible_codes = {c for c in eligible_codes if not c.startswith(("4", "8", "9"))}
            skipped_bj = before - len(eligible_codes)
        else:
            skipped_bj = 0

        # 流通市值过滤
        if u.min_float_mv:
            before = len(eligible_codes)
            eligible_codes = {
                c for c in eligible_codes
                if c not in info_map or (info_map[c].get("float_mv") or 0) >= u.min_float_mv
            }
            skipped_mv = before - len(eligible_codes)
        else:
            skipped_mv = 0

        # 成交额过滤
        if u.min_amount:
            before = len(eligible_codes)
            eligible_codes = {
                c for c in eligible_codes
                if c not in info_map or (info_map[c].get("amount") or 0) >= u.min_amount
            }
            skipped_amount = before - len(eligible_codes)
        else:
            skipped_amount = 0

        # 股价过滤
        if u.max_price:
            before = len(eligible_codes)
            eligible_codes = {
                c for c in eligible_codes
                if c not in info_map or 0 < (info_map[c].get("price") or 0) <= u.max_price
            }
            skipped_price = before - len(eligible_codes)
        else:
            skipped_price = 0
    else:
        skipped_st = skipped_star = skipped_bj = skipped_mv = skipped_amount = skipped_price = 0

    # 如果指定了 codes，取交集
    if codes:
        eligible_codes = eligible_codes & set(codes)

    pool = sorted(eligible_codes)
    if not pool:
        log.info("股票池为空（ST %d, 科创板 %d, 北交所 %d, 市值不足 %d, 成交额不足 %d, 股价超限 %d）",
                 skipped_st, skipped_star, skipped_bj, skipped_mv, skipped_amount, skipped_price)
        return pd.DataFrame()

    workers = workers or max(1, (os.cpu_count() or 2))
    # limit_bars 必须 >= min_bars，否则判断"K线不足"会不等价于历史根数不足
    limit_bars = max(required_bars(cfg), int(u.min_bars))

    # 市场环境快照：扫描前预计算一次，所有 worker 共享
    market_snapshot = None
    if hasattr(cfg, "market") and cfg.market.enabled:
        try:
            from .market import compute_market_snapshot
            market_snapshot = compute_market_snapshot(cfg)
            if market_snapshot is not None:
                log.info(
                    "市场环境: %s (%.1f) | 情绪: %.1f | 行业: %d 个",
                    market_snapshot.regime_label, market_snapshot.regime_score,
                    market_snapshot.sentiment_score, len(market_snapshot.sector_scores),
                )
            else:
                log.warning("市场环境快照为 None（指数数据缺失），跳过环境修正")
        except Exception as exc:
            log.warning("计算市场环境快照失败，跳过环境修正: %s", exc)

    chunks = [pool[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    tasks = [
        # cfg 是 dict 子类，可直接 pickle 传给子进程；后端由 worker 自行打开
        (cfg, chunk, detector_keys, names, limit_bars, market_snapshot)
        for chunk in chunks
    ]

    rows: list[dict] = []
    skipped_bars = 0  # K线不足的只数由 worker 读完后统计（见 _scan_chunk）
    if workers == 1:
        for task in tqdm(tasks, desc="扫描", unit="块"):
            chunk_rows, short = _scan_chunk(task)
            rows.extend(chunk_rows)
            skipped_bars += short
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool_exec:
            for chunk_rows, short in tqdm(
                pool_exec.map(_scan_chunk, tasks), total=len(tasks), desc="扫描", unit="块"
            ):
                rows.extend(chunk_rows)
                skipped_bars += short

    log.info(
        "扫描 %d 只（跳过: K线不足 %d, ST %d, 科创板 %d, 北交所 %d, 市值不足 %d, 成交额不足 %d, 股价超限 %d）",
        len(pool),
        skipped_bars,
        skipped_st,
        skipped_star,
        skipped_bj,
        skipped_mv,
        skipped_amount,
        skipped_price,
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Pattern-specific key levels are inserted in detector order.  That order
    # differs between process chunks, so normalize columns before returning to
    # make single- and multi-process scans byte-for-byte comparable.
    base_columns = [
        "代码", "名称", "形态", "形态key", "方向", "状态", "最新价", "综合评分",
        "机会质量", "收益质量", "阶段确定性", "风险修正", "信号日期",
        "预期收益%", "风险收益比", "说明",
    ]
    extra_columns = sorted(col for col in df.columns if col not in base_columns)
    df = df.reindex(columns=base_columns + extra_columns)
    if statuses:
        wanted = {"formed": "已形成", "forming": "构筑中", "invalid": "已失效"}
        keep = {wanted.get(s, s) for s in statuses}
        df = df[df["状态"].isin(keep)]
    threshold = cfg.scoring.min_score if min_score is None else min_score
    if threshold:
        df = df[df["综合评分"] >= threshold]

    return df.sort_values("综合评分", ascending=False).reset_index(drop=True)


def scan_market_realtime(
    cfg,
    spot: pd.DataFrame,
    detector_keys: list[str] | None = None,
    statuses: list[str] | None = None,
    min_score: float | None = None,
    market_snapshot=None,
) -> pd.DataFrame:
    """结合全市场最新价扫描形态，供盘中定时刷新观察池。

    market_snapshot 与收盘 scan 用同一份：进出池的"综合评分"要同量纲，
    否则盘中入池的票和收盘报表里的分数对不上。
    """
    if spot is None or spot.empty:
        return pd.DataFrame()

    filtered = universe.filter_spot(spot, cfg)
    if filtered.empty:
        return pd.DataFrame()

    with store.open_store(cfg, readonly=True) as st:
        # 同 scan_market：不再用 bar_counts() 预筛（csv 后端逐文件数行 ~112s，
        # 而本函数被盘中监控每轮刷新池子调用），min_bars 改为读后判定。
        names = st.get_stock_names()
        quote_map = {
            str(row["code"]).zfill(6): row
            for row in filtered.to_dict("records")
        }
        detectors = resolve_detectors(detector_keys)
        rows: list[dict] = []
        limit_bars = max(required_bars(cfg), int(cfg.universe.min_bars))
        min_bars = int(cfg.universe.min_bars)
        for code, quote in quote_map.items():
            hist = st.read_kline(code, limit_bars)
            if hist.empty or len(hist) < min_bars:
                continue
            try:
                ctx = build_context(
                    code, names.get(code, quote.get("name", code)),
                    build_intraday_df(hist, quote, today=pd.Timestamp.today()),
                    cfg,
                    market_snapshot=market_snapshot,
                )
                rows.extend(hit.to_row() for hit in scan_context(ctx, detectors))
            except Exception as exc:  # noqa: BLE001
                log.warning("实时扫描 %s 失败: %s", code, exc)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    base_columns = [
        "代码", "名称", "形态", "形态key", "方向", "状态", "最新价", "综合评分",
        "机会质量", "收益质量", "阶段确定性", "风险修正", "信号日期",
        "预期收益%", "风险收益比", "说明",
    ]
    extra_columns = sorted(col for col in df.columns if col not in base_columns)
    df = df.reindex(columns=base_columns + extra_columns)
    if statuses:
        wanted = {"formed": "已形成", "forming": "构筑中", "invalid": "已失效"}
        df = df[df["状态"].isin({wanted.get(s, s) for s in statuses})]
    threshold = cfg.scoring.min_score if min_score is None else min_score
    if threshold:
        df = df[df["综合评分"] >= threshold]
    return df.sort_values("综合评分", ascending=False).reset_index(drop=True)


def scan_one(
    cfg, code: str, detector_keys: list[str] | None = None, explain: bool = False
) -> tuple[list[PatternHit], ScanContext | None]:
    """单股扫描，返回 (命中列表, 上下文)。上下文带 trace 供 --explain 打印。"""
    with store.open_store(cfg, readonly=True) as st:
        df = st.read_kline(code)
        name = st.get_stock_names().get(code, code)

    if df.empty:
        log.warning("%s 本地无K线数据，先跑 update", code)
        return [], None

    ctx = build_context(code, name, df, cfg, explain=explain)
    hits = scan_context(ctx, resolve_detectors(detector_keys))
    return hits, ctx
