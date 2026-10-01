"""破底翻 v2 第二买点：v2 B1 确认 → 第一波 H1 → 回踩 pl（不破 L0、缩量）→ 放量破 H1 再启动。

与 v2 B1 **独立计算**：B1 的 `_find_v2_bases` 是结构定位的唯一实现，B2 只是换一组
前提（B1 已确认 + H1 已形成 + 之后走完一个回踩）与更长的 L0 回看窗口来复用同一份结构。

口径（与旧 `break_bottom_b2.py` 的差别不只是阈值，判据本身换了）：

- 前提：v2 B1 `B1_CONFIRMED` 且 H1 已形成（`h1_price > 0`），B1 距今
  `min_bars_after_b1` ~ `break_within_v2` 根。
- 回踩低点 `pl` = H1 之后首个已确认 ZigZag 谷，且
  `low > L0 − pullback_l0_atr_k×ATR`（回踩不得有效跌破 L0）、
  幅度 ≥ `max(pullback_min_pct, pullback_min_atr_k×ATR/H1)`。
  跌破 L0 的回踩直接不作数（不是"失效"，是"这里没有 B2"）。
- 缩量：回踩段均量 ≤ 第一波均量 × `pullback_max_vol_ratio`。
- 再启动：`pl` 之后首根 `close > H1×(1+breakout_buffer_pct)` 且量 ≥ `vol_ma5×breakout_vol_ratio`。
- `Stop = max(L0 − k×ATR, L1 − k×ATR, pl − k×ATR)`；
  `TP1 = H1 + (H1 − L1)`（等幅量度）；`TP2` = v2 的 TP2 候选池里高于 TP1 的最近一档。
- 追高闸门 `chase_abort_pct`、时效 `breakout_within` / `forming_max_age`，与 v1 B2 同思路。

失效只报一种：**B2 确认之后收盘跌穿 Stop**（交易上早已离场，不该再报"已形成"）。
"""

from __future__ import annotations

import numpy as np

from ..swings import LOW
from .base import (
    STATUS_FORMED,
    STATUS_FORMING,
    STATUS_INVALID,
    PatternHit,
    ScanContext,
    blend,
    taper,
)
from .break_bottom_v2 import (
    STAGE_CONFIRMED,
    _atr_at,
    _find_v2_bases,
    _pick_tp2,
    _safe,
    _tp2_candidates,
    trailing_stop_ref,
)

PATTERN = "break_bottom_v2_b2"
PATTERN_CN = "破底翻v2第二买点"


def _find_pullback(
    ctx: ScanContext, h1_idx: int, h1_price: float, l0: float, threshold: float, last: int
) -> tuple[int, float] | None:
    """H1 之后首个"回踩不破 L0、幅度够"的已确认 ZigZag 谷。

    只取已确认的谷：未确认的末端点还没"止跌回升"，把它当回踩低点会让
    "仍在下探"的即时段冒充 pl。
    """
    df = ctx.df
    atr = df["atr"].to_numpy(dtype=float) if "atr" in df.columns else np.full(ctx.n, np.nan)
    close = df["close"].to_numpy(dtype=float)
    for p in ctx.pivots:
        if p.kind != LOW or not p.confirmed or p.idx <= h1_idx or p.price <= 0:
            continue
        if p.price <= l0 - ctx.cfg.break_bottom_v2_b2.pullback_l0_atr_k * _atr_at(
            atr, close, p.idx
        ):
            continue  # 回踩已有效跌破 L0 —— 这里不存在 B2
        if (h1_price - p.price) / h1_price >= threshold:
            return p.idx, float(p.price)
    return None


def _find_restart(
    ctx: ScanContext, after: int, h1_price: float, buffer: float, vol_ratio: float, last: int
) -> int | None:
    """pl 之后首根"收盘突破 H1 且放量"的根。"""
    df = ctx.df
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    for i in range(after + 1, last + 1):
        if close[i] < h1_price * (1 + buffer):
            continue
        vma5 = _safe(df["vol_ma5"], i, 0.0)
        if vma5 > 0 and volume[i] < vma5 * vol_ratio:
            continue
        return i
    return None


def detect(ctx: ScanContext) -> list[PatternHit]:
    cfg = ctx.cfg
    c = cfg.break_bottom_v2_b2
    if not c.enabled:
        return []
    v2c = cfg.break_bottom_v2

    df, n = ctx.df, ctx.n
    volume = df["volume"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float) if "atr" in df.columns else np.full(n, np.nan)
    last = n - 1

    # B2 允许 B1 更早（break_within_v2），因此 L0 的回看窗口也要相应放宽
    bases = _find_v2_bases(
        ctx, v2c, max_l0_age=int(c.break_within_v2) + int(v2c.l0_lookback)
    )
    confirmed = [
        b
        for b in bases
        if b.stage == STAGE_CONFIRMED
        and b.h1_price > 0
        and b.b1_idx is not None
        and c.min_bars_after_b1 <= (last - b.b1_idx) <= c.break_within_v2
    ]
    if not confirmed:
        ctx.note("[破底翻v2-B2] 无「B1已确认且H1已形成」且时效内的结构")
        return []

    hits: list[PatternHit] = []
    for base in confirmed:
        hit = _build(ctx, c, v2c, base, last, volume, close, atr)
        if hit is not None:
            hits.append(hit)
    if not hits:
        return []

    # 每票最多 1 行：未失效优先 → 已形成优先 → 信号日新优先
    return [max(hits, key=lambda h: (not h.pattern_invalid, h.status == STATUS_FORMED,
                                     h.signal_date.value))]


def _build(ctx, c, v2c, base, last, volume, close, atr) -> PatternHit | None:
    df = ctx.df
    h1_idx, h1_price = base.h1_idx, base.h1_price
    atr_h1 = _atr_at(atr, close, h1_idx)
    threshold = max(c.pullback_min_pct, c.pullback_min_atr_k * atr_h1 / h1_price)

    pullback = _find_pullback(ctx, h1_idx, h1_price, base.l0, threshold, last)
    if pullback is None:
        ctx.note(
            f"[破底翻v2-B2] H1={h1_price:.2f} 之后回踩不足 {threshold*100:.1f}% "
            f"或回踩已跌破 L0={base.l0:.2f}"
        )
        return None
    pl_idx, pl_price = pullback

    wave1_vol = float(volume[base.b1_idx : h1_idx + 1].mean())
    pull_vol = float(volume[h1_idx + 1 : pl_idx + 1].mean())
    if wave1_vol > 0 and pull_vol > wave1_vol * c.pullback_max_vol_ratio:
        ctx.note(
            f"[破底翻v2-B2] 回踩放量 {pull_vol:.0f} > 第一波 {wave1_vol:.0f}"
            f"×{c.pullback_max_vol_ratio}"
        )
        return None
    vol_ratio = pull_vol / wave1_vol if wave1_vol > 0 else 1.0

    b2_idx = _find_restart(
        ctx, pl_idx, h1_price, c.breakout_buffer_pct, c.breakout_vol_ratio, last
    )

    # 确认价 / 止损 / 目标
    entry = float(close[b2_idx]) if b2_idx is not None else 0.0
    atr_ref = _atr_at(atr, close, b2_idx if b2_idx is not None else last)
    stop = max(
        base.l0 - v2c.stop_atr_k * atr_ref,
        base.l1 - v2c.extreme_stop_atr_k * atr_ref,
        pl_price - c.stop_swing_atr_k * atr_ref,
    )
    tp1 = h1_price + (h1_price - base.l1)
    tp2 = _pick_tp2(_tp2_candidates(ctx, v2c, last), tp1)

    status = STATUS_FORMING
    invalid_reason = ""
    if b2_idx is not None:
        gain = (close[last] - entry) / entry if entry > 0 else 0.0
        if gain > c.chase_abort_pct:
            ctx.note(
                f"[破底翻v2-B2] 突破后涨幅{gain*100:.1f}% > "
                f"{c.chase_abort_pct*100:.0f}%，不追"
            )
            return None
        if last - b2_idx >= c.breakout_within:
            ctx.note(
                f"[破底翻v2-B2] 突破 H1 已{last - b2_idx}根，形态已走完"
            )
            return None
        breach = next((i for i in range(b2_idx + 1, last + 1) if close[i] < stop), None)
        if breach is not None:
            status = STATUS_INVALID
            invalid_reason = "B2确认后跌破止损位"
            if not c.emit_invalid or (last - breach) > int(c.invalid_within):
                return None
        else:
            status = STATUS_FORMED
    elif last - pl_idx > c.forming_max_age:
        ctx.note(
            f"[破底翻v2-B2] 回踩已{last - pl_idx}根仍未突破 H1"
            f"（> {c.forming_max_age}），超期丢弃"
        )
        return None

    if status == STATUS_INVALID and not c.emit_invalid:
        return None

    signal_idx = b2_idx if b2_idx is not None else last
    # ---- RR / 百分比 ----
    risk = entry - stop if entry > 0 else 0.0
    reward1 = tp1 - entry if entry > 0 else 0.0
    reward2 = tp2 - entry if entry > 0 else 0.0
    rr1 = reward1 / risk if risk > 0 and reward1 > 0 else 0.0
    rr2 = reward2 / risk if risk > 0 and reward2 > 0 else 0.0
    risk_pct = risk / entry * 100 if entry > 0 else 0.0
    reward1_pct = reward1 / entry * 100 if entry > 0 else 0.0
    reward2_pct = reward2 / entry * 100 if entry > 0 else 0.0

    # ---- 子分 ----
    rebound_pct = (h1_price - base.l1) / base.l1
    pullback_pct = (h1_price - pl_price) / h1_price
    geom = np.mean(
        [
            taper(rebound_pct, 0.10, 0.05),
            taper(1.0 - pullback_pct / 0.20, 0.7, 0.2),
            taper(base.depth, 0.0, v2c.max_break_depth),
            1.0 if status == STATUS_FORMED else 0.7,
        ]
    )
    shrink_floor = min(0.40, c.pullback_max_vol_ratio * 0.5)
    vol_score = taper(vol_ratio, shrink_floor, c.pullback_max_vol_ratio)

    ma20 = _safe(df["ma20"], last, 0.0)
    ma20_prev = _safe(df["ma20"], max(0, last - 5), ma20)
    rsi_last = _safe(df["rsi"], last, 50.0)
    hist_last = _safe(df["macd_hist"], last, 0.0)
    hist_prev = _safe(df["macd_hist"], max(0, last - 1), 0.0)
    trend = np.mean(
        [
            1.0 if ma20 > ma20_prev > 0 else 0.0,
            taper(rsi_last, 70, 50),
            1.0 if hist_last > hist_prev else 0.0,
            1.0 if ma20 > 0 and close[last] > ma20 else 0.0,
        ]
    )
    score, gs, vs, ts = blend(geom, vol_score, trend, status, ctx.cfg, priority="S")

    dates = {
        "L0": df.index[base.l0_idx],
        "L1": df.index[base.l1_idx],
        "B1": df.index[base.b1_idx],
        "H1": df.index[h1_idx],
        "PL": df.index[pl_idx],
    }
    if b2_idx is not None:
        dates["B2"] = df.index[b2_idx]

    key_levels = {
        "L0": base.l0,
        "L1": base.l1,
        "B1": base.b1_price,
        "H1": h1_price,
        "PL": pl_price,
        "B2": entry,
        "stop_price": stop,
        "tp1_price": tp1,
        "tp2_price": tp2,
        "risk_pct": risk_pct,
        "reward1_pct": reward1_pct,
        "reward2_pct": reward2_pct,
        "rr1": rr1,
        "rr2": rr2,
        "回撤%": pullback_pct * 100,
        "缩量比": vol_ratio,
        "trailing_stop": trailing_stop_ref(ctx, v2c, last, pl_price),
    }

    detail = (
        f"L0={base.l0:.2f} H1={h1_price:.2f} 回踩{pl_price:.2f}"
        f"({pullback_pct*100:.1f}%) 缩量{vol_ratio:.2f} "
    )
    if b2_idx is not None:
        detail += (
            f"B2={entry:.2f}({df.index[b2_idx]:%m-%d}) 止损{stop:.2f} "
            f"RR1={rr1:.2f} TP1={tp1:.2f}"
        )
    else:
        detail += f"待突破 H1={h1_price:.2f}"
    if invalid_reason:
        detail += f" 失效原因:{invalid_reason}"

    return PatternHit(
        code=ctx.code,
        name=ctx.name,
        pattern=PATTERN,
        pattern_cn=PATTERN_CN,
        status=status,
        score=score,
        geom_score=gs,
        vol_score=vs,
        trend_score=ts,
        signal_date=df.index[signal_idx],
        key_levels=key_levels,
        detail=detail,
        pivots=[],
        close_price=float(close[last]),
        rsi=rsi_last,
        macd_hist=hist_last,
        pattern_stage="B2_CONFIRMED" if status == STATUS_FORMED else "PULLBACK",
        pattern_invalid=(status == STATUS_INVALID),
        invalid_reason=invalid_reason,
        key_level_dates=dates,
    )
