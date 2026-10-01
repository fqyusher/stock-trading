"""破底翻第二买点（B2）严格定义实现。

B2 不是"回踩颈线"，而是 B1 突破颈线后形成第一波上涨高点 H1，随后发生第一次
回踩，价格重新进入颈线 N 附近的支撑区域，且没有破坏破底翻结构，回踩过程中出现
缩量或卖压衰减，随后出现止跌并重新转强。

严格定义 11 点：
1. B1 必须已成立：Close > N + 0.2×ATR
2. H1-B1 >= max(5%, 1.5×ATR)
3. H1 之后必须是第一次回踩
4. 颈线支撑区：N-0.5×ATR <= P <= N+0.3×ATR
5. 回踩幅度：20-50%最佳 / 50-70%可接受 / >70%降级
6. 回踩缩量，不允许持续放量破位
7. B2 确认：Close > 前一日 High AND Close > MA5
8. 强确认：Vol >= MA5_Vol AND MACD柱改善 AND RSI 重新上行
9. B2 失效：Close < N-0.3×ATR 连续2日 / 跌破 L0 / 回踩>70%后继续走弱
10. 止损：回踩低点/颈线下方 ATR 缓冲
11. 目标：TP1=H1，TP2=更高前高或结构测量目标

前提「破底翻已形成」由 `break_bottom._find_break_bottom_base` 统一判定。
"""

from __future__ import annotations

import numpy as np

from .base import STATUS_FORMED, STATUS_FORMING, PatternHit, ScanContext, blend, taper
from .break_bottom import _find_break_bottom_base
from ..swings import Pivot, HIGH, LOW

PATTERN = "break_bottom_b2"
PATTERN_CN = "破底翻第二买点"


def _safe(series, idx: int, default: float = 0.0) -> float:
    try:
        value = float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return default
    return default if np.isnan(value) else value


def _zag_pivots(ctx, start: int, end: int) -> list:
    """[start, end] 区间内的 ZigZag 摆动点（含末端未确认点），下标重映射回原序列。"""
    from dataclasses import replace

    from ..swings import zigzag

    if end - start + 1 < 3:
        return []
    pivots = zigzag(ctx.df.iloc[start : end + 1], ctx.cfg)
    return [replace(p, idx=p.idx + start) for p in pivots]


def _pick_h1(
    ctx, start_idx: int, base_price: float, threshold: float, last: int
) -> tuple[int, float] | None:
    """第一波高点 H1：B1 之后第一个满足涨幅阈值的已确认 ZigZag 峰。

    规范第2点：H1 - B1 >= max(5%, 1.5×ATR)。base_price = B1 的收盘价。
    """
    for p in _zag_pivots(ctx, start_idx + 1, last):
        if not p.is_high or not p.confirmed:
            continue
        if (p.price - base_price) / base_price >= threshold:
            return p.idx, float(p.price)
    return None


def _pick_l1(
    ctx, h1_idx: int, h1_price: float, threshold: float, last: int
) -> tuple[int, float] | None:
    """第一次回踩低点 L1：H1 之后第一个满足回撤幅度的已确认 ZigZag 谷。

    规范第3点：必须是第一次回踩。ZigZag 按时间顺序扫描，第一个符合条件的谷
    就是"第一次回踩"。兜底逻辑保留——回踩刚发生、ZigZag 尚未确认时仍需捕获。
    """
    for p in _zag_pivots(ctx, h1_idx + 1, last):
        if not p.is_low or not p.confirmed:
            continue
        if (h1_price - p.price) / h1_price >= threshold:
            return p.idx, float(p.price)
    tail_start = h1_idx + 1
    if tail_start > last:
        return None
    lows = ctx.df["low"].iloc[tail_start : last + 1].to_numpy()
    lo = float(lows.min())
    if lo > 0 and (h1_price - lo) / h1_price >= threshold:
        return tail_start + int(lows.argmin()), lo
    return None


def detect(ctx: ScanContext) -> list[PatternHit]:
    cfg = ctx.cfg
    c = cfg.break_bottom_b2
    if not c.enabled:
        return []

    df, n = ctx.df, ctx.n
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    last = n - 1

    # ── 基础结构定位 ──────────────────────────────────────────────
    bb_base = _find_break_bottom_base(ctx, cfg.break_bottom, cfg_b2=c)
    if bb_base is None:
        return []

    if bb_base.first_breakout_idx is None:
        ctx.note("[B2] B1 未形成（收盘从未放量突破颈线）")
        return []
    if last - bb_base.first_breakout_idx < c.min_bars_after_breakout:
        ctx.note(
            f"[B2] 颈线突破仅{last - bb_base.first_breakout_idx}根，"
            f"不足{c.min_bars_after_breakout}根，来不及形成第一波+回踩"
        )
        return []

    L0 = bb_base.l0
    neck = bb_base.neck
    min_low = bb_base.low
    min_idx = bb_base.low_idx
    b1_idx = bb_base.first_breakout_idx
    b1_close = close[b1_idx]

    # ── 规范第1点：B1 确认 = Close > N + 0.2×ATR ─────────────────
    atr_at_b1 = _safe(df["atr"], b1_idx, neck * 0.03)
    b1_threshold = neck + c.b1_atr_k * atr_at_b1
    if b1_close < b1_threshold:
        ctx.note(
            f"[B2] B1 未确认: Close({b1_close:.2f}) < "
            f"N+{c.b1_atr_k}×ATR({b1_threshold:.2f})"
        )
        return []

    ctx.note(
        f"[B2] B1确认: Close({b1_close:.2f}) > N+{c.b1_atr_k}×ATR({b1_threshold:.2f}) "
        f"L0={L0:.2f} 颈线={neck:.2f}"
    )

    # ── 全局结构失效检查（从 B1 后到当日）────────────────────────
    # 规范第9点：Close < N-0.3×ATR 连续2日 / 跌破 L0
    neck_invalid_level = neck - c.neck_invalid_atr_k * atr_at_b1
    consecutive_below_neck = 0
    consecutive_below_l0 = 0
    for i in range(b1_idx + 1, last + 1):
        if close[i] < neck_invalid_level:
            consecutive_below_neck += 1
            if consecutive_below_neck >= c.pullback_invalid_consecutive:
                ctx.note(
                    f"[B2] 连续{consecutive_below_neck}日收盘 < "
                    f"N-{c.neck_invalid_atr_k}×ATR({neck_invalid_level:.2f})，颈线失效"
                )
                return []
        else:
            consecutive_below_neck = 0

        if close[i] < L0:
            consecutive_below_l0 += 1
            if consecutive_below_l0 >= c.pullback_invalid_consecutive:
                ctx.note(f"[B2] 连续{consecutive_below_l0}日收盘跌破L0，结构破坏")
                return []
        else:
            consecutive_below_l0 = 0

        atr_i = _safe(df["atr"], i, neck * 0.03)
        if close[i] < L0 - c.pullback_l0_atr_buffer * atr_i:
            ctx.note(f"[B2] 收盘跌破 L0-{c.pullback_l0_atr_buffer}×ATR，结构破坏")
            return []

    # ── 规范第2点：H1 从 B1 起算，H1-B1 >= max(5%, 1.5×ATR) ────
    h1_threshold = max(c.h1_min_pct, c.h1_min_atr_k * atr_at_b1 / b1_close)
    h1 = _pick_h1(ctx, b1_idx, b1_close, h1_threshold, last)
    if h1 is None:
        ctx.note(f"[B2] 第一波涨幅不足 {h1_threshold*100:.1f}% 或未形成摆动高点")
        return []
    h1_idx, h1_price = h1
    wave1_gain = (h1_price - b1_close) / b1_close

    ctx.note(
        f"[B2] H1={h1_price:.2f} @ {df.index[h1_idx]:%Y-%m-%d} "
        f"第一波涨幅{wave1_gain*100:.1f}%"
    )

    # ── 规范第3点：第一次回踩 ────────────────────────────────────
    atr_h1 = _safe(df["atr"], h1_idx, neck * 0.03)
    pullback_threshold = max(c.pullback_min_pct, c.pullback_min_atr_k * atr_h1 / h1_price)
    l1 = _pick_l1(ctx, h1_idx, h1_price, pullback_threshold, last)
    if l1 is None:
        ctx.note(f"[B2] 回踩不足 {pullback_threshold*100:.1f}% 或未形成摆动低点")
        return []
    l1_idx, l1_price = l1

    # ── 规范第4点：颈线支撑区 N-0.5×ATR <= L1 <= N+0.3×ATR ─────
    # ATR 取 B1 位置的（与支撑区定义一致），而非 L1 位置的——支撑区是 B1 时刻划定的。
    neck_support_lower = neck - c.neck_support_lower_atr_k * atr_at_b1
    neck_support_upper = neck + c.neck_support_upper_atr_k * atr_at_b1
    if not (neck_support_lower <= l1_price <= neck_support_upper):
        ctx.note(
            f"[B2] L1({l1_price:.2f}) 不在颈线支撑区"
            f"[{neck_support_lower:.2f}, {neck_support_upper:.2f}]"
        )
        return []

    ctx.note(
        f"[B2] L1={l1_price:.2f} @ {df.index[l1_idx]:%Y-%m-%d} "
        f"回撤{(h1_price/l1_price-1)*100:.1f}% 在颈线支撑区内"
    )

    # ── 规范第5点：回踩幅度分级 ──────────────────────────────────
    wave1_range = h1_price - b1_close
    pullback_range = h1_price - l1_price
    pullback_pct = pullback_range / wave1_range if wave1_range > 0 else 1.0

    if pullback_pct <= c.pullback_optimal_max:
        pullback_rating = "最佳"
        pullback_score_mult = 1.0
    elif pullback_pct <= c.pullback_acceptable_max:
        pullback_rating = "可接受"
        pullback_score_mult = 0.7
    else:
        pullback_rating = "过深"
        pullback_score_mult = 0.3

    ctx.note(f"[B2] 回踩幅度{pullback_pct*100:.0f}% ({pullback_rating})")

    # ── 规范第9点补充：回踩>70%后继续走弱 → 失效 ────────────────
    if pullback_pct > c.pullback_acceptable_max and close[last] < l1_price:
        ctx.note(f"[B2] 回踩{pullback_pct*100:.0f}%>70%且现价< L1，继续走弱，失效")
        return []

    # ── H1 后结构完整性（L1 之后到当日也不能破坏）────────────────
    invalid_break = False
    consecutive_below_neck_post = 0
    consecutive_below_l0_post = 0
    for i in range(h1_idx + 1, last + 1):
        if close[i] < neck_invalid_level:
            consecutive_below_neck_post += 1
            if consecutive_below_neck_post >= c.pullback_invalid_consecutive:
                invalid_break = True
                ctx.note(f"[B2] H1后连续{consecutive_below_neck_post}日 < N-{c.neck_invalid_atr_k}×ATR，颈线失效")
                break
        else:
            consecutive_below_neck_post = 0

        if close[i] < L0:
            consecutive_below_l0_post += 1
            if consecutive_below_l0_post >= c.pullback_invalid_consecutive:
                invalid_break = True
                ctx.note(f"[B2] 连续{consecutive_below_l0_post}日收盘跌破L0，结构破坏")
                break
        else:
            consecutive_below_l0_post = 0

        atr_i = _safe(df["atr"], i, neck * 0.03)
        if close[i] < L0 - c.pullback_l0_atr_buffer * atr_i:
            invalid_break = True
            ctx.note(f"[B2] 收盘跌破 L0-{c.pullback_l0_atr_buffer}×ATR，结构破坏")
            break

    if invalid_break:
        return []

    # ── 规范第6点：回踩缩量 + 不允许持续放量破位 ────────────────
    wave1_vol = float(volume[b1_idx : h1_idx + 1].mean())
    pullback_vol = float(volume[h1_idx + 1 : l1_idx + 1].mean())

    if pullback_vol > wave1_vol * c.vol_shrink_ratio:
        ctx.note(f"[B2] 回踩量{pullback_vol:.0f} > 第一波量{wave1_vol:.0f}×{c.vol_shrink_ratio}")
        return []

    # 持续放量破位：回踩段内出现放量（>1.5×vol_ma5）且收盘创新低
    sustained_vol_break = False
    for i in range(h1_idx + 1, l1_idx + 1):
        vol_ma5_i = _safe(df["vol_ma5"], i, volume[i])
        if volume[i] > vol_ma5_i * c.sustained_vol_ratio:
            if i > 0 and close[i] < low_arr[i - 1]:
                sustained_vol_break = True
                ctx.note(f"[B2] 回踩中第{i-h1_idx}根放量破位（量比{volume[i]/vol_ma5_i:.1f}×）")
                break

    if sustained_vol_break:
        return []

    ctx.note(
        f"[B2] 缩量OK: 回踩量{pullback_vol:.0f} ≤ "
        f"第一波量{wave1_vol:.0f}×{c.vol_shrink_ratio:.2f}"
    )

    # ── 规范第7点：B2 确认 = Close > 前一日 High AND Close > MA5 ─
    b2_confirm_ma = c.b2_confirm_ma
    b2_confirmed_idx = None
    strong_confirm = False

    for i in range(l1_idx + 1, last + 1):
        prev_high = high[i - 1] if i > 0 else high[i]
        ma_val = _safe(df[f"ma{b2_confirm_ma}"], i, 0)
        if close[i] > prev_high and close[i] > ma_val:
            b2_confirmed_idx = i
            # ── 规范第8点：强确认检查 ────────────────────────────
            vol_ma5_i = _safe(df["vol_ma5"], i, 0)
            hist_i = _safe(df["macd_hist"], i, 0)
            hist_prev = _safe(df["macd_hist"], i - 1, 0)
            rsi_i = _safe(df["rsi"], i, 50)
            rsi_prev = _safe(df["rsi"], i - 1, 50)
            strong_confirm = (
                volume[i] >= vol_ma5_i
                and hist_i > hist_prev
                and rsi_i > rsi_prev
            )
            break

    # ── 失败突破检测：B2 确认后 N 根内跌回 L1 下方 ──────────────
    if b2_confirmed_idx is not None:
        failed = False
        for i in range(
            b2_confirmed_idx + 1,
            min(b2_confirmed_idx + c.failed_breakout_bars + 1, last + 1),
        ):
            if close[i] < l1_price:
                ctx.note(
                    f"[B2] B2确认后第{i-b2_confirmed_idx}个交易日收盘 < L1，失败确认"
                )
                failed = True
                break
        if failed:
            b2_confirmed_idx = None
            strong_confirm = False

    # ── 追高判定 ─────────────────────────────────────────────────
    gain_from_confirm = 0.0
    if b2_confirmed_idx is not None:
        gain_from_confirm = (close[last] - close[b2_confirmed_idx]) / close[b2_confirmed_idx]
        if gain_from_confirm > c.chase_abort_pct:
            ctx.note(f"[B2] 确认后涨幅{gain_from_confirm*100:.1f}% > {c.chase_abort_pct*100:.0f}%，不追")
            return []
        if gain_from_confirm > c.chase_reduce_pct:
            ctx.note(f"[B2] 确认后涨幅{gain_from_confirm*100:.1f}% > {c.chase_reduce_pct*100:.0f}%，降低评分")

    # ── 时效控制 ─────────────────────────────────────────────────
    if b2_confirmed_idx is None:
        if (last - l1_idx) > c.forming_max_age:
            ctx.note(f"[B2] 回踩已{last - l1_idx}根（> {c.forming_max_age}）仍未确认，超期丢弃")
            return []
    elif (last - b2_confirmed_idx) >= c.breakout_within:
        ctx.note(f"[B2] B2确认已{last - b2_confirmed_idx}根，形态已走完")
        return []

    # ── 状态判定 ─────────────────────────────────────────────────
    if b2_confirmed_idx is not None:
        status = STATUS_FORMED
        signal_idx = b2_confirmed_idx
    else:
        status = STATUS_FORMING
        signal_idx = last

    # ── 评分 ─────────────────────────────────────────────────────
    depth = (L0 - min_low) / L0
    hl_height = (l1_price - neck) / neck if neck > 0 else 0.0

    geom = np.mean([
        taper(depth, 0.0, cfg.break_bottom.max_break_depth),
        taper(wave1_gain, 0.10, 0.05),
        taper(1.0 - pullback_pct / 0.70, 0.7, 0.2),
        taper(hl_height, 0.05, 0.0),
    ])
    geom *= pullback_score_mult

    shrink_floor = min(0.40, c.vol_shrink_ratio * 0.5)
    vol_ratio = pullback_vol / wave1_vol if wave1_vol > 0 else 1.0
    vol = taper(vol_ratio, shrink_floor, c.vol_shrink_ratio)

    ma20 = _safe(df["ma20"], last, 0)
    rsi = _safe(df["rsi"], last, 50)
    hist_last = _safe(df["macd_hist"], last)
    hist_prev = _safe(df["macd_hist"], max(0, last - 1))

    trend = np.mean([
        1.0 if close[last] > ma20 > 0 else 0.0,
        taper(rsi, 70, 50),
        1.0 if hist_last > hist_prev else 0.0,
        1.0 if strong_confirm else 0.0,
    ])

    score, gs, vs, ts = blend(geom, vol, trend, status, cfg, priority="S")

    if b2_confirmed_idx is not None and gain_from_confirm > c.chase_reduce_pct:
        score *= 0.85

    # ── 规范第10点：止损 = min(L1, 颈线) - ATR 缓冲 ─────────────
    atr_at_signal = _safe(df["atr"], b2_confirmed_idx or last, neck * 0.03)
    stop_anchor = min(l1_price, neck)
    stop_loss = stop_anchor - c.stop_atr_buffer * atr_at_signal

    # ── 规范第11点：TP1=H1，TP2=更高前高或测量目标 ──────────────
    tp1 = h1_price
    measured_target = h1_price + (h1_price - l1_price)
    prior_high = bb_base.prior_high if bb_base.prior_high > 0 else measured_target
    tp2 = max(prior_high, measured_target)

    key_levels = {
        "颈线": neck,
        "收复位": L0,
        "破底最低": min_low,
        "B1价格": b1_close,
        "第一波高点": h1_price,
        "回踩低点": l1_price,
        "回踩幅度%": round(pullback_pct * 100, 1),
        "回踩评级": pullback_rating,
        "参考止损": stop_loss,
        "第一目标": tp1,
        "第二目标": tp2,
        "强确认": "是" if strong_confirm else "否",
    }

    detail = (
        f"颈线={neck:.2f} B1={b1_close:.2f} H1={h1_price:.2f} L1={l1_price:.2f} "
        f"第一波{wave1_gain*100:.1f}% 回撤{pullback_pct*100:.0f}%({pullback_rating}) "
        f"缩量{vol_ratio:.2f} "
        f"{'强确认' if strong_confirm else '已确认' if status == STATUS_FORMED else '待确认'}"
    )

    # 构造 B2 关键摆动点，让图表上能画出标记点
    b2_pivots = [
        Pivot(idx=b1_idx, date=df.index[b1_idx], price=b1_close, kind=LOW, confirmed=True),
        Pivot(idx=h1_idx, date=df.index[h1_idx], price=h1_price, kind=HIGH, confirmed=True),
        Pivot(idx=l1_idx, date=df.index[l1_idx], price=l1_price, kind=LOW, confirmed=True),
    ]

    return [
        PatternHit(
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
            pivots=b2_pivots,
            close_price=float(close[last]),
            rsi=rsi,
            macd_hist=_safe(df["macd_hist"], last, 0.0),
        )
    ]
