"""整理突破形态：平台突破、突破回踩确认、杯柄/圆弧底。"""
from __future__ import annotations

import numpy as np

from .base import STATUS_FORMED, STATUS_FORMING, PatternHit, ScanContext, blend, taper

PLATFORM = "platform_breakout"
BREAKOUT_RETEST = "breakout_retest"
CUP_HANDLE = "cup_handle"
PATTERN_CN = {
    PLATFORM: "平台突破",
    BREAKOUT_RETEST: "突破回踩确认",
    CUP_HANDLE: "杯柄突破",
}
PATTERNS = tuple(PATTERN_CN)


def _safe(a, i, default=0.0):
    try:
        v = float(a.iloc[i])
        return default if not np.isfinite(v) else v
    except (IndexError, TypeError, ValueError):
        return default


def _hit(ctx, key, status, signal, resistance, support, geom, detail, priority="S", pivots=()):
    df, i = ctx.df, signal
    close = float(df.close.iloc[-1])
    vol_ma = _safe(df.vol_ma20, i, 0.0)
    vr = float(df.volume.iloc[i] / vol_ma) if vol_ma > 0 else 1.0
    vol = taper(vr, 1.8, 0.8)
    ma20, ma60 = _safe(df.ma20, -1), _safe(df.ma60, -1)
    trend = 0.5 * (1.0 if close > ma20 > 0 else 0.0) + 0.5 * (1.0 if ma20 > ma60 > 0 else 0.0)
    score, gs, vs, ts = blend(geom, vol, trend, status, ctx.cfg, priority)
    height = max(0.0, resistance - support)
    return PatternHit(ctx.code, ctx.name, key, PATTERN_CN[key], status, score, gs, vs, ts,
        df.index[signal], {"突破位": resistance, "整理下沿": support,
        "参考止损": support * 0.97, "等幅目标": resistance + height}, detail,
        list(pivots), close, _safe(df.rsi, -1, 50), _safe(df.macd_hist, -1, 0))


def _range_setup(ctx, key, window, trend_min, slope_limit):
    """平台突破检测（含箱体）：前期上涨后进入窄幅整理，突破上沿。"""
    df, n, c = ctx.df, ctx.n, ctx.cfg.breakout
    if n < window + 10:
        return None
    seg = df.iloc[-window-1:-1]
    hi, lo = float(seg.high.max()), float(seg.low.min())
    mid = float(seg.close.mean())
    if mid <= 0 or (hi - lo) / mid > c.max_range:
        return None
    slope = np.polyfit(np.arange(window), seg.close.to_numpy(float), 1)[0] / mid
    if abs(slope) > slope_limit:
        return None
    prior_start = max(0, n - window - c.prior_lookback)
    prior = float(df.close.iloc[prior_start])
    rise = (float(seg.close.iloc[0]) - prior) / prior if prior > 0 else 0
    if rise < trend_min:
        return None
    last = float(df.close.iloc[-1])
    broke = last > hi * (1 + c.breakout_buffer)
    near = last >= hi * (1 - c.forming_near)
    if broke:
        status, signal = STATUS_FORMED, n - 1
    elif near:
        status, signal = STATUS_FORMING, n - 1
    else:
        return None
    geom = np.mean([taper(rise, trend_min * 2, trend_min), taper((hi-lo)/mid, c.target_range, c.max_range),
                    taper(abs(slope), 0, slope_limit)])
    return _hit(ctx, key, status, signal, hi, lo, float(geom),
                f"整理{window}根，前期涨幅{rise*100:.1f}%，区间{lo:.2f}-{hi:.2f}，"
                + ("收盘突破" if broke else "接近突破位"), priority="S")


def _pullback_breakout(ctx, key, window, trend_min, slope_limit):
    """突破回踩确认：整理区间内先回踩下沿确认支撑，再突破上沿。"""
    df, n, c = ctx.df, ctx.n, ctx.cfg.breakout
    if n < window + 10:
        return None
    seg = df.iloc[-window-1:-1]
    hi, lo = float(seg.high.max()), float(seg.low.min())
    mid = float(seg.close.mean())
    if mid <= 0 or (hi - lo) / mid > c.max_range:
        return None
    slope = np.polyfit(np.arange(window), seg.close.to_numpy(float), 1)[0] / mid
    if abs(slope) > slope_limit or slope <= 0:
        return None
    prior_start = max(0, n - window - c.prior_lookback)
    prior = float(df.close.iloc[prior_start])
    rise = (float(seg.close.iloc[0]) - prior) / prior if prior > 0 else 0
    if rise < trend_min:
        return None

    touch_tol = float(c.pullback_touch_tol)
    lows = seg.low.to_numpy(float)
    closes = seg.close.to_numpy(float)
    range_h = hi - lo
    if range_h <= 0:
        return None

    touch_level = lo * (1 + touch_tol)
    touch_indices = np.where(lows <= touch_level)[0]
    has_pullback = len(touch_indices) > 0

    pullback_recovered = False
    if has_pullback:
        last_touch = int(touch_indices[-1])
        mid_level = (hi + lo) / 2
        for j in range(last_touch + 1, len(closes)):
            if closes[j] >= mid_level:
                pullback_recovered = True
                break

    last = float(df.close.iloc[-1])
    broke = last > hi * (1 + c.breakout_buffer)
    near = last >= hi * (1 - c.forming_near)

    if broke and has_pullback and pullback_recovered:
        status, signal = STATUS_FORMED, n - 1
    elif near:
        status, signal = STATUS_FORMING, n - 1
    elif broke and not has_pullback:
        status, signal = STATUS_FORMING, n - 1
    else:
        return None

    pb_bonus = 0.15 if (has_pullback and pullback_recovered) else 0.0
    geom_base = np.mean([
        taper(rise, trend_min * 2, trend_min),
        taper((hi-lo)/mid, c.target_range, c.max_range),
        taper(abs(slope), 0, slope_limit),
    ])
    geom = min(1.0, float(geom_base) + pb_bonus)

    pb_tag = "回踩确认+突破" if (has_pullback and pullback_recovered and broke) else (
        "接近突破位" if near else "突破待确认"
    )
    detail = f"整理{window}根，前期涨幅{rise*100:.1f}%，区间{lo:.2f}-{hi:.2f}，{pb_tag}"
    return _hit(ctx, key, status, signal, hi, lo, geom, detail, priority="S")


def _cup(ctx):
    """杯柄/圆弧底突破：U 形底后突破杯沿。"""
    df, n, c = ctx.df, ctx.n, ctx.cfg.breakout
    w = c.cup_window
    if n < w + 5:
        return None
    seg = df.close.iloc[-w-1:]
    vals = seg.to_numpy(float)
    left, bottom, right = vals[0], float(vals.min()), vals[-1]
    bi = int(np.argmin(vals))
    if not (0.2*w < bi < 0.8*w) or bottom <= 0:
        return None
    depth = (max(left, right) - bottom) / max(left, right)
    if not (c.cup_min_depth <= depth <= c.cup_max_depth):
        return None
    if right < left * (1 - c.cup_rim_tol):
        return None

    rim = max(left, right)
    last = float(df.close.iloc[-1])
    if last > rim * (1 + c.breakout_buffer):
        status = STATUS_FORMED
    elif last >= rim * (1 - c.forming_near):
        status = STATUS_FORMING
    else:
        return None

    # 圆弧底 vs 杯柄：二次拟合 R² 高且对称 → 圆弧底标签
    t = np.arange(w + 1, dtype=float)
    coeffs = np.polyfit(t, vals, 2)
    is_rounding = coeffs[0] > 0 and depth < float(c.rounding_max_depth) * 0.8
    tag = "圆弧底" if is_rounding else "杯柄"

    geom = 0.75
    detail = f"{tag}深度{depth*100:.1f}%，左/右杯沿{left:.2f}/{right:.2f}"
    return _hit(ctx, CUP_HANDLE, status, n - 1, rim, bottom, geom, detail, priority="B")


def detect(ctx: ScanContext) -> list[PatternHit]:
    c = ctx.cfg.breakout
    if not c.enabled:
        return []
    if ctx.trend_direction == "down":
        ctx.note("做多资格: 下降趋势，跳过趋势型突破形态")
        return []

    out = []

    # 平台突破（含箱体）
    hit = _range_setup(ctx, PLATFORM, c.consolidation_window, c.platform_trend, c.platform_slope)
    if hit:
        out.append(hit)

    # 突破回踩确认
    hit = _pullback_breakout(ctx, BREAKOUT_RETEST, c.consolidation_window, c.platform_trend, c.platform_slope)
    if hit:
        out.append(hit)

    # 杯柄/圆弧底
    cup = _cup(ctx)
    if cup:
        out.append(cup)

    return out
