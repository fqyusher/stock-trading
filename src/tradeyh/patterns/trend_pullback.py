"""上升趋势回踩再启动：上升趋势中价格回踩关键支撑后企稳反弹。

支撑位优先级（任一触碰即算）：
1. 前一个 Swing Low / 更高低点
2. 突破后的原压力位 → 支撑转换
3. MA20
4. 上升通道中轴/下轨
5. MA60
"""
from __future__ import annotations

import numpy as np

from .base import STATUS_FORMED, STATUS_FORMING, PatternHit, ScanContext, blend, taper
from ..swings import LOW

PATTERN = "trend_pullback"
PATTERN_CN = "趋势回踩再启动"


def _safe(a, i, default=0.0):
    try:
        v = float(a.iloc[i])
        return default if not np.isfinite(v) else v
    except (IndexError, TypeError, ValueError):
        return default


def _find_support_levels(ctx, lookback: int) -> list[tuple[str, float]]:
    """找出所有可能的支撑位，返回 [(类型, 价格), ...]。"""
    df, n = ctx.df, ctx.n
    supports = []

    ma20 = _safe(df.ma20, -1, 0)
    ma60 = _safe(df.ma60, -1, 0)
    if ma20 > 0:
        supports.append(("MA20", ma20))
    if ma60 > 0:
        supports.append(("MA60", ma60))

    # 前一个 Swing Low
    for pivot in reversed(ctx.pivots):
        if pivot.kind == LOW and pivot.idx < n:
            supports.append(("SwingLow", float(pivot.price)))
            break

    # 突破转支撑：近 lookback 根内的前高（排除最近 5 根）
    start = max(0, n - lookback)
    end = max(start, n - 5)
    if end > start:
        prev_high = float(df.high.iloc[start:end].max())
        if prev_high > 0:
            supports.append(("突破转支撑", prev_high))

    return supports


def _reversal_signal(df) -> bool:
    """最近 1~2 根出现止跌信号：收阳且收在昨收之上，或长下影线。"""
    for offset in (-1, -2):
        try:
            o = float(df.open.iloc[offset])
            c = float(df.close.iloc[offset])
            h = float(df.high.iloc[offset])
            lo = float(df.low.iloc[offset])
            prev_c = float(df.close.iloc[offset - 1])
        except (IndexError, TypeError, ValueError):
            continue
        full = h - lo
        if full <= 0:
            continue
        body = abs(c - o)
        lower_shadow = min(o, c) - lo
        if c > o and c > prev_c:
            return True
        if lower_shadow > body * 1.5 and lower_shadow > full * 0.4:
            return True
    return False


def detect(ctx: ScanContext) -> list[PatternHit]:
    c_cfg = ctx.cfg.trend_pullback
    if not c_cfg.enabled:
        return []
    if ctx.trend_direction == "down":
        ctx.note("做多资格: 下降趋势，跳过趋势回踩")
        return []

    df, n = ctx.df, ctx.n
    lookback = int(c_cfg.lookback)
    if n < lookback + 10:
        ctx.note(f"趋势回踩: K线不足 ({n} < {lookback + 10})")
        return []

    # 1. 确认上升趋势：MA20 > MA60
    ma20 = _safe(df.ma20, -1, 0)
    ma60 = _safe(df.ma60, -1, 0)
    if not (ma20 > ma60 > 0):
        ctx.note("趋势回踩: MA20 < MA60，非上升趋势")
        return []

    # 2. 近期有上涨（高点必须够新——掉队的老高点算不上"回踩再启动"）
    start = max(0, n - lookback)
    high_arr = df.high.to_numpy()[start:n]
    high_idx = start + int(np.argmax(high_arr))
    recent_high = float(high_arr.max())
    prior_close = float(df.close.iloc[start])
    if prior_close <= 0:
        return []
    rise = (recent_high - prior_close) / prior_close
    min_rise = float(c_cfg.min_rise)
    if rise < min_rise:
        ctx.note(f"趋势回踩: 前期涨幅 {rise*100:.1f}% < {min_rise*100:.0f}%")
        return []

    high_max_age = int(c_cfg.high_max_age)
    age = (n - 1) - high_idx
    if age > high_max_age:
        ctx.note(f"趋势回踩: 高点距今 {age} 根 > {high_max_age}（高点太旧，已掉队）")
        return []
    if high_idx >= n - 1:
        ctx.note("趋势回踩: 高点即最新一根，尚未回踩")
        return []

    # 3. 当前回撤幅度（从高点之后的最低点量起，不用最近5根的最低）
    close = float(df.close.iloc[-1])
    low_after = float(df.low.iloc[high_idx + 1:n].min())
    pullback = (recent_high - low_after) / recent_high
    pb_min = float(c_cfg.pullback_min)
    pb_max = float(c_cfg.pullback_max)
    if pullback < pb_min:
        ctx.note(f"趋势回踩: 回撤 {pullback*100:.1f}% < {pb_min*100:.0f}%（太浅）")
        return []
    if pullback > pb_max:
        ctx.note(f"趋势回踩: 回撤 {pullback*100:.1f}% > {pb_max*100:.0f}%（趋势可能已坏）")
        return []

    # 4. 检查回踩是否触及支撑位
    supports = _find_support_levels(ctx, lookback)
    hit_support = None
    hit_type = ""
    ma20_tol = float(c_cfg.ma20_tol)
    ma60_tol = float(c_cfg.ma60_tol)
    generic_tol = 0.02

    for stype, sprice in supports:
        if sprice <= 0:
            continue
        tol = ma20_tol if stype == "MA20" else (ma60_tol if stype == "MA60" else generic_tol)
        for offset in range(-5, 0):
            lo = _safe(df.low, offset, 0)
            if lo > 0 and abs(lo - sprice) / sprice < tol:
                hit_support = sprice
                hit_type = stype
                break
        if hit_support:
            break

    if hit_support is None:
        ctx.note("趋势回踩: 未触及任何支撑位")
        return []

    # 5. 止跌信号
    if not _reversal_signal(df):
        ctx.note("趋势回踩: 无止跌信号")
        return []

    # 6. 状态判定
    stop_buffer = float(c_cfg.stop_buffer)
    stop_loss = hit_support * (1 - stop_buffer)
    target = recent_high + (recent_high - hit_support)

    if close > ma20 > 0 and _safe(df.close, -2, 0) < ma20:
        status = STATUS_FORMED
    elif close >= hit_support * (1 - generic_tol):
        status = STATUS_FORMING
    else:
        ctx.note(
            f"趋势回踩: 现价 {close:.2f} 仍明显低于支撑 {hit_support:.2f} "
            f"({(close/hit_support-1)*100:.1f}%)，未企稳"
        )
        return []

    # 7. 评分
    geom = np.mean([
        taper(rise, min_rise * 2, min_rise),
        taper(1.0 - pullback / pb_max, 0.7, 0.2),
        1.0 if hit_type in ("SwingLow", "突破转支撑") else 0.7,
    ])

    vol_ma = _safe(df.vol_ma20, -1, 0)
    vol_ratio = float(df.volume.iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
    vol = taper(vol_ratio, 0.6, 1.5)

    trend = np.mean([
        1.0 if ma20 > ma60 > 0 else 0.0,
        taper(rise, min_rise * 3, min_rise),
        1.0 if close > ma20 > 0 else 0.5,
    ])

    score, gs, vs, ts = blend(geom, vol, trend, status, ctx.cfg, priority="S")

    pullback_pct = (recent_high - close) / recent_high * 100
    detail = (f"回踩{hit_type} {hit_support:.2f}  回撤{pullback_pct:.1f}%  "
              f"前期涨{rise*100:.1f}%  高点{recent_high:.2f}")

    key_levels = {
        "回踩支撑": hit_support,
        "近期高点": recent_high,
        # Report fields suffixed with % use percentage points (e.g. 5.2).
        "回撤幅度%": pullback_pct,
        "参考止损": stop_loss,
        "等幅目标": target,
    }

    return [PatternHit(
        ctx.code, ctx.name, PATTERN, PATTERN_CN, status, score, gs, vs, ts,
        df.index[-1], key_levels, detail, [], close,
        _safe(df.rsi, -1, 50), _safe(df.macd_hist, -1, 0),
    )]
