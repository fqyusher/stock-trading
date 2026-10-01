"""震荡通道：上升通道 / 下降通道 / 横盘箱体。

做法：对收盘价做最小二乘回归定出中轴，上下轨取残差分位数平移（用分位数而非极值，
抗单根插针）。多窗口各拟合一次，取质量分最高的那个。

**区分真通道和随机游走的关键是上下轨各自的触碰次数**——一条随机游走也能被回归出
一个"通道"，但它不会反复在两条轨上受阻和获得支撑。容纳率用 high/low 而不是 close 来算，
否则分位数定轨会让容纳率变成恒等于 90% 的废指标。
"""

from __future__ import annotations

import math

import numpy as np

from .base import STATUS_FORMED, STATUS_FORMING, PatternHit, ScanContext, blend, taper

RISING = "rising_channel"
FALLING = "falling_channel"
BOX = "box_range"
FALLING_REVERSAL = "falling_channel_reversal"
PATTERN_CN = {RISING: "上升通道", FALLING: "下降通道", BOX: "横盘箱体", FALLING_REVERSAL: "下降通道反转"}


def _safe(series, idx: int, default: float = 0.0) -> float:
    try:
        value = float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return default
    return default if np.isnan(value) else value


def _ols(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """返回 (slope, intercept, r2)。"""
    t_mean, y_mean = t.mean(), y.mean()
    var_t = ((t - t_mean) ** 2).sum()
    if var_t <= 0:
        return 0.0, y_mean, 0.0
    slope = ((t - t_mean) * (y - y_mean)).sum() / var_t
    intercept = y_mean - slope * t_mean
    resid = y - (slope * t + intercept)
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1.0 - resid.dot(resid) / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(max(0.0, r2))


def _count_touches(values: np.ndarray, line: np.ndarray, tol: float) -> int:
    """价格贴近轨道的次数，连续多根合并为一次。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        # 与 NaN 比较结果为 False，正好把无效轨道点当作未触碰
        near = np.abs(values - line) / np.where(line <= 0, np.nan, line) <= tol
    if near.size == 0:
        return 0
    return int(np.sum(near & ~np.concatenate(([False], near[:-1]))))


def fit_window(ctx: ScanContext, window: int) -> dict | None:
    """对最近 window 根拟合通道。plot.py 也用它重算轨道线来画图。"""
    df, cfg = ctx.df, ctx.cfg
    c = cfg.channel
    seg = df.iloc[-window:]
    y = seg["close"].to_numpy(dtype=float)
    high = seg["high"].to_numpy(dtype=float)
    low = seg["low"].to_numpy(dtype=float)
    if np.isnan(y).any() or len(y) < 20:
        return None

    t = np.arange(window, dtype=float)
    slope, intercept, r2 = _ols(t, y)
    mid = slope * t + intercept
    resid = y - mid
    upper_off = float(np.quantile(resid, c.resid_q_upper))
    lower_off = float(np.quantile(resid, c.resid_q_lower))
    upper, lower = mid + upper_off, mid + lower_off

    axis_mean = float(mid.mean())
    if axis_mean <= 0:
        return None
    width = (upper_off - lower_off) / axis_mean
    if not (c.min_width <= width <= c.max_width):
        ctx.note(f"[通道{window}] 通道宽度{width*100:.1f}% 不在允许区间")
        return None

    # 容纳率用 high/low 衡量：整根K线是否大体尊重通道
    inside = (high <= upper * (1 + c.touch_tol)) & (low >= lower * (1 - c.touch_tol))
    containment = float(inside.mean())

    up_touches = _count_touches(high, upper, c.touch_tol)
    low_touches = _count_touches(low, lower, c.touch_tol)

    total_change = (mid[-1] - mid[0]) / mid[0] if mid[0] > 0 else 0.0
    if total_change > c.trend_threshold:
        kind = RISING
    elif total_change < -c.trend_threshold:
        kind = FALLING
    else:
        kind = BOX

    # R² 只对趋势通道有意义；横盘箱体的 R² 天然接近 0，用平坦度代替
    if kind == BOX:
        fit_quality = taper(abs(total_change), 0.0, c.trend_threshold)
    else:
        if r2 < c.min_r2:
            ctx.note(f"[通道{window}] R²{r2:.2f} < {c.min_r2}，趋势不成形")
            return None
        fit_quality = taper(r2, 0.85, c.min_r2)

    band = upper[-1] - lower[-1]
    pos = (y[-1] - lower[-1]) / band if band > 0 else 0.5
    broken_up = y[-1] > upper[-1] * (1 + c.touch_tol)
    broken_down = y[-1] < lower[-1] * (1 - c.touch_tol)

    if (
        up_touches >= c.min_touches
        and low_touches >= c.min_touches
        and containment >= c.min_containment
    ):
        status = STATUS_FORMED
    elif (
        max(up_touches, low_touches) >= c.min_touches
        and min(up_touches, low_touches) >= 1
        and containment >= c.min_containment - 0.05
    ):
        status = STATUS_FORMING
    else:
        ctx.note(
            f"[通道{window}] 触碰上{up_touches}次/下{low_touches}次、"
            f"容纳{containment*100:.0f}%，不足以确认通道"
        )
        return None

    mid_width = (c.min_width + c.max_width) / 2
    width_score = 1.0 - abs(width - mid_width) / max(c.max_width - mid_width, 1e-9)
    geom = float(
        np.mean(
            [
                taper(containment, 1.0, c.min_containment),
                taper(min(up_touches, low_touches), 4, c.min_touches),
                max(0.0, width_score),
                fit_quality,
            ]
        )
    )

    return {
        "window": window,
        "kind": kind,
        "slope": slope,
        "r2": r2,
        "width": width,
        "containment": containment,
        "up_touches": up_touches,
        "low_touches": low_touches,
        "total_change": total_change,
        "upper": float(upper[-1]),
        "lower": float(lower[-1]),
        "mid": float(mid[-1]),
        "pos": float(pos),
        "broken_up": bool(broken_up),
        "broken_down": bool(broken_down),
        "status": status,
        "geom": geom,
        "seg": seg,
        # 完整轨道线，供 plot.py 画图
        "upper_line": upper,
        "mid_line": mid,
        "lower_line": lower,
    }


def _volume_score(seg, kind: str) -> tuple[float, float]:
    """量价配合度。返回 (score, up_vol/down_vol)。"""
    close = seg["close"].to_numpy(dtype=float)
    volume = seg["volume"].to_numpy(dtype=float)
    delta = np.diff(close)
    up_vol = volume[1:][delta > 0]
    down_vol = volume[1:][delta < 0]
    if up_vol.size == 0 or down_vol.size == 0:
        return 0.5, float("nan")
    ratio = float(up_vol.mean() / down_vol.mean()) if down_vol.mean() > 0 else float("nan")
    if not np.isfinite(ratio) or ratio <= 0:
        return 0.5, ratio
    if kind == RISING:
        return taper(ratio, 1.4, 0.7), ratio          # 上涨放量才是健康的升通道
    if kind == FALLING:
        return taper(1 / ratio, 1.4, 0.7), ratio      # 下跌放量确认弱势
    return taper(abs(math.log(ratio)), 0.0, math.log(2)), ratio  # 箱体越均衡越好


def _reversal_signal(seg) -> bool:
    """最近 1~2 根是否出现止跌信号：收阳或长下影。"""
    close = seg["close"].to_numpy(dtype=float)
    open_ = seg["open"].to_numpy(dtype=float)
    low = seg["low"].to_numpy(dtype=float)
    high = seg["high"].to_numpy(dtype=float)
    for i in range(-1, max(-3, -len(close) - 1), -1):
        rng = high[i] - low[i]
        lower_shadow = min(open_[i], close[i]) - low[i]
        if close[i] > open_[i]:
            return True
        if rng > 0 and lower_shadow / rng > 0.6:
            return True
    return False


def detect(ctx: ScanContext) -> list[PatternHit]:
    cfg = ctx.cfg
    c = cfg.channel
    if not c.enabled:
        return []

    df, n = ctx.df, ctx.n
    fits = []
    for window in c.windows:
        if n < window:
            ctx.note(f"[通道{window}] K线仅{n}根，窗口不足")
            continue
        candidate = fit_window(ctx, window)
        if candidate is not None:
            fits.append(candidate)
    if not fits:
        return []

    fit = max(fits, key=lambda f: (f["geom"], f["window"]))
    kind = fit["kind"]
    last = n - 1
    close_last = float(df["close"].iloc[last])

    # 下降通道本身仍是有效的图形结果；触下轨并止跌时额外标记为
    # 反转机会，否则保留 FALLING_CHANNEL，避免图形识别与交易方向耦合。
    if kind == FALLING:
        pos = fit["pos"]
        if pos < 0.15 and _reversal_signal(fit["seg"]):
            ctx.note(f"[通道{fit['window']}] 下降通道触下轨(pos={pos:.2f})，降级为反转信号")
            vol_score, vol_ratio = _volume_score(fit["seg"], kind)
            ma20 = _safe(df["ma20"], last)
            ma60 = _safe(df["ma60"], last)
            trend = (
                0.4 * (1.0 if ma20 and close_last < ma20 else 0.0)
                + 0.3 * (1.0 if 0 < ma20 < ma60 else 0.0)
                + 0.3 * taper(pos, 1.0, 0.0)
            )
            score, gs, vs, ts = blend(fit["geom"], vol_score, trend, STATUS_FORMING, cfg, priority="A")
            return [
                PatternHit(
                    code=ctx.code, name=ctx.name,
                    pattern=FALLING_REVERSAL, pattern_cn=PATTERN_CN[FALLING_REVERSAL],
                    status=STATUS_FORMING,
                    score=score, geom_score=gs, vol_score=vs, trend_score=ts,
                    signal_date=df.index[last],
                    key_levels={
                        "上轨": fit["upper"], "中轴": fit["mid"], "下轨": fit["lower"],
                        "通道位置": pos, "通道宽度%": fit["width"] * 100,
                        "参考止损": fit["lower"] * 0.97,
                        "等幅目标": fit["upper"],
                        "窗口": fit["window"], "R2": fit["r2"],
                    },
                    detail=(
                        f"下降通道反转 {fit['window']}根窗口 "
                        f"下轨{fit['lower']:.2f} 上轨{fit['upper']:.2f} "
                        f"位置{pos:.2f} R²{fit['r2']:.2f}"
                    ),
                    pivots=[], close_price=float(close_last),
                    rsi=_safe(df["rsi"], last, 50.0),
                    macd_hist=_safe(df["macd_hist"], last, 0.0),
                )
            ]
        ctx.note(f"[通道{fit['window']}] 下降通道(pos={fit['pos']:.2f}，无止跌信号)")

    vol_score, vol_ratio = _volume_score(fit["seg"], kind)

    ma20 = _safe(df["ma20"], last)
    ma60 = _safe(df["ma60"], last)
    pos = fit["pos"]
    if kind == RISING:
        trend = (
            0.4 * (1.0 if ma20 and close_last > ma20 else 0.0)
            + 0.3 * (1.0 if ma20 > ma60 > 0 else 0.0)
            + 0.3 * taper(pos, 0.0, 1.0)
        )
    elif kind == FALLING:
        # A falling channel is a valid chart result, but its trend score must
        # remain bearish so it cannot rank as a long opportunity by accident.
        trend = (
            0.4 * (1.0 if ma20 and close_last < ma20 else 0.0)
            + 0.3 * (1.0 if 0 < ma20 < ma60 else 0.0)
            + 0.3 * taper(pos, 1.0, 0.0)
        )
    else:
        flat = 1.0 if ma60 > 0 and abs(ma20 - ma60) / ma60 <= 0.03 else 0.0
        trend = 0.5 * flat + 0.5 * taper(pos, 0.0, 1.0)

    # 通道靠回归拟合成立，结构约束弱于双底/双顶：方向性通道给 B，
    # 无趋势的箱体最松，给 C。
    priority = "C" if kind == BOX else "B"
    score, gs, vs, ts = blend(fit["geom"], vol_score, trend, fit["status"], cfg, priority=priority)

    # 止损放在通道另一侧：做多（上升通道/箱体）破下轨证伪，做空（下降通道）
    # 站上上轨证伪。缺这个键会让 risk_reward_ratio() 恒为 0，通道形态在
    # 收益质量上永远只拿到收益分那 60%。
    stop = fit["upper"] * 1.03 if kind == FALLING else fit["lower"] * 0.97

    broke = "已破上轨" if fit["broken_up"] else ("已破下轨" if fit["broken_down"] else "")
    return [
        PatternHit(
            code=ctx.code,
            name=ctx.name,
            pattern=kind,
            pattern_cn=PATTERN_CN[kind],
            status=fit["status"],
            score=score,
            geom_score=gs,
            vol_score=vs,
            trend_score=ts,
            signal_date=df.index[last],
            key_levels={
                "上轨": fit["upper"],
                "中轴": fit["mid"],
                "下轨": fit["lower"],
                "通道位置": pos,
                "通道宽度%": fit["width"] * 100,
                "参考止损": stop,
                "窗口": fit["window"],
                "R2": fit["r2"],
                "容纳率": fit["containment"],
                "上轨触碰": fit["up_touches"],
                "下轨触碰": fit["low_touches"],
                "通道破位": broke or "-",
            },
            detail=(
                f"{fit['window']}根窗口 累计{fit['total_change']*100:+.1f}% "
                f"上轨{fit['upper']:.2f}/下轨{fit['lower']:.2f} 宽{fit['width']*100:.1f}% "
                f"位置{pos:.2f} 触碰上{fit['up_touches']}次/下{fit['low_touches']}次 "
                f"容纳{fit['containment']*100:.0f}% R²{fit['r2']:.2f} "
                f"涨跌量比{vol_ratio:.2f}" + (f" {broke}" if broke else "")
            ),
            pivots=[],
            close_price=float(close_last),
            rsi=_safe(df["rsi"], last, 50.0),
            macd_hist=_safe(df["macd_hist"], last, 0.0),
        )
    ]
