"""W底（双底）：L1 → H(颈线) → L2，突破颈线确认。

已形成 = 最近 breakout_within 根内收盘有效突破颈线；
构筑中 = 右底已出现、价格自右底回升但尚未突破颈线。
"""

from __future__ import annotations

import math

import numpy as np

from ..swings import LOW, HIGH, count_level_touches, prior_drop_pct
from .base import STATUS_FORMED, STATUS_FORMING, PatternHit, ScanContext, blend, taper

PATTERN = "w_bottom"
PATTERN_CN = "W底"


def _pivot_volume(df, idx: int, span: int = 2) -> float:
    lo = max(0, idx - span)
    hi = min(len(df), idx + span + 1)
    return float(df["volume"].iloc[lo:hi].mean())


def _safe(series, idx: int, default: float = 0.0) -> float:
    try:
        value = float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return default
    return default if np.isnan(value) else value


def _candidates(pivots):
    """在严格交替的 pivot 序列里取出所有 (L, H, L) 三元组，从最近往前。"""
    out = []
    for k in range(len(pivots) - 2):
        p1, p2, p3 = pivots[k], pivots[k + 1], pivots[k + 2]
        if p1.kind == LOW and p2.kind == HIGH and p3.kind == LOW:
            out.append((p1, p2, p3))
    return list(reversed(out))


def detect(ctx: ScanContext) -> list[PatternHit]:
    cfg = ctx.cfg
    c = cfg.w_bottom
    if not c.enabled:
        return []

    df, n = ctx.df, ctx.n
    if n < max(c.min_width + 10, 60):
        ctx.note(f"[W底] K线仅{n}根，不足最小长度")
        return []

    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    last = n - 1
    best: PatternHit | None = None

    candidates = _candidates(ctx.pivots)
    ctx.note(f"[W底] 候选(L,H,L)三元组 {len(candidates)} 组")
    for low1, neck_p, low2 in candidates:
        tag = f"[W底] {low1.date:%y%m%d}/{neck_p.date:%y%m%d}/{low2.date:%y%m%d}"
        neck = neck_p.price
        base = min(low1.price, low2.price)
        if base <= 0 or neck <= base:
            continue

        bottom_diff = abs(low2.price - low1.price) / low1.price
        if bottom_diff > c.bottom_tol:
            ctx.note(f"{tag} 双底价差{bottom_diff*100:.1f}% > {c.bottom_tol*100:.0f}%")
            continue

        neck_height = (neck - base) / base
        if not (c.min_neck_height <= neck_height <= c.max_neck_height):
            ctx.note(f"{tag} 颈线高度{neck_height*100:.1f}% 不在允许区间")
            continue

        width = low2.idx - low1.idx
        if not (c.min_width <= width <= c.max_width):
            ctx.note(f"{tag} 宽度{width}根 不在[{c.min_width},{c.max_width}]")
            continue

        left_len = neck_p.idx - low1.idx
        right_len = low2.idx - neck_p.idx
        if left_len <= 0 or right_len <= 0:
            continue
        time_ratio = right_len / left_len
        if not (c.time_ratio_min <= time_ratio <= c.time_ratio_max):
            ctx.note(f"{tag} 左右时间比{time_ratio:.2f} 失衡")
            continue

        drop = prior_drop_pct(df, low1.idx, c.prior_lookback)
        if drop < c.prior_drop:
            ctx.note(f"{tag} 前期跌幅{drop*100:.1f}% < {c.prior_drop*100:.0f}%")
            continue

        # ---- 状态判定 ----
        trigger = neck * (1 + c.breakout_buffer)
        breakout_idx = None
        for i in range(low2.idx + 1, n):
            if close[i] > trigger:
                if breakout_idx is None:
                    # Start of the current uninterrupted close-above run.
                    breakout_idx = i
            else:
                # A failed breakout invalidates the prior signal. A later
                # reclaim may form a new, actionable signal.
                breakout_idx = None

        if breakout_idx is not None:
            if last - breakout_idx >= c.breakout_within:
                ctx.note(
                    f"{tag} 突破发生在{last - breakout_idx}根前，形态已走完"
                )
                continue  # 突破太久，不再是可交易信号
            status = STATUS_FORMED
            signal_idx = breakout_idx
        else:
            if last - low2.idx > c.forming_max_age:
                ctx.note(f"{tag} 右底距今{last - low2.idx}根，视为失效")
                continue
            rebound = (close[last] - low2.price) / low2.price
            near_neck = (neck - close[last]) / neck <= c.forming_near_neck
            turning = close[last] > _safe(df["ma5"], last, close[last]) or (
                last > 0 and close[last] > close[last - 1]
            )
            if rebound < 0.01 or not (near_neck or turning):
                ctx.note(
                    f"{tag} 未突破且自右底反弹{rebound*100:.1f}%、"
                    f"距颈线{(neck-close[last])/neck*100:.1f}%，不算构筑中"
                )
                continue
            status = STATUS_FORMING
            signal_idx = last

        # ---- 几何分 ----
        neck_touches = count_level_touches(
            df, neck, low1.idx, low2.idx + 1, tol=0.015, use="high"
        )
        log_span = math.log(max(c.time_ratio_max, 1 / max(c.time_ratio_min, 1e-9)))
        geom = np.mean(
            [
                taper(bottom_diff, 0.0, c.bottom_tol),
                1.0 if low2.price >= low1.price else 0.6,  # 抬高的右底更强
                taper(neck_height, 0.25, c.min_neck_height),
                taper(abs(math.log(time_ratio)), 0.0, log_span),
                taper(drop, 0.40, c.prior_drop),
                taper(neck_touches, 3, 1),
            ]
        )

        # ---- 量能分 ----
        vol1 = _pivot_volume(df, low1.idx)
        vol2 = _pivot_volume(df, low2.idx)
        vol_lr = vol2 / vol1 if vol1 > 0 else float("nan")
        shrink = taper(1 - vol_lr, 0.40, 0.0) if vol1 > 0 else 0.5
        if status == STATUS_FORMED:
            vma = _safe(df["vol_ma20"], breakout_idx)
            bo_ratio = volume[breakout_idx] / vma if vma > 0 else 0.0
            expand = taper(bo_ratio, c.vol_ratio_breakout * 1.5, 1.0)
        else:
            vma = _safe(df["vol_ma20"], last)
            recent = float(np.nanmean(volume[max(0, last - 4) : last + 1]))
            bo_ratio = recent / vma if vma > 0 else 0.0
            expand = taper(bo_ratio, c.vol_ratio_breakout, 0.8)
        vol_score = 0.45 * shrink + 0.55 * expand

        # ---- 趋势分 ----
        ma20_now = _safe(df["ma20"], last)
        ma20_prev = _safe(df["ma20"], max(0, last - 5))
        hist1 = _safe(df["macd_hist"], low1.idx)
        hist2 = _safe(df["macd_hist"], low2.idx)
        divergence = low2.price <= low1.price * 1.005 and hist2 > hist1
        trend = (
            0.35 * (1.0 if ma20_now and close[last] > ma20_now else 0.0)
            + 0.25 * (1.0 if ma20_now > ma20_prev else 0.0)
            + 0.25 * (1.0 if divergence else 0.0)
            + 0.15 * (1.0 if _safe(df["dif"], last) > _safe(df["dea"], last) else 0.0)
        )

        score, gs, vs, ts = blend(geom, vol_score, trend, status, cfg, priority="A")
        if best is not None and score <= best.score:
            continue

        best = PatternHit(
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
            key_levels={
                "颈线": neck,
                "左底": low1.price,
                "右底": low2.price,
                "颈线高度%": neck_height * 100,
                "参考止损": base * 0.97,
                "等幅目标": neck * (1 + neck_height),
            },
            detail=(
                f"双底{low1.price:.2f}/{low2.price:.2f}(差{bottom_diff*100:.1f}%) "
                f"颈线{neck:.2f}(高{neck_height*100:.1f}%) 宽{width}根 "
                f"前期跌{drop*100:.0f}% 颈线触碰{neck_touches}次 "
                f"右底量比左底{vol_lr:.2f} "
                + (f"突破日量{bo_ratio:.1f}倍" if status == STATUS_FORMED
                   else f"近5日量{bo_ratio:.1f}倍,距颈线{(neck-close[last])/neck*100:.1f}%")
            ),
            pivots=[low1, neck_p, low2],
            close_price=float(close[last]),
            rsi=_safe(df["rsi"], last, 50.0),
            macd_hist=_safe(df["macd_hist"], last, 0.0),
        )

    return [best] if best else []
