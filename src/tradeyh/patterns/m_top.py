"""M头（双顶）：H1 → L(颈线) → H2，跌破颈线确认。W底的完全镜像。

右头缩量在这里属于**看跌**确认（量价背离），与W底右底缩量的含义相反。
"""

from __future__ import annotations

import math

import numpy as np

from ..swings import HIGH, LOW, count_level_touches, prior_rise_pct
from .base import STATUS_FORMED, STATUS_FORMING, PatternHit, ScanContext, blend, taper

PATTERN = "m_top"
PATTERN_CN = "M头"


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
    """所有 (H, L, H) 三元组，从最近往前。"""
    out = []
    for k in range(len(pivots) - 2):
        p1, p2, p3 = pivots[k], pivots[k + 1], pivots[k + 2]
        if p1.kind == HIGH and p2.kind == LOW and p3.kind == HIGH:
            out.append((p1, p2, p3))
    return list(reversed(out))


def detect(ctx: ScanContext) -> list[PatternHit]:
    cfg = ctx.cfg
    c = cfg.m_top
    if not c.enabled:
        return []

    df, n = ctx.df, ctx.n
    if n < max(c.min_width + 10, 60):
        ctx.note(f"[M头] K线仅{n}根，不足最小长度")
        return []

    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    last = n - 1
    best: PatternHit | None = None

    candidates = _candidates(ctx.pivots)
    ctx.note(f"[M头] 候选(H,L,H)三元组 {len(candidates)} 组")
    for top1, neck_p, top2 in candidates:
        tag = f"[M头] {top1.date:%y%m%d}/{neck_p.date:%y%m%d}/{top2.date:%y%m%d}"
        neck = neck_p.price
        top = max(top1.price, top2.price)
        if neck <= 0 or top <= neck:
            continue

        top_diff = abs(top2.price - top1.price) / top1.price
        if top_diff > c.top_tol:
            ctx.note(f"{tag} 双顶价差{top_diff*100:.1f}% > {c.top_tol*100:.0f}%")
            continue

        neck_depth = (top - neck) / top
        if not (c.min_neck_depth <= neck_depth <= c.max_neck_depth):
            ctx.note(f"{tag} 颈线深度{neck_depth*100:.1f}% 不在允许区间")
            continue

        width = top2.idx - top1.idx
        if not (c.min_width <= width <= c.max_width):
            ctx.note(f"{tag} 宽度{width}根 不在[{c.min_width},{c.max_width}]")
            continue

        left_len = neck_p.idx - top1.idx
        right_len = top2.idx - neck_p.idx
        if left_len <= 0 or right_len <= 0:
            continue
        time_ratio = right_len / left_len
        if not (c.time_ratio_min <= time_ratio <= c.time_ratio_max):
            ctx.note(f"{tag} 左右时间比{time_ratio:.2f} 失衡")
            continue

        rise = prior_rise_pct(df, top1.idx, c.prior_lookback)
        if rise < c.prior_rise:
            ctx.note(f"{tag} 前期涨幅{rise*100:.1f}% < {c.prior_rise*100:.0f}%")
            continue

        # ---- 状态判定 ----
        trigger = neck * (1 - c.breakdown_buffer)
        breakdown_idx = None
        for i in range(top2.idx + 1, n):
            if close[i] < trigger:
                if breakdown_idx is None:
                    # Start of the current uninterrupted close-below run.
                    breakdown_idx = i
            else:
                # A reclaimed neckline invalidates the prior breakdown.
                breakdown_idx = None

        if breakdown_idx is not None:
            if last - breakdown_idx >= c.breakdown_within:
                ctx.note(f"{tag} 破位发生在{last - breakdown_idx}根前，形态已走完")
                continue
            status = STATUS_FORMED
            signal_idx = breakdown_idx
        else:
            if last - top2.idx > c.forming_max_age:
                ctx.note(f"{tag} 右头距今{last - top2.idx}根，视为失效")
                continue
            decline = (top2.price - close[last]) / top2.price
            near_neck = (close[last] - neck) / neck <= c.forming_near_neck
            turning = close[last] < _safe(df["ma5"], last, close[last]) or (
                last > 0 and close[last] < close[last - 1]
            )
            if decline < 0.01 or not (near_neck or turning):
                ctx.note(
                    f"{tag} 未破位且自右头回落{decline*100:.1f}%、"
                    f"距颈线{(close[last]-neck)/neck*100:.1f}%，不算构筑中"
                )
                continue
            status = STATUS_FORMING
            signal_idx = last

        # ---- 几何分 ----
        neck_touches = count_level_touches(
            df, neck, top1.idx, top2.idx + 1, tol=0.015, use="low"
        )
        log_span = math.log(max(c.time_ratio_max, 1 / max(c.time_ratio_min, 1e-9)))
        geom = np.mean(
            [
                taper(top_diff, 0.0, c.top_tol),
                1.0 if top2.price <= top1.price else 0.6,  # 走低的右头更弱势
                taper(neck_depth, 0.25, c.min_neck_depth),
                taper(abs(math.log(time_ratio)), 0.0, log_span),
                taper(rise, 0.40, c.prior_rise),
                taper(neck_touches, 3, 1),
            ]
        )

        # ---- 量能分 ----
        vol1 = _pivot_volume(df, top1.idx)
        vol2 = _pivot_volume(df, top2.idx)
        vol_lr = vol2 / vol1 if vol1 > 0 else float("nan")
        # 右头缩量 = 追高动能衰减，是M头的核心量能特征
        shrink = taper(1 - vol_lr, 0.40, 0.0) if vol1 > 0 else 0.5
        if status == STATUS_FORMED:
            vma = _safe(df["vol_ma20"], breakdown_idx)
            bd_ratio = volume[breakdown_idx] / vma if vma > 0 else 0.0
        else:
            vma = _safe(df["vol_ma20"], last)
            recent = float(np.nanmean(volume[max(0, last - 4) : last + 1]))
            bd_ratio = recent / vma if vma > 0 else 0.0
        expand = taper(bd_ratio, c.vol_ratio_breakdown * 1.5, 0.8)
        vol_score = 0.6 * shrink + 0.4 * expand

        # ---- 趋势分 ----
        ma20_now = _safe(df["ma20"], last)
        ma20_prev = _safe(df["ma20"], max(0, last - 5))
        hist1 = _safe(df["macd_hist"], top1.idx)
        hist2 = _safe(df["macd_hist"], top2.idx)
        divergence = top2.price >= top1.price * 0.995 and hist2 < hist1
        trend = (
            0.35 * (1.0 if ma20_now and close[last] < ma20_now else 0.0)
            + 0.25 * (1.0 if ma20_now < ma20_prev else 0.0)
            + 0.25 * (1.0 if divergence else 0.0)
            + 0.15 * (1.0 if _safe(df["dif"], last) < _safe(df["dea"], last) else 0.0)
        )

        # 双顶是 W 底的镜像，结构约束同样硬，权重取 A
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
                "左头": top1.price,
                "右头": top2.price,
                "颈线深度%": neck_depth * 100,
                "反弹压力": neck,
                "等幅目标": neck * (1 - neck_depth),
                # 做空形态：止损在颈线上方（跌破后站回颈线即证伪）。
                # 缺这个键会让 risk_reward_ratio() 恒为 0，M头永远拿不到收益质量分。
                "参考止损": neck * 1.03,
            },
            detail=(
                f"双顶{top1.price:.2f}/{top2.price:.2f}(差{top_diff*100:.1f}%) "
                f"颈线{neck:.2f}(深{neck_depth*100:.1f}%) 宽{width}根 "
                f"前期涨{rise*100:.0f}% 颈线触碰{neck_touches}次 "
                f"右头量比左头{vol_lr:.2f} "
                + (f"破位日量{bd_ratio:.1f}倍" if status == STATUS_FORMED
                   else f"近5日量{bd_ratio:.1f}倍,距颈线{(close[last]-neck)/neck*100:.1f}%")
            ),
            pivots=[top1, neck_p, top2],
            close_price=float(close[last]),
            rsi=_safe(df["rsi"], last, 50.0),
            macd_hist=_safe(df["macd_hist"], last, 0.0),
        )

    return [best] if best else []
