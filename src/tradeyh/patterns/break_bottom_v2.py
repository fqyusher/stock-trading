"""破底翻 v2（ATR 严格口径）：L0 → 有效破底 → 快速收复 → 跌破失败确认 → B1 确认买点。

与旧 `break_bottom.py` **并行**存在，不改旧口径任何行为（旧口径是 2026-09-21 定稿的
305 条基线，300353 是基准票）。两套在报表里并列，跑数据对比后再决定是否替换。

完整流程（每一步都对应一个配置闸门）：

1. L0 = 最近 `l0_lookback` 根内**已确认**的 Swing Low，且它前面 `prior_lookback`
   根内没有更低的原始最低价（`l0_must_be_leg_low`），并且 L0 之前跌过 `prior_drop`。
2. **有效破底**：存在 i > L0 使 `low[i] < L0 − break_low_atr_k×ATR`
   **或** `close[i] < L0 − break_close_atr_k×ATR`。与旧口径不同，**盘中破也算**
   （旧口径要求收盘跌破）。
3. `L1 = min(low[L0+1:])`，深度 ∈ `[min_break_depth, max_break_depth]`。
4. **快速收复**：自 `L1` 起 `recover_bars` 根内出现 `close > L0`。
   计时起点取**破底最低点 L1 而不是首个触发 ATR 阈值的根**：缓跌型二次探底里
   阈值在破底段很早就被穿过，从首破根起算会把正常结构判死（实测
   `strict_break_bottom`：首破@99 最低@104 收复@106，自首破起 7 根>5 闸门，自最低起 2 根）。
5. **跌破失败确认**：收复后 `fail_confirm_bars` 根内每根都要满足
   `close > L0 − fail_confirm_atr_k×ATR`，否则失效（"假收复、真破位"）。
   判据取**收盘价**：只盘中插针破不算（300353 2026-09-23 就是盘中 16.35 破了阈值
   16.42、收盘 16.89 的长阳，按最低价判会把这条基准票直接判死）。
6. **B1 确认**：自确认窗口之后首个满足 `close > L0 + b1_close_atr_k×ATR`
   **或** 连续 `b1_hold_days` 日 `close > L0` 的根。B1 就是买点（Entry）。
7. 止损 `max(L0 − stop_atr_k×ATR, L1 − extreme_stop_atr_k×ATR)`；`Entry ≤ 止损` 视为无效；
   确认之后若有收盘跌破止损，则该信号失效（**定义外新增**，理由见下）。
8. H1 = B1 之后首个已确认高点满足 `H1 ≥ L0×(1+h1_min_pct)` 或 `H1 − L0 ≥ h1_min_atr_k×ATR`；
   `tp1_price = H1`，未形成则 0。TP2 = 候选池（确认摆动高点 ∪ 60 日高 ∪ 120 日高 ∪
   MA120 ∪ 成交密集区中心）里**严格高于 H1 的最近一档**。

阶段机：`BROKEN`（已破底未收复）→ `RECLAIMED`（已收复待确认）→ `B1_CONFIRMED`。
status：`B1_CONFIRMED` → formed；`BROKEN`/`RECLAIMED` → forming。
失效（`status="invalid"`）也输出一行，但只在 `invalid_within` 根内发生的失效才输出，
否则报表会被历史失败塞满。

**硬约束：每只票最多输出 1 行。** 出图文件名是 `代码_形态key.png`，同一 (代码, 形态)
出两行会直接撞文件名；多候选时按 `(未失效优先, 阶段靠后优先, 信号日新优先)` 取一条。

`_find_v2_bases` 是 v2 的结构定位入口，v2 的 B2 也从这里导入——
第二买点的前提就是"B1 已确认"，两套口径必须逐字一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..swings import HIGH, LOW, Pivot, prior_drop_pct
from .base import (
    STATUS_FORMED,
    STATUS_FORMING,
    STATUS_INVALID,
    PatternHit,
    ScanContext,
    blend,
    taper,
)

PATTERN = "break_bottom_v2"
PATTERN_CN = "破底翻v2"

# 阶段机标签（不落库，只进 detail；落库的是 status）
STAGE_BROKEN = "BROKEN"
STAGE_RECLAIMED = "RECLAIMED"
STAGE_CONFIRMED = "B1_CONFIRMED"
STAGE_INVALID = "INVALID"

_STAGE_CN = {
    STAGE_BROKEN: "已破底待收复",
    STAGE_RECLAIMED: "已收复待确认",
    STAGE_CONFIRMED: "B1已确认",
    STAGE_INVALID: "已失效",
}
_STAGE_RANK = {STAGE_BROKEN: 1, STAGE_RECLAIMED: 2, STAGE_CONFIRMED: 3, STAGE_INVALID: 0}
_STAGE_STATUS = {
    STAGE_BROKEN: STATUS_FORMING,
    STAGE_RECLAIMED: STATUS_FORMING,
    STAGE_CONFIRMED: STATUS_FORMED,
    STAGE_INVALID: STATUS_INVALID,
}

_VOL_FALLBACK = 1.0


def _safe(series, idx: int, default: float = 0.0) -> float:
    try:
        value = float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return default
    return default if np.isnan(value) else value


@dataclass
class V2Base:
    """v2 破底翻的结构快照。B1 之外还带 H1/TP 与失效信息，B2 复用同一份。"""

    l0_idx: int
    l0: float
    l1_idx: int
    l1: float
    depth: float
    break_idx: int  # 首个触发有效破底的根
    recover_idx: int | None  # 收复 L0 的那根（j）
    recover_used: int
    b1_idx: int | None
    b1_price: float
    stop: float
    h1_idx: int | None
    h1_price: float
    tp1_price: float
    tp2_price: float
    stage: str
    invalid_reason: str
    invalid_idx: int
    b1_vol_ratio: float

    @property
    def entry(self) -> float:
        return self.b1_price

    @property
    def rank_idx(self) -> int:
        """同票多候选排序用的"信号日"下标：越新越优先。"""
        if self.stage == STAGE_CONFIRMED:
            return self.b1_idx or self.l1_idx
        if self.stage == STAGE_RECLAIMED:
            return self.recover_idx or self.l1_idx
        if self.stage == STAGE_INVALID:
            return self.invalid_idx
        return self.l1_idx

    @property
    def rank_key(self) -> tuple:
        # 未失效优先 → 阶段靠后优先 → 信号日新优先
        return (
            0 if self.stage == STAGE_INVALID else 1,
            _STAGE_RANK[self.stage],
            self.rank_idx,
        )


def _atr_at(atr: np.ndarray, close: np.ndarray, idx: int) -> float:
    """第 idx 根的 ATR；取不到时退化成收盘价的 3%（与旧检测器同口径）。"""
    if 0 <= idx < len(atr):
        value = float(atr[idx])
        if np.isfinite(value) and value > 0:
            return value
    return float(close[idx]) * 0.03 if 0 <= idx < len(close) else 0.0


def _find_b1(
    close: np.ndarray, l0: float, atr: np.ndarray, c, start: int, last: int
) -> int | None:
    """B1 确认根：`close > L0 + k×ATR` 或连续 `b1_hold_days` 日 `close > L0`。"""
    hold = max(1, int(c.b1_hold_days))
    for i in range(start, last + 1):
        if close[i] > l0 + c.b1_close_atr_k * _atr_at(atr, close, i):
            return i
        if i - hold + 1 >= 0 and all(close[k] > l0 for k in range(i - hold + 1, i + 1)):
            return i
    return None


def _find_h1(
    pivots: list[Pivot], b1_idx: int, l0: float, atr_b1: float, c
) -> tuple[int, float] | None:
    """H1 = B1 之后首个已确认高点，高度满足 `L0×(1+h1_min_pct)` 或 `L0 + h1_min_atr_k×ATR`。"""
    threshold = max(l0 * (1 + c.h1_min_pct), l0 + c.h1_min_atr_k * atr_b1)
    for p in pivots:
        if p.kind == HIGH and p.confirmed and p.idx > b1_idx and p.price >= threshold:
            return p.idx, float(p.price)
    return None


def _dense_price_center(df, lookback: int, bins: int, last: int) -> float:
    """成交密集区中心：最近 lookback 根按价格分 bins 箱，取成交量最大的箱中心。"""
    start = max(0, last - lookback + 1)
    if last < start or bins < 2:
        return 0.0
    seg_close = df["close"].to_numpy(dtype=float)[start : last + 1]
    seg_vol = df["volume"].to_numpy(dtype=float)[start : last + 1]
    seg_close = seg_close[np.isfinite(seg_close)]
    if seg_close.size == 0:
        return 0.0
    lo, hi = float(seg_close.min()), float(seg_close.max())
    if hi <= lo:
        return lo
    edges = np.linspace(lo, hi, bins + 1)
    # digitize 后 1..bins，落到 min 上的点归第 1 箱
    which = np.clip(np.digitize(seg_close, edges) - 1, 0, bins - 1)
    totals = np.zeros(bins)
    for b, v in zip(which, seg_vol):
        totals[b] += v if np.isfinite(v) else 0.0
    best = int(totals.argmax())
    if totals[best] <= 0:
        return 0.0
    return float((edges[best] + edges[best + 1]) / 2)


def _tp2_candidates(ctx: ScanContext, c, last: int) -> list[float]:
    """TP2 候选池：确认摆动高点 ∪ 60 日高 ∪ 120 日高 ∪ MA120 ∪ 成交密集区中心。"""
    df = ctx.df
    high = df["high"].to_numpy(dtype=float)
    out: list[float] = []
    for p in ctx.pivots:
        if p.kind == HIGH and p.confirmed and p.price > 0:
            out.append(float(p.price))
    for window in (60, 120):
        start = max(0, last - window + 1)
        if start <= last:
            seg = high[start : last + 1]
            if seg.size:
                top = float(seg.max())
                if np.isfinite(top) and top > 0:
                    out.append(top)
    ma120 = _safe(df["ma120"], last, 0.0) if "ma120" in df.columns else 0.0
    if ma120 > 0:
        out.append(ma120)
    dense = _dense_price_center(df, c.tp2_lookback, c.tp2_dense_bins, last)
    if dense > 0:
        out.append(dense)
    return out


def _pick_tp2(candidates: list[float], h1_price: float) -> float:
    """严格高于 H1 的最近一档；没有则 0。"""
    above = [p for p in candidates if p > h1_price + 1e-9]
    return min(above) if above else 0.0


def _recent_swing_low(pivots: list[Pivot], last: int, fallback: float) -> float:
    for p in reversed(pivots):
        if p.kind == LOW and p.idx <= last and p.price > 0:
            return float(p.price)
    return fallback


def trailing_stop_ref(ctx: ScanContext, v2c, last: int, fallback: float) -> float:
    """新口径持仓的移动止盈参考位：`max(MA20 − k×ATR, 最近SwingLow − k×ATR)`。

    写进 `trailing_stop` 列供交易层与图表使用；取不到 MA20 时该项退化成 0，
    不会把 max 拉低（`max` 的候选取的都是正数）。
    """
    df = ctx.df
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float) if "atr" in df.columns else np.full(ctx.n, np.nan)
    atr_last = _atr_at(atr, close, last)
    ma20_last = _safe(df["ma20"], last, 0.0)
    swing_low = _recent_swing_low(ctx.pivots, last, fallback)
    candidates = [
        ma20_last - v2c.trail_ma_atr_k * atr_last if ma20_last > 0 else 0.0,
        swing_low - v2c.trail_swing_atr_k * atr_last,
    ]
    return max([v for v in candidates if v > 0] or [0.0])


def _find_v2_bases(ctx: ScanContext, c, max_l0_age: int | None = None) -> list[V2Base]:
    """枚举所有 L0 候选，返回通过闸门（含失效）的结构。失败原因逐条写进 trace。

    `max_l0_age` 覆盖 L0 的新鲜度上限（B2 允许更早的结构）；默认取 `l0_lookback`。
    """
    df, n = ctx.df, ctx.n
    if n < 30:
        ctx.note(f"[破底翻v2] K线仅{n}根，不足30根")
        return []
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    atr = (
        df["atr"].to_numpy(dtype=float)
        if "atr" in df.columns
        else np.full(n, np.nan)
    )
    last = n - 1

    l0_age = int(c.l0_lookback if max_l0_age is None else max_l0_age)
    l0_pivots = [p for p in ctx.pivots if p.kind == LOW and p.confirmed]
    l0_pivots.sort(key=lambda p: p.idx, reverse=True)  # 最近优先
    if not l0_pivots:
        ctx.note("[破底翻v2] 无已确认的低点摆动点")
        return []

    why = {
        "l0_old": 0, "fakelow": 0, "drop": 0, "nobreak": 0,
        "deep": 0, "shallow": 0, "badstop": 0,
    }
    out: list[V2Base] = []

    for p in l0_pivots:
        l0_idx = p.idx
        l0 = float(p.price)
        if l0 <= 0:
            continue
        # 1. L0 新鲜度
        if last - l0_idx > l0_age:
            why["l0_old"] += 1
            continue
        # 2. L0 必须是这段下跌的最低点（原始 low，插针砸出的更低点也算支撑已破）
        if c.l0_must_be_leg_low:
            start = max(0, l0_idx - c.prior_lookback)
            if float(low[start : l0_idx + 1].min()) < l0 - 1e-9:
                why["fakelow"] += 1
                continue
        # 3. 前置结构：L0 之前必须是一段下跌
        if prior_drop_pct(df, l0_idx, c.prior_lookback) < c.prior_drop:
            why["drop"] += 1
            continue

        # 4. 有效破底（盘中破也算）
        break_idx = next(
            (
                i
                for i in range(l0_idx + 1, n)
                if low[i] < l0 - c.break_low_atr_k * _atr_at(atr, close, i)
                or close[i] < l0 - c.break_close_atr_k * _atr_at(atr, close, i)
            ),
            None,
        )
        if break_idx is None:
            why["nobreak"] += 1
            continue

        # 5. L1 = 破底段最低点；深度须落在闸门内
        tail = low[l0_idx + 1 :]
        l1_idx = l0_idx + 1 + int(tail.argmin())
        l1 = float(tail.min())
        depth = (l0 - l1) / l0
        if depth > c.max_break_depth:
            why["deep"] += 1
            continue
        if depth < c.min_break_depth:
            why["shallow"] += 1
            continue

        # 6. 快速收复：自破底最低点起 recover_bars 根内收盘站回 L0
        recover_idx = next(
            (
                i
                for i in range(l1_idx + 1, min(l1_idx + c.recover_bars, last) + 1)
                if close[i] > l0
            ),
            None,
        )
        if recover_idx is None:
            if last - l1_idx < c.recover_bars:
                # 窗口还没走完，还有机会 → 构筑中
                out.append(
                    _make_base(
                        l0_idx, l0, l1_idx, l1, depth, break_idx, None,
                        None, 0.0, 0.0, None, 0.0, 0.0, 0.0,
                        STAGE_BROKEN, "", -1, 0.0,
                    )
                )
            else:
                # 超过 recover_bars 仍未收复 → 失效
                out.append(
                    _make_base(
                        l0_idx, l0, l1_idx, l1, depth, break_idx, None,
                        None, 0.0, 0.0, None, 0.0, 0.0, 0.0,
                        STAGE_INVALID,
                        f"破底后{c.recover_bars}日内未收复L0",
                        l1_idx + c.recover_bars,
                        0.0,
                    )
                )
            continue

        # 7. 跌破失败确认：收复后 fail_confirm_bars 根内不得再有效跌破
        #
        # 判据用**收盘价**而不是最低价。按定义的字面（"最低价 > L0−0.3×ATR"）写过一版，
        # 实测把 300353 东土科技判死了：确认窗口第 3 根（2026-09-23）盘中最低 16.35 跌到
        # 阈值 16.42 下方 0.07，可当天**收盘 16.89**（远在 L0=16.61 之上，且已 > L0+0.3ATR，
        # 本该直接成为 B1）——一根长阳里的插针不算"再次有效跌破"。
        # 用户 2026-09-24 裁定改用收盘价；严格相等不算跌破。
        fail_end = recover_idx + int(c.fail_confirm_bars)
        rebreak = next(
            (
                i
                for i in range(recover_idx + 1, min(fail_end, last) + 1)
                if close[i] < l0 - c.fail_confirm_atr_k * _atr_at(atr, close, i)
            ),
            None,
        )
        if rebreak is not None:
            out.append(
                _make_base(
                    l0_idx, l0, l1_idx, l1, depth, break_idx, recover_idx,
                    None, 0.0, 0.0, None, 0.0, 0.0, 0.0,
                    STAGE_INVALID, "收复后再次有效跌破L0", rebreak, 0.0,
                )
            )
            continue

        if fail_end > last:
            # 确认窗口还没走完 → 构筑中
            out.append(
                _make_base(
                    l0_idx, l0, l1_idx, l1, depth, break_idx, recover_idx,
                    None, 0.0, 0.0, None, 0.0, 0.0, 0.0,
                    STAGE_RECLAIMED, "", -1, 0.0,
                )
            )
            continue

        # 8. B1 确认
        b1_idx = _find_b1(close, l0, atr, c, fail_end, last)
        if b1_idx is None:
            out.append(
                _make_base(
                    l0_idx, l0, l1_idx, l1, depth, break_idx, recover_idx,
                    None, 0.0, 0.0, None, 0.0, 0.0, 0.0,
                    STAGE_RECLAIMED, "", -1, 0.0,
                )
            )
            continue

        atr_b1 = _atr_at(atr, close, b1_idx)
        entry = float(close[b1_idx])
        stop = max(l0 - c.stop_atr_k * atr_b1, l1 - c.extreme_stop_atr_k * atr_b1)
        if entry <= stop:
            # 确认价已经在止损下方，这笔交易没有可度量的风险
            why["badstop"] += 1
            continue

        # 9. 确认后不得跌破止损。定义里只要求"收复后 fail_confirm_bars 内不得再有效跌破"，
        #    但确认之后若收盘跌穿止损，这笔交易在交易层早就离场了，再报"已形成"是错的
        #    （实测 `bad_strict_close_back_below_l0`：B1=10.65 止损 9.93，最后两根收
        #    9.9/9.8 仍被判已形成）。失效位就用止损本身——它正是"结构在什么条件下证伪"
        #    的那条线，不另造一套判据。
        breach = next((i for i in range(b1_idx + 1, last + 1) if close[i] < stop), None)
        if breach is not None:
            out.append(
                _make_base(
                    l0_idx, l0, l1_idx, l1, depth, break_idx, recover_idx,
                    b1_idx, entry, stop, None, 0.0, 0.0, 0.0,
                    STAGE_INVALID, "B1确认后跌破止损位", breach, 0.0,
                )
            )
            continue

        h1 = _find_h1(ctx.pivots, b1_idx, l0, atr_b1, c)
        h1_idx, h1_price = h1 if h1 else (None, 0.0)
        tp1_price = h1_price if h1 else 0.0
        tp2_price = (
            _pick_tp2(_tp2_candidates(ctx, c, last), h1_price) if h1 else 0.0
        )

        vma_b1 = _safe(df["vol_ma5"], b1_idx, 0.0)
        b1_vol_ratio = (
            float(volume[b1_idx]) / vma_b1 if vma_b1 > 0 else _VOL_FALLBACK
        )
        out.append(
            _make_base(
                l0_idx, l0, l1_idx, l1, depth, break_idx, recover_idx,
                b1_idx, entry, stop, h1_idx, h1_price, tp1_price, tp2_price,
                STAGE_CONFIRMED, "", -1, b1_vol_ratio,
            )
        )

    if not out:
        detail = " ".join(f"{k}:{v}" for k, v in why.items() if v)
        ctx.note(
            f"[破底翻v2] {len(l0_pivots)}个L0候选均不成立（{detail or '无'}）"
        )
    return out


def _make_base(
    l0_idx, l0, l1_idx, l1, depth, break_idx, recover_idx,
    b1_idx, b1_price, stop, h1_idx, h1_price, tp1_price, tp2_price,
    stage, invalid_reason, invalid_idx, b1_vol_ratio,
) -> V2Base:
    recover_used = (recover_idx - l1_idx) if recover_idx is not None else 0
    return V2Base(
        l0_idx=l0_idx,
        l0=l0,
        l1_idx=l1_idx,
        l1=l1,
        depth=depth,
        break_idx=break_idx,
        recover_idx=recover_idx,
        recover_used=recover_used,
        b1_idx=b1_idx,
        b1_price=b1_price,
        stop=stop,
        h1_idx=h1_idx,
        h1_price=h1_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        stage=stage,
        invalid_reason=invalid_reason,
        invalid_idx=invalid_idx,
        b1_vol_ratio=b1_vol_ratio,
    )


def _emit_invalid(c, last: int, base: V2Base) -> bool:
    """失效行只在 `invalid_within` 根内发生、且开关打开时才输出。"""
    if not c.emit_invalid:
        return False
    if base.invalid_idx < 0:
        return False
    return (last - base.invalid_idx) <= int(c.invalid_within)


def _rr_grade(rr1: float, c) -> str:
    if rr1 <= 0:
        return ""
    if rr1 >= c.rr_value:
        return "高性价比"
    if rr1 >= c.rr_high:
        return "高质量"
    if rr1 >= c.rr_observe:
        return "观察"
    return "偏低"


def build_hit(ctx: ScanContext, c, base: V2Base) -> PatternHit:
    """把结构快照组装成 PatternHit（评分 + 关键价位 + 日期）。B1 / B2 共用。"""
    df, last = ctx.df, ctx.n - 1
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float) if "atr" in df.columns else np.full(ctx.n, np.nan)

    status = _STAGE_STATUS[base.stage]
    entry = base.b1_price
    risk = entry - base.stop if entry and base.stop else 0.0
    reward1 = base.tp1_price - entry if base.tp1_price else 0.0
    reward2 = base.tp2_price - entry if base.tp2_price else 0.0
    rr1 = reward1 / risk if risk > 0 and reward1 > 0 else 0.0
    rr2 = reward2 / risk if risk > 0 and reward2 > 0 else 0.0
    risk_pct = risk / entry * 100 if entry > 0 else 0.0
    reward1_pct = reward1 / entry * 100 if entry > 0 else 0.0
    reward2_pct = reward2 / entry * 100 if entry > 0 else 0.0

    # ---- 信号日 ----
    if base.stage == STAGE_CONFIRMED:
        signal_idx = base.b1_idx
    elif base.stage == STAGE_INVALID:
        signal_idx = base.invalid_idx
    else:
        signal_idx = last

    # ---- 移动止盈参考位（新口径持仓用） ----
    trailing_stop = trailing_stop_ref(ctx, c, last, base.l1)

    # ---- 子分 ----
    h1_pct = ((base.h1_price - base.l0) / base.l0) if base.h1_price else 0.0
    geom = np.mean(
        [
            taper(base.recover_used, 1, c.recover_bars),
            taper(base.depth, c.min_break_depth, c.max_break_depth),
            taper(h1_pct, c.h1_min_pct, 0.0),
            1.0 if base.stage == STAGE_CONFIRMED else 0.7,
        ]
    )

    vma_break = _safe(df["vol_ma5"], base.l1_idx, 0.0)
    break_vol_ratio = (
        float(volume[base.l1_idx]) / vma_break if vma_break > 0 else _VOL_FALLBACK
    )
    shrink = taper(break_vol_ratio, 0.6, 1.3)
    if base.b1_idx is not None:
        b1_ratio = base.b1_vol_ratio
    else:
        vma_last = _safe(df["vol_ma5"], last, 0.0)
        b1_ratio = float(volume[last]) / vma_last if vma_last > 0 else _VOL_FALLBACK
    expand = taper(b1_ratio, 1.5, 1.0)
    vol_score = 0.4 * shrink + 0.6 * expand

    hist_last = _safe(df["macd_hist"], last)
    hist_prev = _safe(df["macd_hist"], max(0, last - 1))
    divergence = _safe(df["macd_hist"], base.l1_idx) > _safe(df["macd_hist"], base.l0_idx)
    ma5 = _safe(df["ma5"], last)
    ma10 = _safe(df["ma10"], last)
    rsi_last = _safe(df["rsi"], last, 50.0)
    rsi_prev1 = _safe(df["rsi"], max(0, last - 1), 50.0)
    rsi_prev2 = _safe(df["rsi"], max(0, last - 2), 50.0)
    ma20_last = _safe(df["ma20"], last, 0.0)
    ma20_prev = _safe(df["ma20"], max(0, last - 5), ma20_last)
    trend = (
        0.15 * (1.0 if ma5 and close[last] > ma5 else 0.0)
        + 0.10 * (1.0 if ma10 and close[last] > ma10 else 0.0)
        + 0.15 * (1.0 if hist_last > hist_prev else 0.0)
        + 0.20 * (1.0 if divergence else 0.0)
        + 0.20 * (1.0 if not (rsi_last < rsi_prev1 < rsi_prev2) else 0.0)
        + 0.20 * (1.0 if ma20_last > ma20_prev > 0 else 0.0)
    )

    score, gs, vs, ts = blend(geom, vol_score, trend, status, ctx.cfg, priority="A")

    dates = {
        "L0": df.index[base.l0_idx],
        "L1": df.index[base.l1_idx],
    }
    if base.b1_idx is not None:
        dates["B1"] = df.index[base.b1_idx]
    if base.h1_idx is not None:
        dates["H1"] = df.index[base.h1_idx]

    key_levels = {
        "L0": base.l0,
        "L1": base.l1,
        "B1": base.b1_price,
        "H1": base.h1_price,
        "B2": 0.0,
        "stop_price": base.stop,
        "tp1_price": base.tp1_price,
        "tp2_price": base.tp2_price,
        "破底深度%": base.depth * 100,
        "收复用根数": float(base.recover_used),
        "risk_pct": risk_pct,
        "reward1_pct": reward1_pct,
        "reward2_pct": reward2_pct,
        "rr1": rr1,
        "rr2": rr2,
        "trailing_stop": trailing_stop,
    }

    stage_cn = _STAGE_CN[base.stage]
    grade = _rr_grade(rr1, c)
    detail = (
        f"L0={base.l0:.2f}({df.index[base.l0_idx]:%m-%d}) "
        f"L1={base.l1:.2f}(深{base.depth*100:.1f}%) "
        f"收复用{base.recover_used}根 {stage_cn}"
    )
    if base.b1_idx is not None:
        detail += (
            f" B1={base.b1_price:.2f}({df.index[base.b1_idx]:%m-%d}) "
            f"止损{base.stop:.2f} RR1={rr1:.2f}"
        )
        if base.h1_idx is not None:
            detail += f" H1={base.h1_price:.2f} TP2={base.tp2_price:.2f}"
        if grade:
            detail += f" [{grade}]"
        if divergence:
            detail += " MACD底背离"
    if base.invalid_reason:
        detail += f" 失效原因:{base.invalid_reason}"

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
        pattern_stage=base.stage,
        pattern_invalid=(status == STATUS_INVALID),
        invalid_reason=base.invalid_reason,
        key_level_dates=dates,
    )


def detect(ctx: ScanContext) -> list[PatternHit]:
    cfg = ctx.cfg
    c = cfg.break_bottom_v2
    if not c.enabled:
        return []

    bases = _find_v2_bases(ctx, c)
    if not bases:
        return []

    last = ctx.n - 1
    # 每票最多 1 行：未失效优先 → 阶段靠后优先 → 信号日新优先
    candidates = [
        b for b in bases if b.stage != STAGE_INVALID or _emit_invalid(c, last, b)
    ]
    if not candidates:
        ctx.note("[破底翻v2] 候选均不成立（失效行超出时效窗口）")
        return []

    best = max(candidates, key=lambda b: b.rank_key)

    if best.stage != STAGE_CONFIRMED and last - best.l1_idx > c.forming_max_age:
        ctx.note(
            f"[破底翻v2] 破底最低距今{last - best.l1_idx}根仍未确认"
            f"（> {c.forming_max_age}），超期丢弃"
        )
        return []

    ctx.note(
        f"[破底翻v2] L0={best.l0:.2f}({ctx.df.index[best.l0_idx]:%m-%d}) "
        f"L1={best.l1:.2f}(深{best.depth*100:.1f}%) 阶段={best.stage}"
    )
    return [build_hit(ctx, c, best)]
