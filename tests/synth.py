"""合成K线：构造理想形态与近似反例，用于验证检测规则本身。

wick 默认很小（0.2%），避免上下影线把通道容纳率和触碰判定搅乱——
这些测试要验证的是规则逻辑，不是抗噪能力。
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd


def ramp(start: float, end: float, bars: int) -> np.ndarray:
    """含端点的线性段，共 bars 根。"""
    return np.linspace(start, end, bars)


def make_ohlcv(
    closes,
    volumes=None,
    start: str = "2023-01-02",
    wick: float = 0.002,
) -> pd.DataFrame:
    """收盘价序列 → 完整OHLCV。开盘价取前一根收盘，上下影线按 wick 比例外扩。"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.bdate_range(start=start, periods=n)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    highs = np.maximum(opens, closes) * (1 + wick)
    lows = np.minimum(opens, closes) * (1 - wick)
    vol = np.full(n, 1_000_000.0) if volumes is None else np.asarray(volumes, dtype=float)

    return pd.DataFrame(
        {
            "open": opens,
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": vol,
            "amount": vol * closes,
            "turnover": np.full(n, 1.5),
            "pct_chg": np.concatenate(([0.0], np.diff(closes) / closes[:-1] * 100)),
        },
        index=idx,
    )


# ---------------- W底 ----------------

def ideal_w_bottom() -> pd.DataFrame:
    """前期下跌70根 → 左底13.0 → 颈线15.0 → 右底13.1 → 突破颈线。"""
    closes = np.concatenate(
        [
            ramp(20.0, 13.0, 70),     # 前期下跌 35%
            ramp(13.0, 15.0, 21)[1:],  # 左底 → 颈线
            ramp(15.0, 13.1, 21)[1:],  # 颈线 → 右底（缩量）
            ramp(13.1, 15.7, 9)[1:],   # 放量突破
        ]
    )
    volumes = np.concatenate(
        [
            np.full(70, 1.2e6),
            np.full(20, 1.0e6),
            np.full(20, 0.55e6),   # 右底缩量
            np.full(8, 2.4e6),     # 突破放量
        ]
    )
    return make_ohlcv(closes, volumes)


def bad_w_bottom_uneven_bottoms() -> pd.DataFrame:
    """右底15.0比左底13.0高15%，超出 bottom_tol，不该命中。"""
    closes = np.concatenate(
        [
            ramp(20.0, 13.0, 70),
            ramp(13.0, 17.0, 21)[1:],
            ramp(17.0, 15.0, 21)[1:],
            ramp(15.0, 17.6, 9)[1:],
        ]
    )
    return make_ohlcv(closes)


# ---------------- M头 ----------------

def ideal_m_top() -> pd.DataFrame:
    """前期上涨70根 → 左头20.0 → 颈线17.4 → 右头19.9 → 跌破颈线。"""
    closes = np.concatenate(
        [
            ramp(13.0, 20.0, 70),
            ramp(20.0, 17.4, 21)[1:],
            ramp(17.4, 19.9, 21)[1:],
            ramp(19.9, 16.6, 9)[1:],
        ]
    )
    volumes = np.concatenate(
        [
            np.full(70, 1.2e6),
            np.full(20, 1.0e6),
            np.full(20, 0.55e6),   # 右头缩量：追高动能衰减
            np.full(8, 1.8e6),
        ]
    )
    return make_ohlcv(closes, volumes)


def bad_m_top_no_prior_rise() -> pd.DataFrame:
    """双顶形状成立但前期没有上涨（横盘出来的双顶），不该命中。"""
    closes = np.concatenate(
        [
            ramp(19.9, 20.0, 70),      # 前期几乎不动
            ramp(20.0, 17.4, 21)[1:],
            ramp(17.4, 19.9, 21)[1:],
            ramp(19.9, 16.6, 9)[1:],
        ]
    )
    return make_ohlcv(closes)


# ---------------- 破底翻（严格定义：收盘破底 + 放量站上颈线） ----------------

class _StrictBB(NamedTuple):
    """严格破底翻骨架：收盘序列 + 由序列本身反推出的关键下标。"""

    closes: np.ndarray
    l0_idx: int      # 前低
    neck_idx: int    # 颈线（前低之后的反弹高点）
    low_idx: int     # 二次探底新低
    recover_idx: int  # 收盘收复前低
    breakout_idx: int  # 收盘放量站上颈线


def _strict_bb_bars(
    *,
    l0: float = 10.0,
    neck: float = 11.4,
    under: float = 9.6,
    drop_bars: int = 60,
    rebound_bars: int = 20,
    span: int = 45,
    recover_lag: int = 4,
    breakout_lag: int = 5,
    prior_dip: float | None = None,
    tail: Sequence[float] | None = None,
) -> _StrictBB:
    """拼出严格破底翻的完整走势。

    下跌 → 前低 L0 → 反弹到颈线 → 二次探底**收盘**浅破 L0 → 快速收复
    → 放量站上颈线 → tail（默认是回踩守住颈线的几根）。

    默认参数下：间隔 45 根（在 1~3 个月窗口内）、破底深度 4%、颈线高 14%、
    收复用 2 根（在 recover_bars=5 内）。关键下标一律按段长推得，
    不靠 argmin 反查——破了前低却没跌破前低 *收盘* 的样本里，
    区间最小值会落回前低那一根，argmin 会指错位置。

    prior_dip 给定时，下跌段先砸到该价位的深坑再反弹回 L0：L0 就成了
    "抬高的低点"，前面还有更低的价（反例，结构其余部分一字不改）。
    """
    up_leg = ramp(l0, neck, rebound_bars)[1:]  # L0 → 颈线
    down_leg = ramp(neck, under, span - rebound_bars + 1)  # 颈线 → 二次探底
    recover_leg = ramp(under, l0 * 1.04, recover_lag + 1)[1:]  # 快速止跌
    breakout_leg = ramp(l0 * 1.04, neck * 1.02, breakout_lag + 1)[1:]
    if tail is None:
        # 突破后回踩但收盘始终不破颈线
        tail = [neck * 1.008, neck * 1.004, neck * 1.005, neck * 1.009]
    if prior_dip is None:
        drop_leg = ramp(20.0, l0, drop_bars)
    else:
        dip_bars = drop_bars // 2
        up_bars = (drop_bars - dip_bars) // 2
        down_bars = drop_bars - dip_bars - up_bars
        drop_leg = np.concatenate(
            [
                ramp(20.0, prior_dip, dip_bars),
                ramp(prior_dip, neck * 1.10, up_bars + 1)[1:],  # 反弹过 L0 高度
                ramp(neck * 1.10, l0, down_bars + 1)[1:],  # 回落到 L0
            ]
        )
    closes = np.concatenate(
        [drop_leg, up_leg, down_leg, recover_leg, breakout_leg, tail]
    )
    l0_idx = drop_bars - 1
    neck_idx = l0_idx + int(np.argmax(closes[l0_idx : l0_idx + rebound_bars]))
    low_idx = neck_idx + len(down_leg)  # 二次探底最低那根（down_leg 首根与颈线同值）
    recover_idx = next(i for i in range(low_idx + 1, len(closes)) if closes[i] > l0)
    breakout_idx = next(
        i for i in range(recover_idx, len(closes)) if closes[i] > neck * 1.005
    )
    return _StrictBB(closes, l0_idx, neck_idx, low_idx, recover_idx, breakout_idx)


def _strict_bb_df(
    bars: _StrictBB,
    *,
    base: float = 1.0e6,
    break_vol: float = 0.5e6,
    spike: float = 2.6e6,
    volumes: np.ndarray | None = None,
) -> pd.DataFrame:
    """默认量能：破底段缩量、颈线突破日放量 2.6 倍（前 5 日均量的口径）。"""
    if volumes is None:
        volumes = np.full(len(bars.closes), base)
        volumes[max(0, bars.low_idx - 6) : bars.low_idx + 1] = break_vol
        volumes[bars.breakout_idx] = spike
    return make_ohlcv(bars.closes, volumes)


def _patch(df: pd.DataFrame, idx: int, column: str, value: float) -> pd.DataFrame:
    out = df.copy()
    out.iloc[idx, out.columns.get_loc(column)] = value
    return out


def strict_break_bottom(**kwargs) -> pd.DataFrame:
    """标准正例：破底 4%、间隔 45 根、3 根收复、放量站上颈线并回踩守住。"""
    return _strict_bb_df(_strict_bb_bars(**kwargs))


def strict_break_bottom_direct_breakout() -> pd.DataFrame:
    """放量直接突破颈线、不回踩，同样属于有效形态。"""
    bars = _strict_bb_bars(
        tail=list(ramp(11.62, 11.9, 4)[1:]),
    )
    return _strict_bb_df(bars)


def strict_break_bottom_forming_reclaim() -> pd.DataFrame:
    """刚收复 L0、还没摸到颈线 → 构筑中。"""
    bars = _strict_bb_bars()
    closes = bars.closes[: bars.recover_idx + 1]
    return make_ohlcv(closes)


def strict_break_bottom_forming_weak_volume() -> pd.DataFrame:
    """收盘站上颈线但突破日没放量 → 构筑中（等放量确认）。"""
    bars = _strict_bb_bars()
    volumes = np.full(len(bars.closes), 1.0e6)
    volumes[max(0, bars.low_idx - 6) : bars.low_idx + 1] = 0.5e6
    return _strict_bb_df(bars, volumes=volumes)


def bad_strict_break_too_deep() -> pd.DataFrame:
    """破底 14% > 10%，属真破位而非空头陷阱。"""
    return _strict_bb_df(_strict_bb_bars(under=8.6))


def bad_strict_break_too_shallow() -> pd.DataFrame:
    """破底只比前低低 1% < min_break_depth，属噪声级新低而非"有效跌破"。

    允许夹心摆动点之后，箱体震荡里这种 0.x% 的"新低"会大量凑成候选
    （实测近一成，典型如 300509 新美星：破底 0.8%、颈线只 +5.2%）。
    """
    return _strict_bb_df(_strict_bb_bars(under=9.9))


def bad_strict_span_over_2months() -> pd.DataFrame:
    """间隔 50 根 > max_span(45)，钉住"约2个月"这个上限。

    3 个月上限会把箱体里"陈旧前低 + 三个月后又创个低点"收进来
    （典型如 002099 海翔药业：前低 06-10、间隔 53 根、破底深 2.0%）。
    """
    return _strict_bb_df(_strict_bb_bars(span=50))


def bad_strict_span_too_short() -> pd.DataFrame:
    """前低与新低只隔 15 根，不足 1 个月。"""
    return _strict_bb_df(_strict_bb_bars(span=15, rebound_bars=8))


def bad_strict_span_too_long() -> pd.DataFrame:
    """前低与新低隔 70 根，超过 3 个月（时间上已经不是同一个诱空结构）。"""
    return _strict_bb_df(_strict_bb_bars(span=70))


def intraday_only_undercut_break() -> pd.DataFrame:
    """只有盘中最低破了前低、收盘从未跌破 —— **算**有效破底。

    破底取 low（影线口径）：插针砸穿支撑同样是支撑被破。v1/v2 现在是同一套口径
    （v2 从落地起就这么判）。
    """
    bars = _strict_bb_bars(under=10.02)
    df = _strict_bb_df(bars)
    return _patch(df, bars.low_idx, "low", 9.5)


def bad_strict_no_recover() -> pd.DataFrame:
    """破底后在低位磨蹭，recover_bars 内没有收盘收复 L0。"""
    bars = _strict_bb_bars()
    closes = bars.closes.copy()
    closes[bars.low_idx + 1 : bars.low_idx + 1 + 5] = np.linspace(9.62, 9.9, 5)
    return _strict_bb_df(bars._replace(closes=closes))


def bad_strict_higher_low_before_l0() -> pd.DataFrame:
    """前低之前还有更低的低点：L0 只是下跌途中的中间低点，"破底"没破到真支撑。

    下跌段先砸到 8.8 再反弹过 L0 的高度、回落到 L0 = 10.0，其余结构一字不改：
    间隔、深度、颈线高度、前置跌幅、收盘破位、快速收复全都合规，
    唯一的毛病就是 L0 不是这段下跌的最低点。
    """
    return _strict_bb_df(_strict_bb_bars(prior_dip=8.8))


def _strict_bb_bars_gapped() -> _StrictBB:
    """前低与二次低点之间夹着「更高的低点 + 更低的高点」的破底翻骨架。

    真实走势里这种夹心结构比"一头栽到新低"常见得多：反弹见顶后先回调出一个
    更高的低点、再弹一次（没能回到颈线高度），然后才砸穿前低。相邻摆动点口径下
    真前低与二次低点被这两个夹心摆动隔开，(前低, 颈线, 新低) 三元组根本凑不出来，
    只能凑出夹心的那个更高低点——而它到新低只隔 11 根，会被 min_span 挡掉，
    于是整个形态静默漏报。东土科技 300353 就是这个形状
    （16.61 → 20.99 → 17.77 → 19.15 → 15.66）。
    """
    drop_leg = ramp(20.0, 10.0, 60)      # 前期下跌 → 前低 L0
    up_leg = ramp(10.0, 11.4, 20)[1:]    # L0 → 颈线（区间内最高）
    down_a = ramp(11.4, 10.4, 9)[1:]     # 颈线 → 夹心的更高低点
    bounce = ramp(10.4, 11.0, 6)[1:]     # 夹心反弹（更低的高点）
    down_b = ramp(11.0, 9.6, 7)[1:]      # → 二次探底新低
    recover_leg = ramp(9.6, 10.4, 5)[1:]  # 3 根内收盘收复 L0
    breakout_leg = ramp(10.4, 11.628, 6)[1:]  # 放量站上颈线
    tail = [11.50, 11.45, 11.47, 11.52]
    closes = np.concatenate(
        [drop_leg, up_leg, down_a, bounce, down_b, recover_leg, breakout_leg, tail]
    )
    l0_idx = 59
    neck_idx = l0_idx + len(up_leg)
    low_idx = neck_idx + len(down_a) + len(bounce) + len(down_b)
    recover_idx = next(i for i in range(low_idx + 1, len(closes)) if closes[i] > closes[l0_idx])
    breakout_idx = next(
        i for i in range(recover_idx, len(closes)) if closes[i] > closes[neck_idx] * 1.005
    )
    return _StrictBB(closes, l0_idx, neck_idx, low_idx, recover_idx, breakout_idx)


def strict_break_bottom_gapped_pivots() -> pd.DataFrame:
    """前低与二次低点之间夹着其它摆动点的破底翻 —— 仍须命中。"""
    return _strict_bb_df(_strict_bb_bars_gapped())


def bad_strict_close_back_below_l0() -> pd.DataFrame:
    """站上颈线后又连跌两日收在前低下方 → 形态失效。"""
    bars = _strict_bb_bars()
    closes = bars.closes.copy()
    closes[-2:] = [9.9, 9.8]
    return _strict_bb_df(bars._replace(closes=closes))


def bad_strict_close_breaks_before_neck() -> pd.DataFrame:
    """L0 与颈线之间就先收盘跌破过 L0 → 这道支撑没撑住，"颈线"只是下跌途中的反弹。

    骨架：下跌到 L0=10.0 → 小反弹到 10.7（确认 L0 是摆动低点）→ 跌破 L0 到 9.80
    → 反弹到颈线 11.4 → 二次探底 9.75（浅破 2.5%）→ 快速收复 → 放量站上颈线。

    其余全合规：L0 是下跌段最低点（fakelow 过）、区间真低就是 9.75（"L1 必须是区间
    最低"也过）、深度/间隔/颈线高度/收盘破位/快速收复都过。唯一的毛病是首次收盘
    跌破 L0 发生在颈线**之前**（002030 达安基因的形状）。

    夹心那个 9.80 低点故意做成不合格的候选：它到 9.75 只破 0.5%（< min_break_depth），
    到 L0 又只隔 7 根（< min_span），所以整个样本只有"L0 → 颈线 → 9.75"这一条
    三元组能过其余闸门——规则一失效，样本就命中。
    """
    drop_leg = ramp(20.0, 10.0, 60)        # → 前低 L0 = 10.0（idx 59）
    confirm = ramp(10.0, 10.7, 4)[1:]      # 小反弹，把 L0 确认为摆动低点
    early_dip = ramp(10.7, 9.80, 5)[1:]    # ← 先跌破 L0（收盘 9.80 < 10.0）
    to_neck = ramp(9.80, 11.4, 16)[1:]     # 反弹到颈线 11.4
    to_low = ramp(11.4, 9.75, 20)[1:]      # 二次探底 9.75（浅破 2.5%）
    recover_leg = ramp(9.75, 10.4, 4)[1:]  # 快速收复
    breakout_leg = ramp(10.4, 11.63, 5)[1:]  # 放量站上颈线
    tail = [11.55, 11.50, 11.52, 11.58]
    closes = np.concatenate(
        [drop_leg, confirm, early_dip, to_neck, to_low, recover_leg, breakout_leg, tail]
    )
    return _strict_bb_df(_StrictBB(closes, 59, 81, 100, 102, 107))


def bad_strict_relapse_after_recover() -> pd.DataFrame:
    """收复 L0 之后又连收两根在 L0 下方（随后再翻回来）→ 结构作废。

    与 `bad_strict_close_back_below_l0` 的关键区别：**最后两根是收在 L0 上方**的。
    detect() 末尾的失效检查只看最后两根，看不到中间那次"翻失败"，全靠结构定位
    阶段的"收复后连续 N 根跌破 L0"挡住（300354 东华测试：08-26 收复 L0=26.59，
    09-10/09-11 连收 26.55/25.80，到 09-16 又收 26.95 把末尾检查绕了过去）。
    """
    drop_leg = ramp(20.0, 10.0, 60)         # → 前低 L0 = 10.0（idx 59）
    up_leg = ramp(10.0, 11.4, 20)[1:]       # → 颈线 11.4（idx 78）
    down_leg = ramp(11.4, 9.6, 26)          # → 二次探底 9.6（浅破 4%，idx 104）
    reclaim = [9.80, 9.95, 10.30]           # 收复 L0（首根收上 10.0 的在 idx 107）
    relapse = [9.90, 9.85]                  # ← 又连收两根在 L0 下方
    second = ramp(9.85, 10.6, 4)[1:]        # 再翻回来
    breakout_leg = ramp(10.6, 11.7, 6)[1:]  # 放量站上颈线
    tail = [11.60, 11.55, 11.58]            # 最后两根都在 L0 上方
    closes = np.concatenate(
        [drop_leg, up_leg, down_leg, reclaim, relapse, second, breakout_leg, tail]
    )
    # 二次探底缩量、颈线突破日放量（与 _strict_bb_df 同一套量能口径）
    low_idx = 104
    breakout_idx = next(i for i in range(108, len(closes)) if closes[i] > 11.4 * 1.005)
    volumes = np.full(len(closes), 1.0e6)
    volumes[max(0, low_idx - 6) : low_idx + 1] = 0.5e6
    volumes[breakout_idx] = 2.6e6
    return make_ohlcv(closes, volumes)


def bad_strict_no_prior_drop() -> pd.DataFrame:
    """前低之前只跌了 10%，不构成"明确的下跌趋势"前置结构。"""
    bars = _strict_bb_bars()
    closes = bars.closes.copy()
    closes[: bars.l0_idx + 1] = ramp(11.0, 10.0, bars.l0_idx + 1)
    return _strict_bb_df(bars._replace(closes=closes))


# ---------------- 破底翻第二买点（B2） ----------------

def _strict_b2_frame():
    """严格破底翻基座 + B2 尾巴：颈线放量突破 → 第一波 → 缩量回踩 → 放量破前高。

    返回 (DataFrame, 关键下标)。H1/L1 的下标由构造段长推得，量能按段铺设——
    与检测器用 ZigZag 找出的 H1/L1 一致（测试里另有断言守住这一点）。

    新规范（B2 严格定义）要求：
    1. B1 确认：Close > N + 0.2×ATR
    2. H1 - B1 >= max(5%, 1.5×ATR)
    3. L1 在颈线支撑区 [N-0.5×ATR, N+0.3×ATR]
    4. 回踩幅度 20-50% 最佳（>70% 降级但仍可接受）

    第一波涨幅 16%，回踩到颈线支撑区，回撤幅度 113%（过深但可接受）。
    B2 确认后涨幅控制在 8% 以内，避免触发追高拒绝。
    """
    bars = _strict_bb_bars(tail=[])
    h1, l1 = 13.5, 11.4  # H1 更高，L1 在颈线支撑区内
    bo = bars.breakout_idx
    wave1 = ramp(bars.closes[bo], h1, 15)[1:]   # 颈线 → 第一波高点（更长涨幅）
    pull = ramp(h1, l1, 10)[1:]                 # 缩量回踩到 L1（颈线支撑区）
    surge = ramp(l1, l1 * 1.06, 8)[1:]          # 放量突破（涨幅控制在 6%）
    closes = np.concatenate([bars.closes, wave1, pull, surge])
    h1_idx = bo + len(wave1)
    l1_idx = h1_idx + len(pull)

    volumes = np.full(len(closes), 1.0e6)
    volumes[max(0, bars.low_idx - 6) : bars.low_idx + 1] = 0.5e6  # 破底缩量
    volumes[bo] = 2.6e6                                           # 颈线放量突破
    volumes[bo + 1 : h1_idx + 1] = 1.2e6                          # 第一波
    volumes[h1_idx + 1 : l1_idx + 1] = 0.6e6                      # 回踩缩量
    volumes[l1_idx + 1 :] = 2.5e6                                 # 放量突破
    idx = {"bo": bo, "h1": h1_idx, "l1": l1_idx,
           "l0": bars.l0_idx, "low": bars.low_idx, "n": len(closes)}
    return make_ohlcv(closes, volumes), idx


def ideal_break_bottom_b2() -> pd.DataFrame:
    """完整 B2：破底翻 → 第一波反弹 → 缩量回踩不破 L0 → 放量突破前高。"""
    return _strict_b2_frame()[0]


def bad_b2_pullback_breaks_l0() -> pd.DataFrame:
    """回踩直接跌破 L0（连续 2 日收盘在 L0 下方），结构破坏。"""
    df, idx = _strict_b2_frame()
    out = df.copy()
    out.iloc[idx["h1"] + 1 : idx["l1"] + 2, out.columns.get_loc("close")] = 9.8
    return out


def bad_b2_no_volume_shrink() -> pd.DataFrame:
    """回踩不缩量（回踩段与第一波等量），不符合 B2 条件。"""
    df, idx = _strict_b2_frame()
    out = df.copy()
    # 除颈线突破与突破前高两根外全部等量 → 回踩量 = 第一波量
    out.iloc[idx["bo"] + 1 : idx["l1"] + 1, out.columns.get_loc("volume")] = 1.2e6
    return out


def bad_b2_chase_too_late() -> pd.DataFrame:
    """突破前高后继续大涨超 8%，不追。"""
    bars = _strict_bb_bars(tail=[])
    h1, l1 = 13.5, 11.4
    bo = bars.breakout_idx
    wave1 = ramp(bars.closes[bo], h1, 15)[1:]
    pull = ramp(h1, l1, 10)[1:]
    closes = np.concatenate([bars.closes, wave1, pull, ramp(l1, h1 * 1.25, 20)[1:]])
    h1_idx = bo + len(wave1)
    l1_idx = h1_idx + len(pull)
    volumes = np.full(len(closes), 1.0e6)
    volumes[max(0, bars.low_idx - 6) : bars.low_idx + 1] = 0.5e6
    volumes[bo] = 2.6e6
    volumes[bo + 1 : h1_idx + 1] = 1.2e6
    volumes[h1_idx + 1 : l1_idx + 1] = 0.6e6
    volumes[l1_idx + 1 :] = 2.5e6
    return make_ohlcv(closes, volumes)


def bad_b2_breaks_l0_after_l1() -> pd.DataFrame:
    """L1 之后价格又把 L0 跌回去（连续收盘在 L0 下方），结构已破坏。

    结构闸门以前只查到 L1 为止，L1 之后到当日这一段没人管，这种形态会以
    "构筑中"进池——而它的止损（锚在 L1）此时已经落在现价上方。
    """
    bars = _strict_bb_bars(tail=[])
    bo = bars.breakout_idx
    wave1 = ramp(bars.closes[bo], 13.5, 15)[1:]
    pull = ramp(13.5, 11.4, 10)[1:]
    h1_idx = bo + len(wave1)
    l1_idx = h1_idx + len(pull)
    closes = np.concatenate([bars.closes, wave1, pull, [10.0, 9.9, 9.85]])
    volumes = np.full(len(closes), 1.0e6)
    volumes[max(0, bars.low_idx - 6) : bars.low_idx + 1] = 0.5e6
    volumes[bo] = 2.6e6
    volumes[bo + 1 : h1_idx + 1] = 1.2e6
    volumes[h1_idx + 1 : l1_idx + 1] = 0.6e6
    return make_ohlcv(closes, volumes)


def b2_pullback_volume_boundary() -> pd.DataFrame:
    """回踩量刚好卡在缩量闸门附近，且 H1 当根是放量长阳。

    量能口径是 第一波 = [B1, H1]，回踩 = (H1, L1]。H1 那根放量长阳只算进
    第一波；若把 H1 也算进回踩均量，这个样本会被误判成"回踩没缩量"。
    """
    df, idx = _strict_b2_frame()
    out = df.copy()
    vol_col = out.columns.get_loc("volume")
    out.iloc[idx["bo"], vol_col] = 2.6e6
    out.iloc[idx["bo"] + 1 : idx["h1"] + 1, vol_col] = 1.0e6   # 第一波
    out.iloc[idx["h1"], vol_col] = 3.0e6                        # H1 当根放量长阳
    # 回踩量刚好卡在闸门内：回踩均量 <= 第一波均量 * 0.85
    # 第一波均量 = (14 * 1.0e6 + 3.0e6) / 15 = 1.2e6
    # 回踩均量需 <= 1.2e6 * 0.85 = 1.02e6
    out.iloc[idx["h1"] + 1 : idx["l1"] + 1, vol_col] = 0.8e6    # 回踩缩量
    out.iloc[idx["l1"] + 1 :, vol_col] = 2.5e6
    return out

# ---------------- 通道 ----------------

def ideal_rising_channel(bars: int = 130, period: int = 20) -> pd.DataFrame:
    """稳定上行 + 规律振荡，上下轨各被反复触碰。"""
    t = np.arange(bars, dtype=float)
    trend = 10.0 * (1 + 0.004 * t)
    closes = trend + 0.45 * np.sin(2 * np.pi * t / period)
    volumes = 1.0e6 + 3.0e5 * np.sign(np.concatenate(([0.0], np.diff(closes))))
    return make_ohlcv(closes, volumes)


def ideal_falling_channel(bars: int = 130, period: int = 20) -> pd.DataFrame:
    t = np.arange(bars, dtype=float)
    trend = 20.0 * (1 - 0.004 * t)
    closes = trend + 0.45 * np.sin(2 * np.pi * t / period)
    volumes = 1.0e6 - 3.0e5 * np.sign(np.concatenate(([0.0], np.diff(closes))))
    return make_ohlcv(closes, volumes)


def ideal_box_range(bars: int = 130, period: int = 20) -> pd.DataFrame:
    t = np.arange(bars, dtype=float)
    closes = 15.0 + 0.9 * np.sin(2 * np.pi * t / period)
    return make_ohlcv(closes)


def bad_channel_too_narrow(bars: int = 130) -> pd.DataFrame:
    """一条几乎笔直的线：回归拟合完美但通道宽度不足，不该当作震荡通道。"""
    closes = 10.0 * (1 + 0.004 * np.arange(bars, dtype=float))
    return make_ohlcv(closes)


def triangle_wave(bars: int = 120, period: int = 30, amp: float = 0.15) -> pd.DataFrame:
    """三角波，用于验证 ZigZag 的峰谷交替。"""
    t = np.arange(bars, dtype=float)
    saw = 2 * np.abs((t / period) % 1 - 0.5)  # 0→1→0
    closes = 10.0 * (1 + amp * saw)
    return make_ohlcv(closes)


# ---------------- 破底翻 v2（ATR 严格口径） ----------------
#
# v2 的结构骨架与旧口径不同：不要求"放量站上颈线"，而是看
# L0 → 有效破底（ATR 阈值，盘中破也算）→ 快速收复 → 跌破失败确认 → B1 确认。
# 所以这里不复用 `_strict_bb_bars`（它是按颈线放量突破搭的），单独搭一套。


class _V2BB(NamedTuple):
    closes: np.ndarray
    l0_idx: int  # 前低 L0
    l1_idx: int  # 破底最低点
    recover_idx: int  # 收盘收复 L0
    b1_idx: int  # B1 确认根
    h1_peak_idx: int  # B1 之后第一波高点（ZigZag 会确认成 H1）


def _v2_bars(
    *,
    l0: float = 10.0,
    under: float = 9.6,
    drop_from: float = 12.0,  # 前置下跌起点（须使 L0 之前跌过 15%）
    drop_bars: int = 60,
    rally_bars: int = 10,  # L0 → 反弹高点（不构成 H1）
    down_bars: int = 12,
    recover_len: int = 2,  # 收复用几根（须 ≤ recover_bars=5）
    confirm_bars: int = 3,
    confirm_dip: float | None = None,  # 给定时把确认段首根收盘砸到该价（反例）
    rally_to: float = 11.6,  # B1 之后第一波高点（H1）
    fall_to: float = 10.7,
    fall_bars: int = 8,  # 从 H1 回落，幅度须 > 5% 才能让 ZigZag 确认这个高点
    tail_bars: int = 3,
) -> _V2BB:
    """拼出 v2 破底翻的完整走势。

    默认参数：前置下跌 16.7%（> prior_drop）、破底深度 4.2%（∈[1.5%,10%]）、
    收复用 2 根（≤5）、确认 3 根不破 L0、B1 当日收盘 10.25、随后第一波到 11.6。
    """
    drop = ramp(drop_from, l0, drop_bars)
    rally = ramp(l0, l0 * 1.08, rally_bars)[1:]
    down = ramp(l0 * 1.08, under, down_bars)[1:]
    recover = ramp(under, l0 * 1.02, recover_len + 1)[1:]
    confirm = np.full(confirm_bars, l0 * 1.025)
    if confirm_dip is not None and confirm_bars:
        confirm[0] = confirm_dip
    b1 = np.array([l0 * 1.04])
    rally2 = ramp(l0 * 1.04, rally_to, 8)[1:]
    fall = ramp(rally_to, fall_to, fall_bars)[1:]
    tail = np.full(tail_bars, fall_to * 0.995)
    closes = np.concatenate([drop, rally, down, recover, confirm, b1, rally2, fall, tail])

    l0_idx = drop_bars - 1
    l1_idx = l0_idx + len(rally) + len(down)
    recover_idx = next(i for i in range(l1_idx + 1, len(closes)) if closes[i] > l0)
    # 确认段最后一根就是检测器认定的 B1（它同样满足 close > L0 + 0.3×ATR）；
    # 其后紧跟的 `b1` 那一根是"确认价为 L0×1.04"的显式买点，再往后才是第一波 rally2。
    b1_idx = l1_idx + len(recover) + confirm_bars
    h1_peak_idx = b1_idx + 1 + len(rally2)
    return _V2BB(closes, l0_idx, l1_idx, recover_idx, b1_idx, h1_peak_idx)


def ideal_v2_break_bottom(**kwargs) -> pd.DataFrame:
    """标准正例：B1 已确认 + H1 已形成（tp1/tp2 都有值）。"""
    return make_ohlcv(_v2_bars(**kwargs).closes)


def v2_broken_forming() -> pd.DataFrame:
    """刚破底、收复窗口还没走完 → 构筑中（BROKEN）。"""
    bars = _v2_bars()
    return make_ohlcv(bars.closes[: bars.l1_idx + 1])


def v2_reclaimed_forming() -> pd.DataFrame:
    """已收复 L0、确认窗口还没走完 → 构筑中（RECLAIMED）。"""
    bars = _v2_bars()
    return make_ohlcv(bars.closes[: bars.recover_idx + 1])


def bad_v2_rebreak_in_confirm(**kwargs) -> pd.DataFrame:
    """收复后确认窗口内又有效跌破 L0 → 失效（假收复）。

    确认段首根收盘砸到 9.90：最低 ~9.88 < L0 − 0.3×ATR(9.93)，触发失效判据。
    """
    bars = _v2_bars(confirm_dip=9.90, **kwargs)
    return make_ohlcv(bars.closes)


def v2_confirm_intrabar_dip() -> pd.DataFrame:
    """确认窗口内一根**盘中**破了 L0 − 0.3×ATR，收盘却站得很稳（300353 的形状）。

    判据按收盘价，这种插针不算"再次有效跌破"——该结构反而是标准的 B1。
    300353 东土科技 2026-09-23：盘中最低 16.35 < 阈值 16.42，收盘 16.89 > L0=16.61。
    """
    bars = _v2_bars()
    df = make_ohlcv(bars.closes)
    # 确认段最后一根（= 检测器认定的 B1）压一根长下影，收盘不动
    return _patch(df, bars.b1_idx, "low", bars.closes[bars.b1_idx] * 0.96)


def bad_v2_no_recover_old() -> pd.DataFrame:
    """破底后在低位磨蹭，5 根内没有收盘收复 L0 → 失效（超时）。"""
    bars = _v2_bars()
    closes = bars.closes.copy()
    closes[bars.l1_idx + 1 : bars.l1_idx + 6] = np.linspace(9.62, 9.9, 5)
    return make_ohlcv(closes)


# ---------------- 破底翻 v2 第二买点（B2） ----------------


class _V2B2(NamedTuple):
    closes: np.ndarray
    volumes: np.ndarray
    pl_idx: int  # 回踩低点
    b2_idx: int  # 突破 H1 再启动


def _v2_b2_bars(
    *,
    pull_to: float = 10.9,  # 回踩低点（须 > L0 − 0.3×ATR ≈ 9.93，且回撤 ≥ 5% 才能确认 H1）
    pull_bars: int = 6,
    rally_to: float = 12.3,  # 再启动高点
    rally_bars: int = 9,
    pull_vol: float = 0.6e6,  # 回踩段量（须 ≤ 第一波量 × 0.85）
    spike_vol: float = 2.0e6,  # 突破 H1 那根的量
) -> _V2B2:
    """在 `_v2_bars` 的 H1 峰值之后接「回踩 → 放量破 H1」。

    回踩幅度取 6%（11.6 → 10.9）：既要 ≥ 5% 才能让 ZigZag 确认 H1，
    又要回踩低点站得住 L0 上方（10.9 > 9.93）。
    """
    bars = _v2_bars()
    prefix = bars.closes[: bars.h1_peak_idx + 1]
    h1_peak = float(bars.closes[bars.h1_peak_idx])
    pull = ramp(h1_peak, pull_to, pull_bars)[1:]
    rally = ramp(pull_to, rally_to, rally_bars + 1)[1:]
    closes = np.concatenate([prefix, pull, rally])

    volumes = np.full(len(closes), 1.0e6)
    pl_idx = bars.h1_peak_idx + len(pull)
    volumes[bars.h1_peak_idx + 1 : pl_idx + 1] = pull_vol  # 回踩缩量
    # 突破日放量：阈值与检测器一致——H1 的摆动点价是 high（= 收盘 ×(1+wick)），
    # 检测器要求收盘 > H1×(1+buffer)。用别的阈值打点会出现"放量打在上一根"、
    # 检测器在下一根找不到量能而不认突破。
    b2_threshold = h1_peak * (1 + 0.002) * 1.005
    b2_idx = -1
    for i in range(pl_idx + 1, len(closes)):
        if closes[i] >= b2_threshold:
            b2_idx = i
            break
    if b2_idx >= 0:
        for i in range(b2_idx, min(b2_idx + 2, len(closes))):
            volumes[i] = spike_vol
    return _V2B2(closes, volumes, pl_idx, b2_idx)


def ideal_v2_b2() -> pd.DataFrame:
    """标准正例：v2 B1 已确认 + H1 已形成 + 回踩缩量不破 L0 + 放量破 H1。"""
    frame = _v2_b2_bars()
    return make_ohlcv(frame.closes, frame.volumes)


def v2_b2_forming() -> pd.DataFrame:
    """回踩已完成（ZigZag 已确认这个低点）、还没突破 H1 → 构筑中。

    回踩取 9.5%（11.6 → 10.5）：深一点是为了让"pl 被确认"（需自低点反弹 5%）
    与"突破 H1"（需收盘 > 11.66）之间留出足够宽的窗口，构造出真的还没突破的状态。
    """
    frame = _v2_b2_bars(pull_to=10.5, rally_to=11.3)
    return make_ohlcv(frame.closes, frame.volumes)


def bad_v2_b2_pullback_breaks_l0() -> pd.DataFrame:
    """回踩有效跌破 L0（10.9 → 9.7 < 9.93）→ 这里不存在 B2。"""
    frame = _v2_b2_bars(pull_to=9.7)
    return make_ohlcv(frame.closes, frame.volumes)


def bad_v2_b2_no_shrink() -> pd.DataFrame:
    """回踩不但没缩量、还比第一波放量 → 不符合"缩量回踩"。"""
    frame = _v2_b2_bars(pull_vol=2.0e6)
    return make_ohlcv(frame.closes, frame.volumes)


def bad_v2_b2_chase_too_late() -> pd.DataFrame:
    """突破 H1 后已经涨超 chase_abort_pct(8%) → 不追。"""
    frame = _v2_b2_bars(rally_to=13.6)
    return make_ohlcv(frame.closes, frame.volumes)


