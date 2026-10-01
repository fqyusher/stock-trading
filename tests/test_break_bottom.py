"""破底翻检测（严格定义：收盘破底 → 快速收复 → 放量收盘站上颈线）。"""

from __future__ import annotations

import numpy as np
import pytest
import synth
from conftest import assert_hit, assert_no_hit

from tradeyh.patterns import break_bottom
from tradeyh.patterns.base import STATUS_FORMED, STATUS_FORMING


def test_strict_break_bottom_is_formed(make_ctx, cfg):
    """标准形态：浅破前低 → 3 根内收复 → 放量站上颈线并回踩守住。"""
    ctx = make_ctx(synth.strict_break_bottom())
    hit = assert_hit(ctx, break_bottom, STATUS_FORMED)

    assert hit.pattern == "break_bottom"
    assert hit.score > 55, f"理想破底翻评分应偏高，实际 {hit.score:.1f} ({hit.detail})"

    levels = hit.key_levels
    assert levels["破底最低"] < levels["参考低点"], "二次探底必须创新低"
    assert 0 < levels["破底深度%"] <= 100 * cfg.break_bottom.max_break_depth
    assert levels["收复价"] > levels["参考低点"], "已形成状态下应已收复参考低点"
    assert levels["颈线"] > levels["参考低点"], "颈线是前低之上的反弹高点"
    # 已形成的止损挂在颈线下方（结构失效位是颈线，不是前低）：
    # 挂到 L0 下方会让止损比入场低 颈线高度+0.5×ATR，仓位与 RR 都被压垮
    assert levels["参考止损"] < levels["颈线"], "已形成止损应挂在颈线下方"
    assert levels["参考止损"] > levels["参考低点"], "已形成止损不应再落到前低下方"
    assert (
        cfg.break_bottom.neck_breakout_vol_ratio
        <= levels["突破量比"]
    ), "已形成必须带放量突破"
    assert "保守买点" in hit.detail or "颈线突破" in hit.detail


def test_forming_stop_is_below_l0(make_ctx):
    """构筑中（尚未确认）的结构失效以"跌回前低下方"为准，止损挂在 L0 下方。"""
    ctx = make_ctx(synth.strict_break_bottom_forming_reclaim())
    hit = assert_hit(ctx, break_bottom, STATUS_FORMING)
    assert hit.key_levels["参考止损"] < hit.key_levels["参考低点"]


def test_target_is_neckline_measured_move(make_ctx):
    """目标价 = 颈线 + (颈线 − 破底最低)。确认价已在颈线之上，目标必须更远。"""
    hit = assert_hit(make_ctx(synth.strict_break_bottom()), break_bottom, STATUS_FORMED)
    levels = hit.key_levels
    expected = levels["颈线"] + (levels["颈线"] - levels["破底最低"])
    assert levels["上方压力"] == pytest.approx(expected)
    assert levels["上方压力"] > hit.close_price, "目标价必须在现价之上，否则预期收益恒为 0"


def test_direct_breakout_without_pullback_is_formed(make_ctx):
    """放量直接突破颈线、不回踩，同样属于有效形态。"""
    ctx = make_ctx(synth.strict_break_bottom_direct_breakout())
    assert_hit(ctx, break_bottom, STATUS_FORMED)


def test_reclaiming_l0_without_neckline_is_forming(make_ctx):
    """已收复 L0、还没摸到颈线 → 构筑中（观察池的主要对象）。"""
    ctx = make_ctx(synth.strict_break_bottom_forming_reclaim())
    hit = assert_hit(ctx, break_bottom, STATUS_FORMING)
    assert "激进买点" in hit.detail or "收复支撑" in hit.detail
    assert hit.key_levels["收复价"] > hit.key_levels["参考低点"]
    assert hit.close_price < hit.key_levels["颈线"], "收盘尚未站上颈线"


def test_neckline_breakout_requires_volume(make_ctx):
    """收盘站上颈线但突破日没放量 → 构筑中，等放量确认。

    量能是硬性要求："突破颈线阶段必须放量"是排除弱势反弹的核心判据。
    """
    ctx = make_ctx(synth.strict_break_bottom_forming_weak_volume())
    hit = assert_hit(ctx, break_bottom, STATUS_FORMING)
    assert "待量" in hit.detail or "C点" in hit.detail
    assert hit.key_levels["突破量比"] < ctx.cfg.break_bottom.neck_breakout_vol_ratio


def test_volume_gate_bites_at_configured_ratio(make_ctx, cfg):
    """把放量门槛调高到突破量比之上，同一个样本就必须退回构筑中。"""
    cfg.break_bottom["neck_breakout_vol_ratio"] = 3.0
    ctx = make_ctx(synth.strict_break_bottom())
    assert_hit(ctx, break_bottom, STATUS_FORMING)


def test_breakout_today_is_forming(make_ctx):
    """放量突破就发生在今天 —— 按构筑中处理，等次日站稳再确认。"""
    bars = synth._strict_bb_bars()
    df = synth._strict_bb_df(bars._replace(closes=bars.closes[: bars.breakout_idx + 1]))
    ctx = make_ctx(df)
    hit = assert_hit(ctx, break_bottom, STATUS_FORMING)
    assert "今日突破待站稳" in hit.detail or "保守买点" in hit.detail


def test_too_deep_break_is_rejected(make_ctx):
    """破底深度超上限 = 真破位，不是空头陷阱。这是最关键的一条过滤。"""
    ctx = make_ctx(synth.bad_strict_break_too_deep())
    assert_no_hit(ctx, break_bottom)
    assert any("depth:1" in line for line in ctx.trace), ctx.trace


def test_too_shallow_break_is_rejected(make_ctx):
    """破底只有 1% < min_break_depth = 噪声级新低，不算"有效跌破"。

    允许夹心摆动点后，箱体里的 0.x% 新低会大量凑成候选；这条下限把它们挡掉。
    """
    ctx = make_ctx(synth.bad_strict_break_too_shallow())
    assert_no_hit(ctx, break_bottom)
    assert any("shallow:1" in line for line in ctx.trace), ctx.trace


def test_shallow_gate_bites_at_configured_ratio(make_ctx, cfg):
    """把下限抬到样本深度（4%）之上，同一个标准样本必须退回不命中。"""
    cfg.break_bottom["min_break_depth"] = 0.05
    assert_no_hit(make_ctx(synth.strict_break_bottom()), break_bottom)


def test_span_over_2months_is_rejected(make_ctx):
    """间隔 50 根 > max_span(45)："前低 + 三个月后又创个低点"不是二次探底。"""
    ctx = make_ctx(synth.bad_strict_span_over_2months())
    assert_no_hit(ctx, break_bottom)
    assert any("span:1" in line for line in ctx.trace), ctx.trace


@pytest.mark.parametrize(
    "factory, reason",
    [
        (synth.bad_strict_span_too_short, "span:1"),
        (synth.bad_strict_span_too_long, "span:1"),
        (synth.bad_strict_no_prior_drop, "drop:1"),
        (synth.bad_strict_no_recover, "norecover:1"),
    ],
)
def test_structural_gates_reject(make_ctx, factory, reason):
    """两个低点的间隔 / 前置下跌 / 快速止跌 —— 每条闸门都被真的用上。"""
    ctx = make_ctx(factory())
    assert_no_hit(ctx, break_bottom)
    assert any(reason in line for line in ctx.trace), ctx.trace


def test_intraday_only_undercut_is_accepted(make_ctx):
    """只有盘中最低破了前低、收盘从未跌破 —— 算有效破底（破底取 low）。

    旧口径要求**收盘**跌破，这条样本当时是反例；2026-09-29 按用户口径把破底
    统一到影线（与 v2 一致），于是它翻成正例。要收回复盘价口径得同时改
    `_find_break_bottom_base` 的 `early_break` 与这里。
    """
    ctx = make_ctx(synth.intraday_only_undercut_break())
    hit = assert_hit(ctx, break_bottom, STATUS_FORMED)
    assert hit.key_levels["破底最低"] < hit.key_levels["参考低点"], "破底必须创新低"


def test_higher_low_before_l0_is_rejected(make_ctx):
    """前低之前还有更低的价 —— L0 只是下跌途中的中间低点。

    这种结构右边照样收得出"新低"，但离真支撑还差一截，所谓破底根本没破到
    任何支撑；不拦掉的话，下跌半途的任意一个摆动低点都能被当成前低，
    全市场会捞出一大批伪破底翻（实测占命中数的七成以上）。
    """
    ctx = make_ctx(synth.bad_strict_higher_low_before_l0())
    assert_no_hit(ctx, break_bottom)
    assert any("fakelow:1" in line for line in ctx.trace), ctx.trace


def test_real_low_before_l0_still_hits(make_ctx):
    """正例方向：L0 就是这段下跌的最低点时必须照常命中（新闸门不能误杀）。"""
    df = synth.strict_break_bottom()
    bars = synth._strict_bb_bars()
    lows = df["low"].to_numpy(float)
    assert lows[bars.l0_idx] == pytest.approx(lows[: bars.l0_idx + 1].min())
    assert_hit(make_ctx(df), break_bottom, STATUS_FORMED)


def test_gapped_pivots_still_hit(make_ctx):
    """前低与二次低点之间夹着"更高的低点 + 更低的高点"时，仍须命中。

    这是真实走势里最常见的形状：反弹见顶后先回调出一个更高的低点、再弹一次，
    然后才砸穿前低。只枚举相邻摆动点的话，真前低与二次低点被夹心摆动隔开，
    (前低, 颈线, 新低) 根本凑不出来，只能凑出夹心的那个更高低点——而它到新低
    只隔 11 根（< min_span），于是整个形态静默漏报。东土科技 300353 即此形状。
    """
    bars = synth._strict_bb_bars_gapped()
    df = synth.strict_break_bottom_gapped_pivots()
    hit = assert_hit(make_ctx(df), break_bottom, STATUS_FORMED)

    lows, highs = df["low"].to_numpy(float), df["high"].to_numpy(float)
    levels = hit.key_levels
    mid_low_i = bars.low_idx - 11  # 夹在中间的那个"更高的低点"
    mid_high_i = bars.low_idx - 6  # 破底前最近的那个反弹高点
    assert lows[bars.low_idx] < lows[bars.l0_idx] < lows[mid_low_i], "样本没造出夹心结构"
    assert mid_high_i > bars.neck_idx, "样本没造出夹心的次级反弹高点"
    assert levels["参考低点"] == pytest.approx(lows[bars.l0_idx]), "前低必须取真前低"
    assert levels["参考低点"] < lows[mid_low_i], "更高的低点不能被当成前低"
    assert levels["破底最低"] == pytest.approx(lows[bars.low_idx])
    # 颈线取破底前最近的高点，而不是区间内那个更高的前顶——后者会把量度目标钉到
    # 与底部结构脱节的位置，全市场会冒出一堆"颈线 +35%、破底只 0.6%"的伪信号。
    assert levels["颈线"] == pytest.approx(highs[mid_high_i]), "颈线取破底前最近的高点"
    assert levels["颈线"] < highs[bars.neck_idx], "不能取区间内那个更高的前顶"


def test_neck_height_gate_bites(make_ctx, cfg):
    """颈线高度门槛：把下限抬到样本的 14.5% 之上，同一样本就不再成立。"""
    cfg.break_bottom["min_neck_height"] = 0.20
    ctx = make_ctx(synth.strict_break_bottom())
    assert_no_hit(ctx, break_bottom)
    assert any("neck:1" in line for line in ctx.trace), ctx.trace


def test_close_back_below_l0_is_invalidated(make_ctx):
    """站上颈线后又连跌两日收在前低下方 —— 直接作废，不再进池。

    这种"收复之后才翻失败"的结构现在在**结构定位阶段**就被挡掉（relapse），
    不再走到 detect 末尾的失效检查。两条路径的结果都是"没有信号"，测试只守结果。
    """
    ctx = make_ctx(synth.bad_strict_close_back_below_l0())
    assert_no_hit(ctx, break_bottom)
    assert any("relapse" in line or "形态失效" in line for line in ctx.trace), ctx.trace


def test_close_breaks_before_neck_is_rejected(make_ctx):
    """L0 与颈线之间就先收盘跌破过 L0 → 结构不成立（"破位必须在颈线之后"）。

    真实反例 002030 达安基因：L0 5.17(06-17)，06-25 就收在 5.06，而它认的颈线
    5.43 要到 07-16、"破底日"更是 07-21——中间早就收盘跌破过。这种结构的
    "浅破 2.7% / 1 根收复"全是假象：区间真低其实是 06-29 的 4.80（深 7.2%）。
    """
    ctx = make_ctx(synth.bad_strict_close_breaks_before_neck())
    assert_no_hit(ctx, break_bottom)
    assert any("earlybreak" in line for line in ctx.trace), ctx.trace


def test_relapse_after_recover_is_rejected(make_ctx):
    """收复 L0 之后又连收两根在 L0 下方 → 结构作废（"收复后不得再翻失败"）。

    真实反例 300354 东华测试：08-26 收复 L0=26.59，09-10/09-11 连收 26.55/25.80，
    到 09-16 又收 26.95 翻回来。detect 末尾的失效检查只看**最后两根**，被它绕过去；
    这条规则在结构定位阶段就把中间那次失败挡住。
    """
    df = synth.bad_strict_relapse_after_recover()
    ctx = make_ctx(df)
    assert_no_hit(ctx, break_bottom)
    assert any("relapse" in line for line in ctx.trace), ctx.trace
    # 最后两根收在 L0 上方：守住"末尾检查抓不到"这个前提，否则本用例会悄悄失去区分度
    assert df["close"].iloc[-2:].min() > 10.0


def test_no_break_no_hit(make_ctx):
    """没跌破前低的普通回调不该命中。"""
    ctx = make_ctx(synth.ideal_w_bottom())
    assert_no_hit(ctx, break_bottom)


def test_breakout_expiry(make_ctx, cfg):
    """突破发生在很久以前 → 候选在结构定位阶段就被跳过，不再产生信号。

    "已形成但已过期"的候选如果不跳过，会按可交易性排到最高、再被时效
    闸门整个丢弃——把本该命中的新鲜结构也一起埋掉。
    """
    cfg.break_bottom["breakout_within"] = 1
    ctx = make_ctx(synth.strict_break_bottom())
    assert_no_hit(ctx, break_bottom)
    assert any("stale:1" in line for line in ctx.trace), ctx.trace


def test_forming_expiry(make_ctx, cfg):
    """收复后迟迟不突破颈线 → 超期丢弃，不占观察池。"""
    cfg.break_bottom["forming_max_age"] = 1
    ctx = make_ctx(synth.strict_break_bottom_forming_reclaim())
    assert_no_hit(ctx, break_bottom)
    assert any("超期丢弃" in line for line in ctx.trace), ctx.trace


def test_signal_date_is_the_breakout_bar(make_ctx):
    """已形成的信号日就是放量站上颈线那一根。"""
    bars = synth._strict_bb_bars()
    df = synth.strict_break_bottom()
    hit = assert_hit(make_ctx(df), break_bottom, STATUS_FORMED)
    assert hit.signal_date == df.index[bars.breakout_idx]


def test_core_key_levels_kept_for_downstream(make_ctx):
    """监控 / 观察池 / 出图都在读这几个键，改名会静默降级。"""
    hit = assert_hit(make_ctx(synth.strict_break_bottom()), break_bottom, STATUS_FORMED)
    for key in ("参考低点", "破底最低", "参考止损", "上方压力", "颈线", "收复价"):
        assert key in hit.key_levels, f"缺少下游依赖的价位键 {key}"
    assert "B1触发价" not in hit.key_levels, "ATR 触发价口径已废弃"


def test_forming_discounted_via_stage_certainty(make_ctx):
    """构筑中要比已形成吃亏，但折扣只算一次（只由 stage_certainty 承担）。"""
    formed = assert_hit(make_ctx(synth.strict_break_bottom()), break_bottom)
    forming = assert_hit(
        make_ctx(synth.strict_break_bottom_forming_reclaim()), break_bottom
    )
    assert formed.stage_certainty() == 100.0
    assert forming.stage_certainty() < formed.stage_certainty()
    assert forming.stage_certainty() == pytest.approx(100 * forming.forming_discount)


def test_volume_pattern_affects_score(make_ctx):
    """破底缩量 + 颈线突破放量应比全程等量得分更高。"""
    good = assert_hit(make_ctx(synth.strict_break_bottom()), break_bottom)

    bars = synth._strict_bb_bars()
    flat = assert_hit(
        make_ctx(synth._strict_bb_df(bars, volumes=np.full(len(bars.closes), 1.0e6))),
        break_bottom,
    )

    assert good.vol_score > flat.vol_score
    assert good.score > flat.score


def test_recovery_price_and_bars_are_consistent(make_ctx, cfg):
    """收复价必须是"新低之后第 N 根"的收盘，且 N 在 recover_bars 之内。"""
    df = synth.strict_break_bottom()
    hit = assert_hit(make_ctx(df), break_bottom)
    levels = hit.key_levels
    used = levels["收复用根数"]
    assert 1 <= used <= cfg.break_bottom.recover_bars, "必须在 3~5 日内止跌"
    second_low_idx = int(np.argmin(df["low"].to_numpy()))
    assert levels["收复价"] == pytest.approx(df["close"].iloc[second_low_idx + used])
    assert levels["收复价"] > levels["参考低点"]


def test_disabled_detector_returns_nothing(make_ctx, cfg):
    cfg.break_bottom["enabled"] = False
    assert break_bottom.detect(make_ctx(synth.strict_break_bottom())) == []
