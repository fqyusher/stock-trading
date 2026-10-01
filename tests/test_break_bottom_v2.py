"""破底翻 v2（ATR 严格口径）：L0 → 有效破底 → 快速收复 → 跌破失败确认 → B1 确认。

与旧 `test_break_bottom.py` 分开：v2 刻意**放宽**了"必须收盘破底"（盘中破也算），
又**收紧**了确认环节（看 ATR 阈值而不是放量站上颈线），两套口径的期望值不同，
混在一个文件里会互相污染。
"""

from __future__ import annotations

import pytest
import synth
from conftest import assert_hit, assert_no_hit

from tradeyh.patterns import break_bottom_v2
from tradeyh.patterns.base import STATUS_FORMED, STATUS_FORMING, STATUS_INVALID


@pytest.fixture(autouse=True)
def _enable_v2(cfg):
    """v2 在生产配置里是暂停状态（enabled: false），但它的检测器契约仍要被守住。"""
    cfg.break_bottom_v2.enabled = True
    cfg.break_bottom_v2_b2.enabled = True


def test_ideal_is_formed_with_b1_targets(make_ctx, cfg):
    """标准正例：B1 已确认、H1/TP1/TP2 都已形成。"""
    ctx = make_ctx(synth.ideal_v2_break_bottom())
    hit = assert_hit(ctx, break_bottom_v2, STATUS_FORMED)

    assert hit.pattern == "break_bottom_v2"
    assert hit.pattern_stage == "B1_CONFIRMED"
    assert not hit.pattern_invalid

    levels = hit.key_levels
    # 关键价位必须齐全（未知一律写 0.0，保证报表列集合稳定）
    for key in (
        "L0", "L1", "B1", "H1", "B2", "stop_price", "tp1_price", "tp2_price",
        "risk_pct", "reward1_pct", "reward2_pct", "rr1", "rr2", "trailing_stop",
    ):
        assert key in levels, f"缺少关键价位 {key}"

    assert levels["L1"] < levels["L0"], "破底必须创新低"
    assert 0 < levels["破底深度%"] <= 100 * cfg.break_bottom_v2.max_break_depth
    assert levels["B1"] > levels["L0"], "B1 是收复确认价，必在 L0 上方"
    assert levels["stop_price"] < levels["B1"], "止损必须在买点下方"
    assert levels["tp1_price"] > levels["B1"], "TP1（H1）必须在买点上方"
    assert levels["tp2_price"] > levels["tp1_price"], "TP2 是 H1 之上的最近一档"
    assert levels["rr1"] > 0 and levels["rr2"] >= levels["rr1"]
    assert hit._get_stop_loss() == pytest.approx(levels["stop_price"])
    assert hit.key_level_dates["L0"] < hit.key_level_dates["L1"] < hit.key_level_dates["B1"]


def test_intraday_only_undercut_is_accepted(make_ctx):
    """"只有盘中破、收盘从未破"算有效破底——v1/v2 现在是同一套影线口径。

    这条样本当初是 v1 的反例（v1 曾要求收盘跌破），v2 落地时特意反向守着；
    2026-09-29 v1 也统一到影线口径后，两套检测器在这份样本上结论一致。
    """
    ctx = make_ctx(synth.intraday_only_undercut_break())
    hit = assert_hit(ctx, break_bottom_v2, STATUS_FORMED)
    assert hit.key_levels["L1"] < hit.key_levels["L0"]


def test_broken_stage_is_forming(make_ctx):
    """刚破底、收复窗口还没走完 → 构筑中，阶段 BROKEN，且不给价位。"""
    ctx = make_ctx(synth.v2_broken_forming())
    hit = assert_hit(ctx, break_bottom_v2, STATUS_FORMING)
    assert hit.pattern_stage == "BROKEN"
    assert hit.key_levels["B1"] == 0.0
    assert hit.key_levels["stop_price"] == 0.0


def test_reclaimed_stage_is_forming(make_ctx):
    """已收复 L0、确认窗口还没走完 → 构筑中，阶段 RECLAIMED。"""
    hit = assert_hit(make_ctx(synth.v2_reclaimed_forming()), break_bottom_v2, STATUS_FORMING)
    assert hit.pattern_stage == "RECLAIMED"
    assert hit.key_levels["L1"] < hit.key_levels["L0"]


def test_rebreak_in_confirm_window_invalidates(make_ctx):
    """收复后确认窗口内**收盘**又有效跌破 L0 → 失效（假收复）。"""
    hit = assert_hit(make_ctx(synth.bad_v2_rebreak_in_confirm()), break_bottom_v2, STATUS_INVALID)
    assert hit.pattern_invalid
    assert hit.invalid_reason == "收复后再次有效跌破L0"
    assert hit.stage_certainty() == 0.0


def test_intrabar_dip_in_confirm_window_is_not_a_breach(make_ctx):
    """确认窗口内只有盘中插针破、收盘稳稳站住 → 不算"再次有效跌破"。

    300353 东土科技 2026-09-23 就是这个形状（盘中最低 16.35 < 阈值 16.42，
    收盘 16.89 却远在 L0=16.61 上方）。判据按收盘价，所以这里必须直接成为 B1。
    """
    hit = assert_hit(make_ctx(synth.v2_confirm_intrabar_dip()), break_bottom_v2, STATUS_FORMED)
    assert hit.pattern_stage == "B1_CONFIRMED"
    assert hit.key_levels["B1"] > hit.key_levels["L0"]


def test_no_recover_in_time_invalidates(make_ctx, cfg):
    """破底后在低位磨蹭，recover_bars 内没有收复 → 失效。"""
    hit = assert_hit(make_ctx(synth.bad_v2_no_recover_old()), break_bottom_v2, STATUS_INVALID)
    assert hit.invalid_reason == f"破底后{cfg.break_bottom_v2.recover_bars}日内未收复L0"


def test_close_back_below_stop_invalidates(make_ctx):
    """B1 确认后收盘跌穿止损 → 失效（这笔交易在交易层早已离场）。

    这条是定义外新增的闸门：定义只约束"收复后 fail_confirm_bars 内不得再跌破"，
    不拦住的话，确认之后砸穿止损的票仍会报"已形成"。
    """
    hit = assert_hit(make_ctx(synth.bad_strict_close_back_below_l0()), break_bottom_v2, STATUS_INVALID)
    assert hit.invalid_reason == "B1确认后跌破止损位"


@pytest.mark.parametrize(
    "builder",
    [
        synth.bad_strict_break_too_deep,      # 破底 14% > max_break_depth
        synth.bad_strict_break_too_shallow,   # 破底 1% < min_break_depth
        synth.bad_strict_higher_low_before_l0,  # L0 不是这段下跌的最低点
        synth.bad_strict_no_prior_drop,       # 前置下跌不足 15%
    ],
)
def test_gate_rejects(make_ctx, builder):
    """每条闸门都要有一条反例守着——否则调阈值时没人知道它还在不在起作用。"""
    assert_no_hit(make_ctx(builder()), break_bottom_v2)


@pytest.mark.parametrize(
    "builder",
    [
        synth.ideal_w_bottom,
        synth.ideal_m_top,
        synth.ideal_box_range,
        synth.ideal_rising_channel,
        synth.ideal_break_bottom_b2,
    ],
)
def test_other_patterns_not_matched(make_ctx, builder):
    """其它形态的样本不该被 v2 误伤。"""
    assert_no_hit(make_ctx(builder()), break_bottom_v2)


def test_emit_invalid_disabled_hides_invalid_rows(make_ctx, cfg):
    """关闭 emit_invalid 后，失效行不再输出（报表不被历史失败塞满）。"""
    cfg.break_bottom_v2.emit_invalid = False
    assert_no_hit(make_ctx(synth.bad_v2_rebreak_in_confirm()), break_bottom_v2)


def test_stale_invalid_row_is_suppressed(make_ctx, cfg):
    """失效发生得太早（超过 invalid_within）就不再输出。"""
    cfg.break_bottom_v2.invalid_within = 0
    assert_no_hit(make_ctx(synth.bad_v2_no_recover_old()), break_bottom_v2)


def test_one_row_per_code_at_most(make_ctx):
    """每只票最多 1 行：出图文件名是 代码_形态key.png，两行会撞文件名。"""
    hits = break_bottom_v2.detect(make_ctx(synth.ideal_v2_break_bottom()))
    assert len(hits) == 1


def test_min_break_depth_gate_is_active(make_ctx, cfg):
    """把下限抬到正例的深度之上，正例必须立刻不命中（证明闸门真的在起作用）。"""
    assert_hit(make_ctx(synth.ideal_v2_break_bottom()), break_bottom_v2, STATUS_FORMED)
    cfg.break_bottom_v2.min_break_depth = 0.5
    assert_no_hit(make_ctx(synth.ideal_v2_break_bottom()), break_bottom_v2)
