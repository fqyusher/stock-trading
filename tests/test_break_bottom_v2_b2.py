"""破底翻 v2 第二买点（B2）：B1 确认 → H1 → 回踩 pl（不破 L0、缩量）→ 放量破 H1 再启动。

B2 与 B1 **独立计算**（都复用 `break_bottom_v2._find_v2_bases` 定位结构），
所以这里既验证 B2 自己的闸门，也验证它能与 B1 同时命中、不互相冒充。
"""

from __future__ import annotations

import pytest
import synth
from conftest import assert_hit, assert_no_hit

from tradeyh.patterns import break_bottom_v2, break_bottom_v2_b2
from tradeyh.patterns.base import STATUS_FORMED, STATUS_FORMING


@pytest.fixture(autouse=True)
def _enable_v2(cfg):
    """v2 在生产配置里是暂停状态（enabled: false），但它的检测器契约仍要被守住。"""
    cfg.break_bottom_v2.enabled = True
    cfg.break_bottom_v2_b2.enabled = True


def test_ideal_b2_is_formed(make_ctx):
    """标准正例：回踩缩量不破 L0，放量突破 H1 → B2 已形成。"""
    hit = assert_hit(make_ctx(synth.ideal_v2_b2()), break_bottom_v2_b2, STATUS_FORMED)

    assert hit.pattern == "break_bottom_v2_b2"
    assert hit.pattern_stage == "B2_CONFIRMED"
    assert not hit.pattern_invalid

    levels = hit.key_levels
    for key in (
        "L0", "L1", "B1", "H1", "PL", "B2", "stop_price",
        "tp1_price", "tp2_price", "risk_pct", "rr1", "rr2", "trailing_stop",
    ):
        assert key in levels, f"缺少关键价位 {key}"

    assert levels["L1"] < levels["L0"] < levels["PL"], "回踩低点必须站在 L0 上方"
    assert levels["PL"] < hit.key_levels["H1"], "回踩必须低于第一波高点"
    assert levels["B2"] > levels["H1"], "再启动买点必须收在 H1 之上"
    assert levels["stop_price"] < levels["B2"] < levels["tp1_price"]
    # TP1 = H1 + (H1 − L1) 等幅量度
    assert levels["tp1_price"] == pytest.approx(levels["H1"] + (levels["H1"] - levels["L1"]))
    assert hit.key_level_dates["L1"] < hit.key_level_dates["B1"] < hit.key_level_dates["H1"]
    assert hit.key_level_dates["H1"] < hit.key_level_dates["PL"] < hit.key_level_dates["B2"]


def test_pullback_not_yet_broken_out_is_forming(make_ctx):
    """回踩已完成、还没突破 H1 → 构筑中。"""
    hit = assert_hit(make_ctx(synth.v2_b2_forming()), break_bottom_v2_b2, STATUS_FORMING)
    assert hit.pattern_stage == "PULLBACK"
    assert hit.key_levels["B2"] == 0.0
    assert hit.key_levels["tp1_price"] > hit.close_price, "构筑中也该给出目标位"


@pytest.mark.parametrize(
    "builder",
    [
        synth.bad_v2_b2_pullback_breaks_l0,  # 回踩有效跌破 L0
        synth.bad_v2_b2_no_shrink,           # 回踩放量（不缩量）
        synth.bad_v2_b2_chase_too_late,      # 突破后已涨超 8%，不追
    ],
)
def test_b2_gate_rejects(make_ctx, builder):
    assert_no_hit(make_ctx(builder()), break_bottom_v2_b2)


def test_b2_needs_b1_confirmed(make_ctx, cfg):
    """把"B1 之后至少 N 根"抬到样本长度之上，B2 必须消失（前提是 B1 已确认）。"""
    assert_hit(make_ctx(synth.ideal_v2_b2()), break_bottom_v2_b2, STATUS_FORMED)
    cfg.break_bottom_v2_b2.min_bars_after_b1 = 1000
    assert_no_hit(make_ctx(synth.ideal_v2_b2()), break_bottom_v2_b2)


def test_b2_without_pullback_is_not_matched(make_ctx):
    """B1 正例（第一波之后直接平走、没有回踩再启动）不该被 B2 冒充。"""
    assert_no_hit(make_ctx(synth.ideal_v2_break_bottom()), break_bottom_v2_b2)


def test_b1_and_b2_are_independent(make_ctx):
    """同一份样本上两套可以同时命中，各自给各自的价位。"""
    ctx = make_ctx(synth.ideal_v2_b2())
    b1 = assert_hit(ctx, break_bottom_v2, STATUS_FORMED)
    b2 = assert_hit(ctx, break_bottom_v2_b2, STATUS_FORMED)
    assert b1.pattern != b2.pattern
    assert b1.key_levels["B1"] == b2.key_levels["B1"], "两套必须复用同一份结构定位"
    assert b2.key_levels["tp1_price"] != b1.key_levels["tp1_price"], "B2 的目标是等幅量度"


def test_b2_disabled_returns_nothing(make_ctx, cfg):
    cfg.break_bottom_v2_b2.enabled = False
    assert_no_hit(make_ctx(synth.ideal_v2_b2()), break_bottom_v2_b2)
