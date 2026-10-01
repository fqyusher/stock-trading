"""通道检测：上升 / 下降 / 横盘箱体。"""

from __future__ import annotations

import synth
from conftest import assert_hit, assert_no_hit

from tradeyh.patterns import channel
from tradeyh.patterns.base import STATUS_FORMED


def test_rising_channel(make_ctx):
    ctx = make_ctx(synth.ideal_rising_channel())
    hit = assert_hit(ctx, channel, STATUS_FORMED)

    assert hit.pattern == channel.RISING
    assert hit.pattern_cn == "上升通道"
    levels = hit.key_levels
    assert levels["上轨"] > levels["中轴"] > levels["下轨"]
    assert levels["上轨触碰"] >= 2 and levels["下轨触碰"] >= 2
    assert levels["容纳率"] >= ctx.cfg.channel.min_containment
    assert levels["R2"] > 0.8, f"稳定上行的R²应很高，实际 {levels['R2']:.2f}"
    assert 0.0 <= levels["通道位置"] <= 1.0
    assert levels["通道破位"] == "-"


def test_falling_channel(make_ctx):
    hit = assert_hit(make_ctx(synth.ideal_falling_channel()), channel, STATUS_FORMED)
    assert hit.pattern == channel.FALLING
    assert hit.pattern_cn == "下降通道"


def test_box_range(make_ctx):
    """横盘箱体的 R² 天然接近0，不能被 min_r2 挡掉。"""
    ctx = make_ctx(synth.ideal_box_range())
    hit = assert_hit(ctx, channel, STATUS_FORMED)
    assert hit.pattern == channel.BOX
    assert hit.key_levels["R2"] < ctx.cfg.channel.min_r2


def test_straight_line_is_not_a_channel(make_ctx):
    """笔直上行没有振荡空间，通道宽度不足，不该算震荡通道。"""
    ctx = make_ctx(synth.bad_channel_too_narrow())
    assert_no_hit(ctx, channel)
    assert any("通道宽度" in line for line in ctx.trace), ctx.trace


def test_touch_count_gate(make_ctx, cfg):
    """把触碰次数门槛提到不可能达到的水平，通道就不该被确认。"""
    cfg.channel["min_touches"] = 50
    ctx = make_ctx(synth.ideal_rising_channel())
    assert_no_hit(ctx, channel)
    assert any("不足以确认通道" in line for line in ctx.trace), ctx.trace


def test_r2_gate_only_applies_to_trending(make_ctx, cfg):
    """把 min_r2 提到 0.99：趋势通道被挡，箱体不受影响。"""
    cfg.channel["min_r2"] = 0.99
    assert_no_hit(make_ctx(synth.ideal_rising_channel()), channel)
    assert_hit(make_ctx(synth.ideal_box_range()), channel)


def test_position_near_lower_rail_scores_higher(make_ctx):
    """上升通道贴近下轨是机会位，趋势分应高于贴近上轨时。"""
    df = synth.ideal_rising_channel(bars=130, period=20)
    # 正弦周期20：往前挪5根让最后一根落在波谷附近
    at_trough = assert_hit(make_ctx(df.iloc[:-5]), channel)
    at_peak = assert_hit(make_ctx(df.iloc[:-15]), channel)
    lower, upper = sorted([at_trough.key_levels["通道位置"], at_peak.key_levels["通道位置"]])
    assert upper - lower > 0.3, "两个采样点应分别落在通道上下两侧"


def test_broken_channel_is_flagged(make_ctx):
    """向上突破上轨后仍然报出，但标记通道破位，不静默丢弃。"""
    df = synth.ideal_rising_channel()
    df.iloc[-1, df.columns.get_loc("close")] *= 1.15
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] * 1.002
    hit = assert_hit(make_ctx(df), channel)
    assert hit.key_levels["通道破位"] == "已破上轨"
    assert hit.key_levels["通道位置"] > 1.0


def test_disabled_detector_returns_nothing(make_ctx, cfg):
    cfg.channel["enabled"] = False
    assert channel.detect(make_ctx(synth.ideal_rising_channel())) == []
