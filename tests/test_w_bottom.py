"""W底检测。"""

from __future__ import annotations

import synth
from conftest import assert_hit, assert_no_hit

from tradeyh.patterns import w_bottom
from tradeyh.patterns.base import STATUS_FORMED, STATUS_FORMING


def test_ideal_w_bottom_is_formed(make_ctx):
    ctx = make_ctx(synth.ideal_w_bottom())
    hit = assert_hit(ctx, w_bottom, STATUS_FORMED)

    assert hit.pattern == "w_bottom"
    assert hit.score > 60, f"理想W底评分应偏高，实际 {hit.score:.1f} ({hit.detail})"
    assert 14.5 < hit.key_levels["颈线"] < 15.5
    assert 12.5 < hit.key_levels["左底"] < 13.5
    assert 12.5 < hit.key_levels["右底"] < 13.6
    assert hit.key_levels["等幅目标"] > hit.key_levels["颈线"]
    assert len(hit.pivots) == 3


def test_shrinking_right_bottom_scores_higher(make_ctx):
    """右底缩量是W底的核心量能特征，同一形状缩量版应拿更高量能分。"""
    shrink = make_ctx(synth.ideal_w_bottom())
    hit_shrink = assert_hit(shrink, w_bottom)

    flat_df = synth.ideal_w_bottom()
    flat_df["volume"] = 1.2e6  # 右底不缩量、突破不放量
    hit_flat = assert_hit(make_ctx(flat_df), w_bottom)

    assert hit_shrink.vol_score > hit_flat.vol_score
    assert hit_shrink.score > hit_flat.score


def test_uneven_bottoms_rejected(make_ctx):
    ctx = make_ctx(synth.bad_w_bottom_uneven_bottoms())
    assert_no_hit(ctx, w_bottom)
    assert any("双底价差" in line for line in ctx.trace), ctx.trace


def test_forming_state_before_breakout(make_ctx):
    """砍掉突破段，只留右底后的小幅回升 —— 应判为构筑中。"""
    df = synth.ideal_w_bottom().iloc[:-6]
    ctx = make_ctx(df)
    hit = assert_hit(ctx, w_bottom, STATUS_FORMING)
    assert hit.score > 0
    assert "距颈线" in hit.detail


def test_stale_breakout_rejected(make_ctx, cfg):
    """把 breakout_within 收紧到 1，两根前的突破就该被判为形态已走完。"""
    cfg.w_bottom["breakout_within"] = 1
    ctx = make_ctx(synth.ideal_w_bottom())
    assert_no_hit(ctx, w_bottom)
    assert any("形态已走完" in line for line in ctx.trace), ctx.trace


def test_failed_breakout_is_no_longer_formed(make_ctx):
    """突破后收回颈线下方时，旧突破不能继续污染当前状态。"""
    df = synth.ideal_w_bottom()
    df.iloc[-1, df.columns.get_loc("close")] = 14.9
    df.iloc[-1, df.columns.get_loc("low")] = 14.85
    df.iloc[-1, df.columns.get_loc("high")] = 15.05

    hit = assert_hit(make_ctx(df), w_bottom, STATUS_FORMING)
    assert hit.signal_date == df.index[-1]


def test_disabled_detector_returns_nothing(make_ctx, cfg):
    cfg.w_bottom["enabled"] = False
    assert w_bottom.detect(make_ctx(synth.ideal_w_bottom())) == []


def test_m_top_shape_does_not_trigger_w_bottom(make_ctx):
    assert_no_hit(make_ctx(synth.ideal_m_top()), w_bottom)
