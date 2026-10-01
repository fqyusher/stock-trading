"""M头检测。"""

from __future__ import annotations

import synth
from conftest import assert_hit, assert_no_hit

from tradeyh.patterns import m_top
from tradeyh.patterns.base import STATUS_FORMED, STATUS_FORMING


def test_ideal_m_top_is_formed(make_ctx):
    ctx = make_ctx(synth.ideal_m_top())
    hit = assert_hit(ctx, m_top, STATUS_FORMED)

    assert hit.pattern == "m_top"
    assert hit.score > 60, f"理想M头评分应偏高，实际 {hit.score:.1f} ({hit.detail})"
    assert 17.0 < hit.key_levels["颈线"] < 17.8
    assert 19.5 < hit.key_levels["左头"] < 20.5
    assert 19.4 < hit.key_levels["右头"] < 20.4
    assert hit.key_levels["等幅目标"] < hit.key_levels["颈线"]
    assert len(hit.pivots) == 3


def test_no_prior_rise_rejected(make_ctx):
    ctx = make_ctx(synth.bad_m_top_no_prior_rise())
    assert_no_hit(ctx, m_top)
    assert any("前期涨幅" in line for line in ctx.trace), ctx.trace


def test_forming_state_before_breakdown(make_ctx):
    """砍掉破位段，只留右头后的小幅回落 —— 应判为构筑中。"""
    df = synth.ideal_m_top().iloc[:-5]
    ctx = make_ctx(df)
    hit = assert_hit(ctx, m_top, STATUS_FORMING)
    assert "距颈线" in hit.detail


def test_shrinking_right_head_scores_higher(make_ctx):
    """右头缩量 = 追高动能衰减，是看跌确认，应比等量版本得分更高。"""
    hit_shrink = assert_hit(make_ctx(synth.ideal_m_top()), m_top)

    flat_df = synth.ideal_m_top()
    flat_df["volume"] = 1.2e6
    hit_flat = assert_hit(make_ctx(flat_df), m_top)

    assert hit_shrink.vol_score > hit_flat.vol_score


def test_w_bottom_shape_does_not_trigger_m_top(make_ctx):
    assert_no_hit(make_ctx(synth.ideal_w_bottom()), m_top)


def test_disabled_detector_returns_nothing(make_ctx, cfg):
    cfg.m_top["enabled"] = False
    assert m_top.detect(make_ctx(synth.ideal_m_top())) == []
