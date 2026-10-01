"""评分口径：止损已经被击穿时，这笔交易没有可度量的风险。

守的是一个排序失真的 bug：`risk_reward_ratio` 用 `abs()` 取"现价到止损"的距离，
于是"做多但止损在现价上方"（结构已破）被当成一个极小的正风险，RR 被兜到
MAX_RR=6.0、收益质量拿满分，反而排到榜单前列。
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradeyh.patterns.base import DIR_LONG, DIR_SHORT, PatternHit


def _hit(direction: str, close: float, stop: float, target: float) -> PatternHit:
    return PatternHit(
        code="000001",
        name="测试",
        pattern="w_bottom",
        pattern_cn="W底",
        status="formed",
        score=60.0,
        geom_score=60.0,
        vol_score=60.0,
        trend_score=60.0,
        signal_date=pd.Timestamp("2026-01-05"),
        key_levels={"参考止损": stop, "等幅目标": target},
        close_price=close,
        direction=direction,
    )


def test_long_stop_above_price_is_not_a_valid_risk():
    hit = _hit(DIR_LONG, close=10.0, stop=10.5, target=13.0)
    assert hit.stop_breached()
    assert hit.risk_reward_ratio() == 0.0
    assert hit.return_quality() == 0.0


def test_short_stop_below_price_is_not_a_valid_risk():
    hit = _hit(DIR_SHORT, close=10.0, stop=9.5, target=8.0)
    assert hit.stop_breached()
    assert hit.risk_reward_ratio() == 0.0
    assert hit.return_quality() == 0.0


def test_normal_long_risk_is_unchanged():
    """做多止损在现价下方是正常结构，评分口径不受影响。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    assert not hit.stop_breached()
    # reward 1.5 / risk 0.5 = 3.0
    assert hit.risk_reward_ratio() == pytest.approx(3.0)
    assert hit.return_quality() > 0


def test_normal_short_risk_is_unchanged():
    """做空止损在现价上方才是正常结构（不能跟着做多一起判成击穿）。"""
    hit = _hit(DIR_SHORT, close=10.0, stop=10.5, target=8.0)
    assert not hit.stop_breached()
    # reward 2.0 / risk 0.5 = 4.0
    assert hit.risk_reward_ratio() == pytest.approx(4.0)
    assert hit.return_quality() > 0


def test_stop_breached_needs_both_price_and_stop():
    hit = _hit(DIR_LONG, close=0.0, stop=10.5, target=13.0)
    assert not hit.stop_breached()
    hit = _hit(DIR_LONG, close=10.0, stop=0.0, target=13.0)
    assert not hit.stop_breached()


def test_regime_multiplier_neutral_is_one():
    """中性市场不影响评分。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    hit.market_regime = "neutral"
    assert hit.regime_multiplier() == 1.0


def test_regime_multiplier_trending_boosts_trend_patterns():
    """趋势市提升趋势类形态评分。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    hit.pattern = "trend_pullback"
    hit.market_regime = "trending"
    assert hit.regime_multiplier() > 1.0


def test_regime_multiplier_trending_penalizes_reversal_patterns():
    """趋势市压制反转类形态评分。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    hit.pattern = "w_bottom"
    hit.market_regime = "trending"
    assert hit.regime_multiplier() < 1.0


def test_regime_multiplier_ranging_boosts_reversal_patterns():
    """震荡市提升反转类形态评分。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    hit.pattern = "w_bottom"
    hit.market_regime = "ranging"
    assert hit.regime_multiplier() > 1.0


def test_time_decay_reduces_stage_certainty():
    """时间衰减降低已形成形态的阶段确定性。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    hit.status = "formed"
    hit.time_decay = 0.5
    assert hit.stage_certainty() == pytest.approx(50.0)


def test_time_decay_does_not_affect_forming():
    """构筑中形态不受时间衰减影响（用 forming_discount 折扣）。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    hit.status = "forming"
    hit.forming_discount = 0.85
    hit.time_decay = 0.5
    assert hit.stage_certainty() == pytest.approx(85.0)


def test_time_decay_floor():
    """衰减有下限，不会归零。"""
    hit = _hit(DIR_LONG, close=10.0, stop=9.5, target=11.5)
    hit.status = "formed"
    hit.time_decay = 0.1
    assert hit.stage_certainty() == pytest.approx(10.0)
