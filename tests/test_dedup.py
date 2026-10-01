"""同股形态去重：互斥组内只保留评分最高的一个。"""

from __future__ import annotations

import pandas as pd

from tradeyh.patterns.base import PatternHit
from tradeyh.scan import dedup_hits, apply_invalidation


def _make_hit(
    code: str,
    pattern: str,
    score: float,
    close: float = 10.0,
    status: str = "formed",
    key_levels: dict | None = None,
) -> PatternHit:
    return PatternHit(
        code=code,
        name="测试",
        pattern=pattern,
        pattern_cn=pattern,
        status=status,
        score=score,
        geom_score=score,
        vol_score=score,
        trend_score=score,
        signal_date=pd.Timestamp("2026-01-05"),
        key_levels=key_levels or {},
        close_price=close,
    )


def test_conflict_group_keeps_highest_score():
    """破底翻新旧口径互斥，保留评分最高的。"""
    hits = [
        _make_hit("000001", "break_bottom", 50.0),
        _make_hit("000001", "break_bottom_v2", 70.0),
    ]
    result = dedup_hits(hits)
    assert len(result) == 1
    assert result[0].pattern == "break_bottom_v2"


def test_independent_patterns_both_survive():
    """非冲突形态都保留。"""
    hits = [
        _make_hit("000001", "w_bottom", 50.0),
        _make_hit("000001", "trend_pullback", 60.0),
    ]
    result = dedup_hits(hits)
    assert len(result) == 2


def test_dedup_is_per_stock():
    """不同股票的冲突形态各自保留。"""
    hits = [
        _make_hit("000001", "break_bottom", 50.0),
        _make_hit("000002", "break_bottom_v2", 70.0),
    ]
    result = dedup_hits(hits)
    assert len(result) == 2


def test_b2_conflict_group():
    """B2 新旧口径互斥。"""
    hits = [
        _make_hit("000001", "break_bottom_b2", 80.0),
        _make_hit("000001", "break_bottom_v2_b2", 60.0),
    ]
    result = dedup_hits(hits)
    assert len(result) == 1
    assert result[0].pattern == "break_bottom_b2"


def test_invalidation_marks_long_pattern():
    """做多形态 close < 关键支撑 → 失效。"""
    hits = [
        _make_hit(
            "000001", "w_bottom", 60.0,
            close=8.0,
            key_levels={"左底": 9.0, "参考止损": 8.5, "等幅目标": 12.0},
        ),
    ]
    result = apply_invalidation(hits)
    assert result[0].pattern_invalid is True
    assert result[0].status == "invalid"


def test_invalidation_marks_short_pattern():
    """做空形态 close > 关键压力 → 失效。"""
    hits = [
        _make_hit(
            "000001", "m_top", 60.0,
            close=11.0,
            key_levels={"参考止损": 10.5, "等幅目标": 8.0},
        ),
    ]
    result = apply_invalidation(hits)
    assert result[0].pattern_invalid is True
    assert result[0].status == "invalid"


def test_invalidation_skips_already_invalid():
    """v2 已标记 invalid 的不重复判定。"""
    hit = _make_hit("000001", "break_bottom_v2", 60.0, close=8.0)
    hit.pattern_invalid = True
    hit.invalid_reason = "v2 自行标记"
    result = apply_invalidation([hit])
    assert result[0].invalid_reason == "v2 自行标记"


def test_invalidation_no_rule_means_no_change():
    """没有失效规则的形态不受影响。"""
    hits = [_make_hit("000001", "rising_channel", 60.0, close=8.0)]
    result = apply_invalidation(hits)
    assert result[0].pattern_invalid is False
    assert result[0].status == "formed"
