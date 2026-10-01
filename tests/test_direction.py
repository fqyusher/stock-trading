"""方向与评分表的完整性。

base.py 把目标价、方向、优先级都做成了表驱动，理由是"漏登记会在测试里暴露"
——这个文件就是那个测试。少了它，新加一个形态忘了登记方向，只会表现为
做空形态悄悄混进做多榜单，没有任何报错。
"""

from __future__ import annotations

import pandas as pd
import pytest

import synth
from tradeyh import report, watch
from tradeyh.patterns import PATTERN_CN, break_bottom, channel, m_top, w_bottom
from tradeyh.patterns.base import (
    DIR_LONG,
    DIR_SHORT,
    MAX_RR,
    PATTERN_DIRECTION,
    STATUS_FORMED,
    STATUS_FORMING,
    STOP_KEY,
    TARGET_KEYS,
    PatternHit,
    blend,
)


def _hit(pattern="w_bottom", close=10.0, levels=None, status=STATUS_FORMED, **kw):
    return PatternHit(
        code="000001", name="测试", pattern=pattern, pattern_cn="测试形态",
        status=status, score=80.0, geom_score=80.0, vol_score=80.0, trend_score=80.0,
        signal_date=pd.Timestamp("2024-01-02"),
        key_levels=levels or {}, close_price=close, **kw,
    )


# ---------------- 表完整性 ----------------

def test_every_pattern_has_direction():
    """每个能出现在报表里的形态都必须登记方向。"""
    missing = sorted(set(PATTERN_CN) - set(PATTERN_DIRECTION))
    assert not missing, f"以下形态没登记交易方向，会默认当成做多: {missing}"


def test_every_pattern_has_target_key():
    """每个形态都必须有目标价键，否则收益质量恒为 0，永远排不进榜。"""
    missing = sorted(set(PATTERN_CN) - set(TARGET_KEYS))
    assert not missing, f"以下形态没登记目标价键，收益质量会恒为 0: {missing}"


def test_direction_values_are_valid():
    assert set(PATTERN_DIRECTION.values()) <= {DIR_LONG, DIR_SHORT}


def test_bearish_patterns_are_short():
    """M头和下降通道是有效图形，但不是做多机会。"""
    assert PATTERN_DIRECTION["m_top"] == DIR_SHORT
    assert PATTERN_DIRECTION["falling_channel"] == DIR_SHORT
    # 下降通道里的止跌反转是做多，别跟上面那条搞混
    assert PATTERN_DIRECTION["falling_channel_reversal"] == DIR_LONG


def test_direction_filled_from_table():
    """检测器不传 direction 时按形态查表，而不是一律 long。"""
    assert _hit(pattern="m_top").direction == DIR_SHORT
    assert _hit(pattern="w_bottom").direction == DIR_LONG
    assert _hit(pattern="m_top").is_long is False
    assert _hit(pattern="w_bottom").is_long is True


def test_unknown_pattern_falls_back_to_long():
    assert _hit(pattern="不存在的形态").direction == DIR_LONG


# ---------------- 收益/风险计算 ----------------

def test_short_profit_positive_when_target_below_close():
    """做空形态兑现 = 价格下跌，预期收益应为正数。"""
    hit = _hit(pattern="m_top", close=10.0, levels={"等幅目标": 9.0, STOP_KEY: 10.3})
    assert hit.expected_profit() == pytest.approx(10.0)


def test_long_profit_positive_when_target_above_close():
    hit = _hit(pattern="w_bottom", close=10.0, levels={"等幅目标": 11.0, STOP_KEY: 9.7})
    assert hit.expected_profit() == pytest.approx(10.0)


def test_rr_clamped_when_price_sits_on_stop():
    """现价贴着止损时真实 risk→0，RR 不兜住会冲到几十上百污染排序。"""
    hit = _hit(pattern="w_bottom", close=10.0, levels={"等幅目标": 13.0, STOP_KEY: 9.9999})
    assert hit.risk_reward_ratio() <= MAX_RR


def test_missing_stop_gives_zero_rr_not_crash():
    hit = _hit(pattern="w_bottom", close=10.0, levels={"等幅目标": 11.0})
    assert hit.risk_reward_ratio() == 0.0


# ---------------- 构筑中只打一次折 ----------------

def test_blend_ignores_status(cfg):
    """blend 不再乘 forming_discount，折扣只由 stage_certainty 承担。

    以前两处都打折，构筑中被扣两遍分。
    """
    formed = blend(0.8, 0.8, 0.8, STATUS_FORMED, cfg, priority="A")
    forming = blend(0.8, 0.8, 0.8, STATUS_FORMING, cfg, priority="A")
    assert formed == forming


def test_stage_certainty_uses_forming_discount():
    hit = _hit(status=STATUS_FORMING)
    hit.forming_discount = 0.5
    assert hit.stage_certainty() == pytest.approx(50.0)
    assert _hit(status=STATUS_FORMED).stage_certainty() == 100.0


def test_priority_lowers_score(cfg):
    """优先级低的形态同样几何质量下应拿更低分。"""
    s_score = blend(0.8, 0.8, 0.8, STATUS_FORMED, cfg, priority="S")[0]
    c_score = blend(0.8, 0.8, 0.8, STATUS_FORMED, cfg, priority="C")[0]
    assert c_score < s_score


# ---------------- 检测器真的把价位填进去了 ----------------

@pytest.mark.parametrize(
    "fixture_name,module",
    [
        ("ideal_w_bottom", w_bottom),
        ("ideal_m_top", m_top),
        ("strict_break_bottom", break_bottom),
        ("ideal_rising_channel", channel),
        ("ideal_falling_channel", channel),
        ("ideal_box_range", channel),
    ],
)
def test_detector_hits_carry_target_and_stop(fixture_name, module, make_ctx):
    """命中必须同时带目标价和止损键，否则风险收益比恒为 0。

    这里只查"键在不在"和"RR 有没有被裁剪"：目标价离现价多远取决于合成
    数据落在形态的哪个位置（比如价格正贴着通道上轨时上行空间本就为 0），
    不该在这个测试里约束。
    """
    ctx = make_ctx(getattr(synth, fixture_name)())
    hits = module.detect(ctx)
    assert hits, f"{fixture_name} 未命中:\n  " + "\n  ".join(ctx.trace)
    for hit in hits:
        target_key = TARGET_KEYS[hit.pattern]
        assert target_key in hit.key_levels, f"{hit.pattern} 缺目标价键 {target_key}"
        assert STOP_KEY in hit.key_levels, f"{hit.pattern} 缺 {STOP_KEY}"
        assert hit._get_target_price() is not None
        assert hit._get_stop_loss() is not None
        assert 0 <= hit.risk_reward_ratio() <= MAX_RR


@pytest.mark.parametrize(
    "fixture_name,module",
    [
        ("ideal_w_bottom", w_bottom),
        ("ideal_m_top", m_top),
        ("strict_break_bottom", break_bottom),
    ],
)
def test_reversal_hits_have_room_to_target(fixture_name, module, make_ctx):
    """反转形态的合成样本都留了到目标价的空间，收益质量必须为正。

    做空形态（M头）按跌幅算，同样应该是正数——这是 direction 分支生效的证据。
    """
    ctx = make_ctx(getattr(synth, fixture_name)())
    hits = module.detect(ctx)
    assert hits
    for hit in hits:
        assert hit.expected_profit() > 0, f"{hit.pattern} 预期收益 {hit.expected_profit()}"
        assert hit.risk_reward_ratio() > 0, f"{hit.pattern} RR 为 0，止损键没生效"
        assert hit.return_quality() > 0


def test_scan_context_stamps_forming_discount(make_ctx, cfg):
    """阶段折扣由 scan_context 统一盖上，改 config 要对所有形态生效。"""
    from tradeyh import scan

    cfg.scoring["forming_discount"] = 0.42
    ctx = make_ctx(synth.ideal_w_bottom())
    hits = scan.scan_context(ctx, {"w_bottom": w_bottom})
    assert hits
    assert all(h.forming_discount == pytest.approx(0.42) for h in hits)


# ---------------- 做多筛选 ----------------

def _frame():
    return pd.DataFrame([
        {"形态key": "w_bottom", "综合评分": 90},
        {"形态key": "m_top", "综合评分": 95},
        {"形态key": "falling_channel", "综合评分": 93},
        {"形态key": "rising_channel", "综合评分": 80},
    ])


def test_report_long_only_drops_shorts():
    kept = set(report.long_only(_frame())["形态key"])
    assert kept == {"w_bottom", "rising_channel"}


def test_watch_long_only_drops_shorts():
    kept = set(watch._long_only(_frame())["形态key"])
    assert kept == {"w_bottom", "rising_channel"}


def test_long_only_survives_missing_column():
    df = pd.DataFrame([{"综合评分": 90}])
    assert len(report.long_only(df)) == 1


def test_long_only_drops_invalid_rows():
    """v2 的失效行仍是做多形态，只按形态key 筛方向会把它们放进榜单/观察池。"""
    df = pd.DataFrame([
        {"形态key": "break_bottom_v2", "失效": "", "综合评分": 80},
        {"形态key": "break_bottom_v2", "失效": "是", "综合评分": 91},
        {"形态key": "break_bottom_v2_b2", "失效": "是", "综合评分": 88},
    ])
    kept = report.long_only(df)
    assert list(kept["综合评分"]) == [80]
    assert set(watch._long_only(df).index) == set(kept.index)


def test_long_only_without_invalid_column_is_unchanged():
    """老报表没有"失效"列（旧口径不输出这一档），筛选行为不能变。"""
    assert len(report.long_only(_frame())) == 2
    assert len(watch._long_only(_frame())) == 2
