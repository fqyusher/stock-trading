"""破底翻第二买点（B2）检测测试。

B2 严格定义 11 点：
1. B1 确认：Close > N + 0.2×ATR
2. H1-B1 >= max(5%, 1.5×ATR)
3. 第一次回踩
4. 颈线支撑区：N-0.5×ATR <= L1 <= N+0.3×ATR
5. 回踩幅度分级：20-50%最佳 / 50-70%可接受 / >70%降级
6. 缩量 + 不允许持续放量破位
7. B2 确认：Close > 前日 High AND Close > MA5
8. 强确认：Vol >= MA5_Vol AND MACD改善 AND RSI上行
9. 失效：Close < N-0.3×ATR 连续2日 / 跌破 L0 / 回踩>70%后走弱
10. 止损：回踩低点/颈线下方 ATR 缓冲
11. 目标：TP1=H1，TP2=更高前高或测量目标
"""

from __future__ import annotations

import numpy as np
import pytest
import synth
from tradeyh.patterns import break_bottom, break_bottom_b2
from tradeyh.patterns.base import STATUS_FORMED, STATUS_FORMING


def test_ideal_b2_formed(make_ctx):
    """理想 B2：B1 确认 → 第一波 → 缩量回踩颈线支撑区 → B2 确认。"""
    ctx = make_ctx(synth.ideal_break_bottom_b2())
    hits = break_bottom_b2.detect(ctx)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.status == STATUS_FORMED
    assert hit.pattern == "break_bottom_b2"
    for key in ("颈线", "B1价格", "第一波高点", "回踩低点", "第一目标", "第二目标", "参考止损"):
        assert key in hit.key_levels


def test_b2_key_levels(make_ctx):
    """验证 key_levels 的合理性与风险结构。"""
    hit = break_bottom_b2.detect(make_ctx(synth.ideal_break_bottom_b2()))[0]
    levels = hit.key_levels
    assert levels["颈线"] > 0
    assert levels["B1价格"] > levels["颈线"]
    assert levels["第一波高点"] > levels["B1价格"]
    assert levels["参考止损"] < levels["回踩低点"]
    assert levels["第一目标"] == levels["第一波高点"]
    assert levels["第二目标"] >= levels["第一目标"]


def test_b2_h1_and_l1_are_measured_from_breakout(make_ctx):
    """H1/L1 必须落在颈线突破之后的构造段上。

    以前 H1 从"收复点"起算，会塌到颈线附近，B2 与 B1 同日触发、退化成重复信号。
    """
    df, plan = synth._strict_b2_frame()
    hit = break_bottom_b2.detect(make_ctx(df))[0]
    assert hit.key_levels["第一波高点"] == pytest.approx(df["high"].iloc[plan["h1"]]), (
        "H1 应为第一波构造段的最高点"
    )
    assert hit.key_levels["回踩低点"] == pytest.approx(df["low"].iloc[plan["l1"]]), (
        "L1 应为回踩构造段的最低点"
    )


def test_b2_b1_premise_uses_first_neckline_breakout(make_ctx, cfg):
    """B2 的起算点是**首次**放量突破颈线（`first_breakout_idx`），不是当前这段行情的。

    本样本回踩到 L1（< 颈线）后又放量突破一次，于是"当前行情段"的触发点落在
    B2 自己的突破前高那根上；拿它当第一波起点，H1 会直接塌到当前那根，
    B2 与"突破前高"这一步变成同一个信号。
    """
    df, _ = synth._strict_b2_frame()
    ctx = make_ctx(df)
    base = break_bottom._find_break_bottom_base(
        ctx, cfg.break_bottom, cfg_b2=cfg.break_bottom_b2
    )
    assert base is not None
    assert base.first_breakout_idx is not None
    assert base.breakout_idx is not None
    assert base.first_breakout_idx < base.breakout_idx, (
        "首次颈线突破应当早于 B2 自己的突破前高"
    )


def test_b2_forming_before_confirmation(make_ctx):
    """回踩完成、结构完好、尚未 B2 确认 → 构筑中（观察池的主要对象）。"""
    df, plan = synth._strict_b2_frame()
    ctx = make_ctx(df.iloc[: plan["l1"] + 2])
    hits = break_bottom_b2.detect(ctx)
    assert len(hits) == 1
    assert hits[0].status == STATUS_FORMING
    assert "待确认" in hits[0].detail


def test_b2_requires_neckline_breakout(make_ctx):
    """颈线只是站上去、没有放量突破 → B1 未形成，B2 也不该出现。"""
    ctx = make_ctx(synth.strict_break_bottom_forming_weak_volume())
    assert break_bottom_b2.detect(ctx) == []
    assert any("B1 未形成" in line for line in ctx.trace), ctx.trace


def test_b2_min_bars_after_breakout(make_ctx, cfg):
    """颈线突破后必须留出足够根数，才可能形成第一波 + 回踩。"""
    cfg.break_bottom_b2["min_bars_after_breakout"] = 100
    ctx = make_ctx(synth.ideal_break_bottom_b2())
    assert break_bottom_b2.detect(ctx) == []
    assert any("不足100根" in line for line in ctx.trace), ctx.trace


def test_b2_formed_expires(make_ctx, cfg):
    """突破前高太久 → 形态已走完，不再当成可交易的"已形成"信号。

    以前只挡"构筑中"超期；突破后只要 3 个交易日内没跌回 H1 下方，就永远算已形成
    （实测 300508 维宏股份 06-15 站上 H1，三个月后回落到 -25% 还在报表里）。
    """
    cfg.break_bottom_b2["breakout_within"] = 1
    ctx = make_ctx(synth.ideal_break_bottom_b2())
    assert break_bottom_b2.detect(ctx) == []
    assert any("形态已走完" in line for line in ctx.trace), ctx.trace


def test_b2_disabled(make_ctx, cfg):
    cfg.break_bottom_b2.enabled = False
    assert break_bottom_b2.detect(make_ctx(synth.ideal_break_bottom_b2())) == []


def test_b2_pullback_breaks_l0(make_ctx):
    """回踩跌破 L0 应该被拒绝。"""
    ctx = make_ctx(synth.bad_b2_pullback_breaks_l0())
    assert break_bottom_b2.detect(ctx) == []
    # 收复后连续跌破 L0 现在在结构定位阶段就挡掉了（relapse），不再走到 B2 自己的闸门
    assert any("relapse" in line or "结构破坏" in line for line in ctx.trace), ctx.trace


def test_b2_no_volume_shrink(make_ctx):
    """回踩不缩量（在派发，不是洗盘）应该被拒绝。"""
    ctx = make_ctx(synth.bad_b2_no_volume_shrink())
    assert break_bottom_b2.detect(ctx) == []
    assert any("回踩量" in line for line in ctx.trace), ctx.trace


def test_b2_chase_too_late(make_ctx):
    """突破前高后涨幅超 8% 不追。"""
    ctx = make_ctx(synth.bad_b2_chase_too_late())
    assert break_bottom_b2.detect(ctx) == []
    assert any("不追" in line for line in ctx.trace), ctx.trace


def test_b2_breaks_l0_after_l1(make_ctx):
    """L1 之后又跌破 L0 —— 结构闸门必须查到当日，不能只查到 L1。"""
    ctx = make_ctx(synth.bad_b2_breaks_l0_after_l1())
    assert break_bottom_b2.detect(ctx) == []
    # 新规范下，连续跌破颈线支撑区下沿也会触发失效
    assert any(
        "relapse" in line or "颈线失效" in line or "结构破坏" in line
        for line in ctx.trace
    ), ctx.trace


def test_b2_pullback_volume_excludes_h1_bar(make_ctx):
    """回踩均量只算 (H1, L1]，H1 当根已经计进第一波。

    H1 是放量长阳；两边都算会把回踩均量抬高，把本来合格的回踩判成"没缩量"。
    """
    ctx = make_ctx(synth.b2_pullback_volume_boundary())
    hits = break_bottom_b2.detect(ctx)
    assert len(hits) == 1, f"回踩段口径正确时应当命中：{ctx.trace}"
    assert any("缩量OK" in line for line in ctx.trace), ctx.trace


def test_b2_volume_characteristics(make_ctx):
    """回踩缩量 + 突破放量 → 量能评分偏高。"""
    hit = break_bottom_b2.detect(make_ctx(synth.ideal_break_bottom_b2()))[0]
    assert hit.vol_score > 50


def _old_break_df():
    """破底发生在约 77 根之前：进得了 100 根的窗口，却会被 B1 的 20 根窗口漏掉。"""
    bars = synth._strict_bb_bars(tail=[])
    closes = np.concatenate([bars.closes, np.full(68, 11.0)])
    return synth._strict_bb_df(bars._replace(closes=closes))


def test_b2_break_window_honours_break_within_b2(make_ctx, cfg):
    """破底新低的回看窗口要真的按 break_within_b2 生效。"""
    ctx = make_ctx(_old_break_df())
    base = break_bottom_b2._find_break_bottom_base(
        ctx, cfg.break_bottom, cfg_b2=cfg.break_bottom_b2
    )
    assert base is not None, "距离末尾 77 根的破底应落在 100 根窗口内"

    cfg.break_bottom_b2["break_within_b2"] = 20
    narrow = break_bottom_b2._find_break_bottom_base(
        ctx, cfg.break_bottom, cfg_b2=cfg.break_bottom_b2
    )
    assert narrow is None


def test_b2_requires_break_bottom_base(make_ctx):
    """没有破底翻基础结构应该返回空。"""
    ctx = make_ctx(synth.make_ohlcv(np.linspace(10, 15, 200)))
    assert break_bottom_b2.detect(ctx) == []


def test_b2_base_failure_records_reason(make_ctx):
    """基础结构定位失败时必须把**具体**原因写进 trace。"""
    ctx = make_ctx(synth.make_ohlcv(np.linspace(10, 15, 200)))
    assert break_bottom_b2.detect(ctx) == []
    assert ctx.trace, "explain 模式下应留下拒绝原因"
    assert any("摆动点三元组" in line for line in ctx.trace), ctx.trace
