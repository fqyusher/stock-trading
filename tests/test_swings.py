"""ZigZag 摆动点与趋势背景。"""

from __future__ import annotations

import synth

from tradeyh import indicators, swings


def test_zigzag_alternates_and_finds_extremes(cfg):
    df = indicators.compute_indicators(synth.triangle_wave(), cfg)
    pivots = swings.zigzag(df, cfg)

    assert len(pivots) >= 5, f"三角波应产出多个摆动点，实际 {len(pivots)}"
    kinds = [p.kind for p in pivots]
    assert all(a != b for a, b in zip(kinds, kinds[1:])), f"峰谷未严格交替: {kinds}"
    assert [p.idx for p in pivots] == sorted(p.idx for p in pivots)

    # 峰的价格必须高于相邻谷
    for prev, cur in zip(pivots, pivots[1:]):
        if prev.is_high:
            assert prev.price > cur.price
        else:
            assert prev.price < cur.price


def test_zigzag_tail_pivot_is_unconfirmed(cfg):
    df = indicators.compute_indicators(synth.ideal_w_bottom(), cfg)
    pivots = swings.zigzag(df, cfg)
    assert pivots[-1].confirmed is False, "末端极值应标记为未确认"
    assert all(p.confirmed for p in pivots[:-1])


def test_flat_series_yields_no_pivots(cfg):
    df = indicators.compute_indicators(synth.make_ohlcv([10.0] * 80), cfg)
    assert swings.zigzag(df, cfg) == []


def test_threshold_adapts_to_volatility(cfg):
    """高波动序列的反转阈值应高于 pct_min 下限。"""
    calm = indicators.compute_indicators(synth.make_ohlcv([10.0] * 60), cfg)
    wild = indicators.compute_indicators(
        synth.make_ohlcv([10.0, 13.0] * 30), cfg
    )
    assert swings.reversal_thresholds(calm, cfg)[-1] == cfg.swings.pct_min
    assert swings.reversal_thresholds(wild, cfg)[-1] > cfg.swings.pct_min


def test_prior_trend_helpers(cfg):
    df = indicators.compute_indicators(synth.ideal_w_bottom(), cfg)
    drop = swings.prior_drop_pct(df, 69, cfg.w_bottom.prior_lookback)
    assert drop > 0.20, f"前70根从20跌到13，跌幅应显著，实际 {drop:.3f}"

    df_up = indicators.compute_indicators(synth.ideal_m_top(), cfg)
    rise = swings.prior_rise_pct(df_up, 69, cfg.m_top.prior_lookback)
    assert rise > 0.20, f"前70根从13涨到20，涨幅应显著，实际 {rise:.3f}"


def test_count_level_touches_merges_adjacent(cfg):
    # 在 10.0 附近连续贴了3根，应算作一次触碰
    closes = [12.0, 10.0, 10.01, 9.99, 12.0, 12.0, 10.0, 12.0]
    df = synth.make_ohlcv(closes)
    touches = swings.count_level_touches(df, 10.0, 0, len(df), tol=0.01, use="low")
    assert touches == 2, f"应合并相邻触碰得到2次，实际 {touches}"
