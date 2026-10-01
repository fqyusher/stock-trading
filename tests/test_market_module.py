"""市场环境 / 情绪 / 行业热度模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradeyh.market.regime import compute_market_regime
from tradeyh.market.sentiment import compute_market_sentiment
from tradeyh.market.sector import compute_sector_heat
from tradeyh.market.snapshot import MarketSnapshot


# ---------------------------------------------------------------------------
# MarketRegime
# ---------------------------------------------------------------------------


def _make_index_df(last_close: float, trend: str = "up", bars: int = 80) -> pd.DataFrame:
    """合成指数 K 线。trend: up / down / flat。"""
    np.random.seed(42)
    if trend == "up":
        closes = np.linspace(last_close * 0.85, last_close, bars)
    elif trend == "down":
        closes = np.linspace(last_close * 1.15, last_close, bars)
    else:
        closes = np.full(bars, last_close)
    closes += np.random.randn(bars) * last_close * 0.005
    highs = closes * (1 + np.abs(np.random.randn(bars) * 0.005))
    lows = closes * (1 - np.abs(np.random.randn(bars) * 0.005))
    opens = closes + np.random.randn(bars) * last_close * 0.003
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.random.randint(100000, 999999, bars).astype(float),
        "amount": np.random.randint(1000000, 9999999, bars).astype(float),
        "pct_chg": np.zeros(bars),
    }, index=pd.date_range("2026-01-01", periods=bars, freq="B"))


class TestMarketRegime:
    def test_strong_market(self):
        """三指数全部上涨趋势 → 强势。"""
        dfs = {
            "csi300": _make_index_df(4000, "up"),
            "csi1000": _make_index_df(6000, "up"),
            "chinext": _make_index_df(2500, "up"),
        }
        score, label, detail = compute_market_regime(
            dfs, {"csi300": 0.4, "csi1000": 0.3, "chinext": 0.3},
            {"strong": 80, "bullish": 65, "neutral": 45, "weak": 25},
        )
        assert score >= 65
        assert label in ("强势", "偏强")

    def test_weak_market(self):
        """三指数全部下跌趋势 → 弱势。"""
        dfs = {
            "csi300": _make_index_df(3000, "down"),
            "csi1000": _make_index_df(5000, "down"),
            "chinext": _make_index_df(1800, "down"),
        }
        score, label, detail = compute_market_regime(
            dfs, {"csi300": 0.4, "csi1000": 0.3, "chinext": 0.3},
            {"strong": 80, "bullish": 65, "neutral": 45, "weak": 25},
        )
        assert score <= 45
        assert label in ("弱势", "偏弱")

    def test_empty_indices(self):
        """无指数数据 → 默认震荡。"""
        score, label, _ = compute_market_regime(
            {}, {"csi300": 0.4}, {"strong": 80, "bullish": 65, "neutral": 45, "weak": 25},
        )
        assert score == 50.0
        assert label == "震荡"


# ---------------------------------------------------------------------------
# MarketSentiment
# ---------------------------------------------------------------------------


def _make_summary(n: int, median_pct: float = 0.0) -> pd.DataFrame:
    """合成全市场日度数据。"""
    np.random.seed(42)
    pct = np.random.normal(median_pct, 2.0, n)
    codes = [f"{i:06d}" for i in range(n)]
    closes = 10 + np.random.randn(n)
    highs = closes * (1 + np.abs(pct) / 100)
    return pd.DataFrame({
        "code": codes, "pct_chg": pct, "close": closes, "high": highs,
        "amount": np.random.randint(1000000, 9999999, n).astype(float),
    })


class TestMarketSentiment:
    weights = {
        "advance_ratio": 0.20, "median_return": 0.20,
        "limit_count": 0.15, "failed_limit_rate": 0.15,
        "limit_follow_through": 0.15, "max_streak": 0.15,
    }

    def test_bullish_sentiment(self):
        """大部分股票上涨 → 情绪偏高。"""
        df = _make_summary(5000, median_pct=1.5)
        score, detail = compute_market_sentiment(df, self.weights)
        assert score >= 60

    def test_bearish_sentiment(self):
        """大部分股票下跌 → 情绪偏低。"""
        df = _make_summary(5000, median_pct=-1.5)
        score, detail = compute_market_sentiment(df, self.weights)
        assert score <= 40

    def test_empty_summary(self):
        """空数据 → 默认 50。"""
        df = pd.DataFrame(columns=["code", "pct_chg", "close", "high", "amount"])
        score, _ = compute_market_sentiment(df, self.weights)
        assert 40 <= score <= 60


# ---------------------------------------------------------------------------
# SectorHeat
# ---------------------------------------------------------------------------


def _make_sector_data():
    """合成行业数据。"""
    np.random.seed(42)
    n_stocks = 100
    codes = [f"{i:06d}" for i in range(n_stocks)]
    sectors = {c: "电子" if i < 50 else "银行" for i, c in enumerate(codes)}
    pct = np.random.normal(0.5, 2.0, n_stocks)
    summary = pd.DataFrame({
        "code": codes, "pct_chg": pct,
        "close": 10 + np.random.randn(n_stocks),
        "high": 10.5 + np.abs(np.random.randn(n_stocks)),
        "amount": np.random.randint(1000000, 9999999, n_stocks).astype(float),
    })
    return summary, sectors


class TestSectorHeat:
    weights = {
        "relative_return": 0.20, "volume_ratio": 0.15,
        "breadth": 0.15, "trend": 0.20,
        "capital_intensity": 0.15, "outperform_days": 0.15,
    }

    def test_basic_computation(self):
        summary, sectors = _make_sector_data()
        result = compute_sector_heat(summary, None, sectors, None, self.weights)
        assert "电子" in result
        assert "银行" in result
        for name, data in result.items():
            assert 0 <= data["score"] <= 100
            assert data["stage"] in ("启动", "扩散", "主升", "高潮", "分歧", "退潮")

    def test_empty_data(self):
        result = compute_sector_heat(
            pd.DataFrame(), None, {}, None, self.weights,
        )
        assert result == {}


# ---------------------------------------------------------------------------
# MarketSnapshot
# ---------------------------------------------------------------------------


class TestMarketSnapshot:
    def test_factors_default(self):
        """中性 50 分 → 环境乘子必须恰好是 1.00（恒等变换）。

        这是承重墙：综合评分要靠它留在 0-100 量纲上，否则 --min-score /
        watch.min_score / Excel 色阶锚点全部失准（曾经三乘子相乘把 60 分压到 2 分）。
        """
        snap = MarketSnapshot(
            date="2026-09-29",
            regime_score=50,
            sentiment_score=50,
            sector_scores={"电子": 50},
            sector_stages={"电子": "扩散"},
            code_to_sector={"000001": "电子"},
        )
        assert snap.market_factor == 1.0
        assert snap.sentiment_factor == 1.0
        assert snap.sector_factor("000001") == 1.0

    def test_sector_bonus(self):
        snap = MarketSnapshot(
            date="2026-09-29",
            sector_stages={"电子": "主升", "银行": "退潮", "地产": "高潮"},
            code_to_sector={"000001": "电子", "600000": "银行", "001979": "地产"},
        )
        assert snap.sector_bonus("000001") == 1.05  # 主升
        assert snap.sector_bonus("600000") == 1.0    # 退潮 → no bonus
        assert snap.sector_bonus("001979") == 0.95   # 高潮 → penalty

    def test_block_long(self):
        snap = MarketSnapshot(date="2026-09-29", regime_score=20)
        assert snap.should_block_long() is True
        snap2 = MarketSnapshot(date="2026-09-29", regime_score=60)
        assert snap2.should_block_long() is False

    def test_unknown_sector_neutral(self):
        """未知板块 → 按 50 分算，即中性 1.00。"""
        snap = MarketSnapshot(
            date="2026-09-29",
            sector_scores={"电子": 80},
            code_to_sector={"000001": "电子"},
        )
        assert snap.sector_factor("999999") == 1.0


# ---------------------------------------------------------------------------
# Score integration
# ---------------------------------------------------------------------------


class TestScoreIntegration:
    def test_no_snapshot_preserves_score(self):
        """market_snapshot=None 时 composite_score 与原始逻辑一致。"""
        from tradeyh.patterns.base import PatternHit

        hit = PatternHit(
            code="000001", name="测试", pattern="break_bottom",
            pattern_cn="破底翻", status="formed", score=80,
            geom_score=80, vol_score=70, trend_score=75,
            signal_date=pd.Timestamp("2026-09-20"),
            close_price=10.0,
        )
        score_no_snap = hit.composite_score()
        assert score_no_snap > 0
        assert hit.market_snapshot is None

    def test_snapshot_reduces_score(self):
        """弱市场环境 + 弱情绪应降低综合评分（中性才是 1.00）。"""
        from tradeyh.patterns.base import PatternHit

        hit_base = PatternHit(
            code="000001", name="测试", pattern="break_bottom",
            pattern_cn="破底翻", status="formed", score=80,
            geom_score=80, vol_score=70, trend_score=75,
            signal_date=pd.Timestamp("2026-09-20"),
            close_price=10.0,
        )
        score_base = hit_base.composite_score()

        weak_snap = MarketSnapshot(
            date="2026-09-29",
            regime_score=30,
            sentiment_score=30,
            sector_scores={},
            sector_stages={},
            code_to_sector={},
        )
        hit_weak = PatternHit(
            code="000001", name="测试", pattern="break_bottom",
            pattern_cn="破底翻", status="formed", score=80,
            geom_score=80, vol_score=70, trend_score=75,
            signal_date=pd.Timestamp("2026-09-20"),
            close_price=10.0,
            market_snapshot=weak_snap,
        )
        score_weak = hit_weak.composite_score()
        assert score_weak < score_base
