"""联网 smoke 测试：验证东财/腾讯抓取渠道的接口契约没有变。

默认不跑（pytest.ini 里 -m "not network"）。手动跑：
    pytest tests/test_network.py -m network -v
"""

from __future__ import annotations

import pytest

from tradeyh import fetch, store, universe

pytestmark = pytest.mark.network


def test_fetch_kline_contract(cfg):
    df, channel = fetch.fetch_kline("000001", "20240101", "20241231", cfg)
    assert channel is not None
    assert not df.empty
    assert list(df.columns) == store.KLINE_COLUMNS
    assert df["date"].is_monotonic_increasing
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()


def test_fetch_kline_tencent_channel(cfg):
    """单独走腾讯渠道也要能出标准表（缺额/换手列允许为 NaN）。"""
    cfg.data["sources"] = ["tencent"]
    df, channel = fetch.fetch_kline("000001", "20240101", "20241231", cfg)
    assert channel == "腾讯"
    assert not df.empty
    assert list(df.columns) == store.KLINE_COLUMNS
    assert df["date"].is_monotonic_increasing
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()
    assert df[["open", "close", "high", "low", "volume"]].notna().all().all()


def test_fetch_kline_unknown_code_returns_empty(cfg):
    df, channel = fetch.fetch_kline("000000", "20240101", "20241231", cfg)
    assert df.empty
    assert channel is None, "无数据时不应报告成功渠道"


def test_fetch_kline_sina_channel(cfg):
    """单独走新浪渠道（仅不复权）：单请求拿全区间，量单位已换算成手。"""
    cfg.data["sources"] = ["sina"]
    cfg.data["adjust"] = ""
    df, channel = fetch.fetch_kline("000001", "20240101", "20241231", cfg)
    assert channel == "新浪"
    assert not df.empty
    assert list(df.columns) == store.KLINE_COLUMNS
    assert df["date"].is_monotonic_increasing
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()
    assert df["date"].min() >= "2024-01-01" and df["date"].max() <= "2024-12-31"


def test_fetch_kline_channel_order_does_not_matter(cfg):
    cfg.data["sources"] = ["tencent", "eastmoney"]
    df, _ = fetch.fetch_kline("600000", "20240101", "20240131", cfg)
    assert not df.empty and list(df.columns) == store.KLINE_COLUMNS


def test_fetch_spot_covers_whole_market(cfg):
    spot = fetch.fetch_spot(cfg)
    assert len(spot) > 4500, f"沪深京A股应有5000+只，实际 {len(spot)}"
    assert set(spot.columns) >= {"code", "name", "price", "amount", "float_mv"}
    assert spot["code"].str.len().eq(6).all()

    pool = universe.filter_spot(spot, cfg)
    assert 0 < len(pool) < len(spot)
    assert not pool["name"].str.contains("ST").any()


def test_fetch_spot_tencent_fallback(cfg):
    """腾讯备用快照：代码清单仍来自东财，报价来自腾讯。"""
    cfg.data["sources"] = ["tencent"]
    spot = fetch.fetch_spot(cfg)
    assert len(spot) > 4500
    assert set(spot.columns) >= {"code", "name", "price", "amount", "float_mv"}
    assert (spot.loc[spot["price"] > 0, "price"] > 0).all()
