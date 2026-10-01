"""渠道选择逻辑测试（离线，不发网络请求）。"""

from __future__ import annotations

import pytest

from tradeyh import fetch


def test_sina_channel_skipped_when_adjusted(cfg):
    """qfq 下新浪不适用，应立即切渠道而不是反复重试。"""
    cfg.data["sources"] = ["sina"]
    cfg.data["adjust"] = "qfq"
    with pytest.raises(fetch.FetchError, match="全部渠道失败"):
        fetch.fetch_kline("000001", "20240101", "20241231", cfg)


def test_sina_channel_rejects_non_daily(cfg):
    cfg.data["adjust"] = ""
    cfg.data["period"] = "weekly"
    with pytest.raises(fetch.ChannelUnavailable):
        fetch._kline_sina("000001", "20240101", "20241231", cfg)


def test_unknown_source_rejected(cfg):
    cfg.data["sources"] = ["nonsense"]
    with pytest.raises(ValueError, match="未知数据源"):
        fetch.fetch_kline("000001", "20240101", "20241231", cfg)


def test_spot_skips_channels_without_snapshot(cfg):
    """新浪没有全市场列表接口，fetch_spot 应跳过它而不是报"未知数据源"。"""
    cfg.data["sources"] = ["sina"]
    with pytest.raises(fetch.FetchError, match="全部渠道失败"):
        fetch.fetch_spot(cfg)


def test_spot_rejects_unknown_source(cfg):
    cfg.data["sources"] = ["nonsense"]
    with pytest.raises(ValueError, match="未知数据源"):
        fetch.fetch_spot(cfg)
