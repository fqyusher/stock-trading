"""指数实时报价 + 全市场快照 pct_chg 的解析（全部离线，靠 http_stub）。

**本仓库的实网路径在开发环境不可用**，所以这里覆盖的是解析与回退逻辑：
东财 ulist 的字段映射、腾讯 GBK 行情串的下标、以及"全渠道失败抛 FetchError"，
保证调用方能据此回退到本地收盘价。
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradeyh import fetch, universe


def _em_index_payload():
    return {
        "data": {
            "diff": [
                {"f12": "000300", "f14": "沪深300", "f2": 4532.11, "f3": -0.42},
                {"f12": "000905", "f14": "中证1000", "f2": 6120.88, "f3": 0.15},
                {"f12": "399006", "f14": "创业板指", "f2": 2210.30, "f3": -0.88},
            ]
        }
    }


def _tx_line(symbol: str, name: str, code: str, price: float, pct: float, fields: int = 50):
    """腾讯行情串。字段按固定下标填，避免靠数波浪线。"""
    p = ["0"] * fields
    p[fetch._TX_F_NAME] = name
    p[2] = code
    p[fetch._TX_F_PRICE] = str(price)
    p[fetch._TX_F_PCT_CHG] = str(pct)
    return f'v_{symbol}="' + "~".join(p) + '"'


# ---------------------------------------------------------------------------
# 指数符号映射
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "idx_code,expected",
    [("idx_000300", "sh000300"), ("idx_000905", "sh000905"), ("idx_399006", "sz399006")],
)
def test_idx_tx_symbol(idx_code, expected):
    assert fetch._idx_tx_symbol(idx_code) == expected


def test_idx_tx_symbol_rejects_unknown_prefix():
    with pytest.raises(ValueError):
        fetch._idx_tx_symbol("idx_999999")


# ---------------------------------------------------------------------------
# 东财
# ---------------------------------------------------------------------------


def test_index_quotes_eastmoney_parses_and_maps_codes(http_stub, cfg):
    http_stub.add_json("ulist.np", _em_index_payload())
    codes = list(cfg.market.index_codes.values())

    df = fetch.fetch_index_quotes(codes, cfg, sources=["eastmoney"])

    assert list(df["code"]) == codes, "指数代码要映射回 idx_xxxxxx"
    assert list(df["name"]) == ["沪深300", "中证1000", "创业板指"]
    assert df["price"].tolist() == [4532.11, 6120.88, 2210.30]
    assert df["pct_chg"].tolist() == [-0.42, 0.15, -0.88]
    # secid 必须用指数口径（000xxx 属沪市，与个股规则相反）
    _, params = http_stub.json_calls[0]
    assert params["secids"] == "1.000300,1.000905,0.399006"


def test_index_quotes_skip_invalid_price(http_stub, cfg):
    payload = {"data": {"diff": [
        {"f12": "000300", "f14": "沪深300", "f2": "-", "f3": 0.0},
        {"f12": "399006", "f14": "创业板指", "f2": 2210.30, "f3": -0.88},
    ]}}
    http_stub.add_json("ulist.np", payload)
    df = fetch.fetch_index_quotes(list(cfg.market.index_codes.values()), cfg,
                                  sources=["eastmoney"])
    assert list(df["code"]) == ["idx_399006"], "无点位（停牌/休市）的要剔除"


# ---------------------------------------------------------------------------
# 腾讯回退
# ---------------------------------------------------------------------------


def test_index_quotes_tencent_parses_gbk_string(http_stub, cfg):
    text = ";".join([
        _tx_line("sh000300", "沪深300", "000300", 4532.11, -0.42),
        _tx_line("sz399006", "创业板指", "399006", 2210.30, -0.88),
    ]) + ";"
    http_stub.add_text("qt.gtimg.cn", text)

    df = fetch.fetch_index_quotes(["idx_000300", "idx_399006"], cfg, sources=["tencent"])

    assert list(df["code"]) == ["idx_000300", "idx_399006"]
    assert df["pct_chg"].tolist() == [-0.42, -0.88]
    assert df["price"].tolist() == [4532.11, 2210.30]


def test_index_quotes_falls_back_to_tencent_when_eastmoney_fails(http_stub, cfg):
    """东财不注册（抛 FetchError）→ 自动落到腾讯。"""
    http_stub.add_text("qt.gtimg.cn", _tx_line("sh000300", "沪深300", "000300", 4532.11, -0.42))
    df = fetch.fetch_index_quotes(["idx_000300"], cfg, sources=["eastmoney", "tencent"])
    assert df["code"].tolist() == ["idx_000300"]


def test_index_quotes_all_channels_fail_raises(http_stub, cfg):
    """全渠道失败必须抛 FetchError，调用方才能回退本地收盘价。"""
    with pytest.raises(fetch.FetchError):
        fetch.fetch_index_quotes(["idx_000300"], cfg, sources=["eastmoney", "tencent"])


def test_index_quotes_empty_codes_is_empty_frame(cfg):
    df = fetch.fetch_index_quotes([], cfg)
    assert df.empty
    assert list(df.columns) == fetch.INDEX_QUOTE_COLUMNS


# ---------------------------------------------------------------------------
# 全市场快照补 pct_chg（涨跌停/行业榜要用）
# ---------------------------------------------------------------------------


def test_spot_eastmoney_includes_pct_chg(http_stub, cfg):
    payload = {"data": {"total": 2, "diff": [
        {"f2": 3.71, "f3": -1.33, "f6": 1.67e9, "f8": 0.5, "f12": "000002", "f14": "万科A",
         "f21": 1.0e10},
        {"f2": 11.35, "f3": 0.44, "f6": 7.8e8, "f8": 0.7, "f12": "000001", "f14": "平安银行",
         "f21": 2.0e11},
    ]}}
    http_stub.add_json("clist/get", payload)
    cfg.data["sources"] = ["eastmoney"]

    spot = fetch.fetch_spot(cfg)

    assert "pct_chg" in spot.columns
    assert spot.set_index("code")["pct_chg"].to_dict() == {"000002": -1.33, "000001": 0.44}
    # 字段顺序沿用原样，且 f3 要真的被请求
    assert "f3" in http_stub.json_calls[0][1]["fields"]
    # 加列不能破坏股池过滤
    assert len(universe.filter_spot(spot, cfg)) == 2


def test_spot_tencent_includes_pct_chg(http_stub, cfg):
    http_stub.add_json("clist/get", {"data": {"total": 1, "diff": [{"f12": "000001"}]}})
    http_stub.add_text("qt.gtimg.cn", _tx_line("sz000001", "平安银行", "000001", 11.35, 0.44))
    cfg.data["sources"] = ["tencent"]

    spot = fetch.fetch_spot(cfg)

    assert spot["pct_chg"].tolist() == [0.44]
    assert spot["price"].tolist() == [11.35]


def _seed_codes(cfg, tmp_path, codes):
    """造一个只有代码清单的临时 csv 库（K线内容不重要）。"""
    from tradeyh import store

    kdir = tmp_path / "store" / "csv" / "kline"
    kdir.mkdir(parents=True, exist_ok=True)
    for code in codes:
        (kdir / f"{code}.csv").write_text("date,open,close,high,low\n", encoding="utf-8")
    cfg.data["storage"] = "csv"
    cfg.data["data_dir"] = str(tmp_path / "store")
    cfg.data["sources"] = ["tencent"]
    assert store.stock_codes(cfg) == sorted(codes)


def test_spot_tencent_falls_back_to_local_code_list(http_stub, cfg, tmp_path):
    """东财列表不可用时，腾讯渠道改用本地库代码清单——否则全市场快照整体失效，
    面板的涨跌停/成交额就只能显示昨天的收盘数。"""
    codes = ["000001", "600000", "300750"]
    _seed_codes(cfg, tmp_path, codes)
    # 只注册腾讯行情串：东财 clist 未注册 → 抛 FetchError → 触发本地清单兜底
    http_stub.add_text("qt.gtimg.cn", ";".join([
        _tx_line("sz000001", "平安银行", "000001", 11.60, 2.20),
        _tx_line("sh600000", "浦发银行", "600000", 10.10, 1.00),
        _tx_line("sz300750", "宁德时代", "300750", 200.0, 19.97),   # 创业板 20% 板涨停
    ]) + ";")

    spot = fetch.fetch_spot(cfg)

    assert sorted(spot["code"]) == sorted(codes)
    assert spot.set_index("code")["pct_chg"].to_dict() == {
        "000001": 2.20, "600000": 1.00, "300750": 19.97,
    }
    assert not spot["pct_chg"].isna().any(), "本地清单路径也要拿到当日涨跌幅"


def test_spot_tencent_raises_when_no_codes_anywhere(http_stub, cfg, tmp_path):
    """东财不通、本地库也空 → 明确抛错，让调用方回退本地收盘聚合。"""
    cfg.data["storage"] = "csv"
    cfg.data["data_dir"] = str(tmp_path / "store")
    cfg.data["sources"] = ["tencent"]
    http_stub.add_text("qt.gtimg.cn", "")
    with pytest.raises(fetch.FetchError):
        fetch.fetch_spot(cfg)
