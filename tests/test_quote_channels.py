"""报价渠道：同花顺解析 + fetch_quotes 的补漏语义（全部离线）。

`fetch_quotes` 从"首个非空即返回"改成了"缺的票用后续渠道补齐"，因为各渠道覆盖都不满：
停牌、新股，以及同花顺那种网关 502 抖动（实测 10~20%）。这里把两条语义都钉住。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradeyh import fetch


# ---------------------------------------------------------------------------
# 同花顺解析
# ---------------------------------------------------------------------------


def _ths_payload(code="000001", name="平安银行", price=11.57, prev=11.35,
                 open_=11.36, high=11.65, low=11.33, vol_shares=104535745,
                 amount=1205814860.0, pct=1.94):
    return {"items": {
        "5": code, "name": name, "10": str(price), "6": str(prev), "7": str(open_),
        "8": str(high), "9": str(low), "13": str(vol_shares), "19": str(amount),
        "199112": str(pct),
    }}


def _ths_text(payload) -> str:
    return f"quotebridge_v6_realhead_hs_000001_last({json.dumps(payload, ensure_ascii=False)});"


def test_ths_symbol():
    assert fetch._ths_symbol("000001") == "hs_000001"
    assert fetch._ths_symbol(600000) == "hs_600000"


def test_jsonp_payload_strips_wrapper():
    assert fetch._jsonp_payload('cb({"a":1});') == {"a": 1}
    assert fetch._jsonp_payload('cb({"a":1})') == {"a": 1}
    assert fetch._jsonp_payload("not jsonp at all") is None
    assert fetch._jsonp_payload("cb({bad json});") is None


def test_parse_ths_quote_fields_and_units():
    """字段 ID 是实测对出来的；volume 同花顺给"股"，本地库/QUOTE_COLUMNS 是"手"。"""
    row = fetch._parse_ths_quote(_ths_payload())
    assert row["code"] == "000001"
    assert row["name"] == "平安银行"
    assert row["price"] == pytest.approx(11.57)
    assert row["prev_close"] == pytest.approx(11.35)
    assert row["open"] == pytest.approx(11.36)
    assert row["high"] == pytest.approx(11.65)
    assert row["low"] == pytest.approx(11.33)
    assert row["pct_chg"] == pytest.approx(1.94)
    assert row["amount"] == pytest.approx(1205814860.0)
    assert row["volume"] == pytest.approx(1045357.45), "股 → 手要除 100"


@pytest.mark.parametrize("price", ["", None, "0", "-1"])
def test_parse_ths_quote_skips_no_price(price):
    """停牌/退市没有有效现价 → 跳过，不能当成 0 元报价。"""
    payload = _ths_payload()
    payload["items"]["10"] = price
    assert fetch._parse_ths_quote(payload) is None


def test_quotes_ths_tolerates_single_code_failure(monkeypatch, cfg):
    """同花顺约 10~20% 的票会被网关 502 打掉，整批不能因此失败。"""
    calls: list[str] = []

    def fake_get_text(url, *a, **kw):
        code = url.rsplit("hs_", 1)[1].split("/", 1)[0]
        calls.append(code)
        if code == "600000":
            raise fetch.FetchError("502 Bad Gateway")
        return _ths_text(_ths_payload(code=code, name=f"股票{code}"))

    monkeypatch.setattr(fetch, "_http_get_text", fake_get_text)
    df = fetch._quotes_ths(["000001", "600000", "300750"], cfg)

    assert sorted(df["code"]) == ["000001", "300750"]
    assert len(calls) == 3, "失败的票不该拖垮其余请求"
    assert list(df.columns) == fetch.QUOTE_COLUMNS


def test_quotes_ths_raises_when_all_fail(monkeypatch, cfg):
    monkeypatch.setattr(
        fetch, "_http_get_text",
        lambda *a, **kw: (_ for _ in ()).throw(fetch.FetchError("502")),
    )
    with pytest.raises(fetch.FetchError, match="全部失败"):
        fetch._quotes_ths(["000001", "600000"], cfg)


def test_quotes_ths_uses_requested_code_not_response(monkeypatch, cfg):
    """返回体里的代码不可信，一律以请求的代码为准。"""
    monkeypatch.setattr(
        fetch, "_http_get_text",
        lambda *a, **kw: _ths_text(_ths_payload(code="999999")),
    )
    df = fetch._quotes_ths(["000001"], cfg)
    assert df["code"].tolist() == ["000001"]


# ---------------------------------------------------------------------------
# fetch_quotes 补漏语义
# ---------------------------------------------------------------------------


def _register(monkeypatch, mapping: dict[str, list[str]], cfg):
    """把 QUOTE_CHANNELS 换成假渠道，记录每个渠道实际收到的代码。"""
    asked: dict[str, list[str]] = {}

    def make(name, codes_it_can_serve):
        def fn(codes, cfg_):
            asked.setdefault(name, []).extend(codes)
            served = [c for c in codes if c in codes_it_can_serve]
            if not served:
                raise fetch.FetchError(f"{name} 拿不到任何一只")
            return pd.DataFrame([
                {"code": c, "name": f"{name}-{c}", "price": 10.0, "open": 10.0,
                 "high": 10.5, "low": 9.5, "volume": 100.0, "amount": 1e6,
                 "prev_close": 9.9, "pct_chg": 1.0}
                for c in served
            ])

        return fn

    channels = {name: (name, make(name, set(codes))) for name, codes in mapping.items()}
    monkeypatch.setattr(fetch, "QUOTE_CHANNELS", channels)
    return asked


def test_fetch_quotes_fills_gaps_from_later_channels(monkeypatch, cfg, http_stub):
    """靠前渠道缺的票只用缺的那些去问下一个渠道，凑齐即止。"""
    asked = _register(monkeypatch, {"a": ["000001", "600000"], "b": ["300750"]}, cfg)
    df = fetch.fetch_quotes(["000001", "600000", "300750"], cfg, sources=["a", "b"])

    assert set(df["code"]) == {"000001", "600000", "300750"}
    assert set(asked["a"]) == {"000001", "600000", "300750"}, "第一个渠道拿到全部请求的票"
    assert asked["b"] == ["300750"], "补齐时只该问缺的那只"


def test_fetch_quotes_stops_once_complete(monkeypatch, cfg, http_stub):
    asked = _register(monkeypatch, {"a": ["000001", "600000"], "b": ["000001"]}, cfg)
    fetch.fetch_quotes(["000001", "600000"], cfg, sources=["a", "b"])
    assert "b" not in asked, "已经凑齐就不该再问下一个渠道"


def test_fetch_quotes_keeps_requested_order(monkeypatch, cfg, http_stub):
    _register(monkeypatch, {"a": ["600000"], "b": ["000001", "300750"]}, cfg)
    df = fetch.fetch_quotes(["300750", "000001", "600000"], cfg, sources=["a", "b"])
    assert df["code"].tolist() == ["300750", "000001", "600000"]


def test_fetch_quotes_tolerates_channel_returning_nothing(monkeypatch, cfg, http_stub):
    """某渠道整批失败（如东财被墙）应继续往后走，不该中断。"""
    asked = _register(monkeypatch, {"a": [], "b": ["000001"]}, cfg)
    df = fetch.fetch_quotes(["000001"], cfg, sources=["a", "b"])
    assert df["code"].tolist() == ["000001"]
    assert asked["a"] == ["000001"]


def test_fetch_quotes_all_channels_fail_raises(monkeypatch, cfg, http_stub):
    _register(monkeypatch, {"a": [], "b": []}, cfg)
    with pytest.raises(fetch.FetchError, match="全部渠道失败"):
        fetch.fetch_quotes(["000001"], cfg, sources=["a", "b"])


def test_fetch_quotes_skips_channel_without_support(monkeypatch, cfg, http_stub):
    """sina 之类不提供报价的渠道应被跳过，而不是报错。"""
    _register(monkeypatch, {"a": ["000001"]}, cfg)
    df = fetch.fetch_quotes(["000001"], cfg, sources=["sina", "a"])
    assert df["code"].tolist() == ["000001"]


def test_fetch_quotes_empty_codes(cfg):
    df = fetch.fetch_quotes([], cfg)
    assert df.empty and list(df.columns) == fetch.QUOTE_COLUMNS


def test_quote_channel_order_matches_config(cfg):
    """同花顺放最后（一股一请求 + 10~20% 网关抖动，只当兵底）。

    字典顺序必须和 `watch.quote_sources` 一致：`trade.py` 调 fetch_quotes 时没传
    sources，走的就是字典顺序。配合上面"凑齐即止"那条，同花顺在日常轮次里
    完全不产生开销。
    """
    assert list(cfg.watch.quote_sources) == ["eastmoney", "tencent", "ths"]
    assert list(fetch.QUOTE_CHANNELS) == ["eastmoney", "tencent", "ths"]
