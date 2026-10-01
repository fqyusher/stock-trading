"""存储后端：csv / json / sqlite 三种格式行为对齐。"""

from __future__ import annotations

import pandas as pd
import pytest
import synth

from tradeyh import store


@pytest.fixture(params=["sqlite", "csv", "json"])
def backend(request, cfg, tmp_path):
    """按格式打开一个临时后端，用法与模块级 sqlite 函数一致。"""
    cfg.data["storage"] = request.param
    cfg.data["db_path"] = str(tmp_path / "kline.db")
    cfg.data["data_dir"] = str(tmp_path / "store")
    st = store.open_store(cfg)
    yield request.param, st
    st.close()


def _kline_frame(n=10, scale=1.0):
    df = synth.make_ohlcv([10.0 + i * 0.1 for i in range(n)])
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    for col in ("open", "close", "high", "low"):
        df[col] = df[col] * scale
    return df[store.KLINE_COLUMNS]


def test_upsert_is_idempotent(backend):
    _, st = backend
    df = _kline_frame(10)
    st.upsert_kline("000001", df)
    st.upsert_kline("000001", df)
    st.commit()

    assert st.bar_counts()["000001"] == 10
    assert st.last_date("000001") == df["date"].iloc[-1]


def test_incremental_append_merges_by_date(backend):
    """重叠日期覆盖、新日期追加，幂等合并。"""
    _, st = backend
    first = _kline_frame(10)
    st.upsert_kline("000001", first)

    tail = _kline_frame(15)  # 前10根重叠，后5根为新增
    st.upsert_kline("000001", tail)

    assert st.bar_counts()["000001"] == 15
    full = st.read_kline("000001")
    assert len(full) == 15
    assert full.index.is_monotonic_increasing


def test_read_kline_ordering_and_limit(backend):
    _, st = backend
    st.upsert_kline("000001", _kline_frame(30))

    full = st.read_kline("000001")
    assert isinstance(full.index, pd.DatetimeIndex)
    assert full.index.is_monotonic_increasing
    assert len(full) == 30
    # 文件后端往返后数值不应失真
    assert abs(full["close"].iloc[-1] - 12.9) < 1e-9

    tail = st.read_kline("000001", limit_bars=5)
    assert len(tail) == 5
    assert tail.index[-1] == full.index[-1], "limit_bars 应取最近N根"

    assert st.read_kline("999999").empty


def test_detect_adjust_drift(backend):
    _, st = backend
    original = _kline_frame(10)
    st.upsert_kline("000001", original)

    assert not st.detect_adjust_drift("000001", original, tolerance=0.002)

    adjusted = _kline_frame(10, scale=0.93)
    assert st.detect_adjust_drift("000001", adjusted, tolerance=0.002)

    fresh = _kline_frame(10)
    fresh["date"] = "2099-01-01"
    assert not st.detect_adjust_drift("000001", fresh, tolerance=0.002)


def test_delete_and_all_codes(backend):
    _, st = backend
    st.upsert_kline("000001", _kline_frame(5))
    st.upsert_kline("600000", _kline_frame(5))
    st.delete_code("000001")
    assert st.all_codes() == ["600000"]


def test_stock_info_and_meta(backend):
    _, st = backend
    info = pd.DataFrame(
        {"code": ["000001", "600000"], "name": ["平安银行", "浦发银行"],
         "float_mv": [1e11, 2e11], "amount": [1e9, 2e9]}
    )
    st.upsert_stock_info(info)
    assert st.get_stock_names()["000001"] == "平安银行"

    st.upsert_stock_info(pd.DataFrame({"code": ["000002"], "name": ["万科A"]}))
    assert st.get_stock_names()["000002"] == "万科A"

    st.set_meta("last_inc_update", "2026-08-28")
    assert st.get_meta("last_inc_update") == "2026-08-28"
    assert st.get_meta("missing") is None


def test_date_range(backend):
    _, st = backend
    df = _kline_frame(10)
    st.upsert_kline("000001", df)
    lo, hi = st.date_range()
    assert lo == df["date"].iloc[0] and hi == df["date"].iloc[-1]


def test_open_store_rejects_unknown_format(cfg, tmp_path):
    cfg.data["storage"] = "xml"
    cfg.data["data_dir"] = str(tmp_path / "store")
    with pytest.raises(ValueError, match="未知存储格式"):
        store.open_store(cfg)


def test_pipeline_on_csv_backend(cfg, tmp_path):
    """默认 csv 后端也能完整支撑扫描（与 sqlite 种子库结果一致）。"""
    from tradeyh import scan

    cfg.data["storage"] = "csv"
    cfg.data["data_dir"] = str(tmp_path / "store")
    cfg.data["out_dir"] = str(tmp_path / "out")
    cfg.universe["min_bars"] = 10

    with store.open_store(cfg) as st:
        st.upsert_kline("000001", synth.ideal_w_bottom().reset_index()
                        .rename(columns={"index": "date"})
                        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
                        [store.KLINE_COLUMNS])
        st.upsert_stock_info(pd.DataFrame({"code": ["000001"], "name": ["W底股"]}))

    hits = scan.scan_market(cfg, workers=1)
    assert not hits.empty
    assert "w_bottom" in set(hits.loc[hits["代码"] == "000001", "形态key"])
