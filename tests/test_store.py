"""存储层：入库幂等、读取、复权漂移检测。"""

from __future__ import annotations

import pandas as pd
import pytest
import synth

from tradeyh import store


@pytest.fixture
def conn(tmp_path):
    connection = store.connect(tmp_path / "test.db")
    store.init_db(connection)
    yield connection
    connection.close()


def _kline_frame(n=10, scale=1.0):
    df = synth.make_ohlcv([10.0 + i * 0.1 for i in range(n)])
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    for col in ("open", "close", "high", "low"):
        df[col] = df[col] * scale
    return df[store.KLINE_COLUMNS]


def test_normalize_maps_chinese_columns():
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "股票代码": ["000001", "000001"],
            "开盘": [10.0, 10.5],
            "收盘": [10.5, 10.8],
            "最高": [10.6, 11.0],
            "最低": [9.9, 10.4],
            "成交量": [1000, 1200],
            "成交额": [10500, 12800],
            "振幅": [1.0, 1.1],
            "涨跌幅": [1.5, 2.0],
            "涨跌额": [0.15, 0.3],
            "换手率": [0.5, 0.6],
        }
    )
    out = store.normalize_kline(raw)
    assert list(out.columns) == store.KLINE_COLUMNS
    assert out["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert out["close"].tolist() == [10.5, 10.8]


def test_normalize_empty_input_is_safe():
    assert store.normalize_kline(pd.DataFrame()).empty
    assert store.normalize_kline(None).empty


def test_upsert_is_idempotent(conn):
    df = _kline_frame(10)
    store.upsert_kline(conn, "000001", df)
    store.upsert_kline(conn, "000001", df)
    conn.commit()

    assert store.bar_counts(conn)["000001"] == 10
    assert store.last_date(conn, "000001") == df["date"].iloc[-1]


def test_read_kline_ordering_and_limit(conn):
    store.upsert_kline(conn, "000001", _kline_frame(30))
    conn.commit()

    full = store.read_kline(conn, "000001")
    assert isinstance(full.index, pd.DatetimeIndex)
    assert full.index.is_monotonic_increasing
    assert len(full) == 30

    tail = store.read_kline(conn, "000001", limit_bars=5)
    assert len(tail) == 5
    assert tail.index.is_monotonic_increasing
    assert tail.index[-1] == full.index[-1], "limit_bars 应取最近N根"


def test_detect_adjust_drift(conn):
    original = _kline_frame(10)
    store.upsert_kline(conn, "000001", original)
    conn.commit()

    # 同样的数据不该报漂移
    assert not store.detect_adjust_drift(conn, "000001", original, tolerance=0.002)

    # 前复权价整体缩放 = 发生了除权除息
    adjusted = _kline_frame(10, scale=0.93)
    assert store.detect_adjust_drift(conn, "000001", adjusted, tolerance=0.002)

    # 库里没有重叠日期时无法判断，应返回 False 而不是误报
    fresh = _kline_frame(10)
    fresh["date"] = "2099-01-01"
    assert not store.detect_adjust_drift(conn, "000001", fresh, tolerance=0.002)


def test_delete_code(conn):
    store.upsert_kline(conn, "000001", _kline_frame(5))
    store.upsert_kline(conn, "600000", _kline_frame(5))
    conn.commit()
    store.delete_code(conn, "000001")
    conn.commit()
    assert store.all_codes(conn) == ["600000"]


def test_stock_info_and_meta(conn):
    info = pd.DataFrame(
        {"code": ["000001", "600000"], "name": ["平安银行", "浦发银行"],
         "float_mv": [1e11, 2e11], "amount": [1e9, 2e9]}
    )
    store.upsert_stock_info(conn, info)
    assert store.get_stock_names(conn)["000001"] == "平安银行"

    # 缺少可选列也要能写入
    store.upsert_stock_info(conn, pd.DataFrame({"code": ["000002"], "name": ["万科A"]}))
    assert store.get_stock_names(conn)["000002"] == "万科A"

    store.set_meta(conn, "last_inc_update", "2026-08-28")
    assert store.get_meta(conn, "last_inc_update") == "2026-08-28"
    assert store.get_meta(conn, "missing") is None
