from __future__ import annotations

from datetime import datetime

import pandas as pd

from tradeyh import monitor, store, watch
from tradeyh.patterns import PatternHit


def test_build_intraday_df_replaces_same_day_bar():
    idx = pd.date_range("2026-09-08", periods=2, freq="D")
    hist = pd.DataFrame(
        {
            "open": [10.0, 11.0], "close": [10.5, 11.5],
            "high": [11.0, 12.0], "low": [9.5, 10.5],
            "volume": [100, 120], "amount": [1000, 1200],
            "turnover": [1.0, 1.1], "pct_chg": [0.0, 9.5],
        }, index=idx
    )
    out = monitor.build_intraday_df(
        hist, {"price": 12.0, "open": 11.6, "high": 11.8, "low": 11.4}, today=idx[-1]
    )
    assert len(out) == len(hist)
    assert out.iloc[-1]["close"] == 12.0
    assert out.iloc[-1]["high"] >= out.iloc[-1]["close"]


def test_sync_from_hits_adds_realtime_candidate(cfg, tmp_path):
    cfg.watch.pool_path = str(tmp_path / "pool.json")
    cfg.watch.patterns = ["w_bottom"]
    cfg.watch.statuses = ["forming"]
    cfg.watch.max_pool = 10
    hit = PatternHit(
        code="000001", name="测试", pattern="w_bottom", pattern_cn="W底",
        status="forming", score=72.0, geom_score=70.0, vol_score=70.0,
        trend_score=70.0, signal_date=pd.Timestamp("2026-09-09"),
    )
    stats = watch.sync_from_hits(cfg, [hit])
    assert stats["新增"] == 1
    assert watch.load_pool(cfg)[0]["code"] == "000001"

    stats = watch.sync_from_hits(cfg, pd.DataFrame([hit.to_row()]))
    assert stats["更新"] == 1


# ---------------- 失败突破：交易日口径（旧实现按自然日，周末/连休会多算） ----------------


def _hist(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"close": [10.0] * len(dates)}, index=pd.to_datetime(dates))


def test_trading_days_ignores_weekend():
    """周五破位、下周一：只过 1 个交易日（按自然日会算成 3）。"""
    hist = _hist(["2026-09-09", "2026-09-10", "2026-09-11"])
    n = monitor._trading_days_since(hist, "2026-09-11", now=datetime(2026, 9, 14, 10, 0))
    assert n == 1


def test_trading_days_ignores_holiday_gap():
    """国庆连休：09-30 到 10-12 之间有 K 线的只有 10-09、10-12 两个交易日。"""
    hist = _hist(["2026-09-29", "2026-09-30", "2026-10-09", "2026-10-12"])
    n = monitor._trading_days_since(hist, "2026-09-30", now=datetime(2026, 10, 12, 10, 0))
    assert n == 2


def test_trading_days_counts_today_intraday():
    """盘中当日 K 线还没入库：把"今天"这个交易日补上。"""
    hist = _hist(["2026-09-29", "2026-09-30"])
    n = monitor._trading_days_since(hist, "2026-09-30", now=datetime(2026, 10, 9, 10, 0))
    assert n == 1


def test_trading_days_weekend_today_not_counted():
    """停跑在周末：今天不是交易日，不能补。"""
    hist = _hist(["2026-09-29", "2026-09-30"])
    n = monitor._trading_days_since(hist, "2026-09-30", now=datetime(2026, 10, 10, 10, 0))
    assert n == 0


def test_trading_days_none_without_hist():
    """没有交易日历就不判失败突破（返回 None），交给止损兜底。"""
    assert monitor._trading_days_since(None, "2026-09-30") is None
    assert monitor._trading_days_since(_hist([]), "2026-09-30") is None
    assert monitor._trading_days_since(_hist(["2026-09-30"]), "") is None


# ---------------- 盘中监控也要带市场环境快照（与收盘 scan 同量纲） ----------------


def test_market_snapshot_cache_recomputes_only_when_inputs_change(monkeypatch, cfg, tmp_path):
    """输入文件没变就不重算——盘中每轮重算要白花约 2 秒。"""
    cfg.data["data_dir"] = str(tmp_path / "store")
    calls: list[int] = []
    monkeypatch.setattr(
        monitor, "_compute_snapshot", lambda c: (calls.append(1), f"SNAP{len(calls)}")[1]
    )
    cache = monitor.MarketSnapshotCache(cfg)

    assert cache.get() == "SNAP1"
    assert cache.get() == "SNAP1"
    assert len(calls) == 1, "输入没变不该重算"

    # 收盘后 update/rebuild-market 写了新数据 → 指纹变了，下一轮必须重算
    store.market_dir(cfg).joinpath("daily_summary.csv").write_text(
        "code,date\n", encoding="utf-8")
    assert cache.get() == "SNAP2"
    assert len(calls) == 2, "输入变了必须重算"


def _fake_stack(monkeypatch, hits_factory, cfg):
    """把 monitor 的检测器栈换成假的，但保留真实的 scan_context 后处理。

    返回 captured 列表（每轮 detect 拿到的 ctx.market_snapshot）。
    """
    from types import SimpleNamespace

    captured: list = []

    class FakeDetector:
        @staticmethod
        def detect(ctx):
            captured.append(ctx.market_snapshot)
            return hits_factory()

    class FakeStore:
        def read_kline(self, code, n):
            return pd.DataFrame({"close": [10.0]}, index=pd.to_datetime(["2026-09-30"]))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

    def fake_build_context(code, name, df, cfg_, explain=False, market_snapshot=None):
        # 字段要够 scan_context 跑完（它要读 cfg.scoring / market_regime / df.index）
        return SimpleNamespace(
            code=code, name=name, df=pd.DataFrame(), trace=["假检测器未命中"], n=0,
            cfg=cfg_, market_regime="neutral", market_snapshot=market_snapshot,
        )

    monkeypatch.setattr(monitor.scan, "build_context", fake_build_context)
    monkeypatch.setattr(monitor.scan, "resolve_detectors", lambda keys: {"fake": FakeDetector})
    monkeypatch.setattr(monitor.store, "open_store", lambda c, readonly=False: FakeStore())
    monkeypatch.setattr(monitor, "build_intraday_df", lambda hist, quote, today=None: hist)
    return captured


_QUOTES = pd.DataFrame([{"code": "000001", "price": 10.0}])
_POOL = [{"code": "000001", "name": "测试", "pattern": "w_bottom"}]


def test_check_once_threads_market_snapshot_to_detectors(monkeypatch, cfg):
    """盘中检测必须带上环境快照，否则同一只票在监控里和收盘报表里差一个环境系数。"""
    captured = _fake_stack(monkeypatch, lambda: [], cfg)
    sentinel = object()
    monitor.check_once(cfg, list(_POOL), _QUOTES, interval=60, market_snapshot=sentinel)
    assert captured == [sentinel], "检测器没拿到环境快照"


def test_check_once_applies_scan_context_postprocessing(monkeypatch, cfg):
    """必须走 scan_context：阶段折扣要取自配置，环境乘子要真的进评分。

    判别力：把 discount 改成 0.5——没走 scan_context 的话 hit 会保留字段默认值 0.85。
    """
    from tradeyh.market.snapshot import MarketSnapshot
    from tradeyh.patterns.base import PatternHit

    cfg.scoring["forming_discount"] = 0.5

    def make_hit():
        return PatternHit(
            code="000001", name="测试", pattern="w_bottom", pattern_cn="W底",
            status="forming", score=80.0, geom_score=80.0, vol_score=70.0,
            trend_score=75.0, signal_date=pd.Timestamp("2026-09-20"), close_price=10.0,
        )

    _fake_stack(monkeypatch, lambda: [make_hit()], cfg)
    neutral = MarketSnapshot(date="2026-09-30", regime_score=50, sentiment_score=50)
    weak = MarketSnapshot(date="2026-09-30", regime_score=20, sentiment_score=20)

    ok = monitor.check_one(cfg, dict(_POOL[0]), {"price": 10.0}, _FakeStoreCtx(), neutral)
    assert ok.hit is not None
    assert ok.hit.forming_discount == 0.5, "折扣没从配置盖上去，说明没走 scan_context"
    assert ok.hit.market_snapshot is neutral

    weak_decision = monitor.check_one(
        cfg, dict(_POOL[0]), {"price": 10.0}, _FakeStoreCtx(), weak
    )
    assert weak_decision.hit.composite_score() < ok.hit.composite_score()


class _FakeStoreCtx:
    """check_one 只需要 st.read_kline。"""

    def read_kline(self, code, n):
        return pd.DataFrame({"close": [10.0]}, index=pd.to_datetime(["2026-09-30"]))


def test_check_once_drops_price_invalidated_entry(monkeypatch, cfg):
    """价格已证伪的 hit 必须判 INVALID 并移出池子，不能当"还在构筑"留着。"""
    from tradeyh.patterns.base import PatternHit

    invalid_hit = PatternHit(
        code="000001", name="测试", pattern="w_bottom", pattern_cn="W底",
        status="formed", score=80.0, geom_score=80.0, vol_score=70.0,
        trend_score=75.0, signal_date=pd.Timestamp("2026-09-20"), close_price=9.0,
        pattern_invalid=True, invalid_reason="close(9.00) < 颈线(10.00)",
    )
    _fake_stack(monkeypatch, lambda: [invalid_hit], cfg)
    decisions, kept = monitor.check_once(cfg, [dict(_POOL[0])], _QUOTES, interval=60)

    assert len(decisions) == 1
    assert decisions[0].kind == monitor.INVALID, "失效的 hit 被当成构筑中留下了"
    assert decisions[0].reason == "close(9.00) < 颈线(10.00)"
    assert kept == [], "失效条目应移出池子"


def test_check_once_without_snapshot_still_works(monkeypatch, cfg):
    """没传快照时按旧行为走（None），不该炸。"""
    captured = _fake_stack(monkeypatch, lambda: [], cfg)
    monitor.check_once(cfg, [{"code": "000001", "pattern": "w_bottom"}], _QUOTES, interval=60)
    assert captured == [None]


def test_market_panel_builders_return_expected_types(cfg, tmp_path):
    """面板组装返回对象、渲染返回字符串——run() 里 print 的是后者。"""
    from datetime import datetime

    from tradeyh.market.panel import MarketPanel

    cfg.data["data_dir"] = str(tmp_path / "store")
    now = datetime(2026, 9, 30, 14, 32, 5)

    panel = monitor.build_market_panel(cfg, None, None, None, None, now)
    assert isinstance(panel, MarketPanel)

    text = monitor.format_market_panel(cfg, None, None, None, None, now)
    assert isinstance(text, str)
    assert "盘面概览" in text


def test_panel_labels_live_vs_local_sources(cfg, tmp_path):
    """面板必须标出每块数据的时间来源，否则分不清实时值和昨天的收盘值。"""
    from datetime import datetime

    import pandas as pd
    import synth

    from tradeyh import store
    from tradeyh.market.snapshot import MarketSnapshot

    cfg.data["storage"] = "csv"
    cfg.data["data_dir"] = str(tmp_path / "store")
    now = datetime(2026, 9, 30, 14, 32, 5)
    snap = MarketSnapshot(date="2026-09-29", regime_score=50, sentiment_score=60)

    # 本地指数K线（供回退用）
    kdir = tmp_path / "store" / "csv" / "kline"
    kdir.mkdir(parents=True, exist_ok=True)
    for key, code in cfg.market.index_codes.items():
        df = synth.make_ohlcv(synth.ramp(4000, 4532, 30), start="2026-08-01")
        df.reset_index().rename(columns={"index": "date"}).assign(
            date=lambda d: d["date"].dt.strftime("%Y-%m-%d")
        )[store.KLINE_COLUMNS].to_csv(kdir / f"{code}.csv", index=False)

    idx = pd.DataFrame({"code": ["idx_000300"], "name": ["沪深300"],
                        "price": [4532.11], "pct_chg": [-0.42]})
    spot = pd.DataFrame({"code": ["600000"], "name": ["浦发银行"],
                         "price": [10.0], "pct_chg": [10.0], "amount": [1e8]})

    live = monitor.format_market_panel(cfg, snap, idx, spot, now, now)
    assert "实时 14:32:05" in live
    assert "快照 14:32:05" in live

    # 实时指数拿不到 → 回落本地收盘K线，并标出来源
    local = monitor.format_market_panel(cfg, snap, None, spot, now, now)
    assert "[本地收盘]" in local
    assert "指数为本地收盘价" in local
    # 宽度来自快照、指数来自本地：两个来源标签都要能区分开
    assert "盘面 [快照 14:32:05]" in local

    # 本地指数也没有 → 明确写"接口不可用"，不能假装有数据
    empty_cfg = cfg
    empty_cfg.data["data_dir"] = str(tmp_path / "empty")
    none_at_all = monitor.format_market_panel(empty_cfg, snap, None, spot, now, now)
    assert "指数 [—]" in none_at_all
    assert "指数实时接口不可用" in none_at_all
