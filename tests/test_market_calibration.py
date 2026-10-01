"""市场环境乘子标定 / 行业热度历史 / 回填。

与 test_market_module.py 分开：那边验的是各个 0-100 分数本身算得对不对，
这边守的是"分数怎么变成综合评分"以及"历史数据从哪来"。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import synth

from tradeyh import store
from tradeyh.market import backfill
from tradeyh.market.sector import _classify_stage, compute_sector_heat
from tradeyh.market.sector_fetch import code_to_sector_map
from tradeyh.market.snapshot import (
    DEFAULT_BANDS,
    MarketSnapshot,
    _bands,
    _factor,
    _limit_up_as_of,
    limit_up_streaks,
    load_heat_history,
    prev_heat_from_history,
    save_heat_history,
)


# ---------------------------------------------------------------------------
# 乘子标定
# ---------------------------------------------------------------------------


def test_factor_bands_endpoints_and_neutral():
    """0 分 → lo，50 分 → 1.00，100 分 → hi。"""
    assert _factor(0, 0.85, 1.15) == pytest.approx(0.85)
    assert _factor(50, 0.85, 1.15) == pytest.approx(1.0)
    assert _factor(100, 0.85, 1.15) == pytest.approx(1.15)
    assert _factor(0, 0.95, 1.05) == pytest.approx(0.95)
    assert _factor(50, 0.95, 1.05) == pytest.approx(1.0)
    assert _factor(100, 0.95, 1.05) == pytest.approx(1.05)


def test_factor_is_monotonic_and_clamped():
    scores = list(range(0, 101))
    factors = [_factor(s, *DEFAULT_BANDS["market"]) for s in scores]
    assert factors == sorted(factors), "分数越高乘子必须不降"
    assert _factor(-20, 0.85, 1.15) == pytest.approx(0.85)   # 下界截断
    assert _factor(180, 0.85, 1.15) == pytest.approx(1.15)   # 上界截断


def test_snapshot_factors_use_configured_bands(cfg):
    """带宽必须走配置，不能硬编码。"""
    cfg.market["factor_bands"] = {"market": {"lo": 0.5, "hi": 2.0}}
    assert _bands(cfg.market, "market") == (0.5, 2.0)
    assert _bands(cfg.market, "sentiment") == DEFAULT_BANDS["sentiment"], "缺的键要回落默认"
    assert _bands(None, "sector") == DEFAULT_BANDS["sector"]

    snap = MarketSnapshot(
        date="2026-09-30", regime_score=100, market_band=_bands(cfg.market, "market"),
    )
    assert snap.market_factor == pytest.approx(2.0)


def _hit(**over):
    from tradeyh.patterns.base import PatternHit

    kw = dict(
        code="000001", name="测试", pattern="break_bottom", pattern_cn="破底翻",
        status="formed", score=80, geom_score=80, vol_score=70, trend_score=75,
        signal_date=pd.Timestamp("2026-09-20"), close_price=10.0,
    )
    kw.update(over)
    return PatternHit(**kw)


def test_neutral_snapshot_is_exact_identity():
    """全中性快照下 composite_score 必须与"没有快照"逐位相等。

    这是承重墙：min_score 门槛、watch.min_score、Excel 色阶锚点(0/60/100)
    全都假设综合评分留在 0-100 量纲上。曾经三个乘子相乘把 60 分压成 2 分。
    """
    base = _hit().composite_score()
    neutral = MarketSnapshot(date="2026-09-30", regime_score=50, sentiment_score=50)
    assert _hit(market_snapshot=neutral).composite_score() == base


def test_weak_environment_shrinks_but_not_collapses():
    """弱势日应当扣分，但只在带宽内扣（不再是 20 倍缩水）。"""
    base = _hit().composite_score()
    weak = MarketSnapshot(date="2026-09-30", regime_score=9.8, sentiment_score=28.4)
    got = _hit(market_snapshot=weak).composite_score()
    floor = base * (
        _factor(9.8, *DEFAULT_BANDS["market"])
        * _factor(28.4, *DEFAULT_BANDS["sentiment"])
        * _factor(50, *DEFAULT_BANDS["sector"])
    )
    assert got == pytest.approx(floor)
    assert got > base * 0.7, f"扣分不该超过带宽，实得 {got:.1f} / 基准 {base:.1f}"


def test_hot_sector_gets_bonus_cold_sector_gets_penalty():
    """板块热度是双向的：热板块>1，冷板块<1（旧公式只会扣分）。"""
    snap = MarketSnapshot(
        date="2026-09-30",
        sector_scores={"热": 90, "冷": 20},
        code_to_sector={"111111": "热", "222222": "冷"},
    )
    assert snap.sector_factor("111111") > 1.0
    assert snap.sector_factor("222222") < 1.0
    assert snap.sector_factor("333333") == 1.0  # 未知板块 → 中性


# ---------------------------------------------------------------------------
# 行业热度：历史 + 阶段
# ---------------------------------------------------------------------------


def _sector_frame(n_days: int = 15) -> pd.DataFrame:
    """两个行业、各 2 只股票、n_days 天：电子量价齐升，银行横盘缩量。"""
    rows = []
    bdays = pd.bdate_range("2026-08-03", periods=n_days)
    for i, day in enumerate(bdays):
        for code, sector in (("000001", "电子"), ("000002", "电子"),
                             ("600000", "银行"), ("600001", "银行")):
            hot = sector == "电子"
            pct = (3.0 + i * 0.3) if hot else (-0.2 - i * 0.05)
            amount = (5.0e8 * (1 + i * 0.12)) if hot else (1.0e9 * (1 - i * 0.01))
            rows.append({
                "code": code, "date": str(day.date()), "pct_chg": pct,
                "high": 10.0 * (1 + pct / 100), "close": 10.0, "amount": amount,
            })
    return pd.DataFrame(rows)


def test_sector_subindicators_stop_being_constants():
    """有 20 日历史后，量比/趋势不能再是恒定的 50（历史只有一天时的老毛病）。"""
    df = _sector_frame(15)
    weights = {
        "relative_return": 0.20, "volume_ratio": 0.15, "breadth": 0.15,
        "trend": 0.20, "capital_intensity": 0.15, "outperform_days": 0.15,
    }
    today = df[df["date"] == df["date"].max()]
    result = compute_sector_heat(today, df, {"000001": "电子", "000002": "电子",
                                             "600000": "银行", "600001": "银行"},
                                 None, weights)
    hot, cold = result["电子"]["detail"], result["银行"]["detail"]
    for key in ("volume_ratio", "trend"):
        assert hot[key] != 50.0 or cold[key] != 50.0, f"{key} 仍是常数"
        assert hot[key] > cold[key], f"{key} 未区分冷热行业"
    assert result["电子"]["score"] > result["银行"]["score"]


def test_stage_startup_and_mainup_are_reachable():
    """有分数轨迹后，启动/主升/分歧/退潮都必须可达（旧版只有退潮/扩散）。"""
    assert _classify_stage(45, [30, 35], 65, 1.0) == "启动"
    assert _classify_stage(75, [60, 68], 70, 1.0) == "主升"
    assert _classify_stage(90, [85, 88], 80, 0.8) == "高潮"
    assert _classify_stage(85, [95, 92], 60, 1.0) == "分歧"
    assert _classify_stage(30, [50, 45], 30, 1.0) == "退潮"


def test_prev_heat_uses_only_days_before_as_of():
    history = {"2026-09-24": {"电子": 10.0}, "2026-09-25": {"电子": 20.0},
               "2026-09-28": {"电子": 30.0}}
    assert prev_heat_from_history(history, "2026-09-28") == {"电子": [10.0, 20.0]}
    assert prev_heat_from_history(history, "2026-09-25") == {"电子": [10.0]}


def test_heat_history_roundtrip(cfg, tmp_path):
    cfg.data["data_dir"] = str(tmp_path / "store")
    save_heat_history(cfg, {"2026-09-28": {"电子": 62.34}, "2026-09-29": {"电子": 66.0}})
    got = load_heat_history(cfg)
    assert set(got) == {"2026-09-28", "2026-09-29"}
    assert got["2026-09-28"]["电子"] == 62.34
    # 损坏文件按空历史处理，不该炸
    store.market_dir(cfg).joinpath("sector_heat_history.json").write_text("{bad", encoding="utf-8")
    assert load_heat_history(cfg) == {}


# ---------------------------------------------------------------------------
# 涨停榜口径
# ---------------------------------------------------------------------------


def test_limit_up_takes_previous_trading_day():
    history = {"2026-09-25": ["000001"], "2026-09-28": ["600000"]}
    prev_codes, streaks = _limit_up_as_of(history, "2026-09-28")
    assert prev_codes == ["000001"], "拿到的必须是上一交易日，不是当日"
    assert streaks == {"000001": 1}
    # 当日这份还没结束，不应出现在 as_of=09-28 的口径里
    assert "600000" not in streaks


def test_limit_up_streak_resets_after_dropping_out():
    """连板必须连续，中间掉队要清零，不能算成累计涨停天数。"""
    history = {"d1": ["A", "B"], "d2": ["A"], "d3": ["A", "B"]}
    assert limit_up_streaks(history) == {"A": 3, "B": 1}


def test_parse_limit_up_history_accepts_legacy_shape():
    from tradeyh.market.snapshot import parse_limit_up_history

    assert parse_limit_up_history({"date": "2026-09-28", "codes": ["000001"]}) == {
        "2026-09-28": ["000001"]
    }
    assert parse_limit_up_history(["nonsense"]) == {}


# ---------------------------------------------------------------------------
# 尾读 & 回填
# ---------------------------------------------------------------------------


def test_read_tail_rows_matches_pandas(tmp_path):
    path = tmp_path / "k.csv"
    full = pd.DataFrame({
        "date": [f"2026-01-{i:02d}" for i in range(1, 21)],
        "open": np.arange(20, dtype=float), "close": np.arange(20, dtype=float) + 1,
    })
    full.to_csv(path, index=False)
    for n in (1, 5, 20, 50):
        rows = store.read_tail_rows_csv(path, n)
        assert len(rows) == min(n, len(full))
        assert rows[-1]["date"] == full["date"].iloc[-1]
    assert store.read_tail_rows_csv(path, 0) == []
    assert store.read_tail_rows_csv(tmp_path / "missing.csv", 5) == []


def test_read_tail_rows_handles_crlf_and_no_trailing_newline(tmp_path):
    path = tmp_path / "crlf.csv"
    path.write_bytes(b"date,close\r\n2026-01-01,1\r\n2026-01-02,2\r\n2026-01-03,3")
    rows = store.read_tail_rows_csv(path, 2)
    assert rows == [{"date": "2026-01-02", "close": "2"},
                    {"date": "2026-01-03", "close": "3"}]


def _seed_csv_store(cfg, root: Path, n_days: int = 12) -> Path:
    """在临时目录造一个 csv 库：4 只股票、2 个行业。

    电子板块刻意"前段横盘、末尾才启动"——这样行业热度历史会呈现一条上升轨迹，
    可以验证回填没有把未来数据算进历史日。
    """
    kdir = root / "csv" / "kline"
    kdir.mkdir(parents=True, exist_ok=True)
    warming = n_days - 6
    closes = {
        "000001": [10.0] * (n_days - 2) + [11.0, 12.1],   # 末尾连两个 +10%
        "000002": np.concatenate([np.full(warming, 10.0), synth.ramp(10.0, 13.0, 6)]),
        "600000": synth.ramp(10.0, 10.5, n_days),
        "600001": synth.ramp(10.0, 9.4, n_days),
    }
    vols = {
        "000001": np.concatenate([np.full(warming + 4, 1.0e6), np.full(2, 3.0e6)]),
        "000002": np.concatenate([np.full(warming, 1.0e6), np.linspace(1.5e6, 3.0e6, 6)]),
        "600000": np.linspace(3.0e6, 1.0e6, n_days),
        "600001": np.full(n_days, 1.0e6),
    }
    for code, cl in closes.items():
        df = synth.make_ohlcv(cl, volumes=vols[code], start="2026-08-03")
        df.reset_index().rename(columns={"index": "date"}).assign(
            date=lambda d: d["date"].dt.strftime("%Y-%m-%d")
        )[store.KLINE_COLUMNS].to_csv(kdir / f"{code}.csv", index=False)
    # 行业分类缓存：不写的话 load_sector_map 会去联网拉 baostock
    store.market_dir(cfg).joinpath("sector_map.json").write_text(
        json.dumps({
            "000001": {"sector": "电子", "name": "A"},
            "000002": {"sector": "电子", "name": "B"},
            "600000": {"sector": "银行", "name": "C"},
            "600001": {"sector": "银行", "name": "D"},
        }, ensure_ascii=False), encoding="utf-8",
    )
    return kdir


@pytest.fixture
def backfill_cfg(cfg, tmp_path):
    root = tmp_path / "store"
    cfg.data["storage"] = "csv"
    cfg.data["data_dir"] = str(root)
    cfg.data["out_dir"] = str(tmp_path / "out")
    cfg.data["max_workers"] = 2
    cfg.market["history_days"] = 12
    _seed_csv_store(cfg, root, 12)
    return cfg


def test_rebuild_market_rebuilds_all_three_files(backfill_cfg):
    stats = backfill.rebuild_market_data(backfill_cfg)
    assert stats["codes"] == 4
    assert stats["days"] == 12, "daily_summary 应当覆盖整个窗口，而不是只有一天"

    mdir = store.market_dir(backfill_cfg)
    summary = pd.read_csv(mdir / "daily_summary.csv", dtype={"code": str})
    assert summary["date"].nunique() == 12
    assert set(summary["code"]) == {"000001", "000002", "600000", "600001"}

    # 涨停历史必须是多日结构，且 000001 连板 2
    hist = json.loads((mdir / "limit_up_history.json").read_text(encoding="utf-8"))
    assert isinstance(hist, dict) and len(hist) == 12
    assert all(isinstance(v, list) for v in hist.values())
    assert "000001" in hist[max(hist)]
    streaks = json.loads((mdir / "limit_up_streak.json").read_text(encoding="utf-8"))
    assert streaks.get("000001") == 2, f"连板应逐日推出来，实得 {streaks}"

    # 热度历史被预热 → 阶段判定当场可用
    heat = load_heat_history(backfill_cfg)
    assert len(heat) == 12
    assert stats["heat_days"] == 12


def test_seeded_heat_history_has_no_lookahead(backfill_cfg):
    """历史日的热度只能用"当日为止"的数据算。

    如果整天整段传进 compute_sector_heat，tail(5)/tail(20) 会取到数据末尾，
    历史日就被未来的行情污染了（实测早期分数从 35 被抬到 53）。
    """
    backfill.rebuild_market_data(backfill_cfg)
    heat = load_heat_history(backfill_cfg)
    days = sorted(heat)
    assert len(days) == 12

    summary = pd.read_csv(store.market_dir(backfill_cfg) / "daily_summary.csv",
                          dtype={"code": str})
    c2s = code_to_sector_map(backfill_cfg)
    weights = dict(backfill_cfg.market.sector_weights)
    day = days[0]
    truncated = compute_sector_heat(
        summary[summary["date"] == day], summary[summary["date"] <= day], c2s, None, weights)
    leaked = compute_sector_heat(summary[summary["date"] == day], summary, c2s, None, weights)

    # 先确认两种口径确实不同，否则这条断言是空转的
    assert leaked["电子"]["score"] > truncated["电子"]["score"] + 5, (
        f"两种口径结果太接近，测试失去判别力: {leaked['电子']['score']} vs {truncated['电子']['score']}"
    )
    assert heat[day]["电子"] == pytest.approx(truncated["电子"]["score"])

    path = [heat[d]["电子"] for d in days]
    assert len(set(path)) > 3, f"轨迹被抹平了，实际序列 {path}"


def test_rebuild_market_result_feeds_back_into_snapshot(backfill_cfg):
    """回填出来的数据要能让行业热度真的分冷热，而不是一片 50。"""
    backfill.rebuild_market_data(backfill_cfg)
    from tradeyh.market.snapshot import _load_daily_summary

    today, history = _load_daily_summary(backfill_cfg)
    assert history["date"].nunique() == 12
    result = compute_sector_heat(
        today, history, {"000001": "电子", "000002": "电子",
                         "600000": "银行", "600001": "银行"},
        None, dict(backfill_cfg.market.sector_weights),
    )
    assert result["电子"]["score"] > result["银行"]["score"]


def test_rebuild_market_keeps_existing_file_when_no_data(cfg, tmp_path):
    """库里没有 K 线时不该把已有文件清空。"""
    cfg.data["storage"] = "csv"
    cfg.data["data_dir"] = str(tmp_path / "empty_store")
    (tmp_path / "empty_store" / "csv" / "kline").mkdir(parents=True)
    existing = store.market_dir(cfg) / "daily_summary.csv"
    existing.write_text("code,date\n", encoding="utf-8")
    stats = backfill.rebuild_market_data(cfg)
    assert stats["rows"] == 0
    assert existing.read_text(encoding="utf-8") == "code,date\n"


def test_rebuild_market_works_on_sqlite_backend(cfg, tmp_path):
    """read_kline 把日期放在索引上（没有 date 列），非 csv 后端也要能回填。

    否则写出来的 daily_summary 全是空日期，整段历史都废掉。
    """
    db = tmp_path / "kline.db"
    cfg.data["storage"] = "sqlite"
    cfg.data["db_path"] = str(db)
    cfg.data["data_dir"] = str(tmp_path / "store")
    cfg.market["history_days"] = 8

    conn = store.connect(db)
    store.init_db(conn)
    rows = []
    for code, closes in {
        "000001": synth.ramp(10.0, 12.0, 8), "600000": synth.ramp(10.0, 10.4, 8),
    }.items():
        df = synth.make_ohlcv(closes, start="2026-08-03").reset_index().rename(
            columns={"index": "date"})
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        store.upsert_kline(conn, code, df[store.KLINE_COLUMNS])
        rows.append({"code": code, "name": code})
    store.upsert_stock_info(conn, pd.DataFrame(rows))
    conn.commit()
    conn.close()
    store.market_dir(cfg).joinpath("sector_map.json").write_text(
        json.dumps({"000001": {"sector": "电子"}, "600000": {"sector": "银行"}}),
        encoding="utf-8",
    )

    stats = backfill.rebuild_market_data(cfg)
    assert stats["codes"] == 2
    assert stats["days"] == 8
    summary = pd.read_csv(store.market_dir(cfg) / "daily_summary.csv", dtype={"code": str})
    assert len(summary) == 16
    assert summary["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(), \
        f"日期必须统一成 YYYY-MM-DD，实得 {summary['date'].unique()[:3]}"
    assert summary["close"].notna().all()


def test_update_writes_multiday_limit_up_schema(backfill_cfg):
    """update 的增量路径也要写 {日期:[代码]}，且键用数据日而不是墙钟日。"""
    from tradeyh import update

    mdir = store.market_dir(backfill_cfg)
    with store.open_store(backfill_cfg) as st:
        update._write_daily_summary(backfill_cfg, st, ["000001", "000002", "600000", "600001"])

    hist = json.loads((mdir / "limit_up_history.json").read_text(encoding="utf-8"))
    assert isinstance(hist, dict) and not any(k == "date" for k in hist)
    data_date = pd.read_csv(mdir / "daily_summary.csv")["date"].max()
    assert list(hist) == [data_date], "键必须是数据日"
    assert "000001" in hist[data_date], "000001 末日涨停，应进榜"

    streaks = json.loads((mdir / "limit_up_streak.json").read_text(encoding="utf-8"))
    assert streaks == {"000001": 1}, "首日涨停连板从 1 开始"

    # 同一天重跑是幂等的（不该把连板数累加）
    with store.open_store(backfill_cfg) as st:
        update._write_daily_summary(backfill_cfg, st, ["000001", "000002", "600000", "600001"])
    streaks2 = json.loads((mdir / "limit_up_streak.json").read_text(encoding="utf-8"))
    assert streaks2 == streaks
