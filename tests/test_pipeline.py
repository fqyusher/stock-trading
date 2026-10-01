"""端到端：合成库 → 扫描 → 报表。"""

from __future__ import annotations

import pandas as pd
import pytest
import synth

from tradeyh import report, scan, store
from tradeyh.patterns import DETECTORS

CASES = {
    "000001": ("W底股", synth.ideal_w_bottom),
    "000002": ("M头股", synth.ideal_m_top),
    "600000": ("升通道股", synth.ideal_rising_channel),
    "600001": ("破底翻股", synth.strict_break_bottom),
    "600002": ("箱体股", synth.ideal_box_range),
}


@pytest.fixture
def seeded_cfg(cfg, tmp_path):
    """把合成K线写进临时 sqlite 库，并把 cfg 指向它。"""
    db_path = tmp_path / "kline.db"
    cfg.data["storage"] = "sqlite"  # 默认是 csv，这里明确用 sqlite 种子库
    cfg.data["db_path"] = str(db_path)
    cfg.data["out_dir"] = str(tmp_path / "out")
    cfg.universe["min_bars"] = 100  # 合成序列只有 100~160 根

    conn = store.connect(db_path)
    store.init_db(conn)
    rows = []
    for code, (name, builder) in CASES.items():
        df = builder().reset_index().rename(columns={"index": "date"})
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        store.upsert_kline(conn, code, df[store.KLINE_COLUMNS])
        rows.append({"code": code, "name": name})
    store.upsert_stock_info(conn, pd.DataFrame(rows))
    conn.commit()
    conn.close()
    return cfg


def test_scan_market_finds_expected_patterns(seeded_cfg):
    hits = scan.scan_market(seeded_cfg, workers=1)
    assert not hits.empty

    # 按综合评分降序排列
    scores = hits["综合评分"].tolist()
    assert scores == sorted(scores, reverse=True), "应按综合评分降序排列"

    assert set(hits["状态"]) <= {"已形成", "构筑中", "已失效"}
    assert hits["名称"].notna().all()

    found = hits.groupby("代码")["形态key"].apply(set)
    assert "w_bottom" in found["000001"]
    assert "m_top" in found["000002"]
    assert "rising_channel" in found["600000"]
    assert "break_bottom" in found["600001"]
    assert "box_range" in found["600002"]


def test_pattern_filter(seeded_cfg):
    hits = scan.scan_market(seeded_cfg, detector_keys=["w_bottom"], workers=1)
    assert set(hits["形态key"]) == {"w_bottom"}


def test_min_score_and_status_filters(seeded_cfg):
    hits = scan.scan_market(seeded_cfg, statuses=["formed"], min_score=50, workers=1)
    assert (hits["综合评分"] >= 50).all()
    assert set(hits["状态"]) == {"已形成"}


def test_min_bars_filter_skips_short_history(seeded_cfg):
    seeded_cfg.universe["min_bars"] = 10_000
    assert scan.scan_market(seeded_cfg, workers=1).empty


def test_multiprocess_matches_single_process(seeded_cfg):
    single = scan.scan_market(seeded_cfg, workers=1)
    multi = scan.scan_market(seeded_cfg, workers=2)
    pd.testing.assert_frame_equal(
        single.sort_values(["代码", "形态key"]).reset_index(drop=True),
        multi.sort_values(["代码", "形态key"]).reset_index(drop=True),
    )


def test_scan_one_explain_collects_trace(seeded_cfg):
    hits, ctx = scan.scan_one(seeded_cfg, "000001", explain=True)
    assert ctx is not None and ctx.name == "W底股"
    assert any(h.pattern == "w_bottom" for h in hits)
    assert ctx.trace, "explain 模式下应记录判定过程"


def test_scan_one_missing_code(seeded_cfg):
    hits, ctx = scan.scan_one(seeded_cfg, "999999")
    assert hits == [] and ctx is None


def test_resolve_detectors_accepts_both_key_styles():
    assert set(scan.resolve_detectors(["w_bottom"])) == {"w_bottom"}
    assert set(scan.resolve_detectors(["rising_channel"])) == {"channel"}
    # 不写死名单：新注册一个检测器不该让这条测试失效
    assert set(scan.resolve_detectors(None)) == set(DETECTORS)
    with pytest.raises(ValueError, match="未知形态"):
        scan.resolve_detectors(["no_such_pattern"])


def test_write_report_produces_readable_files(seeded_cfg):
    hits = scan.scan_market(seeded_cfg, workers=1)
    paths = report.write_report(hits, seeded_cfg.data.out_dir, "20260828")

    assert paths["csv"].exists() and paths["xlsx"].exists()

    csv_back = pd.read_csv(paths["csv"], dtype={"代码": str})
    assert len(csv_back) == len(hits)
    assert list(csv_back.columns)[:6] == [
        "代码", "名称", "形态", "形态key", "方向", "状态"
    ]

    sheets = pd.read_excel(paths["xlsx"], sheet_name=None)
    assert "汇总" in sheets
    for pattern_cn in hits["形态"].unique():
        assert pattern_cn in sheets, f"缺少 {pattern_cn} 分表"
    # 合成库里有 M头，做多榜单必须把它挡在外面
    longs = report.long_only(hits)
    if not longs.empty:
        assert "做多榜单" in sheets
        assert "M头" not in set(sheets["做多榜单"]["形态"])


def test_write_report_handles_no_hits(seeded_cfg):
    paths = report.write_report(pd.DataFrame(), seeded_cfg.data.out_dir, "20260828")
    assert paths["csv"].exists()
    assert "xlsx" not in paths


def test_print_summary_runs(seeded_cfg, capsys):
    report.print_summary(scan.scan_market(seeded_cfg, workers=1), top=3)
    out = capsys.readouterr().out
    assert "命中统计" in out and "综合评分前" in out
