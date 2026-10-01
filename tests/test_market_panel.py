"""盘面概览面板：纯离线单测（不联网、不碰真实 data/）。

重点守的是**涨跌停的板块阈值**——固定 9.5% 的口径会把创业板 +10% 错算成涨停，
而本地数据里 20% 板的票有 1500 多只，错得会很难看。
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradeyh.market.panel import (
    LIMIT_TOLERANCE,
    MarketPanel,
    _limit_threshold,
    build_panel,
    format_panel,
)
from tradeyh.market.snapshot import MarketSnapshot


@pytest.mark.parametrize(
    "code,name,expected",
    [
        ("600000", "浦发银行", 10.0),
        ("000001", "平安银行", 10.0),
        ("300750", "宁德时代", 20.0),          # 创业板
        ("301111", "某创业板", 20.0),
        ("688111", "金山办公", 20.0),          # 科创板
        ("830799", "艾融软件", 30.0),          # 北交所
        ("430047", "诺思兰德", 30.0),
        ("600001", "ST 某某", 5.0),            # ST 优先
        ("300002", "*ST 创业", 5.0),           # ST 且创业板 → 5%
    ],
)
def test_limit_threshold_by_board(code, name, expected):
    assert _limit_threshold(code, name) == expected


def _spot(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """(code, name, pct_chg, amount) → 全市场快照。"""
    return pd.DataFrame(
        {
            "code": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "price": [10.0] * len(rows),
            "pct_chg": [r[2] for r in rows],
            "amount": [r[3] for r in rows],
        }
    )


def test_limit_counts_respect_board_thresholds():
    spot = _spot([
        ("600000", "主板涨停", 10.0, 1e8),      # 涨停
        ("600001", "主板准涨停", 9.9, 1e8),     # 带宽内也算（涨停价取整会抖）
        ("300750", "创业板+10%", 10.0, 1e8),    # 20% 板，+10% 不是涨停
        ("300751", "创业板涨停", 20.0, 1e8),    # 是涨停
        ("688111", "科创板涨停", 19.97, 1e8),   # 带宽内
        ("830799", "北交所涨停", 30.0, 1e8),    # 是涨停
        ("600002", "ST涨停", 5.0, 1e8),         # ST 5%
        ("600003", "ST涨10%", 10.0, 1e8),       # ST 涨 10% 不可能，不该计
        ("600004", "新股首日", 180.0, 1e8),     # 越界，不该计
        ("600005", "跌停", -10.0, 1e8),         # 跌停
        ("300752", "创业跌停", -20.0, 1e8),     # 是跌停
        ("600006", "平盘", 0.0, 1e8),
    ])
    panel = build_panel(spot, None, None, None)
    assert panel.limit_up == 6, f"实得 {panel.limit_up}"
    assert panel.limit_down == 2, f"实得 {panel.limit_down}"
    assert panel.advance == 9
    assert panel.flat == 1
    assert panel.decline == 2


def test_new_listing_180pct_is_not_counted():
    """新股首日 +180% 落在 10% 板阈值之外，带宽判据自动排除。"""
    spot = _spot([("600999", "新股", 180.0, 1e8)])
    panel = build_panel(spot, None, None, None)
    assert panel.limit_up == 0
    assert abs(180.0 - 10.0) > LIMIT_TOLERANCE


def test_amount_sums_whole_market():
    spot = _spot([
        ("600000", "A", 1.0, 6.0e10),
        ("000001", "B", -1.0, 7.0e10),
    ])
    panel = build_panel(spot, None, None, None)
    assert panel.amount == pytest.approx(1.3e11)
    assert "1300亿" in format_panel(panel)


def test_sector_ranking_by_median_pct():
    """行业榜按当日涨幅中位数排序，不是按热度分。"""
    rows = []
    for i in range(4):                       # 煤炭：涨得多
        rows.append((f"6000{i:02d}", f"煤{i}", 5.0 + i, 1e8))
    for i in range(4):                       # 银行：跌得多
        rows.append((f"6001{i:02d}", f"银{i}", -4.0 - i, 1e8))
    for i in range(4):                       # 地产：中间
        rows.append((f"6002{i:02d}", f"地{i}", 0.5, 1e8))
    road = {code: ("煤炭" if code.startswith("6000") else
                   "银行" if code.startswith("6001") else "地产")
            for code, _, _, _ in rows}
    # 热度分故意与涨幅相反，验证排序用的是涨幅
    snap = MarketSnapshot(date="2026-09-30", sector_scores={"煤炭": 10.0, "银行": 90.0, "地产": 50.0},
                          code_to_sector=road)
    panel = build_panel(_spot(rows), None, snap, None)

    assert [s for s, _, _ in panel.top_sectors][0] == "煤炭"
    assert [s for s, _, _ in panel.bottom_sectors][0] == "银行"
    assert panel.top_sectors[0][1] == pytest.approx(6.5)     # 5,6,7,8 的中位数
    assert panel.top_sectors[0][2] == pytest.approx(10.0)    # 热度分照样带出来
    assert [s for s, _, _ in panel.top_sectors] == ["煤炭", "地产", "银行"]


def test_tiny_sectors_are_ignored():
    """1-2 只票的小行业不该靠单票冲进榜单。"""
    rows = [("600000", "独苗", 30.0, 1e8)]
    for i in range(4):
        rows.append((f"6001{i:02d}", f"银{i}", -1.0, 1e8))
    road = {c: ("独苗行业" if c == "600000" else "银行") for c, _, _, _ in rows}
    snap = MarketSnapshot(date="2026-09-30", code_to_sector=road)
    panel = build_panel(_spot(rows), None, snap, None)
    assert "独苗行业" not in [s for s, _, _ in panel.top_sectors]
    assert [s for s, _, _ in panel.top_sectors] == ["银行"]


def test_offline_panel_still_renders_with_notes():
    """没有快照也没有指数时，骨架照常渲染并说明缺什么。"""
    panel = build_panel(None, None, None, None)
    text = format_panel(panel)
    assert "盘面概览" in text
    assert "指数" in text and "环境" in text
    assert any("指数" in n for n in panel.notes)
    assert any("全市场快照" in n for n in panel.notes)


def test_panel_renders_indices_in_config_order(cfg):
    """指数按 config 顺序展示（csi300 → csi1000 → chinext），名称友好化。"""
    quotes = pd.DataFrame({
        "code": ["idx_399006", "idx_000300", "idx_000905"],
        "name": ["创业板指", "沪深300", "中证1000"],
        "price": [2210.30, 4532.11, 6120.88],
        "pct_chg": [-0.88, -0.42, 0.15],
    })
    panel = build_panel(None, quotes, None, cfg, index_as_of="14:32:05")
    assert [r["code"] for r in panel.index_rows] == ["idx_000300", "idx_000905", "idx_399006"]
    assert [r["name"] for r in panel.index_rows] == ["沪深300", "中证1000", "创业板指"]
    text = format_panel(panel)
    assert "沪深300 4532.11 -0.42%" in text
    assert "实时 14:32:05" in text


def test_missing_pct_column_does_not_crash():
    """快照缺 pct_chg（老接口）时应降级而不是炸。"""
    spot = pd.DataFrame({"code": ["600000"], "name": ["A"], "price": [10.0], "amount": [1e8]})
    panel = build_panel(spot, None, None, None)
    assert panel.limit_up == 0 and panel.amount == pytest.approx(1e8)
    assert format_panel(panel)


def test_panel_flags_missing_beijing_market():
    """北交所缺没缺从数据本身判断，缺了要说——否则成交额/涨跌停是偏小的。"""
    without_bj = _spot([("600000", "浦发银行", 1.0, 1e8)])
    assert "快照不含北交所" in build_panel(without_bj, None, None, None).notes

    with_bj = _spot([("600000", "浦发银行", 1.0, 1e8), ("830799", "北交所票", 1.0, 1e8)])
    assert "快照不含北交所" not in build_panel(with_bj, None, None, None).notes


def test_format_amount_units():
    assert "万亿" in format_panel(MarketPanel(amount=1.5e12))
    assert "亿" in format_panel(MarketPanel(amount=8.0e10))
    assert "—" in format_panel(MarketPanel(amount=0.0))
