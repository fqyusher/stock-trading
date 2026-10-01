"""盘面概览面板：指数点位 / 涨跌停家数 / 成交额 / 行业涨幅榜。

盘中监控每轮调一次 `format_panel(build_panel(...))` 打印。

**纯计算模块**：不联网、不读写文件，所有 I/O 由调用方（monitor.py）负责——
这样面板逻辑可以完全离线单测，网络不通时也能靠本地收盘数据渲染出一样的骨架。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

# 涨跌停判据的带宽（百分点）：涨跌幅落在 [阈值-k, 阈值+k] 内才算封板。
# 为什么要带宽而不是 pct >= 阈值：各板块阈值不同（主板10/创业板科创板20/北交所30/ST5），
# 而涨停价按 0.01 元取整，实际涨跌幅会在阈值附近抖动。带宽同时天然排除了
# 新股上市首日那种 +180% 的越界值。
LIMIT_TOLERANCE = 0.15

# 行业榜最少成分股数：证监会行业里有些只有 1-2 只票，中位涨幅会被单票带飞，
# 排进"领涨板块"会误导。
MIN_SECTOR_SIZE = 3

INDEX_CN = {"csi300": "沪深300", "csi1000": "中证1000", "chinext": "创业板指", "csi500": "中证500"}


@dataclass
class MarketPanel:
    index_rows: list[dict] = field(default_factory=list)   # [{code,name,price,pct_chg}]
    limit_up: int = 0
    limit_down: int = 0
    advance: int = 0
    decline: int = 0
    flat: int = 0
    amount: float = 0.0                                    # 全市场成交额（元）
    top_sectors: list[tuple[str, float, float]] = field(default_factory=list)
    bottom_sectors: list[tuple[str, float, float]] = field(default_factory=list)
    regime_label: str = ""
    regime_score: float = 0.0
    sentiment_score: float = 0.0
    market_factor: float = 1.0
    sentiment_factor: float = 1.0
    index_as_of: str = ""
    breadth_as_of: str = ""
    breadth_source: str = ""
    notes: list[str] = field(default_factory=list)


def _limit_threshold(code: str, name: str) -> float:
    """该股票的涨跌停幅度（百分点）。按代码前缀 + 名称判板块。"""
    code = str(code).zfill(6)
    upper = str(name or "").upper()
    if "ST" in upper:                      # ST / *ST 一律 5%
        return 5.0
    if code.startswith("688"):             # 科创板
        return 20.0
    if code.startswith("300") or code.startswith("301"):  # 创业板
        return 20.0
    if code.startswith("8") or code.startswith("4"):      # 北交所
        return 30.0
    return 10.0                            # 沪深主板


def _count_limits(frame: pd.DataFrame) -> tuple[int, int]:
    """按每只票自己的板块阈值数涨跌停。"""
    if frame.empty or "pct_chg" not in frame.columns or "code" not in frame.columns:
        return 0, 0
    names = frame["name"] if "name" in frame.columns else pd.Series([""] * len(frame))
    up = down = 0
    for code, name, pct in zip(frame["code"], names, frame["pct_chg"]):
        if pd.isna(pct):
            continue
        thr = _limit_threshold(code, name)
        pct = float(pct)
        if abs(pct - thr) <= LIMIT_TOLERANCE:
            up += 1
        elif abs(pct + thr) <= LIMIT_TOLERANCE:
            down += 1
    return up, down


def _sector_ranking(
    frame: pd.DataFrame, sector_map: dict[str, str], heat: dict[str, float],
    top_n: int = 3, min_size: int = MIN_SECTOR_SIZE,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """按行业当日涨幅中位数排名，返回 (领涨, 垫底)，每项 (行业, 中位涨幅%, 热度分)。"""
    if frame.empty or "pct_chg" not in frame.columns or not sector_map:
        return [], []
    df = frame[["code", "pct_chg"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["sector"] = df["code"].map(sector_map)
    df = df.dropna(subset=["sector", "pct_chg"])
    if df.empty:
        return [], []
    grouped = df.groupby("sector")["pct_chg"]
    sizes = grouped.size()
    medians = grouped.median()
    keep = [s for s in medians.index if int(sizes.get(s, 0)) >= min_size]
    if not keep:
        return [], []
    ranked = sorted(keep, key=lambda s: -float(medians[s]))
    fmt = [(s, float(medians[s]), float(heat.get(s, 0.0))) for s in ranked]
    return fmt[:top_n], list(reversed(fmt[-top_n:]))


def _fmt_pct(value: float) -> str:
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "—"


_SECTOR_CODE_RE = re.compile(r"^[A-Z]\d{2}")  # 证监会行业名带的 A05/B06 前缀，展示时去掉


def _short_sector(name: str) -> str:
    """展示用行业名：去掉 A05/B06 这类分类码前缀，太长就截断。"""
    text = _SECTOR_CODE_RE.sub("", str(name)) or str(name)
    return text if len(text) <= 14 else text[:13] + "…"


def _fmt_amount(amount: float) -> str:
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return "—"
    if amount <= 0:
        return "—"
    if amount >= 1e12:
        return f"{amount / 1e12:.2f}万亿"
    return f"{amount / 1e8:.0f}亿"


def build_panel(
    spot: pd.DataFrame | None,
    index_quotes: pd.DataFrame | None,
    snapshot=None,
    cfg=None,
    *,
    sector_map: dict[str, str] | None = None,
    index_as_of: str = "",
    breadth_as_of: str = "",
    breadth_source: str = "",
    index_order: list[str] | None = None,
) -> MarketPanel:
    """把"全市场快照 + 指数报价 + 环境快照"组装成面板数据。

    三者都可以是 None（离线），缺失的部分留空并在 notes 里说明，不抛异常。
    """
    panel = MarketPanel(index_as_of=index_as_of, breadth_as_of=breadth_as_of,
                        breadth_source=breadth_source)

    # 指数：按配置顺序展示（csi300 / csi1000 / chinext）
    if index_quotes is not None and not index_quotes.empty:
        names = INDEX_CN
        if cfg is not None and hasattr(cfg, "market"):
            order = index_order or list(cfg.market.index_codes.values())
            name_by_code = {
                code: names.get(key, str(code))
                for key, code in cfg.market.index_codes.items()
            }
        else:
            order, name_by_code = [], {}
        rows = {str(r["code"]): dict(r) for r in index_quotes.to_dict("records")}
        ordered = [rows[c] for c in order if c in rows]
        ordered += [r for c, r in rows.items() if c not in set(order)]
        for row in ordered:
            row["name"] = name_by_code.get(str(row["code"]), row.get("name") or str(row["code"]))
        panel.index_rows = ordered
    else:
        panel.notes.append("指数实时接口不可用")

    # 环境 / 情绪
    if snapshot is not None:
        panel.regime_label = snapshot.regime_label
        panel.regime_score = float(snapshot.regime_score)
        panel.sentiment_score = float(snapshot.sentiment_score)
        panel.market_factor = float(snapshot.market_factor)
        panel.sentiment_factor = float(snapshot.sentiment_factor)
    else:
        panel.notes.append("无环境快照（评分不做环境修正）")

    # 盘面宽度
    if spot is not None and not spot.empty:
        panel.amount = float(pd.to_numeric(spot.get("amount"), errors="coerce").sum())
        panel.limit_up, panel.limit_down = _count_limits(spot)
        if "pct_chg" in spot.columns:
            pct = pd.to_numeric(spot["pct_chg"], errors="coerce").dropna()
            panel.advance = int((pct > 0.001).sum())
            panel.decline = int((pct < -0.001).sum())
            panel.flat = int((pct.abs() <= 0.001).sum())
        if sector_map is None and snapshot is not None:
            sector_map = snapshot.code_to_sector
        heat = getattr(snapshot, "sector_scores", None) or {}
        panel.top_sectors, panel.bottom_sectors = _sector_ranking(spot, sector_map or {}, heat)

        # 北交所有没有，直接从数据看：东财清单含北交所（8xxxxx/4xxxxx/920xxx），
        # 本地库清单没有。缺了就得说明，否则成交额/涨跌停是偏小的。
        codes = spot["code"].astype(str).str.zfill(6)
        if not codes.str.match(r"^[84]|^920").any():
            panel.notes.append("快照不含北交所")
    else:
        panel.notes.append("无全市场快照（涨跌停/成交额/行业榜缺）")

    return panel


def format_panel(panel: MarketPanel, width: int = 78) -> str:
    """渲染成多行文本。每块自带数据时间标签，便于区分实时与收盘数据。"""
    lines: list[str] = []
    stamp = panel.index_as_of or panel.breadth_as_of or ""
    title = f" 盘面概览 {stamp} " if stamp else " 盘面概览 "
    lines.append(title.center(max(30, width), "─"))

    if panel.index_as_of:
        tag = f"[实时 {panel.index_as_of}]"
    elif panel.index_rows:
        tag = "[本地收盘]"
    else:
        tag = "[—]"
    if panel.index_rows:
        parts = [
            f"{r.get('name', r.get('code'))} {float(r['price']):.2f} {_fmt_pct(r.get('pct_chg'))}"
            for r in panel.index_rows
            if r.get("price") is not None and not pd.isna(r.get("price"))
        ]
        lines.append(f"指数 {tag}  " + " | ".join(parts))
    else:
        lines.append(f"指数 {tag}  —")

    if panel.regime_label:
        lines.append(
            f"环境 [{panel.breadth_as_of or '数据日'}]  {panel.regime_label} "
            f"{panel.regime_score:.1f} | 情绪 {panel.sentiment_score:.1f} | 乘子 "
            f"市场{panel.market_factor:.3f} 情绪{panel.sentiment_factor:.3f}"
        )
    else:
        lines.append("环境 [无]  —")

    btag = f"[{panel.breadth_source} {panel.breadth_as_of}]" if panel.breadth_source else "[—]"
    if panel.amount or panel.advance or panel.decline or panel.limit_up or panel.limit_down:
        lines.append(
            f"盘面 {btag}  涨停 {panel.limit_up} 跌停 {panel.limit_down} | "
            f"涨 {panel.advance} 跌 {panel.decline} 平 {panel.flat} | "
            f"成交 {_fmt_amount(panel.amount)}"
        )
    else:
        lines.append(f"盘面 {btag}  —")

    def _sector_line(title: str, items: list[tuple[str, float, float]]) -> str:
        if not items:
            return f"{title} —"
        body = " | ".join(
            f"{_short_sector(s)} {_fmt_pct(p)}(热度{h:.0f})" for s, p, h in items
        )
        return f"{title} {body}"

    lines.append(f"{_sector_line('领涨板块 ' + btag, panel.top_sectors)}")
    lines.append(f"{_sector_line('垫底板块 ' + btag, panel.bottom_sectors)}")
    for note in panel.notes:
        lines.append(f"  · {note}")
    return "\n".join(lines)
