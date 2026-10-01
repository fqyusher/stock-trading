"""SectorHeat (0~100)：行业热度评估 + 行业阶段判定。

6 个子指标：
1. 5d/20d 相对大盘涨幅 (20%)
2. 成交额相对 20 日均值 (15%)
3. 行业内上涨股票占比 (15%)
4. 行业指数趋势 (MA20/MA60/斜率) (20%)
5. 行业资金强度 (前3大个股 amount 占比) (15%)
6. 连续强于大盘天数 (15%)

行业阶段：启动 / 扩散 / 主升 / 高潮 / 分歧 / 退潮
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _map_linear(value: float, lo: float, mid: float, hi: float) -> float:
    if value <= lo:
        return 0.0
    if value >= hi:
        return 100.0
    if value <= mid:
        return 50.0 * (value - lo) / (mid - lo) if mid != lo else 50.0
    return 50.0 + 50.0 * (value - mid) / (hi - mid) if hi != mid else 50.0


def _sector_metrics(
    sector_today: pd.DataFrame,
    market_today: pd.DataFrame,
    sector_hist: pd.DataFrame,
    has_history: bool,
    mkt_daily: pd.Series | None,
) -> dict[str, float]:
    """计算单个行业的 6 个子指标。

    sector_hist（本行业的历史切片）与 mkt_daily（全市场逐日中位数）都由
    compute_sector_heat 预先算好传进来：原来每个行业把十几万行的历史重筛 4 次、
    全市场中位数重算 2 次，83 个行业就是几百次全表扫描（实测一天 9.7s，占满
    整个市场环境快照的耗时）。
    """
    metrics: dict[str, float] = {}

    today = sector_today
    if today.empty:
        return {k: 50.0 for k in [
            "relative_return", "volume_ratio", "breadth",
            "trend", "capital_intensity", "outperform_days",
        ]}

    # 1. 相对涨幅 (5d/20d)
    if not sector_hist.empty:
        sec_daily = sector_hist.groupby("date")["pct_chg"].median()
        sec_5d = sec_daily.tail(5).sum()
        sec_20d = sec_daily.tail(20).sum()
        mkt_5d = mkt_daily.tail(5).sum() if mkt_daily is not None else 0.0
        mkt_20d = mkt_daily.tail(20).sum() if mkt_daily is not None else 0.0
        rel = (sec_5d + sec_20d) / 2 - (mkt_5d + mkt_20d) / 2
        metrics["relative_return"] = _map_linear(rel, -5.0, 0.0, 5.0)
    elif has_history:
        metrics["relative_return"] = 50.0
    else:
        sec_median = today["pct_chg"].median()
        mkt_median = market_today["pct_chg"].median() if not market_today.empty else 0
        rel = sec_median - mkt_median
        metrics["relative_return"] = _map_linear(rel, -3.0, 0.0, 3.0)

    # 2. 量能变化
    if "amount" in today.columns and not sector_hist.empty and "amount" in sector_hist.columns:
        today_amt = today["amount"].sum()
        avg_amt = sector_hist.groupby("date")["amount"].sum().tail(20).mean()
        if avg_amt > 0:
            ratio = today_amt / avg_amt
            metrics["volume_ratio"] = _map_linear(ratio, 0.5, 1.0, 1.5)
        else:
            metrics["volume_ratio"] = 50.0
    else:
        metrics["volume_ratio"] = 50.0

    # 3. 上涨占比
    total = len(today)
    up = (today["pct_chg"] > 0).sum() if total > 0 else 0
    ratio = up / total if total > 0 else 0.5
    metrics["breadth"] = _map_linear(ratio, 0.3, 0.5, 0.7)

    # 4. 趋势强度 (近 5 日累计涨幅中位数)
    if not sector_hist.empty:
        daily_med = sector_hist.groupby("date")["pct_chg"].median()
        if len(daily_med) >= 5:
            cum5 = daily_med.tail(5).sum()
            metrics["trend"] = _map_linear(cum5, -3.0, 0.0, 3.0)
        else:
            metrics["trend"] = 50.0
    else:
        metrics["trend"] = _map_linear(today["pct_chg"].median(), -2.0, 0.0, 2.0)

    # 5. 资金强度 (前3大 amount 占比)
    if "amount" in today.columns and len(today) >= 3:
        sorted_amt = today["amount"].sort_values(ascending=False)
        top3_ratio = sorted_amt.head(3).sum() / sorted_amt.sum() if sorted_amt.sum() > 0 else 0
        metrics["capital_intensity"] = _map_linear(top3_ratio, 0.1, 0.35, 0.6)
    else:
        metrics["capital_intensity"] = 50.0

    # 6. 持续性 (近 20 日行业涨幅 > 大盘的天数占比)
    if not sector_hist.empty and mkt_daily is not None and not mkt_daily.empty:
        sec_daily = sector_hist.groupby("date")["pct_chg"].median()
        aligned = sec_daily.reindex(mkt_daily.index).dropna()
        mkt_aligned = mkt_daily.reindex(aligned.index)
        total_days = len(aligned)
        if total_days > 0:
            ratio = (aligned > mkt_aligned).sum() / total_days
            metrics["outperform_days"] = _map_linear(ratio, 0.3, 0.5, 0.7)
        else:
            metrics["outperform_days"] = 50.0
    else:
        metrics["outperform_days"] = 50.0

    return metrics


def _classify_stage(
    score: float,
    prev_scores: list[float] | None = None,
    breadth: float = 50.0,
    volume_ratio: float = 1.0,
) -> str:
    """基于热度分数轨迹 + 广度 + 量能判定行业阶段。"""
    prev = prev_scores or []
    trend = score - (prev[-1] if prev else score)

    if score < 40 and trend < -3:
        return "退潮"
    if score > 80 and volume_ratio < 0.9:
        return "高潮"
    if score > 70 and trend > 2:
        return "主升"
    if 50 <= score <= 70 and breadth > 60:
        return "扩散"
    if 40 <= score <= 55 and len(prev) >= 1 and prev[-1] < 40:
        return "启动"
    if score > 70 and trend < -3:
        return "分歧"
    if score >= 45:
        return "扩散"
    return "退潮"


def compute_sector_heat(
    summary_df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    sector_map: dict[str, str],
    prev_heat: dict[str, list[float]] | None,
    weights: dict[str, float],
) -> dict[str, dict]:
    """计算所有行业的热度分数和阶段。

    参数:
        summary_df: 当日全市场 (code, pct_chg, high, close, amount, ...)
        history_df: 近 20 日全市场 (code, date, pct_chg, amount, ...)
        sector_map: {code: sector_name}
        prev_heat: {sector_name: [最近几日的热度分数]}
        weights: 子指标权重

    返回:
        {sector_name: {"score": float, "stage": str, "detail": dict}}
    """
    if summary_df.empty or not sector_map:
        return {}

    sectors: dict[str, list[str]] = {}
    for code, sector in sector_map.items():
        sectors.setdefault(sector, []).append(code)

    market_df = summary_df
    has_history = history_df is not None and not history_df.empty
    # 全市场逐日中位数与行业无关，循环外只算一次
    mkt_daily = history_df.groupby("date")["pct_chg"].median() if has_history else None
    result: dict[str, dict] = {}

    for sector_name, codes in sectors.items():
        sector_today = summary_df[summary_df["code"].astype(str).isin(codes)]
        sector_hist = history_df[history_df["code"].isin(codes)] if has_history else pd.DataFrame()
        metrics = _sector_metrics(sector_today, market_df, sector_hist, has_history, mkt_daily)

        total_w = sum(weights.get(k, 0) for k in metrics)
        if total_w <= 0:
            total_w = 1.0
        score = sum(metrics[k] * weights.get(k, 0) for k in metrics) / total_w
        score = round(min(100, max(0, score)), 1)

        prev_scores = (prev_heat or {}).get(sector_name, [])
        breadth = metrics.get("breadth", 50.0)
        volume_ratio_raw = metrics.get("volume_ratio", 50.0)
        vr = 0.5 + (volume_ratio_raw - 50) / 100

        stage = _classify_stage(score, prev_scores, breadth, vr)

        result[sector_name] = {
            "score": score,
            "stage": stage,
            "detail": metrics,
        }

    return result
