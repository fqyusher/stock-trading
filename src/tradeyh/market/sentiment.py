"""MarketSentiment (0~100)：全市场情绪评估。

6 个子指标：
1. 上涨家数占比 (20%)
2. 涨跌幅中位数 (20%)
3. 涨停/跌停数量 (15%)
4. 炸板率 (15%)
5. 昨日涨停股今日平均涨幅 (15%)
6. 最高连板高度 (15%)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _map_linear(value: float, lo: float, mid: float, hi: float) -> float:
    """三段线性映射: lo→0, mid→50, hi→100，clamp 到 [0, 100]。"""
    if value <= lo:
        return 0.0
    if value >= hi:
        return 100.0
    if value <= mid:
        return 50.0 * (value - lo) / (mid - lo) if mid != lo else 50.0
    return 50.0 + 50.0 * (value - mid) / (hi - mid) if hi != mid else 50.0


def _advance_ratio(df: pd.DataFrame) -> float:
    total = len(df)
    if total == 0:
        return 50.0
    up = (df["pct_chg"] > 0).sum()
    ratio = up / total
    return _map_linear(ratio, 0.3, 0.5, 0.7)


def _median_return(df: pd.DataFrame) -> float:
    if df.empty:
        return 50.0
    med = float(df["pct_chg"].median())
    return _map_linear(med, -2.0, 0.0, 2.0)


def _limit_count(df: pd.DataFrame) -> float:
    if df.empty:
        return 50.0
    limit_up = (df["pct_chg"] >= 9.5).sum()
    limit_down = (df["pct_chg"] <= -9.5).sum()
    net = limit_up - limit_down
    return _map_linear(net, -30, 0, 30)


def _failed_limit_rate(df: pd.DataFrame) -> float:
    """炸板率：盘中触及涨停区但收盘未封住的比例。"""
    if df.empty or "high" not in df.columns:
        return 50.0
    prev_close = df["close"] / (1 + df["pct_chg"] / 100)
    limit_threshold = prev_close * 1.095
    touched = df["high"] >= limit_threshold
    sealed = df["pct_chg"] >= 9.5
    failed = touched & ~sealed
    total_touched = touched.sum()
    if total_touched == 0:
        return 50.0
    rate = failed.sum() / total_touched
    return _map_linear(rate, 0.0, 0.5, 1.0)


def _limit_follow_through(
    df: pd.DataFrame,
    prev_limit_codes: list[str],
) -> float:
    """昨日涨停股今日平均涨幅。"""
    if not prev_limit_codes or df.empty:
        return 50.0
    today = df[df["code"].astype(str).isin(prev_limit_codes)]
    if today.empty:
        return 50.0
    avg = float(today["pct_chg"].mean())
    return _map_linear(avg, -3.0, 0.0, 3.0)


def _max_streak(streaks: dict[str, int]) -> float:
    if not streaks:
        return 20.0
    max_s = max(streaks.values())
    if max_s <= 1:
        return 20.0
    if max_s <= 3:
        return 20 + 30 * (max_s - 1) / 2
    if max_s <= 5:
        return 50 + 20 * (max_s - 3) / 2
    return min(100.0, 70 + 30 * (max_s - 5) / 2)


def compute_market_sentiment(
    summary_df: pd.DataFrame,
    weights: dict[str, float],
    prev_limit_codes: list[str] | None = None,
    streaks: dict[str, int] | None = None,
) -> tuple[float, dict]:
    """计算市场情绪分数。

    参数:
        summary_df: 当日全市场聚合 (code, pct_chg, high, close, ...)
        weights: 子指标权重
        prev_limit_codes: 昨日涨停股票代码列表
        streaks: 当前连板数 {code: count}
    """
    sub: dict[str, float] = {
        "advance_ratio": _advance_ratio(summary_df),
        "median_return": _median_return(summary_df),
        "limit_count": _limit_count(summary_df),
        "failed_limit_rate": _failed_limit_rate(summary_df),
        "limit_follow_through": _limit_follow_through(
            summary_df, prev_limit_codes or []
        ),
        "max_streak": _max_streak(streaks or {}),
    }

    total_w = sum(weights.get(k, 0) for k in sub)
    if total_w <= 0:
        total_w = 1.0
    score = sum(sub[k] * weights.get(k, 0) for k in sub) / total_w
    return round(min(100, max(0, score)), 1), sub
