"""MarketRegime (0~100)：基于主要指数的市场环境评估。

使用沪深300、中证1000、创业板指的 K 线数据，综合判断：
1. 指数相对 MA20/MA60 的位置
2. MA20/MA60 方向（斜率）
3. 20日/60日趋势幅度
4. 多指数趋势一致性
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ma(s: pd.Series, n: int) -> float:
    return float(s.rolling(n).mean().iloc[-1]) if len(s) >= n else float(s.iloc[-1])


def _slope(s: pd.Series, n: int) -> float:
    """最近 n 日的 MA 斜率（百分比/日）。"""
    ma = s.rolling(20).mean().dropna()
    if len(ma) < n + 1:
        return 0.0
    return (float(ma.iloc[-1]) - float(ma.iloc[-n - 1])) / float(ma.iloc[-n - 1]) / n


def _score_index(df: pd.DataFrame) -> tuple[float, str]:
    """单只指数评分 (0~100) 和趋势方向 (up/down/neutral)。"""
    close = df["close"]
    last = float(close.iloc[-1])

    ma20 = _ma(close, 20)
    ma60 = _ma(close, 60)

    # 1. 位置分 (40%)
    above_ma20 = last > ma20
    above_ma60 = last > ma60
    if above_ma20 and above_ma60:
        position = 100.0
    elif above_ma60:
        position = 60.0
    elif above_ma20:
        position = 40.0
    else:
        position = 0.0

    # 2. 斜率分 (30%)
    slope20 = _slope(close, 5)
    slope60 = _slope(close, 10)
    if slope20 > 0.002 and slope60 > 0.001:
        slope_score = 100.0
    elif slope20 > 0:
        slope_score = 65.0
    elif slope20 > -0.002:
        slope_score = 35.0
    else:
        slope_score = 0.0

    # 3. 趋势分 (30%)
    ret20 = (last / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0
    ret60 = (last / float(close.iloc[-60]) - 1) if len(close) >= 60 else 0
    trend_raw = ret20 * 0.6 + ret60 * 0.4
    trend_score = max(0, min(100, 50 + trend_raw * 500))

    score = position * 0.4 + slope_score * 0.3 + trend_score * 0.3

    if slope20 > 0.002:
        direction = "up"
    elif slope20 < -0.002:
        direction = "down"
    else:
        direction = "neutral"

    return score, direction


def _regime_label(score: float, thresholds: dict) -> str:
    if score >= thresholds.get("strong", 80):
        return "强势"
    if score >= thresholds.get("bullish", 65):
        return "偏强"
    if score >= thresholds.get("neutral", 45):
        return "震荡"
    if score >= thresholds.get("weak", 25):
        return "偏弱"
    return "弱势"


def compute_market_regime(
    index_dfs: dict[str, pd.DataFrame],
    weights: dict[str, float],
    thresholds: dict[str, int],
) -> tuple[float, str, dict]:
    """计算市场环境分数。

    参数:
        index_dfs: {"csi300": df, "csi1000": df, "chinext": df}
        weights: 各指数权重
        thresholds: 标签阈值

    返回:
        (score, label, detail)
    """
    if not index_dfs:
        return 50.0, "震荡", {}

    scores: dict[str, float] = {}
    directions: dict[str, str] = {}
    for name, df in index_dfs.items():
        if df is None or df.empty or len(df) < 20:
            continue
        s, d = _score_index(df)
        scores[name] = s
        directions[name] = d

    if not scores:
        return 50.0, "震荡", {}

    total_w = sum(weights.get(k, 0) for k in scores)
    if total_w <= 0:
        total_w = 1.0
    weighted = sum(scores[k] * weights.get(k, 0) for k in scores) / total_w

    consistency_bonus = 0.0
    if len(directions) >= 2:
        dirs = list(directions.values())
        if len(set(dirs)) == 1 and dirs[0] != "neutral":
            consistency_bonus = 5.0

    final = min(100.0, weighted + consistency_bonus)
    label = _regime_label(final, thresholds)

    detail = {
        "index_scores": scores,
        "directions": directions,
        "consistency_bonus": consistency_bonus,
        "weighted_raw": weighted,
    }
    return round(final, 1), label, detail
