"""摆动点（ZigZag）识别与趋势背景判定 —— 所有形态识别的地基。

阈值随波动率自适应：`max(pct_min, atr_k * ATR / close)`。
固定百分比阈值在低波动股上会把噪声拟合成摆动点，在高波动股上又会漏掉真形态。

末端的极值点会作为**未确认** pivot 附在序列尾部（`confirmed=False`）——
"构筑中"的形态（比如右底刚出现还没反弹够幅度）全靠这个点才能被看到。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HIGH = "H"
LOW = "L"


@dataclass(frozen=True)
class Pivot:
    idx: int
    date: pd.Timestamp
    price: float
    kind: str  # 'H' 峰 / 'L' 谷
    confirmed: bool = True

    @property
    def is_high(self) -> bool:
        return self.kind == HIGH

    @property
    def is_low(self) -> bool:
        return self.kind == LOW


def reversal_thresholds(df: pd.DataFrame, cfg) -> np.ndarray:
    """每根K线的反转阈值（小数比例）。"""
    close = df["close"].to_numpy(dtype=float)
    if "atr" in df.columns:
        atr_ratio = df["atr"].to_numpy(dtype=float) / np.where(close == 0, np.nan, close)
    else:
        atr_ratio = np.full(len(df), np.nan)
    atr_part = cfg.swings.atr_k * atr_ratio
    return np.where(
        np.isnan(atr_part), cfg.swings.pct_min, np.maximum(cfg.swings.pct_min, atr_part)
    )


def zigzag(df: pd.DataFrame, cfg) -> list[Pivot]:
    """返回严格高低交替的摆动点序列，最后一个可能未确认。"""
    n = len(df)
    if n < 3:
        return []

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    dates = df.index
    thr = reversal_thresholds(df, cfg)

    pivots: list[Pivot] = []
    direction: str | None = None  # 'up' 正在走升段 / 'down' 正在走跌段
    hi_idx, hi_price = 0, high[0]
    lo_idx, lo_price = 0, low[0]

    for i in range(1, n):
        t = thr[i]
        if direction is None:
            if high[i] > hi_price:
                hi_idx, hi_price = i, high[i]
            if low[i] < lo_price:
                lo_idx, lo_price = i, low[i]
            if hi_price > 0 and (hi_price - low[i]) / hi_price >= t:
                pivots.append(Pivot(hi_idx, dates[hi_idx], hi_price, HIGH))
                direction = "down"
                lo_idx, lo_price = i, low[i]
            elif lo_price > 0 and (high[i] - lo_price) / lo_price >= t:
                pivots.append(Pivot(lo_idx, dates[lo_idx], lo_price, LOW))
                direction = "up"
                hi_idx, hi_price = i, high[i]
            continue

        if direction == "up":
            if high[i] > hi_price:
                hi_idx, hi_price = i, high[i]
            elif hi_price > 0 and (hi_price - low[i]) / hi_price >= t:
                pivots.append(Pivot(hi_idx, dates[hi_idx], hi_price, HIGH))
                direction = "down"
                lo_idx, lo_price = i, low[i]
        else:  # direction == 'down'
            if low[i] < lo_price:
                lo_idx, lo_price = i, low[i]
            elif lo_price > 0 and (high[i] - lo_price) / lo_price >= t:
                pivots.append(Pivot(lo_idx, dates[lo_idx], lo_price, LOW))
                direction = "up"
                hi_idx, hi_price = i, high[i]

    # 末端未确认极值：形态"构筑中"的判定依赖它
    if direction == "up":
        tail = Pivot(hi_idx, dates[hi_idx], hi_price, HIGH, confirmed=False)
    elif direction == "down":
        tail = Pivot(lo_idx, dates[lo_idx], lo_price, LOW, confirmed=False)
    else:
        tail = None
    if tail is not None and (not pivots or tail.idx > pivots[-1].idx):
        pivots.append(tail)

    return pivots


def fractal_pivots(df: pd.DataFrame, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """分形局部极值下标 (高点下标数组, 低点下标数组)。用于颈线/轨道触碰计数。"""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    highs, lows = [], []
    for i in range(window, n - window):
        seg_h = high[i - window : i + window + 1]
        seg_l = low[i - window : i + window + 1]
        if high[i] >= seg_h.max():
            highs.append(i)
        if low[i] <= seg_l.min():
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


def prior_drop_pct(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """idx 之前 lookback 根内的最高价 → idx 处最低价的跌幅。W底要求前期有下跌。"""
    start = max(0, idx - lookback)
    if idx <= start:
        return 0.0
    peak = float(df["high"].iloc[start:idx].max())
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - float(df["low"].iloc[idx])) / peak)


def prior_rise_pct(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """idx 之前 lookback 根内的最低价 → idx 处最高价的涨幅。M头要求前期有上涨。"""
    start = max(0, idx - lookback)
    if idx <= start:
        return 0.0
    trough = float(df["low"].iloc[start:idx].min())
    if trough <= 0:
        return 0.0
    return max(0.0, (float(df["high"].iloc[idx]) - trough) / trough)


def count_level_touches(
    df: pd.DataFrame, level: float, start: int, end: int, tol: float, use: str = "high"
) -> int:
    """[start, end) 区间内价格触碰某水平价位的次数（相邻触碰合并为一次）。"""
    if level <= 0 or end <= start:
        return 0
    values = df[use].to_numpy(dtype=float)[start:end]
    near = np.abs(values - level) / level <= tol
    # 连续为 True 的段算一次触碰
    return int(np.sum(near & ~np.concatenate(([False], near[:-1]))))
