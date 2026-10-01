"""技术指标：纯 pandas/numpy 向量化，无第三方 TA 依赖。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR。"""
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False).mean()


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (DIF, DEA, MACD柱)。柱按国内习惯取 2*(DIF-DEA)。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return dif, dea, 2 * (dif - dea)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Wilder ADX：趋势强度 0-100。>25 趋势明确，<20 震荡。"""
    tr = true_range(high, low, close)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm >= minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(20.0)


def boll(
    close: pd.Series, period: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std()
    return mid + k * std, mid, mid - k * std


def compute_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """给K线附加检测器要用到的全部指标列，返回新 DataFrame。"""
    out = df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    # 120 只给 v2 破底翻的 TP2 候选池用（MA120 是常见的长期压力位）
    for period in (5, 10, 20, 60, 120):
        out[f"ma{period}"] = sma(close, period)
    out["vol_ma5"] = sma(volume, 5)
    out["vol_ma20"] = sma(volume, 20)
    out["atr"] = atr(high, low, close, cfg.swings.atr_period)
    out["dif"], out["dea"], out["macd_hist"] = macd(close)
    out["rsi"] = rsi(close)
    out["adx"] = adx(high, low, close)
    out["vol_ratio"] = volume / out["vol_ma20"].replace(0, np.nan)
    out["vol_ratio"] = out["vol_ratio"].fillna(1.0)
    out["vol_ratio_ma5"] = volume / out["vol_ma5"].replace(0, np.nan)
    out["vol_ratio_ma5"] = out["vol_ratio_ma5"].fillna(1.0)
    return out
