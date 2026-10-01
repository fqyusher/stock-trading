"""股票池构建与过滤。"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def filter_spot(spot: pd.DataFrame, cfg) -> pd.DataFrame:
    """对全市场快照做静态过滤（不涉及K线长度，那个在扫描阶段过滤）。"""
    u = cfg.universe
    df = spot.copy()
    total = len(df)
    reasons: list[str] = []

    if u.exclude_st:
        mask = ~df["name"].str.upper().str.contains("ST", na=False)
        reasons.append(f"ST {(~mask).sum()}")
        df = df[mask]

    if u.exclude_delisted:
        mask = ~df["name"].str.contains("退", na=False)
        reasons.append(f"退市 {(~mask).sum()}")
        df = df[mask]

    if u.exclude_star_market:
        mask = ~df["code"].str.startswith("688")
        reasons.append(f"科创板 {(~mask).sum()}")
        df = df[mask]

    if u.exclude_bj:
        mask = ~df["code"].str.startswith(("4", "8", "9"))
        reasons.append(f"北交所 {(~mask).sum()}")
        df = df[mask]

    # 最新价为 NaN 视为停牌
    mask = df["price"].notna() & (df["price"] > 0)
    reasons.append(f"停牌 {(~mask).sum()}")
    df = df[mask]

    if u.min_float_mv:
        mask = df["float_mv"].fillna(0) >= u.min_float_mv
        reasons.append(f"流通市值不足 {(~mask).sum()}")
        df = df[mask]

    if u.min_amount:
        mask = df["amount"].fillna(0) >= u.min_amount
        reasons.append(f"成交额不足 {(~mask).sum()}")
        df = df[mask]

    if u.max_price:
        mask = df["price"].fillna(0) <= u.max_price
        reasons.append(f"股价>{u.max_price} {(~mask).sum()}")
        df = df[mask]

    log.info("股票池: %d → %d (剔除: %s)", total, len(df), ", ".join(reasons))
    return df.reset_index(drop=True)
