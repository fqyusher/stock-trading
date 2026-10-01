"""市场环境 / 情绪 / 行业热度模块。

提供 MarketSnapshot 数据类，在扫描前预计算一次，传入所有 worker 进程共享。
"""

from .snapshot import MarketSnapshot, compute_market_snapshot

__all__ = ["MarketSnapshot", "compute_market_snapshot"]
