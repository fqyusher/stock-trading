"""形态检测器注册表。

新增形态只需写一个暴露 `detect(ctx) -> list[PatternHit]` 的模块，然后加进 DETECTORS。

## 破底翻 v2 已整体停用（2026-09-30）

`break_bottom_v2` / `break_bottom_v2_b2` **不在下面的注册表里**，模块文件保留但不再被任何
链路加载：scan 不会跑、monitor 不会命中、plot 不会出图。这是"停用"的总闸——三个消费方
（`scan.resolve_detectors`、`watch` 的形态展开、`report`/`plot` 的展示名）都从这三张表派生，
摘掉注册就没有路径能产出 v2。

**要重新启用，必须同时做两件事，缺一不可：**
1. 在下面三张表里重新登记这两个模块（把 `DETECTORS` 上方注释里的那几行加上；
   `PATTERN_KEYS` 加 `"break_bottom_v2": [break_bottom_v2.PATTERN]`、
   `PATTERN_CN` 加 `break_bottom_v2.PATTERN: break_bottom_v2.PATTERN_CN`，v2_b2 同理）；
2. 把 `config.yaml` 里 `break_bottom_v2.enabled` / `break_bottom_v2_b2.enabled` 改回 `true`，
   并把它们加回 `watch.patterns`。

只翻 config 开关不会有任何效果（检测器根本没被加载）；只登记表而不加回 `watch.patterns`，
观察池仍然收不到它们。`base.py` 的 `PATTERN_DIRECTION` / `TARGET_KEYS` 里 v2 的条目**特意保留**，
重启用时不用再补。
"""

from __future__ import annotations

from . import (
    break_bottom,
    break_bottom_b2,
    breakout,
    channel,
    m_top,
    trend_pullback,
    w_bottom,
)
from .base import (
    STATUS_CN,
    STATUS_FORMED,
    STATUS_FORMING,
    STATUS_INVALID,
    CONFLICT_GROUPS,
    INVALIDATION_RULES,
    PatternHit,
    ScanContext,
)

# 重新启用 v2 时把这两行加回 import 与下面三张表：
#   from . import break_bottom_v2, break_bottom_v2_b2
#   DETECTORS |= {"break_bottom_v2": break_bottom_v2,
#                 "break_bottom_v2_b2": break_bottom_v2_b2}

# key → 模块。CLI --patterns 用这里的 key 过滤。
DETECTORS = {
    "break_bottom": break_bottom,
    "break_bottom_b2": break_bottom_b2,
    "w_bottom": w_bottom,
    "m_top": m_top,
    "channel": channel,
    "breakout": breakout,
    "trend_pullback": trend_pullback,
}

# 检测器 key → 它可能产出的 pattern 值
PATTERN_KEYS = {
    # Keep the historical name accepted for watch/config files.
    "break_bottom": [break_bottom.PATTERN, "breakdown_reversal"],
    "break_bottom_b2": [break_bottom_b2.PATTERN],
    "w_bottom": [w_bottom.PATTERN],
    "m_top": [m_top.PATTERN],
    "channel": [channel.RISING, channel.FALLING, channel.BOX, channel.FALLING_REVERSAL],
    "breakout": list(breakout.PATTERNS),
    "trend_pullback": [trend_pullback.PATTERN],
}

PATTERN_CN = {
    break_bottom.PATTERN: break_bottom.PATTERN_CN,
    "breakdown_reversal": break_bottom.PATTERN_CN,
    break_bottom_b2.PATTERN: break_bottom_b2.PATTERN_CN,
    w_bottom.PATTERN: w_bottom.PATTERN_CN,
    m_top.PATTERN: m_top.PATTERN_CN,
    **channel.PATTERN_CN,
    **breakout.PATTERN_CN,
    trend_pullback.PATTERN: trend_pullback.PATTERN_CN,
}

__all__ = [
    "DETECTORS",
    "PATTERN_KEYS",
    "PATTERN_CN",
    "CONFLICT_GROUPS",
    "INVALIDATION_RULES",
    "PatternHit",
    "ScanContext",
    "STATUS_CN",
    "STATUS_FORMED",
    "STATUS_FORMING",
    "STATUS_INVALID",
]
