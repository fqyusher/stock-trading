"""形态检测器的公共结构与评分工具。

所有检测器签名一致：`detect(ctx) -> list[PatternHit]`，
方便 patterns/__init__.py 注册后统一调度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..swings import Pivot

STATUS_FORMED = "formed"
STATUS_FORMING = "forming"
# 失效态：结构已被证伪（破底后没收复、收复后再次有效跌破）。只有 v2 口径会输出，
# 目的是让"刚刚失效"这件事在报表上可见；旧的 formed/forming 二态没有这一档。
STATUS_INVALID = "invalid"
STATUS_CN = {STATUS_FORMED: "已形成", STATUS_FORMING: "构筑中", STATUS_INVALID: "已失效"}

# 交易方向。图形识别与交易方向解耦：下降通道/M头是有效图形，但不是做多机会。
# 做多榜单与观察池只取 LONG，做空形态仍然进报表（规避信息）。
DIR_LONG = "long"
DIR_SHORT = "short"
DIR_NEUTRAL = "neutral"
DIRECTION_CN = {DIR_LONG: "做多", DIR_SHORT: "做空", DIR_NEUTRAL: "中性"}

# 形态 → 交易方向。和 TARGET_KEYS 一样表驱动：新形态漏登记会被测试抓住，
# 而不是静默默认成 DIR_LONG，把做空形态混进做多榜单和观察池。
PATTERN_DIRECTION = {
    "w_bottom": DIR_LONG,
    "break_bottom": DIR_LONG,
    "breakdown_reversal": DIR_LONG,
    "break_bottom_b2": DIR_LONG,
    # v2 口径（ATR 严格版）与旧口径并行，同样是做多机会
    "break_bottom_v2": DIR_LONG,
    "break_bottom_v2_b2": DIR_LONG,
    "trend_pullback": DIR_LONG,
    "platform_breakout": DIR_LONG,
    "breakout_retest": DIR_LONG,
    "cup_handle": DIR_LONG,
    # 下降通道里出现止跌信号是反转做多机会，与单纯的下降通道方向相反
    "falling_channel_reversal": DIR_LONG,
    "rising_channel": DIR_LONG,
    # 箱体目标价取上轨，按"下沿买入/上沿卖出"的区间做多处理
    "box_range": DIR_LONG,
    "m_top": DIR_SHORT,
    "falling_channel": DIR_SHORT,
}

# 每个形态的目标价/止损价取自 key_levels 的哪个键。
# 以前这段是散在 _get_target_price 里的 if 链，漏一个形态就静默退化成"无目标价"
# → 收益质量恒为 0 → 该形态永远排不进榜。表驱动后缺键会在测试里暴露。
TARGET_KEYS = {
    "w_bottom": "等幅目标",
    "m_top": "等幅目标",
    "platform_breakout": "等幅目标",
    "breakout_retest": "等幅目标",
    "cup_handle": "等幅目标",
    "falling_channel_reversal": "等幅目标",
    "break_bottom_b2": "等幅目标",
    # 回踩的第一目标是前高，不是"前高再叠一个满幅"——后者会把预期收益
    # 放大一倍多，让该形态在收益质量上恒定顶满 100 分。
    "trend_pullback": "近期高点",
    "break_bottom": "上方压力",
    "breakdown_reversal": "上方压力",
    # v2 口径直接写 tp1_price/tp2_price 两个候选目标，主目标取第一目标（H1）。
    "break_bottom_v2": "tp1_price",
    "break_bottom_v2_b2": "tp1_price",
    "rising_channel": "上轨",
    "box_range": "上轨",
    "falling_channel": "下轨",
}

STOP_KEY = "参考止损"

# 形态优先级 → 评分权重。反映信号可靠性，不是命中数量。
# 形态越少见、结构约束越硬的给越高权重；靠均线关系就能成立的给低权重。
PRIORITY_WEIGHT = {"S": 1.0, "A": 0.95, "B": 0.88, "C": 0.80}

# 互斥形态组：同组内新旧口径检测同一波行情，同一只股票只保留评分最高的一个。
# 不在任何组里的形态互相独立，不受去重影响。
CONFLICT_GROUPS: list[set[str]] = [
    {"break_bottom", "break_bottom_v2", "breakdown_reversal"},
    {"break_bottom_b2", "break_bottom_v2_b2"},
]
_CONFLICT_MAP: dict[str, int] = {
    pat: idx for idx, group in enumerate(CONFLICT_GROUPS) for pat in group
}

# 统一失效判定规则。key_levels 里的键名 → 失效条件。
# 做多形态：close < 关键支撑 → 失效；做空形态：close > 关键压力 → 失效。
# v2 口径自己管理失效（emit_invalid），这里不重复判定。
INVALIDATION_RULES: dict[str, dict[str, str]] = {
    "w_bottom": {"key": "左底", "side": "long"},
    "break_bottom": {"key": "参考止损", "side": "long"},
    "platform_breakout": {"key": "整理下沿", "side": "long"},
    "breakout_retest": {"key": "整理下沿", "side": "long"},
    "cup_handle": {"key": "整理下沿", "side": "long"},
    "trend_pullback": {"key": "回踩支撑", "side": "long"},
    "m_top": {"key": "参考止损", "side": "short"},
    "falling_channel": {"key": "下轨", "side": "short"},
}

# 风险收益比上限。close 贴着止损时 risk→0 会让 RR 冲到几十上百，
# 直接污染排序（实测 trend_pullback 的 RR 均值曾达 57.7）。
MAX_RR = 6.0
# risk 至少按现价的这个比例计，防止 risk→0 把 RR 放飞
MIN_RISK_PCT = 0.005


@dataclass
class ScanContext:
    """单只股票的扫描上下文，由 scan.py 组装一次给所有检测器复用。

    explain=True 时检测器会把每一步的拒绝原因写进 trace，
    这是调参时唯一能看清"为什么没命中"的途径。
    """

    code: str
    name: str
    df: pd.DataFrame  # 已附加指标列
    pivots: list[Pivot]
    cfg: Any
    explain: bool = False
    trace: list[str] = field(default_factory=list)
    market_snapshot: Any = None  # MarketSnapshot，扫描前预计算

    @property
    def n(self) -> int:
        return len(self.df)

    def note(self, msg: str) -> None:
        if self.explain:
            self.trace.append(msg)

    @property
    def trend_direction(self) -> str:
        """up / down / neutral — 基于 MA20 斜率 + MA20 vs MA60。"""
        try:
            ma20 = float(self.df.ma20.iloc[-1])
            ma60 = float(self.df.ma60.iloc[-1])
            ma20_5 = float(self.df.ma20.iloc[-6]) if len(self.df) > 5 else ma20
        except (IndexError, TypeError, ValueError):
            return "neutral"
        if not (np.isfinite(ma20) and np.isfinite(ma60) and ma20 > 0 and ma60 > 0):
            return "neutral"
        if not np.isfinite(ma20_5) or ma20_5 <= 0:
            ma20_5 = ma20
        slope = (ma20 - ma20_5) / ma20_5
        if ma20 > ma60 and slope > 0.005:
            return "up"
        if ma20 < ma60 and slope < -0.005:
            return "down"
        return "neutral"

    @property
    def market_regime(self) -> str:
        """trending / ranging / neutral — 基于 ADX。"""
        try:
            adx_val = float(self.df["adx"].iloc[-1])
        except (KeyError, IndexError, TypeError, ValueError):
            return "neutral"
        if not np.isfinite(adx_val):
            return "neutral"
        if adx_val >= 25:
            return "trending"
        if adx_val <= 20:
            return "ranging"
        return "neutral"


@dataclass
class PatternHit:
    code: str
    name: str
    pattern: str
    pattern_cn: str
    status: str
    score: float
    geom_score: float
    vol_score: float
    trend_score: float
    signal_date: pd.Timestamp
    key_levels: dict[str, float] = field(default_factory=dict)
    detail: str = ""
    pivots: list[Pivot] = field(default_factory=list)
    close_price: float = 0.0
    # 风险修正所需数据
    rsi: float = 50.0
    macd_hist: float = 0.0
    # 交易方向。新字段追加在末尾，保证既有位置参数调用不受影响。
    # 留空表示"按形态查 PATTERN_DIRECTION"，检测器一般不需要显式传。
    direction: str = ""
    # 构筑中的阶段折扣。由 scan.scan_context() 统一从 cfg.scoring 盖到 hit 上，
    # 而不是评分时读全局——多进程扫描下 worker 之间没有共享状态。
    forming_discount: float = 0.85
    # 阶段机标签（BROKEN/RECLAIMED/B1_CONFIRMED…）。旧口径没有阶段机，留空即可；
    # 与 status 是两个维度：status 只回答"已形成/构筑中/已失效"，stage 说明走到哪一步。
    pattern_stage: str = ""
    # 结构是否已失效。只由显式输出失效行的检测器（v2）置位，默认 False。
    # 报表/观察池靠这一列把失效行挡在做多榜单之外——只看形态 key 筛方向会放过它们。
    pattern_invalid: bool = False
    invalid_reason: str = ""
    # 关键点日期（L0/L1/B1/H1/B2…）。key_levels 只能是 float，日期单开一列，
    # to_row() 展开成 <key>_date，避免和价位混在一个列里。
    key_level_dates: dict[str, pd.Timestamp] = field(default_factory=dict)
    # 市场环境标签，由 scan_context() 从 ScanContext.market_regime 盖上。
    market_regime: str = "neutral"
    # 时间衰减因子（0.1-1.0），由 scan_context() 按信号天数计算。
    # formed 状态才衰减，forming/invalid 保持 1.0。
    time_decay: float = 1.0
    # 市场环境快照，由 scan_context() 从 ScanContext.market_snapshot 盖上。
    # None 时 composite_score 不做环境修正（向后兼容）。
    market_snapshot: Any = None
    sector_name: str = ""
    sector_stage: str = ""

    def __post_init__(self) -> None:
        if not self.direction:
            self.direction = PATTERN_DIRECTION.get(self.pattern, DIR_LONG)

    @property
    def status_cn(self) -> str:
        return STATUS_CN.get(self.status, self.status)

    @property
    def direction_cn(self) -> str:
        return DIRECTION_CN.get(self.direction, self.direction)

    @property
    def is_long(self) -> bool:
        return self.direction == DIR_LONG

    def expected_profit(self) -> float:
        """预期收益率（%）。做空形态按跌幅计，正数=形态兑现。"""
        if not self.close_price or self.close_price <= 0:
            return 0.0

        target = self._get_target_price()
        if target is None:
            return 0.0

        if self.direction == DIR_SHORT:
            return (self.close_price - target) / self.close_price * 100
        return (target - self.close_price) / self.close_price * 100

    def _get_target_price(self) -> float | None:
        """目标价位。表驱动，缺键返回 None。

        支持回退链：break_bottom 优先取"前波高点"（图2标注的目标），
        没有则回退到"上方压力"（等幅量度）。
        """
        key = TARGET_KEYS.get(self.pattern)
        if key is None:
            return None
        # 回退链：break_bottom 的"前波高点"优先于"上方压力"
        fallback_keys = {"break_bottom": ["前波高点", "上方压力"]}.get(self.pattern, [key])
        for k in fallback_keys:
            value = self.key_levels.get(k)
            if value is not None and isinstance(value, (int, float)) and value > 0:
                return float(value)
        return None

    def _get_stop_loss(self) -> float | None:
        """止损价位。

        先取传统的 `参考止损`，没有就回退到 `stop_price`（v2 口径用的键）。
        新口径不写 `参考止损`，避免报表里出现两列同值的止损。
        """
        for key in (STOP_KEY, "stop_price"):
            value = self.key_levels.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return None

    def stop_breached(self) -> bool:
        """参考止损是否已经落到现价的错误一侧。

        做多的止损在现价下方、做空的止损在现价上方才是正常的风险结构。一旦
        止损已经在现价另一侧，这笔交易没有可度量的风险——以前用 `abs()` 取距离，
        会把"止损在现价上方"当成一个极小的正风险，RR 被兜到 MAX_RR、收益质量
        拿满分，反而让结构已破的形态排到榜单前列。
        """
        stop = self._get_stop_loss()
        if not stop or not self.close_price:
            return False
        if self.direction == DIR_SHORT:
            return stop <= self.close_price
        return stop >= self.close_price

    def risk_reward_ratio(self) -> float:
        """风险收益比（收益/风险），越大越好。裁剪到 MAX_RR 以内。"""
        target = self._get_target_price()
        stop = self._get_stop_loss()
        if not target or not stop or not self.close_price:
            return 0.0
        if self.stop_breached():
            return 0.0
        reward = abs(target - self.close_price)
        # risk 下限兜底：现价贴着止损时真实 risk 趋于 0，不兜住 RR 会冲到几十
        risk = max(abs(self.close_price - stop), self.close_price * MIN_RISK_PCT)
        if risk <= 0:
            return 0.0
        return min(MAX_RR, reward / risk)

    def opportunity_quality(self) -> float:
        """机会质量（0-100）：形态本身的质量评分（不含阶段折扣）。"""
        return self.score

    def return_quality(self) -> float:
        """收益质量（0-100）：预期收益 × 风险收益比。

        止损已被击穿时直接记 0：风险结构不成立，再看目标空间没有意义
        （否则光靠预期收益那 60% 权重仍能拿到不低的分）。
        """
        if self.stop_breached():
            return 0.0
        profit = self.expected_profit()
        rr = self.risk_reward_ratio()
        # 收益分：预期收益映射到 0-100（15% 收益=100分）。
        # 以前是 10% 顶满，导致大量形态挤在 100 分上失去区分度。
        profit_score = clamp01(profit / 15.0) * 100
        # 风险收益比分：RR=3 为 100 分（MAX_RR=6 时仍有上半区间做区分）
        rr_score = clamp01(rr / 3.0) * 100
        # 综合：收益占60%，风险收益比占40%
        return profit_score * 0.6 + rr_score * 0.4

    def stage_certainty(self) -> float:
        """阶段确定性（0-100）：已形成=100×时间衰减，构筑中打折，已失效=0。

        这里是 forming 唯一被打折的地方。以前 blend() 里也乘一次
        forming_discount，构筑中被扣两遍分。
        """
        if self.status == STATUS_FORMED:
            return 100.0 * self.time_decay
        if self.status == STATUS_INVALID:
            return 0.0
        return 100.0 * clamp01(self.forming_discount)

    def risk_adjustment(self) -> float:
        """风险修正（0-100）：基于技术指标的风险评估。高分 = 低风险环境。

        按"实际可得权重"归一化：拿不到的分项（比如检测器没提供压力距离）
        不参与分母，否则该分项永远缺失就等于给所有形态封了个分数上限。
        """
        bullish = self.direction != DIR_SHORT
        earned = 0.0
        total = 0.0

        # RSI 评估（权重 30）
        total += 30.0
        if bullish:
            if 40 <= self.rsi <= 60:
                earned += 30
            elif 30 <= self.rsi < 40 or 60 < self.rsi <= 70:
                earned += 15
            elif self.rsi < 30:
                earned += 20  # 超卖可能反弹
            # >70 超买，不加分
        else:
            if 40 <= self.rsi <= 60:
                earned += 30
            elif self.rsi > 70:
                earned += 20  # 超买可能回落
            elif self.rsi > 60:
                earned += 15

        # MACD 动量（权重 30）
        total += 30.0
        if bullish:
            if self.macd_hist > 0:
                earned += 30
            elif self.macd_hist > -0.1:
                earned += 15
        else:
            if self.macd_hist < 0:
                earned += 30
            elif self.macd_hist < 0.1:
                earned += 15

        # 距目标的空间（权重 20）——用预期收益代替原先那个
        # 从未被任何检测器写入的 "压力距离%" 死键
        profit = self.expected_profit()
        if profit:
            total += 20.0
            if profit > 10:
                earned += 20
            elif profit > 5:
                earned += 10

        if total <= 0:
            return 50.0
        # 0~total 映射到 20~100：全拿不到分也不该等于"零风险信息"
        return 20.0 + (earned / total) * 80.0

    def regime_multiplier(self) -> float:
        """市场环境调节系数。趋势市提升趋势类形态评分，震荡市提升反转类。"""
        if self.market_regime == "neutral":
            return 1.0
        boost = 0.08
        penalty = 0.95
        trend_patterns = {
            "trend_pullback", "platform_breakout", "breakout_retest",
            "rising_channel", "break_bottom_b2", "break_bottom_v2_b2",
        }
        reversal_patterns = {
            "w_bottom", "box_range", "falling_channel_reversal",
            "break_bottom", "break_bottom_v2", "breakdown_reversal",
        }
        if self.market_regime == "trending":
            if self.pattern in trend_patterns:
                return 1.0 + boost
            if self.pattern in reversal_patterns:
                return penalty
        elif self.market_regime == "ranging":
            if self.pattern in reversal_patterns:
                return 1.0 + boost
            if self.pattern in trend_patterns:
                return penalty
        return 1.0

    def composite_score(self) -> float:
        """综合评分（0-100）：多维度加权。

        权重分配：
        - 机会质量 40%：形态本身的质量
        - 收益质量 25%：预期收益和风险收益比
        - 阶段确定性 20%：已形成 vs 构筑中
        - 风险修正 15%：技术指标风险
        最后乘以市场环境调节系数。
        """
        raw = (
            self.opportunity_quality() * 0.40
            + self.return_quality() * 0.25
            + self.stage_certainty() * 0.20
            + self.risk_adjustment() * 0.15
        )
        raw *= self.regime_multiplier()
        if self.market_snapshot is not None:
            snap = self.market_snapshot
            raw *= snap.market_factor
            raw *= snap.sentiment_factor
            raw *= snap.sector_factor(self.code)
            raw *= snap.sector_bonus(self.code)
        return raw

    def to_row(self) -> dict[str, Any]:
        row = {
            "代码": self.code,
            "名称": self.name,
            "形态": self.pattern_cn,
            "形态key": self.pattern,
            "方向": self.direction_cn,
            "状态": self.status_cn,
            "最新价": round(self.close_price, 2) if self.close_price else "",
            "综合评分": round(self.composite_score(), 1),
            "机会质量": round(self.opportunity_quality(), 1),
            "收益质量": round(self.return_quality(), 1),
            "阶段确定性": round(self.stage_certainty(), 1),
            "风险修正": round(self.risk_adjustment(), 1),
            "预期收益%": round(self.expected_profit(), 1),
            "风险收益比": round(self.risk_reward_ratio(), 2),
            "信号日期": pd.Timestamp(self.signal_date).strftime("%Y-%m-%d"),
            "阶段": self.pattern_stage,
            "失效": "是" if self.pattern_invalid else "",
            "失效原因": self.invalid_reason,
            "说明": self.detail,
        }
        if self.market_snapshot is not None:
            snap = self.market_snapshot
            row["市场环境"] = snap.regime_label
            row["市场情绪"] = snap.sentiment_score
            row["所属板块"] = self.sector_name
            row["板块热度"] = round(
                snap.sector_scores.get(self.sector_name, 0), 1
            ) if self.sector_name else ""
            row["板块阶段"] = self.sector_stage
        for key, value in self.key_levels.items():
            row[key] = round(value, 3) if isinstance(value, (int, float)) else value
        for key, value in self.key_level_dates.items():
            row[f"{key}_date"] = (
                pd.Timestamp(value).strftime("%Y-%m-%d") if value is not None else ""
            )
        return row


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else float(x))


def taper(value: float, good: float, bad: float) -> float:
    """把一个连续量线性映射到 0~1：达到 good 得 1，劣于 bad 得 0。

    good 可以大于或小于 bad，方向自动处理。用来避免"刚好过阈值"和
    "远超阈值"拿一样的分。
    """
    if good == bad:
        return 1.0 if value == good else 0.0
    return clamp01((value - bad) / (good - bad))


def blend(
    geom: float, vol: float, trend: float, status: str, cfg, priority: str = "S"
) -> tuple[float, float, float, float]:
    """子项 0~1 → (形态质量分, 几何分, 量能分, 趋势分)，后三项各为 0~100。

    priority 为形态优先级（S/A/B/C），影响最终评分权重。

    注意这里**不再**乘 forming_discount：构筑中的折扣统一由
    PatternHit.stage_certainty() 承担。以前两处都打折，构筑中被扣两遍分，
    而 stage_certainty 又把 score 再叠进去，等于让几何质量在综合评分里
    被重复计了三次。
    """
    s = cfg.scoring
    total_weight = s.geom_weight + s.vol_weight + s.trend_weight
    raw = (
        clamp01(geom) * s.geom_weight
        + clamp01(vol) * s.vol_weight
        + clamp01(trend) * s.trend_weight
    ) / total_weight * 100
    raw *= PRIORITY_WEIGHT.get(priority, 1.0)
    return raw, clamp01(geom) * 100, clamp01(vol) * 100, clamp01(trend) * 100
