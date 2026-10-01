"""MarketSnapshot：市场环境 / 情绪 / 行业热度的快照数据类。

在扫描前由主进程预计算一次，pickle 后传入 worker 进程。
所有字段均为基本类型 (float/str/dict)，保证可序列化。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .. import store as store_mod
from .regime import compute_market_regime
from .sector import compute_sector_heat
from .sector_fetch import code_to_sector_map
from .sentiment import compute_market_sentiment

log = logging.getLogger(__name__)

# 中性 50 分 → 系数恰好 1.0（恒等变换），0 分 → lo，100 分 → hi，两段线性。
# 中性必须是 1.0 否则综合评分整体缩水，min_score / Excel 色阶锚点全部失准。
DEFAULT_BANDS = {
    "market": (0.85, 1.15),
    "sentiment": (0.95, 1.05),
    "sector": (0.90, 1.10),
}


def _factor(score: float, lo: float, hi: float, neutral: float = 50.0) -> float:
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = neutral
    s = min(100.0, max(0.0, s))
    if s <= neutral:
        return lo + (1.0 - lo) * (s / neutral)
    return 1.0 + (hi - 1.0) * ((s - neutral) / (100.0 - neutral))


def _bands(cfg_market, key: str) -> tuple[float, float]:
    """从 cfg.market.factor_bands 取带宽，缺失时回落到默认值。"""
    default = DEFAULT_BANDS[key]
    if cfg_market is None:
        return default
    bands = getattr(cfg_market, "factor_bands", None)
    if not bands:
        return default
    band = bands.get(key) if hasattr(bands, "get") else None
    if not band:
        return default
    lo = float(band.get("lo", default[0]))
    hi = float(band.get("hi", default[1]))
    return lo, hi


@dataclass
class MarketSnapshot:
    date: str
    regime_score: float = 50.0
    regime_label: str = "震荡"
    sentiment_score: float = 50.0
    sector_scores: dict[str, float] = field(default_factory=dict)
    sector_stages: dict[str, str] = field(default_factory=dict)
    code_to_sector: dict[str, str] = field(default_factory=dict)
    market_band: tuple[float, float] = DEFAULT_BANDS["market"]
    sentiment_band: tuple[float, float] = DEFAULT_BANDS["sentiment"]
    sector_band: tuple[float, float] = DEFAULT_BANDS["sector"]

    @property
    def market_factor(self) -> float:
        return _factor(self.regime_score, *self.market_band)

    @property
    def sentiment_factor(self) -> float:
        return _factor(self.sentiment_score, *self.sentiment_band)

    def sector_factor(self, code: str) -> float:
        sector = self.code_to_sector.get(str(code), "")
        score = self.sector_scores.get(sector, 50.0)
        return _factor(score, *self.sector_band)

    def sector_bonus(self, code: str, cfg_market=None) -> float:
        stage = self.sector_stages.get(self.code_to_sector.get(str(code), ""), "")
        bonus = 1.05
        penalty = 0.95
        if cfg_market is not None:
            bonus = getattr(cfg_market, "sector_bonus", 1.05)
            penalty = getattr(cfg_market, "sector_penalty", 0.95)
        if stage in ("启动", "扩散", "主升"):
            return bonus
        if stage == "高潮":
            return penalty
        return 1.0

    def should_block_long(self, threshold: float = 25) -> bool:
        return self.regime_score < threshold

    def should_downgrade(self, code: str, threshold: float = 30) -> bool:
        sector = self.code_to_sector.get(str(code), "")
        return self.sector_scores.get(sector, 50.0) < threshold

    def should_block_high_grade(self, code: str) -> bool:
        sector = self.code_to_sector.get(str(code), "")
        return self.sector_stages.get(sector, "") == "退潮"


def _load_index_klines(cfg) -> dict[str, pd.DataFrame]:
    """从 store 读取指数 K 线。"""
    st = store_mod.open_store(cfg, readonly=True)
    try:
        index_codes = cfg.market.index_codes
        result: dict[str, pd.DataFrame] = {}
        for name, code in index_codes.items():
            try:
                df = st.read_kline(code, 120)
                if not df.empty:
                    result[name] = df
                else:
                    log.warning("指数 %s (%s) 无数据", name, code)
            except Exception:
                log.warning("读取指数 %s (%s) 失败", name, code)
        return result
    finally:
        st.close()


def _load_daily_summary(cfg) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """加载当日和历史的 daily_summary。"""
    market_dir = store_mod.market_dir(cfg)
    summary_path = market_dir / "daily_summary.csv"
    if not summary_path.exists():
        log.warning("daily_summary.csv 不存在，情绪/行业指标使用默认值")
        return pd.DataFrame(), None
    df = pd.read_csv(summary_path, dtype={"code": str})
    if df.empty:
        return pd.DataFrame(), None
    today = df[df["date"] == df["date"].max()]
    history = df
    return today, history


HEAT_HISTORY_FILENAME = "sector_heat_history.json"
HEAT_HISTORY_MAX_DAYS = 60
PREV_HEAT_DAYS = 5  # 阶段判定只需要"前一日 + 一点轨迹"，5 天足够覆盖所有分支


def heat_history_path(cfg) -> Path:
    return store_mod.market_dir(cfg) / HEAT_HISTORY_FILENAME


def load_heat_history(cfg) -> dict[str, dict[str, float]]:
    """读取行业热度历史 {日期: {行业: 分数}}。缺失或损坏一律返回空历史。"""
    p = heat_history_path(cfg)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("行业热度历史读取失败（%s），按空历史处理", exc)
        return {}
    hist = data.get("history") if isinstance(data, dict) else None
    if not isinstance(hist, dict):
        return {}
    return {str(d): dict(v or {}) for d, v in hist.items()}


def save_heat_history(cfg, history: dict[str, dict[str, float]]) -> Path:
    """原子写：先写 .tmp 再 os.replace，按日期升序只留最近 N 天。"""
    days = sorted(history)[-HEAT_HISTORY_MAX_DAYS:]
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history": {
            d: {k: round(float(v), 2) for k, v in sorted(history[d].items())}
            for d in days
        },
    }
    p = heat_history_path(cfg)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


def prev_heat_from_history(
    history: dict[str, dict[str, float]], as_of: str, days: int = PREV_HEAT_DAYS
) -> dict[str, list[float]]:
    """{日期:{行业:分}} → {行业:[近几日分数]}，只取 as_of 之前的最后 days 天。

    没有这份轨迹，_classify_stage 里的 trend 恒为 0，启动/主升/分歧永远不可达。
    """
    dates = [d for d in sorted(history) if d < as_of][-days:]
    out: dict[str, list[float]] = {}
    for d in dates:
        for sector, score in history.get(d, {}).items():
            out.setdefault(sector, []).append(float(score))
    return out


def parse_limit_up_history(data) -> dict[str, list[str]]:
    """涨停榜 → {日期: [代码]}。兼容旧的单日 {"date":..,"codes":..} 结构。"""
    if not isinstance(data, dict):
        return {}
    if "codes" in data:  # 旧 schema：只存了当日
        day = str(data.get("date") or "")
        return {day: list(data.get("codes") or [])} if day else {}
    return {str(k): list(v or []) for k, v in data.items() if k}


def limit_up_streaks(history: dict[str, list[str]]) -> dict[str, int]:
    """最后一个交易日各涨停股的连板数。

    必须按日期升序推：`run` 每天只保留当日仍涨停的代码，掉队的自然清零，
    否则连板会算成"累计涨停天数"。
    """
    run: dict[str, int] = {}
    for d in sorted(k for k in history if k):
        run = {c: run.get(c, 0) + 1 for c in set(history.get(d) or [])}
    return dict(run)


def _load_limit_up_history(cfg) -> dict[str, list[str]]:
    """读取涨停榜历史 {日期: [代码]}。"""
    p = store_mod.market_dir(cfg) / "limit_up_history.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("涨停历史读取失败（%s），按空处理", exc)
        return {}
    return parse_limit_up_history(data)


def _limit_up_as_of(
    history: dict[str, list[str]], as_of: str
) -> tuple[list[str], dict[str, int]]:
    """返回 (上一交易日涨停代码, 该日连板数)。"""
    prev_dates = sorted(d for d in history if d and d < as_of)
    if not prev_dates:
        return [], {}
    before = {d: history[d] for d in prev_dates}
    return list(history.get(prev_dates[-1]) or []), limit_up_streaks(before)


def _load_limit_up_data(cfg, as_of: str) -> tuple[list[str], dict[str, int]]:
    """加载 as_of 当日的行情所需的涨停跟踪数据（取上一交易日那一份）。"""
    return _limit_up_as_of(_load_limit_up_history(cfg), as_of)


def compute_market_snapshot(cfg) -> MarketSnapshot | None:
    """编排入口：计算市场环境 / 情绪 / 行业热度，返回 MarketSnapshot。

    在 scan_market() 开始时调用一次，结果传入所有 worker。
    数据不足时返回 None（composite_score 不做环境修正）。
    """
    mkt = cfg.market
    if not mkt.enabled:
        return None

    today_str = date.today().strftime("%Y-%m-%d")

    # 1. MarketRegime
    index_dfs = _load_index_klines(cfg)
    if not index_dfs:
        log.info("无指数K线数据，跳过市场环境评估")
        return None
    regime_score, regime_label, regime_detail = compute_market_regime(
        index_dfs,
        weights=dict(mkt.index_weights) if hasattr(mkt.index_weights, "items") else mkt.index_weights,
        thresholds=dict(mkt.regime_thresholds) if hasattr(mkt.regime_thresholds, "items") else mkt.regime_thresholds,
    )
    log.info(
        "市场环境: %s (%.1f分) | 指数数据: %s",
        regime_label, regime_score, list(index_dfs.keys()),
    )

    # 2. MarketSentiment（口径按 daily_summary 的数据日，不是墙钟日）
    today_summary, history_summary = _load_daily_summary(cfg)
    data_date = today_str
    if not today_summary.empty:
        data_date = str(today_summary["date"].max())
    index_last = max((str(df.index[-1].date()) for df in index_dfs.values()), default="")
    if index_last and index_last != data_date:
        log.warning(
            "指数最新日 %s 与 daily_summary 数据日 %s 不一致，市场环境与情绪/行业不同源；"
            "先跑 update 或 rebuild-market 对齐",
            index_last, data_date,
        )
    prev_limit_codes, streaks = _load_limit_up_data(cfg, data_date)
    sent_weights = dict(mkt.sentiment_weights) if hasattr(mkt.sentiment_weights, "items") else mkt.sentiment_weights
    sentiment_score, sent_detail = compute_market_sentiment(
        today_summary, sent_weights, prev_limit_codes, streaks,
    )
    log.info("市场情绪: %.1f分 (数据日 %s)", sentiment_score, data_date)

    # 3. SectorHeat
    c2s = code_to_sector_map(cfg)
    sector_weights = dict(mkt.sector_weights) if hasattr(mkt.sector_weights, "items") else mkt.sector_weights
    heat_history = load_heat_history(cfg)
    n_prev = len(heat_history)
    sector_result = compute_sector_heat(
        today_summary, history_summary, c2s,
        prev_heat_from_history(heat_history, data_date), sector_weights,
    )
    sector_scores = {k: v["score"] for k, v in sector_result.items()}
    sector_stages = {k: v["stage"] for k, v in sector_result.items()}
    log.info(
        "行业热度: %d 个行业（历史 %d 天）| 前3: %s",
        len(sector_scores), n_prev,
        sorted(sector_scores.items(), key=lambda x: -x[1])[:3],
    )

    if sector_scores:
        heat_history[data_date] = sector_scores
        try:
            save_heat_history(cfg, heat_history)
        except OSError as exc:
            log.warning("行业热度历史写入失败: %s", exc)

    return MarketSnapshot(
        date=today_str,
        regime_score=regime_score,
        regime_label=regime_label,
        sentiment_score=sentiment_score,
        sector_scores=sector_scores,
        sector_stages=sector_stages,
        code_to_sector=c2s,
        market_band=_bands(mkt, "market"),
        sentiment_band=_bands(mkt, "sentiment"),
        sector_band=_bands(mkt, "sector"),
    )
