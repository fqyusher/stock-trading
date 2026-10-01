"""配置加载：config.yaml 覆盖 DEFAULTS，敏感字段从环境变量 / .env 注入，支持点号访问。

形态阈值全部走配置，代码里不硬编码数字。DEFAULTS 保证即使 config.yaml
缺键或整个文件不存在，程序依然能跑。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "data": {
        "storage": "csv",
        "sources": ["baostock", "eastmoney", "tencent", "sina"],
        "db_path": "data/kline.db",
        "data_dir": "data/store",
        "out_dir": "out",
        "history_start": "20220101",
        "adjust": "qfq",
        "period": "daily",
        "max_workers": 4,
        "request_timeout": 15,
        "retry": 3,
        "drift_check_bars": 30,
        "drift_tolerance": 0.002,
    },
    "universe": {
        "exclude_st": True,
        "exclude_delisted": True,
        "exclude_star_market": True,
        "exclude_bj": True,
        "min_bars": 250,
        "min_float_mv": 0,
        "min_amount": 0,
        "max_price": 0,
    },
    "swings": {
        "pct_min": 0.05,
        "atr_period": 14,
        "atr_k": 1.5,
        "fractal_window": 5,
    },
    "w_bottom": {
        "enabled": True,
        "bottom_tol": 0.05,
        "min_neck_height": 0.08,
        "max_neck_height": 0.60,
        "min_width": 15,
        "max_width": 150,
        "time_ratio_min": 0.3,
        "time_ratio_max": 3.0,
        "prior_drop": 0.15,
        "prior_lookback": 60,
        "breakout_buffer": 0.01,
        "breakout_within": 10,
        "forming_near_neck": 0.05,
        "forming_max_age": 30,
        "vol_ratio_breakout": 1.5,
    },
    "m_top": {
        "enabled": True,
        "top_tol": 0.05,
        "min_neck_depth": 0.08,
        "max_neck_depth": 0.60,
        "min_width": 15,
        "max_width": 150,
        "time_ratio_min": 0.3,
        "time_ratio_max": 3.0,
        "prior_rise": 0.15,
        "prior_lookback": 60,
        "breakdown_buffer": 0.01,
        "breakdown_within": 10,
        "forming_near_neck": 0.05,
        "forming_max_age": 30,
        "vol_ratio_breakdown": 1.2,
    },
    "break_bottom": {
        "enabled": True,
        # 前低 L0 与破底新低的间隔（根数）：1 个月 ≈ 20 根、2 个月 ≈ 45 根
        "min_span": 20,
        "max_span": 45,
        # 二次探底（破底新低）须发生在最近 N 根内。取与回看量级相当：
        # 探底本身超过一个季度，整段结构就不再是"短期诱空"，而"够不够新鲜、
        # 还值不值得动手"交给 breakout_within（已形成）与 forming_max_age（构筑中）。
        "break_within": 63,
        # 前置结构：L0 之前须有的跌幅（确认处于下行趋势）
        "prior_drop": 0.15,
        "prior_lookback": 60,
        # 破底深度：低于下限只是噪声级新低（不算"有效跌破"），高于上限是真破位不是"翻"
        "min_break_depth": 0.015,
        "max_break_depth": 0.10,
        # 颈线 = 前低之后的反弹高点
        "min_neck_height": 0.05,
        "max_neck_height": 0.60,
        "recover_bars": 5,
        # 收复 L0 后连续 N 根收盘又落在 L0 下方 = 这次"翻"失败（与 break_bottom_b2 同名同义）
        "pullback_invalid_consecutive": 2,
        "neck_breakout_buffer": 0.005,
        "neck_breakout_vol_ratio": 1.3,
        "breakout_within": 10,
        "forming_max_age": 30,
        "stop_atr_buffer": 0.5,
    },
    "break_bottom_b2": {
        "enabled": True,
        "break_within_b2": 100,
        "min_bars_after_breakout": 8,
        "breakout_within": 10,
        # --- B1 确认（规范第1点）---
        "b1_atr_k": 0.2,              # Close > N + k×ATR
        # --- H1 定义（规范第2点）---
        "h1_min_pct": 0.05,           # H1-B1 >= 5%
        "h1_min_atr_k": 1.5,          # H1-B1 >= 1.5×ATR
        # --- 颈线支撑区（规范第4点）---
        "neck_support_lower_atr_k": 0.5,   # N - 0.5×ATR
        "neck_support_upper_atr_k": 0.3,   # N + 0.3×ATR
        # --- 回踩幅度分级（规范第5点）---
        "pullback_optimal_max": 0.50,      # 最佳上限 50%
        "pullback_acceptable_max": 0.70,   # 可接受上限 70%
        "pullback_min_pct": 0.02,          # 最小回踩幅度（噪声过滤）
        "pullback_min_atr_k": 0.8,
        # --- 回踩量能（规范第6点）---
        "vol_shrink_ratio": 0.85,          # 回踩均量 <= 第一波均量 × 此值
        "sustained_vol_ratio": 1.5,        # 持续放量阈值（相对 vol_ma5）
        # --- B2 确认（规范第7点）---
        "b2_confirm_ma": 5,                # 确认用 MA 周期
        # --- 强确认（规范第8点）---
        # （无需额外配置，直接检查 Vol>=MA5_Vol AND MACD改善 AND RSI上行）
        # --- 失效条件（规范第9点）---
        "neck_invalid_atr_k": 0.3,         # Close < N - k×ATR
        "pullback_invalid_consecutive": 2,  # 连续N日
        "pullback_l0_atr_buffer": 0.5,     # 跌破L0的ATR缓冲
        # --- 止损（规范第10点）---
        "stop_atr_buffer": 0.5,
        # --- 追高控制 ---
        "chase_reduce_pct": 0.05,
        "chase_abort_pct": 0.08,
        # --- 时效 ---
        "failed_breakout_bars": 3,
        "forming_max_age": 40,
    },
    # 破底翻 v2（ATR 严格口径）—— 2026-09-30 起整体停用：模块文件保留，但已从
    # patterns/__init__.py 的检测器注册表摘除，scan / 监控 / 出图都不会再碰它。
    # 链路：L0(确认摆动低点) → 有效破底(ATR 阈值，仅盘中破也算) → 快速收复 L0
    # → 跌破失败确认 → B1 确认买点。
    # 重新启用：①在 patterns/__init__.py 重新登记 ②把 enabled 改回 True ③加回 watch.patterns。
    "break_bottom_v2": {
        "enabled": False,
        "l0_lookback": 60,          # L0 = 最近N根内确认的 Swing Low
        "l0_must_be_leg_low": True, # L0 回看区内不得有更低的原始最低价
        "prior_drop": 0.15,
        "prior_lookback": 60,
        "break_low_atr_k": 0.3,     # Low < L0 - k×ATR → 有效破底
        "break_close_atr_k": 0.2,   # 或 Close < L0 - k×ATR
        "min_break_depth": 0.015,
        "max_break_depth": 0.10,    # 定义外新增：真破位 30% 再反弹不该叫破底翻
        "recover_bars": 5,          # 自破底最低点起N根内 Close > L0
        "fail_confirm_bars": 3,     # 收复后N根内不得再有效跌破（按收盘价判定，插针不算）
        "fail_confirm_atr_k": 0.3,  # 判据：收盘 > L0 - k×ATR
        "b1_close_atr_k": 0.3,      # B1①：Close > L0 + k×ATR
        "b1_hold_days": 2,          # B1②：连续N日 Close > L0
        "h1_min_pct": 0.05,
        "h1_min_atr_k": 1.5,
        "stop_atr_k": 0.3,          # 结构止损 = L0 - k×ATR
        "extreme_stop_atr_k": 0.5,  # 极限破底止损 = L1 - k×ATR
        "tp2_lookback": 120,
        "tp2_dense_bins": 24,
        "rr_observe": 1.5,
        "rr_high": 2.0,
        "rr_value": 3.0,
        "breakout_within": 63,      # B1 确认须发生在最近N根内
        "forming_max_age": 30,
        "emit_invalid": True,
        "invalid_within": 20,
        "trail_ma_atr_k": 1.0,      # 移动止盈①：MA20 - k×ATR
        "trail_swing_atr_k": 0.3,   # 移动止盈②：最近SwingLow - k×ATR
    },
    # 破底翻 v2 第二买点 —— 已随 v2 一起停用（重新启用见 break_bottom_v2 上面的说明）。
    "break_bottom_v2_b2": {
        "enabled": False,
        "break_within_v2": 100,
        "min_bars_after_b1": 8,
        "pullback_min_pct": 0.02,
        "pullback_min_atr_k": 0.8,
        "pullback_l0_atr_k": 0.3,   # 回踩不得有效跌破 L0
        "pullback_max_vol_ratio": 0.85,
        "breakout_buffer_pct": 0.005,
        "breakout_vol_ratio": 1.2,
        "breakout_within": 10,
        "chase_abort_pct": 0.08,
        "stop_swing_atr_k": 0.3,
        "forming_max_age": 40,
        "emit_invalid": True,
        "invalid_within": 20,
    },
    "channel": {
        "enabled": True,
        "windows": [60, 90, 120],
        "min_r2": 0.35,
        "min_containment": 0.90,
        "min_width": 0.05,
        "max_width": 0.35,
        "touch_tol": 0.015,
        "min_touches": 2,
        "trend_threshold": 0.08,
        "resid_q_upper": 0.95,
        "resid_q_lower": 0.05,
    },
    "breakout": {
        "enabled": True,
        "consolidation_window": 25,
        "prior_lookback": 60,
        "platform_trend": 0.12,
        "platform_slope": 0.012,
        "box_slope": 0.006,
        "flag_trend": 0.15,
        "flag_slope": 0.02,
        "max_range": 0.18,
        "target_range": 0.08,
        "breakout_buffer": 0.01,
        "forming_near": 0.03,
        "cup_window": 80,
        "cup_min_depth": 0.12,
        "cup_max_depth": 0.45,
        "cup_rim_tol": 0.08,
        "rounding_window": 80,
        "rounding_min_depth": 0.10,
        "rounding_max_depth": 0.40,
        "rounding_symmetry": 0.70,
        "pullback_touch_tol": 0.015,
    },
    "trend_pullback": {
        "enabled": True,
        "lookback": 60,
        "min_rise": 0.10,
        "pullback_min": 0.03,
        "pullback_max": 0.20,
        "ma20_tol": 0.015,
        "ma60_tol": 0.02,
        "stop_buffer": 0.03,
        "high_max_age": 15,
    },
    "scoring": {
        "geom_weight": 50,
        "vol_weight": 25,
        "trend_weight": 25,
        "forming_discount": 0.85,
        "min_score": 0,
        "dedup_enabled": True,
        "invalidation_enabled": True,
        "regime_boost_pct": 8,
        "regime_trend_threshold": 25,
        "regime_range_threshold": 20,
        "time_decay_half_life_days": 30,
        "time_decay_floor": 0.1,
    },
    "market": {
        "enabled": True,
        "index_codes": {
            "csi300": "idx_000300",
            "csi1000": "idx_000905",
            "chinext": "idx_399006",
        },
        "index_weights": {"csi300": 0.4, "csi1000": 0.3, "chinext": 0.3},
        "regime_thresholds": {
            "strong": 80, "bullish": 65, "neutral": 45, "weak": 25,
        },
        "sentiment_weights": {
            "advance_ratio": 0.20, "median_return": 0.20,
            "limit_count": 0.15, "failed_limit_rate": 0.15,
            "limit_follow_through": 0.15, "max_streak": 0.15,
        },
        "sector_weights": {
            "relative_return": 0.20, "volume_ratio": 0.15,
            "breadth": 0.15, "trend": 0.20,
            "capital_intensity": 0.15, "outperform_days": 0.15,
        },
        "sector_bonus": 1.05,
        "sector_penalty": 0.95,
        # 环境乘子的带宽：中性 50 分必须映射成 1.00（否则综合评分整体缩水，
        # min_score / Excel 色阶锚点全失准）。0 分 → lo，100 分 → hi，两段线性。
        "factor_bands": {
            "market": {"lo": 0.85, "hi": 1.15},
            "sentiment": {"lo": 0.95, "hi": 1.05},
            "sector": {"lo": 0.90, "hi": 1.10},
        },
        "block_long_regime_threshold": 25,
        "downgrade_sector_threshold": 30,
        "history_days": 60,  # rebuild-market 默认回填的交易日数
    },
    "charts": {
        "enabled": False,  # 默认不出图；--plot 或改 true 才画
        "dir": "",  # 空 = data.out_dir/charts
        "bars": 180,  # 每张图显示的K线根数
        "max_charts": 0,  # 单次扫描最多出图数，0=不限（默认全出）
        "dpi": 130,
        "include_ma20": True,
        "index_html": True,
    },
    "watch": {
        "enabled": True,
        "pool_path": "data/watch/pool.json",
        "patterns": ["break_bottom"],
        "statuses": ["forming"],
        "min_score": 0,
        "max_pool": 50,
        "interval": 60,
        "pool_refresh_interval": 900,
        "confirm_rounds": 2,
        "quote_sources": ["eastmoney", "tencent", "ths"],
        "bell": False,
    },
    "trade": {
        "enabled": True,
        "account_path": "data/trade/account.json",
        "positions_path": "data/trade/positions.json",
        "history_path": "data/trade/history.json",
        "initial_capital": 100000,
        "risk_per_trade": 0.01,
        "max_positions": 10,
        "breakeven_pct": 0.03,
        "trail_trigger_pct": 0.05,
        "trail_pct": 0.02,
        "min_target_upside": 0.08,
        "max_entry_drift": 0.10,
        # B2 持仓在突破 bar 后 N 个【交易日】内跌回 H1 下方 → 失败突破离场。
        # 这是交易层的退出策略，与检测器 break_bottom_b2.failed_breakout_bars
        # （信号过滤）各自独立，别互相牵动。
        "failed_breakout_bars": 3,
        "partial_close_pct": 0.5,
        # RR1（第一目标空间/风险）不足此值则不买入。只对带 tp1_price 的 v2 口径生效：
        # 旧口径没有第一目标，缺失时闸门自动跳过，行为不变。
        "min_rr1": 1.5,
        "trail_use_ma_hl": True,
        "emt": {
            "enabled": False,
            "ip": "61.152.230.41",
            "port": 19088,
            "account_id": "",
            "password": "",
            "local_ip": "127.0.0.1",
        },
    },
}


ENV_FILE = PROJECT_ROOT / ".env"

# 敏感字段只认环境变量（或项目根的 .env）：config.yaml 纳入版本控制，不能放凭证。
# 键名 -> 在配置树里的路径。
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "TRADEYH_EMT_ACCOUNT_ID": ("trade", "emt", "account_id"),
    "TRADEYH_EMT_PASSWORD": ("trade", "emt", "password"),
}


def load_env_file(path: str | Path | None = None) -> None:
    """把 .env 的 KEY=VALUE 灌进 os.environ，已存在的环境变量优先（不覆盖）。

    优先级：真实 shell 环境变量 > .env > config.yaml。行首 # 为注释；值只做
    首尾去空白与去成对引号，密码里的 # 不会被当注释截断。
    """
    env_path = Path(path) if path else ENV_FILE
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _apply_env_overrides(cfg: dict) -> None:
    for env_key, path in ENV_OVERRIDES.items():
        value = os.environ.get(env_key, "").strip()
        if not value:
            continue
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value


class Section(dict):
    """点号访问的字典，嵌套字典自动包装。

    嵌套字典首次访问时包装成 Section 并**原地替换**原值，
    保证 cfg.data["x"] = v 这类修改真正生效（而不是写进临时拷贝）。
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = dict.__getitem__(self, key)
        except KeyError as exc:
            raise AttributeError(f"配置项不存在: {key}") from exc
        if isinstance(value, dict) and not isinstance(value, Section):
            value = Section(value)
            dict.__setitem__(self, key, value)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> Section:
    """读取配置。data.db_path / data.data_dir / data.out_dir / watch.pool_path
    会解析成相对项目根目录的绝对路径。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    user_cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}

    merged = _deep_merge(DEFAULTS, user_cfg)
    load_env_file()
    _apply_env_overrides(merged)

    for key in ("db_path", "data_dir", "out_dir"):
        raw = Path(merged["data"][key])
        merged["data"][key] = str(raw if raw.is_absolute() else PROJECT_ROOT / raw)

    raw = Path(merged["watch"]["pool_path"])
    merged["watch"]["pool_path"] = str(raw if raw.is_absolute() else PROJECT_ROOT / raw)

    # charts.dir 留空表示"用 out_dir/charts"，只有填了才解析成绝对路径
    charts = merged.setdefault("charts", {})
    if str(charts.get("dir") or "").strip():
        raw = Path(charts["dir"])
        charts["dir"] = str(raw if raw.is_absolute() else PROJECT_ROOT / raw)

    for key in ("account_path", "positions_path", "history_path"):
        raw = Path(merged["trade"][key])
        merged["trade"][key] = str(raw if raw.is_absolute() else PROJECT_ROOT / raw)

    return Section(merged)
