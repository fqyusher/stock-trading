"""破底翻 v2 已整体停用（扫描 / 监控 / 出图都不跑），这里把这条状态钉住。

停用的做法是**从检测器注册表摘除**（`patterns/__init__.py`），而不是只把 config 开关
翻成 false —— 否则 `scan --patterns break_bottom_v2` 还会被接受、CLI 帮助里还列着它、
`watch.patterns` 里也能写进去。三条消费链（scan / watch / plot）都从注册表派生，
所以注册表是唯一需要守的地方。

同时要保证"重新启用"仍是一条可行的两步骤操作，所以这里也断言 v2 模块本身还在、
直接调用可用（只是被 enabled=false 挡住）。
"""

from __future__ import annotations

import pytest

from tradeyh import config, scan, watch
from tradeyh.patterns import DETECTORS, PATTERN_CN, PATTERN_KEYS

V2_KEYS = ("break_bottom_v2", "break_bottom_v2_b2")
V2_PATTERNS = ("break_bottom_v2", "break_bottom_v2_b2")


# ---------------------------------------------------------------------------
# 注册表里不能有 v2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", V2_KEYS)
def test_v2_not_registered_as_detector(key):
    assert key not in DETECTORS, f"{key} 又回到检测器注册表了"


@pytest.mark.parametrize("key", V2_KEYS)
def test_v2_not_registered_as_pattern_key(key):
    assert key not in PATTERN_KEYS


@pytest.mark.parametrize("pat", V2_PATTERNS)
def test_v2_not_in_display_names(pat):
    """展示名表里没有 v2 → 报表分表、观察池中文名、出图都不会给它留位置。"""
    assert pat not in PATTERN_CN


def test_resolve_detectors_rejects_v2():
    """CLI `--patterns break_bottom_v2` 必须报"未知形态"，而不是静默返回空。"""
    with pytest.raises(ValueError, match="未知形态"):
        scan.resolve_detectors(["break_bottom_v2"])


def test_resolve_detectors_none_has_no_v2():
    got = scan.resolve_detectors(None)
    assert not any(k in got for k in V2_KEYS)


# ---------------------------------------------------------------------------
# config / 观察池 里不能有 v2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", V2_PATTERNS)
def test_v2_config_disabled(cfg, key):
    assert getattr(cfg, key).enabled is False


def test_watch_patterns_excludes_v2(cfg):
    assert not (set(cfg.watch.patterns) & set(V2_PATTERNS))


def test_watch_sync_rejects_v2_in_patterns(cfg, tmp_path):
    """watch.patterns 里手写 v2 应当**明确报错**（v2 已不在可选形态里），
    而不是静默收不到候选、也不是静默收进来。"""
    report = tmp_path / "scan_20260930.csv"
    report.write_text("代码,名称,形态,形态key,状态,综合评分,失效\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知形态"):
        watch.sync_from_report(cfg, report_path=str(report), patterns=["break_bottom_v2"])


# ---------------------------------------------------------------------------
# 重新启用的路径仍然存在（只是被 switches 挡住）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod_name", V2_KEYS)
def test_v2_modules_still_importable(mod_name):
    """停用的是"接入"，不是"删代码"——模块要还在，否则重启用就变成重写了。"""
    mod = __import__(f"tradeyh.patterns.{mod_name}", fromlist=[mod_name])
    assert callable(mod.detect)
    assert mod.PATTERN == mod_name


@pytest.mark.parametrize("mod_name", V2_KEYS)
def test_v2_detectors_return_nothing_when_disabled(mod_name, make_ctx):
    """即使在测试里直接调它，config 关着也必须返回空。"""
    mod = __import__(f"tradeyh.patterns.{mod_name}", fromlist=[mod_name])
    assert mod.detect(make_ctx(_flat_history())) == []


def _flat_history():
    import numpy as np
    import pandas as pd

    n = 300
    close = np.full(n, 10.0)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": close, "close": close, "high": close, "low": close,
            "volume": np.full(n, 1e6), "amount": np.full(n, 1e7),
            "turnover": np.full(n, 1.0), "pct_chg": np.zeros(n),
        },
        index=idx,
    )


def test_live_state_has_no_v2_entries(cfg, tmp_path):
    """观察池里不该再出现 v2 条目（历史上清过一次，防止又被 sync 带回来）。"""
    pool = watch.load_pool(cfg)
    assert not [e for e in pool if e.get("pattern") in V2_PATTERNS]


def test_monitor_drops_stale_v2_pool_entry(monkeypatch, cfg):
    """池子里若残留 v2 条目（手工编辑/旧池文件），监控应把它按失效移出，
    而不是每轮刷警告还留着当僵尸。"""
    import pandas as pd

    from tradeyh import monitor

    class FakeStore:
        def read_kline(self, code, n):
            return _flat_history()          # 完整 OHLCV，build_context 要算指标

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

    monkeypatch.setattr(monitor.store, "open_store", lambda c, readonly=False: FakeStore())

    pool = [{"code": "000001", "name": "测试", "pattern": "break_bottom_v2"}]
    quotes = pd.DataFrame([{"code": "000001", "price": 10.0}])
    decisions, kept = monitor.check_once(cfg, pool, quotes, interval=60)

    assert decisions[0].kind == monitor.INVALID
    assert "break_bottom_v2" in decisions[0].reason
    assert kept == [], "残留的 v2 条目应被移出池子"
