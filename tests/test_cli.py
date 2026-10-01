"""CLI 出图开关：默认不出图，--plot / --no-plot 显式覆盖。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cli  # noqa: E402

from tradeyh import config  # noqa: E402


def _scan_args(*argv: str):
    return cli.build_parser().parse_args(["scan", *argv])


def test_plot_off_by_default(cfg):
    """不敲 --plot 就不出图（args.plot 为 None = 未指定，跟随 charts.enabled）。"""
    args = _scan_args()
    assert args.plot is None
    assert cli._want_plot(cfg, args) is False


def test_defaults_charts_disabled():
    """DEFAULTS 与 config.yaml 要同步：出图默认关，回到"扫描只出报表"。"""
    assert config.DEFAULTS["charts"]["enabled"] is False
    assert config.load_config(config.DEFAULT_CONFIG_PATH).charts.enabled is False


def test_plot_flag_enables(cfg):
    """--plot 显式指定时压过配置。"""
    cfg.charts.enabled = False
    assert cli._want_plot(cfg, _scan_args("--plot")) is True


def test_no_plot_flag_disables(cfg):
    """--no-plot 同样压过配置（charts.enabled=true 时的临时关闭）。"""
    cfg.charts.enabled = True
    assert cli._want_plot(cfg, _scan_args()) is True
    assert cli._want_plot(cfg, _scan_args("--no-plot")) is False
