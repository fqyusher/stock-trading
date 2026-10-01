"""配置加载：yaml 覆盖 DEFAULTS、敏感字段从 .env / 环境变量注入。"""

from __future__ import annotations

import os

import yaml

from tradeyh import config


def test_env_file_parsing(tmp_path, monkeypatch):
    """注释行、空行、export 前缀、成对引号、引号内含 # 的密码。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 行首注释\n"
        "\n"
        "TRADEYH_EMT_ACCOUNT_ID=510100006862\n"
        'TRADEYH_EMT_PASSWORD="p#ss=word"\n'
        "export TRADEYH_OTHER=1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "environ", {})

    config.load_env_file(env_file)

    assert os.environ["TRADEYH_EMT_ACCOUNT_ID"] == "510100006862"
    assert os.environ["TRADEYH_EMT_PASSWORD"] == "p#ss=word"
    assert os.environ["TRADEYH_OTHER"] == "1"


def test_real_env_wins_over_env_file(tmp_path, monkeypatch):
    """.env 只做 setdefault：shell 里已存在的变量不被覆盖。"""
    env_file = tmp_path / ".env"
    env_file.write_text("TRADEYH_EMT_PASSWORD=from_file\n", encoding="utf-8")
    monkeypatch.setattr(os, "environ", {"TRADEYH_EMT_PASSWORD": "from_shell"})

    config.load_env_file(env_file)

    assert os.environ["TRADEYH_EMT_PASSWORD"] == "from_shell"


def test_env_overrides_reach_config(tmp_path, monkeypatch):
    """环境变量注入到 cfg.trade.emt，且覆盖 config.yaml 里的占位。"""
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "不存在.env")

    cfg = config.load_config(config.DEFAULT_CONFIG_PATH)
    assert cfg.trade.emt.account_id == ""

    monkeypatch.setenv("TRADEYH_EMT_ACCOUNT_ID", "510100006862")
    monkeypatch.setenv("TRADEYH_EMT_PASSWORD", "p#ss")
    cfg = config.load_config(config.DEFAULT_CONFIG_PATH)
    assert cfg.trade.emt.account_id == "510100006862"
    assert cfg.trade.emt.password == "p#ss"


def test_repo_config_has_no_credentials():
    """config.yaml 纳入版本控制，文件里的凭证必须是空占位（真实值只放 .env）。

    故意读原始 yaml 而不是 load_config：后者会把本地 .env 的值注进来。
    """
    raw = yaml.safe_load(config.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["trade"]["emt"]["account_id"] == ""
    assert raw["trade"]["emt"]["password"] == ""
