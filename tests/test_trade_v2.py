"""破底翻 v2 的交易链路：TP1 分批止盈 / TP2 清仓 / min_rr1 闸门 / 结构移动止盈。

只碰 v2 新增的那几条分叉；旧口径的退出路径由 `test_trade.py` 守着。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradeyh import broker, monitor, trade
from tradeyh.patterns.base import STATUS_FORMED, PatternHit


def _point_trade_files(cfg, tmp_path) -> None:
    cfg.trade.enabled = True
    cfg.trade.account_path = str(tmp_path / "account.json")
    cfg.trade.positions_path = str(tmp_path / "positions.json")
    cfg.trade.history_path = str(tmp_path / "history.json")


def _v2_hit(pattern="break_bottom_v2", levels=None, close=10.5, score=80.0) -> PatternHit:
    base = {
        "L0": 10.0, "L1": 9.6, "B1": 10.4, "H1": 12.0, "B2": 0.0,
        "stop_price": 9.9, "tp1_price": 12.0, "tp2_price": 13.0,
        "rr1": 2.0, "rr2": 3.0,
    }
    base.update(levels or {})
    return PatternHit(
        code="000001", name="测试", pattern=pattern, pattern_cn="破底翻v2",
        status=STATUS_FORMED, score=score, geom_score=score, vol_score=score,
        trend_score=score, signal_date=pd.Timestamp("2026-09-14"),
        key_levels=base, close_price=close,
    )


def _decision(hit: PatternHit, price: float) -> monitor.Decision:
    return monitor.Decision(
        code=hit.code, name=hit.name, kind=monitor.PROMOTED, hit=hit, price=price
    )


def _position(code="000001", entry=10.5, shares=1000, stop=9.9, target=0.0,
              tp1=0.0, tp2=0.0, partial=False) -> trade.Position:
    return trade.Position(
        code=code, name="测试", pattern="break_bottom_v2", pattern_cn="破底翻v2",
        entry_price=entry, shares=shares, cost=entry * shares,
        stop_loss=stop, highest_price=entry, target_price=target,
        entry_time="2026-09-14 09:35:00", entry_score=80.0,
        tp1_price=tp1, tp2_price=tp2, partial_closed=partial,
    )


# ---------------- 开仓 ----------------

def test_open_stores_tp1_tp2(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(broker, "_broker", None)
    _point_trade_files(cfg, tmp_path)

    quote = pd.DataFrame([{"code": "000001", "price": 10.5}])
    monitor.handle_trading(cfg, [_decision(_v2_hit(), 10.5)], quote)

    pos = trade.load_positions(cfg)[0]
    assert pos.tp1_price == 12.0
    assert pos.tp2_price == 13.0
    # TP2 存在时用 TP2 当固定目标价（TP1 只管分批），别在 TP1 就清光
    assert pos.target_price == 13.0


def test_min_rr1_blocks_low_reward_setup(cfg, tmp_path, monkeypatch):
    """RR1 不足则不买入。用 min_target_upside=0 隔离掉"上涨空间"那道闸门。"""
    monkeypatch.setattr(broker, "_broker", None)
    _point_trade_files(cfg, tmp_path)
    cfg.trade.min_target_upside = 0
    cfg.trade.min_rr1 = 1.5

    # 现价 10.5、止损 9.9 → risk 0.6；TP1 10.8 → reward 0.3 → RR1=0.5
    hit = _v2_hit(levels={"tp1_price": 10.8})
    quote = pd.DataFrame([{"code": "000001", "price": 10.5}])
    monitor.handle_trading(cfg, [_decision(hit, 10.5)], quote)
    assert trade.load_positions(cfg) == []

    # 关掉闸门就该放行——证明挡住它的是 min_rr1 而不是别的
    cfg.trade.min_rr1 = 0
    monitor.handle_trading(cfg, [_decision(hit, 10.5)], quote)
    assert len(trade.load_positions(cfg)) == 1


def test_old_pattern_skips_rr1_gate(cfg, tmp_path, monkeypatch):
    """旧口径没有 tp1，min_rr1 闸门必须自动跳过（否则会拦掉所有旧形态）。"""
    monkeypatch.setattr(broker, "_broker", None)
    _point_trade_files(cfg, tmp_path)
    cfg.trade.min_rr1 = 99.0

    hit = PatternHit(
        code="000001", name="测试", pattern="break_bottom", pattern_cn="破底翻",
        status=STATUS_FORMED, score=80.0, geom_score=80.0, vol_score=80.0,
        trend_score=80.0, signal_date=pd.Timestamp("2026-09-14"),
        key_levels={"参考低点": 10.0, "参考止损": 9.9, "上方压力": 12.0},
        close_price=10.5, pattern_stage="",
    )
    quote = pd.DataFrame([{"code": "000001", "price": 10.5}])
    monitor.handle_trading(cfg, [_decision(hit, 10.5)], quote)
    assert len(trade.load_positions(cfg)) == 1


def test_b2_open_records_h1_as_breakout_level(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(broker, "_broker", None)
    _point_trade_files(cfg, tmp_path)

    hit = _v2_hit(pattern="break_bottom_v2_b2")
    quote = pd.DataFrame([{"code": "000001", "price": 10.5}])
    monitor.handle_trading(cfg, [_decision(hit, 10.5)], quote)
    pos = trade.load_positions(cfg)[0]
    assert pos.breakout_level == 12.0
    # 突破日期取突破 bar 的日期（signal_date），不是开仓日——失败突破窗口从它起算
    assert pos.breakout_date == hit.signal_date.strftime("%Y-%m-%d")


def test_b2_open_records_breakout_bar_date_not_entry_date(cfg, tmp_path, monkeypatch):
    """旧 b2 已形成时 signal_date = 突破 bar 日期；开仓日若晚于它，不能拿开仓日当突破日。"""
    monkeypatch.setattr(broker, "_broker", None)
    _point_trade_files(cfg, tmp_path)

    hit = PatternHit(
        code="000001", name="测试", pattern="break_bottom_b2", pattern_cn="破底翻第二买点",
        status=STATUS_FORMED, score=80.0, geom_score=80.0, vol_score=80.0,
        trend_score=80.0, signal_date=pd.Timestamp("2026-09-10"),
        key_levels={"第一波高点": 12.0, "参考止损": 9.9, "等幅目标": 13.0},
        close_price=10.5,
    )
    quote = pd.DataFrame([{"code": "000001", "price": 10.5}])
    monitor.handle_trading(cfg, [_decision(hit, 10.5)], quote)
    pos = trade.load_positions(cfg)[0]
    assert pos.breakout_level == 12.0
    assert pos.breakout_date == "2026-09-10"


# ---------------- 退出 ----------------

def test_tp1_triggers_partial_not_full_close(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(broker, "_broker", None)
    _point_trade_files(cfg, tmp_path)

    trade.save_positions(cfg, [_position(shares=1000, tp1=12.0)])
    quote = pd.DataFrame([{"code": "000001", "price": 12.1}])
    trades = monitor.handle_trading(cfg, [], quote)

    assert [t[1] for t in trades] == ["分批止盈"]
    pos = trade.load_positions(cfg)[0]
    assert pos.shares == 500, "TP1 只该卖一半"
    assert pos.partial_closed is True


def test_tp2_after_partial_closes_rest():
    """TP1 分批之后，TP2 到价走"目标价"整仓清掉。"""
    pos = _position(target=13.0, tp1=12.0, tp2=13.0, partial=True)
    assert trade.check_exit(pos, 13.2, _minimal_cfg()) == "目标价"


def test_tp1_price_does_not_fire_twice():
    pos = _position(target=0.0, tp1=12.0, partial=True)
    assert trade.check_exit(pos, 12.5, _minimal_cfg()) is None


def test_failed_breakout_window_comes_from_trade_section():
    """失败突破窗口取 trade.failed_breakout_bars，不再借 break_bottom_b2 的检测器参数。"""
    pos = _position(target=0.0)
    pos.breakout_level = 12.0
    pos.breakout_date = "2026-09-10"

    # 检测器块给个大值、trade 给 0：若还读检测器就会误判失败突破
    cfg = _minimal_cfg()
    cfg.trade.failed_breakout_bars = 0
    cfg.break_bottom_b2.failed_breakout_bars = 99
    assert trade.check_exit(pos, 11.5, cfg, trading_day_count=1) is None

    # trade 放行才触发
    cfg.trade.failed_breakout_bars = 3
    assert trade.check_exit(pos, 11.5, cfg, trading_day_count=1) == "失败突破"


def _minimal_cfg():
    class _Section(dict):
        def __getattr__(self, k):
            return self[k]

    return _Section(
        trade=_Section(
            breakeven_pct=0.03, trail_trigger_pct=0.05, trail_pct=0.02,
            partial_close_pct=0.5, trail_use_ma_hl=True, min_rr1=1.5,
            failed_breakout_bars=3,
        ),
        break_bottom_b2=_Section(failed_breakout_bars=3),
    )


# ---------------- 移动止盈 ----------------

def test_trail_ref_replaces_percent_trailing():
    """v2 持仓的止损跟结构位走，不再用 highest_price×(1−trail_pct)。"""
    pos = _position(entry=10.5, stop=9.9, tp1=12.0)
    trade.update_trailing_stop(pos, 11.5, _minimal_cfg(), trail_ref=10.8)
    assert pos.stop_loss == pytest.approx(10.8)
    assert pos.stop_loss != pytest.approx(11.5 * 0.98), "不该退化成百分比回撤"


def test_trail_ref_never_lowers_stop():
    pos = _position(entry=10.5, stop=11.0, tp1=12.0)
    trade.update_trailing_stop(pos, 11.5, _minimal_cfg(), trail_ref=10.8)
    assert pos.stop_loss == pytest.approx(11.0)


def test_old_position_keeps_percent_trailing():
    pos = _position(entry=10.0, stop=9.7, target=12.0)  # tp1=0
    trade.update_trailing_stop(pos, 10.6, _minimal_cfg())
    assert pos.stop_loss == pytest.approx(10.6 * 0.98)


# ---------------- 兼容与显示 ----------------

def test_legacy_positions_json_loads_without_tp_fields(cfg, tmp_path):
    """磁盘上已有的 positions.json 没有 tp1/tp2 键，加载不能炸。"""
    _point_trade_files(cfg, tmp_path)
    legacy = {
        "code": "000001", "name": "老仓", "pattern": "break_bottom",
        "pattern_cn": "破底翻", "entry_price": 10.0, "shares": 100,
        "cost": 1000.0, "stop_loss": 9.7, "highest_price": 10.0,
        "target_price": 11.5, "entry_time": "2026-09-14 09:35:00",
        "entry_score": 70.0, "trailing_activated": False, "order_id": "",
        "breakout_level": 0.0, "breakout_date": "", "partial_closed": False,
    }
    trade.positions_path(cfg).write_text(
        json.dumps({"positions": [legacy]}), encoding="utf-8"
    )
    pos = trade.load_positions(cfg)[0]
    assert pos.tp1_price == 0.0 and pos.tp2_price == 0.0


def test_alert_row_falls_back_to_v2_keys():
    """ALERT_COLUMNS 不变，取值回退到 v2 的 L0/stop_price/tp1_price。"""
    hit = _v2_hit()
    row = monitor._alert_row(_decision(hit, 10.5))
    assert row["参考低点"] == 10.0
    assert row["参考止损"] == 9.9
    assert row["上方压力"] == 12.0


def test_print_round_shows_v2_levels(capsys):
    hit = _v2_hit()
    monitor.print_round([_decision(hit, 10.5)], pool_size=1)
    out = capsys.readouterr().out
    assert "L0=10.00" in out and "TP1=12.00" in out and "止损9.90" in out
