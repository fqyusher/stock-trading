"""模拟交易账户完整性回归。

覆盖 monitor.handle_trading 的平仓路径。曾经的 bug：当"某只到价平仓"与
"另一只移动止损上移"发生在同一轮时，函数末尾用加载时的旧持仓列表整体写回，
把刚平掉的仓复活、现金重复入账。修复后只保存幸存持仓，平掉的不再回写。
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradeyh import broker, monitor, trade


def _point_trade_files(cfg, tmp_path) -> None:
    """把账户/持仓/历史三个文件指向临时目录，别碰真实 data/trade。"""
    cfg.trade.enabled = True
    cfg.trade.account_path = str(tmp_path / "account.json")
    cfg.trade.positions_path = str(tmp_path / "positions.json")
    cfg.trade.history_path = str(tmp_path / "history.json")


def _position(code, name, entry, shares, stop, target) -> trade.Position:
    return trade.Position(
        code=code, name=name, pattern="w_bottom", pattern_cn="W底",
        entry_price=entry, shares=shares, cost=entry * shares,
        stop_loss=stop, highest_price=entry, target_price=target,
        entry_time="2026-09-14 09:35:00", entry_score=70.0,
    )


def test_closed_position_not_resurrected_when_other_stop_moves(cfg, tmp_path, monkeypatch):
    """一只到目标价分批止盈、同轮另一只移动止损上移：分批卖出的不能复活，现金只入账一次。"""
    monkeypatch.setattr(broker, "_broker", None)  # 隔离券商单例；EMT 未启用走本地模拟
    _point_trade_files(cfg, tmp_path)

    # A：现价 10.6（+6%），本轮触发移动止损上移到 10.6*0.98=10.388，但不平仓（幸存）
    pos_a = _position("000001", "甲", entry=10.0, shares=1000, stop=9.7, target=12.0)
    # B：现价 22.5（+12.5%），越过目标价 22.0，本轮分批止盈（50%）；同时它的止损也会上移
    pos_b = _position("000002", "乙", entry=20.0, shares=500, stop=19.0, target=22.0)
    trade.save_positions(cfg, [pos_a, pos_b])

    trade.save_account(cfg, trade.Account(
        initial_capital=100000.0,
        cash=50000.0,
        frozen=pos_a.cost + pos_b.cost,  # 2 万，与两笔持仓成本一致
    ))

    quotes = pd.DataFrame(
        [{"code": "000001", "price": 10.6}, {"code": "000002", "price": 22.5}]
    )

    trades_made = monitor.handle_trading(cfg, [], quotes)

    # 本轮只分批止盈 B 一笔（50% = 250股，向下取整到100 = 200股）
    assert len(trades_made) == 1
    assert trades_made[0][0].code == "000002"
    assert trades_made[0][1] == "分批止盈"

    # 持仓里仍有 B（剩余300股）—— 分批止盈不清仓
    positions = trade.load_positions(cfg)
    codes = [p.code for p in positions]
    assert "000001" in codes
    # B 应该还在持仓里（部分平仓后剩余）
    pos_b_remaining = [p for p in positions if p.code == "000002"]
    assert len(pos_b_remaining) == 1
    assert pos_b_remaining[0].shares == 300  # 500 - 200 = 300
    assert pos_b_remaining[0].partial_closed == True

    # A 本轮的移动止损更新要落库，证明幸存持仓携带了本轮更新
    pos_a_updated = [p for p in positions if p.code == "000001"]
    assert len(pos_a_updated) == 1
    assert pos_a_updated[0].stop_loss == pytest.approx(10.6 * 0.98)

    # 现金入账：50000 + 22.5*200 = 54500
    account = trade.load_account(cfg)
    assert account.cash == pytest.approx(54500.0)
    assert account.trade_count == 1
    assert account.total_pnl == pytest.approx(500.0)  # (22.5-20.0)*200

    # 历史仅一笔分批平仓记录
    history = trade.load_history(cfg)
    assert len(history) == 1
    assert history[0].code == "000002"
    assert history[0].exit_reason == "分批止盈"
    assert history[0].pnl == pytest.approx(500.0)
