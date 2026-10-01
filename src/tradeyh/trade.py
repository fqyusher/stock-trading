"""模拟交易：监控确认成型时自动买入，止损/移动止盈时自动卖出。

核心原则是**不引入新的判定逻辑**——入场信号完全走 monitor 的 PROMOTED 决策，
退出条件基于形态检测器提供的关键价位（参考止损/目标价）加上移动止盈规则。

仓位管理用固定风险法：每笔交易风险金额 = 现金 × risk_per_trade，
股数 = risk_amount / (entry_price - stop_loss)，取100整数倍。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import broker
from . import fetch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Account:
    initial_capital: float
    cash: float
    frozen: float = 0.0
    total_pnl: float = 0.0
    trade_count: int = 0

    @property
    def equity(self) -> float:
        return self.cash + self.frozen

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["equity"] = self.equity
        return d


@dataclass
class Position:
    code: str
    name: str
    pattern: str
    pattern_cn: str
    entry_price: float
    shares: int
    cost: float
    stop_loss: float
    highest_price: float
    target_price: float
    entry_time: str
    entry_score: float
    trailing_activated: bool = False
    order_id: str = ""  # 券商订单号
    breakout_level: float = 0.0  # B2 突破的 H1 价格（用于失败突破检测）
    breakout_date: str = ""  # B2 突破 bar 的日期（非开仓日；失败突破窗口从它起算）
    partial_closed: bool = False  # 是否已分批止盈
    # 破底翻 v2 口径的分级目标：TP1 到价分批止盈、TP2 到价清仓。
    # 必须有默认值——`load_positions` 走 `Position(**item)`，磁盘上已有的
    # positions.json 里没有这两个键，缺默认值会直接 TypeError。
    tp1_price: float = 0.0
    tp2_price: float = 0.0

    @property
    def stop_distance_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.entry_price - self.stop_loss) / self.entry_price * 100


@dataclass
class Trade:
    code: str
    name: str
    pattern: str
    pattern_cn: str
    entry_price: float
    shares: int
    cost: float
    stop_loss: float
    target_price: float
    entry_time: str
    entry_score: float
    exit_price: float
    exit_time: str
    exit_reason: str
    pnl: float
    pnl_pct: float
    entry_order_id: str = ""
    exit_order_id: str = ""


# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------


def account_path(cfg) -> Path:
    return Path(cfg.trade.account_path)


def positions_path(cfg) -> Path:
    return Path(cfg.trade.positions_path)


def history_path(cfg) -> Path:
    return Path(cfg.trade.history_path)


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_account(cfg) -> Account:
    path = account_path(cfg)
    if not path.exists():
        capital = float(cfg.trade.initial_capital)
        return Account(initial_capital=capital, cash=capital)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Account(
        initial_capital=data["initial_capital"],
        cash=data["cash"],
        frozen=data.get("frozen", 0.0),
        total_pnl=data.get("total_pnl", 0.0),
        trade_count=data.get("trade_count", 0),
    )


def save_account(cfg, account: Account) -> None:
    _atomic_write(account_path(cfg), account.to_dict())


def load_positions(cfg) -> list[Position]:
    path = positions_path(cfg)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Position(**item) for item in data.get("positions", [])]


def save_positions(cfg, positions: list[Position]) -> None:
    _atomic_write(positions_path(cfg), {"positions": [asdict(p) for p in positions]})


def load_history(cfg) -> list[Trade]:
    path = history_path(cfg)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Trade(**item) for item in data.get("trades", [])]


def save_history(cfg, trades: list[Trade]) -> None:
    _atomic_write(history_path(cfg), {"trades": [asdict(t) for t in trades]})


# ---------------------------------------------------------------------------
# 仓位计算
# ---------------------------------------------------------------------------


def calc_position_size(
    cash: float, entry_price: float, stop_loss: float, cfg
) -> int:
    """固定风险法计算股数。

    risk_amount = cash × risk_per_trade
    shares = risk_amount / (entry_price - stop_loss)
    取100整数倍（A股最小交易单位）
    """
    if stop_loss >= entry_price:
        log.warning("止损价 %.2f >= 入场价 %.2f，无法计算仓位", stop_loss, entry_price)
        return 0

    risk_amount = cash * float(cfg.trade.risk_per_trade)
    stop_distance = entry_price - stop_loss
    shares = int(risk_amount / stop_distance)
    shares = (shares // 100) * 100  # 取100整数倍

    if shares <= 0:
        log.warning("计算股数为0（风险金额 %.2f，止损距离 %.2f）", risk_amount, stop_distance)
        return 0

    cost = shares * entry_price
    if cost > cash:
        shares = int(cash / entry_price / 100) * 100
        if shares <= 0:
            log.warning("现金不足：一手需要 %.2f，可用 %.2f", entry_price * 100, cash)
            return 0

    return shares


# ---------------------------------------------------------------------------
# 开仓
# ---------------------------------------------------------------------------


def open_position(
    cfg,
    code: str,
    name: str,
    pattern: str,
    pattern_cn: str,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    entry_score: float,
    tp1_price: float = 0.0,
    tp2_price: float = 0.0,
) -> Position | None:
    """开仓。返回新建的 Position，资金不足或仓位已满时返回 None。

    `tp1_price` / `tp2_price` 只有 v2 口径会传：TP1 到价分批止盈、TP2 到价清仓。
    旧口径传默认 0，退出逻辑与以前完全一致。
    """
    account = load_account(cfg)
    positions = load_positions(cfg)

    if any(p.code == code for p in positions):
        log.debug("%s 已持仓，跳过", code)
        return None

    max_pos = int(cfg.trade.max_positions)
    if len(positions) >= max_pos:
        log.info("持仓已满（%d/%d），跳过 %s", len(positions), max_pos, code)
        return None

    shares = calc_position_size(account.cash, entry_price, stop_loss, cfg)
    if shares <= 0:
        return None

    cost = shares * entry_price
    if cost > account.cash:
        log.warning("现金不足：需要 %.2f，可用 %.2f", cost, account.cash)
        return None

    position = Position(
        code=code,
        name=name,
        pattern=pattern,
        pattern_cn=pattern_cn,
        entry_price=entry_price,
        shares=shares,
        cost=cost,
        stop_loss=stop_loss,
        highest_price=entry_price,
        target_price=target_price,
        entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        entry_score=entry_score,
        tp1_price=float(tp1_price or 0.0),
        tp2_price=float(tp2_price or 0.0),
    )

    # 通过券商接口下单
    brk = broker.get_broker(cfg)
    order = brk.buy(code, name, entry_price, shares)
    position.order_id = order.order_id

    if order.status == "rejected":
        log.warning("买入委托被拒绝: %s", order.message)
        return None

    account.cash -= cost
    account.frozen += cost
    save_account(cfg, account)

    positions.append(position)
    save_positions(cfg, positions)

    order_info = f"订单号 {order.order_id}" if order.order_id else "本地模拟"
    log.info(
        "买入 %s %s %d股 @ %.2f  成本%.2f  止损%.2f  目标%.2f  [%s]",
        code, name, shares, entry_price, cost, stop_loss, target_price, order_info,
    )
    return position


# ---------------------------------------------------------------------------
# 平仓
# ---------------------------------------------------------------------------


def close_position(
    cfg, position: Position, exit_price: float, exit_reason: str
) -> Trade:
    """平仓。返回 Trade 记录。"""
    account = load_account(cfg)
    positions = load_positions(cfg)

    # 通过券商接口下单卖出
    brk = broker.get_broker(cfg)
    order = brk.sell(position.code, position.name, exit_price, position.shares)

    pnl = (exit_price - position.entry_price) * position.shares
    pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100

    trade_record = Trade(
        code=position.code,
        name=position.name,
        pattern=position.pattern,
        pattern_cn=position.pattern_cn,
        entry_price=position.entry_price,
        shares=position.shares,
        cost=position.cost,
        stop_loss=position.stop_loss,
        target_price=position.target_price,
        entry_time=position.entry_time,
        entry_score=position.entry_score,
        exit_price=exit_price,
        exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        exit_reason=exit_reason,
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 2),
        entry_order_id=position.order_id,
        exit_order_id=order.order_id,
    )

    account.cash += exit_price * position.shares
    account.frozen -= position.cost
    account.total_pnl += pnl
    account.trade_count += 1
    save_account(cfg, account)

    positions = [p for p in positions if p.code != position.code]
    save_positions(cfg, positions)

    history = load_history(cfg)
    history.append(trade_record)
    save_history(cfg, history)

    emoji = "+" if pnl >= 0 else ""
    order_info = f"订单号 {order.order_id}" if order.order_id else "本地模拟"
    log.info(
        "卖出 %s %s %d股 @ %.2f  %s  盈亏%s%.2f (%.2f%%)  [%s]",
        position.code, position.name, position.shares, exit_price,
        exit_reason, emoji, pnl, pnl_pct, order_info,
    )
    return trade_record


# ---------------------------------------------------------------------------
# 分批止盈
# ---------------------------------------------------------------------------


def partial_close_position(
    cfg, position: Position, exit_price: float, shares_to_sell: int, reason: str
) -> Trade:
    """分批止盈：卖出部分仓位，更新剩余。返回 Trade 记录。"""
    account = load_account(cfg)
    positions = load_positions(cfg)

    brk = broker.get_broker(cfg)
    order = brk.sell(position.code, position.name, exit_price, shares_to_sell)

    pnl = (exit_price - position.entry_price) * shares_to_sell
    pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100

    trade_record = Trade(
        code=position.code,
        name=position.name,
        pattern=position.pattern,
        pattern_cn=position.pattern_cn,
        entry_price=position.entry_price,
        shares=shares_to_sell,
        cost=shares_to_sell * position.entry_price,
        stop_loss=position.stop_loss,
        target_price=position.target_price,
        entry_time=position.entry_time,
        entry_score=position.entry_score,
        exit_price=exit_price,
        exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        exit_reason=reason,
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 2),
        entry_order_id=position.order_id,
        exit_order_id=order.order_id,
    )

    account.cash += exit_price * shares_to_sell
    account.frozen -= position.entry_price * shares_to_sell
    account.total_pnl += pnl
    account.trade_count += 1
    save_account(cfg, account)

    position.shares -= shares_to_sell
    position.cost = position.shares * position.entry_price
    position.partial_closed = True

    positions = [p if p.code != position.code else position for p in positions]
    save_positions(cfg, positions)

    history = load_history(cfg)
    history.append(trade_record)
    save_history(cfg, history)

    emoji = "+" if pnl >= 0 else ""
    order_info = f"订单号 {order.order_id}" if order.order_id else "本地模拟"
    log.info(
        "分批卖出 %s %s %d股 @ %.2f  %s  盈亏%s%.2f (%.2f%%)  剩余%d股  [%s]",
        position.code, position.name, shares_to_sell, exit_price,
        reason, emoji, pnl, pnl_pct, position.shares, order_info,
    )
    return trade_record


def update_trailing_stop(
    position: Position, current_price: float, cfg,
    ma20: float | None = None, recent_hl: float | None = None,
    trail_ref: float | None = None,
) -> None:
    """更新移动止损。原地修改 position。

    规则：
    0. `trail_ref` 给定时（v2 口径持仓）：止损直接跟随结构位
       `max(MA20 − k×ATR, 最近SwingLow − k×ATR)`，只上移不下移。
       这条**替代**下面的百分比回撤规则——形态的失效位是结构给定的，
       与"从最高价回撤 trail_pct"无关。
    1. 涨 >= breakeven_pct → 止损上移到成本价（保本）
    2. 涨 >= trail_trigger_pct → 启动追踪止损
    3. 追踪止损 = highest_price × (1 - trail_pct)
    4. 分批止盈后 → 用 MA20 / 最近有效 Higher Low 作为移动止盈
    """
    if current_price > position.highest_price:
        position.highest_price = current_price

    if trail_ref and trail_ref > 0:
        if trail_ref > position.stop_loss:
            position.stop_loss = trail_ref
        return

    gain_pct = (current_price - position.entry_price) / position.entry_price

    breakeven_pct = float(cfg.trade.breakeven_pct)
    trail_trigger = float(cfg.trade.trail_trigger_pct)
    trail_pct = float(cfg.trade.trail_pct)

    if position.partial_closed and getattr(cfg.trade, "trail_use_ma_hl", True):
        candidates = [v for v in (ma20, recent_hl) if v and v > 0]
        if candidates:
            new_stop = max(candidates)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
        return

    if gain_pct >= trail_trigger:
        position.trailing_activated = True
        new_stop = position.highest_price * (1 - trail_pct)
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
    elif gain_pct >= breakeven_pct:
        if position.stop_loss < position.entry_price:
            position.stop_loss = position.entry_price


# ---------------------------------------------------------------------------
# 退出检查
# ---------------------------------------------------------------------------


def _is_today_entry(position: Position) -> bool:
    """判断是否为当日买入（T+1：当日买入不可卖出）。"""
    entry_date = position.entry_time[:10]
    return entry_date == datetime.now().strftime("%Y-%m-%d")


def check_exit(
    position: Position, current_price: float, cfg,
    trading_day_count: int | None = None,
    ma20: float | None = None, recent_hl: float | None = None,
    trail_ref: float | None = None,
) -> str | None:
    """检查是否应该退出。返回退出原因，None 表示继续持有。

    T+1 规则：当日买入的股票不能当日卖出，只能从下一个交易日起卖出。
    即使触发止损/目标价，当日买入的仓位也不会退出。
    """
    update_trailing_stop(
        position, current_price, cfg, ma20=ma20, recent_hl=recent_hl, trail_ref=trail_ref
    )

    if _is_today_entry(position):
        return None

    if current_price <= position.stop_loss:
        if position.stop_loss >= position.entry_price:
            return "止盈"
        return "止损"

    # B2 失败突破：突破 bar 后 N 个交易日内跌回 H1 下方
    if (
        position.breakout_level > 0
        and position.breakout_date
        and trading_day_count is not None
    ):
        if trading_day_count <= int(getattr(cfg.trade, "failed_breakout_bars", 3)):
            if current_price < position.breakout_level:
                return "失败突破"

    # v2 口径的第一目标 TP1：到价先卖一半，剩下的交给移动止盈。
    # 必须排在下面那条按 target_price 的"分批止盈"之前，且复用同一个 reason 字面量——
    # monitor 的 handle_trading 只认"分批止盈"，新造一个词会掉进"整仓清掉"的分支。
    if (
        position.tp1_price > 0
        and not position.partial_closed
        and current_price >= position.tp1_price
    ):
        return "分批止盈"

    # 分批止盈：到达目标价时卖出部分
    if not position.partial_closed and position.target_price > 0:
        if current_price >= position.target_price:
            return "分批止盈"

    if position.target_price > 0 and current_price >= position.target_price:
        return "目标价"

    return None


# ---------------------------------------------------------------------------
# 显示
# ---------------------------------------------------------------------------


def describe_account(cfg) -> str:
    account = load_account(cfg)
    positions = load_positions(cfg)
    history = load_history(cfg)

    price_map = {}
    if positions:
        codes = [p.code for p in positions]
        result_box: list = [None]
        exc_box: list = [None]

        def _fetch() -> None:
            try:
                result_box[0] = fetch.fetch_quotes(codes, cfg)
            except Exception as exc:
                exc_box[0] = exc

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=5)
        if t.is_alive():
            log.warning("获取实时行情超时，显示成本价")
        elif exc_box[0] is not None:
            log.warning("获取实时行情失败，显示成本价: %s", exc_box[0])
        elif result_box[0] is not None:
            quotes = result_box[0]
            price_map = {row["code"]: row["price"] for _, row in quotes.iterrows()}

    market_value = 0.0
    unrealized_pnl = 0.0
    if positions and price_map:
        for p in positions:
            price = price_map.get(p.code)
            if price:
                mv = price * p.shares
                market_value += mv
                unrealized_pnl += (price - p.entry_price) * p.shares
    if not market_value:
        market_value = account.frozen

    equity = account.cash + market_value
    total_pnl = equity - account.initial_capital
    total_trades = account.trade_count + len(positions)

    lines = [
        "=" * 60,
        "账户状态",
        "=" * 60,
        f"初始资金    {account.initial_capital:>12.2f}",
        f"可用现金    {account.cash:>12.2f}",
        f"持仓市值    {market_value:>12.2f}",
        f"账户净值    {equity:>12.2f}",
        f"累计盈亏    {total_pnl:>+12.2f}",
        f"交易次数    {total_trades:>12d}",
    ]

    if positions:
        lines.append("")
        lines.append(f"当前持仓 ({len(positions)} 只)")
        lines.append("-" * 60)
        lines.append(
            f"{'代码':<8}{'名称':<8}{'形态':<6}{'股数':>6}{'成本':>8}"
            f"{'现价':>8}{'止损':>8}{'目标':>8}{'盈亏%':>7}"
        )
        for p in positions:
            current_price = price_map.get(p.code)
            if current_price:
                pnl_pct = (current_price - p.entry_price) / p.entry_price * 100
                price_str = f"{current_price:>8.2f}"
                pnl_str = f"{pnl_pct:>+6.1f}%"
            else:
                price_str = f"{'--':>8}"
                pnl_str = f"{'--':>7}"
            lines.append(
                f"{p.code:<8}{p.name:<8}{p.pattern_cn:<6}{p.shares:>6}"
                f"{p.entry_price:>8.2f}{price_str}{p.stop_loss:>8.2f}"
                f"{p.target_price:>8.2f}{pnl_str}"
            )
    else:
        lines.append("\n当前无持仓")

    if history:
        lines.append("")
        lines.append(f"最近交易 (共 {len(history)} 笔)")
        lines.append("-" * 60)
        for t in history[-10:]:
            emoji = "+" if t.pnl >= 0 else ""
            lines.append(
                f"{t.exit_time[:10]} {t.code} {t.name:<8} "
                f"{t.exit_reason:<4} {t.entry_price:.2f}→{t.exit_price:.2f} "
                f"{emoji}{t.pnl:.2f} ({emoji}{t.pnl_pct:.1f}%)"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


def describe_history(cfg, limit: int = 20) -> str:
    history = load_history(cfg)
    if not history:
        return "暂无交易记录"

    lines = [
        f"交易记录 (最近 {min(limit, len(history))} / 共 {len(history)} 笔)",
        "-" * 80,
        f"{'退出时间':<12}{'代码':<8}{'名称':<8}{'形态':<6}"
        f"{'入场':>7}{'退出':>7}{'股数':>6}{'盈亏':>9}{'盈亏%':>7}{'原因':<6}",
    ]

    for t in history[-limit:]:
        emoji = "+" if t.pnl >= 0 else ""
        lines.append(
            f"{t.exit_time[:10]:<12}{t.code:<8}{t.name:<8}{t.pattern_cn:<6}"
            f"{t.entry_price:>7.2f}{t.exit_price:>7.2f}{t.shares:>6}"
            f"{emoji}{t.pnl:>8.2f}{emoji}{t.pnl_pct:>6.1f}% {t.exit_reason:<6}"
        )

    total_pnl = sum(t.pnl for t in history)
    win_count = sum(1 for t in history if t.pnl > 0)
    lines.append("-" * 80)
    lines.append(f"累计盈亏: {total_pnl:+.2f}  胜率: {win_count}/{len(history)}")
    return "\n".join(lines)


def reset_account(cfg) -> None:
    capital = float(cfg.trade.initial_capital)
    account = Account(initial_capital=capital, cash=capital)
    save_account(cfg, account)
    save_positions(cfg, [])
    save_history(cfg, [])
    log.info("账户已重置：初始资金 %.2f", capital)
