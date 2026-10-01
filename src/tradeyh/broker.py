"""券商接口：东财 EMT 极速柜台仿真交易。

通过 vnemttrader Python SDK 连接东财 EMT 仿真环境，实现真实模拟下单。
本地 trade.py 仍保留作为交易记录跟踪，实际订单通过此模块发送到券商。

SDK 位于 emt_api_python/lib/linux/（需加入 sys.path）。
使用前在 config.yaml 的 trade.emt 节配置地址/端口，账号密码放项目根 .env
（TRADEYH_EMT_ACCOUNT_ID / TRADEYH_EMT_PASSWORD，见 config.ENV_OVERRIDES）。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 常量
EMT_MKT_SZ_A = 1   # 深圳A股
EMT_MKT_SH_A = 2   # 上海A股
EMT_SIDE_BUY = 1
EMT_SIDE_SELL = 2
EMT_PRICE_LIMIT = 1  # 限价
EMT_BUSINESS_CASH = 0  # 普通股票


@dataclass
class BrokerOrder:
    order_id: str
    code: str
    name: str
    side: str
    price: float
    shares: int
    status: str  # "submitted", "filled", "rejected", "local"
    message: str = ""


def _setup_sdk_path() -> bool:
    """把 vnemttrader.so 所在目录加入 sys.path。"""
    project_root = Path(__file__).resolve().parents[2]
    sdk_dir = project_root / "emt_api_python" / "lib" / "linux"
    if not sdk_dir.exists():
        sdk_dir = project_root / "emt_api_python" / "lib" / "windows"
    if sdk_dir.exists():
        sdk_str = str(sdk_dir)
        if sdk_str not in sys.path:
            sys.path.insert(0, sdk_str)
        return True
    return False


class _TradeSpi:
    """回调处理器：收集异步回报供同步查询使用。"""

    def __init__(self):
        self.orders: dict[int, dict] = {}
        self.trades: list[dict] = []
        self.positions: list[dict] = []
        self.asset: dict = {}
        self.fund: dict = {}
        self._lock = threading.Lock()
        self._order_event = threading.Event()
        self._query_event = threading.Event()
        self._query_type = ""

    def onDisconnected(self, reason):
        log.warning("EMT 断开连接: reason=%s", reason)

    def onError(self, data):
        log.error("EMT 错误: %s", data)

    def onOrderEvent(self, data, error, session):
        if error and error.get("error_id", 0) != 0:
            log.warning("EMT 报单错误: %s", error)
        with self._lock:
            emt_id = data.get("order_emt_id", 0)
            self.orders[emt_id] = dict(data)
        self._order_event.set()

    def onTradeEvent(self, data, session):
        with self._lock:
            self.trades.append(dict(data))
        log.info("EMT 成交: %s %s %d股 @ %.2f",
                 data.get("ticker"), "买" if data.get("side") == 1 else "卖",
                 data.get("quantity", 0), data.get("price", 0))

    def onCancelOrderError(self, data, error, session):
        log.warning("EMT 撤单失败: %s %s", data, error)

    def onQueryPosition(self, data, error, reqid, last, session):
        if data and data.get("ticker"):
            with self._lock:
                self.positions.append(dict(data))
        if last:
            self._query_event.set()

    def onQueryAsset(self, data, error, reqid, session):
        if data:
            with self._lock:
                self.asset = dict(data)
        self._query_event.set()

    def onQueryFund(self, data, error, reqid, session):
        if data:
            with self._lock:
                self.fund = dict(data)
        self._query_event.set()

    def onQueryOrderByPage(self, data, req_count, order_sequence, query_reference, reqid, last, session):
        if last:
            self._query_event.set()

    def onQueryPositionByPage(self, data, req_count, trade_sequence, query_reference, reqid, last, session):
        if data and data.get("ticker"):
            with self._lock:
                self.positions.append(dict(data))
        if last:
            self._query_event.set()

    # 其余回调空实现，避免基类报错
    def onQueryOrder(self, *a): pass
    def onQueryTrade(self, *a): pass
    def onQueryTradeByPage(self, *a): pass
    def onQueryFundTransfer(self, *a): pass
    def onFundTransfer(self, *a): pass
    def onQueryETF(self, *a): pass
    def onQueryETFBasket(self, *a): pass
    def onQueryIPOInfoList(self, *a): pass
    def onQueryIPOQuotaInfo(self, *a): pass
    def onQueryOptionAuctionInfo(self, *a): pass
    def onQueryStructuredFund(self, *a): pass
    def onCreditCashRepay(self, *a): pass


class EMTBroker:
    """东财 EMT 极速柜台接口封装。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._api = None
        self._spi: _TradeSpi | None = None
        self._session = 0
        self._reqid = 0
        self._connected = False
        self._client_id = 27

    def connect(self) -> bool:
        if self._connected:
            return True

        emt_cfg = self.cfg.trade.get("emt", {})
        if not emt_cfg.get("enabled", False):
            return False

        ip = emt_cfg.get("ip", "")
        port = int(emt_cfg.get("port", 19088))
        user = emt_cfg.get("account_id", "") or emt_cfg.get("user", "")
        password = emt_cfg.get("password", "")
        local_ip = emt_cfg.get("local_ip", "127.0.0.1")

        if not ip or not user:
            log.warning("EMT ip 或 account_id 未配置，使用本地模拟")
            return False

        if not _setup_sdk_path():
            log.warning("EMT SDK 目录不存在，使用本地模拟")
            return False

        try:
            from vnemttrader import TraderApi

            self._spi = _TradeSpi()

            class _Api(TraderApi):
                def __init__(self, spi):
                    super().__init__()
                    self._spi = spi

                def onDisconnected(self, reason):
                    self._spi.onDisconnected(reason)
                def onError(self, data):
                    self._spi.onError(data)
                def onOrderEvent(self, data, error, session):
                    self._spi.onOrderEvent(data, error, session)
                def onTradeEvent(self, data, session):
                    self._spi.onTradeEvent(data, session)
                def onCancelOrderError(self, data, error, session):
                    self._spi.onCancelOrderError(data, error, session)
                def onQueryPosition(self, data, error, reqid, last, session):
                    self._spi.onQueryPosition(data, error, reqid, last, session)
                def onQueryAsset(self, data, error, reqid, session):
                    self._spi.onQueryAsset(data, error, reqid, session)
                def onQueryFund(self, data, error, reqid, session):
                    self._spi.onQueryFund(data, error, reqid, session)
                def onQueryOrderByPage(self, *a):
                    self._spi.onQueryOrderByPage(*a)
                def onQueryPositionByPage(self, *a):
                    self._spi.onQueryPositionByPage(*a)
                def onQueryOrder(self, *a): self._spi.onQueryOrder(*a)
                def onQueryTrade(self, *a): self._spi.onQueryTrade(*a)
                def onQueryTradeByPage(self, *a): self._spi.onQueryTradeByPage(*a)
                def onQueryFundTransfer(self, *a): self._spi.onQueryFundTransfer(*a)
                def onFundTransfer(self, *a): self._spi.onFundTransfer(*a)
                def onQueryETF(self, *a): self._spi.onQueryETF(*a)
                def onQueryETFBasket(self, *a): self._spi.onQueryETFBasket(*a)
                def onQueryIPOInfoList(self, *a): self._spi.onQueryIPOInfoList(*a)
                def onQueryIPOQuotaInfo(self, *a): self._spi.onQueryIPOQuotaInfo(*a)
                def onQueryOptionAuctionInfo(self, *a): self._spi.onQueryOptionAuctionInfo(*a)
                def onQueryStructuredFund(self, *a): self._spi.onQueryStructuredFund(*a)
                def onCreditCashRepay(self, *a): self._spi.onCreditCashRepay(*a)

            log_dir = str(Path(self.cfg.trade.get("account_path", "data/trade")).parent / "emt_log")
            os.makedirs(log_dir, exist_ok=True)

            self._api = _Api(self._spi)
            self._api.createTraderApi(self._client_id, log_dir, 2)  # INFO level
            self._api.subscribePublicTopic(2)  # QUICK
            self._api.setSoftwareVersion("tradeyh")

            self._session = self._api.login(ip, port, user, password, 1, local_ip)
            if self._session == 0:
                err = self._api.getApiLastError()
                log.error("EMT 登录失败: %s", err)
                return False

            self._connected = True
            log.info("EMT 仿真连接成功：账号 %s @ %s:%d", user, ip, port)
            return True

        except Exception as exc:
            log.error("EMT 连接失败: %s", exc)
            return False

    def _next_reqid(self) -> int:
        self._reqid += 1
        return self._reqid

    def _to_market(self, code: str) -> int:
        code = str(code).zfill(6)
        if code.startswith(("6", "9")):
            return EMT_MKT_SH_A
        return EMT_MKT_SZ_A

    def buy(self, code: str, name: str, price: float, shares: int) -> BrokerOrder:
        if not self.connect():
            return self._local_order(code, name, "buy", price, shares)

        try:
            order = {
                "ticker": str(code).zfill(6),
                "market": self._to_market(code),
                "side": EMT_SIDE_BUY,
                "price_type": EMT_PRICE_LIMIT,
                "price": price,
                "quantity": shares,
                "business_type": EMT_BUSINESS_CASH,
                "position_effect": 0,  # 股票不需要开平标志，置0
                "ep_id": 0,
            }
            emt_id = self._api.insertOrder(order, self._session)
            if emt_id == 0:
                err = self._api.getApiLastError()
                msg = f"{err.get('error_id', '')}: {err.get('error_msg', '')}" if err else "未知错误"
                log.error("EMT 买入失败: %s", msg)
                return BrokerOrder("", code, name, "buy", price, shares, "rejected", msg)

            log.info("EMT 买入委托: %s %s %d股 @ %.2f  EMT_ID=%s", code, name, shares, price, emt_id)
            return BrokerOrder(str(emt_id), code, name, "buy", price, shares, "submitted")

        except Exception as exc:
            log.error("EMT 买入异常: %s", exc)
            return BrokerOrder("", code, name, "buy", price, shares, "rejected", str(exc))

    def sell(self, code: str, name: str, price: float, shares: int) -> BrokerOrder:
        if not self.connect():
            return self._local_order(code, name, "sell", price, shares)

        try:
            order = {
                "ticker": str(code).zfill(6),
                "market": self._to_market(code),
                "side": EMT_SIDE_SELL,
                "price_type": EMT_PRICE_LIMIT,
                "price": price,
                "quantity": shares,
                "business_type": EMT_BUSINESS_CASH,
                "position_effect": 0,
                "ep_id": 0,
            }
            emt_id = self._api.insertOrder(order, self._session)
            if emt_id == 0:
                err = self._api.getApiLastError()
                msg = f"{err.get('error_id', '')}: {err.get('error_msg', '')}" if err else "未知错误"
                log.error("EMT 卖出失败: %s", msg)
                return BrokerOrder("", code, name, "sell", price, shares, "rejected", msg)

            log.info("EMT 卖出委托: %s %s %d股 @ %.2f  EMT_ID=%s", code, name, shares, price, emt_id)
            return BrokerOrder(str(emt_id), code, name, "sell", price, shares, "submitted")

        except Exception as exc:
            log.error("EMT 卖出异常: %s", exc)
            return BrokerOrder("", code, name, "sell", price, shares, "rejected", str(exc))

    def get_positions(self) -> list[dict[str, Any]]:
        if not self.connect() or not self._spi:
            return []

        try:
            self._spi.positions = []
            self._spi._query_event.clear()
            self._api.queryPosition("", self._session, self._next_reqid())
            self._spi._query_event.wait(timeout=5)

            result = []
            for pos in self._spi.positions:
                ticker = pos.get("ticker", "")
                result.append({
                    "code": ticker,
                    "name": pos.get("ticker_name", ""),
                    "market": pos.get("market", 0),
                    "shares": pos.get("total_qty", 0),
                    "available": pos.get("sellable_qty", 0),
                    "cost": pos.get("avg_price", 0),
                })
            return result

        except Exception as exc:
            log.error("EMT 查询持仓失败: %s", exc)
            return []

    def get_cash(self) -> float:
        if not self.connect() or not self._spi:
            return 0.0

        try:
            self._spi.fund = {}
            self._spi._query_event.clear()
            self._api.queryFund(self._session, self._next_reqid())
            self._spi._query_event.wait(timeout=5)

            return self._spi.fund.get("buying_power", 0.0)

        except Exception as exc:
            log.error("EMT 查询资金失败: %s", exc)
            return 0.0

    def get_asset(self) -> dict:
        if not self.connect() or not self._spi:
            return {}

        try:
            self._spi.asset = {}
            self._spi._query_event.clear()
            self._api.queryAsset(self._session, self._next_reqid())
            self._spi._query_event.wait(timeout=5)
            return dict(self._spi.asset)

        except Exception as exc:
            log.error("EMT 查询资产失败: %s", exc)
            return {}

    def _local_order(self, code: str, name: str, side: str, price: float, shares: int) -> BrokerOrder:
        return BrokerOrder(
            order_id=f"LOCAL-{code}-{side}",
            code=code, name=name, side=side,
            price=price, shares=shares,
            status="local",
            message="本地模拟（EMT未启用）",
        )


_broker: EMTBroker | None = None


def get_broker(cfg) -> EMTBroker:
    global _broker
    if _broker is None:
        _broker = EMTBroker(cfg)
    return _broker
