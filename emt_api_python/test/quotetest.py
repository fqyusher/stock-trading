#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

from vnemtquote import *

def printFuncName(*args):
    """"""
    print('*' * 50)
    print(args)
    print('*' * 50)

### 回调函数
class QuoteApi(EMTQuoteApi):
    def __init__(self):
        super(QuoteApi, self).__init__()

    def onDisconnected(self, reason):
        """"""
        printFuncName('onDisconnected', reason)

    def onQueryAllTickersFullInfo(self, ticker_info, error, is_last):
        # """"""
        printFuncName("onQueryAllTickersFullInfo")
        if error is not None:
            print("error_id: ", error["error_id"])
            print("error_msg: ", error["error_msg"])
        print("exchange_id: ", ticker_info["exchange_id"])
        print("ticker: ", ticker_info["ticker"])
        print("ticker_name: ", ticker_info["ticker_name"])
        print("security_type: ", ticker_info["security_type"])
        print("ticker_qualification_class: ", ticker_info["ticker_qualification_class"])
        print("is_VIE: ", ticker_info["is_VIE"])
        print("is_noprofit: ", ticker_info["is_noprofit"])
        print("is_weighted_voting_rights: ", ticker_info["is_weighted_voting_rights"])
        print("is_have_price_limit: ", ticker_info["is_have_price_limit"])
        print("upper_limit_price: ", ticker_info["upper_limit_price"])
        print("lower_limit_price: ", ticker_info["lower_limit_price"])
        print("pre_close_price: ", ticker_info["pre_close_price"])
        print("price_tick: ", ticker_info["price_tick"])
        print("bid_qty_upper_limit: ", ticker_info["bid_qty_upper_limit"])
        print("bid_qty_lower_limit: ", ticker_info["bid_qty_lower_limit"])
        print("bid_qty_unit: ", ticker_info["bid_qty_unit"])
        print("ask_qty_upper_limit: ", ticker_info["ask_qty_upper_limit"])
        print("ask_qty_lower_limit: ", ticker_info["ask_qty_lower_limit"])
        print("ask_qty_unit: ", ticker_info["ask_qty_unit"])
        print("market_bid_qty_upper_limit: ", ticker_info["market_bid_qty_upper_limit"])
        print("market_bid_qty_lower_limit: ", ticker_info["market_bid_qty_lower_limit"])
        print("market_bid_qty_unit: ", ticker_info["market_bid_qty_unit"])
        print("market_ask_qty_upper_limit: ", ticker_info["market_ask_qty_upper_limit"])
        print("market_ask_qty_lower_limit: ", ticker_info["market_ask_qty_lower_limit"])
        print("market_ask_qty_unit: ", ticker_info["market_ask_qty_unit"])

    def onQueryTickersPriceInfo(self, price_info, error, is_last):
        printFuncName("onQueryAllTickersPriceInfo")
        print("exchange_type: ", price_info["exchange_type"])
        print("ticker: ", price_info["ticker"])
        print("last_price: ", price_info["last_price"])

    def onSubscribeAllMarketData(self, exchange_id, error):
        printFuncName('onSubscribeAllMarketData', exchange_id, error)

    def onUnSubscribeAllMarketData(self, exchange_id, error):
        printFuncName('onUnSubscribeAllMarketData', exchange_id, error)

    def onSubMarketData(self, ticker_list, error, last):
        printFuncName("onSubMarketData", ticker_list, error, last)

    def onUnSubMarketData(self, ticker_list, error, last):
        printFuncName("onUnSubMarketData", ticker_list, error, last)

    def onSubscribeAllTickByTick(self, exchange_id, error):
        printFuncName("onSubscribeAllTickByTick", exchange_id, error)

    def onUnSubscribeAllTickByTick(self, exchange_id, error):
        printFuncName("onUnSubscribeAllTickByTick", exchange_id, error)

    def onSubTickByTick(self, ticker, error, last):
        printFuncName("onSubTickByTick", ticker, error, last)

    def onUnSubTickByTick(self, ticker, error, last):
        printFuncName("onUnSubTickByTick", ticker, error, last)

    def onSubscribeAllIndexData(self, exchange_id, error):
        printFuncName("onSubscribeAllIndexData", exchange_id, error)

    def onUnSubscribeAllIndexData(self, exchange_id, error):
        printFuncName("onUnSubscribeAllIndexData", exchange_id, error)

    def onSubIndexData(self, ticker, error, last):
        printFuncName("onSubIndexData", ticker, error, last)

    def onUnSubIndexData(self, ticker, error, last):
        printFuncName("onUnSubIndexData", ticker, error, last)

    def onSubscribeAllMinuteInfo(self, exchange_id, error):
        printFuncName("onSubscribeAllMinuteInfo", exchange_id, error)

    def onUnSubscribeAllMinuteInfo(self, exchange_id, error):
        printFuncName("onUnSubscribeAllMinuteInfo", exchange_id, error)

    def onSubMinuteInfo(self, ticker, error, last):
        printFuncName("onSubMinuteInfo", ticker, error, last)

    def onUnSubMinuteInfo(self, ticker, error, last):
        printFuncName("onUnSubMinuteInfo", ticker, error, last)

    def onQueryMinuteInfo(self, min_info, error, last):
        printFuncName("onQueryMinuteInfo", error, last)
        if min_info is not None:
            print("exchange_type: ", min_info["exchange_type"])
            print("ticker_type: ", min_info["ticker_type"])
            print("ticker: ", min_info["ticker"])
            print("data_time: ", min_info["data_time"])
            print("last_price: ", min_info["last_price"])
            print("volume_trade: ", min_info["volume_trade"])
            print("value_trade: ", min_info["value_trade"])
            print("avg_price: ", min_info["avg_price"])

    def onQueryMinHistoryInfo(self, his_min_info, error, last):
        printFuncName("onQueryMinHistoryInfo", error, last)
        if his_min_info is not None:
            print("exchange_type: ", his_min_info["exchange_type"])
            print("ticker_type: ", his_min_info["ticker_type"])
            print("ticker: ", his_min_info["ticker"])
            print("data_time: ", his_min_info["data_time"])
            print("last_price: ", his_min_info["last_price"])
            print("volume_trade: ", his_min_info["volume_trade"])
            print("value_trade: ", his_min_info["value_trade"])
            print("avg_price: ", his_min_info["avg_price"])

    def onQueryLatestIndexData(self, index_data, error, last):
        printFuncName("onQueryLatestIndexData", error, last)
        if index_data is not None:
            print("exchange_id: ", index_data["exchange_id"])
            print("ticker: ", index_data["ticker"])
            print("data_time: ", index_data["data_time"])
            print("open_price: ", index_data["open_price"])
            print("last_price: ", index_data["last_price"])
            print("high_price: ", index_data["high_price"])
            print("low_price: ", index_data["low_price"])
            print("qty: ", index_data["qty"])
            print("turnover: ", index_data["turnover"])

    def onQueryLatestMarketData(self, market_data, error, last):
        printFuncName("onQueryLatestMarketData", error, last)
        if market_data is not None:
            print("exchange_id: ", market_data["exchange_id"])
            print("ticker: ", market_data["ticker"])
            print("last_price: ", market_data["last_price"])
            print("pre_close_price: ", market_data["pre_close_price"])
            print("open_price: ", market_data["open_price"])
            print("high_price: ", market_data["high_price"])
            print("low_price: ", market_data["low_price"])
            print("close_price: ", market_data["close_price"])
            print("upper_limit_price: ", market_data["upper_limit_price"])
            print("lower_limit_price: ", market_data["lower_limit_price"])
            print("data_time: ", market_data["data_time"])
            print("qty: ", market_data["qty"])
            print("bid: ", market_data["bid"])
            print("ask: ", market_data["ask"])
            print("bid_qty: ", market_data["bid_qty"])
            print("ask_qty: ", market_data["ask_qty"])
            print("turnover: ", market_data["turnover"])
            print("avg_price: ", market_data["avg_price"])
            print("trades_count: ", market_data["trades_count"])
            print("ticker_status: ", market_data["ticker_status"])
            print("total_bid_qty: ", market_data["total_bid_qty"])
            print("total_ask_qty: ", market_data["total_ask_qty"])
            print("ma_bid_price: ", market_data["ma_bid_price"])
            print("ma_ask_price: ", market_data["ma_ask_price"])
            print("cancel_buy_count: ", market_data["cancel_buy_count"])
            print("cancel_sell_count: ", market_data["cancel_sell_count"])
            print("cancel_buy_qty: ", market_data["cancel_buy_qty"])
            print("cancel_sell_qty: ", market_data["cancel_sell_qty"])
            print("cancel_buy_money: ", market_data["cancel_buy_money"])
            print("cancel_sell_money: ", market_data["cancel_sell_money"])
            print("total_buy_count: ", market_data["total_buy_count"])
            print("total_sell_count: ", market_data["total_sell_count"])
            print("duration_after_buy: ", market_data["duration_after_buy"])
            print("duration_after_sell: ", market_data["duration_after_sell"])
            print("num_bid_orders: ", market_data["num_bid_orders"])
            print("num_ask_orders: ", market_data["num_ask_orders"])
            print("data_type: ", market_data["data_type"])
            if market_data["data_type"] == 3:
                print("ma_bond_bid_price: ", market_data["ma_bond_bid_price"])
                print("ma_bond_ask_price: ", market_data["ma_bond_ask_price"])
                print("yield_to_maturity: ", market_data["yield_to_maturity"])
            elif market_data["data_type"] == 2:
                print("iopv: ", market_data["iopv"])
                print("etf_buy_count: ", market_data["etf_buy_count"])
                print("etf_sell_count: ", market_data["etf_sell_count"])
                print("etf_buy_qty: ", market_data["etf_buy_qty"])
                print("etf_buy_money: ", market_data["etf_buy_money"])
                print("etf_sell_qty: ", market_data["etf_sell_qty"])
                print("etf_sell_money: ", market_data["etf_sell_money"])
                print("pre_iopv: ", market_data["pre_iopv"])
            elif market_data["data_type"] == 7:
                print("total_warrant_exec_qty: ", market_data["total_warrant_exec_qty"])
                print("warrant_lower_price: ", market_data["warrant_lower_price"])
                print("warrant_upper_price: ", market_data["warrant_upper_price"])
            elif market_data["data_type"] == 4:
                print("auction_price: ", market_data["auction_price"])
                print("auction_qty: ", market_data["auction_qty"])
                print("last_enquiry_time: ", market_data["last_enquiry_time"])
                print("pre_total_long_positon: ", market_data["pre_total_long_positon"])
                print("total_long_positon: ", market_data["total_long_positon"])
                print("pre_settl_price: ", market_data["pre_settl_price"])
                print("settl_price: ", market_data["settl_price"])

    def onQueryAllTickers(self, data, error, last):
        printFuncName("onQueryAllTickers", error, last)
        if data is not None:
            print("exchange_type: ", data["exchange_id"])
            print("ticker: ", data["ticker"])
            print("ticker_name: ", data["ticker_name"])
            print("ticker_type: ", data["ticker_type"])
            print("pre_close_price: ", data["pre_close_price"])
            print("lower_limit_price: ", data["lower_limit_price"])
            print("price_tick: ", data["price_tick"])
            print("buy_qty_unit: ", data["buy_qty_unit"])
            print("sell_qty_unit: ", data["sell_qty_unit"])
            print("is_last", last)

    def onMinuteInfo(self, min_data):
        printFuncName("onMinuteInfo")
        print("exchange_type: ", min_data["exchange_type"])
        print("ticker_type: ", min_data["ticker_type"])
        print("ticker: ", min_data["ticker"])
        print("data_time: ", min_data["data_time"])
        print("last_price: ", min_data["last_price"])
        print("volume_trade: ", min_data["volume_trade"])
        print("value_trade: ", min_data["value_trade"])
        print("avg_price: ", min_data["avg_price"])

    def onTickByTick(self, tbt_data):
        printFuncName("onTickByTick")
        if tbt_data is not None:
            print("data_time: ", tbt_data["data_time"])
            print("seq: ", tbt_data["seq"])
            print("exchange_id: ", tbt_data["exchange_id"])
            print("ticker: ", tbt_data["ticker"])
            print("type: ", tbt_data["type"])
            if tbt_data["type"] == 1:
                print("channel_no: ", tbt_data["channel_no"])
                print("side: ", tbt_data["side"])
                print("ord_type: ", tbt_data["ord_type"])
                print("seq: ", tbt_data["seq"])
                print("price: ", tbt_data["price"])
                print("qty: ", tbt_data["qty"])
            else:
                print("channel_no: ", tbt_data["channel_no"])
                print("trade_flag: ", tbt_data["trade_flag"])
                print("seq: ", tbt_data["seq"])
                print("bid_no: ", tbt_data["bid_no"])
                print("ask_no: ", tbt_data["ask_no"])
                print("price: ", tbt_data["price"])
                print("qty: ", tbt_data["qty"])
                print("money: ", tbt_data["money"])

    def onIndexData(self, index_data):
        printFuncName("onIndexData")
        print("exchange_id: ", index_data["exchange_id"])
        print("datatime: ", index_data["data_time"])
        print("ticker: ", index_data["ticker"])
        print("pre_close_price: ", index_data["pre_close_price"])
        print("open_price: ", index_data["open_price"])
        print("last_price: ", index_data["last_price"])
        print("high_price: ", index_data["high_price"])
        print("low_price: ", index_data["low_price"])
        print("qty: ", index_data["qty"])
        print("turnover: ", index_data["turnover"])

    def onDepthMarketData(self, market_data, bid1_qty_list, bid1_count, max_bid1_count, ask1_qty_list, ask1_count,
                          max_ask1_count):
        printFuncName("onDepthMarketData")
        print("ticker: ", market_data["ticker"])
        print("exchange_id: ", market_data["exchange_id"])
        print("pre_close_price: ", market_data["pre_close_price"])
        print("open_price: ", market_data["open_price"])
        print("high_price: ", market_data["high_price"])
        print("low_price: ", market_data["low_price"])
        print("close_price: ", market_data["close_price"])
        print("upper_limit_price: ", market_data["upper_limit_price"])
        print("lower_limit_price: ", market_data["lower_limit_price"])
        print("data_time: ", market_data["data_time"])
        print("qty: ", market_data["qty"])
        print("turnover: ", market_data["turnover"])
        print("avg_price: ", market_data["avg_price"])
        print("bid: ", market_data["bid"])
        print("ask: ", market_data["ask"])
        print("bid_qty: ", market_data["bid_qty"])
        print("ask_qty: ", market_data["ask_qty"])
        print("ticker_status: ", market_data["ticker_status"])
        print("trades_count: ", market_data["trades_count"])
        print("total_bid_qty: ", market_data["total_bid_qty"])
        print("total_ask_qty: ", market_data["total_ask_qty"])
        print("ma_bid_price: ", market_data["ma_bid_price"])
        print("ma_ask_price: ", market_data["ma_ask_price"])
        print("cancel_buy_count: ", market_data["cancel_buy_count"])
        print("cancel_sell_count: ", market_data["cancel_sell_count"])
        print("cancel_buy_money: ", market_data["cancel_buy_money"])
        print("cancel_sell_money: ", market_data["cancel_sell_money"])
        print("total_buy_count: ", market_data["total_buy_count"])
        print("total_sell_count: ", market_data["total_sell_count"])
        print("duration_after_buy: ", market_data["duration_after_buy"])
        print("duration_after_sell: ", market_data["duration_after_sell"])
        print("data_type: ", market_data["data_type"])
        if market_data["data_type"] == 2:
            print("iopv: ", market_data["iopv"])
            print("etf_buy_count: ", market_data["etf_buy_count"])
            print("etf_sell_count: ", market_data["etf_sell_count"])
            print("etf_buy_qty: ", market_data["etf_buy_qty"])
            print("etf_buy_money: ", market_data["etf_buy_money"])
            print("etf_sell_qty: ", market_data["etf_sell_qty"])
            print("etf_sell_money: ", market_data["etf_sell_money"])
            print("pre_iopv: ", market_data["pre_iopv"])
        elif market_data["data_type"] == 3:
            print("ma_bond_bid_price: ", market_data["ma_bond_bid_price"])
            print("ma_bond_ask_price: ", market_data["ma_bond_ask_price"])
            print("yield_to_maturity: ", market_data["yield_to_maturity"])
        elif market_data["data_type"] == 7:
            print("total_warrant_exec_qty: ", market_data["total_warrant_exec_qty"])
            print("warrant_lower_price: ", market_data["warrant_lower_price"])
            print("warrant_upper_price: ", market_data["warrant_upper_price"])
        elif market_data["data_type"] == 4:
            print("auction_price: ", market_data["auction_price"])
            print("auction_qty: ", market_data["auction_qty"])
            print("last_enquiry_time: ", market_data["last_enquiry_time"])
            print("pre_total_long_positon: ", market_data["pre_total_long_positon"])
            print("total_long_positon: ", market_data["total_long_positon"])
            print("pre_settl_price: ", market_data["pre_settl_price"])
            print("settl_price: ", market_data["settl_price"])


# 使用时请挑选所需部分，其余部分可删除或注释
if __name__ == '__main__':

    lev1_ip = "10.10.89.140"
    lev1_port = 9300
    username = "hq_writer"
    password = "hq_writer"

    # 网络协议
    # 1：TCP
    # 2：UDP
    protocol_type = 1
    # 本地接收UDP消息的网卡IP（如有需要自行修改）
    local_udp_ip = "127.0.0.1"
    
    api = QuoteApi()
    ### 创建 API
    # param1：Client_ID
    # param2：日志文件所在目录（自行指定）
    # param3：行情类型
    #       1: 表示沪深L1行情
    #       2: 表示沪深L2行情
    # param4：日志输出等级
    #       0：FATAL
    #       1：ERROR
    #       2：WARNING
    #       3：INFO
    #       4：DEBUG
    #       5：TRACE
    api.createPythonQuoteApi(101, "logs", 1, 3)
    api.setHeartBeatInterval(5)
    version = api.getApiVersion()
    print("version: ", version)
    login_res = api.login(lev1_ip, lev1_port, username, password, protocol_type, local_udp_ip)
    print("login_res: ", login_res)
    if login_res != 0:
        exit()
    # 沪深交易所 id
    # 1：上交所
    # 2：深交所
    sh_exchange_id = 1
    sz_exchange_id = 2

    ### 查询所有证券代码基础信息
    # param：交易所
    # api.queryAllTickers(sh_exchange_id)
    # api.queryAllTickers(sz_exchange_id)
    ### 查询所有证券代码全量信息
    # param：交易所
    api.queryAllTickersFullInfo(sh_exchange_id)
    #api.queryAllTickersFullInfo(sz_exchange_id)

    # ### 订阅全市场快照行情
    # # param：交易所
    # api.subscribeAllMarketData(sh_exchange_id)
    # api.subscribeAllMarketData(sz_exchange_id)

    # # ### 订阅全市场指数行情
    # # # param：交易所
    # api.subscribeAllIndexData(sh_exchange_id)
    # api.subscribeAllIndexData(sz_exchange_id)

    # time.sleep(120)

    # # ### 退订全市场快照行情
    # # # param：交易所
    # api.unsubscribeAllMarketData(sh_exchange_id)
    # api.unsubscribeAllMarketData(sz_exchange_id)

    # # ### 退订全市场指数行情
    # # # param：交易所
    # api.unsubscribeAllIndexData(sh_exchange_id)
    # api.unsubscribeAllIndexData(sz_exchange_id)

    # time.sleep(10)

    ## 沪市指数
    sub_sh_index = [{'ticker': '000001'}]
    unsub_sh_index = [{'ticker':'000001'}]
    ## 沪市股票
    sub_sh_stock = [{'ticker': '600071'}]
    unsub_sh_stock = [{'ticker': '600071'}]
    ## 沪市基金
    sub_sh_fund = [{'ticker': '501000'}]
    unsub_sh_fund = [{'ticker': '501000'}]
    ## 沪市期权
    sub_sh_opt = [{'ticker': '10004237'}]
    unsub_sh_opt = [{'ticker': '10004237'}]
    ## 沪市债券
    sub_sh_bond = [{'ticker': '010303'}]
    unsub_sh_bond = [{'ticker': '010303'}]

    ## 深市指数
    sub_sz_index = [{'ticker': '399001'}]
    unsub_sz_index = [{'ticker':'399001'}]
    ## 深市股票
    sub_sz_stock = [{'ticker': '300059'}]
    unsub_sz_stock = [{'ticker': '300059'}]
    ## 深市基金
    sub_sz_fund = [{'ticker': '159001'}]
    unsub_sz_fund = [{'ticker': '159001'}]
    ## 深市期权
    sub_sz_opt = [{'ticker': '90001282'}, {'ticker': '90001283'}]
    unsub_sz_opt = [{'ticker': '90001282'}, {'ticker': '90001283'}]
    ## 深市债券
    sub_sz_bond = [{'ticker': '100303'}]
    unsub_sz_bond = [{'ticker': '100303'}]

    # ### 查询最新价（仅 L1 支持）
    # # param1: 代码数组
    # # param2: 股票数量
    # # param3: 交易所
    # api.queryTickersPriceInfo(sub_sh_stock, 1, sh_exchange_id)
    # api.queryTickersPriceInfo(sub_sz_stock, 1, sz_exchange_id)

    # api.queryTickersPriceInfo(sub_sh_index, 1, sh_exchange_id)
    # api.queryTickersPriceInfo(sub_sz_index, 1, sz_exchange_id)

    # api.queryTickersPriceInfo(sub_sh_fund, 1, sh_exchange_id)
    # api.queryTickersPriceInfo(sub_sz_fund, 1, sz_exchange_id)

    # api.queryTickersPriceInfo(sub_sh_bond, 1, sh_exchange_id)
    # api.queryTickersPriceInfo(sub_sz_bond, 1, sz_exchange_id)

    # api.queryTickersPriceInfo(sub_sh_opt, 1, sh_exchange_id)
    # api.queryTickersPriceInfo(sub_sz_opt, 1, sz_exchange_id)

    # ### 查询最新行情（仅 L1 支持）
    # # param1: 代码数组
    # # param2: 股票数量
    # # param3: 代码类型
    # #       0: 股票
    # #       1: 指数
    # #       2: 基金
    # #       3: 债券
    # #       4: 期权
    # # param4: 交易所xxxxx
    # api.queryLatestInfo(sub_sh_stock, 1, 0, sh_exchange_id)
    # api.queryLatestInfo(sub_sz_stock, 1, 0, sz_exchange_id)

    # api.queryLatestInfo(sub_sh_index, 1, 1, sh_exchange_id)
    # api.queryLatestInfo(sub_sz_index, 1, 1, sz_exchange_id)

    # api.queryLatestInfo(sub_sh_fund, 1, 2, sh_exchange_id)
    # api.queryLatestInfo(sub_sz_fund, 1, 2, sz_exchange_id)

    # api.queryLatestInfo(sub_sh_bond, 1, 3, sh_exchange_id)
    # api.queryLatestInfo(sub_sz_bond, 1, 3, sz_exchange_id)

    # api.queryLatestInfo(sub_sh_opt, 1, 4, sh_exchange_id)
    # api.queryLatestInfo(sub_sz_opt, 1, 4, sz_exchange_id)

    ### 订阅指定证券代码行情 指数 快照 逐笔
    # # param1：订阅/退订证券代码数组 指数行情
    # # param2：股票数量
    # # param3：交易所
    # api.subscribeIndexData(sub_sh_index, 1, sh_exchange_id)
    # time.sleep(30)
    # api.unsubscribeIndexData(unsub_sh_index, 1, sh_exchange_id)

    # api.subscribeIndexData(sub_sz_index, 1, sz_exchange_id)
    # time.sleep(30)
    # api.unsubscribeIndexData(unsub_sz_index, 1, sz_exchange_id)

    # # param1：订阅/退订证券代码数组 快照行情
    # # param2：股票数量
    # # param3：交易所
    # # 股票
    # api.subscribeMarketData(sub_sh_stock, 1, sh_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sh_stock, 1, sh_exchange_id)

    # api.subscribeMarketData(sub_sz_stock, 1, sz_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sz_stock, 1, sz_exchange_id)
    # # 基金
    # api.subscribeMarketData(sub_sh_fund, 1, sh_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sh_fund, 1, sh_exchange_id)

    # api.subscribeMarketData(sub_sz_fund, 1, sz_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sz_fund, 1, sz_exchange_id)
    # # 债券
    # api.subscribeMarketData(sub_sh_bond, 1, sh_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sh_bond, 1, sh_exchange_id)

    # api.subscribeMarketData(sub_sz_bond, 1, sz_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sz_bond, 1, sz_exchange_id)
    # 期权
    # api.subscribeMarketData(sub_sh_opt, 1, sh_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sh_opt, 1, sh_exchange_id)

    # api.subscribeMarketData(sub_sz_opt, 2, sz_exchange_id)
    # time.sleep(30)
    # api.unsubscribeMarketData(unsub_sz_opt, 2, sz_exchange_id)
    
    # 保持主线程一直运行
    while True:
        time.sleep(3)
        pass

    api.logout()
