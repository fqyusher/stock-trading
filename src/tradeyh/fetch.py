"""多渠道行情抓取：东财 + 同花顺 + 腾讯 + 新浪 + Baostock。

HTTP 渠道按配置顺序依次尝试（cfg.data.sources）：某渠道重试耗尽或明确报错就切换下一家，
全部渠道都拿不到才放弃。拿不到数据（接口正常但返回空）与网络异常区别对待：
前者返回空表交由调用方记入失败清单，后者重试若干次后再切换渠道。
渠道因参数组合不适用（如新浪无复权）时抛 ChannelUnavailable，不重试直接切下一家。

HTTP 层用标准库 urllib（不用 requests）：部分网络/代理环境下 requests 的连接池管理
容易触发连接重置，urllib 短连接更稳，自带代理环境变量支持。

Baostock 为 Python SDK 渠道，需 pip install baostock，适合 HTTP 渠道受限时使用。
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from . import store

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

# 东财 secid 市场前缀：上交所(6/9开头)=1，深交所(0/3，含创业板)与北交所(4/8)=0
_EM_SECID_PREFIX = ("0", "3", "4", "8")
# 腾讯代码前缀
_TX_MARKET = {"6": "sh", "9": "sh", "4": "bj", "8": "bj"}

# 东财 fqt 复权参数
_EM_ADJUST = {"": 0, "qfq": 1, "hfq": 2}
# 腾讯 fqkline 复权段
_TX_ADJUST = {"": "", "qfq": "qfq", "hfq": "hfq"}
# 东财 klt 周期
_EM_KLT = {"daily": 101, "weekly": 102, "monthly": 103}
# 腾讯 fqkline 周期参数（注意不是 daily 而是 day）
_TX_PERIOD = {"daily": "day", "weekly": "week", "monthly": "month"}
# 东财 clist 分页大小：服务端每页上限实测为 100，按 total 翻页取全量
_EM_CLIST_PZ = 100
# clist 接口主机池：部分网络环境会屏蔽个别主机，轮着试（delay 为延时行情，做股票池足够）
_EM_CLIST_HOSTS = (
    "https://push2.eastmoney.com",
    "https://82.push2.eastmoney.com",
    "https://push2delay.eastmoney.com",
)
# K线接口主机池：push2his 是历史K线专用，其他镜像不一定支持 kline 路径
_EM_KLINE_HOSTS = (
    "https://push2his.eastmoney.com",
    "https://push2.eastmoney.com",
    "https://82.push2.eastmoney.com",
)

# 腾讯行情串 v_shXXXXXX="f0~f1~..." 各字段下标（~ 分隔）：
#   1 名称 | 2 代码 | 3 最新价 | 4 昨收 | 5 今开 | 6 成交量(手)
#   32 涨跌幅% | 33 最高 | 34 最低 | 37 成交额(万,备用) | 38 换手率%
#   44 总市值(亿) | 45 流通市值(亿) | 57 成交额(万,优先)
_TX_F_NAME, _TX_F_PRICE, _TX_F_TURNOVER = 1, 3, 38
_TX_F_PREV_CLOSE, _TX_F_OPEN, _TX_F_VOLUME = 4, 5, 6
_TX_F_PCT_CHG, _TX_F_HIGH, _TX_F_LOW = 32, 33, 34
_TX_F_AMOUNT_WAN_ALT, _TX_F_FLOAT_MV_YI, _TX_F_AMOUNT_WAN = 37, 45, 57
_TX_BATCH = 60  # 腾讯单次批量报价的代码数上限
# 东财 ulist 单次可带的 secid 数（观察池通常几十只，一次就够；留余量防 URL 过长）
_EM_QUOTE_BATCH = 100
# 新浪日K接口（参考 a-trade/src/kline.py：极少限流，单请求一次拿完，稳定优先）
_SINA_KLINE = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_SINA_MAX_BARS = 1200  # 单次可回传的最大根数（实测 1200 可用）


class FetchError(RuntimeError):
    """重试耗尽后仍失败。"""


class ChannelUnavailable(FetchError):
    """渠道因参数组合不适用（非网络问题），跳过重试直接切下一家。"""


def _http_get_text(
    url: str,
    params: dict | None = None,
    *,
    timeout: float = 15,
    retries: int = 3,
    backoff: float = 0.4,
    encoding: str = "utf-8",
    headers: dict | None = None,
) -> str:
    """urllib GET 并返回解码文本；瞬时失败按线性退避重试（参考 a-trade/fetcher/http.py）。"""
    hdrs = dict(_HEADERS)
    if headers:
        hdrs.update(headers)
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode(encoding, "ignore")
        except urllib.error.HTTPError as exc:
            # 4xx/5xx 是服务端明确拒绝（如限流 501），重试无益，立即失败切渠道，
            # 避免全量更新时被限流拖死（每请求白等退避好几轮）
            raise FetchError(f"请求被拒 {url}: HTTP {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001 - 网络瞬时错误，退避重试
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise FetchError(f"请求失败 {url}: {last}") from last


def _get_json(url: str, params: dict, timeout: float) -> dict:
    return json.loads(_http_get_text(url, params, timeout=timeout))


def _sleep_jitter() -> None:
    time.sleep(random.uniform(0.2, 0.5))


def _with_retry(fn, *args, retry: int = 3, what: str = "", **kwargs):
    last_exc: Exception | None = None
    for attempt in range(retry):
        try:
            return fn(*args, **kwargs)
        except ChannelUnavailable:
            raise  # 参数不适用不是网络问题，不重试直接切渠道
        except Exception as exc:  # noqa: BLE001 - 网络层异常种类多，统一退避重试
            last_exc = exc
            if attempt < retry - 1:
                time.sleep(2**attempt)  # 1s / 2s / 4s
    raise FetchError(f"{what} 重试 {retry} 次仍失败: {last_exc}") from last_exc


def _finalize_kline(df: pd.DataFrame) -> pd.DataFrame:
    """渠道原始表 → 内部标准表。缺少的列（腾讯无成交额/换手率/涨跌幅）补 NaN，
    不影响只吃 OHLCV 的形态算法。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=store.KLINE_COLUMNS)
    for col in store.KLINE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[store.KLINE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in store.KLINE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    return df.sort_values("date").reset_index(drop=True)


def _f(value) -> float | None:
    """安全转 float；空值/-/空串返回 None（行情串里停牌价就是 '-'）。"""
    if value is None or value == "-" or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _at(parts: list[str], idx: int) -> str | None:
    """按下标取行情串字段，越界返回 None（不同板块的串长度不一致）。"""
    return parts[idx] if idx < len(parts) else None


def _em_symbol(code: str) -> str:
    prefix = "0" if code.startswith(_EM_SECID_PREFIX) else "1"
    return f"{prefix}.{code}"


def _tx_symbol(code: str) -> str:
    return f"{_TX_MARKET.get(code[0], 'sz')}{code}"


def _em_adjust_param(adjust: str) -> int:
    if adjust not in _EM_ADJUST:
        raise ValueError(f"不支持的复权方式 '{adjust}'，可选: qfq / hfq / 空串")
    return _EM_ADJUST[adjust]


# ---------------------------------------------------------------------------
# K线渠道
# ---------------------------------------------------------------------------


def _kline_eastmoney(code: str, start: str, end: str, cfg) -> pd.DataFrame:
    """东财日K线，字段最全（含成交额/换手率/涨跌幅）。多主机回退提高成功率。"""
    params = {
        "secid": _em_symbol(code),
        "klt": _EM_KLT.get(cfg.data.period, 101),
        "fqt": _em_adjust_param(cfg.data.adjust),
        "beg": start,
        "end": end,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "_": int(time.time() * 1000),  # 时间戳防缓存/代理串包
    }
    last_exc: Exception | None = None
    for host in _EM_KLINE_HOSTS:
        url = f"{host}/api/qt/stock/kline/get"
        try:
            payload = _get_json(url, params, cfg.data.request_timeout)
            if payload.get("rc") not in (0, None):
                raise FetchError(f"东财K线接口报错 rc={payload.get('rc')}")
            # rc 正常但 data 为 null：代码不存在/已退市，属"接口正常但无数据"
            data = payload.get("data")
            if not data:
                return pd.DataFrame(columns=store.KLINE_COLUMNS)
            klines = data.get("klines") or []
            if not klines:
                return pd.DataFrame(columns=store.KLINE_COLUMNS)  # 区间内无数据

            rows = [line.split(",") for line in klines]
            df = pd.DataFrame(
                rows,
                columns=[
                    "date", "open", "close", "high", "low",
                    "volume", "amount", "_amplitude", "pct_chg", "_chg", "turnover",
                ],
            )
            return _finalize_kline(df.drop(columns=["_amplitude", "_chg"]))
        except FetchError as exc:
            last_exc = exc
            continue  # 当前主机失败，试下一个
    raise FetchError(f"东财K线 {code} 全部主机失败: {last_exc}")


def _tx_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def _kline_tencent(code: str, start: str, end: str, cfg) -> pd.DataFrame:
    """腾讯日K线（qfq/hfq/不复权），只有量没有额，作备用渠道。

    单次请求最多回 640 根，全量历史（约 1100+ 根）需要把 end 往前推分段补拉。
    """
    adjust = cfg.data.adjust
    if adjust not in _TX_ADJUST:
        raise ValueError(f"不支持的复权方式 '{adjust}'，可选: qfq / hfq / 空串")
    period = _TX_PERIOD.get(cfg.data.period)
    if period is None:
        raise ValueError(f"腾讯渠道不支持周期 '{cfg.data.period}'")

    symbol = _tx_symbol(code)
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    # 复权数据在 qfqday / hfqday 这类键下，不复权在 day 键下
    bar_key = f"{_TX_ADJUST[adjust]}{period}"
    start_d, end_d = _tx_date(start), _tx_date(end)

    all_bars: list = []
    for _ in range(20):  # 640根/次，20次足够覆盖十余年日线
        params = {"param": f"{symbol},{period},{start_d},{end_d},640,{_TX_ADJUST[adjust]}"}
        payload = _get_json(url, params, cfg.data.request_timeout)
        if payload.get("code") != 0:
            raise FetchError(f"腾讯K线接口报错 code={payload.get('code')}")

        data = (payload.get("data") or {}).get(symbol)
        if not data:
            # code 正常但没有该 symbol：代码不存在/已退市，同东财按"无数据"处理。
            # 若前几段已取到数据，说明只是翻到头了，保留已取部分。
            break
        bars = [row[:6] for row in (data.get(bar_key) or []) if len(row) >= 6]
        if not bars:
            break
        # 每次返回的是靠近 end 的一段，拼到已取数据的前面，再把 end 往前推
        all_bars = bars + all_bars
        if len(bars) < 640 or bars[0][0] <= start_d:
            break
        end_d = (pd.Timestamp(bars[0][0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if not all_bars:
        return pd.DataFrame(columns=store.KLINE_COLUMNS)

    # 每根: [日期, 开, 收, 高, 低, 成交量(手)]，个别记录末尾会多出信息字段已截掉
    df = pd.DataFrame(
        all_bars, columns=["date", "open", "close", "high", "low", "volume"]
    )
    return _finalize_kline(df)


def _kline_sina(code: str, start: str, end: str, cfg) -> pd.DataFrame:
    """新浪日K线（仅不复权），单请求拿最近 N 根，极少限流（a-trade 实测经验）。

    无成交额/换手率/涨跌幅；成交量单位是股，统一换算成手对齐东财/腾讯。
    """
    if cfg.data.adjust:
        raise ChannelUnavailable('新浪渠道仅支持不复权（adjust: ""）')
    if cfg.data.period != "daily":
        raise ChannelUnavailable("新浪渠道仅支持日K（period: daily）")
    # 按区间日历日估算交易日（5/7 折算）+ 缓冲，避免取太多或不够（同 a-trade 思路）
    days = (pd.Timestamp(_tx_date(end)) - pd.Timestamp(_tx_date(start))).days
    datalen = max(30, min(int(days * 5 / 7) + 40, _SINA_MAX_BARS))
    params = {
        "symbol": _tx_symbol(code),
        "scale": 240,
        "ma": "no",
        "datalen": datalen,
    }
    txt = _http_get_text(
        _SINA_KLINE,
        params,
        timeout=cfg.data.request_timeout,
        encoding="gbk",
        headers={"Referer": "https://finance.sina.com.cn/"},
    )
    txt = txt.strip()
    if not txt or txt == "null":
        return pd.DataFrame(columns=store.KLINE_COLUMNS)  # 代码不存在/无数据
    rows: list = []
    for r in json.loads(txt):
        try:
            rows.append(
                (r["day"], r["open"], r["close"], r["high"], r["low"], r["volume"])
            )
        except (KeyError, TypeError):
            continue  # 个别脏行跳过（同 a-trade）
    if not rows:
        return pd.DataFrame(columns=store.KLINE_COLUMNS)
    df = pd.DataFrame(
        rows, columns=["date", "open", "close", "high", "low", "volume"]
    )
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100  # 股→手
    # 新浪不支持日期区间参数，取回后自行裁剪到 [start, end]
    df = df[(df["date"] >= _tx_date(start)) & (df["date"] <= _tx_date(end))]
    return _finalize_kline(df)


# ---------------------------------------------------------------------------
# Baostock 渠道（Python SDK，非 HTTP）
# ---------------------------------------------------------------------------

_bs_logged_in = False


def _bs_login() -> None:
    """Baostock 需要登录才能使用，全局只需登录一次。"""
    global _bs_logged_in
    if _bs_logged_in:
        return
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise FetchError(f"Baostock 登录失败: {lg.error_msg}")
        _bs_logged_in = True
    except ImportError:
        raise ChannelUnavailable("Baostock 未安装，请 pip install baostock")


def _bs_symbol(code: str) -> str:
    """转换代码格式：000001 -> sz.000001, 600000 -> sh.600000"""
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


_BS_ADJUST = {"": "3", "qfq": "2", "hfq": "1"}  # 3=不复权, 2=前复权, 1=后复权
_BS_FREQ = {"daily": "d", "weekly": "w", "monthly": "m"}


def _kline_baostock(code: str, start: str, end: str, cfg) -> pd.DataFrame:
    """Baostock 日K线，SDK 方式获取，适合 HTTP 渠道受限时使用。"""
    _bs_login()
    import baostock as bs

    adjust = cfg.data.adjust
    if adjust not in _BS_ADJUST:
        raise ChannelUnavailable(f"Baostock 不支持复权方式 '{adjust}'")
    period = cfg.data.period
    if period not in _BS_FREQ:
        raise ChannelUnavailable(f"Baostock 不支持周期 '{period}'")

    symbol = _bs_symbol(code)
    start_date = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_date = f"{end[:4]}-{end[4:6]}-{end[6:]}"

    fields = "date,open,high,low,close,volume,amount,turn,pctChg"
    try:
        rs = bs.query_history_k_data_plus(
            symbol,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=_BS_FREQ[period],
            adjustflag=_BS_ADJUST[adjust],
        )

        if rs.error_code != "0":
            raise FetchError(f"Baostock 查询失败: {rs.error_msg}")

        rows: list[list] = []
        while rs.next():
            rows.append(rs.get_row_data())
    except UnicodeDecodeError as exc:
        raise FetchError(f"Baostock 编码错误: {exc}")
    except Exception as exc:
        exc_name = type(exc).__name__
        raise FetchError(f"Baostock 异常 ({exc_name}): {exc}")

    if not rows:
        return pd.DataFrame(columns=store.KLINE_COLUMNS)

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                      "volume", "amount", "turnover", "pct_chg"])
    # Baostock 成交量单位是股，转换为手
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100
    return _finalize_kline(df)


KLINE_CHANNELS = {
    "eastmoney": ("东财", _kline_eastmoney),
    "sina": ("新浪", _kline_sina),
    "tencent": ("腾讯", _kline_tencent),
    "baostock": ("Baostock", _kline_baostock),
}


def fetch_kline(code: str, start_date: str, end_date: str, cfg) -> tuple[pd.DataFrame, str | None]:
    """多渠道拉单只股票日K线，返回 (DataFrame, 成功渠道名)。

    渠道返回空表时也继续尝试下一家；但只要任一渠道确认过"接口正常但无数据"，
    最终就返回空表（代码/日期范围确实无数据），其余渠道的异常不再追究；
    只有全部渠道都是异常才抛 FetchError。
    """
    # 错开并发请求，避免同时打到服务器
    time.sleep(random.uniform(0.1, 0.3))

    errors: list[str] = []
    saw_empty = False
    sources = cfg.data.sources
    for idx, source in enumerate(sources, 1):
        entry = KLINE_CHANNELS.get(source)
        if entry is None:
            raise ValueError(
                f"未知数据源 '{source}'，可选: {', '.join(KLINE_CHANNELS)}"
            )
        label, fn = entry
        log.info("[%s] 尝试渠道 %s (%d/%d)", code, label, idx, len(sources))
        try:
            df = _with_retry(
                fn, code, start_date, end_date, cfg,
                retry=cfg.data.retry, what=f"{label}K线 {code}",
            )
        except ChannelUnavailable as exc:
            log.info("[%s] %s 不适用，切换下一渠道: %s", code, label, exc)
            continue
        except FetchError as exc:
            log.warning("[%s] %s 失败: %s", code, label, exc)
            errors.append(f"{label}: {exc}")
            continue
        _sleep_jitter()
        if not df.empty:
            log.info("[%s] %s 成功，获取 %d 根K线", code, label, len(df))
            return df, label
        saw_empty = True
        log.info("[%s] %s 返回空表（无数据），尝试下一渠道", code, label)

    if saw_empty:
        log.info("[%s] 渠道确认无数据，返回空表", code)
        return pd.DataFrame(columns=store.KLINE_COLUMNS), None
    raise FetchError(f"K线 {code} 全部渠道失败: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# 指数 K 线
# ---------------------------------------------------------------------------

# 指数代码 → 交易所映射（与个股不同：000xxx 指数属于上交所）
_IDX_EM_PREFIX = {"000": "1", "001": "1", "399": "0"}  # 000/001=沪, 399=深
_IDX_BS_PREFIX = {"000": "sh", "001": "sh", "399": "sz"}


def _idx_em_symbol(idx_code: str) -> str:
    """指数代码 → 东财 secid。idx_000300 → 1.000300, idx_399006 → 0.399006"""
    raw = idx_code.removeprefix("idx_")
    for prefix, market in _IDX_EM_PREFIX.items():
        if raw.startswith(prefix):
            return f"{market}.{raw}"
    raise ValueError(f"未知指数交易所前缀: {raw}")


def _idx_bs_symbol(idx_code: str) -> str:
    """指数代码 → baostock symbol。idx_000300 → sh.000300, idx_399006 → sz.399006"""
    raw = idx_code.removeprefix("idx_")
    for prefix, exchange in _IDX_BS_PREFIX.items():
        if raw.startswith(prefix):
            return f"{exchange}.{raw}"
    raise ValueError(f"未知指数交易所前缀: {raw}")


def _idx_tx_symbol(idx_code: str) -> str:
    """指数代码 → 腾讯 symbol。idx_000300 → sh000300, idx_399006 → sz399006"""
    raw = idx_code.removeprefix("idx_")
    for prefix, exchange in _IDX_BS_PREFIX.items():
        if raw.startswith(prefix):
            return f"{exchange}{raw}"
    raise ValueError(f"未知指数交易所前缀: {raw}")


def _kline_eastmoney_index(idx_code: str, start: str, end: str, cfg) -> pd.DataFrame:
    """东财指数日K线。与个股版相同但 secid 映射不同且不复权。"""
    params = {
        "secid": _idx_em_symbol(idx_code),
        "klt": _EM_KLT.get(cfg.data.period, 101),
        "fqt": 0,  # 指数不复权
        "beg": start,
        "end": end,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "_": int(time.time() * 1000),
    }
    last_exc: Exception | None = None
    for host in _EM_KLINE_HOSTS:
        url = f"{host}/api/qt/stock/kline/get"
        try:
            payload = _get_json(url, params, cfg.data.request_timeout)
            if payload.get("rc") not in (0, None):
                raise FetchError(f"东财K线接口报错 rc={payload.get('rc')}")
            data = payload.get("data")
            if not data:
                return pd.DataFrame(columns=store.KLINE_COLUMNS)
            klines = data.get("klines") or []
            if not klines:
                return pd.DataFrame(columns=store.KLINE_COLUMNS)
            rows = [line.split(",") for line in klines]
            df = pd.DataFrame(
                rows,
                columns=[
                    "date", "open", "close", "high", "low",
                    "volume", "amount", "amplitude", "pct_chg",
                    "price_change", "turnover",
                ],
            )
            return _finalize_kline(df)
        except FetchError as exc:
            last_exc = exc
            log.warning("东财指数K线 %s 主机 %s 失败: %s", idx_code, host, exc)
    raise FetchError(f"东财指数K线 {idx_code} 全部主机失败: {last_exc}") from last_exc


def _kline_baostock_index(idx_code: str, start: str, end: str, cfg) -> pd.DataFrame:
    """Baostock 指数日K线。不复权。"""
    _bs_login()
    import baostock as bs

    symbol = _idx_bs_symbol(idx_code)
    start_date = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_date = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    fields = "date,open,high,low,close,volume,amount,turn,pctChg"
    rs = bs.query_history_k_data_plus(
        symbol, fields,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3",  # 指数不复权
    )
    if rs.error_code != "0":
        raise FetchError(f"Baostock 指数 {idx_code} 报错: {rs.error_msg}")
    rows: list[list] = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=store.KLINE_COLUMNS)
    df = pd.DataFrame(rows, columns=rs.fields)
    return _finalize_kline(df)


INDEX_KLINE_CHANNELS = {
    "eastmoney": ("东财指数", _kline_eastmoney_index),
    "baostock": ("Baostock指数", _kline_baostock_index),
}


def fetch_index_kline(
    idx_code: str, start_date: str, end_date: str, cfg,
) -> tuple[pd.DataFrame, str | None]:
    """拉取指数日K线。idx_code 格式为 idx_000300 / idx_000905 / idx_399006。

    复用多渠道框架，但只走 eastmoney + baostock（指数不需要复权，新浪/腾讯的
    指数支持不完整）。
    """
    time.sleep(random.uniform(0.05, 0.15))
    errors: list[str] = []
    saw_empty = False
    for label, fn in INDEX_KLINE_CHANNELS.values():
        try:
            df = _with_retry(
                fn, idx_code, start_date, end_date, cfg,
                retry=cfg.data.retry, what=f"{label} {idx_code}",
            )
        except ChannelUnavailable as exc:
            log.info("[%s] %s 不适用: %s", idx_code, label, exc)
            continue
        except FetchError as exc:
            log.warning("[%s] %s 失败: %s", idx_code, label, exc)
            errors.append(f"{label}: {exc}")
            continue
        if not df.empty:
            log.info("[%s] %s 成功，%d 根K线", idx_code, label, len(df))
            return df, label
        saw_empty = True
    if saw_empty:
        return pd.DataFrame(columns=store.KLINE_COLUMNS), None
    raise FetchError(f"指数K线 {idx_code} 全部渠道失败: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# 全市场快照渠道
# ---------------------------------------------------------------------------


def _em_clist_all(fields: str, cfg) -> list[dict]:
    """东财 clist 分页取全量。服务端每页上限约 100，只能按 total 逐页翻。
    主机池轮着试：部分网络环境（代理）会屏蔽个别主机。
    """
    out: list[dict] = []
    page, total = 1, None
    host_idx = 0
    while page <= 100:  # 硬上限保险，满市场约 55 页
        params = {
            "pn": page,
            "pz": _EM_CLIST_PZ,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f12",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
            "fields": fields,
            "_": int(time.time() * 1000),
        }
        payload = None
        last_exc: Exception | None = None
        for offset in range(len(_EM_CLIST_HOSTS)):
            host = _EM_CLIST_HOSTS[(host_idx + offset) % len(_EM_CLIST_HOSTS)]
            try:
                payload = _get_json(
                    f"{host}/api/qt/clist/get", params, cfg.data.request_timeout
                )
                host_idx = (host_idx + offset) % len(_EM_CLIST_HOSTS)
                break
            except Exception as exc:  # noqa: BLE001 - 换下一台主机再试
                last_exc = exc
        if payload is None:
            raise FetchError(f"东财 clist 全部主机不可用: {last_exc}")

        data = payload.get("data") or {}
        diff = data.get("diff") or []
        if not diff:
            break
        out.extend(diff)
        if total is None:
            total = data.get("total") or 0
        if len(out) >= total:
            break
        page += 1
        _sleep_jitter()
    return out


def _spot_eastmoney(cfg) -> pd.DataFrame:
    """东财全市场快照（沪深京A股，约5400只），分页取全量。"""
    # f2 最新价 | f3 涨跌幅 | f6 成交额 | f8 换手率 | f12 代码 | f14 名称 | f21 流通市值
    # f3 是盘中面板算涨跌停/行业涨幅榜要用的，universe.filter_spot 忽略它
    diff = _em_clist_all("f2,f3,f6,f8,f12,f14,f21", cfg)
    if not diff:
        raise FetchError("东财全市场快照返回空")
    raw = pd.DataFrame(diff)
    return _normalize_spot(
        pd.DataFrame(
            {
                "code": raw["f12"].astype(str).str.zfill(6),
                "name": raw["f14"].astype(str),
                "price": pd.to_numeric(raw["f2"], errors="coerce"),
                "pct_chg": pd.to_numeric(raw["f3"], errors="coerce"),
                "amount": pd.to_numeric(raw["f6"], errors="coerce"),
                "float_mv": pd.to_numeric(raw["f21"], errors="coerce"),
                "turnover": pd.to_numeric(raw["f8"], errors="coerce"),
            }
        )
    )


def _all_stock_codes(cfg) -> list[str]:
    """全市场股票代码清单。优先东财（含北交所），不可用则退回本地库。

    腾讯没有公开的全市场列表接口，所以东财一被墙，腾讯渠道会**跟着整体失效**。
    退回本地库至少还能拿到沪深主板/创业板/科创板，代价是不含北交所——
    比"全市场快照全军覆没、面板只能显示昨天的收盘数"好得多。
    """
    try:
        diff = _em_clist_all("f12", cfg)
        if diff:
            return [str(item["f12"]).zfill(6) for item in diff]
    except Exception as exc:  # noqa: BLE001 - 兜底路径，任何失败都退回本地
        log.warning("东财代码清单不可用（%s），改用本地库代码清单", exc)

    codes = store.stock_codes(cfg)
    if codes:
        log.warning("全市场快照的代码清单取自本地库 %d 只（不含北交所）", len(codes))
    return codes


def _spot_tencent(cfg) -> pd.DataFrame:
    """腾讯备用快照：代码清单优先东财，拿不到就用本地库（见 _all_stock_codes）。

    行情串解析与 _quotes_tencent 同源；该接口极少限流，是可靠回退源。
    """
    codes = _all_stock_codes(cfg)
    if not codes:
        raise FetchError("腾讯渠道无法获取代码列表（东财不可用且本地库为空）")
    symbols = [_tx_symbol(c) for c in codes]

    rows: list[dict] = []
    for i in range(0, len(symbols), _TX_BATCH):
        batch = symbols[i:i + _TX_BATCH]
        # 腾讯行情串是 GBK 编码；该接口极少限流，是可靠的回退源（同 a-trade 经验）
        text = _http_get_text(
            "https://qt.gtimg.cn/q=" + ",".join(batch),
            timeout=cfg.data.request_timeout,
            retries=4,
            encoding="gbk",
            headers={"Referer": "https://gu.qq.com/"},
        )
        for line in text.strip().split(";"):
            line = line.strip()
            if "=" not in line:
                continue
            val = line.split("=", 1)[1].strip().strip('"')
            if not val:
                continue
            p = val.split("~")
            if len(p) < 46:
                continue
            price = _f(p[_TX_F_PRICE])
            if price is None:  # 停牌/无效报价
                continue
            # 高位下标（成交额/流通市值/换手率）在短行情串里可能不存在，越界按缺失处理
            amount_wan = (
                _f(_at(p, _TX_F_AMOUNT_WAN)) or _f(_at(p, _TX_F_AMOUNT_WAN_ALT)) or 0.0
            )
            rows.append(
                {
                    "code": p[2],
                    "name": p[_TX_F_NAME],
                    "price": price,
                    "pct_chg": _f(_at(p, _TX_F_PCT_CHG)),
                    "amount": amount_wan * 1e4,
                    "float_mv": (_f(_at(p, _TX_F_FLOAT_MV_YI)) or 0.0) * 1e8,
                    "turnover": _f(_at(p, _TX_F_TURNOVER)),
                }
            )
        _sleep_jitter()

    if not rows:
        raise FetchError("腾讯行情批量接口返回空")
    return _normalize_spot(pd.DataFrame(rows))


def _normalize_spot(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="code").reset_index(drop=True)


SPOT_CHANNELS = {
    "eastmoney": ("东财", _spot_eastmoney),
    "tencent": ("腾讯", _spot_tencent),
}


QUOTE_COLUMNS = [
    "code", "name", "price", "open", "high", "low",
    "volume", "amount", "prev_close", "pct_chg",
]


def _normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """报价表统一列与类型。停牌/无效价（price<=0）直接剔除，盘中监控用不上。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    out = df.copy()
    for col in QUOTE_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    out = out[QUOTE_COLUMNS]
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["name"] = out["name"].astype(str)
    for col in QUOTE_COLUMNS[2:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["price"] > 0]
    return out.drop_duplicates(subset="code").reset_index(drop=True)


def _quotes_eastmoney(codes: list[str], cfg) -> pd.DataFrame:
    """东财批量报价。ulist.np 支持一次传多个 secid，比全市场 clist 轻得多。"""
    # f2 现价 | f3 涨跌幅 | f5 成交量(手) | f6 成交额 | f12 代码 | f14 名称
    # f15 最高 | f16 最低 | f17 今开 | f18 昨收
    fields = "f2,f3,f5,f6,f12,f14,f15,f16,f17,f18"
    rows: list[dict] = []
    host_idx = 0
    for i in range(0, len(codes), _EM_QUOTE_BATCH):
        batch = codes[i:i + _EM_QUOTE_BATCH]
        params = {
            "fltt": 2,
            "invt": 2,
            "fields": fields,
            "secids": ",".join(_em_symbol(c) for c in batch),
            "_": int(time.time() * 1000),
        }
        payload = None
        last_exc: Exception | None = None
        for offset in range(len(_EM_CLIST_HOSTS)):
            host = _EM_CLIST_HOSTS[(host_idx + offset) % len(_EM_CLIST_HOSTS)]
            try:
                payload = _get_json(
                    f"{host}/api/qt/ulist.np/get", params, cfg.data.request_timeout
                )
                host_idx = (host_idx + offset) % len(_EM_CLIST_HOSTS)
                break
            except Exception as exc:  # noqa: BLE001 - 换下一台主机再试
                last_exc = exc
        if payload is None:
            raise FetchError(f"东财 ulist 全部主机不可用: {last_exc}")

        diff = (payload.get("data") or {}).get("diff") or []
        for item in diff:
            rows.append(
                {
                    "code": str(item.get("f12", "")).zfill(6),
                    "name": item.get("f14"),
                    "price": _f(item.get("f2")),
                    "open": _f(item.get("f17")),
                    "high": _f(item.get("f15")),
                    "low": _f(item.get("f16")),
                    "volume": _f(item.get("f5")),   # 已是"手"，与本地库单位一致
                    "amount": _f(item.get("f6")),
                    "prev_close": _f(item.get("f18")),
                    "pct_chg": _f(item.get("f3")),
                }
            )
        if len(codes) > _EM_QUOTE_BATCH:
            _sleep_jitter()

    if not rows:
        raise FetchError("东财批量报价返回空")
    return _normalize_quotes(pd.DataFrame(rows))


def _quotes_tencent(codes: list[str], cfg) -> pd.DataFrame:
    """腾讯批量报价，复用 _spot_tencent 已验证的行情串解析。极少限流，是可靠回退。"""
    rows: list[dict] = []
    symbols = [_tx_symbol(c) for c in codes]
    for i in range(0, len(symbols), _TX_BATCH):
        batch = symbols[i:i + _TX_BATCH]
        text = _http_get_text(
            "https://qt.gtimg.cn/q=" + ",".join(batch),
            timeout=cfg.data.request_timeout,
            retries=4,
            encoding="gbk",
            headers={"Referer": "https://gu.qq.com/"},
        )
        for line in text.strip().split(";"):
            line = line.strip()
            if "=" not in line:
                continue
            val = line.split("=", 1)[1].strip().strip('"')
            if not val:
                continue
            p = val.split("~")
            if len(p) <= _TX_F_LOW:
                continue
            price = _f(p[_TX_F_PRICE])
            if price is None:  # 停牌/无效报价
                continue
            amount_wan = (
                _f(_at(p, _TX_F_AMOUNT_WAN)) or _f(_at(p, _TX_F_AMOUNT_WAN_ALT)) or 0.0
            )
            rows.append(
                {
                    "code": p[2],
                    "name": p[_TX_F_NAME],
                    "price": price,
                    "open": _f(_at(p, _TX_F_OPEN)),
                    "high": _f(_at(p, _TX_F_HIGH)),
                    "low": _f(_at(p, _TX_F_LOW)),
                    "volume": _f(_at(p, _TX_F_VOLUME)),  # 行情串里就是"手"
                    "amount": amount_wan * 1e4,
                    "prev_close": _f(_at(p, _TX_F_PREV_CLOSE)),
                    "pct_chg": _f(_at(p, _TX_F_PCT_CHG)),
                }
            )
        if len(symbols) > _TX_BATCH:
            _sleep_jitter()

    if not rows:
        raise FetchError("腾讯批量报价返回空")
    return _normalize_quotes(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# 同花顺（个股实时报价）
# ---------------------------------------------------------------------------

# 同花顺个股代码统一走 hs_ 前缀（沪深主板/创业板/科创板实测都通；北交所返回 502）。
_THS_REFERER = {"Referer": "https://stockpage.10jqka.com.cn/"}
_THS_QUOTE_URL = "https://d.10jqka.com.cn/v6/realhead/{symbol}/last.js"

# 返回体是数字字段 ID，实测对出来的映射（2026-09-30 用平安银行核对：
# 6=昨收11.35 与本地库一致；199112=1.94 = 11.57/11.35-1；69/70=涨跌停价12.49/10.22）
_THS_F_PRICE = "10"
_THS_F_PREV_CLOSE = "6"
_THS_F_OPEN = "7"
_THS_F_HIGH = "8"
_THS_F_LOW = "9"
_THS_F_VOLUME = "13"     # 单位：股（同花顺 K 线同样是股，本地库是手，差 100 倍）
_THS_F_AMOUNT = "19"     # 单位：元
_THS_F_PCT_CHG = "199112"


def _ths_symbol(code: str) -> str:
    """个股代码 → 同花顺 symbol。000001 → hs_000001。"""
    return f"hs_{str(code).zfill(6)}"


def _jsonp_payload(text: str) -> dict | None:
    """剥掉 JSONP 外壳：`someCallback({...});` → dict。解析失败返回 None。"""
    m = re.search(r"\(\s*(\{.*\})\s*\)\s*;?\s*$", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _parse_ths_quote(payload: dict) -> dict | None:
    """同花顺 items → QUOTE_COLUMNS 行。无有效现价（停牌/退市）返回 None。"""
    items = (payload or {}).get("items") or {}
    price = _f(items.get(_THS_F_PRICE))
    if price is None or price <= 0:
        return None
    volume = _f(items.get(_THS_F_VOLUME))
    return {
        "code": str(items.get("5") or "").zfill(6),
        "name": items.get("name") or "",
        "price": price,
        "open": _f(items.get(_THS_F_OPEN)),
        "high": _f(items.get(_THS_F_HIGH)),
        "low": _f(items.get(_THS_F_LOW)),
        "volume": None if volume is None else volume / 100,  # 股 → 手
        "amount": _f(items.get(_THS_F_AMOUNT)),
        "prev_close": _f(items.get(_THS_F_PREV_CLOSE)),
        "pct_chg": _f(items.get(_THS_F_PCT_CHG)),
    }


def _quotes_ths(codes: list[str], cfg) -> pd.DataFrame:
    """同花顺个股实时报价。**只找到单代码接口**，所以目前是一股一请求。

    已验证的事实（2026-09-30）：`realhead/hs_<code>/last.js` 只吃单个代码——
    多代码的几种写法（逗号 / 分号 / 竖线 / `?code=` 追加）实测**一律 404 或被忽略**，
    `line/` 那族同理；所以不能断言"同花顺没有批量"，只能说"这一族路径上没有批量形式"。
    如果哪天拿到批量端点，这里换成批量即可，`fetch_quotes` 的补漏语义不用动。

    实测成本（89 只观察池）：一股一请求 18~39 秒、覆盖 81~89/89（约 10~20% 是网关
    502 抖动，同批里随机失败）；腾讯 60 只/请求、89 只 1.0 秒全覆盖。所以它排在
    **最后当兵底**，且靠"凑齐即止"在正常轮次里根本不会被调用。
    """
    rows: list[dict] = []
    failed: list[str] = []
    for code in codes:
        try:
            text = _http_get_text(
                _THS_QUOTE_URL.format(symbol=_ths_symbol(code)),
                timeout=cfg.data.request_timeout,
                retries=2,
                headers=_THS_REFERER,
            )
            row = _parse_ths_quote(_jsonp_payload(text) or {})
        except Exception as exc:  # noqa: BLE001 - 单只失败不该中断整批
            log.debug("同花顺报价 %s 失败: %s", code, exc)
            row = None
        if row is None:
            failed.append(code)
            continue
        row["code"] = str(code).zfill(6)  # 以请求的代码为准，不信任返回值
        rows.append(row)

    if not rows:
        raise FetchError(f"同花顺报价全部失败（{len(failed)} 只）")
    if failed:
        log.info("同花顺报价 %d/%d 成功，缺 %d 只留给后续渠道", len(rows), len(codes), len(failed))
    return _normalize_quotes(pd.DataFrame(rows))


QUOTE_CHANNELS = {
    # 顺序 = 默认尝试顺序，必须与 config.yaml 的 watch.quote_sources 一致
    # （trade.py 那边调用时没传 sources，走的就是这个字典顺序）
    "eastmoney": ("东财", _quotes_eastmoney),
    "tencent": ("腾讯", _quotes_tencent),
    "ths": ("同花顺", _quotes_ths),
}


def fetch_quotes(codes: list[str], cfg, sources: list[str] | None = None) -> pd.DataFrame:
    """按代码批量取实时报价，返回 QUOTE_COLUMNS 表（volume 单位=手，与本地K线一致）。

    与 fetch_spot 的区别：那个拉全市场（东财要翻 55 页），盘中每分钟跑一次太重；
    这个只取观察池里的几十只，一两次请求就够。

    渠道之间是**补漏**关系而不是"首个非空即返回"：靠前的渠道缺哪些票，就只用缺的
    那些去问下一个渠道，凑齐即止（凑齐后后面的渠道根本不会被调用）。原因是各渠道
    覆盖都不满——停牌、新股，以及同花顺那种网关 502 抖动（实测 10~20%）——
    旧逻辑会把这些票直接丢给调用方当"无报价"。
    """
    codes = [str(c).zfill(6) for c in dict.fromkeys(codes)]
    if not codes:
        return pd.DataFrame(columns=QUOTE_COLUMNS)

    wanted = set(codes)
    got: dict[str, dict] = {}
    errors: list[str] = []

    for source in sources or list(QUOTE_CHANNELS):
        missing = [c for c in codes if c not in got]
        if not missing:
            break
        entry = QUOTE_CHANNELS.get(source)
        if entry is None:
            log.debug("渠道 %s 不提供批量报价，跳过", source)
            continue
        label, fn = entry
        try:
            df = _with_retry(
                fn, missing, cfg, retry=cfg.data.retry, what=f"{label}批量报价"
            )
        except FetchError as exc:
            log.warning("%s批量报价失败: %s", label, exc)
            errors.append(f"{label}: {exc}")
            continue
        added = 0
        for row in df.to_dict("records"):
            code = str(row["code"]).zfill(6)
            if code in wanted and code not in got:
                got[code] = row
                added += 1
        if added:
            log.debug("%s批量报价补齐 %d 只，累计 %d/%d", label, added, len(got), len(codes))

    if not got:
        raise FetchError(f"批量报价全部渠道失败: {'; '.join(errors) or '返回空'}")

    left = [c for c in codes if c not in got]
    if left:
        log.info("报价仍缺 %d/%d 只（停牌或各渠道都没给）: %s", len(left), len(codes), ",".join(left[:8]))
    out = pd.DataFrame([got[c] for c in codes if c in got])
    return _normalize_quotes(out)


INDEX_QUOTE_COLUMNS = ["code", "name", "price", "pct_chg"]


def _normalize_index_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """指数报价列与类型统一。无效点位（<=0）剔除。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=INDEX_QUOTE_COLUMNS)
    out = df.copy()
    for col in INDEX_QUOTE_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    out = out[INDEX_QUOTE_COLUMNS]
    out["code"] = out["code"].astype(str)
    out["name"] = out["name"].astype(str)
    for col in ("price", "pct_chg"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["price"] > 0]
    return out.drop_duplicates(subset="code").reset_index(drop=True)


def _index_quotes_eastmoney(codes: list[str], cfg) -> pd.DataFrame:
    """东财指数实时点位。复用 ulist.np，secid 走 _idx_em_symbol（1.000300 / 0.399006）。"""
    want = {c.removeprefix("idx_"): c for c in codes}
    params = {
        "fltt": 2,
        "invt": 2,
        "fields": "f2,f3,f12,f14",  # 点位 | 涨跌幅 | 代码 | 名称
        "secids": ",".join(_idx_em_symbol(c) for c in codes),
        "_": int(time.time() * 1000),
    }
    payload = None
    last_exc: Exception | None = None
    host_idx = 0
    for offset in range(len(_EM_CLIST_HOSTS)):
        host = _EM_CLIST_HOSTS[(host_idx + offset) % len(_EM_CLIST_HOSTS)]
        try:
            payload = _get_json(
                f"{host}/api/qt/ulist.np/get", params, cfg.data.request_timeout
            )
            host_idx = (host_idx + offset) % len(_EM_CLIST_HOSTS)
            break
        except Exception as exc:  # noqa: BLE001 - 换下一台主机再试
            last_exc = exc
    if payload is None:
        raise FetchError(f"东财指数报价全部主机不可用: {last_exc}")

    rows: list[dict] = []
    for item in (payload.get("data") or {}).get("diff") or []:
        bare = str(item.get("f12", "")).zfill(6)
        rows.append(
            {
                "code": want.get(bare, f"idx_{bare}"),
                "name": item.get("f14"),
                "price": _f(item.get("f2")),
                "pct_chg": _f(item.get("f3")),
            }
        )
    if not rows:
        raise FetchError("东财指数报价返回空")
    return _normalize_index_quotes(pd.DataFrame(rows))


def _index_quotes_tencent(codes: list[str], cfg) -> pd.DataFrame:
    """腾讯指数实时点位（回退源）。复用个股行情串的字段下标。"""
    want = {_idx_tx_symbol(c): c for c in codes}
    text = _http_get_text(
        "https://qt.gtimg.cn/q=" + ",".join(want),
        timeout=cfg.data.request_timeout,
        retries=4,
        encoding="gbk",
        headers={"Referer": "https://gu.qq.com/"},
    )
    rows: list[dict] = []
    for line in text.strip().split(";"):
        line = line.strip()
        if "=" not in line:
            continue
        head, _, val = line.partition("=")
        symbol = head.strip().removeprefix("v_")
        val = val.strip().strip('"')
        if not val:
            continue
        p = val.split("~")
        if len(p) <= _TX_F_PCT_CHG:
            continue
        price = _f(p[_TX_F_PRICE])
        if price is None:  # 无行情（休市/接口异常）
            continue
        rows.append(
            {
                "code": want.get(symbol, f"idx_{p[2]}"),
                "name": p[_TX_F_NAME],
                "price": price,
                "pct_chg": _f(_at(p, _TX_F_PCT_CHG)),
            }
        )
    if not rows:
        raise FetchError("腾讯指数报价返回空")
    return _normalize_index_quotes(pd.DataFrame(rows))


INDEX_QUOTE_CHANNELS = {
    "eastmoney": ("东财", _index_quotes_eastmoney),
    "tencent": ("腾讯", _index_quotes_tencent),
}


def fetch_index_quotes(
    codes: list[str], cfg, sources: list[str] | None = None,
) -> pd.DataFrame:
    """批量取指数实时点位，返回 INDEX_QUOTE_COLUMNS。

    只要 1 次请求（3 个 secid），盘中每分钟调一次可以接受。全部渠道失败抛
    FetchError，由调用方回退到本地收盘价，不在这里吞错降级成空表。
    """
    codes = [str(c) for c in dict.fromkeys(codes)]
    if not codes:
        return pd.DataFrame(columns=INDEX_QUOTE_COLUMNS)

    errors: list[str] = []
    for source in sources or list(INDEX_QUOTE_CHANNELS):
        entry = INDEX_QUOTE_CHANNELS.get(source)
        if entry is None:
            log.debug("渠道 %s 不提供指数报价，跳过", source)
            continue
        label, fn = entry
        try:
            df = _with_retry(
                fn, codes, cfg, retry=cfg.data.retry, what=f"{label}指数报价"
            )
        except FetchError as exc:
            log.warning("%s指数报价失败: %s", label, exc)
            errors.append(f"{label}: {exc}")
            continue
        if not df.empty:
            return df

    raise FetchError(f"指数报价全部渠道失败: {'; '.join(errors) or '返回空'}")


def fetch_spot(cfg) -> pd.DataFrame:
    """全市场实时快照（沪深京A股）。用于构建股票池和取名称。多渠道兜底。

    不是每个渠道都提供全市场列表（新浪就没有），配置里这类渠道直接跳过——
    它们在 K线渠道里仍然有效，不能当成配置写错。
    """
    errors: list[str] = []
    for source in cfg.data.sources:
        entry = SPOT_CHANNELS.get(source)
        if entry is None:
            if source in KLINE_CHANNELS:
                log.debug("渠道 %s 不提供全市场快照，跳过", source)
                continue
            raise ValueError(
                f"未知数据源 '{source}'，可选: {', '.join(KLINE_CHANNELS)}"
            )
        label, fn = entry
        try:
            df = _with_retry(fn, cfg, retry=cfg.data.retry, what=f"{label}全市场快照")
        except FetchError as exc:
            errors.append(str(exc))
            continue
        if not df.empty:
            return df

    raise FetchError(f"全市场快照全部渠道失败: {'; '.join(errors) or '返回空'}")
