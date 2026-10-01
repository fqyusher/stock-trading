"""盘中监控：把实时报价拼成"今日这根K线"，原样重跑形态检测器。

核心原则是**盘中不写任何新的阈值逻辑**。判定完全走 patterns/ 里那套检测器，
只是喂给它的最后一根 bar 由实时报价合成。这样盘中口径与收盘 scan 完全一致，
不会出现"盘中提醒了、收盘扫不出来"的分裂。

同一次 detect() 调用顺带承担失效判定：返回空列表就说明形态已废
（最低点超期、或破底深度超上限变成真破位），拒绝原因从 ctx.trace 取。
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import fetch, scan, store, trade, watch
from .patterns import STATUS_FORMED, STATUS_INVALID, PatternHit

log = logging.getLogger(__name__)

# 决策类型
PROMOTED = "promoted"    # 站稳成型，进提醒名单
FORMED = "formed"        # 已成型但确认轮次未够
FORMING = "forming"      # 仍在构筑
INVALID = "invalid"      # 形态失效，移出池子
NOQUOTE = "noquote"      # 停牌/取不到报价，本轮跳过

ALERT_COLUMNS = [
    "触发时间", "代码", "名称", "形态", "现价", "参考低点",
    "收复幅度%", "参考止损", "上方压力", "综合评分", "确认轮次", "说明",
]


@dataclass
class Decision:
    code: str
    name: str
    kind: str
    hit: PatternHit | None = None
    price: float = 0.0
    reason: str = ""
    entry: dict[str, Any] = field(default_factory=dict)


def build_intraday_df(
    hist: pd.DataFrame, quote: dict[str, Any], today: pd.Timestamp | None = None
) -> pd.DataFrame:
    """把实时报价拼成今日 bar 接到日线尾部，返回新表（不改原表）。

    本地已有今日行时**替换**而非追加：update.py 在未收盘时也会写当日数据，
    直接 append 会得到两根同日 K 线，把 MA/ATR 全带偏。

    high/low 再与现价取 max/min：报价接口的最高最低偶尔滞后于现价，
    不兜住会出现 close 跑到 high 之上这种非法 bar。
    """
    if hist is None or hist.empty:
        return hist

    today = pd.Timestamp(today or date.today()).normalize()
    price = float(quote["price"])
    prev_close = float(hist["close"].iloc[-1])

    open_ = float(quote.get("open") or 0) or price
    high = max(float(quote.get("high") or 0), price, open_)
    low = min([v for v in (quote.get("low"), price, open_) if v and v > 0])

    bar = {
        "open": open_,
        "close": price,
        "high": high,
        "low": low,
        # 报价的成交量单位已在 fetch 层对齐为"手"，与本地库一致
        "volume": float(quote.get("volume") or 0),
        "amount": float(quote.get("amount") or 0),
        "turnover": float("nan"),
        "pct_chg": (price - prev_close) / prev_close * 100 if prev_close else 0.0,
    }

    out = hist.copy()
    if out.index[-1].normalize() == today:
        prev_close = float(out["close"].iloc[-2]) if len(out) > 1 else prev_close
        bar["pct_chg"] = (price - prev_close) / prev_close * 100 if prev_close else 0.0
        for key, value in bar.items():
            if key in out.columns:
                out.iloc[-1, out.columns.get_loc(key)] = value
        return out

    row = pd.DataFrame([bar], index=pd.DatetimeIndex([today], name=out.index.name))
    return pd.concat([out, row[[c for c in out.columns if c in row.columns]]])


def _compute_snapshot(cfg):
    """算一次市场环境快照。失败返回 None（评分退回不做环境修正），不中断监控。"""
    try:
        from .market import compute_market_snapshot

        return compute_market_snapshot(cfg)
    except Exception as exc:  # noqa: BLE001 - 环境快照是加分项，不该让监控起不来
        log.warning("计算市场环境快照失败，本轮不做环境修正: %s", exc)
        return None


def _snapshot_stamp(cfg) -> tuple:
    """快照输入的文件指纹：任一文件被改过就重算。

    盘中的 daily_summary / 指数K线都是"上一收盘"的聚合，一天之内不会变，所以
    缓存几乎总是命中（省掉每轮约 2 秒重算）；而收盘后 update 一跑，mtime 变了
    下一轮就自动拿到新环境，不用重启监控。
    """
    files = [
        store.market_dir(cfg) / name
        for name in ("daily_summary.csv", "limit_up_history.json",
                     "limit_up_streak.json", "sector_map.json")
    ]
    if cfg.data.storage == "sqlite":
        files.append(Path(cfg.data.db_path))
    else:
        kdir = Path(cfg.data.data_dir) / cfg.data.storage / "kline"
        if kdir.exists():
            files.extend(sorted(kdir.glob("idx_*.csv")))

    stamp = []
    for path in files:
        try:
            stat = path.stat()
            stamp.append((path.name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            stamp.append((path.name, 0, 0))
    return tuple(stamp)


class MarketSnapshotCache:
    """按输入文件指纹缓存市场环境快照，供盘中每轮复用。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._stamp: tuple | None = None
        self._snapshot = None

    def get(self):
        stamp = _snapshot_stamp(self.cfg)
        if stamp != self._stamp:
            self._snapshot = _compute_snapshot(self.cfg)
            self._stamp = stamp
        return self._snapshot


def _local_index_quotes(cfg) -> "pd.DataFrame | None":
    """本地指数收盘价（实时接口不可用时的回退）。涨跌幅用尾两根K线算。"""
    rows: list[dict] = []
    st = store.open_store(cfg, readonly=True)
    try:
        for key, code in cfg.market.index_codes.items():
            try:
                k = st.read_kline(code, 2)
            except Exception as exc:  # noqa: BLE001
                log.debug("读指数 %s 失败: %s", code, exc)
                continue
            if len(k) < 2:
                continue
            last, prev = float(k["close"].iloc[-1]), float(k["close"].iloc[-2])
            if prev <= 0:
                continue
            rows.append(
                {
                    "code": code,
                    "name": key,
                    "price": last,
                    "pct_chg": (last / prev - 1) * 100,
                }
            )
    finally:
        st.close()
    return pd.DataFrame(rows) if rows else None


def _local_breadth(cfg) -> "tuple[pd.DataFrame | None, str]":
    """本地 daily_summary 最后交易日的全市场行（回退）。返回 (df, 数据日)。"""
    # 复用 snapshot 的读取（同一个包内），避免第三份 daily_summary 解析逻辑
    try:
        from .market.snapshot import _load_daily_summary

        today, _ = _load_daily_summary(cfg)
    except Exception as exc:  # noqa: BLE001 - 回退路径失败不该影响监控
        log.warning("读取本地日度聚合失败: %s", exc)
        return None, ""
    if today is None or today.empty:
        return None, ""
    return today, str(today["date"].max())


def build_market_panel(cfg, snapshot, index_quotes, spot, spot_at, now):
    """组装盘面面板。

    实时指数 → 本地收盘指数；全市场快照 → 本地 daily_summary，
    两级都用不上时仍返回可渲染的骨架（面板里会写明缺什么）。
    """
    from .market.panel import build_panel
    from .market.sector_fetch import cached_sector_map

    notes: list[str] = []
    index_as_of = ""
    if index_quotes is None or index_quotes.empty:
        index_quotes = _local_index_quotes(cfg)
        if index_quotes is not None:
            notes.append("指数为本地收盘价（实时接口不可用）")
    else:
        index_as_of = now.strftime("%H:%M:%S")

    breadth_source = breadth_as_of = ""
    if spot is not None and not spot.empty:
        breadth_source = "快照"
        breadth_as_of = spot_at.strftime("%H:%M:%S") if spot_at else ""
    else:
        spot, breadth_as_of = _local_breadth(cfg)
        if spot is not None:
            breadth_source = "本地收盘"
            notes.append("涨跌停/成交额/行业榜为本地收盘数据")

    panel = build_panel(
        spot, index_quotes, snapshot, cfg,
        sector_map=cached_sector_map(cfg),
        index_as_of=index_as_of, breadth_as_of=breadth_as_of,
        breadth_source=breadth_source,
    )
    panel.notes.extend(notes)
    return panel


def format_market_panel(cfg, snapshot, index_quotes, spot, spot_at, now) -> str:
    """组装并渲染盘面面板（每轮打印一次）。"""
    from .market.panel import format_panel

    panel = build_market_panel(cfg, snapshot, index_quotes, spot, spot_at, now)
    return format_panel(panel)


def _pick_hit(hits: list[PatternHit], pattern: str) -> PatternHit | None:
    """只认这条目登记的形态。channel 检测器一次出三种形态，不过滤会张冠李戴。"""
    same = [h for h in hits if h.pattern == pattern] or hits
    return max(same, key=lambda h: h.score) if same else None


def check_one(
    cfg,
    entry: dict[str, Any],
    quote: dict[str, Any] | None,
    st,
    market_snapshot=None,
) -> Decision:
    """对单只重跑检测器。这里不改 entry，状态更新统一在 check_once 里做。"""
    code, name = entry["code"], entry.get("name", entry["code"])
    if quote is None:
        return Decision(code, name, NOQUOTE, reason="无报价（停牌或接口未返回）", entry=entry)

    hist = st.read_kline(code, scan.required_bars(cfg))
    if hist.empty:
        return Decision(code, name, NOQUOTE, reason="本地无K线，先跑 update", entry=entry)

    df = build_intraday_df(hist, quote)
    # 市场环境快照与收盘 scan 同一份：盘中评分（综合评分/市场环境/板块列）
    # 必须和当天报表对得上，否则会出现"报表 60 分、监控里 3 分"的分裂。
    ctx = scan.build_context(code, name, df, cfg, explain=True, market_snapshot=market_snapshot)

    pattern = entry.get("pattern", "break_bottom")
    price = float(quote["price"])
    # 复用 scan 的解析：既接受检测器 key，也接受 pattern 名（如 rising_channel → channel）
    try:
        detectors = scan.resolve_detectors([pattern])
    except ValueError:
        # 形态已不在注册表里（例如已整体停用的 break_bottom_v2，或手工写错的形态名）：
        # 它永远判不出结果，留在池里只会每轮刷一条警告、占着名额 → 按失效移出。
        return Decision(
            code, name, INVALID, price=price,
            reason=f"形态 {pattern} 不在可选列表（可能已停用），移出观察池", entry=entry,
        )
    # 必须走 scan_context：阶段折扣 / ADX regime 乘子 / 环境快照 / 时间衰减 /
    # 价格失效判定 / 同股去重全在这一个函数里，直接调 detect() 会把这六样全漏掉，
    # 盘中分数就和收盘报表对不上（实测最多差 17%）。
    hits = scan.scan_context(ctx, detectors)
    hit = _pick_hit(hits, pattern)

    if hit is None:
        reason = ctx.trace[-1] if ctx.trace else "检测器未命中"
        return Decision(code, name, INVALID, price=price, reason=reason, entry=entry)
    # 价格已证伪（收盘跌破结构止损等）要排在 formed 之前判，否则它会掉进下面的
    # FORMING 分支，永远留在池子里当"还在构筑"。
    if hit.pattern_invalid or hit.status == STATUS_INVALID:
        return Decision(
            code, name, INVALID, price=price,
            reason=hit.invalid_reason or "已被价格行为证伪", entry=entry,
        )
    if hit.status == STATUS_FORMED:
        return Decision(code, name, FORMED, hit=hit, price=price, entry=entry)
    return Decision(code, name, FORMING, hit=hit, price=price, entry=entry)


GAP_FACTOR = 3  # 距上次检测超过 GAP_FACTOR×监控间隔，视为监控断档


def _missed_while_away(
    entry: dict, prev_checked: str, now: datetime, interval: int
) -> bool:
    """判断这只“已形成”是不是在监控断档期间形成的——真漏了才走补漏快速通道。

    断档的两种信号：
    - prev_checked 为空、且不是实时扫描刚发现的（即报表/手工带进来的历史形成）；
    - 距上次检测超过 GAP_FACTOR×监控间隔（停跑重启或监控断了几轮）。
    连续盯着、上一轮刚检测过的不算漏，照常走 confirm_rounds 多轮确认。
    """
    if not prev_checked:
        # 从没实时检测过：报表/手工进池的当历史形成补；实时扫描刚冒出来的当正常形成
        return entry.get("source_report", "") != watch.REALTIME_SOURCE
    try:
        prev = datetime.strptime(prev_checked, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return True  # 时间戳坏了，保守当断档
    gap = (now - prev).total_seconds()
    return gap > max(60, interval * GAP_FACTOR)


def check_once(
    cfg, pool: list[dict], quotes: pd.DataFrame, interval: int | None = None,
    market_snapshot=None,
) -> tuple[list[Decision], list[dict]]:
    """跑一轮全池检测，返回 (决策列表, 更新后的池子)。失效条目已从池中剔除。"""
    confirm_rounds = max(1, int(cfg.watch.confirm_rounds))
    interval = int(interval or cfg.watch.interval)
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    quote_map = {r["code"]: r for r in quotes.to_dict("records")} if not quotes.empty else {}

    decisions: list[Decision] = []
    kept: list[dict] = []

    with store.open_store(cfg, readonly=True) as st:
        for entry in pool:
            try:
                decision = check_one(
                    cfg, entry, quote_map.get(entry["code"]), st, market_snapshot
                )
            except Exception as exc:  # noqa: BLE001 - 单只异常不该打断整轮监控
                log.warning("检测 %s 失败: %s", entry["code"], exc)
                kept.append(entry)
                continue

            if decision.kind == NOQUOTE:
                decisions.append(decision)
                kept.append(entry)
                continue

            prev_checked = entry.get("last_checked", "")
            entry["last_checked"] = stamp
            entry["last_price"] = decision.price

            if decision.kind == INVALID:
                entry["last_status"] = "已失效"
                decisions.append(decision)
                continue  # 不进 kept，即移出池子

            entry["last_score"] = round(decision.hit.composite_score(), 1)
            if decision.kind == FORMED:
                entry["last_status"] = "已形成"
                if not entry.get("promoted_at") and _missed_while_away(
                    entry, prev_checked, now, interval
                ):
                    # 停跑/盘中断档期间已经形成的：这轮直接确认到位，本轮即提醒买入
                    entry["confirm_streak"] = confirm_rounds
                    log.info("%s 断档期间已形成，补提醒", entry["code"])
                else:
                    entry["confirm_streak"] = int(entry.get("confirm_streak", 0)) + 1
                if entry["confirm_streak"] >= confirm_rounds and not entry.get("promoted_at"):
                    entry["promoted_at"] = stamp
                    decision.kind = PROMOTED
            else:
                entry["last_status"] = "构筑中"
                entry["confirm_streak"] = 0

            decisions.append(decision)
            kept.append(entry)

    return decisions, kept


def _v2_trail_ref(cfg, hist: pd.DataFrame) -> float | None:
    """v2 口径持仓的移动止盈参考位：`max(MA20 − k×ATR, 最近SwingLow − k×ATR)`。

    形态的失效位由结构给定，与"从最高价回撤 trail_pct"无关。阈值取
    `break_bottom_v2.trail_ma_atr_k / trail_swing_atr_k`，和检测器写进
    `trailing_stop` 列的是同一套算法。
    """
    from . import indicators, swings

    if hist is None or hist.empty or len(hist) < 20:
        return None
    df = indicators.compute_indicators(hist, cfg)
    v2c = cfg.break_bottom_v2
    atr_last = float(df["atr"].iloc[-1])
    if not atr_last > 0:  # NaN 也会落到这里
        return None
    ma20 = float(df["ma20"].iloc[-1])
    swing_low = 0.0
    for pivot in reversed(swings.zigzag(df, cfg)):
        if pivot.is_low and pivot.price > 0:
            swing_low = float(pivot.price)
            break
    candidates = []
    if ma20 > 0:
        candidates.append(ma20 - float(v2c.trail_ma_atr_k) * atr_last)
    if swing_low > 0:
        candidates.append(swing_low - float(v2c.trail_swing_atr_k) * atr_last)
    positive = [v for v in candidates if v > 0]
    return max(positive) if positive else None


def _trading_days_since(
    hist: pd.DataFrame | None, start_date: str, now: datetime | None = None
) -> int | None:
    """start_date 之后经过的**交易日**数（失败突破检测用）。

    交易日历取本地 K 线的日期序列——和检测器按 bar 序号判失败的是一家。
    盘中 hist 还没有当日 K 线，若"今天"是交易日就把它补进去。旧实现按自然日
    算，周末/连休会把非交易日也计进去，让持仓比检测器更早触发"失败突破"。

    取不到 K 线时返回 None（调用方据此跳过失败突破判定）——没有交易日历就
    数不出交易日，猜一个数不如不判，剩余风险由止损兜住。
    """
    if not start_date or hist is None or hist.empty:
        return None
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in hist.index]
    elapsed = sum(1 for d in dates if d > start_date)
    # 进行中的交易日：当日 K 线还没入库，补上今天（周末/节假日不补）
    if start_date < today and today not in dates and now.weekday() < 5:
        elapsed += 1
    return elapsed


def handle_trading(
    cfg, decisions: list[Decision], quotes: pd.DataFrame
) -> list[tuple[trade.Position, str, float]]:
    """处理交易逻辑：PROMOTED 时开仓，每轮检查持仓退出条件。

    返回本轮的交易记录列表 [(position, reason, exit_price), ...]
    """
    if not cfg.trade.enabled:
        return []

    quote_map = {r["code"]: r for r in quotes.to_dict("records")} if not quotes.empty else {}
    trades_made: list[tuple[trade.Position, str, float]] = []

    # 1. 开仓：PROMOTED 的股票
    for d in decisions:
        if d.kind != PROMOTED or d.hit is None:
            continue

        levels = d.hit.key_levels
        # 旧口径写"参考止损"，v2 写 stop_price；两者都没有就跳过
        stop_loss = levels.get("参考止损", 0) or levels.get("stop_price", 0) or 0
        tp1 = float(levels.get("tp1_price", 0) or 0)
        tp2 = float(levels.get("tp2_price", 0) or 0)
        primary = d.hit._get_target_price() or 0

        if tp1 > 0:
            # v2 口径：TP1 到价分批止盈、TP2 到价清仓。没有 TP2 时不给固定目标价，
            # 剩余仓位交给移动止盈（否则会在 TP1 把剩下的也一次性清掉，TP1 的分批失去意义）。
            target = tp2 if tp2 > tp1 else 0
            nearest_target = tp1
        else:
            target = primary
            nearest_target = primary

        if 0 < target <= d.price:
            log.debug("%s 目标价 %.2f 未高于现价 %.2f，本次不设目标", d.code, target, d.price)
            target = 0

        if stop_loss <= 0:
            log.warning("%s 无止损价，跳过开仓", d.code)
            continue

        price = d.price

        min_upside = float(cfg.trade.min_target_upside)
        if nearest_target > 0 and (nearest_target - price) / price < min_upside:
            log.info(
                "%s 距目标价 %.2f 仅 %.1f%%（< %.0f%%），上涨空间不足，跳过",
                d.code, nearest_target, (nearest_target - price) / price * 100, min_upside * 100,
            )
            continue

        max_drift = float(cfg.trade.max_entry_drift)
        if stop_loss > 0 and (price - stop_loss) / stop_loss > max_drift:
            log.info(
                "%s 现价 %.2f 距形成点位（止损 %.2f）偏离 %.1f%%（> %.0f%%），跳过",
                d.code, price, stop_loss, (price - stop_loss) / stop_loss * 100, max_drift * 100,
            )
            continue

        # RR1 闸门：只对带第一目标的 v2 口径生效（旧口径 tp1=0，自动跳过，行为不变）。
        # 用**现价**而不是检测器记录的 B1 价重算——信号发出后价格已经漂移，
        # 拿 B1 时刻的 rr1 做否决会把已经变差的入场放进来。
        if tp1 > 0:
            min_rr1 = float(getattr(cfg.trade, "min_rr1", 0))
            if price <= stop_loss:
                log.info("%s 现价 %.2f 已在止损 %.2f 下方，跳过", d.code, price, stop_loss)
                continue
            rr1_now = (tp1 - price) / (price - stop_loss)
            if min_rr1 and rr1_now < min_rr1:
                log.info(
                    "%s RR1=%.2f（< %.2f，TP1 %.2f / 现价 %.2f / 止损 %.2f），性价比不足，跳过",
                    d.code, rr1_now, min_rr1, tp1, price, stop_loss,
                )
                continue

        position = trade.open_position(
            cfg,
            code=d.code,
            name=d.name,
            pattern=d.hit.pattern,
            pattern_cn=d.hit.pattern_cn,
            entry_price=d.price,
            stop_loss=stop_loss,
            target_price=target,
            entry_score=d.hit.composite_score(),
            tp1_price=tp1,
            tp2_price=tp2,
        )
        if position:
            # B2 开仓：存储突破价位与**突破 bar 日期**，用于失败突破检测
            breakout = 0.0
            if d.hit.pattern == "break_bottom_b2":
                breakout = levels.get("第一波高点", 0)
            elif d.hit.pattern == "break_bottom_v2_b2":
                breakout = levels.get("H1", 0)
            if breakout > 0:
                position.breakout_level = breakout
                # 取突破 bar 的日期，而不是开仓日：检测器的失败突破窗口从突破 bar
                # 起算，交易层若从开仓日起算，开仓越晚给的窗口越宽，两层会错位。
                # B2 已形成时 signal_date 就是突破 bar 的日期（旧 b2 取 h1_breakout_idx、
                # v2_b2 取 b2_idx，见各自检测器的 signal_idx）。
                position.breakout_date = pd.Timestamp(d.hit.signal_date).strftime("%Y-%m-%d")
                positions = trade.load_positions(cfg)
                positions = [p if p.code != position.code else position for p in positions]
                trade.save_positions(cfg, positions)

            print(
                f"  ▶ 买入 {d.code} {d.name} {position.shares}股 @ {d.price:.2f}"
                f"  止损{stop_loss:.2f} 目标{target:.2f}"
            )

    # 2. 检查持仓退出
    positions = trade.load_positions(cfg)
    survivors: list[trade.Position] = []
    changed = False
    for pos in positions:
        quote = quote_map.get(pos.code)
        if quote is None:
            survivors.append(pos)
            continue

        current_price = float(quote["price"])
        old_stop = pos.stop_loss

        # 获取 MA20 / 最近有效 HL（分批止盈后的移动止盈）、v2 口径的结构止损位，
        # 以及突破后经过的【交易日】数（失败突破检测用，见 _trading_days_since）。
        ma20_val = None
        recent_hl = None
        trail_ref = None
        trading_day_count = None
        need_hist = (
            pos.partial_closed
            or pos.tp1_price > 0
            or (pos.breakout_level > 0 and pos.breakout_date)
        )
        if need_hist:
            hist = None
            try:
                from . import scan as scan_mod
                from . import store as store_mod
                with store_mod.open_store(cfg, readonly=True) as st:
                    hist = st.read_kline(pos.code, scan_mod.required_bars(cfg))
            except Exception:
                hist = None
            if hist is not None and not hist.empty and len(hist) >= 20:
                ma20_val = float(hist["close"].iloc[-20:].mean())
                if pos.tp1_price > 0:
                    trail_ref = _v2_trail_ref(cfg, hist)
            if pos.breakout_level > 0 and pos.breakout_date:
                trading_day_count = _trading_days_since(hist, pos.breakout_date)
            if pos.partial_closed or pos.tp1_price > 0:
                # 从 key_levels 推断 recent_hl（回踩低点 L1 或参考低点 L0）
                # 持仓本身不存 key_levels，用 breakout_level 作为 H1 参考
                # recent_hl 在 B2 场景下 = L1，近似用止损价作为参考
                recent_hl = pos.stop_loss if pos.stop_loss > 0 else None

        exit_reason = trade.check_exit(
            pos, current_price, cfg,
            trading_day_count=trading_day_count,
            ma20=ma20_val, recent_hl=recent_hl, trail_ref=trail_ref,
        )

        if exit_reason == "分批止盈":
            partial_pct = float(getattr(cfg.trade, "partial_close_pct", 0.5))
            shares_to_sell = int(pos.shares * partial_pct) // 100 * 100
            if shares_to_sell <= 0:
                shares_to_sell = pos.shares
            if shares_to_sell >= pos.shares:
                closed = trade.close_position(cfg, pos, current_price, exit_reason)
                trades_made.append((pos, exit_reason, current_price))
                changed = True
                emoji = "+" if closed.pnl >= 0 else ""
                print(
                    f"  ◀ 卖出 {closed.code} {closed.name} {closed.shares}股 @ {current_price:.2f}"
                    f"  {exit_reason}  盈亏{emoji}{closed.pnl:.2f} ({emoji}{closed.pnl_pct:.1f}%)"
                )
            else:
                partial = trade.partial_close_position(cfg, pos, current_price, shares_to_sell, exit_reason)
                trades_made.append((pos, exit_reason, current_price))
                changed = True
                emoji = "+" if partial.pnl >= 0 else ""
                print(
                    f"  ◈ 分批卖出 {partial.code} {partial.name} {shares_to_sell}股 @ {current_price:.2f}"
                    f"  {exit_reason}  盈亏{emoji}{partial.pnl:.2f} ({emoji}{partial.pnl_pct:.1f}%)"
                    f"  剩余{pos.shares}股"
                )
                # 分批止盈后仍需将持仓加入 survivors，以便保存剩余仓位
                survivors.append(pos)
            continue

        if exit_reason:
            closed = trade.close_position(cfg, pos, current_price, exit_reason)
            trades_made.append((pos, exit_reason, current_price))
            changed = True
            emoji = "+" if closed.pnl >= 0 else ""
            print(
                f"  ◀ 卖出 {closed.code} {closed.name} {closed.shares}股 @ {current_price:.2f}"
                f"  {exit_reason}  盈亏{emoji}{closed.pnl:.2f} ({emoji}{closed.pnl_pct:.1f}%)"
            )
            continue

        if trade._is_today_entry(pos):
            if current_price <= pos.stop_loss or (
                pos.target_price > 0 and current_price >= pos.target_price
            ):
                log.info(
                    "%s 触发T+1限制：当日买入不可卖出，持仓过夜",
                    pos.code,
                )

        if pos.stop_loss != old_stop:
            changed = True
            log.debug(
                "%s 移动止损更新: %.2f → %.2f (最高 %.2f)",
                pos.code, old_stop, pos.stop_loss, pos.highest_price,
            )
        survivors.append(pos)

    if changed:
        trade.save_positions(cfg, survivors)

    return trades_made


# ---------------------------------------------------------------------------
# 提醒输出
# ---------------------------------------------------------------------------


def alert_path(cfg, stamp: str | None = None) -> Path:
    stamp = stamp or date.today().strftime("%Y%m%d")
    return Path(cfg.data.out_dir) / f"alerts_{stamp}.csv"


def _alert_row(decision: Decision) -> dict[str, Any]:
    hit = decision.hit
    levels = hit.key_levels if hit else {}
    # v2 口径的键名是 L0 / stop_price / tp1_price，旧口径是 参考低点 / 参考止损 / 上方压力。
    # ALERT_COLUMNS 保持不变（DictWriter 遇到未知字段会直接报错），只做取值回退。
    ref = levels.get("参考低点") or levels.get("L0") or 0.0
    recovered = (decision.price - ref) / ref * 100 if ref else 0.0
    stop = levels.get("参考止损") or levels.get("stop_price") or 0.0
    target = levels.get("上方压力") or levels.get("tp1_price") or 0.0
    return {
        "触发时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "代码": decision.code,
        "名称": decision.name,
        "形态": hit.pattern_cn if hit else "",
        "现价": round(decision.price, 2),
        "参考低点": round(ref, 2),
        "收复幅度%": round(recovered, 2),
        "参考止损": round(stop, 2),
        "上方压力": round(target, 2),
        "综合评分": round(hit.composite_score(), 1) if hit else 0.0,
        "确认轮次": decision.entry.get("confirm_streak", 0),
        "说明": hit.detail if hit else decision.reason,
    }


def write_alerts(cfg, decisions: list[Decision]) -> Path | None:
    """追加写当日提醒清单。文件不存在时先写表头。"""
    rows = [_alert_row(d) for d in decisions if d.kind == PROMOTED]
    if not rows:
        return None
    path = alert_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALERT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return path


def print_alert(decision: Decision, bell: bool = False) -> None:
    row = _alert_row(decision)
    bar = "=" * 64
    print(f"\n{bar}")
    print(
        f"★ 站稳成型  {row['代码']} {row['名称']}  【{row['形态']}】"
        f"  {row['触发时间'][11:]}"
    )
    print(
        f"  现价 {row['现价']}  已收复参考低点 {row['参考低点']} "
        f"(+{row['收复幅度%']}%)  综合 {row['综合评分']}"
    )
    print(f"  止损 {row['参考止损']}   上方压力 {row['上方压力']}")
    print(f"  {row['说明']}")
    print(bar)
    if bell:
        print("\a", end="", flush=True)


def print_round(
    decisions: list[Decision],
    pool_size: int,
    position_map: dict[str, trade.Position] | None = None,
    quote_map: dict[str, dict] | None = None,
    verbose: bool = False,
) -> None:
    counts = {k: 0 for k in (PROMOTED, FORMED, FORMING, INVALID, NOQUOTE)}
    for d in decisions:
        counts[d.kind] = counts.get(d.kind, 0) + 1

    held_count = sum(1 for d in decisions if d.code in (position_map or {}))
    held_info = f"  持仓{held_count:>2}" if held_count else ""
    print(
        f"{datetime.now():%H:%M:%S}  池{pool_size:>3}{held_info}"
        f"  已形成{counts[FORMED] + counts[PROMOTED]:>3}"
        f"  构筑中{counts[FORMING]:>3}"
        f"  失效{counts[INVALID]:>3}"
        f"  无报价{counts[NOQUOTE]:>3}"
        f"  触发{counts[PROMOTED]:>3}"
    )

    # 显示每只股票的详细信息
    for d in sorted(decisions, key=lambda x: (-x.entry.get("last_score", 0), x.code)):
        if d.kind == INVALID:
            print(f"  ✗ {d.code} {d.name:<8} 失效  {d.reason}")
        elif d.kind == NOQUOTE:
            print(f"  - {d.code} {d.name:<8} 跳过  {d.reason}")
        elif d.hit:
            levels = d.hit.key_levels
            pattern_cn = d.hit.pattern_cn
            status = d.hit.status_cn
            score = d.entry.get("last_score", 0)
            confirm = d.entry.get("confirm_streak", 0)

            # 根据形态类型显示关键价位
            if d.hit.pattern in ("break_bottom", "breakdown_reversal"):
                ref_low = levels.get("参考低点", 0)
                min_low = levels.get("破底最低", 0)
                resistance = levels.get("上方压力", 0)
                stop = levels.get("参考止损", 0)
                recovered = (d.price - ref_low) / ref_low * 100 if ref_low else 0
                detail = (
                    f"参考低点{ref_low:.2f} 破底至{min_low:.2f} "
                    f"已收复{recovered:+.1f}% "
                    f"压力{resistance:.2f} 止损{stop:.2f}"
                )
            elif d.hit.pattern == "break_bottom_b2":
                ref_low = levels.get("收复位", 0)
                h1 = levels.get("第一波高点", 0)
                l1 = levels.get("回踩低点", 0)
                target = levels.get("等幅目标", 0)
                stop = levels.get("参考止损", 0)
                dist_h1 = (h1 - d.price) / h1 * 100 if h1 else 0
                detail = (
                    f"收复位{ref_low:.2f} H1={h1:.2f} L1={l1:.2f} "
                    f"距H1{dist_h1:+.1f}% 目标{target:.2f} 止损{stop:.2f}"
                )
            elif d.hit.pattern in ("break_bottom_v2", "break_bottom_v2_b2"):
                l0 = levels.get("L0", 0)
                l1 = levels.get("L1", 0)
                h1 = levels.get("H1", 0)
                tp1 = levels.get("tp1_price", 0)
                tp2 = levels.get("tp2_price", 0)
                stop = levels.get("stop_price", 0)
                rr1 = levels.get("rr1", 0)
                dist_h1 = (h1 - d.price) / h1 * 100 if h1 else 0
                detail = (
                    f"L0={l0:.2f} L1={l1:.2f} H1={h1:.2f} "
                    f"距H1{dist_h1:+.1f}% TP1={tp1:.2f} TP2={tp2:.2f} "
                    f"止损{stop:.2f} RR1={rr1:.2f}"
                )
                if d.hit.pattern == "break_bottom_v2_b2":
                    detail = (
                        f"L0={l0:.2f} H1={h1:.2f} 回踩{levels.get('PL', 0):.2f} "
                        f"B2={levels.get('B2', 0):.2f} 距H1{dist_h1:+.1f}% "
                        f"TP1={tp1:.2f} 止损{stop:.2f} RR1={rr1:.2f}"
                    )
            elif d.hit.pattern == "w_bottom":
                neck = levels.get("颈线", 0)
                left = levels.get("左底", 0)
                right = levels.get("右底", 0)
                target = levels.get("等幅目标", 0)
                stop = levels.get("参考止损", 0)
                dist_neck = (neck - d.price) / neck * 100 if neck else 0
                detail = (
                    f"双底{left:.2f}/{right:.2f} 颈线{neck:.2f} "
                    f"距颈线{dist_neck:+.1f}% "
                    f"目标{target:.2f} 止损{stop:.2f}"
                )
            elif d.hit.pattern in ("rising_channel", "falling_channel", "box_range"):
                upper = levels.get("上轨", 0)
                lower = levels.get("下轨", 0)
                mid = levels.get("中轴", 0)
                pos = levels.get("通道位置", 0.5)
                detail = (
                    f"上轨{upper:.2f} 中轴{mid:.2f} 下轨{lower:.2f} "
                    f"位置{pos:.0%}"
                )
            elif d.hit.pattern == "m_top":
                neck = levels.get("颈线", 0)
                left = levels.get("左头", 0)
                right = levels.get("右头", 0)
                target = levels.get("等幅目标", 0)
                dist_neck = (d.price - neck) / neck * 100 if neck else 0
                detail = (
                    f"双顶{left:.2f}/{right:.2f} 颈线{neck:.2f} "
                    f"距颈线{dist_neck:+.1f}% 目标{target:.2f}"
                )
            elif d.hit.pattern in ("platform_breakout", "breakout_retest", "cup_handle"):
                resistance = levels.get("突破位", 0)
                support = levels.get("整理下沿", 0)
                target = levels.get("等幅目标", 0)
                stop = levels.get("参考止损", 0)
                dist = (resistance - d.price) / resistance * 100 if resistance else 0
                detail = (
                    f"突破位{resistance:.2f} 下沿{support:.2f} "
                    f"目标{target:.2f} 止损{stop:.2f} 距突破{dist:+.1f}%"
                )
            elif d.hit.pattern == "trend_pullback":
                support = levels.get("回踩支撑", 0)
                recent_high = levels.get("近期高点", 0)
                target = levels.get("等幅目标", 0)
                stop = levels.get("参考止损", 0)
                pb_pct = levels.get("回撤幅度%", 0)
                detail = (
                    f"支撑{support:.2f} 高点{recent_high:.2f} "
                    f"回撤{pb_pct:.1f}% 目标{target:.2f} 止损{stop:.2f}"
                )
            elif d.hit.pattern == "falling_channel_reversal":
                lower = levels.get("下轨", 0)
                upper = levels.get("上轨", 0)
                target = levels.get("等幅目标", 0)
                stop = levels.get("参考止损", 0)
                detail = (
                    f"下降通道反转 下轨{lower:.2f} 上轨{upper:.2f} "
                    f"目标{target:.2f} 止损{stop:.2f}"
                )
            else:
                detail = d.hit.detail[:40] if d.hit.detail else ""

            # 状态标记
            status_mark = "★" if d.kind == PROMOTED else ("●" if d.kind == FORMED else "○")
            pos = (position_map or {}).get(d.code)
            pos_tag = ""
            if pos:
                pnl_pct = (d.price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price else 0
                emoji = "+" if pnl_pct >= 0 else ""
                pos_tag = (
                    f"  ◈持仓{pos.shares}股 成本{pos.entry_price:.2f}"
                    f" 盈亏{emoji}{pnl_pct:.1f}% 止损{pos.stop_loss:.2f}"
                )
            print(
                f"  {status_mark} {d.code} {d.name:<8} {pattern_cn:<6} {status} "
                f"现价{d.price:>7.2f} 评分{score:>5.1f} 确认{confirm} "
                f"{detail}{pos_tag}"
            )

    # 显示不在观察池但仍在持仓的股票
    if position_map and quote_map:
        decision_codes = {d.code for d in decisions}
        orphans = [p for code, p in position_map.items() if code not in decision_codes]
        if orphans:
            print(f"  ── 持仓（不在观察池）──")
            for pos in orphans:
                quote = quote_map.get(pos.code)
                price = float(quote["price"]) if quote else 0
                if price > 0:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    emoji = "+" if pnl_pct >= 0 else ""
                    print(
                        f"  ◈ {pos.code} {pos.name:<8} {pos.shares}股 "
                        f"成本{pos.entry_price:.2f} 现价{price:.2f} "
                        f"盈亏{emoji}{pnl_pct:.1f}% 止损{pos.stop_loss:.2f}"
                    )
                else:
                    print(f"  ◈ {pos.code} {pos.name:<8} {pos.shares}股 成本{pos.entry_price:.2f} 无报价")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def run_monitor(
    cfg,
    interval: int | None = None,
    once: bool = False,
    verbose: bool = False,
    bell: bool | None = None,
    repeat: bool = False,
    panel: bool = True,
) -> int:
    """轮询主循环：不间断取实时报价重跑检测器，直到 Ctrl-C。

    once=True 只跑一轮，便于手工验证。
    panel=True 每轮打印盘面概览（指数/涨跌停/成交额/行业榜）。
    """
    interval = interval or int(cfg.watch.interval)
    bell = cfg.watch.bell if bell is None else bell
    sources = list(cfg.watch.quote_sources)
    refresh_interval = max(0, int(getattr(cfg.watch, "pool_refresh_interval", 0)))

    pool = watch.load_pool(cfg)
    if not pool and not refresh_interval:
        print("观察池为空。先跑: python cli.py watch sync")
        return 1
    if not pool:
        print("观察池为空，将先用实时价格扫描建立候选池。")

    if repeat:  # 清掉已提醒标记，让站稳的票每轮都再报一次
        for entry in pool:
            entry["promoted_at"] = ""

    snapshot_cache = MarketSnapshotCache(cfg)
    market_snapshot = snapshot_cache.get()

    print(
        f"监控启动：观察池 {len(pool)} 只，间隔 {interval}s，"
        f"池刷新 {refresh_interval or '关闭'}s，"
        f"确认 {cfg.watch.confirm_rounds} 轮，报价源 {'/'.join(sources)}"
    )
    if not panel and market_snapshot is not None:
        # 面板关掉时给一行环境摘要，保持"情绪进来了"这件事可见
        print(
            f"市场环境：{market_snapshot.regime_label} {market_snapshot.regime_score:.1f} | "
            f"情绪 {market_snapshot.sentiment_score:.1f} | "
            f"行业 {len(market_snapshot.sector_scores)} 个 | 乘子 "
            f"市场{market_snapshot.market_factor:.3f}/情绪{market_snapshot.sentiment_factor:.3f}"
        )
    elif not panel:
        print("市场环境：未取到（评分不做环境修正）")
    print("Ctrl-C 退出（退出前会保存池子状态）\n")

    last_quotes = pd.DataFrame(columns=fetch.QUOTE_COLUMNS)
    last_pool_refresh = 0.0
    rounds = 0
    last_spot: pd.DataFrame | None = None
    last_spot_at: datetime | None = None
    try:
        while True:
            rounds += 1
            now = time.monotonic()
            # 环境快照的输入（收盘聚合 + 指数K线）盘中不变，命中缓存；
            # 收盘后 update 一跑指纹就变了，下一轮自动取到新环境。
            market_snapshot = snapshot_cache.get()
            refresh_due = bool(refresh_interval) and (
                rounds == 1 or now - last_pool_refresh >= refresh_interval
            )
            if refresh_due:
                try:
                    spot = fetch.fetch_spot(cfg)
                    last_spot, last_spot_at = spot, datetime.now()
                    hits = scan.scan_market_realtime(
                        cfg,
                        spot,
                        statuses=None,
                        min_score=None,
                        market_snapshot=market_snapshot,
                    )
                    stats = watch.sync_from_hits(cfg, hits)
                    pool = watch.load_pool(cfg)
                    last_pool_refresh = now
                    print(
                        f"  池子刷新：实时命中 {stats['候选']}，新增 {stats['新增']}，"
                        f"更新 {stats['更新']}，池内 {stats['池内']}"
                    )
                except Exception as exc:  # noqa: BLE001 - 刷新失败不应中断盘中监控
                    log.warning("第%d轮池子刷新失败，沿用旧池: %s", rounds, exc)

            # 指数实时点位：1 个请求，失败就回退本地收盘价（面板会标出来源）
            index_quotes = None
            if panel:
                try:
                    index_quotes = fetch.fetch_index_quotes(
                        list(cfg.market.index_codes.values()), cfg, sources=sources
                    )
                except fetch.FetchError as exc:
                    log.debug("第%d轮指数报价不可用，回退本地: %s", rounds, exc)

            codes = [e["code"] for e in pool]
            # 持仓的股票可能已不在观察池，也要取报价才能检查止盈止损
            if cfg.trade.enabled:
                pos_codes = [p.code for p in trade.load_positions(cfg)]
                codes = list(dict.fromkeys(codes + pos_codes))  # 去重保序
            try:
                last_quotes = fetch.fetch_quotes(codes, cfg, sources=sources)
            except fetch.FetchError as exc:
                log.warning("第%d轮取报价失败，沿用缓存: %s", rounds, exc)

            decisions, pool = check_once(
                cfg, pool, last_quotes, interval=interval, market_snapshot=market_snapshot
            )
            watch.save_pool(cfg, pool)

            # 构建持仓映射，用于监控显示
            position_map: dict[str, trade.Position] = {}
            if cfg.trade.enabled:
                for p in trade.load_positions(cfg):
                    position_map[p.code] = p

            q_map = {r["code"]: r for r in last_quotes.to_dict("records")} if not last_quotes.empty else {}
            if panel:
                # 每轮重打：指数实时刷新，涨跌停/成交额/行业榜沿用池刷新那次全市场快照
                # （每分钟翻 55 页会被东财限流），面板上标了各自的数据时间。
                print(format_market_panel(
                    cfg, market_snapshot, index_quotes, last_spot, last_spot_at, datetime.now()
                ))
            print_round(decisions, len(pool), position_map=position_map, quote_map=q_map, verbose=verbose)
            for decision in decisions:
                if decision.kind == PROMOTED:
                    print_alert(decision, bell=bell)
            path = write_alerts(cfg, decisions)
            if path:
                print(f"  提醒已追加写入 {path}\n")

            # 处理模拟交易
            handle_trading(cfg, decisions, last_quotes)

            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        watch.save_pool(cfg, pool)
        print(f"\n已中断，池子状态已保存（{len(pool)} 只）。")
    return 0
