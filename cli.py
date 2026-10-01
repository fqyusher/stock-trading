#!/usr/bin/env python3
"""A股K线形态扫描 CLI。

  python cli.py update --full          首次全量拉取
  python cli.py update                 每日收盘后增量更新
  python cli.py scan                   全市场扫描全部形态
  python cli.py scan --code 000001 --explain --plot
  python cli.py info                   查看本地库状态
  python cli.py watch sync             把构筑中的形态同步进观察池
  python cli.py watch run              盘中监控，站稳成型即提醒
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from tradeyh import config, monitor, report, scan, store, trade, update, watch  # noqa: E402
from tradeyh.patterns import DETECTORS, PATTERN_KEYS  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _storage_missing(cfg) -> bool:
    """本地库还没建过时给一句可操作的提示，而不是抛 sqlite/文件层的原始异常。"""
    if store.storage_exists(cfg):
        return False
    print(f"本地库不存在: {store.storage_path(cfg)}\n先运行: python cli.py update --full")
    return True


def _want_plot(cfg, args) -> bool:
    """出图开关优先级：--plot / --no-plot 显式指定 > charts.enabled 配置。

    charts.enabled 默认 false——出图要读全量K线、写 png，全市场扫描时又慢又占地，
    所以默认只出报表。需要看图时显式 --plot（cron 走这条），或者把 charts.enabled
    打开，让它变成长期的默认行为。
    """
    return cfg.charts.enabled if args.plot is None else args.plot


def cmd_update(args, cfg) -> int:
    update.run_update(
        cfg,
        full=args.full,
        limit=args.limit,
        codes=_split(args.codes),
        retry_failed=args.retry_failed,
    )
    return 0


def cmd_update_index(args, cfg) -> int:
    """单独拉取指数 K 线（沪深300/中证1000/创业板指）。"""
    result = update.update_index_klines(cfg)
    ok, fail = result["index_ok"], result["index_fail"]
    if fail:
        print(f"指数更新完成: 成功 {ok}, 失败 {fail}（网络问题请稍后重试）")
    else:
        print(f"指数更新完成: 全部成功 ({ok} 只)")
    return 1 if fail else 0


def cmd_fetch_sectors(args, cfg) -> int:
    """从 baostock 拉取行业分类并缓存。"""
    from tradeyh.market.sector_fetch import fetch_sector_map_baostock, save_sector_map
    data = fetch_sector_map_baostock()
    save_sector_map(cfg, data)
    print(f"已保存 {len(data)} 只股票的行业分类到 data/market/sector_map.json")
    return 0


def cmd_rebuild_market(args, cfg) -> int:
    """从本地 K 线重建 daily_summary / 涨停历史 / 行业热度历史（不联网）。"""
    from tradeyh.market.backfill import rebuild_market_data
    stats = rebuild_market_data(
        cfg,
        days=args.days,
        workers=args.workers,
        codes=_split(args.codes),
    )
    if not stats["rows"]:
        print("没有读到任何 K 线，data/market 下的文件未改动（先跑 update）")
        return 1
    lo, hi = stats.get("range", ("", ""))
    print(
        f"已重建 {stats['codes']} 只 / {stats['days']} 个交易日 "
        f"({lo} ~ {hi}), {stats['rows']} 行, 行业热度历史 {stats['heat_days']} 天, "
        f"用时 {stats['elapsed']:.1f}s"
    )
    return 0


def cmd_scan(args, cfg) -> int:
    if _storage_missing(cfg):
        return 1

    patterns = _split(args.patterns)

    if args.code:
        return _scan_single(args, cfg, patterns)

    codes = _split(args.codes)
    hits = scan.scan_market(
        cfg,
        codes=codes,
        detector_keys=patterns,
        statuses=_split(args.status),
        min_score=args.min_score,
        workers=args.workers,
    )
    report.print_summary(hits, top=args.top)
    stamp = args.stamp or date.today().strftime("%Y%m%d")
    paths = report.write_report(hits, cfg.data.out_dir, stamp)
    for kind, path in paths.items():
        print(f"{kind}: {path}")

    if _want_plot(cfg, args):
        from tradeyh import plot as plot_mod
        from tradeyh.report import long_only

        # 默认只给做多榜单出图：做空形态不进候选，画了也没人看
        chart_src = long_only(hits) if args.plot_long_only else hits
        max_charts = args.plot_top  # None = 用 config 里的 charts.max_charts
        # 图按报表同一个 stamp 分组到日期子目录，同日重跑覆盖同名文件
        created = plot_mod.generate_charts(
            cfg, chart_src, max_charts=max_charts, stamp=stamp
        )
        chart_dir = plot_mod.charts_dir(cfg, stamp)
        print(f"标注图 {len(created)} 张，目录: {chart_dir}")
        if created and (chart_dir / "index.html").exists():
            print(f"索引页: {chart_dir / 'index.html'}")
    return 0


def _scan_single(args, cfg, patterns) -> int:
    hits, ctx = scan.scan_one(cfg, args.code, patterns, explain=args.explain)
    if ctx is None:
        return 1

    print(f"\n{args.code} {ctx.name}  本地K线 {ctx.n} 根  "
          f"{ctx.df.index[0]:%Y-%m-%d} ~ {ctx.df.index[-1]:%Y-%m-%d}")
    print(f"最新收盘 {ctx.df['close'].iloc[-1]:.2f}  "
          f"MA20 {ctx.df['ma20'].iloc[-1]:.2f}  摆动点 {len(ctx.pivots)} 个")

    if args.explain:
        print("\n--- 摆动点(ZigZag) ---")
        for pivot in ctx.pivots[-12:]:
            kind = "峰" if pivot.is_high else "谷"
            flag = "" if pivot.confirmed else " (未确认)"
            print(f"  {pivot.date:%Y-%m-%d} {kind} {pivot.price:8.2f}{flag}")

        print("\n--- 判定过程 ---")
        if ctx.trace:
            for line in ctx.trace:
                print(f"  {line}")
        else:
            print("  (无拒绝记录)")

    if not hits:
        print("\n未命中任何形态。")
        return 0

    print("\n--- 命中 ---")
    for hit in sorted(hits, key=lambda h: h.score, reverse=True):
        print(
            f"\n[{hit.pattern_cn} / {hit.status_cn}] 评分 {hit.score:.1f} "
            f"(几何{hit.geom_score:.0f} 量能{hit.vol_score:.0f} 趋势{hit.trend_score:.0f}) "
            f"信号日 {pd.Timestamp(hit.signal_date):%Y-%m-%d}"
        )
        print(f"  {hit.detail}")
        levels = "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in hit.key_levels.items()
        )
        print(f"  关键价位: {levels}")

    if args.plot:
        from tradeyh import plot as plot_mod

        for hit in hits:
            path = plot_mod.plot_hit(
                ctx, hit, plot_mod.charts_pattern_dir(cfg, hit.pattern, args.stamp)
            )
            if path:
                print(f"  图: {path}")
    return 0


def cmd_watch(args, cfg) -> int:
    action = args.watch_action

    if action == "sync":
        stats = watch.sync_from_report(
            cfg,
            report_path=args.report,
            patterns=_split(args.patterns),
            statuses=_split(args.status),
            min_score=args.min_score,
        )
        print(f"报表  {stats['报表']}")
        print(
            f"候选 {stats['候选']} 只  →  新增 {stats['新增']}  "
            f"更新 {stats['更新']}  池内合计 {stats['池内']}"
        )
        print(f"池文件 {watch.pool_path(cfg)}")
        return 0

    if action == "list":
        print(watch.describe(watch.load_pool(cfg)))
        return 0

    if action == "add":
        if _storage_missing(cfg):
            return 1
        stats = watch.add(cfg, _split(args.codes) or [], pattern=args.pattern)
        print(f"新增 {stats['新增'] or '无'}  已存在 {stats['已存在'] or '无'}  池内 {stats['池内']}")
        return 0

    if action == "remove":
        stats = watch.remove(cfg, _split(args.codes) or [])
        print(f"移除 {stats['移除'] or '无'}  未找到 {stats['未找到'] or '无'}  池内 {stats['池内']}")
        return 0

    if action == "run":
        if _storage_missing(cfg):
            return 1
        return monitor.run_monitor(
            cfg,
            interval=args.interval,
            once=args.once,
            verbose=args.detail or args.once,
            bell=True if args.bell else None,
            repeat=args.repeat,
            panel=not args.no_panel,
        )

    raise ValueError(f"未知 watch 操作: {action}")


def cmd_info(args, cfg) -> int:
    if _storage_missing(cfg):
        return 1

    with store.open_store(cfg, readonly=True) as st:
        counts = st.bar_counts()
        names = st.get_stock_names()
        total_rows = sum(counts.values())
        min_date, max_date = st.date_range()
        print(f"存储格式    {cfg.data.storage}")
        print(f"存储位置    {store.storage_path(cfg)}")
        print(f"股票数      {len(counts)}  (股票信息表 {len(names)} 条)")
        print(f"K线行数     {total_rows}")
        print(f"日期范围    {min_date or '-'} ~ {max_date or '-'}")
        print(f"最后全量    {st.get_meta('last_full_update') or '-'}")
        print(f"最后增量    {st.get_meta('last_inc_update') or '-'}")
        short = sum(1 for n in counts.values() if n < cfg.universe.min_bars)
        print(f"K线不足{cfg.universe.min_bars}根  {short} 只（扫描时会跳过）")
    return 0


def cmd_trade(args, cfg) -> int:
    action = args.trade_action

    if action == "status":
        print(trade.describe_account(cfg))
        return 0

    if action == "history":
        print(trade.describe_history(cfg, limit=args.limit or 20))
        return 0

    if action == "reset":
        if not args.yes:
            print("确定要重置账户吗？这将清空所有持仓和交易记录。")
            print("确认请运行: python cli.py trade reset --yes")
            return 1
        trade.reset_account(cfg)
        print(f"账户已重置，初始资金: {cfg.trade.initial_capital:.2f}")
        return 0

    raise ValueError(f"未知 trade 操作: {action}")


def build_parser() -> argparse.ArgumentParser:
    all_keys = list(DETECTORS) + [p for ps in PATTERN_KEYS.values() for p in ps]
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="A股K线形态扫描：破底翻 / W底 / M头 / 上升下降横盘通道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", help="配置文件路径，默认 ./config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="打开 DEBUG 日志")
    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser("update", help="抓取/更新本地K线")
    p_update.add_argument("--full", action="store_true", help="全量重拉（忽略已有数据）")
    p_update.add_argument("--limit", type=int, help="只处理前N只，用于试跑")
    p_update.add_argument("--codes", help="只处理指定代码，逗号分隔")
    p_update.add_argument("--retry-failed", action="store_true", help="重跑上次失败清单")
    p_update.set_defaults(func=cmd_update)

    p_update_idx = sub.add_parser("update-index", help="单独拉取指数K线（沪深300/中证1000/创业板指）")
    p_update_idx.set_defaults(func=cmd_update_index)

    p_sectors = sub.add_parser("fetch-sectors", help="从baostock拉取行业分类")
    p_sectors.set_defaults(func=cmd_fetch_sectors)

    p_rebuild = sub.add_parser(
        "rebuild-market",
        help="从本地K线重建市场环境辅助文件（daily_summary/涨停历史/行业热度历史），不联网",
    )
    p_rebuild.add_argument("--days", type=int, help="回填的交易日数，默认取 market.history_days")
    p_rebuild.add_argument("--workers", type=int, help="进程数，默认 data.max_workers")
    p_rebuild.add_argument("--codes", help="只处理指定代码，逗号分隔")
    p_rebuild.set_defaults(func=cmd_rebuild_market)

    p_scan = sub.add_parser("scan", help="扫描形态并出报表")
    p_scan.add_argument(
        "--patterns", help=f"只扫指定形态，逗号分隔。可选: {', '.join(all_keys)}"
    )
    p_scan.add_argument("--status", help="只要 formed / forming，逗号分隔")
    p_scan.add_argument("--min-score", type=float, help="最低评分门槛")
    p_scan.add_argument("--workers", type=int, help="进程数，默认CPU核数")
    p_scan.add_argument("--codes", help="只扫指定代码，逗号分隔")
    p_scan.add_argument("--code", help="单股调试模式")
    p_scan.add_argument("--explain", action="store_true", help="单股模式下打印判定过程")
    p_scan.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="生成个股标注图（K线+关键点位+趋势线）；默认不出图，指定 --plot 才出（charts.enabled=true 则为默认出图）",
    )
    p_scan.add_argument(
        "--plot-top", type=int, help="出图数量上限，按综合评分顺序（覆盖 charts.max_charts）"
    )
    p_scan.add_argument(
        "--plot-long-only",
        action="store_true",
        help="只给做多形态出图（默认给全部命中出图）",
    )
    p_scan.add_argument("--top", type=int, default=15, help="终端打印前N条，默认15")
    p_scan.add_argument("--stamp", help="报表文件名日期戳，默认今天")
    p_scan.set_defaults(func=cmd_scan)

    p_info = sub.add_parser("info", help="查看本地库状态")
    p_info.set_defaults(func=cmd_info)

    p_watch = sub.add_parser("watch", help="观察池 + 盘中站稳提醒")
    w_sub = p_watch.add_subparsers(dest="watch_action", required=True)
    p_watch.set_defaults(func=cmd_watch)

    w_sync = w_sub.add_parser("sync", help="从 scan 报表同步观察池")
    w_sync.add_argument("--report", help="指定报表路径，默认 out/ 下最新的 scan_*.csv")
    w_sync.add_argument(
        "--patterns", help="只同步指定形态，逗号分隔。默认取配置 watch.patterns"
    )
    w_sync.add_argument("--status", help="只同步 formed / forming，默认 forming")
    w_sync.add_argument("--min-score", type=float, help="入池最低综合评分")

    w_sub.add_parser("list", help="查看观察池")

    w_add = w_sub.add_parser("add", help="手工加入观察池")
    w_add.add_argument("codes", help="股票代码，逗号分隔")
    w_add.add_argument("--pattern", default="break_bottom", help="登记的形态，默认 break_bottom")

    w_remove = w_sub.add_parser("remove", help="从观察池移除")
    w_remove.add_argument("codes", help="股票代码，逗号分隔")

    w_run = w_sub.add_parser("run", help="盘中轮询监控，站稳成型即提醒")
    w_run.add_argument("--interval", type=int, help="轮询间隔（秒），默认取配置 watch.interval")
    w_run.add_argument("--once", action="store_true", help="只跑一轮，用于验证")
    w_run.add_argument("--detail", action="store_true", help="每轮打印每只的明细")
    w_run.add_argument("--bell", action="store_true", help="触发提醒时终端响铃")
    w_run.add_argument("--repeat", action="store_true", help="已提醒过的票每轮重复提醒")
    w_run.add_argument(
        "--no-panel", action="store_true",
        help="不打印盘面概览（指数/涨跌停/成交额/行业榜），默认打印",
    )

    p_trade = sub.add_parser("trade", help="模拟交易：自动买卖")
    t_sub = p_trade.add_subparsers(dest="trade_action", required=True)
    p_trade.set_defaults(func=cmd_trade)

    t_sub.add_parser("status", help="查看账户和持仓状态")

    t_history = t_sub.add_parser("history", help="查看历史成交")
    t_history.add_argument("--limit", type=int, help="显示条数，默认20")

    t_reset = t_sub.add_parser("reset", help="重置账户（清空持仓和历史）")
    t_reset.add_argument("--yes", action="store_true", help="确认重置")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = config.load_config(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
