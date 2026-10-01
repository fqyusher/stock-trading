#!/usr/bin/env python3
"""独立脚本：使用 Baostock 更新K线数据。

用法示例：
    # 增量更新全市场（从本地最后日期到今天）
    python baostock_update.py

    # 全量更新全市场
    python baostock_update.py --full

    # 指定日期范围
    python baostock_update.py --start 20240101 --end 20241231

    # 只更新指定股票
    python baostock_update.py --codes 000001,600000,300001

    # 指定输出目录和存储格式
    python baostock_update.py --output ./my_data --storage csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tradeyh import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def bs_symbol(code: str) -> str:
    """转换代码格式：000001 -> sz.000001, 600000 -> sh.600000
    
    如果已经是 sh. 或 sz. 开头则直接返回。
    """
    if code.startswith(("sh.", "sz.")):
        return code
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def fetch_kline_bs(
    code: str, start: str, end: str, adjust: str = "2"
) -> pd.DataFrame:
    """使用 Baostock 获取单只股票K线。

    Args:
        code: 股票代码，如 000001
        start: 开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end: 结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        adjust: 复权类型 1=后复权 2=前复权 3=不复权
    """
    import baostock as bs

    symbol = bs_symbol(code)
    start_date = start.replace("-", "")
    end_date = end.replace("-", "")
    start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    fields = "date,open,high,low,close,volume,amount,turn,pctChg"
    rs = bs.query_history_k_data_plus(
        symbol,
        fields,
        start_date=start_fmt,
        end_date=end_fmt,
        frequency="d",
        adjustflag=adjust,
    )

    if rs.error_code != "0":
        log.warning("[%s] Baostock 查询失败: %s", code, rs.error_msg)
        return pd.DataFrame()

    rows: list[list] = []
    try:
        while rs.next():
            rows.append(rs.get_row_data())
    except Exception as exc:
        log.warning("[%s] Baostock 数据读取错误: %s", code, exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close",
                 "volume", "amount", "turnover", "pct_chg"],
    )

    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"] / 100
    df = df.dropna(subset=["date", "close"])
    df = df[store.KLINE_COLUMNS]

    return df.sort_values("date").reset_index(drop=True)


def get_all_stocks() -> pd.DataFrame:
    """获取全市场股票列表，返回含 code 和 name 的 DataFrame。"""
    import baostock as bs

    rs = bs.query_stock_basic()
    records = []
    while rs.next():
        row = rs.get_row_data()
        full_code = row[0]  # 可能是 sh.600000 格式
        name = row[1] if len(row) > 1 else ""
        status = row[4] if len(row) > 4 else "1"
        if status == "1":
            # 提取6位纯代码
            code = full_code.split(".")[-1] if "." in full_code else full_code
            records.append({"code": code, "name": name})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="使用 Baostock 更新K线数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full", action="store_true",
        help="全量更新（从 2006-01-01 开始）",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="开始日期，格式 YYYYMMDD 或 YYYY-MM-DD（默认：增量=本地最后日期，全量=20060101）",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="结束日期（默认：今天）",
    )
    parser.add_argument(
        "--codes", type=str, default=None,
        help="指定股票代码，逗号分隔（如 000001,600000）",
    )
    parser.add_argument(
        "--output", type=str, default="./data/store",
        help="输出目录（默认 ./data/store）",
    )
    parser.add_argument(
        "--storage", type=str, default="csv",
        choices=["csv", "json", "sqlite"],
        help="存储格式（默认 csv）",
    )
    parser.add_argument(
        "--adjust", type=str, default="qfq",
        choices=["qfq", "hfq", ""],
        help="复权方式：qfq=前复权, hfq=后复权, 空=不复权（默认 qfq）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="限制处理股票数量（调试用）",
    )

    args = parser.parse_args()

    adjust_map = {"qfq": "2", "hfq": "1", "": "3"}
    adjust_flag = adjust_map[args.adjust]

    end_date = args.end or date.today().strftime("%Y%m%d")
    start_default = "20060101" if args.full else None

    log.info("Baostock K线更新工具")
    log.info("存储格式: %s, 输出目录: %s", args.storage, args.output)
    log.info("复权方式: %s", args.adjust or "不复权")

    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        log.error("Baostock 登录失败: %s", lg.error_msg)
        return 1
    log.info("Baostock 登录成功")

    try:
        cfg_dict = {
            "storage": args.storage,
            "data_dir": args.output,
            "db_path": str(Path(args.output) / "kline.db"),
        }

        class Cfg:
            data = type("Data", (), cfg_dict)()

        cfg = Cfg()
        st = store.open_store(cfg)

        if args.codes:
            codes = [c.strip() for c in args.codes.split(",")]
            stock_info = pd.DataFrame([{"code": c, "name": ""} for c in codes])
            log.info("指定股票: %d 只", len(codes))
        else:
            log.info("获取全市场股票列表...")
            stock_info = get_all_stocks()
            codes = stock_info["code"].tolist()
            log.info("全市场共 %d 只股票", len(codes))

        if args.limit:
            codes = codes[:args.limit]
            stock_info = stock_info[stock_info["code"].isin(codes)]
            log.info("限制处理前 %d 只", len(codes))

        # 先保存股票基本信息（不更新时间戳）
        st.upsert_stock_info(stock_info, update_time=False)
        log.info("已保存 %d 只股票基本信息", len(stock_info))

        success = 0
        failed = 0
        skipped = 0

        for i, code in enumerate(codes, 1):
            # if int(code) < 600700:
                # log.info(code)
                # st.update_stock_time(code)
                # continue
            if args.full:
                start = args.start or start_default
            else:
                last = st.last_date(code)
                if last and not args.start:
                    last_dt = pd.Timestamp(last).date()
                    start_dt = last_dt - timedelta(days=10)
                    start = start_dt.strftime("%Y%m%d")
                else:
                    start = args.start or start_default

            try:
                df = fetch_kline_bs(code, start, end_date, adjust_flag)

                if df.empty:
                    log.warning("[%s] (%d/%d) 无数据", code, i, len(codes))
                    skipped += 1
                    continue

                n = st.upsert_kline(code, df)
                st.update_stock_time(code)
                log.info(
                    "[%s] (%d/%d) 成功，%d 根K线 -> %s",
                    code, i, len(codes), n, st.kline_path(code),
                )
                success += 1

            except Exception as exc:
                log.error("[%s] (%d/%d) 失败: %s", code, i, len(codes), exc)
                failed += 1

            if i % 100 == 0:
                st.commit()

        st.commit()
        st.close()

        log.info("=" * 50)
        log.info("更新完成: 成功 %d, 失败 %d, 跳过 %d, 共 %d",
                 success, failed, skipped, len(codes))

    finally:
        bs.logout()
        log.info("Baostock 已登出")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
