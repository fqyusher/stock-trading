"""扫描结果输出：Excel（汇总 + 每形态一个 sheet）与 CSV。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .patterns.base import DIR_LONG, PATTERN_DIRECTION

log = logging.getLogger(__name__)

BASE_COLUMNS = [
    "代码",
    "名称",
    "形态",
    "形态key",
    "方向",
    "状态",
    "最新价",
    "综合评分",
    "机会质量",
    "收益质量",
    "阶段确定性",
    "风险修正",
    "信号日期",
    "预期收益%",
    "风险收益比",
    "市场环境",
    "市场情绪",
    "所属板块",
    "板块热度",
    "板块阶段",
]

FILL_FORMED = PatternFill("solid", start_color="C6EFCE")   # 已形成 绿
FILL_FORMING = PatternFill("solid", start_color="FFEB9C")  # 构筑中 黄
HEADER_FILL = PatternFill("solid", start_color="DDEBF7")


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = [c for c in BASE_COLUMNS if c in df.columns]
    tail = ["说明"] if "说明" in df.columns else []
    middle = [c for c in df.columns if c not in base and c not in tail]
    return df[base + middle + tail]


def long_only(df: pd.DataFrame) -> pd.DataFrame:
    """只保留做多形态，并丢掉已失效（v2 口径会输出失效行）的信号。

    M头/下降通道留在汇总里当规避参考，但不进做多榜单。
    按形态key 查表而不是读"方向"列，这样老报表也能筛；"失效"列同理——
    只有 v2 口径会写它，旧报表没这一列时按"没有失效行"处理。
    """
    if df.empty or "形态key" not in df.columns:
        return df
    keep = df["形态key"].map(
        lambda k: PATTERN_DIRECTION.get(str(k), DIR_LONG) == DIR_LONG
    )
    if "失效" in df.columns:
        keep &= df["失效"].fillna("").astype(str).ne("是")
    return df[keep]


def _style_sheet(worksheet, df: pd.DataFrame) -> None:
    n_rows, n_cols = len(df), len(df.columns)
    worksheet.freeze_panes = "A2"

    for col_idx, name in enumerate(df.columns, start=1):
        cell = worksheet.cell(row=1, column=col_idx)
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        width = 46 if name == "说明" else max(8, min(18, len(str(name)) * 2 + 4))
        worksheet.column_dimensions[get_column_letter(col_idx)].width = width

    if n_rows == 0:
        return

    # 综合评评分色阶
    if "综合评分" in df.columns:
        letter = get_column_letter(list(df.columns).index("综合评分") + 1)
        worksheet.conditional_formatting.add(
            f"{letter}2:{letter}{n_rows + 1}",
            ColorScaleRule(
                start_type="num", start_value=0, start_color="FFFFFF",
                mid_type="num", mid_value=60, mid_color="FFD966",
                end_type="num", end_value=100, end_color="63BE7B",
            ),
        )

    # 状态底色
    if "状态" in df.columns:
        letter = get_column_letter(list(df.columns).index("状态") + 1)
        rng = f"{letter}2:{letter}{n_rows + 1}"
        worksheet.conditional_formatting.add(
            rng,
            CellIsRule(operator="equal", formula=['"已形成"'], fill=FILL_FORMED),
        )
        worksheet.conditional_formatting.add(
            rng,
            CellIsRule(operator="equal", formula=['"构筑中"'], fill=FILL_FORMING),
        )

    worksheet.auto_filter.ref = (
        f"A1:{get_column_letter(n_cols)}{n_rows + 1}"
    )


def write_report(df: pd.DataFrame, out_dir: str | Path, stamp: str) -> dict[str, Path]:
    """写出 xlsx + csv，返回 {'xlsx': path, 'csv': path}。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"scan_{stamp}.xlsx"
    csv_path = out_dir / f"scan_{stamp}.csv"

    if df.empty:
        pd.DataFrame(columns=BASE_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.warning("没有任何形态命中，只写出空表 %s", csv_path)
        return {"csv": csv_path}

    summary = _order_columns(df)
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="汇总", index=False)
        _style_sheet(writer.sheets["汇总"], summary)

        # 做多榜单：汇总里混着 M头/下降通道，照着它排序会把"该躲开的票"
        # 排在前面。真正能下单的候选单独成表。
        longs = long_only(summary)
        if not longs.empty:
            longs = longs.sort_values("综合评分", ascending=False)
            longs.to_excel(writer, sheet_name="做多榜单", index=False)
            _style_sheet(writer.sheets["做多榜单"], longs)

        for pattern_cn, group in df.groupby("形态", sort=False):
            # 每个 sheet 只保留本形态实际有值的关键价位列
            sheet_df = _order_columns(group.dropna(axis=1, how="all"))
            sheet_df = sheet_df.sort_values("综合评分", ascending=False)
            sheet_name = str(pattern_cn)[:31]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            _style_sheet(writer.sheets[sheet_name], sheet_df)

    log.info("报表已写出: %s", xlsx_path)
    return {"xlsx": xlsx_path, "csv": csv_path}


def print_summary(df: pd.DataFrame, top: int = 15) -> None:
    """终端打印分组统计与前若干条，方便不开 Excel 就能看一眼。"""
    if df.empty:
        print("没有命中任何形态。")
        return

    print("\n=== 命中统计 ===")
    pivot = (
        df.pivot_table(index="形态", columns="状态", values="代码", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    pivot["合计"] = pivot.sum(axis=1)
    print(pivot.to_string())

    # 榜单只列做多候选：做空形态在上面的统计里能看到，但排进 top 没有操作意义
    longs = long_only(df)
    shorts = len(df) - len(longs)
    print(f"\n=== 做多综合评分前 {top} ===")
    if longs.empty:
        print("无做多形态命中。")
    else:
        cols = ["代码", "名称", "形态", "状态", "最新价", "综合评分", "预期收益%", "风险收益比", "信号日期"]
        cols = [c for c in cols if c in longs.columns]
        print(longs[cols].head(top).to_string(index=False))
    if shorts:
        print(f"（另有 {shorts} 条做空形态未列入榜单，见报表「汇总」sheet）")
