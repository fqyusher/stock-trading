"""观察池：把 scan 报表里"构筑中"的形态存下来，供盘中监控反复重判。

池子是 JSON（条目只有几十条，人可读可手改），写入走临时文件 + os.replace，
保证监控循环被 Ctrl-C 打断时不会留下半截文件。

条目只存**入池时的快照**（关键价位/评分/信号日），盘中的真实判定一律由
monitor.py 重跑检测器得出，不依赖这里的缓存值——快照只用于提醒信息展示。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .patterns import DETECTORS, PATTERN_CN
from .patterns.base import DIR_LONG, PATTERN_DIRECTION

log = logging.getLogger(__name__)

# 从报表里带进池子的关键价位（破底翻用；其他形态列不存在时自动跳过）
LEVEL_COLUMNS = ("参考低点", "破底最低", "参考止损", "上方压力", "颈线", "等幅目标",
                 "突破位", "整理下沿", "回踩支撑", "近期高点",
                 "收复位", "第一波高点", "回踩低点",
                 # 破底翻 v2 口径的价位。必须是纯数字列——sync_from_report 会
                 # float(row[col])，所以 L0_date/B1_date 这类日期列不能放进来。
                 "L0", "L1", "B1", "H1", "PL", "B2",
                 "stop_price", "tp1_price", "tp2_price", "trailing_stop",
                 "回撤%", "缩量比")

STATUS_TO_KEY = {"已形成": "formed", "构筑中": "forming"}

# 实时扫描进池的标的 source_report 写这个值；监控里据此区分
# “盘中刚冒出来的形成”（走正常 confirm_rounds）和“报表/停跑期间带进来的历史形成”（补漏）
REALTIME_SOURCE = "实时价格扫描"


def _long_only(df: pd.DataFrame) -> pd.DataFrame:
    """只保留做多形态。

    M头/下降通道是有效图形，留在报表里当规避参考，但监控"站稳成型即提醒"
    对它们没有意义——池子里混进做空形态，提醒出来也没法照着做多下单。

    按形态key 查表，而不是读报表的"方向"列：早于该列存在的老报表也要能 sync。
    """
    if df.empty or "形态key" not in df.columns:
        return df
    is_long = df["形态key"].map(
        lambda k: PATTERN_DIRECTION.get(str(k), DIR_LONG) == DIR_LONG
    )
    dropped = int((~is_long).sum())
    if dropped:
        # 不出这条日志的话，watch.patterns 里配了 m_top 的人只会看到"候选 0"，
        # 无从判断是阈值太高还是方向被筛掉了
        kinds = sorted(set(df.loc[~is_long, "形态key"].astype(str)))
        log.info("观察池跳过 %d 条做空形态（%s），只监控做多机会", dropped, "、".join(kinds))
    keep = is_long
    # v2 口径会输出"已失效"的行（破底后没收复 / 跌破止损）。它们仍是做多形态，
    # 只按形态key 筛方向会放进来，进池后只能等下一轮 INVALID 才被移出。
    if "失效" in df.columns:
        invalid = df["失效"].fillna("").astype(str).eq("是")
        n_invalid = int((invalid & is_long).sum())
        if n_invalid:
            log.info("观察池跳过 %d 条已失效信号", n_invalid)
        keep = is_long & ~invalid
    return df[keep]


def pool_path(cfg) -> Path:
    return Path(cfg.watch.pool_path)


def load_pool(cfg) -> list[dict[str, Any]]:
    """读观察池。文件不存在或损坏都返回空池——监控不该因为池子坏了就起不来。"""
    path = pool_path(cfg)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("观察池 %s 读取失败（%s），按空池处理", path, exc)
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    return list(entries or [])


def save_pool(cfg, entries: list[dict[str, Any]]) -> Path:
    """原子写：先写 .tmp 再 os.replace，避免写到一半被打断留下坏文件。"""
    path = pool_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "entries": entries,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def _blank_entry(code: str, name: str, pattern: str) -> dict[str, Any]:
    return {
        "code": code,
        "name": name,
        "pattern": pattern,
        "pattern_cn": PATTERN_CN.get(pattern, pattern),
        "added_date": date.today().strftime("%Y-%m-%d"),
        "signal_date": "",
        "added_score": 0.0,
        "levels": {},
        "confirm_streak": 0,
        "promoted_at": "",
        "last_status": "",
        "last_checked": "",
    }


def latest_report(cfg) -> Path | None:
    """out/ 下最新的 scan_*.csv。按文件名排序即是按日期排序（stamp 是 YYYYMMDD）。"""
    reports = sorted(Path(cfg.data.out_dir).glob("scan_*.csv"))
    return reports[-1] if reports else None


def sync_from_report(
    cfg,
    report_path: str | Path | None = None,
    patterns: list[str] | None = None,
    statuses: list[str] | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """从 scan 报表同步观察池，返回统计字典。

    已在池中的代码只刷新价位快照与评分，保留 confirm_streak / promoted_at——
    盘中重跑 sync 不该把已经攒起来的确认轮次清零。
    """
    w = cfg.watch
    path = Path(report_path) if report_path else latest_report(cfg)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"找不到扫描报表 {path or Path(cfg.data.out_dir) / 'scan_*.csv'}，先跑 python cli.py scan"
        )

    want_patterns = set(patterns or w.patterns)
    unknown = want_patterns - set(DETECTORS) - set(PATTERN_CN)
    if unknown:
        raise ValueError(f"未知形态 {', '.join(sorted(unknown))}，可选: {', '.join(DETECTORS)}")
    from .patterns import PATTERN_KEYS
    expanded = set()
    for p in want_patterns:
        if p in PATTERN_KEYS:
            expanded.update(PATTERN_KEYS[p])
        else:
            expanded.add(p)
    want_patterns = expanded
    want_statuses = {STATUS_TO_KEY.get(s, s) for s in (statuses or w.statuses)}
    threshold = w.min_score if min_score is None else min_score

    # report.py 写的是 utf-8-sig，读的时候必须对上，否则首列名会带 BOM
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"代码": str})
    if df.empty:
        return {"报表": str(path), "候选": 0, "新增": 0, "更新": 0, "池内": len(load_pool(cfg))}

    df["代码"] = df["代码"].astype(str).str.zfill(6)
    hit = _long_only(df[
        df["形态key"].isin(want_patterns)
        & df["状态"].map(lambda s: STATUS_TO_KEY.get(s, s)).isin(want_statuses)
    ])
    if threshold:
        hit = hit[hit["综合评分"] >= threshold]
    hit = hit.sort_values("综合评分", ascending=False)

    if w.max_pool and len(hit) > w.max_pool:
        log.info("候选 %d 只超过 max_pool=%d，按综合评分截断", len(hit), w.max_pool)
        hit = hit.head(w.max_pool)

    pool = load_pool(cfg)
    by_code = {e["code"]: e for e in pool}
    added = updated = 0

    for _, row in hit.iterrows():
        code = row["代码"]
        entry = by_code.get(code)
        if entry is None:
            entry = _blank_entry(code, str(row["名称"]), str(row["形态key"]))
            pool.append(entry)
            by_code[code] = entry
            added += 1
        else:
            updated += 1

        entry["name"] = str(row["名称"])
        entry["pattern"] = str(row["形态key"])
        entry["pattern_cn"] = str(row["形态"])
        entry["signal_date"] = str(row["信号日期"])
        entry["added_score"] = float(row["综合评分"])
        entry["source_report"] = path.name
        entry["levels"] = {
            col: float(row[col])
            for col in LEVEL_COLUMNS
            if col in row.index and pd.notna(row[col])
        }

    save_pool(cfg, pool)
    return {
        "报表": str(path),
        "候选": len(hit),
        "新增": added,
        "更新": updated,
        "池内": len(pool),
    }


def sync_from_hits(
    cfg,
    hits,
    patterns: list[str] | None = None,
    statuses: list[str] | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """把盘中实时检测命中结果合并进观察池。已有条目保留确认状态。"""
    if isinstance(hits, pd.DataFrame):
        rows = hits.to_dict("records")
    else:
        rows = [hit.to_row() for hit in (hits or [])]
    if not rows:
        return {"报表": "实时扫描", "候选": 0, "新增": 0, "更新": 0, "池内": len(load_pool(cfg))}

    w = cfg.watch
    want_patterns = set(patterns or w.patterns)
    from .patterns import PATTERN_KEYS
    expanded = set()
    for pattern in want_patterns:
        expanded.update(PATTERN_KEYS.get(pattern, [pattern]))
    want_patterns = expanded
    want_statuses = {STATUS_TO_KEY.get(s, s) for s in (statuses or w.statuses)}
    threshold = w.min_score if min_score is None else min_score
    df = pd.DataFrame(rows)
    hit = _long_only(df[
        df["形态key"].isin(want_patterns)
        & df["状态"].map(lambda s: STATUS_TO_KEY.get(s, s)).isin(want_statuses)
    ])
    if threshold:
        hit = hit[hit["综合评分"] >= threshold]
    hit = hit.sort_values("综合评分", ascending=False)
    if w.max_pool and len(hit) > w.max_pool:
        hit = hit.head(w.max_pool)

    pool = load_pool(cfg)
    by_code = {e["code"]: e for e in pool}
    added = updated = 0
    for _, row in hit.iterrows():
        code = str(row["代码"]).zfill(6)
        entry = by_code.get(code)
        if entry is None:
            entry = _blank_entry(code, str(row["名称"]), str(row["形态key"]))
            pool.append(entry)
            by_code[code] = entry
            added += 1
        else:
            updated += 1
        entry["name"] = str(row["名称"])
        entry["pattern"] = str(row["形态key"])
        entry["pattern_cn"] = str(row["形态"])
        entry["signal_date"] = str(row["信号日期"])
        entry["added_score"] = float(row["综合评分"])
        entry["source_report"] = REALTIME_SOURCE
        entry["levels"] = {
            col: float(row[col])
            for col in LEVEL_COLUMNS
            if col in row.index and pd.notna(row[col])
        }
    save_pool(cfg, pool)
    return {"报表": "实时扫描", "候选": len(hit), "新增": added, "更新": updated, "池内": len(pool)}


def add(cfg, codes: list[str], pattern: str = "break_bottom") -> dict[str, Any]:
    """手工加入观察池。名称从本地库取，价位快照留空（盘中判定不依赖它）。"""
    from . import store

    pool = load_pool(cfg)
    by_code = {e["code"]: e for e in pool}
    with store.open_store(cfg, readonly=True) as st:
        names = st.get_stock_names()

    added, exists = [], []
    for raw in codes:
        code = str(raw).zfill(6)
        if code in by_code:
            exists.append(code)
            continue
        pool.append(_blank_entry(code, names.get(code, code), pattern))
        added.append(code)

    save_pool(cfg, pool)
    return {"新增": added, "已存在": exists, "池内": len(pool)}


def remove(cfg, codes: list[str], reason: str = "手工移除") -> dict[str, Any]:
    pool = load_pool(cfg)
    targets = {str(c).zfill(6) for c in codes}
    kept = [e for e in pool if e["code"] not in targets]
    dropped = [e["code"] for e in pool if e["code"] in targets]
    if dropped:
        log.info("移出观察池 %s：%s", ", ".join(dropped), reason)
    save_pool(cfg, kept)
    return {"移除": dropped, "未找到": sorted(targets - set(dropped)), "池内": len(kept)}


def describe(entries: list[dict[str, Any]]) -> str:
    """池子转成终端表格。空池返回一句提示而不是空字符串。"""
    if not entries:
        return "观察池为空。先跑: python cli.py watch sync"

    header = (
        f"{'代码':<8}{'名称':<10}{'形态':<8}{'状态':<8}"
        f"{'入池分':>7}{'参考低点':>10}{'止损':>9}{'确认':>5}  信号日/提醒"
    )
    lines = [header, "-" * len(header)]
    for e in sorted(entries, key=lambda x: -x.get("added_score", 0)):
        levels = e.get("levels") or {}
        ref = levels.get("参考低点")
        stop = levels.get("参考止损")
        tail = e.get("promoted_at") or e.get("signal_date") or "-"
        flag = "★已提醒" if e.get("promoted_at") else ""
        lines.append(
            f"{e['code']:<8}{e['name']:<10}{e.get('pattern_cn', ''):<8}"
            f"{e.get('last_status') or '待检':<8}"
            f"{e.get('added_score', 0):>7.1f}"
            f"{ref if ref else 0:>10.2f}"
            f"{stop if stop else 0:>9.2f}"
            f"{e.get('confirm_streak', 0):>5}  {tail} {flag}"
        )
    lines.append(f"\n共 {len(entries)} 只")
    return "\n".join(lines)
