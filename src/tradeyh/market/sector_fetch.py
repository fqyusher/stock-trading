"""行业分类拉取与本地缓存。

主渠道 baostock query_stock_industry()，返回证监会行业分类。
缓存到 data/market/sector_map.json，过期 30 天自动刷新。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .. import store

log = logging.getLogger(__name__)

_CACHE_FILENAME = "sector_map.json"
_CACHE_MAX_AGE_DAYS = 30


def _cache_path(cfg) -> Path:
    return store.market_dir(cfg) / _CACHE_FILENAME


def _cache_age_days(cfg) -> int | None:
    p = _cache_path(cfg)
    if not p.exists():
        return None
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    return (datetime.now() - mtime).days


def fetch_sector_map_baostock() -> dict[str, dict[str, str]]:
    """从 baostock 拉取行业分类。返回 {code: {sector, sector_id}}。

    baostock 目前提供证监会行业分类；申万分类已不再通过此接口返回。
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            raise RuntimeError(f"行业分类查询失败: {rs.error_msg}")
        result: dict[str, dict[str, str]] = {}
        while rs.next():
            row = rs.get_row_data()
            # fields: updateDate, code, code_name, industry, industryClassification
            code_raw = row[1]  # e.g. "sh.600000"
            name = row[2]
            industry = row[3]
            code = code_raw.split(".")[-1]
            if industry:
                result[code] = {"sector": industry, "name": name}
        log.info("从 baostock 获取 %d 只股票的行业分类", len(result))
        return result
    finally:
        bs.logout()


def load_sector_map(cfg) -> dict[str, dict[str, str]]:
    """加载行业分类缓存。缺失、为空或过期时自动拉取。"""
    p = _cache_path(cfg)
    age = _cache_age_days(cfg)
    cached: dict | None = None

    if age is not None:
        with open(p, encoding="utf-8") as f:
            cached = json.load(f) or None
        if cached is not None and age < _CACHE_MAX_AGE_DAYS:
            log.info("加载行业分类缓存 (%d 只, %d天前)", len(cached), age)
            return cached

    log.info("行业分类缓存缺失、为空或过期 (age=%s), 重新拉取", age)
    try:
        data = fetch_sector_map_baostock()
    except Exception:
        if cached:
            log.warning("拉取失败，使用过期缓存 (%d 只)", len(cached))
            return cached
        log.warning("拉取失败且无可用缓存，行业分类为空；网络恢复后运行 fetch-sectors 补全")
        return {}
    save_sector_map(cfg, data)
    return data


def cached_sector_map(cfg) -> dict[str, str]:
    """只读缓存文件返回 {code: sector}，**绝不联网**。

    给盘面面板用：`load_sector_map` 在缓存缺失或过期时会去拉 baostock，
    盘中每分钟跑一次的路径不能撞上这种阻塞。
    """
    p = _cache_path(cfg)
    if not p.exists():
        return {}
    try:
        full = json.loads(p.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("行业分类缓存读取失败（%s），按空处理", exc)
        return {}
    return {
        str(code): info["sector"]
        for code, info in full.items()
        if isinstance(info, dict) and info.get("sector")
    }


def save_sector_map(cfg, mapping: dict[str, dict[str, str]]) -> None:
    p = _cache_path(cfg)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    log.info("行业分类缓存已保存 (%d 只)", len(mapping))


def code_to_sector_map(cfg) -> dict[str, str]:
    """返回 {stock_code: sector_name} 的扁平映射。"""
    full = load_sector_map(cfg)
    return {code: info["sector"] for code, info in full.items()}
