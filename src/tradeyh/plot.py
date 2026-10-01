"""个股形态标注图：全形态关键点位（带价格标签）+ 趋势线。

两种用法：
1. 单股复核：`cli.py scan --code 000001 --plot`
2. 全市场扫描后批量出图：`cli.py scan --plot`（榜单前 N 只 + index.html 索引页）

图片先按日期、再按形态分组写进 `<charts 根目录>/<YYYYMMDD>/<形态key>/`，
文件名是 `代码_形态key.png`——不含名次，所以同一天重跑扫描时同一只票的同一形态
直接覆盖，不会堆出一批残留图。

图上画什么：

- **关键点位**：把 `hit.key_levels` 里所有「价格类」键画成水平线，并在右侧标出
  「名称 价格」，不用回头翻报表对数字。价格类判定用**黑名单**——只排除比率
  （`%` 结尾）、`窗口` / `R2` / `容纳率` / 触碰次数这类指标键。白名单的问题是
  新形态往 `key_levels` 里塞了价格却忘了回来登记，图上就静默少一条线；
  黑名单则是「默认画、指标键显式排除」，不会漏。
  同一价位的多个别名合并成一条线，标签并排太挤时按 `_layout_labels` 推开，
  位移超过半个间距的再补一条细引线连回真实价位——位置可以挪，指向不能丢。
- **趋势线**：
  - 通道形态（上升/下降/箱体/下降通道反转）→ 用 `channel.fit_window()` 重算
    上轨 / 中轴 / 下轨三条回归斜线，并在右端标出价位；
  - 其余形态 → 用结构摆动点连出**支撑线**（谷连线）与**压力线**（峰连线），
    并按斜率外推一小段（最多锚点跨度的 `_TREND_MAX_EXTEND`）：既读得出"现在的
    支撑/压力在哪"，又不会因为两个锚点挨得太近而外推出一条飞出去的陡线。
- **摆动点**：形态登记的 `hit.pivots` 与由 `key_levels` 价位匹配到的 ZigZag
  点用圆点标出（实心 = 已确认，空心 = 末端未确认），文字标注按高低点上下让位。
- **信号日**：竖虚线标出形态命中的那根K线。
- **坐标轴**：mpf 的 yahoo 样式默认左右各画一套刻度，这里关掉孪生轴只留右侧一套；
  右侧留白按最长标签实测宽度动态计算。

依赖可选包 mplfinance：`pip install mplfinance`。缺依赖时全部函数返回 None/空，
不会让扫描流程失败。
"""

from __future__ import annotations

import html
import logging
from datetime import date
from pathlib import Path
from typing import Any, TypedDict, TypeGuard

import numpy as np
import pandas as pd

from .patterns import ScanContext
from .patterns import channel as channel_mod
from .patterns.base import PatternHit

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 关键点位样式
# --------------------------------------------------------------------------- #

# 结构位（形态骨架）→ (颜色, 线型)。这些键代表"关键的价位结构"，
# 用区分度高的实线/虚线，方便一眼看出形态骨架。
_STRUCTURE_STYLE: dict[str, tuple[str, str]] = {
    "颈线": ("#d62728", "-"),
    "参考低点": ("#d62728", "-"),
    "收复位": ("#d62728", "-"),
    "破底最低": ("#9467bd", ":"),
    "收复价": ("#1f77b4", ":"),
    "第一波高点": ("#1f77b4", "-"),
    "回踩低点": ("#2ca02c", "-"),
    "左底": ("#8c564b", ":"),
    "右底": ("#8c564b", ":"),
    "左头": ("#8c564b", ":"),
    "右头": ("#8c564b", ":"),
    "回踩支撑": ("#2ca02c", "-"),
    "反弹压力": ("#d62728", "-"),
    "突破位": ("#ff7f0e", "-"),
    "整理下沿": ("#ff7f0e", ":"),
    # 破底翻 v2：L0 是前低、L1 是破底最低、B1 是确认买点、H1 是第一波高点。
    # 用与旧口径对应的配色，两套图放一起时骨架一眼能对上。
    "L0": ("#d62728", "-"),
    "L1": ("#9467bd", ":"),
    "B1": ("#1f77b4", "-"),
    "H1": ("#1f77b4", "-"),
    "B2": ("#2ca02c", "-"),
    "PL": ("#2ca02c", "-"),
    # B2 严格定义新增键
    "B1价格": ("#1f77b4", "-"),
}

# 目标位 / 止损位统一配色，避免每个形态各用一套颜色、图例反而更乱
_TARGET_STYLE = ("#17a44c", "--")
_STOP_STYLE = ("#7f7f7f", ":")

# 目标键 → 标签名（"等幅目标" 直接叫 "目标" 更好读）
_TARGET_DISPLAY = {
    "等幅目标": "目标",
    "上方压力": "目标·压力",
    "近期高点": "目标·前高",
    "上轨": "目标·上轨",
    "tp1_price": "目标1",
    "tp2_price": "目标2",
    "第一目标": "TP1",
    "第二目标": "TP2",
}
_STOP_KEYS = {"参考止损", "stop_price"}

# 不画成价位线的键：比率、计数、指标值，以及"给交易层用、不属于形态骨架"的动态值。
_NON_PRICE_KEYS = {
    "窗口",
    "R2",
    "容纳率",
    "通道位置",
    "通道宽度%",
    "上轨触碰",
    "下轨触碰",
    "通道破位",
    "形态宽度",
    # 破底翻 v2 的比率/计数键：不排除的话会被当成价位画成横线
    "risk_pct",
    "reward1_pct",
    "reward2_pct",
    "rr1",
    "rr2",
    "缩量比",
    "收复用根数",
    # 破底翻 v1 的成交量比值。**必须排除**：量比量级是 0.5~3，画成价位线会把
    # y 轴从「十几块」一路拉到 0 附近，K线只剩画面上方三分之一（每个 break_bottom
    # 命中都中招过）。注意 `破底深度%` / `颈线高度%` 靠下面那条「% 结尾」规则兜住。
    "突破量比",
    "收复量比",
    # 移动止盈参考位是给交易层用的动态值，可能落在现价上方——
    # 画成一条"止损"横线会直接把图读反。骨架该画的是 stop_price。
    "trailing_stop",
    # B2 严格定义新增的非价格键
    "回踩幅度%",
    "回踩评级",
    "强确认",
}

# 通道三轨由回归单独画成斜线，不再画成水平线（画成水平反而是错的）
_CHANNEL_RAIL_KEYS = {"上轨", "中轴", "下轨"}
_RAILS = (
    ("upper_line", "#d62728", "上轨"),
    ("mid_line", "#7f7f7f", "中轴"),
    ("lower_line", "#2ca02c", "下轨"),
)

# 未知价格键的兜底配色（循环使用）
_FALLBACK_PALETTE = ("#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22")

# key_levels 价位 → ZigZag 摆动点的匹配容忍度
_PRICE_TOL = 0.02
# 同一价位的多个别名（如 M头的 颈线/反弹压力）合并阈值
_ALIAS_TOL = 1e-9

# 摆动点锚：key_levels 里哪些键代表"谷"、哪些代表"峰"
_SWING_LOW_KEYS = (
    "参考低点",
    "破底最低",
    "左底",
    "右底",
    "回踩低点",
    "回踩支撑",
    "收复位",
    # 破底翻 v2：L0 前低 / L1 破底最低 / PL 回踩低点
    "L0",
    "L1",
    "PL",
    # B2 严格定义：B1价格是买入确认点，属于低点侧锚
    "B1价格",
)
_SWING_HIGH_KEYS = (
    "颈线",
    "第一波高点",
    "近期高点",
    "左头",
    "右头",
    "上方压力",
    "突破位",
    "反弹压力",
    # 破底翻 v2：H1 第一波高点
    "H1",
    # 注意：B1 是买入确认点，不是结构高点，不应作为压力线锚点
)

_COLOR_SUPPORT = "#e377c2"   # 支撑线（谷连线）
_COLOR_RESIST = "#17becf"    # 压力线（峰连线）
_COLOR_SWING_OK = "#1f77b4"  # 结构关键点

# MA20：只是辅助均线，不该和形态骨架抢注意力——细线 + 低透明度，退到背景层。
# 原先是 #ff7f0e / width 0.9 / alpha 0.8，横贯全图时比形态线还显眼。
_MA20_COLOR = "#f0a35e"
_MA20_WIDTH = 0.7
_MA20_ALPHA = 0.5

# 右侧价位标签字号。原来 7.5 在 16:9 画布上偏小，加大画布后按 8.5 排版。
_LABEL_FONTSIZE = 8.5
# 标签被推开后连回真实价位的引线（细灰线）：位置可以偏，指向不能丢。
_LEADER_COLOR = "#c8c8c8"
_LEADER_LW = 0.6
# 右侧留白下限（数据单位，即K线根数）：留给价位标签的可视宽度。
_MIN_RIGHT_PAD = 12.0

# 趋势线最多再外推"锚点跨度"的多少倍。不限制的话，两个挨得很近的锚点
# 一路外推到最后一根K线，斜率被放大成一条飞出去的陡线（300413 曾到 30+）。
_TREND_MAX_EXTEND = 0.35

# 结构点文字（颈线/左底/右底…）字号，比价位标签小一号。
_PIVOT_FONTSIZE = 7.0

# key_levels 价格 → 标注名称的映射（用于在点位旁标注文字）
# 一个价格可能对应多个名称（如 颈线=反弹压力），取第一个结构性的
_PIVOT_LABEL_KEYS = (
    # 破底翻旧口径
    "参考低点", "颈线", "破底最低", "收复价",
    # 破底翻 v2
    "L0", "L1", "B1", "H1", "B2", "PL",
    # B2 严格定义
    "B1价格", "第一波高点", "回踩低点",
    # W底 / M头
    "左底", "右底", "左头", "右头",
)

# K线涨跌配色：A股习惯「涨红跌绿」，而 mplfinance 的 yahoo 预设是绿涨红跌。
# 不显式覆盖，图上的红绿方向就是反的。
_COLOR_UP = "#d62728"    # 收阳（涨）
_COLOR_DOWN = "#2ca02c"  # 收阴（跌）

# 图中的标题/关键点位标签全是中文，matplotlib 默认字体（DejaVu Sans）没有中文字形，
# 不换字体会整张图都是方框——标注功能等于白做。
_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Hiragino Sans GB",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)

# 中文字体文件兜底路径。matplotlib 只扫自己的系统字体目录，WSL 下
# /mnt/c/Windows/Fonts 不在其中——本机一个中文字体都枚举不到，图上的
# 标题/关键点位标签会整张变方框，标注功能等于白做，所以这里主动注册。
_CJK_FONT_FILES = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
_fonts_cache: list[str] | None = None


def _register_font_file(path: str) -> str | None:
    """把一个字体文件注册进 matplotlib 并返回它的 family 名；不可用返回 None。"""
    try:
        from matplotlib import font_manager

        if not Path(path).exists():
            return None
        font_manager.fontManager.addfont(path)
        name = font_manager.FontProperties(fname=path).get_name()
        return name or None
    except Exception as exc:  # noqa: BLE001 - 单个字体不可用不该影响其余候选
        log.debug("注册字体 %s 失败: %s", path, exc)
        return None


def _cjk_fonts() -> list[str]:
    """本机可用的中文字体名（按优先级），没有则空列表。

    先看 matplotlib 已枚举到的系统字体；一个都没有时再按 `_CJK_FONT_FILES`
    逐个注册（WSL 下命中 /mnt/c/Windows/Fonts），避免中文标签变方框。
    """
    global _fonts_cache
    if _fonts_cache is None:
        found: list[str] = []
        try:
            from matplotlib import font_manager

            available = {f.name for f in font_manager.fontManager.ttflist}
            found = [name for name in _CJK_FONTS if name in available]
        except Exception as exc:  # noqa: BLE001 - 字体问题不该让出图失败
            log.debug("枚举字体失败: %s", exc)

        if not found:
            for path in _CJK_FONT_FILES:
                name = _register_font_file(path)
                if name and name not in found:
                    found.append(name)

        if not found:
            log.warning("未找到中文字体，图中中文标签可能显示为方框")
        _fonts_cache = found
    return _fonts_cache


def _setup_fonts() -> None:
    """给 matplotlib 挂一个能显示中文的字体（只需生效一次）。

    注意：mplfinance 的 `_apply_mpfstyle()` 内部会先 `plt.style.use('default')`，
    把这里设的 rcParams 全部重置掉，所以真正生效的字体还得通过
    `_mpf_style()` 塞进 mpf 的 style `rc` 里。这里设置是为了兜住
    不经过 mplfinance 的绘图路径。
    """
    fonts = _cjk_fonts()
    if not fonts:
        return
    try:
        from matplotlib import rcParams

        current = list(rcParams.get("font.sans-serif", []))
        rcParams["font.sans-serif"] = fonts + [f for f in current if f not in fonts]
        # 中文字体常缺 U+2212 减号字形，负号会变方框
        rcParams["axes.unicode_minus"] = False
    except Exception as exc:  # noqa: BLE001
        log.debug("设置中文字体失败: %s", exc)


def _mpf_style(mpf):
    """mplfinance 的绘图样式：在 yahoo 基础上注入中文字体 + 涨红跌绿配色。

    两处覆盖都必须经 style 通道（`marketcolors` / `rc`），不能直接改 rcParams——
    mpf 的 `_apply_mpfstyle()` 会先 `plt.style.use('default')` 把它们重置掉。
    """
    # yahoo 预设是绿涨红跌，和中式习惯相反，显式换成涨红跌绿。
    try:
        marketcolors = mpf.make_marketcolors(
            up=_COLOR_UP,
            down=_COLOR_DOWN,
            edge="inherit",
            wick="inherit",
            volume="inherit",
        )
    except Exception as exc:  # noqa: BLE001 - 老版本没有 make_marketcolors 就沿用 yahoo 涨跌色
        log.debug("构造涨跌配色失败，沿用 yahoo 默认涨跌色: %s", exc)
        marketcolors = None

    fonts = _cjk_fonts()
    style_kwargs: dict[str, Any] = {"base_mpf_style": "yahoo"}
    if marketcolors is not None:
        style_kwargs["marketcolors"] = marketcolors
    if fonts:
        # yahoo 预设的 font.weight=semibold 在中文字体里常没有对应字重，
        # 会刷一堆 findfont 警告；一并压回 normal。
        style_kwargs["rc"] = {
            "font.sans-serif": fonts,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
            "axes.unicode_minus": False,
        }
    try:
        return mpf.make_mpf_style(**style_kwargs)
    except Exception as exc:  # noqa: BLE001 - 老版本没有 base_mpf_style 就退回默认
        log.debug("构造 mpf style 失败，回退 yahoo: %s", exc)
        return marketcolors or "yahoo"


class LevelLine(TypedDict):
    """一条要画的关键价位线。"""

    name: str
    display: str
    value: float
    color: str
    style: str


def _is_price(value: object) -> TypeGuard[float]:
    """是不是一个可以画成价位线的正数（bool / NaN / 字符串都不算）。

    用 TypeGuard 标注，调用方 `if not _is_price(v): continue` 之后 v 就是 float——
    否则每处都得再写一遍 cast。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return bool(np.isfinite(value)) and value > 0


def _style_for(name: str) -> tuple[tuple[str, str] | None, str]:
    """(颜色, 线型) + 展示名。未知价格键返回 None，由调用方走兜底配色。

    未知键也照样画——新形态往 key_levels 里塞价格就该自动出现在图上。
    """
    if name in _STOP_KEYS:
        return _STOP_STYLE, "止损"
    if name in _TARGET_DISPLAY:
        return _TARGET_STYLE, _TARGET_DISPLAY[name]
    style = _STRUCTURE_STYLE.get(name)
    if style is not None:
        return style, name
    return None, name


def collect_levels(hit: PatternHit, is_channel: bool) -> list[LevelLine]:
    """挑出要画的价位线。

    规则：数值 > 0、非白名单指标键、非 `%` 结尾；通道形态再排除三轨。
    同一价位的多个别名（M头的 颈线/反弹压力）合并成一条线、标签拼接，
    避免同一条水平线被画两遍、两个标签互相压。
    """
    rows: list[LevelLine] = []
    fallback_idx = 0
    for name, value in hit.key_levels.items():
        if not _is_price(value):
            continue
        if name in _NON_PRICE_KEYS or str(name).endswith("%"):
            continue
        if is_channel and name in _CHANNEL_RAIL_KEYS:
            continue

        style, display = _style_for(name)
        if style is None:
            style = (_FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)], "-.")
            fallback_idx += 1
        price = float(value)

        merged = False
        for row in rows:
            if abs(row["value"] - price) <= _ALIAS_TOL * max(1.0, abs(price)):
                if display not in row["display"].split("·"):
                    row["display"] = f"{row['display']}·{display}"
                merged = True
                break
        if merged:
            continue

        rows.append(
            {
                "name": name,
                "display": display,
                "value": price,
                "color": style[0],
                "style": style[1],
            }
        )
    return rows


def _layout_labels(prices: list[float], lo: float, hi: float, gap: float) -> list[float]:
    """标签防重叠：保证任意两个显示位置至少隔 `gap`，返回**与输入同序**的 y。

    价位挨得近时（W底 的 颈线 与 目标 只差一两个点，破底翻 v2 的 止损/PL/支撑线
    还要更挤）标签会叠在一起，推开的距离只是显示位置，线段仍在真实价位上。

    实现在"按价格升序的前向推开 + 越界压回 + 反向再推"之间收敛：前向保证不重叠，
    越界压回把整体塞回 `[lo, hi]`，反向再推补上压回造成的新的重叠。只要
    `hi - lo >= gap * (n - 1)`，收敛后最小间距就一定 >= gap。

    区间根本容不下时（标签太多，或者 y 轴被压得很扁）不再硬挤：按价格顺序在
    `[lo, hi]` 上等距铺开——宁可位置不准，也不能让两张标签叠成一张。

    注意：显示位置 ≠ 真实价位。位移超过 `gap * 0.5` 的标签由调用方补一条引线
    连回真实价位，否则读图时会把标签认到隔壁那条线上。
    """
    n = len(prices)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: prices[i])
    y = [float(p) for p in prices]

    if n > 1 and (hi - lo) < gap * (n - 1):
        # 区间容不下：等距铺满，零重叠优先于位置精度
        step = (hi - lo) / (n - 1)
        for k, i in enumerate(order):
            y[i] = lo + step * k
        return y

    # 前向推开（按价格升序）
    for k in range(1, n):
        prev, cur = order[k - 1], order[k]
        if y[cur] - y[prev] < gap:
            y[cur] = y[prev] + gap
    # 顶到上界就压回，再反向推一次；底到上界同理再前向推一次
    if y[order[-1]] > hi:
        y[order[-1]] = hi
        for k in range(n - 2, -1, -1):
            cur, nxt = order[k], order[k + 1]
            if y[nxt] - y[cur] < gap:
                y[cur] = y[nxt] - gap
    if y[order[0]] < lo:
        y[order[0]] = lo
        for k in range(1, n):
            prev, cur = order[k - 1], order[k]
            if y[cur] - y[prev] < gap:
                y[cur] = y[prev] + gap
    return y



# --------------------------------------------------------------------------- #
# 摆动点锚点 / 趋势线
# --------------------------------------------------------------------------- #


def _nearest_swing(
    ctx: ScanContext, kind: str, price: float, lo_idx: int, used: set[int]
) -> tuple[int, float] | None:
    """把一个 key_levels 价位对应回 ZigZag 摆动点，返回 (下标, 价格)。

    同一价位可能命中多个摆动点（价格反复在同一区间震荡），优先取**已确认**的，
    其次取更靠近当前时点的——趋势线的锚应该落在结构真正被确认的位置上。
    """
    best: tuple[bool, int, float] | None = None
    want_high = kind == "high"
    for pivot in ctx.pivots:
        if pivot.idx < lo_idx or pivot.idx in used:
            continue
        if pivot.is_high != want_high:
            continue
        if abs(pivot.price - price) / price > _PRICE_TOL:
            continue
        candidate = (bool(pivot.confirmed), pivot.idx, float(pivot.price))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return None if best is None else (best[1], best[2])


def _recent_swings(
    ctx: ScanContext, kind: str, lo_idx: int, need: int, used: set[int]
) -> list[tuple[int, float]]:
    """窗口内最近的若干同类**已确认** ZigZag 点，用于补足趋势线锚点。"""
    if need <= 0:
        return []
    want_high = kind == "high"
    pool = [
        p
        for p in ctx.pivots
        if p.idx >= lo_idx
        and p.confirmed
        and p.is_high == want_high
        and p.idx not in used
    ]
    pool.sort(key=lambda p: p.idx, reverse=True)
    return [(p.idx, float(p.price)) for p in pool[:need]]


def _anchors(
    ctx: ScanContext, hit: PatternHit, kind: str, lo_idx: int, need: int = 2
) -> list[tuple[int, float]]:
    """某一类（谷/峰）的趋势线锚点 [(下标, 价格)]，按时间升序。

    来源依次是：形态自带 pivots → key_levels 价位匹配 ZigZag → 窗口内最近已确认点。
    三级兜底是为了让"只给了一个价位的形态"（破底翻只有上方压力）也能画出线，
    而不是静默没有趋势线。
    """
    found: dict[int, float] = {}
    want_high = kind == "high"

    for pivot in hit.pivots:
        if pivot.is_high == want_high and pivot.idx >= lo_idx:
            found[pivot.idx] = float(pivot.price)

    keys = _SWING_HIGH_KEYS if want_high else _SWING_LOW_KEYS
    for name in keys:
        value = hit.key_levels.get(name)
        if not _is_price(value):
            continue
        match = _nearest_swing(ctx, kind, float(value), lo_idx, set(found))
        if match is not None:
            found[match[0]] = match[1]

    if len(found) < need:
        for idx, price in _recent_swings(ctx, kind, lo_idx, need - len(found), set(found)):
            found[idx] = price

    return sorted(found.items())


def _trend_lines(
    ctx: ScanContext, hit: PatternHit, lo_idx: int
) -> list[tuple[list[tuple[pd.Timestamp, float]], str, str]]:
    """支撑线（谷连线）+ 压力线（峰连线），按斜率外推一小段。

    只连两个历史锚点的话，图上读不出"现在"的支撑/压力在哪；但一路外推到最后一根
    K线又会把斜率放大成一条飞出去的陡线（两个挨得近的锚点，外推几十根后能高出
    现价一大截，比如 300413 的压力线到过 30+）。所以只外推"锚点跨度的一小部分"
    （`_TREND_MAX_EXTEND`），够读出当下方向，又不会画出夸张的"预测价"。
    """
    out: list[tuple[list[tuple[pd.Timestamp, float]], str, str]] = []
    bars = ctx.df.index
    last_idx = len(bars) - 1

    for kind, color, name in (
        ("low", _COLOR_SUPPORT, "支撑线"),
        ("high", _COLOR_RESIST, "压力线"),
    ):
        anchors = _anchors(ctx, hit, kind, lo_idx)
        if len(anchors) < 2:
            continue
        (i0, p0), (i1, p1) = anchors[0], anchors[-1]
        if i0 == i1:
            continue
        # 斜率不变外推，但终点不超过最后一根可见K线，也不超过锚点跨度的一小截
        span = i1 - i0
        ext = min(last_idx - i1, max(5, int(span * _TREND_MAX_EXTEND))) if span else 0
        end_p = p1 + (p1 - p0) / span * ext if span else p1
        out.append(
            (
                [(bars[i0], p0), (bars[i1 + ext], end_p)],
                color,
                name,
            )
        )
    return out


def _structural_points(
    ctx: ScanContext, hit: PatternHit, lo_idx: int, index: pd.DatetimeIndex
) -> list[tuple[pd.Timestamp, float]]:
    """图上要打点的结构位置：只画形态登记的关键 pivots。

    不再叠加趋势线锚点和全量 ZigZag 点——那些是画线用的内部数据，
    画成散点只会让图变密。形态登记了几个关键点就画几个，干净利落。
    """
    points: dict[pd.Timestamp, float] = {}
    for pivot in hit.pivots:
        if pivot.idx >= lo_idx and pivot.date in index:
            points[pivot.date] = float(pivot.price)
    return sorted(points.items())


def _pivot_label(pivot, hit: PatternHit) -> str:
    """为一个 pivot 找到对应的标注名称。价格匹配 key_levels 里的结构化键。"""
    price = float(pivot.price)
    for name in _PIVOT_LABEL_KEYS:
        val = hit.key_levels.get(name)
        if isinstance(val, (int, float)) and val > 0 and abs(val - price) / price < 0.005:
            return name
    return ""


def _text_pad(ax, texts: list[str], fontsize: float = _LABEL_FONTSIZE) -> float:
    """右侧留白要留几根K线的宽度：按最长标签实测宽度换算成数据单位。

    量不出来的话（非 Agg 后端等）回落到 `_MIN_RIGHT_PAD` 的两倍——宁可多留一点，
    也别让标签糊到坐标轴外面去。
    """
    fallback = _MIN_RIGHT_PAD * 2
    if not texts:
        return _MIN_RIGHT_PAD
    try:
        fig = ax.figure
        renderer = fig.canvas.get_renderer()
        y_mid = float(np.mean(ax.get_ylim()))
        probe = ax.text(0, y_mid, max(texts, key=len), fontsize=fontsize, alpha=0.0)
        try:
            width_px = float(probe.get_window_extent(renderer).width)
        finally:
            probe.remove()
        # 1 个数据单位（= 1 根K线）在这张图上换算成多少像素
        (px0, _), (px1, _) = ax.transData.transform([(0, y_mid), (1, y_mid)])
        px_per_unit = px1 - px0
        if px_per_unit <= 0:
            return fallback
        return max(_MIN_RIGHT_PAD, width_px / px_per_unit + 1.5)
    except Exception as exc:  # noqa: BLE001 - 量不准就用兜底值，不能因此出不了图
        log.debug("测量标签宽度失败，回落默认留白: %s", exc)
        return fallback


def _legend(
    entries: list[tuple[str, str]], has_points: bool
) -> list[Any]:
    """按 (颜色, 名称) 去重生成图例项；没画的东西不进图例。"""
    import matplotlib.lines as mlines

    handles, seen = [], set()
    for color, name in entries:
        if name in seen:
            continue
        seen.add(name)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.2, label=name))
    if has_points:
        handles.append(
            mlines.Line2D(
                [], [], color=_COLOR_SWING_OK, marker="^", linestyle="None",
                markersize=5, label="结构点",
            )
        )
    return handles


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def plot_hit(
    ctx: ScanContext,
    hit: PatternHit,
    out_dir: str | Path,
    bars: int = 180,
    dpi: int = 130,
    include_ma20: bool = True,
    filename: str | None = None,
) -> Path | None:
    """画一张带形态标注的K线图，返回 png 路径。缺少 mplfinance 时返回 None。"""
    try:
        import mplfinance as mpf
    except ImportError:
        log.warning("未安装 mplfinance，跳过画图：pip install mplfinance")
        return None

    _setup_fonts()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ctx.df
    # 通道要覆盖完整拟合窗口，否则三轨会画到画面外
    window = int(hit.key_levels.get("窗口", 0) or 0)
    show = max(int(bars), window + 20, 30)
    show = min(show, len(df))
    plot_df = df.iloc[-show:][["open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    index = plot_df.index
    lo_idx = len(df) - show  # 可见窗口在 ctx.df 中的起始下标

    is_channel = hit.pattern in channel_mod.PATTERN_CN

    # ---- 关键点位水平线 ----
    levels = collect_levels(hit, is_channel)
    hlines = [row["value"] for row in levels]
    hcolors = [row["color"] for row in levels]
    hstyles = [row["style"] for row in levels]

    # ---- 结构关键点：只画 hit.pivots，高点用 ▽ 低点用 △，带文字标注 ----
    addplots = []
    high_points = pd.Series(np.nan, index=index)
    low_points = pd.Series(np.nan, index=index)
    pivot_labels: list[tuple[pd.Timestamp, float, str, bool]] = []  # (date, price, label, is_high)

    # 为 hit.pivots 建立价格→名称的映射，用于标注
    pivot_name_map: dict[int, str] = {}
    _key_level_to_name: dict[str, str] = {}
    for name, val in hit.key_levels.items():
        if isinstance(val, (int, float)) and val > 0:
            _key_level_to_name[name] = name

    for pivot in hit.pivots:
        if pivot.idx < lo_idx or pivot.date not in index:
            continue
        if pivot.is_high:
            high_points.loc[pivot.date] = float(pivot.price)
        else:
            low_points.loc[pivot.date] = float(pivot.price)
        # 找到这个 pivot 对应的 key_levels 名称
        label = _pivot_label(pivot, hit)
        if label:
            pivot_labels.append(
                (pivot.date, float(pivot.price), label, bool(pivot.is_high))
            )

    if high_points.notna().any():
        addplots.append(
            mpf.make_addplot(
                high_points, type="scatter", markersize=50, marker="v",
                color=_COLOR_SWING_OK,
            )
        )
    if low_points.notna().any():
        addplots.append(
            mpf.make_addplot(
                low_points, type="scatter", markersize=50, marker="^",
                color=_COLOR_SWING_OK,
            )
        )

    if include_ma20 and "ma20" in df.columns:
        ma20 = df["ma20"].iloc[-show:]
        if ma20.notna().any():
            addplots.append(
                mpf.make_addplot(
                    ma20.rename("MA20"),
                    color=_MA20_COLOR,
                    width=_MA20_WIDTH,
                    alpha=_MA20_ALPHA,
                )
            )

    # ---- 斜线：通道三轨 / 支撑压力趋势线 ----
    alines: list[list[tuple[pd.Timestamp, float]]] = []
    aline_colors: list[str] = []
    legend_entries: list[tuple[str, str]] = []
    rail_notes: list[tuple[float, str, str]] = []  # (y, 文本, 颜色)

    if is_channel and window:
        fit = channel_mod.fit_window(ctx, window)
        if fit is not None:
            dates = fit["seg"].index
            for key, color, name in _RAILS:
                line = fit[key]
                alines.append([(dates[0], float(line[0])), (dates[-1], float(line[-1]))])
                aline_colors.append(color)
                legend_entries.append((color, name))
                rail_notes.append(
                    (float(line[-1]), f"{name} {float(line[-1]):.2f}", color)
                )

    if not is_channel:
        for segment, color, name in _trend_lines(ctx, hit, lo_idx):
            alines.append(segment)
            aline_colors.append(color)
            legend_entries.append((color, name))
            rail_notes.append((segment[-1][1], f"{name} {segment[-1][1]:.2f}", color))

    # ---- 组装 ----
    kwargs: dict[str, Any] = dict(
        type="candle",
        style=_mpf_style(mpf),
        volume=True,
        figratio=(18, 9),
        figscale=1.7,
        title=(
            f"{hit.code} {hit.name}  {hit.pattern_cn}[{hit.status_cn}·{hit.direction_cn}]  "
            f"评分{hit.score:.0f}  信号{pd.Timestamp(hit.signal_date):%Y-%m-%d}"
        ),
        tight_layout=True,
        warn_too_much_data=len(plot_df) + 10,
        returnfig=True,
    )
    if addplots:
        kwargs["addplot"] = addplots
    if alines:
        kwargs["alines"] = dict(alines=alines, colors=aline_colors, linewidths=1.1)
    if hlines:
        kwargs["hlines"] = dict(
            hlines=hlines,
            colors=hcolors,
            linestyle=hstyles,
            linewidths=1.0,
        )
    signal_date = pd.Timestamp(hit.signal_date)
    if signal_date in index:
        kwargs["vlines"] = dict(
            vlines=[signal_date], colors=["#444444"], linestyle="--", linewidths=0.9
        )

    fig, axlist = mpf.plot(plot_df, **kwargs)

    ax = axlist[0]
    # mplfinance 的 yahoo 样式是 y_on_right=True：主轴在右（带 Price/Volume 标签），
    # 同时还有一组孪生轴在左把同样的刻度数字又画一遍。左右两套纯属冗余，
    # 关掉孪生轴，只留右边那一套。
    for twin in axlist[1::2]:
        twin.yaxis.set_visible(False)
    # 先按最小留白设一次，作为下面 _text_pad 换算"1 根K线 = 多少像素"的基准；
    # 随后再按实测标签宽度收紧/放宽。
    ax.set_xlim(-1, len(plot_df) + _MIN_RIGHT_PAD)

    # ---- 右侧价格标签：价位线 + 轨道/趋势线右端，统一做防重叠 ----
    notes: list[tuple[float, str, str]] = [
        (row["value"], f"{row['display']} {row['value']:.2f}", row["color"]) for row in levels
    ]
    notes += [note for note in rail_notes if np.isfinite(note[0])]

    # mplfinance 不会为 hlines 自动放宽 y 轴，目标价常常落在画面外——
    # 那样标签只能被挤到边缘、对应的线却看不见。这里显式把 y 轴撑到能装下所有点位。
    if notes:
        price_list = [note[0] for note in notes]
        lo = min(float(plot_df["Low"].min()), min(price_list))
        hi = max(float(plot_df["High"].max()), max(price_list))
        pad = (hi - lo) * 0.04
        ax.set_ylim(lo - pad, hi + pad)

    if notes:
        # 右侧留白按最长标签的实际宽度定：写死 14 根K线的话，别名合并后的长标签
        # （颈线·反弹压力 13.73）会顶到画布边上。
        ax.set_xlim(-1, len(plot_df) + _text_pad(ax, [text for _, text, _ in notes]))

        y_lo, y_hi = ax.get_ylim()
        span = y_hi - y_lo
        gap = span * 0.038
        x_text = len(plot_df) + 0.4
        placed = _layout_labels([note[0] for note in notes], y_lo + span * 0.02, y_hi, gap)
        for (value, text, color), y in zip(notes, placed):
            ax.text(
                x_text,
                y,
                f" {text}",
                color=color,
                fontsize=_LABEL_FONTSIZE,
                va="center",
                ha="left",
                clip_on=False,
            )
            # 标签被推开超过半个间距就不再对着自己的线了：补一条细引线连回真实价位，
            # 否则读图会把标签认到隔壁那条线上。
            if abs(y - value) > gap * 0.5:
                ax.annotate(
                    "",
                    xy=(len(plot_df), value),
                    xytext=(x_text, y),
                    xycoords="data",
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=_LEADER_COLOR,
                        linewidth=_LEADER_LW,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )

    handles = _legend(
        legend_entries, high_points.notna().any() or low_points.notna().any()
    )
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.85)

    # ---- 结构关键点文字标注 ----
    # 偏移固定像素数（而不是数据单位），高低点各自往外挪，让文字落在K线影线之外；
    # 白色描边保证压到蜡烛或网格线上时仍然读得出来。
    import matplotlib.patheffects as path_effects

    for dt, price, label, is_high in pivot_labels:
        if dt not in index:
            continue
        # 高点文字往上、低点往下，让文字落在K线影线之外
        x = index.get_loc(dt)
        ax.annotate(
            label,
            xy=(x, price),
            xytext=(0, 11 if is_high else -11),
            textcoords="offset points",
            fontsize=_PIVOT_FONTSIZE,
            color=_COLOR_SWING_OK,
            fontweight="bold",
            va="center",
            ha="center",
            path_effects=[path_effects.withStroke(linewidth=2.2, foreground="white")],
        )

    name = filename or f"{hit.code}_{hit.pattern}.png"
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    # mplfinance 每张图都新建 figure，不关会一路攒着把内存吃满
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("标注图已写出: %s", path)
    return path


# --------------------------------------------------------------------------- #
# 批量出图
# --------------------------------------------------------------------------- #


def charts_base_dir(cfg) -> Path:
    """出图根目录（未按日期分组）：`charts.dir`，留空则 `out_dir/charts`。"""
    charts_cfg = getattr(cfg, "charts", None)
    cfg_dir = getattr(charts_cfg, "dir", "") if charts_cfg is not None else ""
    return Path(cfg_dir) if cfg_dir else Path(cfg.data.out_dir) / "charts"


def charts_dir(cfg, stamp: str | None = None) -> Path:
    """当天的出图目录：`<根>/<YYYYMMDD>`。

    按日期分组，免得不同交易日的图混在一个目录里越堆越多；日期取报表同一个
    `stamp`，这样"重跑某天的报表"时图也跟着进那一天的目录。
    """
    return charts_base_dir(cfg) / str(stamp or date.today().strftime("%Y%m%d"))


def charts_pattern_dir(cfg, pattern: str, stamp: str | None = None) -> Path:
    """某天某个形态的出图目录：`<根>/<YYYYMMDD>/<形态key>`。

    日期之下再按形态分一层：一次扫描动辄几百张图（不再限制张数），全平铺在
    日期目录里既翻不动、也难以按形态横向对比同类候选。
    """
    return charts_dir(cfg, stamp) / str(pattern)


def generate_charts(
    cfg,
    report_df: pd.DataFrame,
    base_dir: str | Path | None = None,
    max_charts: int | None = None,
    bars: int | None = None,
    index_html: bool | None = None,
    stamp: str | None = None,
) -> list[Path]:
    """按扫描报表为命中的个股批量出图，并可生成索引页。

    图片写入 `<base_dir>/<stamp>/<形态key>/`，文件名为 `代码_形态key.png`——
    **不含名次**，所以同一天重跑扫描时同一只票的同一形态直接覆盖，不会因为
    榜单名次变化而留下一批历史残留图。索引页写在日期目录（`<stamp>/index.html`），
    链接指向各形态子目录。

    默认不限制出图张数（`charts.max_charts=0`），一次扫描把命中的票全部画出来；
    需要限流时用 `--plot-top N` 显式指定。

    多进程扫描时只回传了行数据，画图需要带指标的 DataFrame 与 PatternHit 对象，
    所以这里要按 code 重新读K线、重跑检测器。为省掉重复开销：
    - 整批只打开一次存储、只读一次股票名表；
    - 每只票只读一次K线、只算一遍指标+ZigZag，再一次性跑完它命中的那几个
      检测器（不是每个形态从头来一遍指标/ZigZag）；
    - 读取窗口对齐扫描时的 required_bars，不再整段历史全搬进来——既更省，
      也让命中在同一窗口上复现（原先读全量历史，边缘指标/摆动点可能微移）。
    """
    if report_df is None or report_df.empty:
        return []
    if "代码" not in report_df.columns or "形态key" not in report_df.columns:
        log.warning("报表缺少 代码/形态key 列，跳过错图")
        return []

    # charts 配置整体可选：缺省时逐项回落到这里的默认值，别让 None 传下去
    charts_cfg = getattr(cfg, "charts", None)
    base = Path(base_dir) if base_dir is not None else charts_base_dir(cfg)
    target_dir = base / str(stamp or date.today().strftime("%Y%m%d"))
    if max_charts is None:
        # 默认 0 = 不限张数。以前回落到 60，导致命中几百条时后面的形态
        # （往往正是最需要翻图的少数形态）根本不出图。
        max_charts = getattr(charts_cfg, "max_charts", 0)
    max_charts = None if max_charts is None else int(max_charts)
    bars = int(getattr(charts_cfg, "bars", 180)) if bars is None else int(bars)
    if index_html is None:
        index_html = bool(getattr(charts_cfg, "index_html", True))
    include_ma20 = bool(getattr(charts_cfg, "include_ma20", True))
    dpi = int(getattr(charts_cfg, "dpi", 130))

    if max_charts is not None and max_charts <= 0:
        max_charts = None

    from . import scan as scan_mod
    from . import store as store_mod

    # 每只票命中的形态集合：一次性把该票需要的检测器跑完，而不是每个形态
    # 各读一次K线、各算一遍指标+ZigZag。多形态的票在这里省下重复的重扫。
    patterns_by_code: dict[str, set[str]] = {}
    for _, row in report_df.iterrows():
        code = str(row.get("代码", "")).zfill(6)
        pattern = str(row.get("形态key", ""))
        if code and pattern:
            patterns_by_code.setdefault(code, set()).add(pattern)

    # 读取窗口对齐扫描时的 required_bars（命中才能复现），再保证够画满 bars 根
    limit_bars = max(scan_mod.required_bars(cfg), int(bars) + 20)

    paths: list[Path] = []
    entries: list[tuple[pd.Series, Path]] = []
    # 按 code 缓存 (ctx, hits)：hits 已含该票所有命中形态，行匹配时按 pattern 取。
    # None 表示读/扫失败。整批只在下面的 with 里开一次存储、读一次股票名表。
    cache: dict[str, tuple[ScanContext, list[PatternHit]] | None] = {}

    with store_mod.open_store(cfg, readonly=True) as st:
        names = st.get_stock_names()
        for _, row in report_df.iterrows():
            if max_charts is not None and len(paths) >= max_charts:
                log.info("已达 charts.max_charts=%d，停止出图", max_charts)
                break

            code = str(row.get("代码", "")).zfill(6)
            pattern = str(row.get("形态key", ""))
            status_cn = str(row.get("状态", ""))
            if not code or not pattern:
                continue

            if code not in cache:
                cache[code] = _rescan_code(
                    cfg, st, code, patterns_by_code.get(code, {pattern}),
                    names, scan_mod, limit_bars,
                )
            cached = cache[code]
            if cached is None:
                continue
            ctx, hits = cached

            hit = next(
                (h for h in hits if h.pattern == pattern and h.status_cn == status_cn),
                None,
            ) or next((h for h in hits if h.pattern == pattern), None)
            if hit is None:
                log.debug("%s 未复现 %s/%s，跳过出图", code, pattern, status_cn)
                continue

            # 文件名只由 代码 + 形态key 决定（不带名次/状态）：同一天重跑同名覆盖。
            # 目录再按形态分一层，同类候选放在一起方便横向对比。
            name = f"{code}_{pattern}.png"
            path = plot_hit(
                ctx,
                hit,
                target_dir / pattern,
                bars=bars,
                dpi=dpi,
                include_ma20=include_ma20,
                filename=name,
            )
            if path:
                paths.append(path)
                entries.append((row, path))

    if index_html and entries:
        _write_index(target_dir, entries)
    return paths


def _rescan_code(cfg, st, code: str, patterns, names, scan_mod, limit_bars: int):
    """读一次K线 + 跑完该票命中的那几个检测器，返回 (ctx, hits)；失败返回 None。

    按 code（而不是 code+形态）缓存：同一只票命中多个形态时只读一次K线、
    只算一遍指标+ZigZag，再一次性把需要的检测器跑完，行匹配时按 pattern 取。
    传入已打开的存储 `st` 与预读的股票名表 `names`，省掉每只票各开一次库、
    各读一遍全表名。读取窗口用 limit_bars 截断（对齐扫描时的 required_bars），
    保证命中可复现。
    """
    try:
        df = st.read_kline(code, limit_bars)
        if df.empty:
            return None
        ctx = scan_mod.build_context(code, names.get(code, code), df, cfg)
        # 逐个形态解析检测器再并集：一个未知形态key只丢它自己，不会连累
        # 同一只票其它形态的图（resolve_detectors 对未知key是抛异常的）。
        detectors: dict = {}
        for pat in sorted(patterns):
            try:
                detectors.update(scan_mod.resolve_detectors([pat]))
            except ValueError:
                log.debug("%s 未知形态 %s，跳过", code, pat)
        return ctx, scan_mod.scan_context(ctx, detectors)
    except Exception as exc:  # noqa: BLE001 - 单股出图失败不该中断整批
        log.warning("%s 出图前重扫失败: %s", code, exc)
        return None


def _write_index(out_dir: Path, entries: list[tuple[pd.Series, Path]]) -> Path:
    """写 HTML 索引页，按形态分组折叠，方便一次浏览几百张图。

    图现在按 `<形态key>/` 分目录存放，索引里的链接要带上子目录；同时把每个
    形态收进一个 `<details>`：不再限制出图张数后，一次扫描可能有几百张，
    全部平铺展开既难找也重。折叠 + `loading="lazy"` 让未展开的图不加载。
    """
    cols = ("代码", "名称", "形态", "状态", "最新价", "综合评分", "预期收益%", "风险收益比")

    order: list[str] = []          # 形态出现顺序（沿用报表名次顺序）
    titles: dict[str, str] = {}    # 形态key → 中文名
    blocks: dict[str, list[str]] = {}

    for row, path in entries:
        try:
            rel = path.relative_to(out_dir).as_posix()
        except ValueError:  # 理论上不会发生：图都写在 out_dir 之下
            rel = path.name
        key = path.parent.name
        if key not in blocks:
            order.append(key)
            titles[key] = str(row.get("形态", "") or key)
            blocks[key] = []
        cells = "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in cols
        )
        blocks[key].append(
            f'<tr>{cells}'
            f'<td><a href="{html.escape(rel)}">图</a></td></tr>'
            f'<tr><td colspan="9"><img src="{html.escape(rel)}" loading="lazy"></td></tr>'
        )

    nav, sections = [], []
    for key in order:
        anchor = f"p-{key}"
        label = f"{titles[key]}（{len(blocks[key])}）"
        nav.append(f'<a href="#{html.escape(anchor)}">{html.escape(label)}</a>')
        sections.append(
            f'<details id="{html.escape(anchor)}">'
            f"<summary>{html.escape(label)}</summary>"
            "<table><thead><tr>"
            + "".join(f"<th>{col}</th>" for col in cols)
            + "<th>图</th></tr></thead><tbody>"
            + "".join(blocks[key])
            + "</tbody></table></details>"
        )

    doc = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>形态标注图索引</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:16px}"
        "table{border-collapse:collapse;font-size:13px}"
        "td{border:1px solid #ddd;padding:4px 8px}"
        "img{max-width:100%;margin:8px 0}"
        "summary{cursor:pointer;font-weight:600;margin:12px 0 6px}"
        ".nav a{margin-right:12px;font-size:13px}</style>"
        f"<h3>形态标注图索引（共 {len(entries)} 张）</h3>"
        f"<div class='nav'>{' '.join(nav)}</div>"
        + "".join(sections)
    )
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    log.info("图索引已写出: %s", path)
    return path
