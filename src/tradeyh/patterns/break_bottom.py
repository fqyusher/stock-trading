"""破底翻：二次探底浅破前低、快速止跌、放量收盘站上颈线——空头陷阱 / 假突破。

严格定义下的完整流程：
前置下跌趋势 → 前低 L0（**必须是这段下跌的最低点**：它之前 prior_lookback 根内
不能有更低的价）→ 反弹高点（颈线 N）→ 二次探底**盘中 low** 跌破 L0 创新低
（min_break_depth ≤ 深度 ≤ max_break_depth）→ recover_bars 内收盘收复 L0（快速止跌，
不在新低区盘整）→ 收盘放量站上颈线 N（翻转确认）→ 已形成。

各元素取哪一列价格（2026-09-29 按用户口径统一，与 v2 一致）：

| 元素 | 取价 | 说明 |
|---|---|---|
| L0（前低） | **low** | 下影线最低点 |
| N（颈线） | **high** | 反弹高点 |
| 破底新低 L1 | **low** | 影线最低点 |
| 跌破 L0 是否成立 | **low** | 盘中插针破也算（旧口径要求收盘，与 v2 相反） |
| 收复 L0 | **close** | 收复是收盘概念 |
| 突破颈线 / B1 | **close** | 信号一律看收盘 |

时序上还有两条硬约束，缺一条上面的流程就不成立：

- **破位必须在颈线之后**：L0 与颈线之间就出现更低的 low 的，说明这道支撑压根没撑住，
  后面那个"颈线"只是深跌途中的一次脉冲反弹（002030 达安基因：L0 5.17(06-17)，
  06-29 就插到 4.80，认的颈线 5.43 要到 07-16）。
- **收复之后不得再翻失败**：收盘连续 `pullback_invalid_consecutive` 根落在 L0 之下
  即视为又回到新低区盘整（300354 东华测试：08-26 收复 L0=26.59，09-10/09-11 连收
  26.55/25.80）。`detect()` 末尾也查这条，但只看**最后两根**，翻回来之后才失败的就
  绕过去了——所以在结构定位阶段一并挡掉。单根插破不算（300353 东土科技收复后收在
  L0 下方 1.9% 又立刻收回，仍算正常回踩）。注意这条与 v2 的失效判定一样看**收盘**，
  不要跟着破底一起改成 low。

「L0 必须是这段下跌的最低点」这条不是洁癖：下跌途中的任意一个中间低点也能凑出
(低-高-低) 摆动三元组，右边照样收得出新低，但那个"新低"离真支撑还差一截——
所谓破底根本没破到任何支撑，颈线也多半是深跌中的一次脉冲反弹，量度目标直接
飞到合理区间外。旧口径只校验"L0 之前跌过 prior_drop"，全市场一天能捞出一堆
这种下跌半途的伪结构。

但「L0 与二次低点必须是相邻摆动点」则是过度约束：真实走势里反弹见顶后往往先
回调出一个更高的低点、再弹一次，然后才砸穿前低，两个低点中间夹着摆动点是常态。
所以候选枚举只看"右侧低于左侧"，允许中间夹任意摆动点，颈线取破底前最近的那个
摆动高点（详见 `_pivot_triples`）。

与旧实现的根本差别：确认锚点从「L0 + 0.3×ATR14」换成**颈线**。旧口径只要求价格
相对 L0 推离一个 ATR 幅度，与"前一轮反弹的高点"无关，形态的几何含义被丢掉了。

- 已形成 = 颈线被**放量**收盘突破（突破量 ≥ 前 5 日均量 × neck_breakout_vol_ratio），
  且其后至少还有一根 K 维持收盘站上颈线。
- 构筑中 = 已收复 L0 但尚未触及颈线；或收盘已站上颈线但突破日量能不足（等放量确认）；
  或突破就发生在今天（等次日站稳）。
- 失效 = 连续 2 日收盘跌破 L0，或收盘跌破 L0 − stop_atr_buffer×ATR。
- 收盘跌回颈线下方 = 该次突破作废（run 重置），退回构筑中；这正是"回踩不破颈线
  才视为正式确立"的口径。

`_find_break_bottom_base` 是全项目唯一的破底翻结构定位入口，B2 也从这里导入——
破底翻第二买点的前提就是"这里已经确认"，两套口径必须逐字一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..swings import HIGH, LOW, Pivot, count_level_touches, prior_drop_pct
from .base import STATUS_FORMED, STATUS_FORMING, PatternHit, ScanContext, blend, taper

PATTERN = "break_bottom"
PATTERN_CN = "破底翻"

# 放量突破的噪声兜底：量比分母取不到时的默认值
_VOL_FALLBACK = 1.0


def _safe(series, idx: int, default: float = 0.0) -> float:
    try:
        value = float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return default
    return default if np.isnan(value) else value


def _prior5_vol_ratio(volume: np.ndarray, i: int) -> float:
    """第 i 根的成交量 / 其**前** 5 日均量。

    取前 5 根而不是 VOL_MA5：后者把当日自己也算进均值，当日放量会把分母一起
    抬高，"放量 30%" 变成事实上的放量 30%/(1+30%)，比定义更严。
    """
    start = max(0, i - 5)
    if i <= start:
        return _VOL_FALLBACK
    prev = volume[start:i]
    avg = float(prev.mean()) if prev.size else 0.0
    if avg <= 0:
        return _VOL_FALLBACK
    return float(volume[i]) / avg


@dataclass(frozen=True)
class BreakBottomBase:
    """破底翻的基础结构，B1 / B2 共用。"""

    l0_idx: int
    l0: float  # 前低
    neck_idx: int
    neck: float  # 颈线 = 前低之后的反弹高点
    low_idx: int
    low: float  # 二次探底最低价
    recover_idx: int  # low_idx 后 recover_bars 内首个收盘收复 L0 的 K
    depth: float
    span: int
    neck_height: float
    breakout_idx: int | None  # 当前行情段的颈线放量突破触发 K（B1 的确认依据）
    breakout_vol_ratio: float
    run_intact: bool  # 最新一根收盘仍站在颈线上方
    first_breakout_idx: int | None  # 结构内首次放量突破颈线（B2 第一波的起算点）
    first_breakout_vol_ratio: float
    # 图2标注的C点：站回确认点
    c_point_idx: int | None  # C点索引：首次收盘站上颈线
    c_point_price: float  # C点价格
    c_point_valid: bool  # C点是否有效（放量）
    # 图2标注的前波高点：作为目标价参考
    prior_high_idx: int | None  # L0之前最近的摆动高点索引
    prior_high: float  # 前波高点价格


def _pivot_triples(pivots: list[Pivot]) -> list[tuple[Pivot, Pivot, Pivot]]:
    """所有 (低, 高, 低) 结构，从最近往前，右侧低点必须低于左侧低点。

    与 W 底不同：破底翻的两个低点之间**允许夹着其它摆动点**。前低之后经常先
    反弹、再回调出一个更高的低点、甚至再弹一次，才真正砸穿前低。只取相邻摆动点
    会让这种夹心结构永远凑不出来——东土科技 300353 的 pivot 是
    16.61 → 20.99 → 17.77 → 19.15 → 15.66，相邻口径只能凑出 (17.77, 19.15, 15.66)
    这个假前低，真前低 16.61 与二次低点 15.66 之间隔着两个摆动点，压根进不了候选池。

    颈线取**破底前最近的那个摆动高点**：它才是"前低之后的那一轮反弹高点"，也就是
    价格必须放量收盘站上才确认翻转的那道压力。不能取两个低点之间的最高高点——
    间隔能拉到 63 根，中间很容易冒出一个与底部结构脱节的高点，颈线被钉到那上面后
    量度目标直接虚高（实测 600516 方大炭素：前低 5.02 → 颈线 6.78(+35%) →
    破底 4.99 只低了 0.6%，量度目标 8.57 即 +71%，纯属噪声）。也不能取原始 high，
    那会把单 bar 插针当颈线。
    """
    lows = [p for p in pivots if p.kind == LOW]
    highs = [p for p in pivots if p.kind == HIGH]
    out: list[tuple[Pivot, Pivot, Pivot]] = []
    for i, low1 in enumerate(lows):
        for low2 in lows[i + 1 :]:
            if low2.price >= low1.price:
                continue
            between = [h for h in highs if low1.idx < h.idx < low2.idx]
            if not between:
                continue
            out.append((low1, max(between, key=lambda p: p.idx), low2))
    return list(reversed(out))


class _Breakout(NamedTuple):
    """颈线放量突破的扫描结果。"""

    current_idx: int | None  # 当前这段连续站上颈线的行情里的触发 K
    current_ratio: float
    first_idx: int | None  # 结构内**首次**放量突破（B2 的第一波起算点）
    first_ratio: float
    run_intact: bool  # 最新一根收盘仍站在颈线上方


def _neckline_breakout(
    close: np.ndarray,
    volume: np.ndarray,
    neck: float,
    start: int,
    buffer: float,
    vol_ratio: float,
    last: int,
) -> _Breakout:
    """颈线放量突破扫描。

    维持线用颈线本身（收盘不跌破即算守住），buffer 只用于**初始触发**——
    否则一次回踩收到颈线上方 0.3% 就会被判成突破作废，与"回踩收盘不跌破颈线
    视为正式确立"相矛盾。

    触发 K 必须同时满足「收盘 > 颈线×(1+buffer)」与「量 ≥ 前 5 日均量×vol_ratio」。
    先过价格、后补量能也认（同一段行情内），所以弱量突破只会停在构筑中，
    等到补量那天才转已形成。

    返回两个触发点，对应两种问法：
    - `current_idx`：**当前**这段站上颈线的行情里有没有放量突破。B1 的"已形成"
      必须看它——收盘已经跌回颈线下方就是没确认。
    - `first_idx`：结构内**首次**放量突破颈线。B2 的回踩本来就会把收盘压回颈线
      下方（甚至再放量突破一次），它需要的是"第一波行情从哪起算"，
      取最近一次会把自己后来的突破当起点，H1 直接塌到当前那根上。
    """
    in_run = False
    current_idx: int | None = None
    current_ratio = 0.0
    first_idx: int | None = None
    first_ratio = 0.0
    tail_ratio = _VOL_FALLBACK
    for i in range(start, last + 1):
        if close[i] < neck:
            in_run = False
            current_idx = None
            current_ratio = 0.0
            continue
        if not in_run:
            in_run = True
            current_idx = None
            current_ratio = 0.0
        ratio = _prior5_vol_ratio(volume, i)
        if i == last:
            tail_ratio = ratio
        if current_idx is None and close[i] > neck * (1 + buffer) and ratio >= vol_ratio:
            current_idx = i
            current_ratio = ratio
            if first_idx is None:
                first_idx = i
                first_ratio = ratio
    return _Breakout(
        current_idx=current_idx,
        current_ratio=current_ratio if current_idx is not None else tail_ratio,
        first_idx=first_idx,
        first_ratio=first_ratio,
        run_intact=in_run,
    )


def _tradeability(base: BreakBottomBase) -> tuple:
    """多头结构之间的优先级：当前已确认 > 历史上放量突破过 > 尚未突破。

    第三项在同类里按探底/突破的时新度排序，保证"同样还没确认"时取最新的那一段。
    """
    if base.breakout_idx is not None:
        return (3, base.breakout_idx)
    if base.first_breakout_idx is not None:
        return (2, base.first_breakout_idx)
    return (1, base.low_idx)


def _find_break_bottom_base(
    ctx: ScanContext, cfg_bb, cfg_b2=None
) -> BreakBottomBase | None:
    """定位破底翻基础结构，失败原因逐条写进 trace。B1 / B2 共用。

    cfg_b2 提供时用 B2 自己的回看口径（break_within_b2，破底新低可以更早）；
    其余结构阈值一律取 cfg_bb，保证两个检测器对"什么算破底翻"的判断逐字一致。
    """
    df, n = ctx.df, ctx.n
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    last = n - 1

    max_age = int(cfg_b2.break_within_b2) if cfg_b2 is not None else int(cfg_bb.break_within)
    if n < 30:
        ctx.note(f"[破底翻] K线仅{n}根，不足30根")
        return None

    triples = _pivot_triples(ctx.pivots)
    if not triples:
        ctx.note("[破底翻] 无 (低-高-低) 摆动点三元组")
        return None

    valid: list[BreakBottomBase] = []
    why = {
        "depth": 0, "shallow": 0, "span": 0, "age": 0, "neck": 0, "drop": 0,
        "norecover": 0, "order": 0, "stale": 0,
        "fakelow": 0, "earlybreak": 0, "relapse": 0,
    }
    # "已形成但已过期"的候选在 detect 里注定被时效闸门丢弃，
    # 让这种候选进入候选池只会把更新鲜的结构盖掉（报表上变成什么都没命中）。
    # 只对 B1 生效——B2 就是冲着更久之前的结构来的。
    stale_after = None if cfg_b2 is not None else int(cfg_bb.breakout_within)
    for low1, neck_p, low2 in triples:
        if low1.price <= 0 or not (low1.idx < neck_p.idx < low2.idx):
            why["order"] += 1
            continue

        # 1. 前低必须是这段下跌的最低点：L0 之前 prior_lookback 根内不能有更低的价。
        #    只要求"L0 之前跌过 prior_drop"是不够的——下跌途中任取一个中间低点，
        #    右边照样收得出新低，但那个"新低"离真支撑还差一截，"破底"是假的，
        #    颈线也多半是深跌里的一次脉冲反弹，量度目标直接飞出合理区间。
        #    用原始 low 而不是摆动低点：插针砸出的更低点同样是"支撑已被破"。
        floor_start = max(0, low1.idx - cfg_bb.prior_lookback)
        if float(low[floor_start : low1.idx + 1].min()) < low1.price:
            why["fakelow"] += 1
            continue

        # 2. 二次探底：是浅破位，但也不能只是噪声级的新低。
        #    下限为什么必要：允许夹心摆动点之后，长间隔的 (低,高,低) 组合数量很大，
        #    箱体震荡里跌 0.x% 的"新低"也会凑成候选（实测占命中数近一成，
        #    典型如 300509：破底仅 0.8%、颈线只 +5.2%，纯粹在区间里荡）。
        #    "有效跌破"得能看出是破了支撑，而不是分时噪声。
        depth = (low1.price - low2.price) / low1.price
        if depth > cfg_bb.max_break_depth:
            why["depth"] += 1
            continue
        if depth < cfg_bb.min_break_depth:
            why["shallow"] += 1
            continue

        # 3. 两个低点的间隔须落在 1~3 个月内
        span = low2.idx - low1.idx
        if not (cfg_bb.min_span <= span <= cfg_bb.max_span):
            why["span"] += 1
            continue

        # 4. 新低须够新
        if last - low2.idx > max_age:
            why["age"] += 1
            continue

        # 5. 颈线必须有实质高度——没有像样的前置反弹，就不是"双底"结构
        neck_height = (neck_p.price - low1.price) / low1.price
        if not (cfg_bb.min_neck_height <= neck_height <= cfg_bb.max_neck_height):
            why["neck"] += 1
            continue

        # 6. 前置结构：L0 之前必须是一段下跌
        if prior_drop_pct(df, low1.idx, cfg_bb.prior_lookback) < cfg_bb.prior_drop:
            why["drop"] += 1
            continue

        # 7. L0 与颈线之间允许出现更低的 low（插针、非结构性下探都算），
        #    但如果那个更低的 low 已经构成了 ZigZag 结构性 Swing Low，
        #    说明 L0 支撑在时序上已被打破，后面那个"颈线"只是深跌途中的一次脉冲反弹，
        #    "前低 → 反弹 → 破底 → 收复"的时序根本立不住。
        #    （002030 达安基因：L0 5.17(06-17)，06-29 就插到 4.80；
        #    它认的颈线 5.43 要到 07-16、"破底日"更是 07-21——中间早破过。）
        swing_lows_between = [
            p for p in ctx.pivots
            if p.kind == LOW and low1.idx < p.idx < neck_p.idx
        ]
        early_break = next(
            (p for p in swing_lows_between if p.price < low1.price),
            None,
        )
        if early_break is not None:
            why["earlybreak"] += 1
            continue

        # 至于"L0 之后确实跌破了 L0"，不再单设闸门：破底取 **low**，而候选枚举本身就
        # 要求 low2 是更低的摆动低点（其价位就是那根的 low），所以"跌破"必然成立。
        # 旧口径要求**收盘**跌破（插针不算），与 v2 正相反；2026-09-29 按统一口径改成
        # 影线口径，见 test_intraday_only_undercut_is_accepted。

        # 8. 快速止跌：low_idx 后 recover_bars 内收盘收复 L0，不得在新低区盘整
        recover_idx = next(
            (
                i
                for i in range(
                    low2.idx + 1,
                    min(low2.idx + cfg_bb.recover_bars, last) + 1,
                )
                if close[i] > low1.price
            ),
            None,
        )
        if recover_idx is None:
            why["norecover"] += 1
            continue

        # 9. 收复之后不得再"翻失败"：收盘连续 pullback_invalid_consecutive 根落在 L0
        #    之下，说明这次收复只是插曲，价格又回到新低区盘整（300354 东华测试：
        #    08-26 收 26.85 收复 L0=26.59 之后，09-10/09-11 连收 26.55/25.80）。
        #    detect() 末尾也查这条，但只看**最后两根**，翻回来之后才失败的看不到——
        #    300354 到 09-16 又收 26.95，末尾检查就被绕过去了。
        #    单根插破不算（300353 东土科技收复后收在 L0 下方 1.9% 又立刻收回）。
        run = 0
        relapse = False
        for i in range(recover_idx + 1, last + 1):
            run = run + 1 if close[i] < low1.price else 0
            if run >= cfg_bb.pullback_invalid_consecutive:
                relapse = True
                break
        if relapse:
            why["relapse"] += 1
            continue

        breakout = _neckline_breakout(
            close,
            volume,
            neck_p.price,
            low2.idx + 1,
            cfg_bb.neck_breakout_buffer,
            cfg_bb.neck_breakout_vol_ratio,
            last,
        )
        if (
            stale_after is not None
            and breakout.current_idx is not None
            and last - breakout.current_idx >= stale_after
        ):
            why["stale"] += 1
            continue

        # C点检测：从 recover_idx 开始找首个收盘站上颈线的位置（图2的"站回确认"）
        c_point_idx = None
        c_point_price = 0.0
        c_point_valid = False
        for i in range(recover_idx, last + 1):
            if close[i] > neck_p.price * (1 + cfg_bb.neck_breakout_buffer):
                c_point_idx = i
                c_point_price = float(close[i])
                c_ratio = _prior5_vol_ratio(volume, i)
                c_point_valid = c_ratio >= cfg_bb.neck_breakout_vol_ratio
                break

        # 前波高点检测：L0之前最近的摆动高点（图2的"前波高点"目标位）
        prior_high_idx = None
        prior_high = 0.0
        for p in reversed(ctx.pivots):
            if p.kind == HIGH and p.idx < low1.idx:
                prior_high_idx = p.idx
                prior_high = float(p.price)
                break

        valid.append(
            BreakBottomBase(
                l0_idx=low1.idx,
                l0=float(low1.price),
                neck_idx=neck_p.idx,
                neck=float(neck_p.price),
                low_idx=low2.idx,
                low=float(low2.price),
                recover_idx=recover_idx,
                depth=depth,
                span=span,
                neck_height=neck_height,
                breakout_idx=breakout.current_idx,
                breakout_vol_ratio=breakout.current_ratio,
                run_intact=breakout.run_intact,
                first_breakout_idx=breakout.first_idx,
                first_breakout_vol_ratio=breakout.first_ratio,
                c_point_idx=c_point_idx,
                c_point_price=c_point_price,
                c_point_valid=c_point_valid,
                prior_high_idx=prior_high_idx,
                prior_high=prior_high,
            )
        )

    if not valid:
        detail = " ".join(f"{k}:{v}" for k, v in why.items() if v)
        ctx.note(f"[破底翻] {len(triples)}个(低-高-低)候选均不成立（{detail}）")
        return None

    # 多个结构同时成立时，取"最可交易"的那一个，而不是最近的。
    # 只看最新的候选会让一根新鲜但还没突破颈线的结构，把更早、已经放量站上
    # 颈线的那一段彻底盖住——报表上就只剩"构筑中"，永远看不到"已形成"。
    # 反过来，"已形成但突破太旧"的候选在收集阶段就被跳过（why["stale"]），
    # 否则它会排到可交易性最高、再被 detect 的时效闸门整个丢弃，
    # 把本该命中的新鲜结构也一起埋掉。
    best = max(valid, key=_tradeability)
    ctx.note(
        f"[破底翻] 前低{best.l0:.2f}({df.index[best.l0_idx]:%m-%d}) "
        f"颈线{best.neck:.2f}(+{best.neck_height*100:.1f}%) "
        f"破底至{best.low:.2f}(深{best.depth*100:.1f}%) 间隔{best.span}根 "
        f"收复用{best.recover_idx - best.low_idx}根"
    )
    return best


def detect(ctx: ScanContext) -> list[PatternHit]:
    cfg = ctx.cfg
    c = cfg.break_bottom
    if not c.enabled:
        return []

    df, n = ctx.df, ctx.n
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    last = n - 1

    base = _find_break_bottom_base(ctx, c)
    if base is None:
        return []

    l0 = base.l0
    low = base.low
    neck = base.neck
    recover_idx = base.recover_idx
    recover_used = recover_idx - base.low_idx

    # ---- 失效检查：与是否已突破颈线无关 ----
    atr_last = _safe(df["atr"], last, l0 * 0.03)
    if last >= 1 and close[last] < l0 and close[last - 1] < l0:
        ctx.note("[破底翻] 连续2日收盘跌破L0，形态失效")
        return []
    if close[last] < l0 - c.stop_atr_buffer * atr_last:
        ctx.note(f"[破底翻] 收盘跌破 L0-{c.stop_atr_buffer}×ATR，形态失效")
        return []

    # ---- 状态判定（对应图2的4阶段：下跌→破底→翻回→确认） ----
    # 图1区分两种入场点：激进型买点（支撑收复）vs 保守型买点（颈线突破）
    if base.breakout_idx is not None and base.breakout_idx < last:
        status = STATUS_FORMED
        signal_idx = base.breakout_idx
        stage = "确认-颈线突破(保守买点)"
    elif base.breakout_idx is not None:
        # 突破就发生在今天：等次日站稳再确认，避免单根长阳当确认
        status = STATUS_FORMING
        signal_idx = last
        stage = "确认-今日突破待站稳(保守买点)"
    elif base.c_point_idx is not None:
        # C点已现（站上颈线）但量能不足，等待补量
        status = STATUS_FORMING
        signal_idx = base.c_point_idx
        stage = "翻回-站上颈线待量(C点已现)"
    elif base.run_intact and close[last] > neck:
        # 价格站在颈线上方但未触发突破条件
        status = STATUS_FORMING
        signal_idx = last
        stage = "翻回-接近颈线"
    elif recover_idx is not None and recover_idx <= last:
        # 已收复L0（激进买点），等待进一步突破
        status = STATUS_FORMING
        signal_idx = recover_idx
        stage = "翻回-收复支撑(激进买点)"
    else:
        status = STATUS_FORMING
        signal_idx = last
        stage = "破底-待收复"

    if status == STATUS_FORMING and last - base.low_idx > c.forming_max_age:
        ctx.note(f"[破底翻] 新低距今{last - base.low_idx}根仍未放量突破颈线，超期丢弃")
        return []

    # ---- 几何分 ----
    neck_touches = count_level_touches(
        df, neck, base.l0_idx, base.low_idx + 1, tol=0.02, use="high"
    )
    held = base.breakout_idx is not None and base.run_intact
    geom = np.mean(
        [
            taper(base.depth, 0.0, c.max_break_depth),
            taper(recover_used, 1, c.recover_bars),
            taper(base.neck_height, 0.30, c.min_neck_height),
            taper(neck_touches, 2, 0),
            1.0 if held else 0.7,
        ]
    )

    # ---- 量能分：破底日缩量 + 收复日温和放量 + 颈线突破放量 是理想组合（图1） ----
    vma_break = _safe(df["vol_ma5"], base.low_idx)
    break_vol_ratio = volume[base.low_idx] / vma_break if vma_break > 0 else _VOL_FALLBACK
    shrink = taper(break_vol_ratio, 0.6, 1.3)

    # 收复日量能：温和放量（1.0-1.5倍）最佳，天量反而不好（可能是脉冲）
    vma_recover = _safe(df["vol_ma5"], recover_idx)
    recover_vol_ratio = volume[recover_idx] / vma_recover if vma_recover > 0 else _VOL_FALLBACK
    recover_vol_score = taper(recover_vol_ratio, 1.2, 0.8)

    if base.breakout_idx is not None:
        bo_ratio = base.breakout_vol_ratio
    else:
        vma_last = _safe(df["vol_ma5"], last)
        bo_ratio = volume[last] / vma_last if vma_last > 0 else _VOL_FALLBACK
    expand = taper(bo_ratio, c.neck_breakout_vol_ratio * 1.5, c.neck_breakout_vol_ratio)

    # 三部分加权：破底缩量40% + 收复温和放量20% + 突破放量40%
    vol_score = 0.4 * shrink + 0.2 * recover_vol_score + 0.4 * expand

    # ---- 趋势分（软因子，不满足只降分不拒绝） ----
    hist_last = _safe(df["macd_hist"], last)
    hist_prev = _safe(df["macd_hist"], max(0, last - 1))
    divergence = _safe(df["macd_hist"], base.low_idx) > _safe(df["macd_hist"], base.l0_idx)
    ma5 = _safe(df["ma5"], last)
    ma10 = _safe(df["ma10"], last)

    rsi_last = _safe(df["rsi"], last, 50.0)
    rsi_prev1 = _safe(df["rsi"], max(0, last - 1), 50.0)
    rsi_prev2 = _safe(df["rsi"], max(0, last - 2), 50.0)
    rsi_not_weakening = not (rsi_last < rsi_prev1 < rsi_prev2)
    macd_improving = hist_last > hist_prev
    # DIF/DEA 金叉状态：与 macd_improving（柱子回升）是独立因子，不能重复计入
    dif_last = _safe(df["dif"], last)
    dea_last = _safe(df["dea"], last)
    macd_golden = dif_last > dea_last

    trend = (
        0.15 * (1.0 if ma5 and close[last] > ma5 else 0.0)
        + 0.10 * (1.0 if ma10 and close[last] > ma10 else 0.0)
        + 0.15 * (1.0 if macd_improving else 0.0)
        + 0.20 * (1.0 if divergence else 0.0)
        + 0.20 * (1.0 if rsi_not_weakening else 0.0)
        + 0.20 * (1.0 if macd_golden else 0.0)
    )

    score, gs, vs, ts = blend(geom, vol_score, trend, status, cfg, priority="A")

    # ---- 目标价：颈线量度升幅（双底/破底翻的经典量度） ----
    # 确认信号本身要求收盘站上颈线，目标价必须锚在颈线之上，否则预期收益恒为 0，
    # 收益质量拿不到分，交易模块的 min_target_upside 闸门也会把信号全部否掉。
    resistance = neck + (neck - low)
    # 图2标注的"前波高点"作为备选目标（如果存在）
    prior_high_target = base.prior_high if base.prior_high > 0 else None

    # ---- 止损：与"结构在什么条件下被证伪"对齐，按状态取不同的锚 ----
    # 已形成（收盘已站上颈线）：定义说"回踩收盘不跌破颈线位才视为正式确立"，
    # 所以失效位就是颈线，止损挂在颈线下方。以前不管状态一律用 L0−0.5×ATR，
    # 对已形成来说等于把止损挂到 颈线高度+0.5×ATR 那么远（实测 13~15%），
    # 固定风险法算出的仓位被压得极小、RR 被拖进 ~1.2，交易侧基本不可用。
    # 构筑中（尚未确认）：结构失效以"跌回前低下方"为准，止损挂在 L0 下方。
    atr_at_signal = _safe(df["atr"], base.breakout_idx or recover_idx or last, l0 * 0.03)
    if status == STATUS_FORMED:
        stop_loss = neck - c.stop_atr_buffer * atr_at_signal
    else:
        stop_loss = l0 - c.stop_atr_buffer * atr_at_signal

    # ---- 入场点（图1的两种买点） ----
    aggressive_entry = float(close[recover_idx])  # 激进型买点：支撑收复价
    conservative_entry = neck  # 保守型买点：颈线价位

    # ---- 收复日量能（图1的"支撑收复"阶段） ----
    vma_recover = _safe(df["vol_ma5"], recover_idx)
    recover_vol_ratio = volume[recover_idx] / vma_recover if vma_recover > 0 else _VOL_FALLBACK

    # 构造破底翻关键摆动点，让图表上能画出标记点
    bb_pivots = [
        Pivot(idx=base.l0_idx, date=df.index[base.l0_idx], price=l0, kind=LOW, confirmed=True),
        Pivot(idx=base.neck_idx, date=df.index[base.neck_idx], price=neck, kind=HIGH, confirmed=True),
        Pivot(idx=base.low_idx, date=df.index[base.low_idx], price=low, kind=LOW, confirmed=True),
    ]

    return [
        PatternHit(
            code=ctx.code,
            name=ctx.name,
            pattern=PATTERN,
            pattern_cn=PATTERN_CN,
            status=status,
            score=score,
            geom_score=gs,
            vol_score=vs,
            trend_score=ts,
            signal_date=df.index[signal_idx],
            key_levels={
                "参考低点": l0,
                "颈线": neck,
                "颈线高度%": base.neck_height * 100,
                "破底最低": low,
                "破底深度%": base.depth * 100,
                "突破量比": bo_ratio,
                "收复价": close[recover_idx],
                "收复量比": recover_vol_ratio,
                "收复用根数": recover_used,
                "激进型买点": aggressive_entry,
                "保守型买点": conservative_entry,
                "C点": base.c_point_price if base.c_point_idx is not None else 0.0,
                "C点有效": base.c_point_valid,
                "前波高点": prior_high_target if prior_high_target else 0.0,
                "参考止损": stop_loss,
                "上方压力": resistance,
            },
            detail=(
                f"前低{l0:.2f}({df.index[base.l0_idx]:%m-%d}) "
                f"颈线{neck:.2f}(+{base.neck_height*100:.1f}%) "
                f"破底至{low:.2f}(深{base.depth*100:.1f}%) 间隔{base.span}根 "
                f"收复用{recover_used}根 {stage} "
                f"突破量比{bo_ratio:.2f} 收复量比{recover_vol_ratio:.2f} "
                f"颈线被测试{neck_touches}次 "
                f"破底日量{break_vol_ratio:.2f}倍"
                + (f" C点{base.c_point_price:.2f}" if base.c_point_idx is not None else "")
                + (f" 前高{prior_high_target:.2f}" if prior_high_target else "")
                + (" MACD底背离" if divergence else "")
                + ("" if rsi_not_weakening else " RSI走弱")
            ),
            pivots=bb_pivots,
            close_price=float(close[last]),
            rsi=rsi_last,
            macd_hist=hist_last,
        )
    ]
