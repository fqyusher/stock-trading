# trade-yh

A股K线形态扫描工具：全市场自动识别技术图形，多维度评分，结合市场环境 / 情绪 / 行业热度乘性修正，盘中监控站稳成型即提醒，并可模拟自动交易（含东财 EMT 极速柜台仿真）：固定风险开仓 → 移动止盈 → 分批止盈 → 失败突破离场。

图形识别与交易方向**解耦**：下降通道 / M头是有效图形，但不是做多机会。做多榜单和观察池只取"做多"方向，做空形态仍进报表当规避参考。

## 支持的形态

| 形态 | key | 方向 | 优先级 | 一句话 |
|------|-----|------|:---:|------|
| 破底翻 | `break_bottom` | 做多 | A | 跌破前低后快速收复，空头陷阱（B1 买点） |
| 破底翻第二买点 | `break_bottom_b2` | 做多 | S | 破底翻确认后缩量回踩，放量再破前高（B2 买点） |
| 破底翻v2 | `break_bottom_v2` | 做多 | A | **已停用**。ATR 严格口径，数据对比后暂停使用 |
| 破底翻v2第二买点 | `break_bottom_v2_b2` | 做多 | S | **已停用**。v2 B2，随 v2 一并停用 |
| W底 | `w_bottom` | 做多 | A | 双底，突破颈线确认 |
| M头 | `m_top` | 做空 | A | 双顶，跌破颈线确认 |
| 上升通道 | `rising_channel` | 做多 | B | 上升趋势中的通道整理 |
| 下降通道 | `falling_channel` | 做空 | B | 下降趋势中的通道整理 |
| 下降通道反转 | `falling_channel_reversal` | 做多 | A | 下降通道触下轨且止跌 |
| 横盘箱体 | `box_range` | 做多 | C | 箱体震荡，下沿买上沿卖 |
| 平台突破 | `platform_breakout` | 做多 | S | 上涨后窄幅整理，突破上沿 |
| 突破回踩确认 | `breakout_retest` | 做多 | S | 突破后回踩不破，二次启动 |
| 杯柄突破 | `cup_handle` | 做多 | B | 杯柄 / 圆弧底突破杯沿 |
| 趋势回踩再启动 | `trend_pullback` | 做多 | S | 上升趋势回踩均线后再起 |

> **优先级**（S/A/B/C → 权重 1.0/0.95/0.88/0.80）反映信号可靠性：结构约束越硬的形态权重越高，靠回归拟合成立的通道类偏低。详见[评分体系](#评分体系)。

> **破底翻 v2 已停用**：`break_bottom_v2` / `break_bottom_v2_b2` 在 config 里 `enabled: false`，扫描和监控均不加载。代码保留供参考，不参与报表与观察池。

---

## 形态识别逻辑

所有检测器签名统一：`detect(ctx) -> list[PatternHit]`。`ScanContext` 里已备好附加了指标的 `df`、ZigZag 摆动点 `pivots`、以及 explain 模式的 `trace`。每个命中给出 `key_levels`（关键价位）、`status`（已形成 / 构筑中）、几何 / 量能 / 趋势三个子分。

每个形态的**目标价**和**止损**取自 `key_levels` 的哪个键，是表驱动的（`patterns/base.py`）——漏登记会在测试里暴露，而不是静默退化成"无目标价"。速查：

| 形态 | 目标价键 | 止损键（参考止损）取值 |
|------|---------|---------|
| 破底翻 | `上方压力` = 颈线 + (颈线 − 破底最低) | 已形成 = `颈线 − 0.5×ATR14`，构筑中 = `L0 − 0.5×ATR14` |
| 破底翻第二买点 | `等幅目标` = H1 + (H1 − L1) | `回踩低点 L1 − 0.5×ATR14` |
| 破底翻v2 | `tp1_price` = H1（第一波高点） | `stop_price` = `max(L0 − 0.3×ATR14, L1 − 0.5×ATR14)` |
| 破底翻v2第二买点 | `tp1_price` = H1 + (H1 − L1) | `stop_price` = `max(L0 − 0.3×ATR14, L1 − 0.5×ATR14, pl − 0.3×ATR14)` |
| W底 | `等幅目标` = 颈线×(1+颈线高度) | `底部 × 0.97` |
| M头 | `等幅目标` = 颈线×(1−颈线深度) | `颈线 × 1.03` |
| 上升通道 / 箱体 | `上轨` | `下轨 × 0.97` |
| 下降通道 | `下轨` | `上轨 × 1.03` |
| 下降通道反转 | `等幅目标` = 上轨 | `下轨 × 0.97` |
| 平台突破 / 突破回踩 / 杯柄 | `等幅目标` = 突破位+(突破位−下沿) | `整理下沿 × 0.97` |
| 趋势回踩再启动 | `近期高点` | `回踩支撑 × 0.97` |

参数全部集中在 `config.yaml` 对应的形态块里，下文括号内为默认值。

### 识别地基：ZigZag 摆动点与趋势背景

形态识别的地基是 `swings.py` 的 **ZigZag 摆动点**——把 K 线序列压缩成严格高低交替的转折点（`Pivot`：峰 H / 谷 L）。

- **自适应阈值**：某一波要算作一次反转，幅度必须达到 `max(pct_min, atr_k × ATR/price)`（`swings.pct_min=0.05`，`atr_k=1.5`）。固定百分比阈值在低波动股上会把噪声拟合成摆动点，在高波动股上又会漏掉真形态，所以随波动率自适应。
- **末端未确认点**：序列尾部的极值会作为**未确认** pivot（`confirmed=False`）挂上去。"构筑中"的形态（右底刚出现、还没反弹够幅度）全靠这个点才能被看到。
- **辅助工具**：`count_level_touches`（某价位被反复测试的次数，相邻触碰合并为一次，用于颈线/支撑的重要性）、`prior_drop_pct` / `prior_rise_pct`（前期跌幅 / 涨幅）。

**趋势背景** `ctx.trend_direction`（基于 MA20 斜率 + MA20 vs MA60）：

- `up`：MA20 > MA60 且 MA20 近5根斜率 > +0.5%
- `down`：MA20 < MA60 且斜率 < −0.5%
- 否则 `neutral`

整理突破、趋势回踩这类做多形态在 `down` 趋势下**直接跳过**（不逆势找多头）。

### 通用状态机：已形成 / 构筑中 / 失效

- **已形成（formed）**：突破/破位/收复已经发生并被后续 K 线确认。
- **构筑中（forming）**：右侧结构已现、价格正朝成型方向移动，但关键一步（突破颈线、收复前低……）尚未完成。构筑中在评分里打折（见评分体系），且是观察池盘中监控的主要对象。
- **失效**：关键点距今超期、破得太深、突破发生太久形态已走完等，直接丢弃。
- **已失效（invalid）**：只有 v2 口径会**显式输出**这一档（`status="invalid"`、`失效` 列 = "是"），用来显示"刚刚失效"这件事；它的阶段确定性记 0，且被做多榜单与观察池过滤掉。旧口径仍旧是"失效就不命中、不出行"。

---

### 破底翻 `break_bottom`（做多 · 优先级 A）

跌破前期重要低点后又快速收回站上——空头陷阱 / 假突破。完整流程：**前低 L0 → 反弹到颈线 N → 二次探底收盘跌破 L0 → 快速收复 → 放量收盘站上颈线**。

1. **结构候选**：从 ZigZag 序列里取所有 `(低, 高, 低)` 三元组（与 W底 同一套枚举，`w_bottom._candidates` 的同型写法），逐个校验，不靠"窗口内最低价"反推——那样得到的 L0 只是"某条带里的最低价"，不是真正的前低。三元组给出的是**候选**前低（它是不是这段下跌真正的低点，由第 2 条把关）；颈线取自**峰 pivot**，单 bar 插针不会被当成颈线。
2. **前低必须是这段下跌的最低点**：L0 之前 `prior_lookback`（60）根内不能有更低的价（比的是原始 `low`，插针砸出的更低点同样算"支撑已被破"）。只要求"L0 之前跌过 `prior_drop`"是不够的——下跌途中任取一个中间低点都能凑出 `(低, 高, 低)` 三元组，右边照样收得出新低，但那个"新低"离真支撑还差一截，所谓破底根本没破到任何支撑，颈线也多半是深跌里的一次脉冲反弹、量度目标直接飞出合理区间。实测 2026-09-21 的全市场报告：不加这条时 84 条命中里 62 条是这种伪结构（例：国际复材 301526 的"前低 27.30"之前 9 根就是 24.05，二次探底 26.08 根本没破到 24.05，颈线却取了 4 天脉冲高点 40.69、把量度目标顶到 +77%）。
3. **两低点的空间与周期**：二次探底必须真的创新低（`low < L0`），破底深度 `(L0 − low)/L0 ≤ max_break_depth`（**0.10**）——跌太深是真破位不是"翻"，这一条比几何形状更能决定误报率；两个低点的间隔 `low_idx − l0_idx ∈ [min_span, max_span]`（**20~63 根**，约 1~3 个月，诱空结构是短期的）。
4. **新低须够新**：`last − low_idx ≤ break_within`（63，与 `max_span` 同量级）——探底本身超过一个季度，整段结构就不再是"短期诱空"了。至于"够不够新鲜、还值不值得动手"，交给 `breakout_within`（已形成）与 `forming_max_age`（构筑中）判，而不是压在探底那一步上。
5. **颈线高度**：`(N − L0)/L0 ∈ [min_neck_height, max_neck_height]`（[0.05, 0.60]）——没有像样的前置反弹就谈不上"双底"，这条也顺手挡住颈线退化成 ≈L0 的情形。
6. **前置结构**：L0 之前 `prior_lookback`（60）根内跌幅 ≥ `prior_drop`（0.15）——明确的下跌趋势才是"破底"的前提。
7. **有效跌破 = 收盘跌破**：`(neck_idx, low_idx]` 内出现 `close < L0`。只有盘中插针破 L0、收盘从未跌破的不算——那正是洗盘与"真破位"的分界。
8. **快速止跌**：`low_idx` 之后 `recover_bars`（5）内第一根**收盘 > L0**。不在新低区盘整是形态定义的一部分，收复不了就不报（不再保留旧代码里"尚未有效收复也报构筑中"那条分支）。
9. **颈线放量突破 = 翻转确认**：扫描连续收盘站上颈线的行情段（维持线用颈线本身，`close < N` 即重置；`neck_breakout_buffer`（0.005）只用于**初始触发**，避免回踩收在颈线上方 0.3% 就被判成突破作废）。段内首个同时满足「收盘 > `N×(1+0.005)`」与「量 ≥ **前 5 日均量**×`neck_breakout_vol_ratio`（1.3）」的 K 是触发点。
   - 量比分母取**前** 5 根而不是 `VOL_MA5`：后者把当日自己也算进均值，当日放量会把分母一起抬高，"放量 30%"变成事实上的更严。
   - 先过价格、后补量能也认（同一段行情内），所以弱量突破只会停在构筑中，等到补量那天才转已形成。
10. **失效检查（无条件）**：连续 2 日收盘 < L0，或收盘 < `L0 − stop_atr_buffer×ATR14`（0.5）→ 形态直接作废，不再进池。
11. **状态**：
    - **已形成**：当前这段站上颈线的行情里出现了放量突破，且其后至少还有一根 K 维持站上颈线（避免把单根长阳当确认）。收盘一旦跌回颈线下方，这段行情作废、退回构筑中——这正是"回踩收盘不跌破颈线才视为正式确立"。突破距今 ≥ `breakout_within`（10）根 → 该候选在结构定位阶段就被跳过（"已形成但已过期"），否则它会按可交易性排到最高、再被时效闸门整个丢弃，把本该命中的新鲜结构也一起埋掉。
    - **构筑中**：已收复 L0 但尚未触及颈线；或收盘已站上颈线但突破日量能不足；或放量突破就发生在今天。新低距今 > `forming_max_age`（30）根仍未放量突破 → 丢弃。
- **几何分**：破得越浅 / 收复越快 / 颈线高度越充分 / 颈线被反复测试 / 突破后回踩守住颈线。
- **量能分**：破底日缩量（0.4）+ 颈线突破放量（0.6）是最理想组合。
- **趋势分**（软因子，不满足只降分不拒绝）：站上 MA5（0.15）+ 站上 MA10（0.10）+ MACD 柱回升（0.15）+ MACD 底背离（0.20）+ RSI 不持续走弱（0.20）+ DIF>DEA 金叉状态（0.20）。
- **关键价位**：目标 `上方压力` = **颈线量度升幅** `N + (N − 破底最低)`；止损**按状态取锚**——已形成 = `颈线 − stop_atr_buffer×ATR14`（定义说"回踩收盘不跌破颈线才视为正式确立"，确认后的失效位就是颈线，挂到 L0 下方会让止损比入场低 颈线高度+0.5×ATR，仓位与 RR 都被压垮），构筑中 = `L0 − stop_atr_buffer×ATR14`（尚未确认，结构失效以"跌回前低下方"为准）；`key_levels` 另给 `颈线`、`颈线高度%`、`突破量比`、`收复价`、`收复用根数`、`破底深度%`。
  - 目标价必须锚在颈线之上：确认信号本身要求收盘站上颈线，若沿用旧的"破底前平台上沿"口径，那个价位会被压到 ≈ 颈线，`expected_profit()` 恒为 0、收益质量拿不到分，交易模块的 `min_target_upside`（8%）也会把所有信号否掉。

### 破底翻第二买点 `break_bottom_b2`（做多 · 优先级 S）

破底翻确认之后的**第二次启动买点**：破底 → 快速收复 → 放量站上颈线 → 第一波反弹 → **缩量回踩守住收复位** → 放量突破前高。B2 比 B1 稳健，因为已经过一次完整验证，且"回踩不破 + 放量破前高"是趋势再启动的信号；代价是位置更高、买得更贵、空间更小。

优先级记 **S**（B1 为 A）：B2 比 B1 多出「回踩不破 L0 / HL 抬升 / 放量破前高」三道硬闸门，信号可靠性显著更高，与 B1 同级会让权重设计失去依据。

**前提**：基础结构由 `break_bottom._find_break_bottom_base` 统一判定——结构定位**只有一份实现**，B1 / B2 不会各自漂移成两套口径。要求结构成立（前低是这段下跌的最低点 / 颈线 / 收盘破底 / 快速收复 / 间隔 1~3 个月，全部阈值取 `cfg.break_bottom`），并且颈线**曾被放量突破**（`first_breakout_idx`）。基础结构或突破定位失败时，具体原因（前低之前还有更低的价 / 深度超限 / 间隔超窗口 / 未跌破前低 / 未收复 L0 / 从未放量突破颈线……）都会写进 `trace`，`--explain` 能直接看到卡在哪一条。

- B2 的回踩本来就会把收盘压回**颈线下方**（只要求不破 L0），所以前提取的是"结构内**首次**放量突破颈线"，而不是"当前这段行情里有没有放量突破"——后者在回踩发生时必然为空。取**最近**一次更不行：B2 自己的"突破前高"就是一次颈线之上的放量突破，拿它当第一波起点，H1 会直接塌到当前那根上。
- 突破后一路跌破 L0 的情形不算前提成立，由下面的结构失效闸门负责。
- `min_bars_after_breakout`（8）：颈线突破后至少留出 8 根，才可能形成第一波 + 回踩。
- B2 的 `break_within_b2`（100）比 B1 的 20 宽得多——破底新低可以发生在更早的位置，代价是只认窗口内最低的那个新低。

1. **第一波高点 H1**：**颈线突破点**之后，第一个**已确认的 ZigZag 峰**，且反弹幅度 ≥ `max(rebound_min_pct=5%, rebound_min_atr_k×ATR/min_low)`。用摆动点而非"运行最高价 + 回落 1%"：后者会把盘中插针当 H1，而 H1 是后续突破的基准（要求**收盘**突破），插针高点永远无法被收盘突破，形态只会永远卡在构筑中。
2. **回踩低点 L1**：H1 之后第一个**已确认的 ZigZag 谷**，回撤幅度 ≥ `max(pullback_min_pct=2%, pullback_min_atr_k×ATR/H1)`。若价格已回踩够幅度但还没反弹到能让 ZigZag 确认这个谷（"构筑中"最常见的情形），取该区间实际最低点兜底。
3. **回踩不得破坏结构**（核心闸门）：回踩段内连续 `pullback_invalid_consecutive`（2）日收盘 < L0，或任一日收盘 < `L0 − pullback_l0_atr_buffer×ATR`（0.5）→ 直接丢弃。此外要求 `L1 ≥ L0`（Higher Low，回踩低点必须抬升）。
   - 这道闸门从 H1 之后**一路查到当日**（以前只查到 L1 为止，L1 之后到当日的破坏无人检查——价格反弹失败、又跌回 L0 下方时形态照样以"构筑中"进池，而止损锚在 L1、已经落到现价上方）。
   - 残留边界：价格回落到 **L1 下方但始终没跌破 L0**（只跌到 L0 上方）时，形态仍算"构筑中"，此时参考止损同样会高于现价。这一层由评分口径兜住（止损落在现价错误一侧 → 收益质量记 0），实盘也不会买入（只有"已形成"才会触发），但结构上仍应收敛。
4. **缩量回踩**：回踩段均量 ≤ 第一波均量 × `vol_shrink_ratio`（0.85）。放量回踩说明是在派发，不是洗盘。两段口径：**第一波 = `[颈线突破点, H1]`，回踩段 = `(H1, L1]`**——起点与 H1 的起算点一致（从收复点起算会把颈线以下那一段、含放量的收复/突破也算进分母）；H1 那根只算进第一波，不重复计入回踩（H1 通常是放量的，重复计入会把回踩均量抬高、把合格的回踩误判成"没缩量"）。缩量分在 `[0.40, vol_shrink_ratio]` 区间映射（0.40 及以下满分、刚好卡闸门上限得 0 分）——旧的 `taper(1−vol_ratio, 0.3, 0.0)` 把满分点定在 0.7，导致"刚过闸门"的 0.85 只能拿一半分，有效区分区间被压得极窄。
5. **突破确认（已形成）**：收盘 ≥ `H1×(1+breakout_buffer_pct)`（+0.5%），且同时满足 量 ≥ `VOL_MA20×breakout_vol_ratio`（1.2）、收盘 ≥ MA20、RSI ≥ `rsi_min`（50）、MA20 不再下行。任一不满足就继续往后找，不将就。
6. **失败突破**：突破后 `failed_breakout_bars`（3）个**交易日**收盘跌回 H1 下方 → 撤销该次突破、退回**构筑中**（形态本身还成立，只是这次突破没站住），不是丢弃。口径按 bar 序号（交易日）而非自然日——A股连休时两者差异很大；交易模块的失败突破同样按交易日算（`monitor._trading_days_since`），两层不再错位。
7. **追高控制**：从突破（信号）日收盘算起，现价涨幅 > `chase_reduce_pct`（5%）→ 综合评分 ×0.85；> `chase_abort_pct`（8%）→ 直接丢弃（等回踩再说）。
8. **状态**：**已形成** = 突破确认（当天突破也直接算已形成）；**构筑中** = 回踩完成、结构完好、尚未突破 H1，是观察池的主要对象。**两头都有时效窗口**：构筑中 L1 距今 > `forming_max_age`（40）根仍未突破 → 丢弃（没有这道闸门，几个月前完成破底翻、之后一直在回踩区磨蹭的票会永久占据观察池，靠 `max_pool` 按综合分截断挤掉新鲜机会）；已形成突破前高距今 ≥ `breakout_within`（10）根 → 丢弃（以前只挡构筑中，于是"突破后 3 个交易日内没跌回 H1 下方"就永远算已形成——实测 300508 维宏股份 06-15 站上 H1，到 09-18 已回落 -25%、止损还停在现价上方，报表里照样是"已形成"）。
- **评分**：几何（破底越浅 / 第一波反弹越充分 / 回撤越适中 / HL 抬得越高）、量能（回踩缩量程度）、趋势（MA20 上行、RSI、MACD 柱回升、站上 MA20），优先级 S；构筑中同样在阶段确定性里打折。
- **关键价位**：`收复位`=L0、`第一波高点`=H1、`回踩低点`=L1；目标 `等幅目标 = H1 + (H1 − L1)`，止损 `L1 − stop_atr_buffer×ATR`（0.5）。止损压在 L1 下方而不是 L0 下方：这笔交易的结构被破坏，以**回踩低点失守**为准。注意 L1 距突破价很近，止损空间天然偏小，RR 容易被 `MAX_RR`/`MIN_RISK_PCT` 兜底修饰，实盘需考虑跳空与滑点。

### 破底翻v2 `break_bottom_v2`（做多 · 优先级 A）——已停用

按用户给的 **ATR 严格定义**写的第二套破底翻，**已停用**（`enabled: false`），与旧 `break_bottom` 并行但扫描和监控均不加载。代码保留供参考。差别在于确认锚点从"放量收盘站上颈线"换成"收复前低之后的 ATR 确认"：

1. **前置关键低点 L0**：最近 `l0_lookback`（60）根内**已确认**的 ZigZag 谷；`l0_must_be_leg_low` 要求它前面 `prior_lookback`（60）根内没有更低的原始最低价（否则只是下跌半途的中间低点，"破底"没破到真支撑）；并且 L0 之前跌过 `prior_drop`（15%）。
2. **有效破底**：存在 `low < L0 − break_low_atr_k×ATR`（0.3）**或** `close < L0 − break_close_atr_k×ATR`（0.2）。与旧口径不同，**只盘中插针破也算**。
3. **破底最低 L1 与深度**：`L1 = min(low[L0+1:])`，`(L0−L1)/L0 ∈ [min_break_depth, max_break_depth]`（[1.5%, 10%]）。
4. **快速收复**：自 **L1** 起 `recover_bars`（5）根内出现 `close > L0`。
   计时起点取破底最低点而不是首个触发 ATR 阈值的根——缓跌型二次探底里阈值在破底段很早就被穿过，从首破根起算会把正常结构判死（实测样本：首破@99、最低@104、收复@106，自首破起 7 根 > 5 闸门，自最低起 2 根）。
5. **跌破失败确认**：收复之后 `fail_confirm_bars`（3）根内每根的**收盘价**都要满足 `close > L0 − fail_confirm_atr_k×ATR`（0.3），否则算"假收复"直接失效。判据取收盘价而非最低价——只盘中插针破不算（实测 300353 东土科技 2026-09-23：盘中最低 16.35 破了阈值 16.42，但当天收盘 16.89 是长阳，按最低价判会把这条基准票直接判死）。
6. **B1 确认（买点）**：确认窗口之后首根满足 `close > L0 + b1_close_atr_k×ATR`（0.3）**或** 连续 `b1_hold_days`（2）日 `close > L0`。B1 当日收盘价即 `Entry`。**不要求放量、不要求颈线**——这是与旧口径最本质的区别。
7. **止损**：`max(L0 − stop_atr_k×ATR, L1 − extreme_stop_atr_k×ATR)`；`Entry ≤ 止损` 视为无效。
8. **目标**：`H1` = B1 之后首个已确认摆动高点满足 `H1 ≥ L0×(1+h1_min_pct)`（5%）或 `H1 − L0 ≥ h1_min_atr_k×ATR`（1.5），`tp1_price = H1`（未形成则 0）。`tp2_price` 取候选池（确认摆动高点 ∪ 60 日高 ∪ 120 日高 ∪ MA120 ∪ 成交密集区中心）里**严格高于 H1 的最近一档**。
9. **阶段机**：`BROKEN`（已破底未收复）→ `RECLAIMED`（已收复待确认）→ `B1_CONFIRMED`。`B1_CONFIRMED` = 已形成，前两阶段 = 构筑中。

- **定义外新增的两道闸门**（有实测理由，可配可关）：
  - `max_break_depth`（10%）：定义里没提深度上限，但"真破位 30% 再快速反弹"不该叫破底翻。
  - **确认后不得跌破止损**：定义只约束"收复后 3 日内不得再有效跌破"，但不拦住的话，B1 确认之后砸穿止损的票仍会报"已形成"（实测曾经的 `bad_strict_close_back_below_l0`：B1=10.65、止损 9.93，最后两根收 9.9/9.8 照样"已形成"）。
- **失效行会输出**：`status="invalid"` + `失效` 列 = "是" + `失效原因`，但只保留 `invalid_within`（20）根内发生的失效（否则报表会被历史失败塞满），且被 `report.long_only` / `watch._long_only` 挡在做多榜单和观察池之外。三个原因：`破底后N日内未收复L0` / `收复后再次有效跌破L0` / `B1确认后跌破止损位`。
- **每只票最多 1 行**（出图文件名是 `代码_形态key.png`，同 (代码, 形态) 出两行会撞文件名）。多候选时按 `未失效 → 阶段靠后 → 信号日新` 取一条。
- **关键价位**：`L0/L1/B1/H1/B2/PL`、`stop_price`、`tp1_price`、`tp2_price`、`risk_pct/reward1_pct/reward2_pct/rr1/rr2`、`trailing_stop`，外加 `L0_date/L1_date/B1_date/H1_date/B2_date` 日期列与 `阶段/失效/失效原因` 三列。**不写 `参考止损`**（避免与 `stop_price` 重复列），`base._get_stop_loss()` 已支持回退到 `stop_price`。
- **注意 `rr1` 的口径**：它按 **B1 入场价**计算。信号发出后价格往往已高于 B1，`rr1` 会明显小于报表里那个按**现价**算的 `风险收益比` 列。交易侧的开仓闸门 `trade.min_rr1` 用**现价**重算 RR1，与 `风险收益比` 口径一致。

### 破底翻v2第二买点 `break_bottom_v2_b2`（做多 · 优先级 S）——已停用

v2 B1 确认之后的第二次启动买点，**已停用**，与 B1 **独立计算**（结构定位复用 `break_bottom_v2._find_v2_bases`，前提另算）：

1. 前提：v2 B1 已 `B1_CONFIRMED` 且 H1 已形成，B1 距今 `min_bars_after_b1`（8）~ `break_within_v2`（100）根。L0 回看窗口相应放宽到 `break_within_v2 + l0_lookback`。
2. **回踩低点 pl**：H1 之后首个已确认 ZigZag 谷，且 `low > L0 − pullback_l0_atr_k×ATR`（0.3，回踩不得有效跌破 L0）、幅度 ≥ `max(pullback_min_pct, pullback_min_atr_k×ATR/H1)`。跌破 L0 的回踩**直接不作数**（不是"失效"，是"这里没有 B2"）。
3. **缩量回踩**：回踩段均量 ≤ 第一波均量 × `pullback_max_vol_ratio`（0.85）。第一波 = `[B1, H1]`，回踩段 = `(H1, pl]`。
4. **再启动（已形成）**：pl 之后首根 `close > H1×(1+breakout_buffer_pct)`（+0.5%）且量 ≥ `vol_ma5×breakout_vol_ratio`（1.2）。
5. **止损**：`max(L0 − 0.3×ATR, L1 − 0.5×ATR, pl − stop_swing_atr_k×ATR)`；**TP1** = `H1 + (H1 − L1)`（等幅量度）；**TP2** = v2 的 TP2 候选池里高于 TP1 的最近一档。
6. **时效**：构筑中 pl 距今 > `forming_max_age`（40）→ 丢弃；已形成突破距今 ≥ `breakout_within`（10）→ 丢弃；突破后涨幅 > `chase_abort_pct`（8%）→ 不追。
7. 失效只报一种：**B2 确认之后收盘跌穿止损**。

### W底 `w_bottom`（做多 · 优先级 A）

双底：`L1 → H(颈线) → L2`，突破颈线确认。从 ZigZag 序列里取所有 `(L,H,L)` 三元组，逐个检验，多组命中取评分最高。

- **几何约束**：`neck > base=min(L1,L2)`；双底价差 `|L2−L1|/L1 ≤ bottom_tol`（0.05）；颈线高度 `(neck−base)/base ∈ [min_neck_height, max_neck_height]`（[0.08, 0.60]）；宽度 `L2.idx−L1.idx ∈ [min_width, max_width]`（[15, 150]）；左右时间比 `右段/左段 ∈ [time_ratio_min, time_ratio_max]`（[0.3, 3.0]）。
- **前置背景**：L1 之前 `prior_lookback`（60）根内跌幅 ≥ `prior_drop`（0.15）——没有前期下跌就不构成"底"。
- **状态**：突破触发价 `neck×(1+breakout_buffer)`（+1%）。从右底后扫描收盘价**连续**站上触发价的起点为 `breakout_idx`（中途跌回会重置，之后重新站上算新信号）。
  - **已形成**：存在 `breakout_idx` 且距今 < `breakout_within`（10，太久=形态走完）。
  - **构筑中**：尚未突破，但右底距今 ≤ `forming_max_age`（30），自右底反弹 ≥ 1%，且（临近颈线 within `forming_near_neck`=5% **或** 出现转折 close>MA5 或 >昨收）。
- **几何分**：双底价差 / 抬高的右底更强 / 颈线高度 / 时间对称 / 前期跌幅 / 颈线触碰次数。
- **量能分**：右底相对左底缩量（0.45）+ 突破放量（0.55）。已形成用突破日量比 VOL_MA20，构筑中用近 5 日均量。
- **趋势分**：站上 MA20 / MA20 上行 / 底背离（右底 ≤ 左底且 MACD 柱抬高）/ DIF>DEA。
- **关键价位**：目标 `等幅目标 = neck×(1+颈线高度)`，止损 `base×0.97`。

### M头 `m_top`（做空 · 优先级 A）

双顶：`H1 → L(颈线) → H2`，跌破颈线确认。**W底的完全镜像**——但右头缩量在这里是**看跌**确认（追高动能衰减），与 W 底右底缩量含义相反。

- 取 `(H,L,H)` 三元组。`top=max(H1,H2) > neck`；双顶价差 ≤ `top_tol`（0.05）；颈线深度 `(top−neck)/top ∈ [min_neck_depth, max_neck_depth]`（[0.08, 0.60]）；宽度、时间比同 W 底。
- **前置背景**：H1 之前涨幅 ≥ `prior_rise`（0.15）。
- **状态**：破位触发价 `neck×(1−breakdown_buffer)`。收盘连续跌破的起点为 `breakdown_idx`（站回颈线则重置）。已形成 = 距今 < `breakdown_within`（10）；构筑中 = 右头距今 ≤ `forming_max_age`（30）、自右头回落 ≥ 1%、且（临近颈线或转折 close<MA5 或 <昨收）。
- **量能分**：右头缩量（**0.6**，权重高于 W 底）+ 破位放量（0.4）。
- **趋势分**（看跌）：跌破 MA20 / MA20 下行 / 顶背离（右头 ≥ 左头且 MACD 柱走低）/ DIF<DEA。
- **关键价位**：目标 `等幅目标 = neck×(1−颈线深度)`，止损 `neck×1.03`（跌破后站回颈线即证伪）。做空形态的**预期收益按跌幅计**，正数表示形态兑现。

### 通道 `channel`（上升 / 下降 / 横盘箱体 / 下降通道反转）

对收盘价做**最小二乘回归**定出中轴，上下轨用**残差分位数**（`resid_q_upper=0.95` / `resid_q_lower=0.05`）平移得到（用分位数而非极值，抗单根插针）。多窗口 `[60, 90, 120]` 各拟合一次，取几何质量分最高的那个。

- **区分真通道与随机游走的关键，是上下轨各自的触碰次数**——随机游走也能回归出一条"通道"，但不会反复在两条轨上受阻和获支撑。
- **通道宽度** `(上轨偏移−下轨偏移)/中轴均值 ∈ [min_width, max_width]`（[0.05, 0.35]）。
- **容纳率** containment：整根 K 线（high ≤ 上轨、low ≥ 下轨，各留 `touch_tol`=1.5% 容差）落在通道内的占比。用 high/low 而非 close，否则分位数定轨会让容纳率恒等于 90% 成为废指标。
- **类型判定**：窗口内累计涨跌幅 > `trend_threshold`（+8%）为**上升通道**，< −8% 为**下降通道**，之间为**横盘箱体**。
- **拟合质量**：趋势通道要求 `R² ≥ min_r2`（0.35）；箱体的 R² 天然接近 0，改用"平坦度"衡量。
- **状态**：已形成 = 上下轨触碰各 ≥ `min_touches`（2）且容纳率 ≥ `min_containment`（0.90）；构筑中 = 触碰放宽（一侧 ≥2、另一侧 ≥1）且容纳率 ≥ 0.85。

**下降通道反转（`falling_channel_reversal`，做多 · 优先级 A）**：当最优拟合是下降通道、价格贴近下轨（`pos < 0.15`）、且最近 1~2 根出现止跌信号（收阳或长下影 > 60%）时，把它从"下降通道（做空）"降级标记为反转做多机会，目标 `上轨`、止损 `下轨×0.97`。这样图形识别不与交易方向耦合：没有止跌信号时，它仍然是一条普通的做空下降通道。

- **量价配合**：上升通道要求上涨放量，下降通道要求下跌放量，箱体越均衡越好。
- **趋势分**：上升通道看 close>MA20、MA20>MA60、通道位置高；下降通道镜像（保持看跌，避免被误排进做多榜）；箱体看均线平坦 + 位置。
- **关键价位**：`上轨 / 中轴 / 下轨 / 通道位置 / 通道宽度% / R² / 容纳率 / 上下轨触碰次数`。做多（升 / 箱 / 反转）破下轨证伪→止损 `下轨×0.97`；做空（降）站上上轨证伪→止损 `上轨×1.03`。

### 整理突破 `breakout`（平台突破 / 突破回踩确认 / 杯柄突破）

三个子形态共用一套整理区间逻辑，全部做多，且**下降趋势直接跳过**。

**平台突破 `platform_breakout`（优先级 S）**
- 取最近 `consolidation_window`（25）根（不含当日）为整理区间。区间振幅 `(hi−lo)/mid ≤ max_range`（0.18）、整理段斜率够平 `|slope|/mid ≤ platform_slope`（0.012）。
- 整理开始前须有一段涨幅：相对 `prior_lookback`（60）根前 ≥ `platform_trend`（0.12）。
- **已形成**：收盘 > `hi×(1+breakout_buffer)`（+1%）突破上沿；**构筑中**：收盘 ≥ `hi×(1−forming_near)`（接近突破位，−3% 以内）。

**突破回踩确认 `breakout_retest`（优先级 S）**
- 在平台突破的基础上，要求整理中枢略微上倾（`slope > 0`），并识别**回踩**：区间内低点触及 `lo×(1+pullback_touch_tol)`（1.5%），此后收盘又收回到区间中点以上（`pullback_recovered`）。
- **已形成**：突破上沿 **且** 有过回踩 **且** 回踩已收回；**构筑中**：接近突破位，或已突破但尚无回踩（"突破待确认"）。回踩确认给几何分 +0.15 加成。

**杯柄 / 圆弧底突破 `cup_handle`（优先级 B）**
- 取最近 `cup_window`（80）根收盘，杯底须落在窗口中段（0.2w~0.8w）。杯深 `(max(左沿,右沿)−底)/max(左沿,右沿) ∈ [cup_min_depth, cup_max_depth]`（[0.12, 0.45]），右沿不显著低于左沿（`≥ 左沿×(1−cup_rim_tol)`，8%）。
- 突破杯沿 `rim=max(左沿,右沿)`：> `rim×1.01` 已形成，≥ `rim×0.97` 构筑中。二次拟合 R² 高且对称 → 标"圆弧底"，否则"杯柄"。

三者**关键价位**统一：目标 `等幅目标 = 突破位 + (突破位 − 整理下沿)`，止损 `整理下沿×0.97`。量能看突破日量比 VOL_MA20，趋势看 close>MA20>MA60。

### 趋势回踩再启动 `trend_pullback`（做多 · 优先级 S）

上升趋势中，价格回踩关键支撑后企稳反弹——顺势的低吸机会。下降趋势直接跳过。

1. **确认上升趋势**：MA20 > MA60 > 0。
2. **近期有涨**：`lookback`（60）窗口内 `(近期高点 − 起点收盘)/起点收盘 ≥ min_rise`（0.10）。且**高点必须够新**——距今 ≤ `high_max_age`（15）根，超期算掉队的老高点直接丢弃（否则一只早就见顶、缓慢阴跌的票会长期赖在 3~20% 的"回撤"带里反复命中）；高点也不能就是最新一根，那还没开始回踩。
3. **回撤适中**：`(近期高点 − 高点之后最低)/近期高点 ∈ [pullback_min, pullback_max]`（[0.03, 0.20]）。回撤从**高点之后**的最低点量起（而非笼统的近5日最低），才是这一波的真实回撤。太浅没意义，太深趋势可能已坏。
4. **触及支撑**（近 5 根内最低价贴近下列任一即可，按各自容忍度）：MA20（1.5%）、MA60（2%）、前一个 ZigZag 谷 SwingLow（2%）、突破转支撑（近 lookback 内前高、排除最近 5 根，2%）。
5. **止跌信号**：最近 1~2 根收阳**且收在昨收之上**，或长下影（下影 > 实体 1.5 倍且 > 全幅 40%）。单纯一根小阳线不算——上升趋势里随手就能收阳，必须同时站上昨收才算真正止跌。
6. **状态**：已形成 = 今日 close>MA20 且昨日 close<MA20（重新站回 MA20）；构筑中 = 现价已收在支撑上方（`≥ 回踩支撑×(1−2%)`）。**若现价仍明显低于支撑（还在下跌、未企稳）则直接丢弃**，不再无脑塞进"构筑中"——这一条是控制该形态误报量的关键闸门。
- **关键价位**：目标 `近期高点`（回踩的第一目标是前高，而非"前高再叠一个满幅"——后者会把预期收益放大一倍多，让该形态在收益质量上恒定顶满），止损 `回踩支撑×(1−stop_buffer)`（−3%）。`key_levels` 里也给出 `等幅目标` 作参考。

---

## 评分体系

采用多维度综合评分，所有结果按**综合评分**降序排列（0–100）：

| 维度 | 权重 | 说明 |
|------|:---:|------|
| 机会质量 | 40% | 形态本身质量分（几何+量能+趋势加权，再乘形态优先级权重） |
| 收益质量 | 25% | 预期收益率(60%) + 风险收益比(40%) |
| 阶段确定性 | 20% | 已形成=100，构筑中=100×`forming_discount`，已失效=0 |
| 风险修正 | 15% | RSI(30) + MACD 动量(30) + 距目标空间(20)，按实际可得权重归一到 20~100 |

**机会质量子分**内部：几何 / 量能 / 趋势按 `scoring` 的 `geom_weight:vol_weight:trend_weight`（默认 **50:25:25**）加权到 0~100，再乘**形态优先级权重**（S/A/B/C → 1.0/0.95/0.88/0.80）。

几点口径：

- **构筑中只打一次折**。折扣只在"阶段确定性"里体现（`forming_discount` 默认 0.85），`blend()` 不再重复乘。
- **风险收益比上限 6.0**，且风险至少按现价 0.5% 计——现价贴着止损时真实风险趋于 0，不兜住会让风险收益比冲到几十，直接污染排序。
- **止损已经落到现价的错误一侧时（做多止损 ≥ 现价、做空止损 ≤ 现价），风险收益比与收益质量都记 0**。这种形态的风险结构已经不成立，没有可度量的风险；以前用 `abs()` 取距离会把"止损在现价上方"当成一个极小的正风险，RR 被兜到 6.0 上限、收益质量拿满分，反而让结构已破的形态排到榜单最前面。
- **预期收益对做空形态按跌幅计**，正数一律表示形态兑现。
- **风险修正按"实际可得权重"归一**：拿不到的分项（如某检测器没提供目标空间）不进分母，否则等于给所有形态封了个分数上限。
- **优先级反映确认层次而非命中数量**：`break_bottom_b2` 因多出回踩不破 / HL / 放量破前高三道硬闸门，记 S；`break_bottom`（B1）是左侧早期买点，记 A。

**报表字段：**

| 字段 | 说明 |
|------|------|
| 方向 | 做多 / 做空 / 中性 |
| 最新价 | 信号日收盘价 |
| 综合评分 | 多维度加权总分（0-100），已含市场环境乘性修正 |
| 机会质量 / 收益质量 / 阶段确定性 / 风险修正 | 四个子分 |
| 预期收益% | 基于形态目标价计算（做空按跌幅） |
| 风险收益比 | 目标空间 / 止损空间（上限 6.0） |
| 市场环境 | 当日 MarketRegime 分数（0-100） |
| 市场情绪 | 当日 MarketSentiment 分数（0-100） |
| 所属板块 | 申万行业分类名称 |
| 板块热度 | 所属行业 SectorHeat 分数（0-100） |
| 板块阶段 | 启动 / 扩散 / 主升 / 高潮 / 分歧 / 退潮 |
| 信号日期 / 说明 | 命中日期与逐形态明细 |

此外每个形态会把自己的 `key_levels`（颈线、上下轨、参考止损、等幅目标……）展开成额外列。

### 市场环境修正

扫描前自动计算一次 **MarketSnapshot**，作为乘性因子修正个股综合评分，并在极端弱势时阻止做多。三个子模块：

**MarketRegime（大盘环境 0~100）**：取沪深 300 / 中证 1000 / 创业板指三只指数的日 K 线，每只评估位置（close vs MA20/MA60）、斜率（MA20 方向）、趋势（20d/60d 涨跌幅），加权聚合后按阈值分档：

| 分数区间 | 标签 | 对评分的影响 |
|---------|------|------------|
| 80~100 | 强势 | market_factor ≈ 0.8~1.0 |
| 65~80 | 偏强 | market_factor ≈ 0.65~0.8 |
| 45~65 | 震荡 | market_factor ≈ 0.45~0.65 |
| 25~45 | 偏弱 | market_factor ≈ 0.25~0.45 |
| 0~25 | 弱势 | **阻止做多**（regime < 25 的信号标记 blocked） |

**MarketSentiment（市场情绪 0~100）**：从 `daily_summary.csv`（全市场当日聚合）计算 6 个子指标：

| 子指标 | 权重 | 含义 |
|--------|:---:|------|
| 上涨家数占比 | 20% | 涨/总，映射 0.3→0 / 0.5→50 / 0.7→100 |
| 涨跌幅中位数 | 20% | -2%→0 / 0%→50 / +2%→100 |
| 涨停/跌停数 | 15% | 净涨停数 -30→0 / 0→50 / +30→100 |
| 炸板率 | 15% | 盘中触涨停但未封住的比例，0%→100 / 50%→50 / 100%→0 |
| 昨日涨停跟进 | 15% | 昨日涨停股今日平均涨幅 |
| 连板高度 | 15% | 最高连板天数 |

情绪分数通过 `sentiment_factor = 0.5 + score/200`（范围 0.5~1.0）乘性修正评分。

**SectorHeat（行业热度 0~100 + 阶段）**：基于申万行业分类，每个行业计算 6 个子指标（相对涨幅、量能变化、上涨占比、趋势强度、资金强度、持续性），加权得出热度分数，并判定行业所处阶段：

| 阶段 | 含义 | 对评分的影响 |
|------|------|------------|
| 启动 | 热度从低位回升，相对强度刚转正 | ×1.05 加成 |
| 扩散 | 热度 50~70，上涨占比 >60% | ×1.05 加成 |
| 主升 | 热度 >70，量能放大，加速 | ×1.05 加成 |
| 高潮 | 热度 >80 但量能从峰值回落 | ×0.95 降级 |
| 分歧 | 热度从 >70 回落 | 无修正 |
| 退潮 | 热度 <40 且下降 | 无修正，B1/B2 信号额外降级 |

**评分修正公式**：

```
AdjustedScore = RawScore × MarketFactor × SentimentFactor × SectorFactor × StageBonus
```

- `MarketFactor = regime_score / 100`
- `SentimentFactor = 0.5 + sentiment_score / 200`
- `SectorFactor = 0.5 + sector_score / 200`（无行业归属默认 50 分）
- `StageBonus` = 1.05（启动/扩散/主升）/ 0.95（高潮）/ 1.0（其他）

**过滤规则**：
- `regime_score < 25`：阻止做多方向的信号（标记 blocked）
- `sector_score < 30`：额外 ×0.8 降级
- 行业处于"退潮"阶段：B1/B2 形态信号降级

**向后兼容**：无指数数据时 `compute_market_snapshot()` 返回 None，评分不做环境修正，与未启用时行为一致。

## 安装

```bash
pip3 install -r requirements.txt
```

依赖：pandas, pyyaml, tqdm, openpyxl, mplfinance, baostock

## 快速开始

```bash
# 1. 首次全量拉取K线数据（约5000只股票，需10-30分钟）
python3 cli.py update --full

# 2. 全市场扫描
python3 cli.py scan

# 3. 同步观察池（从扫描报表中取"构筑中"的形态）
python3 cli.py watch sync

# 4. 实时监控（轮询报价，站稳成型即提醒/自动模拟买入）
python3 cli.py watch run

# 5. 单股调试（带解释和图表）
python3 cli.py scan --code 000001 --explain --plot

# 6. 查看本地库状态
python3 cli.py info
```

`--explain` 会打印每个检测器逐条的拒绝原因（"为什么没命中"），是调参时唯一能看清判定过程的途径。

## CLI 命令

### update - 抓取/更新K线

```bash
python3 cli.py update --full          # 全量重拉
python3 cli.py update                 # 增量更新（每日收盘后）
python3 cli.py update --limit 100     # 只处理前100只（测试用）
python3 cli.py update --codes 000001,600519  # 指定股票
python3 cli.py update --retry-failed  # 重跑上次失败的
```

**智能更新逻辑：**
- 收盘后（15:00 后）：本地已有当日数据的股票自动跳过
- 盘中（15:00 前）：即使有当日数据也会拉取更新，获取最新盘中数据
- 增量时回看 `drift_check_bars` 根做**复权漂移检测**，重叠日收盘价偏差超 `drift_tolerance` 判定发生除权，自动全量重拉
- 个股更新完成后自动增量更新 3 只指数 K 线（沪深 300 / 中证 1000 / 创业板指），供市场环境评估使用
- 写入 `data/market/daily_summary.csv`（全市场当日聚合，保留 60 个交易日）和涨停跟踪文件（`limit_up_history.json` / `limit_up_streak.json`），供情绪和行业热度计算使用

### scan - 扫描形态

```bash
python3 cli.py scan                           # 全市场扫描，只出报表（出图默认关）
python3 cli.py scan --patterns w_bottom,m_top # 只扫指定形态（用 key，也接受 rising_channel 等产出名）
python3 cli.py scan --status formed           # 只要已形成的
python3 cli.py scan --min-score 60            # 最低综合评分门槛
python3 cli.py scan --codes 000001,600519     # 只扫指定代码
python3 cli.py scan --workers 4               # 进程数，默认CPU核数
python3 cli.py scan --top 30                  # 终端显示前30条
python3 cli.py scan --code 000001 --explain   # 单股调试，打印判定过程
python3 cli.py scan --plot                    # 本次出图
python3 cli.py scan --no-plot                 # 本次不出图（charts.enabled=true 时临时关掉）
python3 cli.py scan --plot-top 20 --plot-long-only  # 出图上限20张，且只给做多形态出图
python3 cli.py scan --stamp 20260901          # 报表文件名日期戳
```

扫描根据 `universe` 配置自动过滤：ST、退市、科创板、北交所、流通市值、成交额、股价上限、历史K线根数等。

**出图开关（可选功能，默认不出图）**：全市场出图要逐只读K线写 png，又慢又占磁盘，所以 `charts.enabled` 默认 `false`，日常 `scan` 只出 CSV/Excel。需要看图时加 `--plot`（`cron_update.sh` 就是这么跑的），或把 `charts.enabled` 设为 `true` 变成长期默认，再用 `--no-plot` 临时关掉。`--plot-top` 覆盖 `charts.max_charts`，`--plot-long-only` 只给做多形态出图（默认给全部命中出图）。图按报表同一个 `stamp` 落到 `out/charts/<stamp>/<形态key>/`，默认**不限张数**；`<stamp>/index.html` 是索引页，按形态分组折叠。

### info - 查看状态

```bash
python3 cli.py info
```

输出示例：
```
存储格式    csv
存储位置    /path/to/data/store
股票数      4523
K线行数     3412567
日期范围    2022-01-04 ~ 2026-09-01
最后全量    2026-09-01
最后增量    2026-09-01
```

### watch - 观察池 + 盘中监控

观察池用于跟踪 scan 报表中"构筑中"的形态，盘中自动轮询监控，站稳成型即提醒并（若开启）自动模拟买入。

```bash
# 1. 从最新 scan 报表同步观察池（默认取构筑中）
python3 cli.py watch sync

# 2. 查看观察池
python3 cli.py watch list

# 3. 手工添加/移除
python3 cli.py watch add --codes 000001,600000 --pattern break_bottom
python3 cli.py watch remove --codes 000001

# 4. 启动实时监控（持续轮询，Ctrl-C 停止）
python3 cli.py watch run

# 5. 只跑一轮（调试用）
python3 cli.py watch run --once --detail

# 6. 已提醒过的票每轮重复提醒
python3 cli.py watch run --repeat
```

**监控逻辑：**
- 每轮取实时报价，拼成"今日K线"接到历史数据尾部
- **原样重跑形态检测器，口径与收盘 scan 完全一致**
- 连续 `confirm_rounds`（默认 2）轮判定"已形成"才触发提醒/买入，滤掉分时插针
- **断档补漏**：若上一轮检测已经是 3×轮询间隔之前（停跑重启、盘中断了几轮）或跨了交易日，此时发现的"已形成"多半是**断档期间形成的**——不再等 `confirm_rounds`，本轮直接确认提醒/买入。区分"真漏了"和"上一轮刚看过"：从没实时检测过、且是报表/手工带进池的历史形成算漏；盘中实时扫描刚冒出来的不算，照常多轮确认
- 形态失效（破底深度超限、连续跌破 L0、关键点超期等）自动移出池子
- 按 `pool_refresh_interval`（默认 15 分钟）结合全市场最新价重扫并合并新候选
- 提醒写入 `out/alerts_YYYYMMDD.csv`
- 决策分级：PROMOTED（连续确认，触发买入）/ FORMED / FORMING / INVALID / NOQUOTE

**观察池文件：** `data/watch/pool.json`，可手工编辑加入自选股。只收做多形态：`watch.patterns` 里配了 M头 / 下降通道也会被跳过（日志说明跳了几条）。

### trade - 模拟交易

监控确认成型后自动买入，按止损 / 移动止盈规则自动卖出。支持接入东财 EMT 极速柜台仿真交易。

```bash
python3 cli.py trade status              # 查看账户和持仓
python3 cli.py trade history             # 查看历史成交
python3 cli.py trade history --limit 50
python3 cli.py trade reset --yes         # 重置账户（清空持仓和历史，恢复初始资金）
```

> 没有独立的 `trade run`——自动买卖在 `watch run` 的监控循环里进行。

**交易逻辑：**
- **入场**：监控确认成型（PROMOTED）时自动买入，但要过三道否决闸门：
  - **上涨空间不足**：`(目标价−现价)/现价 < min_target_upside`（8%）→ 不买。贴着目标价开仓，赚不到钱却先承担全部风险。
  - **偏离形成点位**：`(现价−止损)/止损 > max_entry_drift`（10%）→ 不买。形态几天前就形成了、价格已经跑远，此时进场等于接最后一棒。
  - **RR1 不足（仅 v2 口径）**：`trade.min_rr1`（1.5）。只对带 `tp1_price` 的信号生效，且用**现价**重算 `(TP1−现价)/(现价−止损)`——信号发出后价格会漂移，拿检测器记录的 B1 时刻 RR1 做否决会把已经变差的入场放进来。旧口径没有 TP1，闸门自动跳过。
- **仓位计算**：固定风险法，股数 = (现金 × `risk_per_trade`) ÷ (入场价 − 止损价)，向下取 100 股整数倍；现金不足时按可用现金上限回退
- **止损**：旧口径取"参考止损"，v2 口径取 `stop_price`
- **移动止盈**：
  - **v2 口径持仓**：止损直接跟随结构位 `max(MA20 − trail_ma_atr_k×ATR, 最近SwingLow − trail_swing_atr_k×ATR)`（只上移不下移），**替代**下面的百分比回撤规则——形态的失效位是结构给定的，与"从最高价回撤 trail_pct"无关
  - 涨 ≥ `breakeven_pct`（3%）→ 止损上移到成本价（保本）
  - 涨 ≥ `trail_trigger_pct`（5%）→ 启动追踪止损，回撤 `trail_pct`（2%）平仓
  - **分批止盈之后**（`trail_use_ma_hl`）→ 不再按回撤百分比追踪，改为把止损抬到 `max(MA20, 当前止损)` 且只上移不下移，让剩余仓位跟着趋势走（"最近有效 HL"目前取的是当前止损价，见[注意事项](#注意事项)）
- **分批止盈**：现价 ≥ 第一目标且尚未分批过 → 先卖出 `partial_close_pct`（50%）的仓位（向下取整到 100 股；不足一手则整笔平仓）；剩余仓位继续按上面的规则管理。
  - v2 口径的第一目标是 `tp1_price`（H1）；旧口径仍按 `目标价`。
  - v2 口径开仓时 `target_price` 存的是 **TP2**（没有 TP2 就存 0，剩余仓位交给移动止盈）——把 TP1 同时当固定目标价会让剩余仓位在 TP1 被一次性清光，分批失去意义。
- **退出条件**：
  - 触止损（分"止损"/"止盈"——止损已抬到成本上方时记为"止盈"）
  - 触目标价（未分批过 → "分批止盈"；已分批过 → "目标价"全平）
  - **失败突破**：B2 持仓在突破后 `trade.failed_breakout_bars`（3）个**交易日**内，实时价跌回 H1 下方 → 立即离场，不等止损（信号已被证伪）。这是**交易层的退出策略**，与检测器 `break_bottom_b2.failed_breakout_bars`（信号过滤）各自独立、互不牵动。突破日取突破 bar 的日期（开仓时存 `signal_date`，与检测器同一根 bar），**不是开仓日**——否则开仓越晚、窗口越宽；交易日历取本地 K 线的日期序列（与检测器 bar 序号口径一致），盘中把"今天"补上；v2 的 B2 用 `H1` 作为突破价位。
  - 当日买入的仓位受 **T+1** 约束：即使盘中触发止损/目标价，当日也不会卖出（形态失效的池子剔除只影响是否继续跟踪，不强制平仓）

**数据存储：** `data/trade/`
- `account.json` - 账户状态（现金 / 冻结 / 累计盈亏 / 交易次数）
- `positions.json` - 当前持仓
- `history.json` - 历史成交

**接入东财 EMT 极速柜台仿真：**

默认使用本地模拟（不连接券商，订单状态标记为 `local`）。如需接入真实仿真交易：

1. 获取东财 EMT 测试账号（联系东方财富机构业务）
2. 将 `emt_api_python/` SDK 目录放在项目根目录下
3. 在 `config.yaml` 的 `trade.emt` 下把 `enabled` 改为 `true`，填入交易地址 / 端口 / 本机IP；资金账号和密码**不要写在 config.yaml 里**，放到 `.env`（见下文「敏感配置」）

> ⚠️ **不要把真实资金账号和密码提交进 git**。凭证明文只放 `.env`（已 gitignore），仓库里 `config.yaml` 只留空占位。

```bash
cp .env.example .env       # 然后填入真实值
# .env
# TRADEYH_EMT_ACCOUNT_ID=你的资金账号
# TRADEYH_EMT_PASSWORD=你的资金密码
```

启用后，买卖订单通过 EMT API 发送到仿真环境，订单号记录在持仓和交易历史中。

## 配置

编辑 `config.yaml` 调整参数（缺失的键回落到 `src/tradeyh/config.py` 的 DEFAULTS）：

```yaml
data:
  storage: csv              # csv / json / sqlite
  sources: [baostock, eastmoney, tencent, sina]  # 数据源优先级（HTTP 渠道按序回退）
  adjust: qfq               # qfq前复权 / hfq后复权 / ""不复权
  history_start: "20220101" # 全量拉取起始日
  max_workers: 4            # 并发数，限流时降到 2-3
  drift_check_bars: 30      # 增量回看根数（复权漂移检测）
  drift_tolerance: 0.002    # 重叠日收盘偏差超 0.2% 判定除权

universe:
  exclude_st: true          # 剔除 ST/*ST
  exclude_delisted: true    # 剔除名称含"退"
  exclude_star_market: true # 剔除科创板 688
  exclude_bj: true          # 剔除北交所/沪B
  min_bars: 250             # 历史K线不足则跳过（滤次新股）
  min_float_mv: 0           # 最小流通市值（元），0=不限
  min_amount: 0             # 最小成交额（元），0=不限
  max_price: 0              # 股价上限，0=不限

swings:                     # ZigZag 摆动点
  pct_min: 0.05             # 最小反转幅度
  atr_k: 1.5                # 实际阈值 = max(pct_min, atr_k * ATR / price)
  fractal_window: 5

scoring:
  geom_weight: 50           # 几何 : 量能 : 趋势 权重
  vol_weight: 25
  trend_weight: 25
  forming_discount: 0.85    # 构筑中形态的阶段折扣
  min_score: 0              # 报表最低分门槛，CLI --min-score 可覆盖

charts:                     # 个股标注图（K线 + 关键点位 + 趋势线）
  enabled: false            # 默认不出图；需要出图加 --plot，或改 true 让"扫描即出图"成为默认
  dir: ""                   # 图片根目录，空 = data.out_dir/charts。实际写入 <根>/<YYYYMMDD>/<形态key>/
  bars: 180                 # 每张图显示的K线根数
  max_charts: 0             # 单次扫描最多出图数（按榜单顺序），0=不限（默认全出），CLI --plot-top 可覆盖
  dpi: 130
  include_ma20: true        # 叠加 MA20
  index_html: true          # 生成 index.html 索引页（写在当天的目录里）

market:                     # 市场环境 / 情绪 / 行业热度
  enabled: true             # 总开关，关闭后评分不做环境修正
  index_codes:              # 指数代码（idx_ 前缀，存入同一 store）
    csi300: idx_000300      # 沪深 300
    csi1000: idx_000905     # 中证 1000
    chinext: idx_399006     # 创业板指
  index_weights: {csi300: 0.4, csi1000: 0.3, chinext: 0.3}
  regime_thresholds: {strong: 80, bullish: 65, neutral: 45, weak: 25}
  sentiment_weights:        # 情绪 6 子指标权重
    advance_ratio: 0.20     # 上涨家数占比
    median_return: 0.20     # 涨跌幅中位数
    limit_count: 0.15       # 涨停/跌停数
    failed_limit_rate: 0.15 # 炸板率
    limit_follow_through: 0.15  # 昨日涨停跟进
    max_streak: 0.15        # 连板高度
  sector_weights:           # 行业热度 6 子指标权重
    relative_return: 0.20   # 相对大盘涨幅
    volume_ratio: 0.15      # 量能变化
    breadth: 0.15           # 上涨占比
    trend: 0.20             # 趋势强度
    capital_intensity: 0.15 # 资金强度
    outperform_days: 0.15   # 持续强于大盘天数
  sector_bonus: 1.05        # 启动/扩散/主升阶段的评分加成
  sector_penalty: 0.95      # 高潮阶段的评分降级
  block_long_regime_threshold: 25   # regime < 25 阻止做多
  downgrade_sector_threshold: 30    # sector < 30 额外降级

watch:                      # 观察池 + 盘中站稳提醒
  pool_path: data/watch/pool.json
  patterns: [break_bottom, breakdown_reversal, break_bottom_b2, platform_breakout, breakout_retest, trend_pullback]
  statuses: [forming]       # 只收"构筑中"——已形成的等不到盘中转折点
  min_score: 0
  max_pool: 50              # 池子上限，超出按综合评分截断
  interval: 60              # 轮询间隔（秒）
  pool_refresh_interval: 900 # 结合最新价重扫并更新观察池（秒）；0=关闭
  confirm_rounds: 2         # 连续N轮成型才提醒
  quote_sources: [eastmoney, tencent]
  bell: false               # 触发提醒时终端响铃

trade:                      # 模拟交易
  enabled: true
  initial_capital: 100000   # 初始资金
  risk_per_trade: 0.01      # 每笔风险比例（1%）
  max_positions: 10         # 最大持仓数
  breakeven_pct: 0.03       # 涨3%后止损移到成本
  trail_trigger_pct: 0.05   # 涨5%后启动追踪
  trail_pct: 0.02           # 追踪止损回撤2%
  min_target_upside: 0.08   # 距目标价不足8%则不买入（上涨空间太小）
  max_entry_drift: 0.10     # 现价距形成点位（止损）偏离超10%则不买入
  failed_breakout_bars: 3   # B2 突破后N个交易日内跌回H1下方即离场（交易层退出策略，独立于检测器同名键）
  partial_close_pct: 0.5    # 到达第一目标时卖出比例（分批止盈）
  min_rr1: 1.5              # RR1 不足则不买入（仅对带 tp1_price 的 v2 口径生效）
  trail_use_ma_hl: true     # 分批止盈后用 MA20 / 当前止损取高者作为移动止盈
  emt:                      # 东财 EMT 极速柜台仿真
    enabled: false          # 设为 true 启用仿真下单
    ip: ""                  # EMT 交易地址
    port: 0                 # EMT 端口
    account_id: ""          # 资金账号 —— 走 .env 的 TRADEYH_EMT_ACCOUNT_ID
    password: ""            # 资金密码 —— 走 .env 的 TRADEYH_EMT_PASSWORD
    local_ip: "127.0.0.1"   # 本机IP（柜台白名单校验用）

# 各形态的详细参数（w_bottom / m_top / break_bottom / break_bottom_b2 /
# break_bottom_v2 / break_bottom_v2_b2 / channel / breakout / trend_pullback 块）
# 见 config.yaml 与上文「形态识别逻辑」
```

### 敏感配置（.env / 环境变量）

`config.yaml` 纳入版本控制，**凭证不放里面**。启动时 `config.py` 会读项目根的 `.env`（gitignore 已忽略），再用环境变量覆盖配置：

```bash
cp .env.example .env       # 模板已提交，.env 不会进 git
```

```
TRADEYH_EMT_ACCOUNT_ID=你的资金账号   # → trade.emt.account_id
TRADEYH_EMT_PASSWORD=你的资金密码     # → trade.emt.password
```

优先级：**真实 shell 环境变量 > `.env` > `config.yaml`**。`.env` 只做 `setdefault`，所以 CI / 容器里直接 `export` 同样生效；行首 `#` 为注释，值可加引号也可不加（引号内的 `#` 不会被当注释）。新增敏感字段时在 `src/tradeyh/config.py` 的 `ENV_OVERRIDES` 里加一行映射即可。

## 输出

扫描结果输出到 `out/` 目录：

- `scan_YYYYMMDD.xlsx` - Excel报表（汇总 + 做多榜单 + 每形态一个 sheet，带色阶；含市场环境/情绪/板块热度列）
- `scan_YYYYMMDD.csv` - CSV报表
- `alerts_YYYYMMDD.csv` - 盘中监控提醒记录
- `charts/<YYYYMMDD>/<形态key>/` - 个股标注图，**先按日期、再按形态分组**。**默认不生成**（`charts.enabled: false`）：全市场 `scan` 加 `--plot`、单股 `--code X --plot` 才会出图；`--no-plot` 可关、`--plot-top` 限制张数（默认不限）。文件名是 `代码_形态key.png`，不含榜单名次，所以同一天重跑扫描时同一只票的同一形态直接覆盖（不会因为名次变化留下一批残留图）；`index.html` 索引页写在当天的目录里、按形态分组折叠，日期取报表同一个 `stamp`（`--stamp`）。

「汇总」含全部形态；「做多榜单」和终端 top 榜只列做多形态，做空形态不会被排进可下单候选。

## 数据存储

支持三种存储后端：

| 后端 | 说明 |
|------|------|
| csv | 一股一文件，方便外部工具查看 |
| json | 一股一文件，JSON格式 |
| sqlite | 单文件数据库，性能最好 |

**stock_info 表字段：** code / name / price / float_mv / amount / updated_at。`updated_at` 只在K线数据实际写入成功后才更新，反映真实的数据新鲜度。

**市场环境数据（`data/market/`）：**

| 文件 | 说明 |
|------|------|
| `daily_summary.csv` | 全市场当日聚合（code/date/pct_chg/high/close/amount），保留 60 个交易日 |
| `limit_up_history.json` | 最近一个交易日的涨停股票代码列表 |
| `limit_up_streak.json` | 每只连板股的连续涨停天数 |
| `sector_map.json` | 申万行业分类缓存（code → sector_name），30 天自动刷新 |

指数 K 线以 `idx_` 前缀存入同一 store（如 `idx_000300`），与个股 K 线共用存储层。

## 项目结构

```
src/tradeyh/
├── fetch.py        # 多渠道数据抓取（东财/新浪/腾讯/Baostock，含指数K线）
├── store.py        # 存储层（sqlite/csv/json）
├── update.py       # 更新编排（含智能时间判断、复权漂移检测、指数更新、日度聚合）
├── scan.py         # 扫描调度（含 universe 过滤、多进程、MarketSnapshot 传入）
├── universe.py     # 股票池过滤
├── indicators.py   # 技术指标（MA/EMA/ATR/MACD/RSI/BOLL）
├── swings.py       # ZigZag 摆动点检测（形态识别地基）
├── report.py       # 报表生成
├── plot.py         # K线图表绘制
├── watch.py        # 观察池管理
├── monitor.py      # 盘中监控（实时报价+形态重判+站稳提醒+触发交易）
├── trade.py        # 模拟交易（自动买卖+仓位管理+移动止盈）
├── broker.py       # 券商接口（东财 EMT 极速柜台仿真 / 本地模拟）
├── market/         # 市场环境评估
│   ├── __init__.py         # 包入口，导出 MarketSnapshot
│   ├── regime.py           # MarketRegime：大盘环境（指数K线 → 0~100）
│   ├── sentiment.py        # MarketSentiment：市场情绪（全市场聚合 → 0~100）
│   ├── sector.py           # SectorHeat：行业热度 + 阶段判定
│   ├── snapshot.py         # MarketSnapshot 数据类 + 编排入口
│   └── sector_fetch.py     # 申万行业分类拉取与缓存
└── patterns/       # 形态检测器
    ├── base.py               # 公共结构、多维度评分、方向/目标/止损表
    ├── break_bottom.py       # 破底翻（B1）
    ├── break_bottom_b2.py    # 破底翻第二买点（B2）
    ├── break_bottom_v2.py    # 破底翻v2（已停用，代码保留）
    ├── break_bottom_v2_b2.py # 破底翻v2第二买点（已停用）
    ├── w_bottom.py           # W底
    ├── m_top.py              # M头
    ├── channel.py            # 通道（上升/下降/箱体/下降通道反转）
    ├── breakout.py           # 整理突破（平台突破/突破回踩/杯柄）
    └── trend_pullback.py     # 趋势回踩再启动
baostock_update.py  # Baostock 独立更新脚本
```

**新增形态**只需写一个暴露 `detect(ctx) -> list[PatternHit]` 的模块，加进 `patterns/__init__.py` 的 `DETECTORS`（并按需补 `PATTERN_KEYS` / `PATTERN_CN`），并在 `base.py` 的 `PATTERN_DIRECTION` / `TARGET_KEYS` 登记方向与目标键即可。若该形态需要更长的历史窗口，记得同步调大 `scan.required_bars()`（破底翻一族按"破底新低回看窗口 + 前低到新低的最大间隔 + 前置跌幅回看"累加：B1 是 `break_within + max_span + prior_lookback`，B2 把 `break_within` 换成 `break_within_b2`）。实际读取量取 `max(required_bars, universe.min_bars)`，这样"K线不足"的判定等同于历史根数不足。

## 数据源

| 渠道 | 特点 | 复权支持 |
|------|------|----------|
| Baostock | Python SDK，最稳定，推荐优先 | qfq/hfq/不复权 |
| 东财 | 字段最全，多主机回退 | qfq/hfq/不复权 |
| 腾讯 | 备用，无成交额 | qfq/hfq/不复权 |
| 新浪 | HTTP 稳定不限流 | 仅不复权 |

### Baostock 独立脚本

不依赖东财快照，单独用 Baostock 更新K线：

```bash
python3 baostock_update.py                       # 增量更新全市场
python3 baostock_update.py --full                # 全量更新
python3 baostock_update.py --start 20240101 --end 20241231
python3 baostock_update.py --codes 000001,600000,300001
python3 baostock_update.py --output ./my_data --storage csv
python3 baostock_update.py --adjust hfq          # 后复权
```

## 注意事项

- 首次全量拉取耗时较长，建议 `--limit 100` 先测试
- Baostock 需 `pip3 install baostock`，HTTP 渠道受限时优先使用
- 东财有限流，失败时降低 `max_workers` 或等待几分钟
- 增量更新会自动检测复权漂移并全量重拉；收盘后运行时，已有当日数据的股票自动跳过
- 扫描结果按综合评分降序排列，已形成和构筑中混合排序
- 观察池只收做多形态，M头 / 下降通道即便配进 `watch.patterns` 也会被跳过
- 实时监控持续轮询报价，Ctrl-C 手动停止（不加时段限制，收盘后拿到的是最新收盘价）
- 模拟交易默认走本地模拟，接入 EMT 前务必确认凭证不进 git
- **市场环境修正**：无指数数据时自动跳过（评分不做修正），首次 `update` 后才会生效。`market.enabled: false` 可完全关闭
- **分批止盈后的移动止盈口径**：止损抬到 `max(MA20, 当前止损)`（只上移）。"最近有效 HL"目前取的是当前止损价而非真实回踩低点，所以剩余仓位实际上是被 MA20 拖着走的——想要更贴形态，需要在持仓里存下 `回踩低点`
- **凭证**：`config.yaml` 已纳入版本控制，EMT 的 `account_id` / `password` 只从环境变量或 `.env` 读取（`TRADEYH_EMT_ACCOUNT_ID` / `TRADEYH_EMT_PASSWORD`），仓库里保持空占位。`.env` 已被 gitignore。若曾把真实账号密码提交过，改仓库历史意义不大——**直接改密**

## 典型工作流

```
收盘后：
  1. python3 cli.py update          # 增量更新K线 + 指数 + 全市场聚合 + 涨停跟踪
  2. python3 cli.py scan            # 全市场扫描（自动计算市场环境/情绪/行业热度，修正评分）
  3. python3 cli.py watch sync      # 同步构筑中的形态到观察池

次日盘中：
  4. python3 cli.py watch run       # 启动监控
     - 每60秒取实时报价，重跑形态检测
     - 连续N轮确认成型 → 终端提醒 + 自动模拟买入
     - 按止损/移动止盈规则自动卖出
     - 形态失效自动移出池子

查看交易：
  5. python3 cli.py trade status    # 查看账户和持仓
  6. python3 cli.py trade history   # 查看历史成交
```
