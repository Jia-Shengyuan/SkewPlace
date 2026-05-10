# Useful Skew 与 Timing-Driven Placement 实验报告

## 1. 本轮结论

- 本轮最重要的澄清是：`520` 不是正式 QoR 对比实验，而是一个 cross-engine smoke case。它的目的只是让 `OpenTimer` 和 `HeteroSTA` 都至少走到第一次 timing feedback，然后看这一轮 feedback 的行为是否正常。
- `Current No-Skew Baseline Suite` 才是当前仓库里更接近正式 end-to-end 的 no-skew 对比。它使用 `1000` 次 global placement iteration，因此会经历很多轮 timing feedback，比 `520` 更接近论文表格里的完整 placement 结果。
- 当前 useful-skew 原型并不是“把 skew 直接施加到最终 STA 上”。它做的是：先在采样出来的寄存器到寄存器路径图上解一个 LP，再把解出来的 `adjusted_setup_slack` 映射成 net weight feedback。最终表里的 `WNS/TNS` 仍然是 timer 对当前 placement 的常规 STA 报告，不是“施加 skew 后”的最终 STA 值。
- 我确认并修复了一个明确的代码问题：当 `useful_skew_weighting_flag=1` 且 `max_skew=0` 时，旧代码仍然会切到 useful-skew 的 sampled weighting 路径，而不是退化回 no-skew。修复后，`max_skew=0` 已经能正确回到 no-skew baseline。
- 我还修了 useful-skew 图构造中的一个实现问题：同一对寄存器之间存在多条 path 时，旧代码按 `path_delay` 选代表路径；现在改成优先按最坏 `slack` 选路径，这更符合 useful-skew/criticality 的语义。
- 但修完上述问题后，`max_skew=50 ps` 的结果仍然显著比 no-skew 差。这说明当前结果变差的主因不是一个简单的单位换算或 `50ps` 数值本身，而是这套 sampled useful-skew weighting heuristic 本身会把 placement 带坏。
- HeteroSTA 现在的结论是：GPU timing 路径可以跑，CPU timing 路径仍然在 `heterosta_extract_rc_from_placement(..., use_cuda=false)` 处 panic。按当前优先级，可以先不继续追 CPU 路径。

## 2. 指标含义

- `WNS`：Worst Negative Slack，最坏单条路径的违例。越大越好，`0` 表示没有最坏负 slack。
- `TNS`：Total Negative Slack，所有负 slack 的总和。越大越好，`0` 表示没有总负 slack。
- `Max violation (ps)`：这里统一写成 `max(0, -WNS_ps)`，也就是把 `WNS` 换成正的违例量。
- `Violation / period`：`max_violation_ps / clock_period_ps`。

### `internal adjusted setup WNS` 是什么

- 这是 useful-skew LP 内部、采样图内部的 setup 最坏 slack。
- 它只看 sampled 的 reg-to-reg setup edges，不是全设计最终 STA 的 WNS。
- 计算公式是：

`adjusted_setup_slack = setup_slack + capture_skew - launch_skew`

- 因为 launch/capture 两端都允许在 `[-max_skew, +max_skew]` 范围内变化，所以一条边的相对改善上界大约是 `2 * max_skew`。
- 因此当 `max_skew=50 ps` 时，`internal adjusted setup WNS` 比 `raw_setup_wns` 改善约 `100 ps` 是合理的，并不奇怪。
- 真正奇怪的不是这个内部值，而是“为什么一打开 useful-skew 分支，最终 placement 的 `WNS/TNS` 会恶化很多”。这部分的原因在第 5 节解释。

## 3. 三类实验分别是什么

### 3.1 `520` cross-engine smoke

- 配置：`superblue1`，`520` 次 global placement iteration。
- 目的：只让流程走到第一次 timing feedback，检查 `OpenTimer` 和 `HeteroSTA` 两条 timing backend 路径是否都能跑通。
- 为什么是 `520`：
  - 当前 timing feedback 触发条件是 `iteration > 500 and iteration % 15 == 0`
  - 所以 `520` 次运行里，只会在 `iter=510` 触发第一次 timing feedback
  - 这正好适合做“第一次 timing feedback 的 smoke 测试”
- 结论：它不是收敛后的正式 QoR 结果，所以不应该直接拿来和论文 Table III 做质量对比。

### 3.2 `Current No-Skew Baseline Suite`

- 配置：8 个 ICCAD2015 `superblue` 设计，`1000` 次 global placement iteration，`useful_skew=false`。
- 目的：得到当前仓库、当前机器、当前参数下的 no-skew baseline，用来和论文量级比较。
- 这组数据更接近“完整 placement 运行”，因此比 `520 smoke` 更适合作为论文对比参考。

### 3.3 `OpenTimer Useful-Skew Range Sweep`

- 配置：`superblue1`，`1000` 次 global placement iteration，`useful_skew=true`，分别取 `max_skew = 50 / 500 / 2000 / 5000 ps`。
- 目的：看 useful-skew 这条 sampled weighting heuristic 打开后，不同 skew budget 对内部 LP 指标和最终 placement 指标有什么影响。
- 注意：这组实验的真正对照不是“最终 STA 上多给了多少 skew”，而是“placement 中的 net-weight update 路径切换成了另一套 heuristic”。

## 4. 为什么 `520` 和论文差很多，但 `1000 no-skew` 又比较接近

### 4.1 `520` 和论文不是同一类实验

- 论文 Table III 对应的是完整 timing-driven placement 结果。
- `520 smoke` 只跑到第一次 timing feedback 后再多走 10 个 iteration，本质上是“第一次 timing feedback 行为检查”。
- 所以 `520 smoke` 比论文差很多是正常的，它根本不是收敛后的 QoR。

### 4.2 `1000 no-skew baseline suite` 更接近完整 placement

- `1000` 次 iteration 会触发很多轮 timing feedback。
- 因此它比 `520 smoke` 更接近完整 timing-driven placement 的行为。
- 这就是为什么 `1000 no-skew` 在量级上会更接近论文，而 `520 smoke` 会偏差很大。

### 4.3 一个直观对比

| 实验 | 目的 | 迭代数 | 是否适合和论文 QoR 对比 |
| --- | --- | ---: | --- |
| `520 cross-engine smoke` | 看第一次 timing feedback 是否正常 | 520 | 不适合 |
| `700 GP Feedback Comparison` | 早期 useful-skew 原型检查 | 700 | 只能粗看趋势 |
| `1000 current no-skew baseline` | 当前仓库正式 no-skew baseline | 1000 | 适合做量级对比 |
| `1000 useful-skew range sweep` | 观察 useful-skew heuristic 的影响 | 1000 | 可以做 prototype 行为分析，不宜直接当论文复现 |

## 5. 为什么 `max_skew=50 ps` 也会让最终违例恶化很多

这是这轮排查里最重要的结论。

### 5.1 不是“只给最终 STA 多了 50ps skew”

当前代码里，一旦 `useful_skew_weighting_flag=1` 且 `max_skew > 0`，走的就不是原生 OpenTimer `lilith` 更新，而是另一套 sampled useful-skew weighting：

1. 从 timer 里取 top `100` 条 `max` test paths 和 top `100` 条 `min` test paths
2. 构造 sampled reg-to-reg timing graph
3. 解 useful-skew LP
4. 把 sampled setup path 上出现的 net 映射到 `adjusted_setup_slack`
5. 用这些 sampled slacks 去更新 net criticality 和 net weights

也就是说，`50ps` 不是“在 no-skew 的基础上只改了 50ps”。

更准确地说，是：

- `useful_skew=false`：走原生 `lilith`
- `useful_skew=true` 且 `max_skew>0`：切换到另一套 sampled heuristic

因此，即使 `max_skew` 很小，只要开关打开，placement 的优化轨迹就已经完全变了。

这里要特别澄清一个容易混淆的点：

- 从“算法语义”上说，`max_skew=0` 时 useful-skew 应该退化成 no-skew，因为 LP 已经没有任何可用 skew 自由度。
- 但旧实现并不是这样。旧实现的判断条件只有 `useful_skew_weighting_flag`，没有额外判断 `max_skew` 是否为 `0`。
- 结果就是：哪怕 `max_skew=0`，代码仍然会切到 sampled useful-skew weighting 路径，只是 LP 里求出来的 skew 恰好全为 `0`。
- 问题在于：即使 skew 全为 `0`，这条分支仍然会用 sampled graph 和 sampled setup paths 去更新 net weights，因此它和原生 no-skew `lilith` 不是同一个算法。
- 所以之前出现“`skew=0` 但结果和 no-skew 不一样”并不是数学上奇怪，而是实现上走了不同代码路径。

### 5.2 这个 heuristic 会被重复应用很多次

在 `1000` 次 iteration 的实验里，timing feedback 会从 `iter=510` 开始每 `15` 次触发一次，总共会发生很多轮。

- 每一轮都会重新修改 net weights
- 每一轮都只基于很小的 sampled path 集合
- 每一轮都是乘法式更新：`net_weights[net_id] *= (1 + net_criticality[net_id])`

因此，一个看起来很小的单次偏差，经过很多轮 feedback 之后，可以把 placement 轨迹拉得很远。这就是为什么 `50ps` 也可能导致最终 `WNS/TNS` 恶化很多。

### 5.3 `50ps` 的恶化并不意味着“50ps 直接造成了 9ns 违例”

这点必须明确：

- `50ps` 只是 LP 里每个寄存器 skew 变量的上下界
- 最终恶化的 `~9ns` 不是这个边界直接加出来的
- 真正造成差异的是“启用 useful-skew 后，net-weight feedback 路径切换了”，而不是“50ps 这个数本身太大”

### 5.4 本轮实验证据

我补跑了一个关键对照：

- `superblue1`, `1000` iterations, `useful_skew=true`, `max_skew=0`

旧代码下，这个 case 仍然会严重变差，说明即使没有任何可用 skew，它也会走 sampled heuristic。这是明确 bug。

修复后，这个 case 现在会正确退化为 no-skew baseline：

| 模式 | TNS (`x1e5 ps`) | WNS (`x1e3 ps`) |
| --- | ---: | ---: |
| no-skew baseline | -68.36015 | -12.805035 |
| useful-skew, `max_skew=0`，修复后 | -68.36015 | -12.805035 |

然后我又重跑了 `max_skew=50 ps`：

| 模式 | TNS (`x1e5 ps`) | WNS (`x1e3 ps`) |
| --- | ---: | ---: |
| useful-skew, `max_skew=50`，修复后复跑 | -177.16288 | -21.50158 |

它几乎没变。这说明：

- `max_skew=0` 的语义 bug 确实存在，我已经修了
- 但 `50ps` 结果仍然很差，说明当前主要问题不是这个 bug，而是 heuristic 本身

## 6. 本轮代码检查与修复

### 6.1 OpenTimer `no-skew` 在 `520` case 崩溃

问题：

- 文件：`dreamplace/ops/timing/src/net_weighting_scheme.h`
- 旧代码：`timer.report_wns().value()`
- 当 `report_wns()` 为空 optional 时，会直接报 `bad optional access`

修复：

- 若 `report_wns()` 无值，则记录 warning 并跳过本轮 `lilith` net-weight update

结果：

- `OpenTimer superblue1 520 no-skew` 已能完整跑完

### 6.2 `max_skew=0` 不退化为 no-skew

问题：

- 旧代码下，只要 `useful_skew_weighting_flag=1`，即使 `max_skew=0` 也会切到 useful-skew sampled weighting 路径

修复：

- 在 `dreamplace/ops/timing/useful_skew.py` 中，将 `max_skew <= 0` 显式定义为 exact no-skew：直接返回全零 skew，不再进入 LP，避免求解器数值噪声
- 在 `dreamplace/ops/timing/src/timing_cpp.{h,cpp}` / `timing_pybind.cpp` 中新增 `update_net_weights_lilith_with_net_slack(...)`
- 在 `dreamplace/ops/timing/timing.py` 中，不再用 Python 复刻 `lilith` 更新，而是仅构造 adjusted net slack，再调用 native C++ `lilith` 更新逻辑

结果：

- `skew=0` 现在已经能在单步 checkpoint 对比和完整 `1000`-iteration placement 两个层面都正确退化为 native no-skew / `lilith`

### 6.3 sampled reg-to-reg 图合并策略不合理

问题：

- 文件：`dreamplace/ops/timing/useful_skew.py`
- 旧代码在合并同一对寄存器之间的多条 path 时，优先按 `path_delay` 选代表路径
- 这对 useful-skew/criticality 不合理，应该优先按最坏 `slack` 选代表路径

修复：

- 改为优先按最坏 `setup_slack` / `hold_slack` 选路径；只有 slack 不可用时才回退到 delay 规则

结果：

- 这是一处应该修的实现问题，但从目前复跑结果看，它不是导致 `skew=50` 大幅恶化的主因

## 7. 关键实验结果

### 7.1 `superblue1`：`520` cross-engine smoke

这组实验只看第一次 timing feedback，不适合直接拿来和论文比 QoR。

| Flow | Clock period (ps) | Runtime (s) | 最后一次报告的 TNS (`x1e5 ps`) | 最后一次报告的 WNS (`x1e3 ps`) | Max violation (ps) | Violation / period | 显式 skew 指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| paper DREAMPlace 4.0 | 9500 | 526.45 | -87.91 | -14.23 | 14230.00 | 1.498 | 论文 no-skew baseline |
| paper DREAMPlace 4.0 + HeteroSTA | 9500 | 112.35 | -59.62 | -12.58 | 12580.00 | 1.324 | 论文 HeteroSTA timing-driven placement |
| current OpenTimer, no-skew, `520 iters` | 9500 | 39.77 | -281.399 | -24.538 | 24537.85 | 2.583 | N/A |
| current OpenTimer, skew `2000 ps`, `520 iters` | 9500 | 51.70 | -281.399 | -24.538 | 24537.85 | 2.583 | `adjusted_setup_wns = -22444.977 ps` |
| current HeteroSTA GPU, no-skew, `520 iters` | 9500 | 20.38 | -348.609 | -20.005 | 20005.29 | 2.106 | N/A |
| current HeteroSTA GPU, skew `2000 ps`, `520 iters` | 9500 | 127.46 | -348.609 | -20.005 | 20005.29 | 2.106 | `adjusted_setup_wns = -16005.287 ps` |

结论：

- `520` 结果只说明“第一次 timing feedback 之后，各 engine 的行为如何”
- 它不说明最终 placement QoR

### 7.2 OpenTimer `1000` iteration no-skew baseline

| Design | Clock period (ps) | Current no-skew TNS (`x1e5 ps`) | Current no-skew WNS (`x1e3 ps`) | Max violation (ps) | Violation / period | Runtime (s) | Paper DREAMPlace 4.0 TNS | Paper DREAMPlace 4.0 WNS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| superblue1 | 9500 | -68.36 | -12.81 | 12805.04 | 1.348 | 452.18 | -87.91 | -14.23 |
| superblue3 | 10000 | -45.11 | -15.20 | 15198.99 | 1.520 | 601.88 | -48.74 | -15.10 |
| superblue4 | 6000 | -125.25 | -11.90 | 11897.33 | 1.983 | 170.82 | -145.90 | -12.84 |
| superblue5 | 9000 | -92.24 | -24.39 | 24392.75 | 2.710 | 490.78 | -95.79 | -29.55 |
| superblue7 | 5500 | -39.97 | -15.40 | 15395.52 | 2.799 | 635.41 | -59.74 | -15.22 |
| superblue10 | 10000 | -563.78 | -20.11 | 20105.55 | 2.011 | 802.75 | -655.36 | -23.11 |
| superblue16 | 5500 | -54.69 | -6.03 | 6029.48 | 1.096 | 343.51 | -63.69 | -10.02 |
| superblue18 | 7000 | -43.39 | -11.33 | 11334.91 | 1.619 | 298.58 | -46.75 | -11.53 |

结论：

- 这组结果在量级上和论文 no-skew baseline 更接近
- 因为这组实验是完整 `1000` iteration placement，不是 `520 smoke`

### 7.3 OpenTimer useful-skew range sweep

说明：

- `no-skew` 与 `skew=0` 现在已经语义一致
- `skew=50` 在本轮修复后已复跑，结果基本不变
- `500/2000/5000` 仍引用之前的结果；结合 `50ps` 复跑结果看，主结论稳定
- 这里的 `Final TNS/WNS` 实际上是 `run_skew_timing_feedback_summary.py` 从 `NonLinearPlace` 的 metrics 列表中取到的“最后一个非空 timing metric”。在 `legalize_flag=0` 的情况下，这个值对应的是“最后一次 timing feedback 时刻”的 STA，不一定等于优化结束后对最终 placement 重新做一次 timing 的结果。

| 模式 | Runtime (s) | Final TNS (`x1e5 ps`) | Final WNS (`x1e3 ps`) | Max violation (ps) | Violation / period | internal adjusted setup WNS (ps) | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no-skew | 452.18 | -68.360 | -12.805 | 12805.035 | 1.348 | N/A | 正常 baseline |
| useful-skew, `max_skew=0 ps` | 489.62 | -68.360 | -12.805 | 12805.035 | 1.348 | N/A | 本轮修复后正确退化为 no-skew |
| useful-skew, `max_skew=50 ps` | 538.62 | -177.163 | -21.502 | 21501.580 | 2.263 | -21401.574 | 本轮修复后复跑，仍显著变差 |
| useful-skew, `max_skew=500 ps` | 548.75 | -176.318 | -21.429 | 21428.812 | 2.256 | -20428.814 | 历史结果 |
| useful-skew, `max_skew=2000 ps` | 529.47 | -170.693 | -21.568 | 21568.098 | 2.270 | -18241.809 | 历史结果 |
| useful-skew, `max_skew=5000 ps` | 546.08 | -167.178 | -21.267 | 21266.764 | 2.239 | -14978.070 | 历史结果 |

关键观察：

- `internal adjusted setup WNS` 会随着 `max_skew` 变大而持续改善，这说明 LP 本身是活的
- 但最终 `WNS/TNS` 一直明显差于 no-skew，这说明真正有问题的是“LP 解如何映射成 net-weight feedback”这一步
- 因而当前 useful-skew 更像一个 diagnostic/upper-bound prototype，不像一个已经可用的 timing-driven placement 策略

### 7.3A 按“只改 slack、其余保持 lilith 不变”重写后的正式结果

这一小节对应用户提出的新设想：

- 保留原生 `lilith` 的 `WNS`、criticality 公式、动量更新、乘法式 net-weight 更新
- useful-skew 只在“每个 net 使用的 slack”这一处做修正
- 直觉上，`max_skew=0` 时应当自然退化为原生 `lilith`

我据此重写了一版 OpenTimer useful-skew。中间发现：

- 只在 Python 侧复刻 `lilith` 更新时，`max_skew=0` 仍会和 native `lilith` 存在残余差异
- 差异来源主要不是 LP 本身，而是“更新公式/遍历对象/浮点路径没有完全走 native 实现”
- 因此最终修正改为：Python 只负责生成 adjusted net slack，真正的 `lilith` 更新仍在 native C++ 中完成

在这一修正之后，重跑 `superblue1` 的 `1000`-iteration 实验，结果如下：

| 模式 | Clock period (ps) | Runtime (s) | TNS (`x1e5 ps`) | WNS (`x1e3 ps`) | Max violation (ps) | Violation / period | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no-skew baseline | 9500 | 452.18 | -68.360 | -12.805 | 12805.035 | 1.348 | 当前已知正常 baseline |
| lilith-compatible useful-skew, `max_skew=0` | 9500 | 503.07 | -68.360 | -12.805 | 12805.035 | 1.348 | 已通过 `skew=0` 退化 sanity check |
| lilith-compatible useful-skew, `max_skew=50` | 9500 | 507.53 | -72.195 | -13.200 | 13200.426 | 1.389 | 相比 baseline 只有轻微变差，不再出现旧版灾难性恶化 |
| lilith-compatible useful-skew, `max_skew=2000` | 9500 | 534.36 | -67.366 | -16.109 | 16108.579 | 1.696 | HPWL 明显下降，但 WNS 仍变差 |

当前判断：

- 最关键的 sanity check 已经满足：`max_skew=0` 现在会自然退化到原生 no-skew baseline。
- 与旧 useful-skew heuristic 相比，这版 `lilith-compatible` 实现明显更符合用户原始语义，因为它不再出现 `50 ps -> WNS -21.5 ns` 的灾难性退化。
- 但这不代表 useful-skew 已经稳定带来 QoR 改善。当前 `50 ps` 仍略差于 baseline，`2000 ps` 虽然把 HPWL 从 `5.243B` 降到 `4.718B`，但最终 `WNS` 仍从 `-12.805 ns` 变差到 `-16.109 ns`。
- 因而当前正式结论是：实现语义已经基本对齐并通过 `skew=0` sanity check，但“在线 useful-skew 反馈是否真的提升 placement QoR”仍未被实验证明。

### 7.4 固定 placement 后只做一次 useful-skew 分析

这是本轮新增的关键验证实验，用来回答一个核心问题：

- 到底是 `solve_useful_skew` / useful-skew 本身写挂了
- 还是“把 useful-skew 信号反复回灌为 net weights”这件事会把 placement 带坏

实验方法：

1. 先按 no-skew 正常完成 `superblue1` 的 `1000` iteration placement
2. placement 完成后，不再修改 net weights，不再继续 placement
3. 只在最终固定的 placement 上做一次 useful-skew 分析

固定 placement 的 sampled baseline（`n=100`）：

| 指标 | 数值 |
| --- | ---: |
| sampled setup worst slack | -15956.827 ps |
| sampled setup TNS | -252908.366 ps |
| sampled setup violating edges | 20 |
| sampled hold worst slack | -3156.250 ps |
| sampled hold TNS | -91852.652 ps |
| sampled hold violating edges | 52 |
| sampled registers | 85 |
| sampled edges | 72 |

在这个固定 placement 上只解一次 useful-skew LP：

| max_skew (ps) | LP margin / skew worst setup slack (ps) | 相对 baseline 改善 |
| ---: | ---: | ---: |
| 50 | -15906.827 | +50 ps |
| 2000 | -13956.827 | +2000 ps |

解释：

- 这组结果是非常合理的。
- 在固定 placement 上，`max_skew=50 ps` 时，LP 只改善了约 `50 ps`；`max_skew=2000 ps` 时，LP 改善了约 `2000 ps`。
- 这说明 useful-skew LP 本身没有表现出“莫名其妙把 50ps 放大成 9ns”这种错误行为。
- 真正异常的是：一旦把 useful-skew 的 sampled slack 信号反复转成 net-weight feedback 并参与后续 placement，最终 placement 轨迹就会明显变差。

因此，这组新实验支持下面这个判断：

- `solve_useful_skew` 本身不是当前主要问题
- 当前主要问题是 useful-skew 到 net-weight feedback 的映射策略，以及它在多轮 placement feedback 中的累积效应

### 7.5 在线 useful-skew 与 single-skew 的并排对比

为了把“在线反复 feedback 的 useful-skew”和“固定最终 placement 上只做一次 useful-skew 分析”放到同一视角下，下面给出并排表。

说明：

- `在线 useful-skew`：和 7.3 一样，在 placement 过程中从 `iter=510` 开始每 `15` 轮反复做 timing feedback，并修改 net weights。
- `single-skew`：先完整跑完 no-skew placement，再在最终固定 placement 上做一次 useful-skew 分析；不回灌 net weights，不继续 placement。
- `single-skew` 的全局 `TNS/WNS` 是“优化结束后对最终固定 placement 重新做一次 timing”的结果，因此不会因为“只做分析”而改变。
- 所以 7.5 里的 fixed-placement 行，应该与“fixed-placement baseline”比较，而不是直接与 7.3 的 `no-skew` 行比较。7.3 的 `no-skew` 行是“最后一次 timing feedback 快照”，7.5 是“优化结束后重新 timing”的结果，二者测量时刻不同。

| 模式 | max_skew (ps) | 是否反复修改 net weights | 全局 TNS (`x1e5 ps`) | 全局 WNS (`x1e3 ps`) | Max violation (ps) | Violation / period | sampled skew 指标 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| no-skew baseline | N/A | 否 | -68.360 | -12.805 | 12805.035 | 1.348 | N/A |
| fixed-placement baseline | N/A | 否 | -67.070 | -15.957 | 15956.829 | 1.680 | N/A |
| fixed-placement single-skew | 0 | 否 | -67.070 | -15.957 | 15956.829 | 1.680 | `skew_margin = -15956.827 ps`, 改善 `+0 ps` |
| 在线 useful-skew feedback | 50 | 是 | -177.163 | -21.502 | 21501.580 | 2.263 | `adjusted_setup_wns = -21401.574 ps` |
| 在线 useful-skew feedback | 2000 | 是 | -170.693 | -21.568 | 21568.098 | 2.270 | `adjusted_setup_wns = -18241.809 ps` |
| fixed-placement single-skew | 50 | 否 | -67.070 | -15.957 | 15956.829 | 1.680 | `skew_margin = -15906.827 ps`, 改善 `+50 ps` |
| fixed-placement single-skew | 2000 | 否 | -67.070 | -15.957 | 15956.829 | 1.680 | `skew_margin = -13956.827 ps`, 改善 `+2000 ps` |

这张表说明了两件事：

- 如果只是对最终固定 placement 做一次 useful-skew 分析，结果是正常且有界的；全局 placement `TNS/WNS` 不会被“分析动作本身”拉坏。
- 一旦把 sampled useful-skew 信号放进在线 placement feedback，真正被改变的是后续优化轨迹；最终恶化来自多轮 net-weight 更新，而不是 single-skew 分析本身。
- `fixed-placement single-skew, max_skew=0` 的全局 `TNS/WNS` 与 `50/2000` 完全一致，这进一步证明 single-skew 分析本身不会改变 placement 结果；变化的只有 sampled skew margin。

## 8. HeteroSTA 当前状态

- GPU timing 路径可跑，`superblue1 520` smoke 已经完成
- CPU timing 路径仍然在第一次 timing feedback 进入 `heterosta_extract_rc_from_placement(..., use_cuda=false)` 时 panic
- 当前阶段可以先不继续追 HeteroSTA CPU 路径，优先把 OpenTimer useful-skew heuristic 本身分析清楚

## 9. 建议的下一步

1. 暂时不要再把 `max_skew` 数值本身当作主要问题。当前主要问题是 sampled useful-skew weighting heuristic 会把 placement 带坏。
2. 如果要继续做 useful-skew，建议先把它定位成“诊断工具”而不是“正式优化器”。
3. 下一步更值得做的是：
   - 记录每一轮 timing feedback 的 `affected_nets`、criticality 分布、weight 变化分布
   - 比较 raw `lilith` 和 useful-skew 分支到底哪些 nets 被重复放大
   - 考虑不要只回灌 sampled setup path 上的全部 nets，而是更严格地筛选真正需要加权的 nets
4. HeteroSTA CPU 路径当前可以先挂起；GPU 路径已经足够支持 smoke 级联调。
