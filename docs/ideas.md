# Useful Skew + DREAMPlace 研究记录

## 1. 当前目标

当前目标不是一步到位把 useful skew 直接并入 DREAMPlace 的主梯度优化，而是按可落地、可验证的顺序推进。

总目标：

- 在 DREAMPlace 现有 placement 流程之上，引入 useful skew 的后处理求解与后续联动优化。
- 先回答 useful skew 在当前 benchmark / 当前 placement 结果上到底能带来多少时序改善。
- 再决定是否继续做 skew-aware net weighting 或更深入的 skew-aware objective。

当前最具体的近期目标：

- 先跑通 baseline benchmark。
- 在 placement 结束后，提取 skew scheduling 所需的时序约束。
- 先实现一个 post-placement useful skew solver。
- 用它评估：当前 placement 解经过 skew 调度后，WNS/TNS 理论上能改善多少。

## 2. 阶段化路线

### 阶段 0：基线与环境确认

- 确认本地 Docker + GPU 环境可运行 DREAMPlace。
- 确认哪些 benchmark 当前仓库已经具备，哪些需要额外下载。
- 跑一个现成 baseline，记录初始结果。

### 阶段 1：post-placement useful skew

这是当前决定先做的方案。

核心思想：

- 先用 DREAMPlace 正常完成 placement。
- placement 完成后，不改 cell 位置。
- 从 timer 中提取寄存器到寄存器的 setup / hold 约束。
- 求一组 skew 分配 `s_i`。
- 评估这组 skew 对当前 placement 解的理论时序改善。

这一阶段的作用：

- 验证 useful skew 本身是否值得做。
- 得到一个清晰 baseline：
  - 无 skew
  - ideal skew
  - 可实现性受限 skew

### 阶段 2：交替优化（alternating optimization）

如果阶段 1 证明 skew 有价值，再进入这一阶段。

核心思想：

- placement 跑若干轮
- 做一次 timing analysis
- 做一次 skew scheduling
- 用 skew-aware timing signal 更新 criticality / net weights
- 继续 placement

这一步不要求 skew 直接进入每一步梯度，但会让 skew 对 placement 产生反馈。

### 阶段 3：更深入的 skew-aware objective

更进一步的版本可以考虑：

- 把 skew 变量或 skew penalty 更直接地并入 placement objective
- 或引入与 CTS 可实现性更紧耦合的模型

这一步难度更高，不作为当前第一阶段目标。

## 3. 当前决定先做的方案

当前方案：

- 先做 post-placement useful skew。
- 暂时不接入真实 CTS，不要求真实 clock tree 立即可实现。
- 先做 ideal skew scheduling。
- 在此基础上，再加入空间一致性 / 可实现性约束。

当前不做的事情：

- 不直接修改 DREAMPlace 主优化目标。
- 不直接把 skew 当作 placement 的梯度变量。
- 不先做 CTS 协同求解。

## 4. 第一版 useful skew 设计稿

### 4.1 输入

第一版 solver 需要的输入应当包括：

- placement 结束后的寄存器位置
- 寄存器集合 `R`
- 寄存器到寄存器的时序边 `E`
- 对每条边 `i -> j`：
  - `dmax_ij`：最大数据路径延迟
  - `dmin_ij`：最小数据路径延迟
  - `setup_j`
  - `hold_j`
- 时钟周期 `T`
- 可选：寄存器物理坐标，用于空间一致性约束

当前 DREAMPlace Python 层默认并不直接暴露完整的 reg-to-reg 约束，所以很可能需要扩 timer 接口，或者新增导出逻辑。

### 4.2 变量

- `s_i`：寄存器 `i` 的 clock arrival time / skew 变量
- `m`：统一 margin 变量

### 4.3 基本约束

对每条寄存器边 `i -> j`：

setup 约束：

```text
s_j - s_i >= dmax_ij + setup_j - T
```

hold 约束：

```text
s_j - s_i <= dmin_ij - hold_j
```

为避免解整体平移导致不唯一，需要固定一个参考点，例如：

```text
s_ref = 0
```

并建议加全局范围限制：

```text
-Smax <= s_i <= Smax
```

### 4.4 第一版目标函数

第一版建议优先使用：

```text
maximize m
```

subject to

```text
s_j - s_i >= dmax_ij + setup_j - T + m
s_j - s_i <= dmin_ij - hold_j - m
```

解释：

- 如果最优 `m >= 0`，说明该 placement 下 skew 调度可行，且还留有统一安全裕量。
- 如果最优 `m < 0`，说明不可行，但 `-m` 就是最小可能的最大违例。

这非常适合作为第一版原型，因为它是一个标准 LP（线性规划）问题，简单、稳、好解释。

### 4.5 不可行时的处理

即使 setup / hold 约束整体不可行，也仍然可以求“最不坏”的 skew。

第一版建议仍优先使用 `maximize m` 版本，因为它天然已经包含“最大违例最小”的意义。

如果后续需要更丰富的目标，可以再考虑：

- 最小化总违例 `sum u_ij`
- 最小化平方违例 `sum u_ij^2`

### 4.6 空间一致性 / 可实现性约束

如果两个寄存器物理上很近，但分配了很大的 skew 差值，真实 CTS 往往更难实现。

因此可以加入空间一致性约束：

```text
|s_i - s_j| <= f(dist(i, j))
```

其中：

- `dist(i, j)` 是物理距离
- `f` 是一个随距离增大而放宽的函数

这类约束可以拆成两条线性不等式：

```text
s_i - s_j <= f_ij
s_j - s_i <= f_ij
```

因此它和 setup / hold 约束在数学本质上是同类约束，仍然适合并入 LP / QP。

第一版实现建议：

- 先做 ideal skew，不加空间约束
- 再做 constrained skew，只对近邻寄存器对加约束

不建议一开始对所有寄存器对都加这类约束，否则规模会过大。

### 4.7 输出

第一版 solver 的输出建议包括：

- 每个寄存器的 skew 值 `s_i`
- 最优 margin `m`
- 约束可行性状态
- 统计信息：
  - 最大 skew
  - skew 分布
  - 邻近寄存器 skew 差分布
- skew 应用前后的 WNS / TNS 对比

## 5. 代码层面的建议落点

当前先不急着改 DREAMPlace 主循环，第一版建议新增一个独立后处理模块，例如：

- `dreamplace/useful_skew/` 或 `dreamplace/skew/`

可能需要的几个组成：

- `extract_timing_graph.py`
  - 从 timer / timing backend 提取 reg-to-reg 约束
- `skew_solver.py`
  - LP/QP 求解 skew
- `evaluate_skew.py`
  - 对比 skew 前后的 timing 指标

如果后续进入阶段 2，再考虑把它接回：

- `dreamplace/NonLinearPlace.py`
- `dreamplace/Timer.py`
- `dreamplace/ops/timing/...`

### 5.1 当前第一版已经落下去的位置

当前已经按最小侵入方式，把 `reg-to-reg timing graph` 导出和最小 `useful skew` 原型放在 OpenTimer 相关模块旁边：

- `dreamplace/ops/timing/src/timing_pybind.cpp`

## 6. 实验报告模板

当前 useful skew 原型建议统一输出两组结果：

- baseline：同一 placement 解上，不做 skew 调度，只统计导出的 reg-to-reg timing graph
- useful skew：在同一 placement 解上，对若干 `n` 做 skew scheduling sweep

建议每个 benchmark 至少记录以下字段。

### 6.1 Baseline 字段

- `path_counts.max`
- `path_counts.min`
- `num_registers`
- `num_edges`
- `setup.worst_slack`
- `setup.tns`
- `setup.violating_edges`
- `hold.worst_slack`
- `hold.tns`
- `hold.violating_edges`
- `worst_setup_edges`
- `worst_hold_edges`

解释：

- `path_counts.*` 表示本次从 OpenTimer 抽出的 sequential test paths 数量
- `num_registers / num_edges` 表示导出的 reg-to-reg 图规模
- `setup/hold.*` 是在该图上的基础 timing 统计，不带任何 skew 求解

### 6.2 Useful Skew 字段

- `n`
- `path_counts.max`
- `path_counts.min`
- `num_registers`
- `num_edges`
- `skew_success`
- `skew_margin`
- `num_constraints`
- `sample_skews`
- `worst_setup_edges`
- `worst_hold_edges`

解释：

- `n` 是每个 split 抽取的 worst sequential test path 数量上限
- `skew_margin` 是 LP 求得的统一 margin；正值更好，负值表示当前约束集本身已严重违例
- `num_constraints` 是最终进入 LP 的 setup/hold 约束数量

### 6.3 当前自动化脚本

当前实验脚本：

- `test/iccad2015.ot/run_useful_skew_summary.py`
- `test/iccad2015.ot/evaluate_useful_skew_checkpoint.py`

功能：

- 读取 benchmark 配置
- 跑 placement（当前默认只跑 global placement，不跑 legalize）
- 在 placement 后更新 timing
- 输出 baseline
- 对多个 `n` 输出 useful skew sweep
- 将摘要写到：
  - `results/iccad2015.ot/<design>_useful_skew_n<n>_iter<iter>.json`

示例：

```bash
python3 test/iccad2015.ot/run_useful_skew_summary.py test/iccad2015.ot/superblue1.json 100 10 50.0 1,10,100
```

参数含义：

- 第 1 个参数：benchmark json
- 第 2 个参数：baseline 使用的 `n`
- 第 3 个参数：global placement iteration
- 第 4 个参数：`max_skew`
- 第 5 个参数：useful skew sweep 的 `n` 列表

### 6.4 多迭代点采样

如果希望一次长跑后再评估多个迭代点，而不是重复跑多次 placement，当前增加了一个实验专用参数：

- `dump_global_place_checkpoint_steps`

用法：

- 在 json 或运行前的参数对象里设置若干迭代点，例如 `[100, 300, 1000]`
- global placement 在这些 iteration 自动导出：
  - `results/<design>/gp_checkpoints/<design>.iter<k>.gp.pklz`
- 之后用：
  - `test/iccad2015.ot/evaluate_useful_skew_checkpoint.py`
  对每个 checkpoint 单独做 baseline / useful skew 评估

这样可以避免把 `iter=300/1000` 各自完整重跑一遍。

## 7. 当前实验结果（superblue1, OpenTimer, GP-only, iter=10）

当前已在服务器隔离副本上完成一组真实 benchmark 实验：

- benchmark: `superblue1`
- timer: `OpenTimer`
- placement: global placement only
- iteration: `10`
- useful skew sweep: `n = 1, 10, 100`
- `max_skew = 50 ps`

结果文件：

- `results/iccad2015.ot/superblue1_useful_skew_n100_iter10.json`

### 7.1 Baseline

- `num_tests = 288532`
- `path_counts = {max: 100, min: 100}`
- `num_registers = 98`
- `num_edges = 61`
- `setup.worst_slack = -54380.902 ps`
- `setup.tns = -543142.707 ps`
- `setup.violating_edges = 10`
- `hold.worst_slack = -2337.788 ps`
- `hold.tns = -92447.111 ps`
- `hold.violating_edges = 51`

### 7.2 Useful Skew Sweep

`n = 1`

- `num_registers = 4`
- `num_edges = 2`
- `skew_margin = -54330.914 ps`
- `num_constraints = 2`

`n = 10`

- `num_registers = 20`
- `num_edges = 11`
- `skew_margin = -54330.902 ps`
- `num_constraints = 11`

`n = 100`

- `num_registers = 98`
- `num_edges = 61`
- `skew_margin = -54330.902 ps`
- `num_constraints = 61`

### 7.3 当前观察

- `n = 1/10/100` 时，`skew_margin` 基本不变。
- 说明当前 placement 状态下，最差的少数 setup path 已经主导了 useful skew LP。
- 在 `max_skew = 50 ps` 的限制下，useful skew 不可能修复当前 `-54 ns` 量级的 setup violation。
- 因此当前 useful skew 原型更像一个 post-placement diagnosis / upper-bound 工具，而不是在这种初始 placement 条件下可直接显著修复 timing 的手段。

## 8. 更收敛实验结果（superblue1, OpenTimer, GP-only, iter=100）

当前又补做了一组更长的 global placement 实验：

- benchmark: `superblue1`
- timer: `OpenTimer`
- placement: global placement only
- iteration: `100`
- useful skew sweep: `n = 1, 10, 100`
- `max_skew = 50 ps`

结果文件：

- `results/iccad2015.ot/superblue1_useful_skew_n100_iter100.json`

### 8.1 Baseline

- `num_registers = 45`
- `num_edges = 48`
- `setup.worst_slack = -22048.561 ps`
- `setup.tns = -342679.239 ps`
- `setup.violating_edges = 23`
- `hold.worst_slack = -3611.133 ps`
- `hold.tns = -83289.539 ps`
- `hold.violating_edges = 25`

### 8.2 Useful Skew Sweep

`n = 1`

- `num_registers = 4`
- `num_edges = 2`
- `skew_margin = -21998.562 ps`

`n = 10`

- `num_registers = 21`
- `num_edges = 16`
- `skew_margin = -21948.561 ps`

`n = 100`

- `num_registers = 45`
- `num_edges = 48`
- `skew_margin = -21948.561 ps`

### 8.3 对比 iter=10 的结论

- `iter=100` 相比 `iter=10`，setup 最差 slack 从约 `-54 ns` 改善到了约 `-22 ns`。
- hold TNS 也有所改善，但仍然较差。
- useful skew 的最优 margin 同样明显改善，但仍然是大负值。
- `n = 10` 和 `n = 100` 的 `skew_margin` 仍然几乎一致，说明决定 LP 结果的仍是最关键的一小批 path。
- 这说明更收敛的 placement 会改善 useful skew 的理论上界，但当前 `max_skew = 50 ps` 仍远小于主导 setup violation 的量级。

### 8.4 Checkpoint 评估结果（superblue1, iter=300）

为了避免把 `iter=300/500/700/...` 各自完整重跑一遍，当前已经在 global placement 中加入 checkpoint 导出，并对中间迭代点单独做 timing / useful skew 评估。

当前已验证：

- 一次 `iter=320` 的长跑可以成功导出：
  - `results/superblue1/gp_checkpoints/superblue1.iter100.gp.pklz`
  - `results/superblue1/gp_checkpoints/superblue1.iter300.gp.pklz`
- 之后可直接用：
  - `test/iccad2015.ot/evaluate_useful_skew_checkpoint.py`
  对导出的 `.gp.pklz` 做单独评估

当前已完成 `iter=300` checkpoint 的评估：

- checkpoint: `results/superblue1/gp_checkpoints/superblue1.iter300.gp.pklz`
- 结果文件：`results/iccad2015.ot/superblue1.iter300.gp_useful_skew.json`

Baseline：

- `num_registers = 73`
- `num_edges = 65`
- `setup.worst_slack = -17527.508 ps`
- `setup.tns = -278755.902 ps`
- `setup.violating_edges = 16`
- `hold.worst_slack = -1651.891 ps`
- `hold.tns = -68989.459 ps`
- `hold.violating_edges = 49`

Useful skew sweep：

`n = 1`

- `num_registers = 4`
- `num_edges = 2`
- `skew_margin = -17477.516 ps`

`n = 10`

- `num_registers = 16`
- `num_edges = 11`
- `skew_margin = -17427.508 ps`

`n = 100`

- `num_registers = 73`
- `num_edges = 65`
- `skew_margin = -17427.508 ps`

当前观察：

- `iter=300` 相比 `iter=100`，setup 最差 slack 继续从约 `-22 ns` 改善到约 `-17.5 ns`。
- hold 最差 slack 与 hold TNS 也有改善。
- useful skew 的 margin 随 placement 收敛继续改善，但仍明显为负。
- `n = 10` 和 `n = 100` 仍基本一致，说明主导 LP 的依旧是少数关键 path。
- 在 `max_skew = 50 ps` 的约束下，当前 useful skew 仍更像 post-placement diagnosis / upper-bound 工具。

### 8.5 Checkpoint 长跑的实际停止行为

当前还验证了一个关键现象：

- 即使把 global placement iteration 上限设为 `1000`
- DREAMPlace 也可能因为内部 stop criterion 提前停止

一次实际试跑中，`iter=1000` 的长跑在 `iteration 764` 就提前结束了。

这意味着：

- 不能默认认为一定能拿到 `iter=1000` checkpoint
- 更合理的 checkpoint 采样点应放在自动停止之前，例如：
  - `100, 300, 500, 700`

因此当前后续实验更适合继续按上述迭代点采样，而不是只盯 `1000` 这一个终点。

### 8.6 `iter=764` 提前停止时的收敛指标

这次 `iter=1000` 长跑并不是异常退出，而是命中了 DREAMPlace 自身的 global placement 停止判定。

当前 `superblue1.json` 中：

- `stop_overflow = 0.1`
- `target_density = 1.0`

`dreamplace/NonLinearPlace.py` 的 `Lgamma_stop_criterion()` 当前逻辑是：

- `Lgamma_step > 100`
- 且满足以下任一条件就停止：
  - `overflow < stop_overflow` 且 `hpwl > prev_hpwl`
  - 或 `max_density < target_density`

服务器日志中，停止前几步指标如下：

| iteration | HPWL | overflow | max_density |
| --- | ---: | ---: | ---: |
| 760 | `5.246124E+09` | `1.015813E-01` | `1.120E+01` |
| 761 | `5.238929E+09` | `1.013798E-01` | `1.205E+01` |
| 762 | `5.232863E+09` | `1.004599E-01` | `1.106E+01` |
| 763 | `5.240048E+09` | `1.000599E-01` | `1.212E+01` |
| 764 | `5.243197E+09` | `9.924360E-02` | `1.115E+01` |

因此这次真正触发停止的是：

- `overflow` 在 `iter=764` 首次降到 `0.1` 以下
- 同时 `HPWL(764) > HPWL(763)`

并不是 `max_density < target_density`，因为这里 `max_density` 仍远大于 `1.0`。

### 8.7 目标周期与 useful skew sweep 补充结果

当前 benchmark `superblue1` 的时钟约束来自：

- `benchmarks/iccad2015.ot/superblue1/superblue1.sdc`

其中：

- `create_clock -name mclk -period 9500.0 [get_ports iccad_clk]`

所以当前目标时钟周期为：

- `clock_period = 9500 ps = 9.5 ns`
- `clock_freq ~= 105.26 MHz`

为了评估 `max_skew` 是否过于保守，当前又在服务器上对多个 checkpoint 做了 sweep：

- `n = 1, 10, 100`
- `max_skew = 50, 100, 250, 500, 1000, 2000 ps`
- 另测了 `max_skew = None`（无上限）

注意：当前 LP 中 `max_skew` 的含义是每个寄存器单独满足：

- `s_i in [-max_skew, +max_skew]`

因此两寄存器间理论最大相对 skew span 可到：

- `2 * max_skew`

#### `iter=300`, `n=100`

- baseline: `setup.worst_slack = -17527.508 ps`

| `max_skew` | `skew_margin` |
| --- | ---: |
| `50 ps` | `-17427.508 ps` |
| `100 ps` | `-17368.793 ps` |
| `250 ps` | `-17218.793 ps` |
| `500 ps` | `-16968.793 ps` |
| `1000 ps` | `-16468.793 ps` |
| `2000 ps` | `-15468.793 ps` |
| `None` | unbounded / solver failed |

#### `iter=500`, `n=100`

- baseline: `setup.worst_slack = -24109.430 ps`

| `max_skew` | `skew_margin` |
| --- | ---: |
| `50 ps` | `-24009.430 ps` |
| `100 ps` | `-23922.561 ps` |
| `250 ps` | `-23772.561 ps` |
| `500 ps` | `-23522.561 ps` |
| `1000 ps` | `-23022.561 ps` |
| `2000 ps` | `-22022.561 ps` |
| `None` | unbounded / solver failed |

#### `iter=700`, `n=100`

- baseline: `setup.worst_slack = -19008.779 ps`

| `max_skew` | `skew_margin` |
| --- | ---: |
| `50 ps` | `-18958.779 ps` |
| `100 ps` | `-18908.779 ps` |
| `250 ps` | `-18758.779 ps` |
| `500 ps` | `-18508.779 ps` |
| `1000 ps` | `-18008.779 ps` |
| `2000 ps` | `-17008.779 ps` |
| `None` | unbounded / solver failed |

当前观察：

- `max_skew = 50 ps` 确实偏保守，但并不是当前 useful skew 收效不大的唯一原因。
- 即使放宽到 `2000 ps`，对当前 `17~24 ns` 量级的 setup 违例也仍然只能提供有限改善。
- 目前 `n` 从 `10` 增到 `100` 的影响仍明显小于 `max_skew` 的影响，说明 LP 主要仍被少量关键 path 主导。
- 在当前 LP 形式下，完全去掉 `max_skew` 上限会使问题无界或数值不稳定，因此不能直接把 `None` 解释为真实可实现的 ideal skew 上界。

### 8.8 服务器整理后复跑确认

当前已把服务器工作目录统一为：

- `/home/shengyuanjia/SkewPlace`

并清理掉旧目录 `/home/shengyuanjia/DREAMPlace` 以及家目录下的 tar/patch/tmp 产物。

一个重要环境注意事项是：

- 服务器 Docker 镜像里的 OpenTimer 是按 `/workspace/DREAMPlace` 编译的
- 因此容器运行时必须挂载：
  - `/home/shengyuanjia/SkewPlace:/workspace/DREAMPlace`
- 如果改挂到 `/DREAMPlace`，OpenTimer 会找不到内置 `ot/sdc` 资源目录，日志会出现：
  - `sdc home "" doesn't exist`
  - `added 0 sdc commands`

修正挂载路径后，`superblue1` 的 SDC 已重新确认正常读入：

- `loading sdc "/workspace/DREAMPlace/benchmarks/iccad2015.ot/superblue1/superblue1.sdc" ...`
- `added 13057 sdc commands`

在这个正确环境下，重新对已有 checkpoint 做了 sweep 确认。

#### `iter=100`

- `clock_period = 9500 ps`
- baseline: `setup.worst_slack = -21983.010 ps`

| `n` | `max_skew=50 ps` | `max_skew=500 ps` | `max_skew=2000 ps` |
| --- | ---: | ---: | ---: |
| `1` | `-21933.016 ps` | `-21483.016 ps` | `-19983.016 ps` |
| `10` | `-21883.010 ps` | `-20983.010 ps` | `-17983.010 ps` |
| `100` | `-21883.010 ps` | `-20983.010 ps` | `-17983.010 ps` |

#### `iter=300`

- `clock_period = 9500 ps`
- baseline: `setup.worst_slack = -17527.508 ps`

| `n` | `max_skew=50 ps` | `max_skew=500 ps` | `max_skew=2000 ps` |
| --- | ---: | ---: | ---: |
| `1` | `-17477.516 ps` | `-17027.516 ps` | `-15527.516 ps` |
| `10` | `-17427.508 ps` | `-16968.793 ps` | `-15468.793 ps` |
| `100` | `-17427.508 ps` | `-16968.793 ps` | `-15468.793 ps` |

#### `iter=500`

- `clock_period = 9500 ps`
- baseline: `setup.worst_slack = -24109.430 ps`

| `n` | `max_skew=50 ps` | `max_skew=500 ps` | `max_skew=2000 ps` |
| --- | ---: | ---: | ---: |
| `1` | `-24059.441 ps` | `-23609.441 ps` | `-22109.441 ps` |
| `10` | `-24059.430 ps` | `-23609.430 ps` | `-22109.430 ps` |
| `100` | `-24009.430 ps` | `-23522.561 ps` | `-22022.561 ps` |

#### `iter=700`

- `clock_period = 9500 ps`
- baseline: `setup.worst_slack = -19008.779 ps`

| `n` | `max_skew=50 ps` | `max_skew=500 ps` | `max_skew=2000 ps` |
| --- | ---: | ---: | ---: |
| `1` | `-18958.777 ps` | `-18508.777 ps` | `-17008.777 ps` |
| `10` | `-18958.779 ps` | `-18508.779 ps` | `-17008.779 ps` |
| `100` | `-18958.779 ps` | `-18508.779 ps` | `-17008.779 ps` |

这些复跑结果与之前整理出的趋势一致：

- `n` 的影响通常仍小于 `max_skew`
- `500 ps` 和 `2000 ps` 能继续改善 margin，但仍远小于当前主 setup 违例量级

### 8.9 GP 内 timing-feedback 复跑结果（`iter=700`）

在修正服务器容器挂载路径后，又重新做了 `iter=700` 的 GP 内 timing-feedback 对照：

- baseline：`useful_skew_weighting_flag = 0`
- skew-aware：`useful_skew_weighting_flag = 1, n = 100, max_skew = 500 ps`
- skew-aware：`useful_skew_weighting_flag = 1, n = 100, max_skew = 2000 ps`

共同实验条件：

- `clock_period = 9500 ps = 9.5 ns`
- `clock_freq ~= 105.26 MHz`
- timing feedback 触发轮次仍为：
  - `508, 523, 538, 553, 568, 583, 598, 613, 628, 643, 658, 673, 688`
  - 共 `13` 次

结果摘要：

| 模式 | runtime | 最后一次 `tns` | 最后一次 `wns` | 最后一次 useful-skew margin |
| --- | ---: | ---: | ---: | ---: |
| baseline | `344.888 s` | `-99.145` | `-20.712` | N/A |
| skew-aware `500 ps` | `520.091 s` | `-192.526` | `-22.254` | `-21254.125 ps` |
| skew-aware `2000 ps` | `500.722 s` | `-187.312` | `-22.245` | `-18903.418 ps` |

最后一次 skew-aware weighting 统计：

- `max_skew = 500 ps`
  - `num_registers = 73`
  - `num_edges = 64`
  - `raw_setup_wns = -22254.125 ps`
  - `adjusted_setup_wns = -21254.125 ps`
  - `affected_nets = 200`
- `max_skew = 2000 ps`
  - `num_registers = 72`
  - `num_edges = 62`
  - `raw_setup_wns = -22244.715 ps`
  - `adjusted_setup_wns = -18903.418 ps`
  - `affected_nets = 200`

当前观察：

- skew-aware 分支在真实大例子上仍然稳定触发了 `13` 次 feedback，不再只是 `iter=520` 时的一次性接通验证。
- 更宽松的 `max_skew` 会明显改善“adjusted setup WNS”这个内部 weighting 信号。
- 但到当前这版原型为止，最终 placement 的外部 `tns/wns` 还没有体现出收益，反而比 baseline 更差。
- 因此这版 skew-aware weighting 目前仍只能说明：
  - useful skew 信息已经成功进入 GP 内 net-weight update
  - 但现有 weighting 映射方式还没有转化成最终时序收益

## 9. 服务器 Git 状态说明

服务器上的 `/home/shengyuanjia/DREAMPlace-useful-skew` 目前不适合直接做 `git pull` / `git push`，原因不是账户本身，而是：

- 当前分支 `master` 落后 `origin/master` 3 个提交。
- 同时工作树里已经有大量未提交修改和未跟踪文件。
- 服务器用户的 `~/.gitconfig` 还配置了失效的本地代理：
  - `http.proxy=http://127.0.0.1:17897`
  - `https.proxy=http://127.0.0.1:17897`

因此当前最稳妥的流程应该是：

- 把本地仓库作为主 git 记录来源
- 服务器隔离副本只负责跑实验
- 在本地 rebase 到 `origin/master` 后再推送
  - 新增 `report_timing_paths(timer, n)`
  - 作用：从 OpenTimer `ot::Path` 导出更细粒度路径字典
- `dreamplace/ops/timing/timing.py`
  - 暴露 `export_reg2reg_timing_graph`
  - 暴露 `solve_useful_skew_from_timer`
- `dreamplace/ops/timing/useful_skew.py`
  - `build_reg2reg_timing_graph(paths, include_paths=False)`
  - `export_reg2reg_timing_graph(timer, n=None, include_paths=False)`
  - `solve_useful_skew(graph, max_skew=None)`
  - `solve_useful_skew_from_timer(timer, n=None, include_paths=False, max_skew=None)`
- `dreamplace/Timer.py`
  - 新增 `report_timing_paths`
  - 新增 `export_reg2reg_timing_graph`
  - 新增 `solve_useful_skew`

安装侧运行副本同步位置：

- `install/dreamplace/ops/timing/timing.py`
- `install/dreamplace/ops/timing/useful_skew.py`
- `install/dreamplace/Timer.py`

### 5.1.1 当前阶段摘要

当前这轮 useful skew 第一版已经落下去的最小闭环如下：

- placement 后调用 OpenTimer 更新当前布局对应的 RC / timing 状态
- 通过 `Timer.report_timing_paths()` 导出 detailed timing paths
- 通过 `Timer.export_reg2reg_timing_graph()` 抽取 `launch_register -> capture_register` 图
- 通过 `Timer.solve_useful_skew()` 在该图上做最小 LP 原型求解

当前对既有模块的主要改动点：

- `dreamplace/BasicPlace.py`
  - 向 timing op 额外传 `node_names`，用于 `gate:pin` fallback
- `dreamplace/Timer.py`
  - 新增 path 导出 / reg-to-reg graph / useful skew 封装接口
- `dreamplace/ops/timing/src/timing_cpp.cpp`
  - RC tree 建树时支持 `raw pin` 到 `node_name:pin_name` 的 fallback 对齐
- `dreamplace/ops/timing/src/timing_pybind.cpp`
  - 新增 detailed path 导出
- `dreamplace/ops/timing/useful_skew.py`
  - 实现 reg-to-reg 图抽取与 LP 原型
- `dreamplace/ops/electric_potential/electric_potential.py`
  - 修复 `fixed_density_map()` 参数类型问题
- `dreamplace/ops/fence_region/fence_region.py`
  - 修复空 / 退化 polygon 导致的 `NaN bounds`

当前最常用命令：

- 重编 timing 扩展：

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/DREAMPlace \
  -e USER="${USER:-dwindz}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "/home/dwindz/workspace/EDA/DREAMPlace":/workspace/DREAMPlace \
  -w /workspace/DREAMPlace/build-gpu-sm120 \
  dreamplace:cuda-ready \
  bash -lc "cmake --build . --target timing_cpp install -- -j4"
```

- 运行最小 timing/useful skew 验证：

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/DREAMPlace \
  -e USER="${USER:-dwindz}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "/home/dwindz/workspace/EDA/DREAMPlace":/workspace/DREAMPlace \
  -w /workspace/DREAMPlace \
  dreamplace:cuda-ready \
  bash -lc "python test/minitimer/run_minitimer_useful_skew.py test/minitimer/minitimer.json"
```

- 当前 `minitimer` 最新有效结果：
  - `num_registers = 2`
  - `num_edges = 1`
  - `reg0 -> reg1`
  - 同一条 `reg0 -> reg1` 边同时带 setup/hold
  - `setup_delay ~= 100.007 ps`, `setup_constraint = 25 ps`, `setup_slack ~= 55.006 ps`
  - `hold_delay ~= 46.007 ps`, `hold_constraint = 4 ps`, `hold_slack ~= 41.994 ps`
  - `skew_success = true`
  - `skew_margin ~= 48.5 ps`
  - `reg0 = 0`, `reg1 ~= -6.506 ps`

- 当前 `minitimer` 结构已回到单向两级寄存器：
  - `in1 -> reg0 -> comb0 -> reg1 -> out1`
  - 通过抬高 `in1` 的 min/max input delay，避免 `PI -> reg0` 抢走最差 hold
  - 去掉 `out1` 的 output delay，避免 `PO` endpoint 抢走最差 min path

- 当前 OpenTimer 小例子还有一个报表层现象：
  - `report_timing_paths_by_split(split, n=1)` 能正常返回 top-1 path
  - 但在这个最小 case 上，`n > 1` 时可能直接返回空列表
  - 因此在 `Timer.report_test_paths_by_split()` 上加了 top-1 fallback，保证 Python 侧 reg-to-reg graph 导出稳定

### 5.2 当前导出接口格式

`report_timing_paths` 的每条 path 当前至少包含：

- `slack`
- `path_delay`
- `required_time`
- `analysis_type`
  - `max` 表示 setup 侧
  - `min` 表示 hold 侧
- `endpoint_transition`
- `start_pin_name`
- `end_pin_name`
- `capture_pin_name`
- `capture_gate_name`
- `related_pin_name`
- `test_constraint`
- `points`

其中 `points` 中每个点当前至少包含：

- `pin_name`
- `gate_name`
- `cell_name`
- `net_name`
- `arrival_time`
- `transition`
- `is_primary_input`
- `is_primary_output`
- `is_datapath_source`

### 5.3 当前 reg-to-reg 图抽取策略

第一版图抽取不是遍历所有寄存器对，而是只从 OpenTimer 报出来的真实 timing path 中收缩边：

- 只处理 `endpoint_type == test` 的路径
- `analysis_type == max` 的路径用于 setup 边信息
- `analysis_type == min` 的路径用于 hold 边信息
- launch 端优先取 path 中第一个 `is_datapath_source == True` 的点
- capture 端取 `Test::constrained_pin()` 对应的寄存器 data pin
- 对相同 `launch_register -> capture_register` 的多条路径做聚合：
  - setup 取最大 `setup_delay`
  - hold 取最小 `hold_delay`

因此当前输出图中每条边会同时尽量带：

- `setup_delay`
- `setup_constraint`
- `hold_delay`
- `hold_constraint`
- 以及对应 worst-case path 的若干统计字段

### 5.4 当前最小 useful skew LP 原型

当前原型使用：

- `scipy.optimize.linprog(method="highs")`

变量：

- 每个寄存器一个 skew 变量 `s_i`
- 一个统一 margin 变量 `m`

约束写法：

- setup:

```text
s_j - s_i >= dmax_ij + setup_ij + m
```

- hold:

```text
s_j - s_i <= dmin_ij - hold_ij - m
```

- 锚点：

```text
s_ref = 0
```

- 可选范围：

```text
-Smax <= s_i <= Smax
```

目标：

```text
maximize m
```

这版还是 prototype，主要用于回答：

- 当前 placement 结果上是否存在有价值的 skew 调度空间
- setup/hold 同时考虑时，理论最优统一 margin 是多少
- 每个寄存器会被推到什么 skew 值

## 6. benchmark 怎么跑

### 6.1 README 中的官方运行方式

README 给出的标准方式是：

```bash
cd <installation directory>
python dreamplace/Placer.py test/ispd2005/adaptec1.json
```

但当前这份本地工作区的 `install/` 目录并没有把 benchmark 数据一起复制进去，因此直接 `cd install` 再跑会因为相对路径问题失败。

### 6.2 当前这份工作区中可用的本地运行方式

当前本地可行方式是：

1. 从仓库根目录进入 Docker

```bash
./docker.sh --cuda
```

2. 在仓库根目录下运行安装产物中的 Placer，并使用根目录下的 benchmark / test 路径：

```bash
python install/dreamplace/Placer.py test/ispd2005/adaptec1.json
```

如果不想进入交互 shell，也可以直接一条命令运行：

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/DREAMPlace \
  -e USER="${USER:-dwindz}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$(pwd)":/workspace/DREAMPlace \
  -w /workspace/DREAMPlace \
  dreamplace:cuda-ready \
  bash -lc "python install/dreamplace/Placer.py test/ispd2005/adaptec1.json"
```

### 6.3 当前本地有哪些 benchmark

当前本地仓库中已经存在的较完整 benchmark 套件：

- `benchmarks/ispd2005`：8 个设计
- `benchmarks/ispd2015`：20 个设计

它们属于真正的性能 benchmark，不是单纯 smoke test。

其中最直接能跑的基线例子：

- `test/ispd2005/adaptec1.json`

更大的真实 benchmark 后续可以考虑：

- `test/ispd2005/bigblue*.json`
- `test/ispd2015/lefdef/*.json`

### 6.4 哪些 timing benchmark 目前缺失

当前仓库里的 timing benchmark 配置文件是有的：

- `test/iccad2015.ot/*.json`：OpenTimer 版本
- `test/iccad2015.hs/*.json`：HeteroSTA 版本

但对应 benchmark 数据当前**不在本地仓库里**，需要额外下载。

下载提示见：

- `benchmarks/iccad2015.ot.md`
- `benchmarks/iccad2015.hs.md`

这也解释了为什么当前直接跑：

```bash
python install/dreamplace/Placer.py test/iccad2015.ot/simple.json
```

会因为找不到 `benchmarks/iccad2015.ot/simple/simple.lef` 而失败。

### 6.5 关于你之前提到的“那个跑很快的 benchmark”

目前看，真正偏“简单例子 / 环境检查”的更像是：

- `test/iccad2015.ot/simple.json`
- `test/iccad2015.hs/simple.json`
- `test/simple.json`

但在当前工作区里，这几个简单例子所依赖的数据并不完整，因此并不适合作为当前主 baseline。

真正有代表性的性能测试仍然是：

- ISPD2005
- ISPD2015
- 以及下载后可跑的 ICCAD2015 timing benchmarks

## 7. 当前本地环境状态

本地环境当前已确认：

- GPU：`NVIDIA GeForce RTX 5070 Ti`，显存约 `12GB`
- Docker 镜像：`dreamplace:cuda-ready`
- 容器内 Python：`3.10.12`
- 容器内 PyTorch：`2.11.0+cu128`
- CUDA available：`True`

当前判断：

- 本地可以先跑 ISPD2005 基线
- 暂时不需要切到导师服务器
- 如果后续更大 benchmark 或 timing benchmark 需要更多资源，再考虑切服务器

## 8. 当前 baseline 运行记录

### 8.1 已成功运行

当前已完成的 baseline 包括 `ISPD2005` 全套 8 个 benchmark，以及 1 个 `ISPD2015` `lef/def` benchmark。

常用运行命令示例：

```bash
python install/dreamplace/Placer.py test/ispd2005/adaptec1.json
```

运行结果摘要：

| benchmark | 类型 | final wHPWL | non-linear placement | detailed placement | total placement |
| --- | --- | ---: | ---: | ---: | ---: |
| `adaptec1` | ISPD2005 | `7.284164E+07` | `12.21s` | `2.681s` | `16.017s` |
| `adaptec2` | ISPD2005 | `8.192209E+07` | `11.51s` | `3.112s` | `15.591s` |
| `adaptec3` | ISPD2005 | `1.929166E+08` | `16.45s` | `5.629s` | `23.874s` |
| `adaptec4` | ISPD2005 | `1.735007E+08` | `16.10s` | `4.805s` | `24.585s` |
| `bigblue1` | ISPD2005 | `8.926234E+07` | `11.66s` | `2.786s` | `16.846s` |
| `bigblue2` | ISPD2005 | `1.369388E+08` | `16.14s` | `4.787s` | `24.882s` |
| `bigblue3` | ISPD2005 | `3.039967E+08` | `26.92s` | `7.831s` | `44.328s` |
| `bigblue4` | ISPD2005 | `7.426307E+08` | `45.08s` | `12.158s` | `86.630s` |
| `mgc_matrix_mult_b` | ISPD2015 | `1.576422E+07` | `24.02s` | `1.360s` | `25.921s` |

- `ISPD2005` 输出为 Bookshelf `.gp.pl`
- `ISPD2015` `mgc_matrix_mult_b` 输出为 DEF：`results/design/design.gp.def`
- `mgc_matrix_mult_b.json` 虽然配置了 `thirdparty/ntuplace_4dr`，但当前本地缺少外部 detailed placer，因此日志中会提示：`External detailed placement engine thirdparty/ntuplace_4dr or aux file NOT found`；不过 DREAMPlace 自带的 legalization 和 detailed placement 仍已完成

说明：

- 这些都不是“纯 smoke test”，而是真实 benchmark。
- 对于当前 12GB 显存的本地机器，`ISPD2005` 全套和至少一个 `ISPD2015` 设计已经可以稳定运行。
- `adaptec1` / `bigblue1` 这一档能在 `15s~17s` 左右跑完，`bigblue4` 则已经到 `86s` 量级，更适合后续做大一点的 baseline 对照。

### 8.2 当前未成功运行的 benchmark

1. `test/iccad2015.ot/simple.json`

- 原因：缺少 `benchmarks/iccad2015.ot/simple/*` 数据文件

2. `test/simple.json`

- 原因：当前工作区缺少 `benchmarks/simple/simple.aux` 等数据

3. `test/ispd2015/lefdef/mgc_matrix_mult_b.json`

- 之前一度失败，但现已修复并跑通
- 暴露并修复了两个 fence region 相关问题：
  - `dreamplace/ops/electric_potential/electric_potential.py`
    - `fixed_density_map()` 的 `num_fixed_impacted_bins_x/y` 需要传原生 `int`，不能直接传 0 维 tensor
  - `dreamplace/ops/fence_region/fence_region.py`
    - `slice_non_fence_region()` 在处理空/退化 polygon 时可能把 `NaN bounds` 混进 `virtual_macro_fence_region`
- 这两个修复已经同步到 `dreamplace/` 和 `install/dreamplace/`

## 9. 建议的下一步

### 近一步

- 以 `adaptec1` 作为小 baseline，以 `bigblue4` 作为大一点的 `ISPD2005` baseline
- 以 `mgc_matrix_mult_b` 作为第一条可用的 `ISPD2015` baseline
- 如果后续要做 skew 实验的第一版对照，建议优先选：
  - `adaptec1`：便于快速迭代
  - `bigblue4`：观察规模效应
  - `mgc_matrix_mult_b`：验证 fence region / modern `lef/def` 流程下也能接入

### useful skew 第一版实现顺序

1. 确认 timer 后端怎样导出 reg-to-reg timing graph
2. 定义 solver 输入格式
3. 先实现 ideal skew LP
4. 再加空间一致性约束版本
5. 做 skew 前后 WNS / TNS 对比

### 是否需要 CTS / iCTS

当前第一版不需要。

当前阶段先做：

- ideal skew
- constrained skew

等确认 useful skew 本身有价值后，再讨论：

- 如何接真实 CTS
- 如何验证 skew target 的物理可实现性
