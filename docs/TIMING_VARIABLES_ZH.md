# Timing Variables

这份文档记录当前 `minitimer` 和后续真实 benchmark 输出里常见的 timing / useful skew 字段含义。

## Path Fields

- `launch_register`
  - 发数寄存器。
- `capture_register`
  - 收数寄存器。
- `launch_pin_name`
  - 当前实现认定的 launch 端 pin，一般是发数寄存器的 `Q`。
- `capture_pin_name`
  - capture 端 pin，一般是收数寄存器的 `D`。
- `related_pin_name`
  - 约束关联的时钟 pin，一般是收数寄存器的 `CK`。
- `analysis_type`
  - `max` 表示 setup 分析。
  - `min` 表示 hold 分析。
- `endpoint_transition`
  - endpoint 上的边沿类型，通常是 `rise` 或 `fall`。
- `path_delay`
  - 当前这条路径的数据路径延迟。
- `required_time`
  - 当前检查下允许的数据到达时刻。
- `arrival_time`
  - 当前路径实际到达 capture 端的时刻。
- `slack`
  - 当前路径在对应 setup/hold 检查下的余量。
  - 大于 `0` 表示满足。
  - 小于 `0` 表示违例。

## Setup Fields

- `setup_delay`
  - 这条 `launch_register -> capture_register` 边在 `max` 分析下的数据路径延迟。
- `setup_constraint`
  - setup 检查要求值，通常来自 capture flop 的 library setup time。
- `setup_arrival_time`
  - setup 分析下，数据实际到达 capture pin 的时刻。
- `setup_required_time`
  - setup 分析下，数据最晚允许到达 capture pin 的时刻。
- `setup_slack`
  - setup 余量。
  - 直观上就是“setup 还剩多少时间”。
  - 近似可理解为：`setup_required_time - setup_arrival_time`。

## Hold Fields

- `hold_delay`
  - 这条 `launch_register -> capture_register` 边在 `min` 分析下的最短数据路径延迟。
- `hold_constraint`
  - hold 检查要求值，通常来自 capture flop 的 library hold time。
- `hold_arrival_time`
  - hold 分析下，数据实际到达 capture pin 的时刻。
- `hold_required_time`
  - hold 分析下，数据不能早于这个时刻到达。
- `hold_slack`
  - hold 余量。
  - 直观上就是“数据有没有到得太早”。
  - 大于 `0` 表示 hold 满足，小于 `0` 表示 hold 违例。

## Graph Fields

- `num_registers`
  - 导出的 reg-to-reg timing graph 中涉及的寄存器数量。
- `num_edges`
  - 图中的寄存器到寄存器边数量。
- `num_paths`
  - 当前用于构图的原始 timing path 数量。
- `path_counts`
  - 每个 split 实际拿到的 path 数量。
  - 例如：
    - `path_counts["max"]` 是 setup 侧 path 数
    - `path_counts["min"]` 是 hold 侧 path 数

## Useful Skew LP Fields

- `skews`
  - LP 给每个寄存器求出的时钟偏移量。
  - 正值表示相对参考点更晚，负值表示更早。
- `margin`
  - 在 setup/hold 约束上同时额外保留的统一安全余量。
  - 当前原型会最大化这个量。
  - `margin > 0` 表示全部约束满足后还能再留出公共余量。
  - `margin < 0` 表示整体不可同时满足，至少还差这么多余量。
- `num_constraints`
  - LP 中最终纳入的约束数量。
- `success`
  - LP 求解是否成功。
- `status`
  - 求解器状态码。
- `message`
  - 求解器返回的信息。
- `max_skew`
  - 对单个寄存器 skew 施加的幅度限制；若为空表示不额外限制。

## Minitimer Example

当前 `minitimer` 的一条边是 `reg0 -> reg1`。

- `setup_delay ~= 100 ps`
  - 数据在 setup 分析下大约走了 `100ps`。
- `setup_constraint = 25 ps`
  - `reg1` 的 setup 检查要求约 `25ps`。
- `setup_slack ~= 55 ps`
  - 说明 setup 还比较宽松。
- `hold_delay ~= 46 ps`
  - 数据在 hold 分析下最快约 `46ps` 到达。
- `hold_constraint = 4 ps`
  - `reg1` 的 hold 检查要求约 `4ps`。
- `hold_slack ~= 42 ps`
  - 说明 hold 也满足，且仍有明显余量。
- `margin ~= 48.5 ps`
  - 说明 setup/hold 一起考虑并允许调整 clock skew 后，还能统一再留出约 `48.5ps` 的公共余量。
