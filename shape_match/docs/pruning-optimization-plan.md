# 第二阶段：搜索剪枝与速度优化计划

## 1. 文档目的

本文档是第一阶段高精度 exhaustive 参考路径之后的第二阶段实施计划。

第一阶段冻结了正确性参考路径和已有性能数据。本阶段只优化搜索过程，目标是：

- 减少需要完整评分的姿态数量；
- 减少每个姿态实际计算的模型点数量；
- 保持召回、误检控制、姿态结果和评分语义与第一阶段基线一致；
- 所有剪枝都能给出可审计的安全拒绝依据。

本阶段不得通过降低阈值、改变模型点分母、删除候选或修改最终评分语义来换取速度。

## 2. 第一阶段冻结 baseline

以下数据来自第一阶段已有的本地 release benchmark，作为第二阶段的时间 baseline。
后续所有剪枝优化必须同时报告 baseline 与 optimized 的 p50、p95、姿态数和模型点评估数。

### 2.1 环境

- CPU：Intel Xeon Platinum 8368Q
- 编译器：GCC 13.3
- OpenCV：4.10
- matcher threads：4
- 统计口径：p50；正式比较至少运行 30 次

### 2.2 冻结耗时

| 场景 | 第一阶段 baseline p50 | 第一阶段记录的优化参考值 | 第二阶段目标 |
| --- | ---: | ---: | ---: |
| Real sample 1，61 angles × 9 scales | 564 ms | 283 ms | 在不改变结果的前提下继续降低完整评分量 |
| Synthetic continuous refinement | 1084 ms | 47 ms | 保留全部安全候选，不依赖固定 Top-K |
| 640×400 clutter，25 transforms | 51.4 ms | 33.2 ms | 降低 p50，同时控制 p95 |

### 2.3 尺度搜索参考数据

第一阶段还记录了尺度搜索场景，可作为联合角度/尺度剪枝的补充 baseline：

| 场景 | 变换数量 | baseline 口径 | p50 |
| --- | ---: | --- | ---: |
| scale | 5 scales | prepared filtered | 9.643 ms |
| angle-scale | 5 angles × 5 scales | prepared filtered | 52.360 ms |

这些数字只代表上述机器、编译选项和固定数据，不构成跨平台性能承诺。

## 3. 正确性边界

第二阶段使用两套结果进行对比：

1. **第一阶段 exhaustive baseline**：关闭新增剪枝，完整遍历并完整评分；
2. **optimized path**：开启安全剪枝后的结果。

两者必须逐候选比较：

- 不得漏掉 baseline 接受的真实目标；
- 不得新增 baseline 没有的接受结果；
- score、位置、角度、尺度在定义容差内一致；
- 最终 level 0 仍使用完整高精度模型和完整评分；
- 任何无法证明安全的候选必须保留。

第一阶段当前亚像素定位误差仍是独立问题。第二阶段不得用剪枝掩盖或改变该误差；`1/30` 像素验收继续由高精度模型和评分基线负责。

## 4. 剪枝总原则

所有剪枝只能产生“确定拒绝”，不能产生“经验拒绝”。每次安全拒绝必须记录：

- 所在金字塔层；
- 姿态索引；
- 剪枝类型；
- 已评估模型点数；
- 当前累计得分；
- 理论 score upper bound；
- visible upper bound；
- 是否经过抽样审计复核。

默认配置保持剪枝关闭，先用审计模式验证，再逐项开启。

## 5. 实施阶段

### Phase 2A：基准与审计 instrumentation

新增 `SearchStats` 字段：

- 总姿态数；
- 完整评分姿态数；
- 安全拒绝姿态数；
- 按 AABB/ROI 拒绝数；
- 按边缘覆盖拒绝数；
- 按 score upper bound 拒绝数；
- 按 visible upper bound 拒绝数；
- 已评估模型点总数；
- 被省略的模型点评估数；
- p50/p95 所需的阶段耗时。

新增参数建议：

```cpp
bool enable_safe_pruning = false;
double pruning_audit_rate = 0.0;
```

当 `pruning_audit_rate > 0` 时，被剪姿态按固定比例重新完整评分，验证上界实现没有误拒绝。

### Phase 2B：姿态级安全预筛选

按以下顺序实现：

1. 变换模型 AABB 与图像边界检查；
2. 计算合法平移索引范围，跳过必然越界的平移；
3. 使用 edge integral 判断模型包围区域是否完全无场景边缘；
4. 使用连续距离场判断是否存在满足最小可见率的可能支持；
5. 对边缘支持不足的姿态只在能够证明上界不足时拒绝。

该阶段主要减少进入完整模型评分的姿态数，不改变模型点评分顺序和评分结果。

### Phase 2C：单姿态安全提前终止

模型点可以重新排序，但不得删除。排序优先级建议为：

1. 高权重点；
2. 高方向区分度点；
3. 空间上分散的点；
4. 不同轮廓区域的代表点。

对已处理点维护：

```text
S_done
W_done
W_remaining
W_total
valid_done
remaining_point_count
```

单极性模式的安全上界：

```text
score_upper_bound = (S_done + W_remaining) / W_total
visible_upper_bound = valid_done + remaining_point_count
```

当：

```text
score_upper_bound < current_level_threshold
```

或：

```text
visible_upper_bound < required_visible_count
```

时可以确定拒绝当前姿态。

`GlobalEither` 模式必须分别维护同极性和反极性累计得分，分别计算两个上界后取较大值。

### Phase 2D：粗到细候选传播

将 exhaustive 路径改为真正的多层安全搜索：

```text
最粗层完整角度/尺度遍历
    ↓
低阈值候选记录
    ↓
宽窗口传播到细层
    ↓
level 0 完整复核
```

允许：

- 粗层使用宽松 `level_min_score`；
- 继承多个候选中心；
- 根据金字塔误差推导平移窗口；
- 无法证明窗口安全时回退完整 ROI。

禁止：

- 只继承上一层最佳角度或尺度；
- 粗层 Top-K；
- 固定比例保留候选；
- 未证明安全时缩小角度、尺度或平移窗口；
- 使用峰值聚类删除可能的真实目标。

### Phase 2E：块级上界和分支定界

在前述安全剪枝通过回归后，再增加平移块级别的上界：

- 每个固定角度/尺度组合划分平移块；
- 对块内模型点支持计算理论最高分；
- 若块上界低于当前层阈值，则跳过整个块；
- 块内仍保留所有可通过的候选；
- 最终 level 0 不允许使用部分评分直接接受。

只有在明确的返回数量策略下，才允许使用第 K 个候选作为分支定界下界；默认 `num_matches=0` 仍不能依赖 Top-K。

## 6. 代码落点

建议新增：

```text
include/openshape/matcher/pruning.hpp
src/matcher/pruning.cpp
tests/exhaustive_pruning_tests.cpp
tests/pruning_equivalence_tests.cpp
benchmark/pruning_benchmark.cpp
```

主要调整：

```text
include/openshape/core/types.hpp
    剪枝参数、审计参数和统计字段

src/matcher/exhaustive_matcher.cpp
    姿态级预筛选、点级上界、粗到细候选传播

src/matcher/matcher.cpp
    复用统一的 score upper bound 和 SearchStats 合并逻辑
```

## 7. 回归数据集

至少覆盖：

- 现有 real sample；
- 现有 synthetic continuous refinement；
- 640×400 clutter；
- 任意小数平移、旋转和尺度；
- 低对比度、模糊和噪声；
- 多目标和重叠目标；
- 空白图、噪声图、单条强边缘；
- 局部相似轮廓、重复纹理和反极性；
- 目标位于 ROI/图像边界的场景。

每个场景保存：

- 图像和模型参数；
- 搜索域；
- baseline 接受候选；
- optimized 接受候选；
- score/位置/角度/尺度差异；
- 剪枝原因统计。

## 8. 验收门槛

正确性门槛：

- baseline 与 optimized 召回结果一致；
- 误检数不增加；
- 被剪姿态的抽样完整复核不得发现误拒绝；
- 不同线程数、分块方式和编译优化级别结果稳定。

性能门槛：

- `score_point_evaluations` 减少至少 50%；
- 完整姿态评分次数减少至少 30%；
- Real sample 1 p50 低于 564 ms；
- Synthetic continuous refinement p50 低于 1084 ms；
- 640×400 clutter p50 低于 51.4 ms；
- p95 不因候选传播和审计机制明显恶化。

任何性能提升都必须同时附带 baseline/optimized 对照表，不能只报告优化后的单次耗时。

## 9. 交付顺序

1. 冻结当前 baseline 输出和耗时；
2. 完成统计和剪枝审计接口；
3. 实现姿态级 AABB/边缘必要条件；
4. 实现点级 score/visible 安全上界；
5. 实现多层候选传播；
6. 实现块级上界；
7. 运行逐候选等价性回归；
8. 生成第二阶段性能报告；
9. 只有验收通过后，才允许打开默认安全剪枝。
