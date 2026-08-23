# 剪枝阶段首批交付：Phase 2A / 2B / 2C

## 已完成范围

- `SearchParams::enable_safe_pruning`：默认 `false`，只影响 exhaustive 高精度路径；
- `SearchParams::pruning_audit_rate`：按姿态索引确定性抽样，范围 `[0, 1]`；
- 姿态级安全拒绝：
  - 变换后模型 AABB 完全位于图像外；
  - 扩张后的模型 AABB 内不存在任何 Canny 或亚像素边缘支持；
- 点级安全拒绝：
  - 固定完整模型权重分母的 score upper bound；
  - visible point count upper bound；
  - `GlobalEither` 分别累计同极性与反极性得分，再取二者最大上界；
- 审计失败保护：抽样完整评分若发现候选仍能达到阈值，则保留候选并增加
  `pruning_audit_failures`，不会继续执行错误拒绝；
- `SearchStats` 增加安全拒绝、AABB/边缘拒绝、遗漏点评估数、审计次数和审计失败数；
- 新增独立等价性测试和轻量 p50/p95 benchmark。

visible upper bound 已在通用 prunable scorer 中实现，但 exhaustive 初始候选扫描暂不传入
最终可见点门槛。原因是当前冻结 baseline 会先按低 score 收集局部峰值、再做最终可见率验收；
提前删除低可见率峰值可能改变邻域峰值归并顺序。后续需先把可见率验收安全地下推到 baseline
候选生成阶段，再在 exhaustive 搜索中启用该上界。

## 正确性验证

Release 构建：

```sh
cmake -S shape_match -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

结果：`5/5` 通过，包括新增 `openshape_pruning_equivalence_tests`。

新增测试覆盖：

- 四种 polarity；
- 多组整数/小数平移、角度和尺度；
- 完整评分与关闭剪枝的 prunable scorer 逐值一致；
- 每个 score upper bound 拒绝由完整评分复核；
- score 和 visible 两类证书；
- target scene 与 blank scene 的 baseline/optimized 结果一致；
- target scene 和 blank scene 的 100% 审计均为零误拒绝。

## 轻量性能数据

命令：

```sh
cmake -S shape_match -B build-benchmark -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DBUILD_BENCHMARK=ON
cmake --build build-benchmark -j4
./build-benchmark/shape_match_pruning_benchmark 30
```

环境：Intel Xeon Platinum 8368Q、GCC 13.3、OpenCV 4.10，单线程 exhaustive
合成场景，`96x96`，3 个角度，160 个模型点，共 27,648 个姿态。

| 指标 | baseline | optimized | 变化 |
| --- | ---: | ---: | ---: |
| p50 | 279.793 ms | 77.482 ms | 3.611x |
| p95 | 288.932 ms | 77.911 ms | 3.708x |
| 完整评分姿态 | 27,648 | 1 | -99.996% |
| 模型点评估 | 4,423,680 | 823,456 | -81.4% |
| 省略模型点评估 | 0 | 3,600,224 | +3,600,224 |
| 返回结果 | baseline | 与 baseline 完全一致 | equivalent=1 |

这是一组新增的微基准，用于验证机制和统计口径，不替代
`pruning-optimization-plan.md` 冻结的 real sample、continuous refinement 和 clutter 正式验收数据。

## 后续工作

1. 在三组冻结场景上保存 baseline/optimized 的逐候选结果和 30 次 p50/p95；
2. 安全地下推 visible 门槛后，在 exhaustive 路径启用 visible upper bound；
3. 对模型点做高权重、方向区分度和空间分散排序，并验证浮点结果容差；
4. 实现多层候选传播与回退完整 ROI；
5. 最后再评估块级上界和是否允许默认开启安全剪枝。

## 第二批姿态域与上界粒度优化

在首批安全剪枝基础上继续完成：

- 对每个变换计算保守的合法平移范围；
- 将边缘支持图按变换 AABB 反向扩张为“可能中心区域”；
- 相同整数 AABB 的变换复用平移区间；
- 生产路径只枚举安全平移区间，审计模式仍完整枚举；
- 小于 1024 个平移点的 ROI 不构建稀疏姿态域，避免预处理成本超过收益；
- score upper bound 从每 8 点检查一次调整为每点检查一次；
- 非均匀权重模型允许高权重点优先，均匀权重模型保持原始缓存友好顺序；
- 增加 theoretical/domain-enumerated/domain-skipped 姿态统计。

### 96x96 全图合成场景，30 次

| 指标 | 首批 optimized | 第二批 optimized | 变化 |
| --- | ---: | ---: | ---: |
| 理论姿态 | 27,648 | 27,648 | 不变 |
| 实际枚举姿态 | 27,648 | 17,934 | -35.1% |
| 姿态域跳过 | 0 | 9,714 | +9,714 |
| 实际点评估 | 823,456 | 714,842 | -13.2% |
| p50 | 77.482 ms | 71.156 ms | -8.2% |
| p95 | 77.911 ms | 72.517 ms | -6.9% |
| 结果 | baseline 等价 | baseline 等价 | 无变化 |

第二批相对未剪枝 baseline 的模型点评估减少 83.8%。9,714 个姿态在进入逐姿态
预筛和评分前被整域跳过，其中 1,536 个来自合法 AABB 范围，8,178 个来自边缘支持域。

### 指定搜索域，17x17 ROI，5 次

搜索域：`[-15°, +15°] / 0.1°`，尺度 `[0.8, 1.2] / 0.05`，即
301 angles × 9 scales = 2,709 transforms。

| 指标 | 首批 optimized | 第二批 optimized | 变化 |
| --- | ---: | ---: | ---: |
| 理论/枚举姿态 | 782,901 | 782,901 | 不变 |
| 实际点评估 | 40,923,200 | 36,530,083 | -10.7% |
| p50 | 4,284.466 ms | 3,929.412 ms | -8.3% |
| p95 | 4,286.121 ms | 3,932.554 ms | -8.2% |

该 ROI 只有 289 个平移位置且完全覆盖目标边缘，因此稀疏姿态域不会减少姿态；收益来自更频繁的
score upper bound。后续姿态数量的主要下降将来自粗到细传播和平移块上界。
