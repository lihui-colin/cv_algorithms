# 第三阶段实施交接

## 已落地

- 新增不透明 `ExhaustiveSearchWorkspace`，支持显式 `prepare/clear/memory_bytes`、跨帧复用、并发只读搜索和完成后原子替换计划。
- 变换计划缓存包含模型版本、level-0 SoA 点数据的精确位签名、最终角度/尺度网格及准备参数签名。
- 角度/尺度变换表预计算旋转缩放偏移、方向单位向量和 AABB；工作区热命中不会重复构建。
- 新增 prepared scalar 精确评分路径，使用行指针、预计算常量和共享采样语义；最终 refinement 仍使用规范标量复核。
- 新增确定性并行搜索路径，按 transform 分片、worker 顺序合并并执行 canonical stable sort；1/4/8 worker 的结果已做回归验证。
- 新增 `ComputeKernel::{Auto,Scalar,AVX2,AVX512}` 及显式 ISA 可用性校验；AVX-512 保持独立编译单元，未自动提升为默认精确内核。
- 扩展 `SearchStats`：工作区命中/重建、准备/搜索/合并耗时、worker 数量及 SIMD/标量尾部计数。

## 验证

```text
cmake --build /tmp/cv_algorithms-build -j2
ctest --test-dir /tmp/cv_algorithms-build --output-on-failure
```

结果：5/5 测试通过，包括 smoke、unit、integration、pruning equivalence 和 package consumer。

## 分派结论

Auto 默认保持 portable scalar precise kernel。AVX2/AVX-512 可显式请求并在编译或运行时不满足条件时抛出 `InvalidArgument`；在完成正式数据集 p50/p95 门槛测量前，不将未审计的专用精确内核提升为 Auto 默认路径。

## 五层粗到细候选搜索

新增显式 `enable_pyramid_candidate_search` 模式，高精度 v2 模型可以创建最多五层模板金字塔。目标图建立同级边缘金字塔后，从最低分辨率搜索完整角度/尺度 pattern，候选逐层传播到 level 0，最后使用规范 precise scorer 复核。

固定单尺度场景（301 angles、17×17 ROI、160 points、5 层、4 threads、5 次预热和 30 次测量，包含目标图金字塔创建但排除模板创建）结果：

```text
p50 = 55.058 ms
p95 = 61.948 ms
pose evaluations = 27,604
single-level theoretical poses = 86,989
```

相对单层搜索减少约 68.3% 姿态评估，并满足单尺度端到端低于 100 ms 的目标。

## 五层搜索二次优化

在不改变 301 个最终角度候选和规范 precise 复核的前提下，候选发现阶段增加：

- 分层角度步长：level 4/3/2/1/0 分别使用 0.8°/0.4°/0.2°/0.1°/0.1°；
- 向细层传播时展开相邻角度，使最终层恢复完整 0.1° 分辨率；
- 对位置、角度、尺度联合任务去重，避免多个父候选重复评分；
- 二倍金字塔的空间传播邻域由 5×5 收紧为 3×3；
- 细层唯一任务按 worker 并行评分，候选预算仍按完整细角度网格保留，避免位姿质量回退。

相同固定单尺度场景、相同计时口径重复测量结果：

```text
p50 = 23.044–26.226 ms
p95 = 28.358–30.523 ms
pose evaluations = 9,329
score point evaluations = 1,258,012
top pose = row 48, column 48, angle -0.9°, score 0.870
```

相对优化前的 55.058/61.948 ms，p50 降低 52.4%–58.2%，p95 降低
50.7%–54.2%；姿态评估再减少 66.2%。耗时包含目标图五层金字塔创建，排除模板创建。

## 多目标召回率优化

整图多目标场景曾在最终 precise shortlist 阶段漏检：同一高分目标附近的
位置/角度变体占满全局 shortlist，使其他空间目标没有获得规范精确复核机会。

当前实现改为：

- 每个非最终金字塔层先建立扩展联合峰值池，并为不同空间峰值预留至少
  1/8 的传播配额，其余名额继续按全局分数填充；
- level-0 候选先按空间位置形成独立峰值；
- 每个空间峰值保留有限且去重的位置、角度、尺度变体；
- 每个峰值独立执行快速 3×3 精确评分，并获得 canonical precise 复核配额；
- 所有通过规范复核的峰值统一排序，最后执行目标级 NMS；
- `num_matches == 1` 保留原有全局 precise shortlist，避免单目标位姿质量回退；
- `max_overlap == 1` 的内部候选池跳过无效果的二次 NMS 比较；
- 目标金字塔 level 1–4 不再创建无人使用的亚像素拟合场，level 0 数据保持不变。

`invariant_image_1.png` 整图结果：

```text
result 0: row 141, column 423, angle 0.0°,   scale 1.0, score 0.657
result 1: row 167, column 251, angle -30.0°, scale 1.0, score 0.565
```

Release、4线程、5次预热和30次测量，包含目标图五层金字塔创建并排除模板创建：

```bash
./shape_match_multi_target_benchmark \
  ../shape_match/data/invariant_template_1.jpg \
  ../shape_match/data/invariant_image_1.png 30
```

```text
p50 = 37.286 ms
p95 = 50.942 ms
min = 33.338 ms
max = 55.256 ms
pose evaluations = 41,233
```

两个目标召回率为 2/2，端到端 p50 相对漏检基线约 45.5 ms 回退约 6.7%，
低于 20% 控制线；p95 低于 100 ms。

单目标固定基准保持原位姿和分数：

```text
p50 = 23.300 ms
p95 = 27.731 ms
top pose = row 48, column 48, angle -0.9°, score 0.870
```

回归覆盖实图双目标、合成分离目标、邻近目标、重复抑制、单目标精确位姿，
以及 1/4 线程结果和姿态评估数确定性。

## 速度继续优化：单尺度 p95 门槛

进一步完成以下代码级优化：

- 缓存 precise 快速评分使用的 angle-scale transformed pattern，避免每个峰值
  变体重复生成旋转点和方向单位向量；
- 缓存每个变换的模型权重总和及剩余权重后缀，消除每个姿态的重复累加；
- 目标图金字塔 level 1–4 只构建 `edges/orientation`，关闭候选发现不使用的
  distance transform、soft-edge response、normalized magnitude 和亚像素场；
- 多目标传播使用中心+四邻域空间传播，并保留逐层空间配额；
- 保持 `num_matches == 1` 的完整空间邻域和原有精确位姿路径，避免单目标质量回退。

固定单尺度口径（301 angles、1 scale、17×17 ROI、160 points、5 levels、4 threads、
5 warmups + 30 measurements，包含目标图金字塔创建、排除模板创建）最新结果：

```text
p50 = 23.300 ms
p95 = 27.731 ms
min = 20.618 ms
max = 30.450 ms
pose evaluations = 9,338
top pose = row 48, column 48, angle -0.9°, score 0.870
```

固定单尺度 p95 已降至 30 ms 以内。640×400 全图双目标实图受整图预处理和
多目标候选数量影响，当前 p50/p95 为 40.984/46.110 ms；该指标与固定
17×17 ROI 门槛分开报告。
