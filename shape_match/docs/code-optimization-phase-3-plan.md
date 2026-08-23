# 第三阶段：代码级耗时优化计划

## 1. 阶段目标

第三阶段只优化代码执行效率，不改变第二阶段已经冻结的搜索空间、剪枝依据、模型点评估语义和最终匹配结果。

固定验收搜索域：

```text
angle_start  = -15.0
angle_extent = 30.0
angle_step   = 0.1

scale_min  = 0.8
scale_max  = 1.2
scale_step = 0.05
```

对应 301 个角度、9 个尺度，共 2,709 个角度–尺度变换。17×17 ROI 下理论姿态数为：

```text
301 × 9 × 17 × 17 = 782,901
```

第三阶段不得通过减少上述姿态数、调整搜索步长、修改阈值、改变剪枝决策或删除候选获得加速。

### 1.1 正确性契约

- 最终结果数量和排序完全一致；
- row、column、angle、scale、score 及 valid point count 完全一致；
- 最终保留候选必须经过规范标量顺序复核；
- SIMD 近阈值结果必须回退规范标量内核；
- 不使用 `-ffast-math`；
- 不允许 FMA 收缩改变规范浮点运算顺序；
- x86 专用内核不可用时必须回退 portable scalar；
- 第二阶段的 theoretical/domain/pose/point 统计语义保持不变。

### 1.2 性能目标

- 当前约 31.2 ms 固定开销：冷启动降低到 5 ms 以下；
- 相同模型和搜索参数的热缓存查找与准备开销低于 1 ms；
- 4 线程相对第三阶段同代码单线程至少达到 3.0× 加速；
- SIMD 内核只有在端到端 p50 至少提升 10% 且 p95 回退不超过 3% 时，才允许进入 Auto 默认分派；
- 正式数据采用至少 5 次预热、30 次测量，并报告 p50/p95。

## 2. 公共接口与统计

### 2.1 显式搜索工作区

新增 `ExhaustiveSearchWorkspace`，用于同一模型和搜索范围的跨帧复用：

```cpp
class ExhaustiveSearchWorkspace {
public:
  ExhaustiveSearchWorkspace();
  ~ExhaustiveSearchWorkspace();

  void prepare(const ShapeModel& model, const SearchParams& params);
  void clear();
  std::size_t memory_bytes() const;
};
```

要求：

- 工作区内部使用 opaque implementation，避免在公共头文件暴露 SIMD 或缓存数据结构；
- 缓存内容构建完成后只读，可以安全地被搜索 worker 共享；
- 同一个工作区允许跨帧复用和并发读取；
- 参数或模型发生变化时，完整构建新计划后再原子替换旧计划；
- 每个工作区只保留一个当前计划，避免无界 LRU；
- `clear()` 释放缓存和 worker scratch；
- 原有 exhaustive API 保持兼容；新增接受 workspace 的重载；
- 未传 workspace 时使用调用内临时计划，不依赖全局缓存。

### 2.2 缓存键

缓存键必须包含：

- 模型版本；
- level 0 SoA 模型点数量及 relative_x、relative_y、orientation、weight 的精确位签名；
- 最终实际使用的 angle start/end/step；
- 最终实际使用的 scale min/max/step；
- 会影响点顺序或准备数据的参数。

浮点字段按 bit pattern 参与签名，禁止使用容差比较判断缓存命中。

### 2.3 计算内核选择

新增：

```cpp
enum class ComputeKernel {
  Auto,
  Scalar,
  AVX2,
  AVX512,
};
```

`SearchParams` 增加内核选择字段，默认 `Auto`。显式请求当前构建或 CPU 不支持的内核时抛出明确的 `InvalidArgument`，不得静默执行另一专用内核。

### 2.4 SearchStats 扩展

新增以下统计：

- workspace cache hit/miss/rebuild；
- workspace memory bytes；
- cache lookup wall time；
- transform table preparation wall time；
- domain preparation wall time；
- parallel search wall time；
- worker score CPU time；
- worker merge/sort wall time；
- actual worker count；
- selected precise kernel；
- SIMD batch count、active lane count、scalar tail count、scalar fallback count。

多线程 CPU 累计时间和调用侧 wall time必须明确区分。

## 3. 实施任务

### Phase 3A：冻结基线与热点拆分

1. 保存第二阶段最新 commit/worktree 状态和完整 benchmark 命令；
2. 对指定 782,901 姿态场景分别测量无统计和开启 SearchStats 的成本；
3. 将耗时拆分为：
   - scene preparation；
   - workspace lookup；
   - transform preparation；
   - pose-domain preparation；
   - prefilter；
   - precise score；
   - candidate collection；
   - merge/sort；
   - refinement；
4. 使用 profiler 记录三角函数、坐标变换、双线性采样、亚像素 3×3 搜索、分支失败和 cache miss；
5. 后续每个子阶段单独记录 p50/p95，禁止将多个优化混在一起后无法归因。

### Phase 3B：角度、尺度和模型点预计算

一次生成角度和尺度网格，禁止热循环中重复生成或浮点累加。

每个角度预计算：

- degrees 和 radians；
- float/double `sin`、`cos`；
- 规范化角度。

每个角度–尺度变换预计算：

- angle、scale 和稳定 transform index；
- 每个模型点的双精度 rotated/scaled x/y offset；
- 每个模型方向的 `sin/cos`；
- transformed AABB；
- support window key；
- 与模型点剪枝顺序对应的索引视图。

每个模型只计算一次：

- total positive weight；
- pruning point order；
- ordered weight；
- remaining weight suffix；
- uniform/non-uniform weight 标志。

准备表按 angle–scale block 并行生成。变换表构建完成后不可变，供所有搜索 worker 读取。

### Phase 3C：Prepared scalar 精确评分内核

新增 prepared precise scorer，直接接收预计算的 point offsets 和 orientation unit vectors：

- 每个姿态只增加 column/row；
- 不再调用 `sin/cos`；
- 不再逐点执行旋转与缩放；
- 图像尺寸、矩阵 data/step、sigma 常量和阈值在循环外提取；
- 使用行指针替代热循环中的 `cv::Mat::at`；
- 完整评分和可剪枝评分共享相同的采样 primitive；
- 保持模型点累计顺序和每步 upper bound 检查顺序不变。

亚像素路径优化：

- 无 subpixel mask 时直接进入无分支快速路径；
- 有 mask 时先读取 3×3 mask，再读取命中位置的 subpixel x/y；
- 预计算 `1 / (2*sigma²)`、edge gate 和 pow exponent；
- 不在模型点循环内构造临时容器。

所有通过低阈值的候选继续用原规范标量函数复核，保证公开最终 score 完全一致。

### Phase 3D：固定开销和内存复用

- 复用 transform table、point order、support-window grouping；
- 复用 worker-local candidate vector、统计结构和评分 scratch；
- 用预计算的 transform→support-window 索引替代每帧 2,709 次 `std::map` 查找；
- scene-dependent 支持域仍按帧计算，不允许跨图像复用；
- 小 ROI 不构建收益为负的稀疏支持域，保留第二阶段阈值逻辑；
- workspace 热命中不得重新计算模型签名之外的模型点变换；
- prepared 数据按 64-byte 对齐，并将只读表和 worker 可写数据分离以避免 false sharing。

### Phase 3E：确定性多线程 exhaustive 搜索

并行任务单位为：

```text
angle-scale transform × translation row tile
```

要求：

- tile 约包含 32–64 个姿态；
- 支持域和合法 translation run 先确定，再生成只读 job descriptors；
- `num_threads > 0` 使用指定线程数；
- `num_threads == 0` 使用 hardware concurrency，但最多 16 个 worker；
- 每个 worker 独享候选数组、SearchStats 和 scratch；
- 热循环中不使用锁；
- pose audit index 从规范 transform/translation index推导，不依赖线程执行顺序；
- 合并顺序先按 worker index，再执行现有 canonical stable sort；
- 1、4、8、16线程必须返回完全一致结果；
- refinement 只有在 profiler 中超过总时间 5% 时才并行。

### Phase 3F：x86 SIMD 精确内核

已有 AVX2 内核按模型点 lane 化，随机 gather 较多且历史 benchmark 为负优化。第三阶段 precise SIMD 不复用该数据路径，而采用相邻平移姿态 lane 化：

- AVX2：同时处理同一 row 上 8 个相邻 column；
- AVX-512：同时处理同一 row 上 16 个相邻 column；
- 对同一模型点，各 lane 的 y 相同、x 连续，优先使用连续或近连续加载；
- 每个 lane 独立维护 weighted score、inverted score、valid count、remaining weight 和 active mask；
- 每处理一个模型点后，对 active lane 执行 score/visible upper bound；
- 已安全终止 lane 不再进行后续采样；
- 不足一个向量宽度的尾部走 scalar；
- 图像边界、取整临界值、不规则亚像素邻域和接近剪枝阈值的 lane 走 scalar fallback；
- SIMD 拒绝必须使用向上舍入保护，任何无法证明安全的结果必须保留或回退；
- 最终候选使用规范 scalar rescore。

独立 ISA 编译单元：

```text
portable main library
AVX2 object:    -mavx2
AVX-512 object: -mavx512f -mavx512dq -mavx512bw -mavx512vl
```

禁止 `-march=native`、`-ffast-math` 和 FMA contraction。运行时使用 CPU feature detection 分派。

Auto 选择规则：

1. AVX-512 相比 AVX2 达到至少 10% p50 收益且 p95 无明显回退，才允许优先 AVX-512；
2. AVX2 相比 scalar 达到至少 10% p50 收益且 p95 无明显回退，才允许优先 AVX2；
3. 未达到门槛的内核仍可显式测试，但 Auto 选择更快的低级路径；
4. portable build 和不支持对应 ISA 的机器始终安全回退 scalar。

### Phase 3G：微架构收尾

- 检查 prepared point arrays 的 alignment 和 cache line 布局；
- 按模型点数和 LLC 容量评估 transform block size；
- 检查 branch miss、L1/L2/LLC miss、IPC、SIMD lane utilization；
- 删除热循环内重复函数调用和不必要的范围检查；
- 检查统计开启/关闭两条路径，避免无 stats 时仍执行计时和原子计数；
- 不在公共 API 中增加绑核或 NUMA 设置；benchmark 可使用单 NUMA node 固定 CPU 提高稳定性。

## 4. 测试计划

### 4.1 精确评分回归

逐姿态比较：

- 原规范完整 scorer；
- prepared scalar scorer；
- AVX2 scorer；
- AVX-512 scorer。

覆盖：

- 四种 polarity；
- 整数和小数模型坐标；
- angle 范围边界及 0.1° 网格；
- 9 个尺度；
- 图像四边和四角；
- 空 subpixel map 和完整 subpixel map；
- 多种 edge_distance_sigma；
- uniform/non-uniform weights；
- 模型点数小于、等于和大于 SIMD width。

要求最终完整 score、valid count逐值一致。

### 4.2 剪枝安全测试

- 对 prepared scalar、AVX2、AVX-512 的所有安全拒绝执行 100% 规范完整评分审计；
- audit failure 必须为 0；
- score upper bound 不得低于规范完整 score；
- SIMD active mask和scalar tail统计总和必须与逻辑点评估数一致。

### 4.3 工作区测试

- 首次 prepare 为 miss/rebuild；
- 同模型同参数再次调用为 hit；
- 改变角度范围、步长、尺度范围或模型后必须 rebuild；
- `clear()` 后 memory bytes 为 0；
- 同一工作区并发只读搜索结果一致；
- 临时工作区 API 与显式工作区 API 结果一致。

### 4.4 多线程确定性

1、4、8、16线程分别验证：

- result count；
- result order；
- row/column/angle/scale；
- score/valid count；
- theoretical/domain/pose统计；
- safe rejection和point evaluation统计。

### 4.5 构建矩阵

- 默认 Auto；
- Scalar only；
- AVX2 compiled/forced；
- AVX-512 compiled/forced；
- AVX2/AVX-512全部禁用的 portable build；
- shared/static library；
- 安装后 package consumer。

## 5. 性能验收

固定场景：

1. 指定宽搜索域：301 angles × 9 scales × 17×17 translations，160 points；
2. 96×96全图稀疏场景；
3. real sample；
4. synthetic continuous refinement；
5. 640×400 clutter；
6. 空白场景和高边缘密度低剪枝收益场景。

每个场景记录：

- cold workspace p50/p95；
- warm workspace p50/p95；
- 1/4/8/16线程；
- scalar/AVX2/AVX-512/Auto；
- total wall time；
- preparation、domain、search、merge、refinement分项；
- pose和point统计；
- workspace bytes；
- SIMD lane utilization和fallback rate。

正式命令必须至少执行：

```sh
cmake --build <release-build> -j4
ctest --test-dir <release-build> --output-on-failure
shape_match_pruning_benchmark 30 wide
```

验收完成后新增第三阶段 handoff 文档，记录：

- 硬件、编译器、OpenCV和构建参数；
- 各阶段前后 p50/p95；
- 固定开销下降；
- 线程扩展率；
- SIMD选择结论；
- 未进入 Auto 的负优化路径及原因；
- 所有正确性和审计结果。

## 6. 实施顺序和停止条件

严格按以下顺序实施：

```text
3A baseline/instrumentation
  → 3B workspace and transform cache
  → 3C prepared scalar
  → 3D allocation/domain reuse
  → 3E deterministic threading
  → 3F AVX2/AVX-512
  → 3G microarchitecture cleanup
  → full acceptance and handoff
```

每一步必须满足：

- CTest全部通过；
- pruning 100% audit零失败；
- 最终结果完全一致；
- p50/p95无不可解释回退。

如果某项 SIMD 或微优化没有达到性能门槛，则保留可测试实现但不进入 Auto，或直接回退该改动；不得为了宣称完成而默认启用负优化。
