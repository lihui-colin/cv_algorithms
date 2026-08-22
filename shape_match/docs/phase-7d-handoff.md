# OpenShapeMatch Phase 7D 交接文档

## 本阶段结论

Phase 7D 已完成候选预筛选、搜索统计和粗层任务并行重构。默认路径在不改变匹配结果的前提下显著减少完整方向 score 调用；2000x2000 基准中，完整 score 调用从 252325 次降至 2354 次，复用场景金字塔后的单线程搜索由约 413.7ms 降至 30.1ms。

## 已完成实现

- `EdgeEngine` 为每层 Canny mask 构建 `CV_32S` summed-area table (`EdgeMap::edge_integral`)。
- 默认启用 `SearchParams::enable_coarse_prefilter`，只作用于最粗的全局搜索层。
- 第一级预筛选使用旋转后模型点 AABB 查询 summed-area table；区域内没有任何 Canny 边缘时立即拒绝。
- 第二级预筛选逐点检查 Canny 命中，并在“剩余点全部命中也无法达到最低有效点数”时提前拒绝。
- 两级拒绝都属于保守上界判断，不使用近似 score 阈值，因此启用/禁用预筛选的最终结果、位置、角度和 score 保持一致。
- 粗层并行单位从“角度”改为“角度 × 图像行”。即使只搜索一个角度，`num_threads > 1` 也能并行全局平移搜索。
- 新增 `SearchStats`，记录 pose、预筛选、完整 score、候选与 NMS 数量，以及预处理、预筛选、score、候选搜索和 NMS 时间。
- `find_shape_models` 的 `ImageView`、`cv::Mat`、`EdgeMap`、`EdgePyramid` 重载均可接受可选的 `SearchStats*`。

## 正确性契约

- `enable_coarse_prefilter=false` 完全绕过预筛选，所有 pose 进入原 score kernel。
- 预筛选只拒绝不可能达到 `max(1, model_points / 4)` 有效点数的 pose。
- summed-area AABB 查询在模型点边界外额外扩展一个像素，覆盖最近邻取整误差。
- 单层和多层搜索均增加启用/禁用预筛选的结果一致性测试。
- 单线程、多线程、portable 和 opt-in AVX2 构建继续使用相同的稳定排序规则。

## SearchStats 字段

```text
pose_evaluations
prefilter_evaluations
prefilter_rejections
integral_prefilter_rejections
full_score_evaluations
accepted_candidates
nms_input_candidates
nms_output_matches
preprocessing_time_ms
rotation_preparation_time_ms
prefilter_cpu_time_ms
score_cpu_time_ms
candidate_search_time_ms
nms_time_ms
```

`prefilter_cpu_time_ms` 和 `score_cpu_time_ms` 是各 worker 的累计 CPU 时间；多线程下可能大于 wall time。其他阶段时间为调用侧 wall time。

## 验证与性能记录

默认 Release + portable SoA，当前机器最近一次单次 benchmark：

```text
prepared scene pyramid
image       threads  filtered    unfiltered  full scores
640x400       1       16.367ms     41.685ms    2354 / 18325
640x400       4        7.524ms     11.713ms    2354 / 18325
1280x1024     1       16.122ms    141.063ms    2354 / 84245
1280x1024     4        7.416ms     43.434ms    2354 / 84245
2000x2000     1       30.085ms    413.701ms    2354 / 252325
2000x2000     4       15.125ms    118.907ms    2354 / 252325
```

2000x2000 场景中，249971 个粗层 pose 被保守预筛选拒绝，其中 248251 个由 summed-area AABB 快速拒绝。上述数据仅用于同机趋势对比。

## 下一阶段启动点：Phase 7E

1. 分离并复用模板旋转数据：为每个角度预计算旋转后的模型点坐标和方向，避免每个 pose 重复 `sin/cos` 与点坐标旋转。
2. 评估 coarse 层的行扫描增量更新或按角度缓存，进一步降低每个 pose 的变换成本。
3. benchmark 改为多轮预热和 p50/p95/p99，避免单次数据受调度和缓存状态影响。
4. 增加不同边缘密度、多个角度和多个目标的性能场景，防止优化只适用于稀疏单目标图像。
5. 在新的数据访问模式稳定后重新评估 AVX2/AVX-512/NEON；当前 AVX2 实验路径继续保持 opt-in。

## 已知边界

- summed-area table 增加每层约 `(rows + 1) * (cols + 1) * 4` 字节内存。
- 当前预筛选只用于最粗层；细化层候选数量小，避免重复 Canny 扫描的额外成本。
- benchmark 仍为单次测量，尚未恢复 Phase 7A 的 percentile 输出。
- scale、subpixel、polarity、遮挡、持久化和 GPU 继续延期。
