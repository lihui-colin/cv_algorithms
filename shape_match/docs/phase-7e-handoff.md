# OpenShapeMatch Phase 7E 交接文档

## 本阶段结论

Phase 7E 已完成按角度/层级预计算旋转模型点和多场景 percentile benchmark。搜索开始时为每个角度缓存旋转后的 `relative_x/y` 与模型方向，pose 评分和预筛选只执行平移与采样，不再为每个 pose 重复计算模型点旋转。

## 已完成实现

- 每个模型层、每个搜索角度构建只读 `RotatedModelPoints`。
- 缓存旋转坐标、旋转方向、权重引用和旋转后点边界。
- coarse prefilter、summed-area AABB 和完整 score 共用同一旋转缓存。
- 细化层通过候选角度映射复用对应缓存。
- `SearchStats::rotation_preparation_time_ms` 单独记录缓存构建耗时。
- 最终结果 score 与公开 `score_pose` 标量/SoA 契约进行回归，误差不超过 `1e-7`，valid count 完全一致。
- benchmark 恢复预热、p50/p95/p99，并增加单/四线程、三种尺寸、多角度、多目标和 clutter 场景。

## Benchmark 摘要

当前机器、portable SoA、复用场景金字塔：

```text
scenario                   filtered p50   unfiltered p50   equivalent
640x400 sparse, 1 thread       8.462ms        27.618ms          yes
640x400 sparse, 4 threads      4.215ms        10.176ms          yes
1280x1024 sparse, 1 thread    13.339ms       114.140ms          yes
1280x1024 sparse, 4 threads    7.403ms        37.602ms          yes
2000x2000 sparse, 1 thread    26.062ms       334.926ms          yes
2000x2000 sparse, 4 threads   13.220ms        97.488ms          yes
640x400 multi-angle           21.035ms        65.616ms          yes
640x400 clutter               13.601ms        41.199ms          yes
```

旋转缓存准备耗时约 `0.003–0.052ms`。所有 benchmark 场景均自动检查 filtered/unfiltered 结果完全一致。

## 决策记录

- 没有加入逐行浮点增量坐标更新。最近邻取整边界会使累积误差难以保持当前逐结果完全一致契约，而旋转缓存已经移除了主要的重复三角函数和点旋转成本。
- 继续保留直接以整数 `x` 遍历 pose 的实现，优先保证坐标确定性。
- AVX2 稀疏 gather 路径继续 opt-in；新的旋转缓存未改变这一性能结论。
- `active_score_kernel_name()` 描述直接 `score_pose(ModelLevelSoA)` 分派；匹配搜索使用按角度缓存后的 `portable-rotated-soa`，由 `active_matcher_score_kernel_name()` 报告。

## 下一阶段

Phase 8 发布基线：安装/导出消费测试、版本/ABI 元数据、兼容矩阵、发布说明和延期功能边界。
