# OpenShapeMatch Phase 7A 交接文档

## 本阶段结论

Phase 0–6 的首阶段功能基线保持通过。本次完成的是 Phase 7 的第一段性能基线工作（Phase 7A）：在不改变公开 AoS 数据契约和标量 score 语义的前提下，引入模型点 SoA 缓存，并支持复用已计算的 `EdgeMap`。标量 AoS score 保留为回归基线；匹配器默认使用 SoA score kernel。

## 已完成的实现

- `ShapeModel` 在创建时为每个金字塔层生成 `ModelLevelSoA`，保存 `relative_x/y`、`orientation` 和 `weight` 的连续数组。
- 新增 `score_pose(const EdgeMap&, const ModelLevelSoA&, ...)`，使用行指针访问边缘和方向矩阵，保持与 AoS 标量 kernel 相同的最近邻采样、方向相似度和权重归一化规则。
- 新增 `find_shape_models(const EdgeMap&, ...)` 重载。调用方可以先执行一次 `EdgeEngine::compute`，再对同一场景重复匹配多个模型或重复运行搜索，避免重复执行入口图像预处理。
- 原 `find_shape_models(const ImageView&, ...)` 委托到预计算重载，公开行为不变。
- 单元测试覆盖 SoA/AoS score 数值一致、SoA 点数量/坐标一致、预计算 `EdgeMap` 结果一致，以及不完整 `EdgeMap` 的错误处理。
- benchmark 增加预处理 p50 和复用 edge map 后搜索 p50，并明确复用搜索仍包含场景金字塔构建。

## 验证记录

在仓库根目录执行：

```sh
cmake --build build -j4
ctest --test-dir build --output-on-failure
cmake --build build-benchmark -j4
./build-benchmark/shape_match_benchmark
```

当前环境结果：3/3 CTest 通过。基准样例（640x400 场景、120x120 模板、177 个模型点、3 层金字塔）最近一次输出约为：

```text
full p50 = 31.502 ms, p95 = 32.011 ms, p99 = 32.304 ms
preprocess p50 = 1.613 ms
reused-search p50 = 29.548 ms
```

这些数字只用于本机趋势比较，不是跨机器验收标准。

## 当前工作区关键文件

- `shape_match/include/openshape/model/shape_model.hpp`
- `shape_match/src/model/shape_model.cpp`
- `shape_match/include/openshape/matcher/matcher.hpp`
- `shape_match/src/matcher/matcher.cpp`
- `shape_match/tests/unit_tests.cpp`
- `shape_match/benchmark/benchmark.cpp`

## 下一会话启动点：Phase 7B

下一阶段应从性能优化的可验证闭环继续，而不是重新设计公共 API：

1. 为 SoA score kernel 建立独立的 benchmark 场景矩阵（640x400、1280x1024、2000x2000；固定模型点数、层数、角度范围、线程数）。
2. 增加标量/SoA/优化实现的结果回归：结果数量、稳定排序、位置/角度容差和 score 最大误差。
3. 在不改变 portable baseline 的前提下加入 SIMD 实现和运行时 CPU dispatch；AVX2/AVX-512/NEON 必须各自有 fallback，并通过同一 score 回归套件验证。
4. 评估缓存完整场景金字塔（而不只是入口 `EdgeMap`）的接口和生命周期，避免把非线程安全的缓存状态放入 `ShapeModel`。
5. 扩展 benchmark 输出到搜索、NMS、峰值内存和线程扩展性；固定编译器、OpenCV 版本和 CPU 信息后再讨论性能门槛。

## 已知边界

- 本阶段没有启用 `-ffast-math`，也没有引入平台专用 SIMD 指令。
- `find_shape_models(const EdgeMap&, ...)` 要求 `gray`、`edges`、`orientation` 非空且尺寸/类型匹配；应使用同一 `ShapeModelParams` 生成该 `EdgeMap`。
- 复用 edge map 的搜索仍在每次调用中构建场景金字塔；完整金字塔缓存留给 Phase 7B。
- `scale`、subpixel、polarity、遮挡、持久化和 GPU 仍按计划延期。

