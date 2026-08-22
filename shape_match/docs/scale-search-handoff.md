# OpenShapeMatch 0.2 尺度搜索交接文档

## 当前状态

0.2 尺度搜索阶段已经实现。默认配置继续执行固定 `scale=1.0` 搜索；设置 `SearchParams::scale_min`、`scale_max` 和 `scale_step` 后，匹配器在全局层联合枚举角度与正等比离散尺度，并将选中的尺度传递到细化、结果、NMS 和绘制。

## 关键实现位置

- `include/openshape/core/types.hpp`：尺度搜索参数与 `MatchResult::scale`。
- `include/openshape/matcher/matcher.hpp`：尺度感知 AoS/SoA `score_pose` 重载。
- `src/core/image.cpp`：尺度范围校验。
- `src/matcher/matcher.cpp`：角度×尺度变换缓存、粗到细搜索、尺度评分、排序、NMS 边界框和绘制。
- `tests/unit_tests.cpp`：参数、API 兼容、评分、NMS、绘制和确定性回归。
- `tests/integration_tests.cpp`：缩小/放大、旋转加尺度、多尺度实例和负样本。
- `benchmark/benchmark.cpp`：固定尺度、尺度范围和角度×尺度场景。

## 设计边界

- 仅支持正的 isotropic scale。
- 离散尺度由 `scale_min + n * scale_step` 生成，不超过 `scale_max`。
- 细化层不搜索相邻尺度，不做连续尺度插值。
- 正尺度不改变模型梯度方向，只缩放相对坐标。
- `SearchStats` 新增角度、尺度、联合变换计数和 `transform_preparation_time_ms`；`rotation_preparation_time_ms` 为兼容 0.1 保留并镜像相同耗时。
- AVX2 仍是 opt-in 的单位尺度直接评分实验路径；搜索主路径使用准备后的 portable SoA 评分。

## 验证入口

```sh
cmake --build build -j4
ctest --test-dir build --output-on-failure

cmake --build build-portable -j4
ctest --test-dir build-portable --output-on-failure

cmake --build build-avx2-test -j4
ctest --test-dir build-avx2-test --output-on-failure

cmake --build build-shared -j4
ctest --test-dir build-shared --output-on-failure

cmake --build build-benchmark -j4
./build-benchmark/shape_match_benchmark
```

本阶段最终验证结果：default static、portable-only、AVX2 opt-in、shared 四套配置均为 CTest 4/4 通过，安装后 package consumer 同时验证 0.2 版本头和 CMake 导出包。`git diff --check` 通过。

2026-08-22 Release benchmark 的新增场景结果：

- `scale`：5 个尺度变换，目标尺度 `1.25`，检测尺度 `1.25`，filtered/unfiltered 结果等价；prepared filtered p50 `9.643 ms`。
- `angle-scale`：5 个角度 × 5 个尺度，共 25 个变换，目标尺度 `1.25`，检测尺度 `1.25`，filtered/unfiltered 结果等价；prepared filtered p50 `52.360 ms`。
- benchmark 全部场景 `equivalent=1`；数据仅代表当前机器和固定 benchmark 参数，不是跨平台性能承诺。

## 新会话建议起点

0.2 阶段结束后应在新会话选择独立后续项目。推荐优先考虑模型持久化或亚像素/亚尺度优化；不要把连续尺度优化直接混入本阶段的离散尺度语义。

## 工作区注意事项

当前工作区仍包含从 0.1 开始的整体未提交实现。不要假定 `git diff` 只属于尺度搜索阶段，也不要覆盖无关改动。
