# OpenShapeMatch Phase 7B 交接文档

## 本阶段结论

Phase 7B 的缓存与可验证性能基础已经完成。匹配结果仍以现有标量 AoS kernel 为正确性基线，默认搜索使用可复用的 portable SoA kernel；运行时 CPU 特性探测和 kernel 选择入口已经固定，但 AVX2/AVX-512/NEON 专用实现尚未启用。

## 已完成的实现

- 新增 `EdgePyramid`，可从 `ImageView` 或已有 `EdgeMap` 构建完整的 Canny/Sobel 场景金字塔。
- 新增 `find_shape_models(const EdgePyramid&, ...)`。多个模型或重复搜索可共享同一个只读场景金字塔；已有 `EdgeMap` 重载现在会构建一次 `EdgePyramid` 后委托搜索。
- 金字塔矩阵在构建后只读，接口不在 `ShapeModel` 内保存场景状态，便于调用方在多线程环境中共享生命周期明确的缓存。
- 新增 `CpuFeatures` 运行时探测：x86 GCC/Clang 使用 `__builtin_cpu_supports`，ARM NEON 使用编译目标宏；新增稳定的 score kernel dispatch 点。
- 当前 dispatch 结果为 `portable-soa`，即使机器报告 AVX2/AVX-512，也不会未经回归直接切换到专用指令实现。
- benchmark 现在覆盖 640x400、1280x1024、2000x2000，以及 1/4 线程，并同时报告完整搜索和复用场景金字塔搜索耗时。

## 验证记录

```sh
cmake --build build -j4
ctest --test-dir build --output-on-failure
cmake --build build-benchmark -j4
./build-benchmark/shape_match_benchmark
```

CTest：3/3 通过。当前机器最近一次 benchmark：

```text
cpu=avx2=1,avx512f=1,neon=0 score_kernel=portable-soa
640x400   threads=1 full=39.192ms prepared=32.918ms
640x400   threads=4 full=32.340ms prepared=28.444ms
1280x1024 threads=1 full=154.323ms prepared=138.120ms
1280x1024 threads=4 full=152.839ms prepared=135.328ms
2000x2000 threads=1 full=476.067ms prepared=406.635ms
2000x2000 threads=4 full=452.990ms prepared=403.959ms
```

数值仅用于同一构建环境的趋势对比，不构成跨硬件验收门槛。

## 下一阶段启动点

下一会话从 Phase 7C 开始：

1. 为 `score_pose_soa_portable` 增加 AVX2 实验 kernel，先只覆盖无出界、固定有效点批次的内部循环，再由标量边界路径兜底。
2. 建立 scalar/portable-SoA/SIMD 的逐 pose score 回归，冻结 score 最大误差、valid count 和结果排序容差。
3. 用编译目标隔离 AVX2、AVX-512、NEON，禁止在不支持的 CPU 上执行专用指令；保留 portable fallback。
4. 把 benchmark 的完整搜索、金字塔构建、score、NMS 分项计时拆开，并增加峰值内存和线程扩展性记录。

## 已知边界

- 当前没有 `-march=native`、`-ffast-math` 或平台专用 SIMD 指令。
- `EdgePyramid` 只读共享是接口约定；调用方不得在搜索期间修改其中的 `cv::Mat` 数据。
- `EdgePyramid::build(const EdgeMap&, ...)` 要求 level zero 至少包含 `gray`、`edges`、`orientation`，并要求类型和尺寸一致。
- scale、subpixel、polarity、遮挡、持久化和 GPU 仍按原计划延期。

