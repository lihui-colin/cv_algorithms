# OpenShapeMatch Phase 7C 交接文档

## 本阶段结论

Phase 7C 已完成首个真实 AVX2 score kernel、独立 ISA 编译单元、运行时 CPU 分派和 scalar/SoA/AVX2 回归。AVX2 实现结果通过当前正确性测试，但在本机端到端 benchmark 中慢于 portable SoA，因此默认不启用；发布和普通构建继续选择 `portable-soa`。

## 已完成实现

- 新增 `src/matcher/score_avx2.cpp`，使用 AVX2 批量完成坐标变换、方向 gather、角度归一化、余弦近似、权重累计。
- AVX2 源文件通过独立 CMake object target 使用 `-mavx2` 编译，主库保持 portable baseline。
- `OPENSHAPE_ENABLE_AVX2=ON/OFF` 控制是否编译 AVX2 实现，默认 `ON`；编译器不支持 `-mavx2` 时自动跳过。
- `OPENSHAPE_USE_AVX2_BY_DEFAULT=ON/OFF` 控制运行时是否选择 AVX2，默认 `OFF`。只有编译了 kernel 且 CPU 报告 AVX2 时才可能执行专用指令。
- `active_score_kernel_name()` 返回实际选择的 `portable-soa` 或 `avx2`。
- 新增完整 8-lane batch 的 AoS scalar 与 SIMD-dispatched score 回归，冻结 valid count 完全一致，score 最大误差 `5e-4`。
- 默认 portable、显式 AVX2、完全不编译 AVX2 三种构建均通过 CTest。

## 验证命令

默认 portable dispatch：

```sh
cmake -S shape_match -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

显式 AVX2 dispatch：

```sh
cmake -S shape_match -B build-avx2-test -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DOPENSHAPE_USE_AVX2_BY_DEFAULT=ON
cmake --build build-avx2-test -j4
ctest --test-dir build-avx2-test --output-on-failure
```

无 AVX2 编译的 portable fallback：

```sh
cmake -S shape_match -B build-portable -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DOPENSHAPE_ENABLE_AVX2=OFF
cmake --build build-portable -j4
ctest --test-dir build-portable --output-on-failure
```

上述三种配置均为 3/3 CTest 通过。

## 性能结论

当前机器为 `avx2=1,avx512f=1,neon=0`。同一 benchmark 的最近一次结果显示 AVX2 实验 kernel 慢于 portable SoA：

```text
prepared pyramid, threads=1
640x400:    portable 32.829ms, AVX2 43.976ms
1280x1024:  portable 136.932ms, AVX2 202.434ms
2000x2000:  portable 402.373ms, AVX2 610.800ms
```

主要原因是稀疏、非连续图像采样带来的 gather、lane mask、边界检查和标量 fallback 成本，高于当前批量算术节省。因此默认保持 `OPENSHAPE_USE_AVX2_BY_DEFAULT=OFF`。这不是未实现，而是经过测量后不将负优化投入默认路径。

## 下一阶段启动点：Phase 7D

1. 优先优化搜索算法和候选生成，而不是继续堆叠指令集：当前 4 线程收益有限，且全局搜索仍对大量位置调用 score。
2. 加入低成本的 coarse prefilter（例如少量 anchor 点或方向直方图），减少完整模型 score 调用次数。
3. benchmark 分离候选数量、score 调用次数、score kernel 时间和 NMS 时间，用数据确定瓶颈。
4. 如果继续 SIMD，应先调整模型点/场景方向的数据访问模式，减少随机 gather；之后再评估 AVX-512 或 NEON。
5. 保留现有 AVX2 实验实现和 opt-in 开关作为后续结构实验基线。

## 已知边界

- AVX2 kernel 使用多项式余弦近似，允许 score 误差上限 `5e-4`；valid count 必须完全一致。
- MSVC 当前不构建 AVX2 object target，仍使用 portable fallback。
- AVX-512 和 NEON 仍只有特性探测，没有专用 score kernel。
- scale、subpixel、polarity、遮挡、持久化和 GPU 继续延期。

