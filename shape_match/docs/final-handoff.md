# OpenShapeMatch 0.1 最终交接文档

## 当前状态

`plan.md` 中 Phase 0–8 的 0.1 固定尺度实施范围已经完成。项目可以构建、测试、benchmark、安装，并由独立 CMake 项目通过 `find_package` 消费。

## 验证入口

```sh
cmake -S shape_match -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure

cmake -S shape_match -B build-benchmark -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DBUILD_BENCHMARK=ON
cmake --build build-benchmark -j4
./build-benchmark/shape_match_benchmark
```

CTest 包括 smoke、unit、integration 和安装后 package consumer 测试。默认 static、portable-only、AVX2 opt-in、shared 四种配置均为 4/4 通过。

## 新会话建议起点

0.1 计划已结束。新会话应从以下延期项目中单独选择一个并建立新计划：scale search、subpixel refinement、polarity、模型持久化、遮挡/变形、GPU、Python binding、Windows CI/ABI automation。

推荐优先顺序：模型持久化 → scale search → subpixel → Windows CI。

## 关键文档

- `docs/release-0.1.0.md`：发布说明与安装方式；
- `docs/compatibility.md`：HALCON-like 兼容矩阵；
- `docs/phase-7e-handoff.md`：最终性能阶段；
- `docs/phase-7d-handoff.md`：coarse prefilter 与性能统计；
- `docs/phase-7c-handoff.md`：AVX2 实验结论。

## 注意事项

- 当前工作区包含从首阶段开始尚未提交的整体实现；不要假定 `git diff` 只属于最后一个阶段。
- 直接 score 默认 kernel 是 `portable-soa`，匹配搜索使用 `portable-rotated-soa`；AVX2 是 opt-in 实验路径。
- `SearchStats` 开启逐调用计时时会产生测量开销；生产环境不需要统计时传 `nullptr`。
- 0.x 不承诺跨 minor C++ ABI 稳定。
