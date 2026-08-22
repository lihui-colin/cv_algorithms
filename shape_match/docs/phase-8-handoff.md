# OpenShapeMatch Phase 8 交接文档

## 阶段结论

Phase 8 发布基线完成，版本为 0.1.0，ABI/SOVERSION 为 0。静态库、共享库、portable fallback 和 opt-in AVX2 配置均通过功能测试与安装后外部消费者测试。

## 发布交付

- 公共版本头 `openshape/version.hpp`；
- 安装导出目标 `openshape::OpenShapeMatch`；
- `OpenShapeMatchConfig.cmake` 自动查找 OpenCV 4 和 Threads；
- static/shared 的 library、archive、runtime、header 安装目录；
- 独立 package consumer 工程，通过 `find_package(OpenShapeMatch CONFIG REQUIRED)` 编译并运行；
- 结果绘制 API `draw_match_results`；
- 模型版本兼容检查；
- HALCON-like 兼容矩阵和 0.1.0 发布说明。

## 最终验证矩阵

以下配置均为 4/4 CTest 通过：

| 配置 | 关键选项 |
| --- | --- |
| 默认 static | `BUILD_TESTING=ON` |
| Portable only | `OPENSHAPE_ENABLE_AVX2=OFF` |
| AVX2 opt-in | `OPENSHAPE_USE_AVX2_BY_DEFAULT=ON` |
| Shared library | `BUILD_SHARED_LIBS=ON`, `OPENSHAPE_ENABLE_AVX2=OFF` |

四个测试为 smoke、unit、integration、package consumer。

## 后续

0.1 固定尺度计划已结束。后续功能必须作为独立版本项目启动，详见 [`final-handoff.md`](final-handoff.md)。

