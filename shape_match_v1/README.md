# HALCON 风格高精度形状匹配 C++ 实现

这是可编译、可运行的独立 C++17 形状匹配工程。输入模板图和完整搜索图，支持旋转、等比尺度、多模板、多实例、亚像素位姿与轮廓输出。

九个公开匹配 API 按 HALCON 24.11 C++ 过程式签名组织，所有公开函数均有实际实现。兼容范围及明确差异见 [docs/API.md](docs/API.md)。没有实现 gen_circle/gen_rectangle1 等示例辅助算子，没有 HDevelop 解释器，也不依赖 HALCON SDK。

## 交付状态

- 完整源码、中文设计说明、函数注释、样例输入、CLI、测试与实测结果。
- 全图样例实际检出 3 个 ring、4 个 nut；非按坐标硬编码返回。
- 独立无噪声合成集 30 案例全部检出；具体位置 RMSE 与误差分布见 [results/VALIDATION.md](results/VALIDATION.md)。
- 该合成集达到 1/30 px RMSE 目标不代表任意实拍图达到该精度。
- 640×480 双模板样例的热缓存搜索在本机 32 线程中位数约 27 ms；实际耗时取决于 CPU、系统负载和搜索范围，测试方法见 [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)。
- 未在本环境执行 HALCON 运行时差分；已有 HALCON CSV 只作为参照，非绝对真值。

## 目录

```text
include/shape_match/shape_match.hpp   公开接口、对象/元组/句柄
src/internal.hpp                     内部数据结构和函数注释
src/types.cpp                        类型、句柄、输入输出
src/model.cpp                        参数管理、模板训练
src/geometry.cpp                     滤波、边缘、金字塔、几何
src/search.cpp                       全图搜索、跟踪、最小二乘、NMS
src/result.cpp                       结果选择器、坐标和轮廓
src/parallel.cpp                     工作线程池、统一线程配置和异常传递
src/precision.cpp                    可选连续轮廓与高斯梯度精定位
examples/match_sample.cpp            可用于任意 PGM 输入的命令行程序
tests/test_match.cpp                 契约、几何与检出回归
tests/synthetic.hpp                  独立连续几何渲染器
tests/accuracy_bench.cpp             亚像素精度测试
tests/refinement_probe.cpp           离线 GT 初始化、收敛性和最近邻审计（非精度验收）
scripts/                            编译、PNG 转换、报告和验证脚本
data/                               PGM 样例与原始用户附件
results/                            本次实际运行产生的结果与日志
docs/                               API、函数实现、优化与测试说明
```

## Linux 一键构建和运行

需要 g++/Clang 的 C++17 支持以及 ar。核心库无第三方 C++ 依赖。已在 Linux g++ 13.3.0 编译运行。

```bash
cd shape_match_v1
bash scripts/build.sh
./build/test_match
./build/match_sample
./build/accuracy_bench results/accuracy.csv
```

搜索默认最多使用 32 个工作线程。可在进程启动前通过 `SHAPE_MATCH_NUM_THREADS=N` 指定
线程数，例如 `SHAPE_MATCH_NUM_THREADS=8 ./build/performance_bench 50`。配置在首次使用时
固定，最多使用硬件报告的线程数；无效值使用 1 线程。性能比较必须使用相同线程数。

PGM 文件已经包含在包内，运行匹配不需要 Python。构建产生静态库 `build/libshape_match.a`。

有 CMake 的环境也可以：

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake -j
ctest --test-dir build-cmake --output-on-failure
```

## 完整验证与报告

Python 报告依赖 Pillow，匹配本身不依赖它。

```bash
python3 -m pip install -r requirements-tools.txt
bash scripts/run_validation.sh
```

输出 CSV、轮廓 JSON、叠加 SVG、计时 JSON、验证报告。CSV 使用 17 位有效数字，不把参考 CSV 的三位小数当作精度上限。原始 HALCON CSV 完整保留；报告按用户确认的历史模型标签错误进行比较层修正。

样图相对参考的当前位置 RMSE 为 0.300350px，尚未达到 1/30px。
旧版对照及精定位诊断见 [docs/REFINEMENT_DIAGNOSIS.md](docs/REFINEMENT_DIAGNOSIS.md)。

## 可选精度增强与 GT 验收

新增路径仍是实验功能：连续轮廓改善了样图整体误差，高斯梯度改善了独立合成精度，
但两者均未满足实图精度与性能的联合验收，因此默认仍为 `nearest_point`。
实现范围、实测数据及尚需 HALCON 导出对齐的部分见
[精度实施报告](docs/PRECISION_IMPLEMENTATION.md)。

```bash
./build/match_sample --refinement-method contour --refinement-radius 3 \
  --diagnostics --output-dir results-contour
python3 scripts/compare_gt.py results-contour/matching_results.csv \
  --bundled-labels --max-rmse 0.1 --max-error 0.2
# 当前样图尚未达标，上述验收命令预期返回 2。
./build/accuracy_bench accuracy-gaussian.csv gradient_gaussian 1.5 extended
./build/test_precision
```

`--diagnostics` 额外输出模板参数、Domain 重心、模型轮廓和结果变换矩阵。
`--bundled-labels` 只用于包内历史 GT 的已知标签修正；新导出的正确标签 CSV 不要加此参数。
HALCON 侧导出程序文本位于 `scripts/export_halcon_alignment.hdev.txt`，须在 HDevelop 中配置路径执行；
本环境没有 HALCON 运行时，尚未验证该程序。

补齐 HALCON 数据后的参数逐项对齐及模板轮廓替换诊断见
[HALCON 对齐实验](docs/HALCON_ALIGNMENT.md)。探针 `./build/halcon_parameter_probe data 新输出目录`
读取实际导出参数，不读取 GT 位姿，也不修改默认算法。
该探针还运行 32 组法向峰定位对照，实验范围和结果见
[法向峰定位诊断](docs/NORMAL_PROFILE_DIAGNOSIS.md)。这些是固定基线位姿的离线实验，不是新的生产匹配模式。
新增[内外轮廓残差诊断](docs/CONTOUR_RESIDUAL_DIAGNOSIS.md)；可选 `--evaluate-reference`
仅在诊断阶段读取 GT 评估残差，不使用 GT 初始化搜索或优化。
探针也包含[梯度方向得分对照](docs/ORIENTATION_DIAGNOSIS.md)，尚未改善与 HALCON 的一致性。
基于新 HALCON 模式导出的[真实精定位轨迹对照](docs/REFINEMENT_TRAJECTORY.md)
由 `refinement_trajectory` 提供，主目标仍为普通 `least_squares`。
新增[初值与残差形式对照](docs/REFINEMENT_SEED_DIAGNOSIS.md)：支持自身插值初值、
仅诊断的导入初值和点到点残差，同时区分过滤前轨迹与最终检出结果。
进一步的[对应点更新与步长接受对照](docs/REFINEMENT_UPDATE_DIAGNOSIS.md)
记录固定对应点、每轮重关联和支持损失检查的结果，均未达到普通 least_squares 对齐目标。

## 换成自己的模板和大图

首先将图像转为 8 位灰度 PGM；透明模板另提供二值有效区域 PGM，掩码尺寸与模板一致。

```bash
./build/match_sample \
  --image your_search.pgm \
  --template target your_template.pgm your_mask.pgm 0.8 1.2 -30 30 \
  --min-score 0.5 \
  --subpixel least_squares_very_high \
  --output-dir your_results
```

`--template` 后依次是：名称、模板 PGM、mask PGM（无 mask 用 `-`）、最小尺度、最大尺度、起始角度（度）、结束角度（度）。可以重复该选项添加多个模板。API 本身的角度单位始终为弧度，CLI 才接受度。

## C++ 最小调用

```cpp
#include "shape_match/shape_match.hpp"
using namespace shape_match;

HTuple model, result, count;
CreateGenericShapeModel(&model);
SetGenericShapeModelParam(model,
    {"model_identifier", "iso_scale_min", "iso_scale_max"},
    {"part", 0.8, 1.2});
TrainGenericShapeModel(ReadPgm("template.pgm", "mask.pgm"), model);
SetGenericShapeModelParam(model, "subpixel", "least_squares_very_high");
FindGenericShapeModel(ReadPgm("search.pgm"), model, &result, &count);

HTuple rows, columns;
GetGenericShapeModelResult(result, "all", "row", &rows);
GetGenericShapeModelResult(result, "all", "column", &columns);
HObject contours;
GetGenericShapeModelResultObject(&contours, result, "all", "contours");
ClearShapeModel(model);
```

## 内存检查

```bash
BUILD_DIR=build-sanitize SANITIZE=1 bash scripts/build.sh
ASAN_OPTIONS=detect_leaks=1 ./build-sanitize/test_match
```

## 接下来读什么

1. [API 与兼容范围](docs/API.md)：外部调用、支持参数、坐标和已知差异。
2. [函数与实现](docs/FUNCTIONS.md)：P0～P5 与源码对应，算法和函数职责。
3. [优化方向](docs/OPTIMIZATION.md)：精度、CPU 性能、兼容性优先级。
4. [测试规范](docs/TESTING.md)：本次实测范围与待扩展的验收条件。
5. [实测验证报告](results/VALIDATION.md)：真实运行产生的结果。
