# HALCON 风格高精度形状匹配 C++ 实现

这是可编译、可运行的独立 C++17 形状匹配工程。输入模板图和完整搜索图，支持旋转、等比尺度、多模板、多实例、亚像素位姿与轮廓输出。

九个公开匹配 API 按 HALCON 24.11 C++ 过程式签名组织，所有公开函数均有实际实现。兼容范围及明确差异见 [docs/API.md](docs/API.md)。没有实现 gen_circle/gen_rectangle1 等示例辅助算子，没有 HDevelop 解释器，也不依赖 HALCON SDK。

## 交付状态

- 完整源码、中文设计说明、函数注释、样例输入、CLI、测试与实测结果。
- 全图样例实际检出 3 个 ring、4 个 nut；非按坐标硬编码返回。
- 独立无噪声合成集 30 案例全部检出；具体位置 RMSE 与误差分布见 [results/VALIDATION.md](results/VALIDATION.md)。
- 该合成集达到 1/30 px RMSE 目标不代表任意实拍图达到该精度。
- CPU 搜索当前为秒级附近，10 ms 目标尚未达到；优化路线见 [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)。
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
examples/match_sample.cpp            可用于任意 PGM 输入的命令行程序
tests/test_match.cpp                 契约、几何与检出回归
tests/synthetic.hpp                  独立连续几何渲染器
tests/accuracy_bench.cpp             亚像素精度测试
scripts/                            编译、PNG 转换、报告和验证脚本
data/                               PGM 样例与原始用户附件
results/                            本次实际运行产生的结果与日志
docs/                               API、函数实现、优化与测试说明
```

## Linux 一键构建和运行

需要 g++/Clang 的 C++17 支持以及 ar。核心库无第三方 C++ 依赖。已在 Linux g++ 13.3.0 编译运行。

```bash
cd shape_match
bash scripts/build.sh
./build/test_match
./build/match_sample
./build/accuracy_bench results/accuracy.csv
```

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
