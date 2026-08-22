# OpenShapeMatch 0.1.0 发布说明

## 发布范围

OpenShapeMatch 0.1.0 是固定尺度形状匹配基线发布，提供 C++17/OpenCV 4 原生库、安装包、示例、CTest 和 benchmark。

核心能力：Canny/Sobel 梯度方向模型、图像和模型金字塔、平移与旋转粗到细搜索、多目标/NMS、场景缓存、保守 coarse prefilter、确定性多线程、`SearchStats`、结果绘制，以及 portable SoA 默认 kernel 和 opt-in AVX2 实验 kernel。

## 构建与安装

```sh
cmake -S shape_match -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix /desired/prefix
```

外部项目：

```cmake
find_package(OpenShapeMatch 0.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE openshape::OpenShapeMatch)
```

## ABI 与平台

- 语言标准：C++17；包版本：0.1.0；ABI/SOVERSION：0。
- Linux GCC/Clang 为当前验证平台；portable kernel 不使用 `-march=native`。
- Linux static/shared 均通过安装后外部消费者测试。Windows 保留 CMake 构建路径并启用 shared-library symbol auto export，但本环境未执行 Windows CI。
- AVX2 使用独立 object target，默认不选择实验 kernel。匹配器的角度缓存路径使用 `portable-rotated-soa`；AVX2 当前仅用于 opt-in 的直接 SoA score 实验路径。

## 延期项目

尺度、亚像素、polarity、遮挡/变形、持久化、GPU 和 Python binding 不属于 0.1.0。后续实施必须分别建立参数契约、正确性回归和性能验收。
