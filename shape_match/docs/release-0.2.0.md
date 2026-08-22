# OpenShapeMatch 0.2.0 发布说明

## 发布范围

OpenShapeMatch 0.2.0 在 0.1 固定尺度基线上加入正等比离散尺度搜索。平移、旋转和尺度共同参与候选搜索，尺度结果贯穿公共评分接口、粗到细匹配、NMS、排序和可视化。

新增公共参数：

```cpp
double scale_min = 1.0;
double scale_max = 1.0;
double scale_step = 0.05;
```

尺度范围满足 `0 < scale_min <= scale_max` 且 `scale_step > 0`。默认值只枚举 `1.0`，保持 0.1 固定尺度行为。范围按 `scale_min + n * scale_step` 离散枚举到不超过 `scale_max`；当前不做连续尺度或亚尺度插值。

新增 `score_pose(..., angle_degrees, scale, valid_count)` AoS/SoA 重载。旧的 angle-only 重载继续可用并等价于 `scale=1.0`。实验 AVX2 直接评分仍只用于旧的单位尺度分派；非单位尺度直接评分使用 portable SoA 路径。

## 搜索语义

- 最粗/全局层枚举角度与尺度的笛卡尔积。
- 各层预计算缩放旋转后的模型坐标、方向和边界。
- 向细层回溯时保留候选角度与离散尺度，只细化位置。
- `MatchResult::scale` 返回命中的离散尺度。
- `SearchStats` 报告角度、尺度和联合变换数量以及变换准备耗时；旧的 `rotation_preparation_time_ms` 保留并镜像新耗时字段。
- 排序顺序为 score 降序，再按 row、column、angle、scale 升序。
- NMS 与结果绘制均使用缩放后的模型 ROI。

## 构建与安装

```sh
cmake -S shape_match -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
cmake --install build --prefix /desired/prefix
```

外部项目：

```cmake
find_package(OpenShapeMatch 0.2 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE openshape::OpenShapeMatch)
```

包版本为 0.2.0，`SOVERSION` 仍为 0。0.x 不承诺跨 minor C++ ABI 稳定。

## 验证范围

测试覆盖参数校验、单位尺度旧/新评分重载一致性、固定尺度搜索精确兼容、非单位尺度 AoS/SoA 评分、缩小/放大目标、旋转加尺度、多目标不同尺度、负样本、确定性多线程、NMS、绘制和安装后外部消费者。

## 亚像素位置细化

设置 `SearchParams::enable_subpixel = true` 后，匹配器会在最终（原图）层
对每个候选使用 Canny 边缘掩膜与 Sobel 梯度的双线性连续评分，并沿行列方向
进行逐级缩小的局部抛物线拟合。默认的离散像素搜索、排序和 NMS 语义保持不变，
但候选的 `row`/`column` 会更新为局部亚像素位置，并将 `MatchResult::refined`
置为 `true`。

## 延期项目

连续/亚尺度优化、亚像素角度优化、polarity、遮挡/变形、模型持久化、GPU 和 Python binding 不属于 0.2.0。
