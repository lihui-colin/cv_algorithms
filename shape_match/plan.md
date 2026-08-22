# OpenShapeMatch C++17 实施计划

## 目标与范围

在 `shape_match/` 内实现一个基于 C++17 + OpenCV 的高性能、可测试的 HALCON-like 形状匹配库。

OpenCV 负责：

- 图像读取和灰度转换；
- Canny 边缘提取；
- Sobel 梯度计算；
- 图像金字塔；
- 旋转、缩放和基础几何处理。

匹配核心先建立标量正确性基线，再逐步加入 SoA 数据布局、SIMD、CPU 特性分派和多线程优化。

“HALCON 兼容”仅表示公开 API 语义和参数命名的有限子集兼容，不承诺复刻 HALCON 未公开的内部算法、score 数值、候选排序或内部性能。

### 首阶段交付范围

- 固定尺度 `scale = 1.0` 的平移和旋转匹配；
- 图像/模型金字塔；
- 粗到细搜索；
- 多目标结果和 NMS；
- Canny 边缘候选和 Sobel 梯度方向评分；
- 原生 C++ 库、单元测试、集成测试、benchmark 和示例程序。

### 延期功能

以下功能不进入首阶段验收：

- 尺度搜索；
- polarity、local polarity 和 global polarity；
- 遮挡建模和变形匹配；
- 亚像素优化；
- 模型持久化；
- GPU；
- Python binding；
- 完整 HALCON API；
- AVX-512/NEON 专用优化。

---

## 技术路线与构建约束

1. 语言标准：C++17。
2. OpenCV：4.x，至少依赖 `core`、`imgproc`、`imgcodecs`。
3. 构建系统：CMake 3.20+。
4. 编译器：GCC、Clang、MSVC；首阶段以 Linux 为主，同时保持 Windows 可构建。
5. 测试框架：GoogleTest 或 Catch2，必须支持 `ctest` 统一执行。
6. Benchmark：Google Benchmark 或等价 C++ benchmark。
7. 默认 Release 使用 `-O3` 或 MSVC `/O2`，不默认使用 `-march=native`。
8. 发布构建使用 portable baseline；AVX2、AVX-512 和 NEON 通过独立实现与运行时 CPU 特性检测逐步加入。
9. 不允许 `-ffast-math` 影响功能基线；若性能阶段启用，必须与标量版本进行结果误差回归。
10. 公共 API 必须明确图像数据所有权、ROI view 生命周期、线程安全和错误处理方式。

---

## 目录结构与交付物

以当前项目目录 `shape_match/` 为工程根，不再创建新的 `openshapematch/` 顶层目录：

```text
shape_match/
├── CMakeLists.txt
├── cmake/
├── include/
│   └── openshape/
│       ├── core/
│       ├── edge/
│       ├── model/
│       ├── pyramid/
│       ├── matcher/
│       ├── optimizer/
│       └── api/
├── src/
│   ├── core/
│   ├── edge/
│   ├── model/
│   ├── pyramid/
│   ├── matcher/
│   ├── optimizer/
│   └── api/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── benchmark/
├── examples/
├── scripts/
├── data/
├── gt/
└── docs/
```

本阶段不创建 `python/` 目录，不实现 Python binding。根 README、`shape_match/README.md` 和 `plan_1.md` 需要在计划确认后同步为 C++17/OpenCV 路线或标记为历史方案。

---

## 公共 API 与数据契约

### 图像输入

- 支持 `CV_8UC1`、`CV_8UC3`、`CV_8UC4`；入口统一转换为 `CV_8UC1`。
- 图像读取使用 `cv::imread`。
- 彩色图像使用 `cv::cvtColor`，默认输入格式为 BGR/BGRA。
- 空图像、非 8-bit 图像和不支持的通道数返回明确错误。
- 内部坐标统一为 `(x, y)`，其中 `x` 为列、`y` 为行。
- 公开结果使用 `row = y`、`column = x`。
- 坐标原点采用像素中心约定。
- 默认 reference origin 为模板 ROI 中心，允许通过参数指定。
- `ImageView` 与 owning `Image` 分离；ROI view 不复制数据，但源图像生命周期必须覆盖 view 使用期。

### `ShapeModelParams`

至少包含：

- `roi`：模板矩形 ROI，首阶段不支持任意 mask；
- `origin`：reference origin，默认 ROI 中心；
- `canny_low_threshold`；
- `canny_high_threshold`；
- `canny_aperture_size = 3`；
- `canny_l2_gradient = true`；
- `sobel_kernel_size = 3`；
- `sobel_border_type = cv::BORDER_REFLECT_101`；
- `gaussian_kernel_size = 1`，默认不额外平滑；
- `gaussian_sigma`；
- `min_gradient_magnitude`；
- `min_model_points`；
- `max_model_points`；
- `min_point_distance`；
- `num_levels`，`0` 表示自动估计；
- `model_point_sampling`；
- `model_version`。

所有参数必须定义默认值、合法范围、非法组合和对应测试。

### `SearchParams`

至少包含：

- `angle_start`；
- `angle_extent`；
- `angle_step`；
- `num_levels`，`0` 表示使用模型层数；
- `min_score`，范围 `[0, 1]`；
- `num_matches`，`0` 表示不限制数量；
- `max_overlap`，范围 `[0, 1]`；
- `search_roi`；
- `coarse_candidate_limit`；
- `refinement_radius`；
- `num_threads`，`0` 表示自动；
- `deterministic`，默认 `true`；
- `enable_subpixel`，首阶段必须保持 `false`。

角度公共单位统一为 degree，内部计算可转换为 radians。首阶段不开放 scale 搜索参数，结果中的 `scale` 固定为 `1.0`。

### `MatchResult`

包含：

- `row`；
- `column`；
- `angle`，单位 degree；
- `scale = 1.0`；
- `score`，范围 `[0, 1]`；
- `valid_point_count`；
- `model_point_count`；
- `level_used`；
- `refined = false`。

匹配 API 返回：

```cpp
std::vector<MatchResult> find_shape_models(
    const ImageView& image,
    const ShapeModel& model,
    const SearchParams& params
);
```

结果按 score 降序排列；相同 score 再按 row、column、angle 稳定排序。无匹配返回空 vector。非法参数、空图像、空模型或模型版本不兼容必须使用统一的异常类型或错误码，具体方案在 Phase 1 冻结。

---

## OpenCV Canny/Sobel 处理管线

统一数据流：

```text
cv::imread
    -> cv::cvtColor
    -> 可选 cv::GaussianBlur
    -> cv::Sobel 计算 Gx/Gy
    -> magnitude/orientation
    -> cv::Canny 生成 edge mask
```

### Canny 边缘提取

- 边缘提取统一使用 `cv::Canny`。
- 默认 `apertureSize = 3`。
- 默认 `L2gradient = true`。
- 低阈值和高阈值通过参数配置，要求 `0 <= low < high`。
- 输出为 `CV_8UC1`，像素值为 `0` 或 `255`。
- Canny 只负责边缘候选筛选，不提供模型描述所需的方向和幅值。

### Sobel 梯度计算

- 梯度计算统一使用 `cv::Sobel`。
- 默认 `ksize = 3`。
- 输出深度为 `CV_32F`。
- 边界模式默认 `cv::BORDER_REFLECT_101`。
- 计算 `Gx`、`Gy`、梯度幅值和方向。
- 梯度幅值使用：

$$
M = \sqrt{G_x^2 + G_y^2}
$$

- 梯度方向使用 `atan2(Gy,Gx)`。
- 内部方向单位为 radians，方向差归一化到 `[-pi, pi)`。
- Canny mask 决定哪些位置可成为模型点；Sobel 幅值和方向用于模型描述及匹配评分。
- 模型创建和搜索图像必须使用一致的灰度、平滑、Canny、Sobel、边界和金字塔参数。
- 零幅值、无效方向、图像边界和出界采样必须有明确行为，并通过单元测试覆盖。

---

## 模型构建

1. 对模板 ROI 执行统一 Canny/Sobel 预处理。
2. 仅从 Canny mask 中选择候选点，并要求 Sobel magnitude 不低于 `min_gradient_magnitude`。
3. 将点转换为相对 reference origin 的坐标，保存方向、归一化权重和所在金字塔层。
4. 候选点先按 magnitude 排序，再按网格或最小点距执行空间均匀采样。
5. 超过 `max_model_points` 时优先保留梯度强、空间覆盖均匀的点。
6. 少于 `min_model_points` 时创建失败并返回明确错误。
7. `ShapeModel` 保存各层模型点、ROI、origin、模型边界和参数快照。
8. 公共稳定布局与后续 SIMD 使用的 SoA 布局分离。
9. 首阶段不保存模型文件；序列化另立阶段。

模型点至少包含：

- `relative_x`；
- `relative_y`；
- `orientation`；
- `weight`；
- `level`。

---

## 金字塔与搜索策略

1. 使用 `cv::pyrDown` 构建图像金字塔，层间缩放固定为 `0.5`。
2. `num_levels` 包含原图层；自动层数受最小图像尺寸和最小有效模型点数限制。
3. 模型层使用与图像层相同的缩放规则。
4. 最粗层按 angle range/step 生成旋转候选，在 search ROI 内进行全局或受限平移搜索。
5. 只保留 top-K 候选。
6. 向下一细层回溯时，坐标乘 2，并在 `refinement_radius` 内局部搜索。
7. 角度候选覆盖 `angle_start` 到 `angle_start + angle_extent`；跨越角度边界时先规范化再排序。
8. 首阶段不做 scale 搜索；模型和结果 scale 固定为 `1.0`。
9. 候选生成、评分、细化和结果合并在确定性模式下必须稳定。

---

## Score 定义

对模型点 `p_i` 应用旋转和平移变换 `T(x, y, theta)`，在场景梯度场采样方向：

$$
S(x,y,\theta)=
\frac{1}{N_{valid}}
\sum_{i \in valid}
 w_i\max\left(0,\cos\left(\theta_i^{model}-\theta_i^{image}\right)\right)
$$

实现前必须冻结以下细节：

- `w_i` 是否为归一化模型梯度幅值权重；
- 分母使用有效点数量还是有效权重；
- 最小有效点比例；
- 首版使用最近邻还是双线性采样；
- 出界点如何处理；
- 不命中 Canny mask 的点是否无效；
- score 是否允许负值。

当前建议的首版规则：使用归一化权重、最近邻采样、仅统计落在图像内且命中 Canny mask 的点、无有效点返回 `0`，最终 score 限制在 `[0, 1]`。

标量 score kernel 是 SIMD 和多线程优化的精度基线。

---

## 候选排序与 NMS

1. 所有候选先按 score 降序生成。
2. 相同 score 使用 row、column、angle 进行稳定 tie-break。
3. 首版使用变换后模板 ROI 的轴对齐 bounding box 计算 overlap，并明确这是近似实现。
4. overlap 定义为两个框交集面积除以较小框面积。
5. 依序保留高分候选，抑制与已保留结果 overlap 大于 `max_overlap` 的候选。
6. `max_overlap = 0` 表示不允许交集；`max_overlap = 1` 允许完全重叠。
7. NMS 完成后再应用 `num_matches`。
8. 旋转矩形或模型点覆盖率 NMS 作为后续改进方向。

---

## 分阶段实施与验收

### Phase 0：路线和工程冻结

#### 任务

- 更新根 README、`shape_match/README.md` 和 `plan_1.md`，消除 Python/C++ 冲突。
- 建立 CMake、OpenCV 依赖查找、Debug/Release、测试和 benchmark 目标。
- 建立安装目录、导出头文件和版本号约定。

#### 验收

- 干净构建成功；
- `ctest` 可以运行；
- 空库和最小示例可以链接；
- 依赖缺失时错误信息明确。

### Phase 1：Core 数据结构

#### 任务

- 实现 Image/ImageView、EdgePoint/ModelPoint、Pose、ShapeModel、MatchResult；
- 实现 ShapeModelParams、SearchParams 和错误类型；
- 冻结坐标、角度、所有权、移动/复制和线程安全规则。

#### 验收

- 构造和默认值测试；
- 非法参数和空对象测试；
- ROI view 生命周期测试；
- 坐标与角度归一化测试。

### Phase 2：Canny/Sobel Edge Engine

#### 任务

- 实现灰度转换、可选平滑、Canny mask 和 Sobel `Gx/Gy/magnitude/orientation`。

#### 验收

- 水平、垂直、正负 45 度合成边缘；
- 验证 Canny 边缘位置；
- 验证 Sobel 方向符号和幅值非负；
- 验证图像边界、参数合法性和重复运行稳定性。

### Phase 3：Shape Model Builder

#### 任务

- 实现 ROI/reference origin；
- 实现候选点筛选、空间均匀采样、模型点元数据和模型层。

#### 验收

- 模型点数量上下限；
- 最小点间距；
- reference origin 改变后的坐标；
- 低纹理模板失败；
- 各层模型点可视化检查。

### Phase 4：Pyramid Engine

#### 任务

- 实现图像/模型金字塔；
- 实现层数自动计算、坐标映射和层间候选传递。

#### 验收

- 每层尺寸；
- 奇数尺寸和最小尺寸；
- 模型点坐标缩放；
- 层间映射误差；
- 层数为 1、自动和非法值的行为。

### Phase 5：Matching Engine

#### 任务

- 实现旋转候选；
- 实现粗到细平移搜索；
- 实现 Canny/Sobel score；
- 实现 top-K 和结果转换。

#### 验收

- 固定合成模板的平移；
- 单一旋转；
- 多目标；
- 负样本；
- 噪声；
- 边界目标；
- 相似干扰物。

### Phase 6：NMS 和结果处理

#### 任务

- 实现候选排序、AABB overlap、NMS、`num_matches` 和绘制接口。

#### 验收

- 无重叠、完全重叠和部分重叠；
- 不同角度候选；
- `max_overlap = 0/1`；
- 结果数量和确定性排序。

### Phase 7：性能基线与优化

#### 任务

1. 测量标量单线程基线；
2. 优化模型为 SoA；
3. 优化 score kernel；
4. 复用预处理结果；
5. 并行角度或候选搜索；
6. 增加 CPU runtime dispatch。

#### 验收

- 优化前后结果数量一致；
- 排序一致；
- row/column/angle 在容差内一致；
- score 最大误差在冻结容差内；
- 单线程和确定性多线程结果稳定。

### Phase 8：扩展与发布

- 亚像素、scale、polarity、遮挡和持久化分别立项；
- 建立 HALCON-like API 兼容矩阵；
- 配置 CMake install/export、版本、ABI、Linux/Windows 和 portable fallback。

---

## 测试与验收矩阵

### 单元测试

- 图像读取失败、空图像、灰度/彩色输入、dtype 和通道错误；
- Canny 阈值、aperture 和 L2 gradient 合法性；
- Sobel kernel、border、已知方向边缘；
- reference origin、坐标变换和角度归一化；
- 金字塔尺寸和层级映射；
- 模型点最小/最大数量、采样距离和低纹理失败；
- score 完全一致、相反方向、部分出界、无有效点和阈值边界；
- overlap/NMS 完全重叠、部分重叠、无重叠、排序和结果数量。

### 集成测试

使用固定随机种子生成数据，覆盖：

- 平移；
- 旋转；
- 多实例；
- 噪声；
- 局部遮挡；
- 图像边界；
- 相似干扰物；
- 纯背景负样本。

每个场景必须冻结：

- 预期检测数量；
- 位置误差阈值；
- 角度误差阈值；
- score 下限；
- 允许误检数；
- 运行时间上限。

### 回归和样例

- `data/` 中现有图像先作为 smoke test；
- `gt/` 补充模板对应关系、中心、角度和实例数量；
- 同一输入重复运行，结果数量、排序和数值稳定；
- 若环境具备 HALCON，只比较检测趋势、位置/角度误差和 score 单调性，不要求逐结果一致。

---

## Benchmark 协议

性能验收必须固定以下条件：

- 图像尺寸：640x400、1280x1024、2000x2000；
- 模板尺寸；
- 模型点数量；
- 金字塔层数；
- 角度范围和角度步长；
- 搜索 ROI；
- 目标数量；
- CPU 型号；
- OpenCV 版本；
- 编译器和 Release 配置；
- 线程数；
- 预热次数和重复次数。

报告至少包含：

- p50/p95/p99 latency；
- 峰值内存；
- 图像读取耗时；
- 预处理耗时；
- 搜索耗时；
- NMS 耗时；
- 总耗时；
- 吞吐量。

`2000x2000 <10ms` 只能作为指定硬件、固定模型点数、固定层数、角度范围、线程数和 Release 配置下的候选目标，不能在条件冻结前作为总验收标准。

---

## 首批实现顺序

计划确认后，按以下顺序进入代码开发：

1. CMake 和依赖配置；
2. 公共错误类型和参数结构；
3. Image/ImageView；
4. EdgePoint/ModelPoint/MatchResult；
5. Canny/Sobel Edge Engine；
6. ShapeModel 和模型点构建；
7. 图像/模型金字塔；
8. 标量 score kernel；
9. 旋转平移 Matcher；
10. 候选排序和 NMS；
11. 集成测试和示例；
12. benchmark；
13. SoA、SIMD 和多线程优化。

---

## 参考资料

- [HALCON shape-based matching documentation](https://www.mvtec.com/doc/halcon/13/en/toc_matching_shapebased.html)
- [Open-source shape-based matching reference](https://github.com/meiqua/shape_based_matching)
- [Shape matching paper](https://ieeexplore.ieee.org/document/4582953/)

## 计划决策记录

- 首阶段正式采用 C++17 + OpenCV，不以 Python 原型为交付物。
- Canny 用于边缘 mask，Sobel 用于梯度幅值与方向。
- 首阶段尺度固定为 1.0；亚像素、遮挡、持久化和完整 HALCON 兼容延期。
- 标量单线程正确性优先于 SIMD 和多线程；所有优化必须有结果一致性回归。
- 当前文档只定义实施计划，不包含代码实现。
