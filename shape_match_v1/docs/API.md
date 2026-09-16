# API 与兼容范围

版本：本工程 0.1.0；目标算子版本 HALCON 24.11；核对网页为 MVTec 24.11.3.0 文档。

## 1. 范围与边界

九个公开匹配算子均有工作实现，声明见 `include/shape_match/shape_match.hpp`。函数名、参数顺序、输入输出形式采用 HALCON C++ 过程式形式。使用 `shape_match` 命名空间。

本工程的 `HObject`、`HTuple`、`HHandle` 是独立实现的必要子集，不能与 `HalconCpp` 对象/句柄混传。无 HALCON SDK 二进制 ABI 兼容声明，不执行 `.hdev`，不读取 HALCON 私有模型文件，不实现区域生成、相机采集或窗口算子。

本次交付是可运行的独立算法实现，而非所有 HALCON matching 功能的等价替换。公开签名的对齐已经落实；HALCON 运行时逐参数/边界行为差分尚未执行。下列差异明确记录，不能称为已验证完全兼容。

## 2. 九个公开函数

| HALCON 算子 | 本工程 C++ 函数 | 功能 | 主要约束 |
|---|---|---|---|
| create_generic_shape_model | CreateGenericShapeModel | 创建未训练模型与唯一标识 | 输出指针非空 |
| set_generic_shape_model_param | SetGenericShapeModelParam | 原子设置参数元组 | 一次一个模型；名称和值数量相同 |
| get_generic_shape_model_param | GetGenericShapeModelParam | 查询实际值、原始值和状态 | 一次一个模型；支持多名称 |
| train_generic_shape_model | TrainGenericShapeModel | 建立金字塔轮廓模型 | 一个灰度图模板；Domain 非空 |
| get_generic_shape_model_object | GetGenericShapeModelObject | 返回模型原点相对轮廓 | 当前只支持 contours |
| find_generic_shape_model | FindGenericShapeModel | 全图多模型多实例搜索与精定位 | 一个搜索图，可携带 Domain |
| get_generic_shape_model_result | GetGenericShapeModelResult | 返回字段元组 | 一次一个字段 |
| get_generic_shape_model_result_object | GetGenericShapeModelResultObject | 返回匹配后的轮廓 | 当前只支持 contours |
| clear_shape_model | ClearShapeModel | 释放并失效一个或多个模型 | 重复清理明确报错 |

所有 void 过程式接口通过输出参数返回结果，错误使用 `HException`。`ErrorCode` 是本工程错误类别，不冒用 HALCON 的数字错误码。

## 3. 输入适配

```cpp
// 应用侧已有灰度内存；像素值范围 0..255。
HObject object = HObject::FromGray(width, height, pixels, mask);
// 或字节图；mask 省略时使用完整图域。
HObject object = HObject::FromBytes(width, height, bytes);
// 示例 I/O，不属于 HALCON 兼容算子。
HObject object = ReadPgm("template.pgm", "mask.pgm");
```

图像对象拥有像素副本，避免调用方临时内存悬空。mask 的非零值表示有效像素。模板 Domain 用于确定原点和保留特征；搜索 Domain 约束模型默认参考位置，而不会把目标轮廓截断在 Domain 内。

PNG 由 `scripts/prepare_samples.py` 转为 PGM，alpha 独立转为 Domain，未缩放、未二值化原始灰度。核心不做文件解码推断。读取 PNG 使用 Pillow，不需要 OpenCV。

## 4. 参数表

| 参数 | 默认 | 实现行为 |
|---|---|---|
| model_identifier | 自动唯一字符串 | 非空，不能为 all/best；多模型搜索时不得重复 |
| contrast_low / contrast_high | auto / auto | 数值 0..255，滞后阈值；一侧 auto 时与另一侧一致 |
| min_size | auto | 本实现自动取 4；按层缩减连通边缘最小点数 |
| metric | auto | 灰度图解析为 use_polarity；另支持 ignore_global_polarity / ignore_local_polarity |
| optimization | auto | 本实现自动取 point_reduction_medium；还支持 none/low/medium/high 点缩减 |
| num_levels | auto | 本实现根据轮廓半径估计；显式 1..8，必须有足够特征 |
| angle_step | auto | 根据半径估计，实际最大 0.20 rad |
| iso_scale_min / iso_scale_max | 1 / 1 | 等比尺度范围；当前资源边界 0.05..20 |
| iso_scale_step | auto | 根据模型半径估计 |
| angle_start / angle_end | 0 / 2π | 使用弧度；连续有序区间，跨度不超过 2π |
| restrict_iso_scale_min / max | auto | 限制训练尺度区间；auto 表示不额外限制 |
| min_score | 0.5 | 0..1；本工程独立评分公式 |
| num_matches | all | all 或 0 表示全部；正整数限制每模型最终返回数 |
| max_overlap | 0.5 | 旋转最小矩形交集 / 较小矩形面积 |
| max_overlap_global_enable | false | true 时跨模型抑制，使用各模型最小重叠阈值 |
| min_contrast | auto | 搜索边缘最小梯度；自动值根据模板噪声估计 |
| greediness | 0.9 | 改变粗层候选保留阈值；非 HALCON 内部启发式复刻 |
| strict_boundaries | false | true 时最终严格过滤角度/尺度，发生在数量限制后 |
| subpixel | least_squares | none / interpolation / least_squares / least_squares_high / least_squares_very_high |
| pyramid_level_highest | auto | 搜索起始层，限制在已训练层数内 |
| pyramid_level_lowest | 1 | 终止层；大于 1 时跳过原分辨率精定位，精度降低 |
| pyramid_level_robust_tracking | false | 仅 false 已实现；true 明确报 Unsupported |
| border_shape_models | system | 独立运行时 system=不允许越界；true/false 可显式指定 |
| origin_row / origin_column | 0 / 0 | 相对于训练图域重心的偏移 |
| prepare_contours_for_visualization | true | false 时轮廓查询明确报错 |

表中 HALCON 默认值参照官方文档；自动估计、评分、点缩减数量和启发式为本工程实现，数值不能保证与官方相同。资源边界是独立实现限制，不宣称等于 HALCON 合法值全集。

训练参数变化后 `needs_training` 变为 true；仅搜索角度、数量、得分等变化不会重训练。min_contrast 改为 auto 时需要重训练。当前对训练参数重复设置同值也会标记重新训练（已知行为差异）。

查询支持 `needs_training`、`has_samples`（false）、`scale_type`，以及已支持参数的 `_param` 原始值形式。例如训练后 `num_levels` 为实际层数，`num_levels_param` 仍可能为 auto。

## 5. 精定位语义

| subpixel | 当前实现 |
|---|---|
| none | 最终平移量化为像素，角度量化到 angle_step |
| interpolation | 局部得分的逐坐标二次插值 |
| least_squares | 点到法线最小二乘，最多 10 轮 |
| least_squares_high | 同一目标函数，最多 20 轮 |
| least_squares_very_high | 同一目标函数，最多 30 轮 |

迭代预算为本实现取值，HALCON 未公开对应内部数字。优化收敛时可以提前结束；模式更高不意味着每个样例结果一定不同或一定更准。

### 实验性精定位扩展（不是 HALCON 参数）

`refinement_method` 默认 `nearest_point`，保留现有路径；可选 `contour`（局部轮廓线段投影）、
`gradient`（中心差分梯度的连续法向剖面）和 `gradient_gaussian`（高斯导数与连续法向剖面）。
仅在 `least_squares*` 且搜索到原分辨率时启用。`refinement_radius` 默认 1.5，合法范围
0.5～10 像素；新路径前三轮搜索半径至少为 3 像素。它不是 HALCON 的 `max_deformation`。
这两个参数可在训练后修改，不触发重训练；当前训练会预计算高斯精定位特征，即使使用默认模式。

新路径以默认精定位结果初始化，使用归一化四自由度求解和含缺失点惩罚的重新关联验收。
当新位姿评分低于 `min_score`、但原位姿仍满足阈值时，回退完整原位姿和评分。
固定尺度仍受原范围约束；这些机制不保证任意输入的精度或召回不退化。
新路径未通过默认替换验收，实图、独立合成集和性能结果见
[精度实施报告](PRECISION_IMPLEMENTATION.md)。

## 6. 结果与坐标

字段支持 `num_match_result`、`model_identifier`、`row`、`column`、`angle`、`scale_row`、`scale_column`、`score`、`hom_mat_2d`。矩阵每个实例返回连续 6 个值。

MatchSelector 支持：0 起始整数索引、all、best、参与本次搜索的模型标识、参与本次搜索的模型句柄，以及这些选择器的组合。结果求并集，按已有分数降序返回，不随选择器顺序重排。同分顺序确定但不保证等于 HALCON。同一图中没有目标时仍返回有效结果句柄和数量 0。

内部 x=column，y=row，角度正方向为图像顺时针；输出 angle 取相反数并规范到 ±π。轮廓采用像素中心坐标。

设内部匹配原点为 o=(row,column)，包含用户原点偏移；A 是行列坐标下的旋转缩放矩阵。为了匹配 XLD 的坐标转换约定，结果变换为：

```text
H = [ A, o + (0.5,0.5) - A*(0.5,0.5) ]
q_pixel = H * (p_model_pixel + 0.5) - 0.5
```

本实现 row/column 输出 H 的平移分量，hom_mat_2d 与轮廓使用同一 H；不是对内部坐标无条件加 0.5。纯平移自匹配输出训练参考位置。这修正了前期泛化方案里简单半像素转换的不完整描述。上传 CSV 的两枚建模目标与该约定相符；实际 HALCON 非零旋转/原点变化的逐项对照仍须在用户 HALCON 环境执行。

修改模型参数不会改变已有结果，模型释放后结果仍可读。HTuple 最后一个结果句柄被销毁时，结果由 shared_ptr 自动释放，无需发明新的公开 clear_result 算子。

## 7. 必须保留的已知差异

- 不支持彩色 metric、XLD 输入训练、各向异性尺度、形变、clutter、扩展样本训练或私有模型序列化。
- 不支持一个图像对象元组为各模型提供不同 Domain；当前一个搜索图由所有模型共享。这不影响用户当前全图任务。
- 参数 setter 当前一次一个模型；类型容器不是完整 HALCON HTuple/HObject。
- 近似最近边缘图和启发式候选束会影响召回；粗层候选 512、跟踪 384、原分辨率 192 的上限不是无限实例保证。
- `num_matches` 当前在末端筛选；HALCON 也可能在中间层按数量丢弃候选。因此两者设置 num_matches 后的结果可能不同。
- Domain 光栅化、图像边界和金字塔边缘处理不是完整 HALCON 算子复刻。
- 模型轮廓可视化连接方式不同，轮廓条数、顺序、评分和实际位姿均不保证数值等同。
- 查询 match/time_measurement 等字典类型未支持，明确报错；本工程计时通过独立 GetSearchDiagnostics 获取。
- HALCON 运行时未安装，未宣称通过其完整兼容认证或 10 ms 性能验收。

## 8. 官方依据

- [CreateGenericShapeModel](https://www.mvtec.com/doc/halcon/2411/en/create_generic_shape_model.html)
- [SetGenericShapeModelParam](https://www.mvtec.com/doc/halcon/2411/en/set_generic_shape_model_param.html)
- [GetGenericShapeModelParam](https://www.mvtec.com/doc/halcon/2411/en/get_generic_shape_model_param.html)
- [TrainGenericShapeModel](https://www.mvtec.com/doc/halcon/2411/en/train_generic_shape_model.html)
- [FindGenericShapeModel](https://www.mvtec.com/doc/halcon/2411/en/find_generic_shape_model.html)
- [GetGenericShapeModelObject](https://www.mvtec.com/doc/halcon/2411/en/get_generic_shape_model_object.html)
- [GetGenericShapeModelResult](https://www.mvtec.com/doc/halcon/2411/en/get_generic_shape_model_result.html)
- [GetGenericShapeModelResultObject](https://www.mvtec.com/doc/halcon/2411/en/get_generic_shape_model_result_object.html)
- [ClearShapeModel](https://www.mvtec.com/doc/halcon/2411/en/clear_shape_model.html)
- [2D coordinate conventions](https://www.mvtec.com/doc/halcon/2411/en/toc_transformations_2dtransformations.html)
