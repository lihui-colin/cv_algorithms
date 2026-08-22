# Plan: HALCON-Like Shape Matching

在当前空仓库中建立一个 Python + OpenCV 的 HALCON 风格形状匹配实现。
首阶段用边缘/梯度方向模型、图像金字塔、旋转候选、粗到细搜索和多目标
NMS 完成可重复闭环；随后按优先级加入尺度、极性策略、亚像素精化、遮挡
评分和模型持久化。目标是兼容公开算法语义与参数，不宣称复刻 HALCON
未公开的内部实现。

## References

- [HALCON shape-based matching documentation](https://www.mvtec.com/doc/halcon/13/en/toc_matching_shapebased.html)
- [Open-source shape-based matching reference](https://github.com/meiqua/shape_based_matching)
- [Shape matching paper](https://ieeexplore.ieee.org/document/4582953/)

## Steps

1. 建立 Python 包和最小运行环境，确定 OpenCV、NumPy 依赖，定义 `ShapeModel`、`MatchResult` 和匹配参数的数据结构；同时补充 README 的算法说明、安装和运行示例。
2. 实现模型创建：从模板 ROI 提取灰度梯度，按 `contrast/min_contrast` 选择有效边缘点，保存点的相对坐标、归一化梯度方向、参考原点，并建立多层图像/模型金字塔；提供模型检查输出，便于观察每层保留的模型点。
3. 实现基础搜索：对输入图像建立金字塔，在最高层枚举允许的旋转角候选，在候选位置计算模型点与搜索图像梯度方向的一致性分数，按阈值保留候选，再逐层回溯和局部精化到原图分辨率。
4. 实现结果处理：支持 `min_score`、`num_matches`、`max_overlap`，使用模型变换后的包围区域计算重叠并做按分数排序的抑制；返回 row、column、angle、score，并提供结果绘制接口。
5. 加入合成数据和单元测试：固定模板、平移/旋转实例、无目标负样本、多实例和噪声/遮挡样本，验证位置误差、角度误差、召回率、误检率和重复运行稳定性。
6. 在基础闭环通过后按独立阶段扩展：各向同性尺度搜索对应 `create_scaled_shape_model/find_scaled_shape_model`；`use_polarity/ignore_global_polarity`；插值或局部优化的亚像素姿态；遮挡比例和边界模型；最后加入模型 JSON/NPZ 持久化与参数查询。

## Relevant Files

- `shape_match/` — 建议的新 Python 包目录，承载模型、金字塔、评分、搜索和结果处理模块。
- `tests/` — 合成图像、检测结果和误差阈值测试。
- `examples/` — 模型创建、搜索和可视化示例。
- `pyproject.toml` — 依赖、测试和可选命令行入口。

项目说明和算法范围见 [README.md](README.md)。

## Verification

1. 用固定随机种子生成模板和变换实例，确认基础平移与旋转实例能被检出，位置误差和角度误差分别低于预设阈值。
2. 在负样本上确认没有超过 `min_score` 的结果；在多实例图像上确认 `num_matches` 和 `max_overlap` 生效。
3. 在不同金字塔层数、边缘阈值和噪声水平下运行测试，检查模型点数、召回率和运行时间变化。
4. 对同一输入连续运行多次，确认输出排序和数值稳定。
5. 后续若环境具备 HALCON，使用相同模板和测试图分别运行 `create_shape_model/find_shape_model`，比较位置、角度、分数趋势，不要求逐结果完全一致。

## Decisions

- 默认技术栈为 Python + OpenCV + NumPy，因为仓库为空且需要先验证算法；若后续部署有性能要求，再迁移核心搜索到 C++。
- 首阶段默认包含平移、旋转、多目标、金字塔和梯度方向评分；缩放、遮挡、亚像素、持久化属于后续阶段，避免一次引入过多变量。
- 采用公开可解释的边缘/梯度方向匹配，不复制 HALCON 的专有实现；文档中明确“行为语义相近”与“内部结果等价”的区别。
- 优先使用合成数据建立回归基线，因为当前没有真实图像、标注或 HALCON 输出。
- 模型 ROI 的参考原点默认使用 ROI 中心，API 预留自定义 origin，以对应 `set_shape_model_origin`。

## Further Considerations

1. 需要在实现开始前确认是否允许安装 Python 依赖；默认优先检查当前环境是否已有 `numpy` 和 `cv2`。
2. 若真实目标是直接替换 HALCON 生产系统，需要额外定义吞吐量、延迟、许可证替代要求和真实图像基准，这不属于首个算法原型的验收范围。
3. HALCON 的 `ignore_local_polarity`、clutter model、deformation 和 least-squares 精化会显著扩大实现与测试范围，应在基础版本有基线后分别加入。
