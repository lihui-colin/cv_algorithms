# OpenShapeMatch 0.2 HALCON-like 兼容矩阵

“兼容”仅指有限的 API 语义和参数命名相似，不表示内部算法、score 数值、排序或性能与 HALCON 相同。

| 能力 | OpenShapeMatch 0.1 | 兼容范围与限制 |
| --- | --- | --- |
| 灰度/BGR/BGRA 输入 | 支持 | 仅 8-bit，内部转 `CV_8UC1` |
| 矩形模板 ROI | 支持 | 不支持任意 mask/domain |
| Reference origin | 支持 | 默认 ROI 中心，可显式指定 |
| Canny/Sobel 模型 | 支持 | 参数公开，创建与搜索保持一致 |
| 图像/模型金字塔 | 支持 | 固定 0.5 层间尺度 |
| 平移搜索 | 支持 | 像素级最近邻采样 |
| 旋转搜索 | 支持 | 公共单位 degree，固定步长 |
| 尺度搜索 | 支持 | 正等比离散尺度，固定步长，无连续尺度插值 |
| 多目标结果 | 支持 | score 排序、AABB overlap NMS |
| `min_score` | 支持 | `[0,1]`，数值不与 HALCON 对齐 |
| `num_matches` | 支持 | `0` 表示不限制 |
| `max_overlap` | 支持 | 交集面积/较小 AABB 面积 |
| 搜索 ROI | 支持 | 矩形 ROI |
| 确定性多线程 | 支持 | 稳定结果合并和 tie-break |
| 结果绘制 | 支持 | 绘制旋转 ROI 多边形和 origin |
| 场景预处理缓存 | 支持 | `EdgeMap` / `EdgePyramid` |
| 搜索性能统计 | 支持 | `SearchStats` |
| 亚像素优化 | 支持 | `enable_subpixel=true` 在最终候选上启用连续评分与位置细化；默认关闭 |
| Polarity 模式 | 不支持 | local/global polarity 均延期 |
| 遮挡/变形模型 | 不支持 | 延期独立项目 |
| 模型持久化 | 不支持 | 延期独立项目 |
| GPU | 不支持 | 延期独立项目 |
| Python binding | 不支持 | 当前发布仅原生 C++17 |
| HALCON 模型文件互操作 | 不支持 | 不读取或写入 HALCON 模型 |

## 数据与线程契约

- `ShapeModel` 构建后只读，可被并发搜索共享。
- `EdgePyramid` 构建后应视为只读，可被并发搜索共享。
- `ImageView` 使用 OpenCV 引用计数矩阵头，不复制像素数据。
- 异常统一继承 `OpenShapeError`；空图像、非法参数和不兼容模型版本分别提供明确错误。
- 默认 `scale_min=scale_max=1.0`，保留 0.1 固定尺度结果语义；角度和尺度均在最粗层联合枚举，细层只细化位置。
- `MatchResult::scale` 返回命中的离散尺度；相同 score/位置/角度时以尺度升序作最终 tie-break。
- 0.x 版本不承诺跨 minor 版本 C++ ABI 稳定；`SOVERSION=0`。
