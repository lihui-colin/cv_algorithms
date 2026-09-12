# 样本来源

`reference/` 下的 PNG、CSV、match.hdev 是用户在当前任务提供的附件副本；其中 HALCON 示例和 rings_and_nuts 图像来自 MVTec 示例体系。它们不是本工程生成或拥有版权的原始素材。继续分发/商用时应遵循其原始许可。

PGM 文件由 `scripts/prepare_samples.py` 从这些 PNG 无缩放转换；模板 alpha 单独转换为 mask。搜索程序只读取图像和模板，不读取参考 CSV 或 recognition_result.png。

用户已确认原始 CSV 的 model 列存在历史导出映射错误。原始文件保留不变；报告比较时，参考序号 1、2、5 为 ring，其余为 nut。检测器不使用这项标签修正或参考位置。

`tests/synthetic.hpp` 生成独立的连续几何合成图，属于本项目精度测试工具。
