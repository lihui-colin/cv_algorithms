# 梯度方向得分精定位对照

2026-09-16。结论：本轮纯方向得分实验未改善 HALCON 一致性，生产默认算法未修改。
完整误差与优化前后得分见 [orientation_measurements.json](orientation_measurements.json)。

## 受控方案

从同一基线最终位姿开始，固定模型点集合，直接在变换后的模型点位置双线性采样图像梯度。
目标为模型旋转法线与归一化图像梯度的平均点积。弱梯度（幅值小于3）和越界位置贡献0，分母保持模型点总数。
此得分不是原来的距离衰减×方向一致性，也不宣称等于 HALCON 实际得分。
仅支持当前样例 use_polarity 和自由等比尺度。

四组对照：场景中心差分/高斯导数 sigma=0.8；模板原法线/同类梯度采样更新法线。
同类模板方案只更新方向，保持模板点坐标不变。逐坐标局部搜索由0.5px等效步长逐次减半到约1e-5px；每级最多100轮。
平移相对种子限制2px、角度0.05rad、log尺度0.05，并保留原参数范围容差。
不使用 GT 初始化或挑选候选，不重新做 NMS，因此7个保留实例不代表独立召回测试。
这是局部搜索，不保证全局最优；本实验失败不能排除所有梯度方向算法。

## 结果

| 场景梯度 / 模板法线 | 位置 RMSE px | 最大误差 px |
|---|---:|---:|
| 中心差分 / 原法线 | 0.663459 | 1.181723 |
| 中心差分 / 同类采样 | 0.665973 | 1.189863 |
| 高斯导数 / 原法线 | 0.575347 | 1.086057 |
| 高斯导数 / 同类采样 | 0.573719 | 1.084210 |

基线为0.300350px，四组均退化。全部28次局部优化的方向得分均不下降，但最终位置一致性变差。
这进一步表明“优化某个看似合理的目标”不等于“复现 HALCON 输出”，不应将这一方案推广为生产精定位。
不能根据这张图事后拟合距离与方向的混合权重来宣称解决问题。

## 下一步所需对照

现有 HALCON 参数导出尚未列出实际 subpixel、min_contrast、原点和尺度步长等信息。
准备了 `scripts/export_halcon_subpixel_modes.hdev.txt`，插入原 HDevelop 程序中 ModelIDs、SearchImageReduced 及搜索范围已经设置完成、模型尚未释放的位置。
只修改新的已存在输出目录；保留原模板、ROI、搜索图及其他参数。程序输出实际参数及四种精定位模式的17位结果，并在正常完成后恢复原subpixel设置。
同名文件会覆盖；异常中断时可能尚未恢复设置。未在 HALCON 运行时执行验证。

四种模式为 interpolation、least_squares、least_squares_high、least_squares_very_high。
比较这些输出可判断当前 GT 对应哪个行为，以及提高最小二乘精度是否改变位姿；不能据此唯一还原内部算法。

## 本地复现

```bash
cd shape_match_v1
bash scripts/build.sh
SHAPE_MATCH_NUM_THREADS=8 ./build/halcon_parameter_probe data results-orientation-new
python3 scripts/compare_gt.py results-orientation-new/orientation_gaussian_matched.csv \
  --reference data/reference/matching_results.csv --max-rmse 0.1 --max-error 0.2
```

最后一条预期精度验收失败。新实验实现在 tests/orientation_probe.hpp，独立得分记录在orientation_scores.csv。
