# 精度提升实施与验收记录

日期：2026-09-12。结论：完成可运行的连续精定位路径和对齐/回归工具，**尚未完成实图精度与 30ms 联合目标**。
默认仍为 `nearest_point`；没有根据 GT 坐标、目标序号或模型名称修正匹配输出。
可复核的逐目标误差、合成集分组统计及 30 次计时原始样本见 [precision_measurements.json](precision_measurements.json)。

## 1. 计划落实范围

| 阶段 | 已实现 | 尚未完成 |
|---|---|---|
| P0 对齐与基线 | 17 位输出；参数、Domain 重心、模型轮廓、变换矩阵导出；旋转/尺度/原点和裁剪域等价测试；独立 GT 关联及精度门限 | HALCON 侧实际运行、全精度结果与真实建模参数的逐项差分 |
| P1 连续边缘 | 可分离高斯导数；模板与场景使用同类连续法向剖面；梯度与边界单测 | 更多真实成像条件及掩码边缘支持范围的验证 |
| P2 连续对应 | 局部相邻边缘线段投影；连续梯度峰定位；重新关联后的缺失点惩罚 | 全局拓扑/XLD 样条对应、复杂交叉轮廓的可靠关联 |
| P3 分组诊断 | 连通轮廓独立拟合与分组 CSV，暴露内外轮廓不一致 | 经独立数据验证的分组权重；未上线按 GT 调权 |
| P4 稳定求解 | 参数单位归一化、小型 LM 求解、步长限制、重新关联验收、低分回退；候选去重保留相隔较大的角度分支 | 完整条件数/可观测性判定、统一的对称分支比较策略 |
| P5 回归与性能 | 70 案例噪声/模糊/遮挡/局部形变回归；缓存剖面采样、共享场景梯度、精定位前去重；Release/ASan/UBSan 检查 | 新模式 30ms 性能验收及默认替换验收 |

新算法集中在 `src/precision.cpp`。粗搜索仍使用原稀疏模型，只在原分辨率精定位阶段启用新观察方式。
梯度剖面采用三次插值；局部响应为正且动态范围合适时插值其对数，否则回退到梯度矢量插值。
以三点二次峰估计细化法向位置，避免直接对原始幅度做插值引入较大的相位偏差。
这属于本工程的数值方案，不声称是 HALCON 私有实现。

## 2. 样图对包内 HALCON 参考

指标为输出 row/column 的二维欧氏误差；只在评估层修正历史 GT 的模型标签。
三种模式均为全图搜索、检出 7/7，无额外实例；`subpixel=least_squares_very_high`。

| 模式 | 半径 px | RMSE px | 最大误差 px | 结论 |
|---|---:|---:|---:|---|
| 默认 nearest_point | 1.5 | 0.300350 | 0.579015 | 保持当前默认结果 |
| contour | 3 | 0.233226 | 0.491953 | 总体改善约 22.3%，但部分目标退化 |
| gradient_gaussian | 1.5 | 0.305271 | 0.585207 | 未改善参考一致性 |

三者均未达到中间门限 RMSE ≤ 0.10px、最大误差 ≤ 0.20px，更未达到最终 1/30px。
检测全部找回不等于精度通过：比较脚本分别输出 `association_passed` 和 `precision_passed`。
不指定精度门限时后者为 null，不能把脚本成功退出解读为精度已达标。

## 3. 独立合成精度与抗扰动

渲染器从连续几何生成图像并进行像素面积采样，不从 GT CSV 或匹配模型轮廓生成答案。
干净集 30 案例覆盖亚像素平移及旋转/尺度；扩展集共 70 案例，额外包含各 10 个噪声、模糊、遮挡、内孔偏移案例。

| 模式 | 干净集 RMSE / 最大 px | 扩展集 RMSE / 最大 px | 检出 |
|---|---|---|---|
| nearest_point | 0.012601 / 0.021334 | 0.040359 / 0.148340 | 70/70 |
| contour，半径 3 | 0.015836 / 0.027438 | 0.055582 / 0.180671 | 70/70 |
| gradient_gaussian | 0.001978 / 0.003320 | 0.032745 / 0.128856 | 70/70 |

高斯模式干净集 RMSE 降低约 84.3%，但不能据此推断实拍图也有该精度。
连续轮廓模式遮挡组 RMSE 从 0.039064 升到 0.107188px，是不推广为默认的原因之一。
扩展回归的分组门限：干净平移/位姿各 1/30px，噪声/模糊各 0.1px，遮挡/局部形变各 0.3px。
这些是当前回归门限，不表示所有抗扰动案例达到 1/30px。

## 4. 性能与安全检查

Release，8 工作线程，一次预热后依次运行各模式 30 次；只计搜索，不计训练、文件读写。
共享机器负载波动明显，因此记录全部样本，不能把最小值作为达标证据。

| 路径 | 中位数 ms | P95 ms |
|---|---:|---:|
| 本轮修改前构建快照 | 30.808 | 102.341 |
| 当前默认 | 42.080 | 116.452 |
| contour，半径 3 | 101.321 | 222.050 |
| gradient_gaussian | 183.429 | 295.076 |

默认路径样图数值保持一致，但本轮计时不足以证明性能无退化；需低负载下交错复测。
新路径额外训练特征、剖面采样和重新关联仍有成本，不能作为 30ms 已交付版本。
ASan/UBSan 已通过原六项 CTest（连续精定位、GT 关联、契约/几何、三组线程池测试）；
该次检查设置 `ASAN_OPTIONS=detect_leaks=0`，不宣称验证了泄漏检测。
默认和高斯模式的完整 70 案例另加入 CTest 自动回归。
最终 Release CTest 为 8/8 通过；重建后的 `./build/test_precision` 通过 1237 项断言，
`./build/test_match` 通过 184 项断言。默认样图 CSV 与本轮基线逐字节一致，
高斯模式 1 线程与 8 线程输出 CSV 也逐字节一致。
精度门限脚本已实测返回 2，并输出 `precision_passed=false`，没有把未达标结果判为通过。

## 5. 复现

从 `shape_match_v1` 目录执行；输出目录使用新的名称，避免覆盖已有基线。

```bash
bash scripts/build.sh
SHAPE_MATCH_NUM_THREADS=8 ./build/match_sample --refinement-method contour \
  --refinement-radius 3 --diagnostics --output-dir results-contour
python3 scripts/compare_gt.py results-contour/matching_results.csv \
  --bundled-labels --max-rmse 0.1 --max-error 0.2 --output results-contour/gt.json
# 当前精度未通过，上一条预期退出码为 2。
SHAPE_MATCH_NUM_THREADS=8 ./build/accuracy_bench accuracy-gaussian.csv gradient_gaussian 1.5 extended
SHAPE_MATCH_NUM_THREADS=8 ./build/performance_bench 30 performance-gaussian.json gradient_gaussian 1.5
./build/refinement_probe data groups.csv
```

`refinement_probe` 输出额外的 `groups.csv.groups.csv`，记录各连通轮廓点数、范围及独立拟合结果。
圆形内孔可能缺乏角度可观测性，分组拟合不能直接作为最终位姿。
本轮诊断发现部分内孔独立拟合与外轮廓在尺度、位置上明显不一致；仅增加迭代或换连续插值不足以消除这种差异。

## 6. 下一验收入口：HALCON 对齐数据

将 `scripts/export_halcon_alignment.hdev.txt` 程序文本粘贴到 HDevelop，配置输入路径和已存在的新输出目录。
分别使用裁剪模板域与原图 ROI 建模，导出版本、实际自动参数、Domain 重心、全精度结果、模型及变换后轮廓。
该脚本尚未在 HALCON 环境执行，可能需要按安装版本调整；输出文件会覆盖同目录同名文件。
变换后模型轮廓不是实际场景观测边缘，不能用于证明边缘拟合残差。

获得这些数据后，先验证模型坐标与轮廓定义，再选择分组权重/对称分支策略。
现阶段没有依据把实图偏差都归因于离散边缘，也不能靠硬编码平移补偿宣称精度提升。

公开原理依据：HALCON 区分插值和不同精度的最小二乘精定位，并通过最低搜索层控制精定位范围，
见 [参数文档](https://www.mvtec.com/doc/halcon/2311/en/set_generic_shape_model_param.html)。
高斯导数与亚像素边缘的公开依据见 [edges_sub_pix](https://www.mvtec.com/doc/halcon/2311/en/edges_sub_pix.html)，
但这不证明匹配算子内部直接调用该算子。
坐标和原点核查依据为 [affine_trans_pixel](https://www.mvtec.com/doc/halcon/2411/en/affine_trans_pixel.html)
及 [train_generic_shape_model](https://www.mvtec.com/doc/halcon/2411/en/train_generic_shape_model.html)。
