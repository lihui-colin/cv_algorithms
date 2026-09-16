# 精定位初值与残差形式对照

目标仍是对齐 `data/reference/halcon_least_squares.csv` 的普通 least_squares，
不是改用 very_high 作为验收真值。本轮只增加诊断能力，未修改生产默认算法。
数值记录见 [refinement_seed_measurements.json](refinement_seed_measurements.json)。

## 实验设计

在真实搜索实现的精定位入口改变初值，保持模板、场景边缘、对应点构建及参数一致：

- `native`：复现版原始初值。
- `score_interpolation`：先执行复现版自身得分插值，不读取 HALCON 位姿。
- `SEED_CSV`：使用 HALCON 导出的 interpolation 位姿，仅用于隔离初值影响。
  这不证明该导出位姿就是 HALCON 内部最小二乘初值，也不属于独立检测精度。
- 默认残差为点到法线；可选 `point_to_point` 同时拟合对应点的两个坐标分量。

点到点分支仅在诊断编译宏内启用；残差、雅可比及步长接受损失一起切换。
两类残差均使用原来的最近邻对应点，没有据 GT 修改边缘或选取每个目标的最佳迭代。

## 充分迭代的结果

以下统一使用 30 次迭代上限和原生对应半径调度；均检出 7/7。

| 初值 | 残差 | 位置 RMSE / px | 最大误差 / px |
|---|---|---:|---:|
| 原生（既有基线） | 点到法线 | 0.300366 | 0.578889 |
| 自身得分插值 | 点到法线 | 0.300366 | 0.578889 |
| HALCON 插值（仅诊断） | 点到法线 | 0.300366 | 0.578889 |
| 原生 | 点到点 | 0.300063 | 0.584118 |
| HALCON 插值（仅诊断） | 点到点 | 0.299803 | 0.599248 |

结论限于当前图像和参数：仅改变初值不能改变点到法线优化最终结果；
仅改为点到点也没有实质改善，最大误差反而增加。不应将这些分支晋升为生产默认。

## 必须同时看漏检与过滤前轨迹

复现版评分会过滤部分导入的 HALCON 位姿；双方评分数值不能直接等同。
过滤前统计按模型和 3px 空间门限关联实例，在每个实例的候选中选复现版最终评分最高者，
不是选离 GT 最近的候选。该统计只帮助解释过滤行为，不能替代完整检测验收。

| HALCON 插值初值，点到法线 | 最终保留 | 保留结果 RMSE / px | 过滤前 7 个目标 RMSE / px |
|---|---:|---:|---:|
| 0 次更新 | 3/7 | 0.045576 | 0.606922 |
| 1 次更新 | 5/7 | 0.122861 | 0.187373 |
| 4 次更新 | 7/7 | 0.282118 | 0.282118 |
| 30 次上限 | 7/7 | 0.300366 | 0.300366 |

不能把前两行的局部低误差称为精度提升。自身插值初值的 4 次更新虽达到
7/7、RMSE 0.277303，但最大误差仍为 0.577676px，且仅单图证据不足以选择固定早停次数。

## 复现命令

在 shape_match_v1 目录执行（输出前缀会被覆盖，勿使用 reference/results 路径）：

```bash
cmake -S . -B /tmp/shape-seed-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/shape-seed-build -j 8
SHAPE_MATCH_NUM_THREADS=8 /tmp/shape-seed-build/refinement_trajectory \
  data /tmp/seed-local-30 30 0 score_interpolation
SHAPE_MATCH_NUM_THREADS=8 /tmp/shape-seed-build/refinement_trajectory \
  data /tmp/seed-halcon-30 30 0 data/reference/halcon_interpolation.csv
SHAPE_MATCH_NUM_THREADS=8 /tmp/shape-seed-build/refinement_trajectory \
  data /tmp/point-native-30 30 0 native point_to_point
python3 scripts/compare_gt.py /tmp/seed-local-30.csv \
  --reference data/reference/halcon_least_squares.csv --max-rmse .1 --max-error .2
```

最后的验收命令预期返回失败。完整参数：
`DATA OUTPUT_PREFIX [BUDGET [RADIUS_OR_0 [native|score_interpolation|SEED_CSV [point_to_point]]]]`。
`0` 半径表示原生 3px → 1.5px 调度，省略预算时使用模式原生预算。
输出 `.csv` 是最终过滤后结果；`.trace.csv` 保留所有精定位候选。

## 接下来的收敛方向

本轮排除了两个简单替换方案，尚未证明 HALCON 的内部实现细节。
下一轮应隔离**对应点选择与更新接受过程**：在相同初值下记录有效点集合、
更新前后重新关联损失、归一化位姿步长；对照固定对应点与每轮重新关联。
重点观察普通/high/very_high 的位姿变化能否由同一更新规则与终止条件解释，
而不是为七个目标分别选迭代次数或写入 GT 偏移。
