# 对应点更新与步长接受诊断

本轮目标：隔离对应点重关联及更新接受规则的影响。所有实验使用复现版原生初值，
不导入 HALCON 位姿；以全精度 `halcon_least_squares.csv` 验收，预算上限统一为 30。
完整逐目标结果见 [refinement_update_measurements.json](refinement_update_measurements.json)。

## 实现与实验边界

诊断可执行程序 `refinement_trajectory` 新增环境变量 `SHAPE_MATCH_TRACE_UPDATE`：

- `native`（默认）：每轮重关联，按固定当前对应点的试探损失接受更新。
- `frozen`：首次关联后，模型点、场景点和法线全部固定；不再用新对应点解增量。
- `support`：保持每轮重关联，额外要求试探位姿重新关联后的支持损失不增加。

支持损失定义为 `有效点法向残差平方和 + 缺失点数量 × radius²`，避免仅通过
丢弃困难点降低损失。它是诊断假设，不是已证实的 HALCON 公式。
不同 radius 下的支持损失不可直接比较；frozen 模式的解算对应点不随 radius 调度改变。

新增 `.updates.csv` 记录位姿边界内的求解尝试：当前/试探有效点数、固定对应点损失、
重关联损失、支持损失、阻尼、像素量级步长及是否接受。
步长取平移长度、角度增量乘模型半径、对数尺度增量乘模型半径的最大值。
越界试探、有效点不足和线性求解失败不写入该表，仍可从 `.trace.csv` 看到迭代位姿。
所有行为改动受诊断编译宏保护，不改变生产库默认行为。

## 实测结果

`0` 半径表示原生 3px → 1.5px 调度。误差单位为 px。
漏检行的 RMSE 仅覆盖保留目标，不能与完整检出行直接当作精度排名。

| 更新策略 | 半径 | 检出 | RMSE | 最大误差 |
|---|---:|---:|---:|---:|
| native | 0 | 7/7 | 0.300366 | 0.578889 |
| native | 1.5 | 7/7 | 0.300366 | 0.578889 |
| native | 3 | 6/7 | 0.234907 | 0.481233 |
| frozen | 0 | 6/7 | 0.244745 | 0.501355 |
| frozen | 1.5 | 7/7 | 0.278318 | 0.587409 |
| frozen | 3 | 6/7 | 0.244745 | 0.501355 |
| support | 0 | 7/7 | 0.300651 | 0.579646 |
| support | 1.5 | 7/7 | 0.300656 | 0.579659 |
| support | 3 | 6/7 | 0.235091 | 0.481233 |

## 轨迹揭示了什么

大螺母（参考 index 4）在原生策略下，iteration 2 使用 388 个对应点；
iteration 3 半径缩到 1.5px 后只剩 354 个对应点，接受的更新步长为 0.098969px。
该步固定对应点损失从 54.512902 降到 49.959844，重新关联支持损失也从
241.262902 降到 236.874471。因此它不是简单的“接受了损失上升步长”错误。

全体候选中，native/原生半径记录 93 次边界内试探，其中拒绝 12 次，
有 5 次已接受更新使重关联支持损失增加。support 策略消除了这种增加，
但变为 138 次试探、45 次拒绝，位置精度没有改善。
这里统计的是所有 11 条候选轨迹，不只是最终 7 个检出。

固定对应点、1.5px 半径使整体 RMSE 略降，但最大误差更大；固定 3px 对应点还会漏检。
不能据此替换生产策略。原生重关联从一开始使用 1.5px 也收敛到同一结果，
说明半径切换会改变过程，却不足以单独解释最终偏差。

## 复现

```bash
cmake -S . -B /tmp/shape-update-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/shape-update-build -j 8
for policy in native frozen support; do
  for radius in 0 1.5 3; do
    SHAPE_MATCH_NUM_THREADS=8 SHAPE_MATCH_TRACE_UPDATE=$policy \
      /tmp/shape-update-build/refinement_trajectory \
      data /tmp/update-$policy-$radius 30 $radius
  done
done
python3 scripts/compare_gt.py /tmp/update-frozen-1.5.csv \
  --reference data/reference/halcon_least_squares.csv --max-rmse .1 --max-error .2
```

最后验收命令仍应返回失败。输出前缀会被覆盖，勿指向参考数据或生产结果。
回归测试覆盖默认结果与生产版本相同、冻结模式有效点数量不变、
support 模式接受步的两种损失均不增加，以及非法策略报错。

## 结论与下一步

本轮未达到对齐目标，不修改生产默认。仅冻结对应点或收紧接受条件都不足以解决问题。
下一步应检查**对应点的观测位置与法线定义**：在相同模板、初值、半径下，
直接对照二维最近点与沿模型法线的局部交点，并保留同一有效点集合评估位姿增量。
此前法向峰值实验未改善，本轮不能据此声称新的法线观测一定有效；
需要先证明其更新方向能解释 ordinary/high/very_high 的差异，再进入生产实现。
