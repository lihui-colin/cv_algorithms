# HALCON 普通 least_squares 对齐：真实迭代轨迹

2026-09-16。主目标仍是 HALCON 普通 least_squares，不以 very_high 替换 GT。
本轮增加真实搜索实现的编译期诊断钩子，生产库不启用这些钩子，不改变默认预算或求解方式。
探针与生产样例的默认输出逐字段完全一致（本次最大数值差0）；新增自动测试使用1e-9容差验证。
完整结果和最终选中候选的轨迹见 [测量数据](refinement_trajectory_measurements.json)。

## 新证据

1. 小 nut（GT #3）进入最小二乘的角度为36.95368°，之后前三次更新为39.81847°、42.68326°、45.54805°。
   HALCON 插值输出角度约45.70137°。因此不能继续假定双方进入精定位的状态相同；但 HALCON 插值输出也不等于其最小二乘的真实内部初值，不能把二者直接等同。
2. 第4次迭代开始使用1.5px对应半径，此前为3px。
   小 nut 从 row=323.67504,column=525.78124,scale=0.680912 跳到324.00210,525.66424,0.686428。
   大 nut 的 column 从109.54165变为109.44601，scale从1.268417变为1.270988。
   这与对应点集合变化一致，提示应继续隔离其影响，而不是仅调总迭代预算。
3. 小 ring（GT #2）的种子尺度为0.80557353，最终0.80546364；HALCON四种模式都是0.8。
   当前解并没有触及0.8下界，单纯增加下界裁剪不会把0.80546变成0.8。不能硬编码该实例尺度。

## 统一预算实验

所有候选采用相同迭代预算，之后执行原评分和NMS，没有根据目标或GT分别选停止轮数。

| 预算 | 关联 / 7 | 已关联位置 RMSE px | 最大误差 px |
|---|---:|---:|---:|
| 0 | 6 | 0.305024 | 0.674075 |
| 1 | 5 | 0.265504 | 0.501365 |
| 2 | 6 | 0.235178 | 0.481917 |
| 3 | 6 | 0.232643 | 0.481233 |
| 4 | 7 | 0.274041 | 0.577676 |
| 5 | 7 | 0.290622 | 0.579982 |
| 6 | 7 | 0.297762 | 0.579637 |
| 8 | 7 | 0.300335 | 0.578889 |
| 默认very_high预算30 | 7 | 0.300366 | 0.578889 |

漏检行的RMSE只统计已关联目标，不是完整集精度改善。所有实验均未达到0.10px RMSE / 0.20px最大误差门限。
预算4的改善不足以通过验收，且更改预算可能改变候选取舍，不能等同于将最终获胜轨迹简单截断。

固定3px半径、预算30：6/7，RMSE=0.234907px，最大=0.481233px；大 ring 漏检。
固定1.5px半径、预算30：7/7，结果与默认几乎相同。
因此半径更新策略有影响，但移除收缩策略也不能直接复现 HALCON。

## 下一步

不能仅凭 very_high 更接近就断言根因只有迭代数，也不能以普通GT处几何残差较大断言算法原理不同。
现有结果同时指向初始化、对应点集合及有限步更新的组合差异。
后续应先固定相同入口位姿，分别对照得分峰插值初始化与迭代调度；保留角度、尺度和召回联合验收，不按目标挑参数。
HALCON实际参数文件目前只有版本与subpixel两项，其他尺度/对比度/原点实际参数仍未获得。

## 复现

```bash
cd shape_match_v1
bash scripts/build.sh
SHAPE_MATCH_NUM_THREADS=8 ./build/refinement_trajectory data trajectory-default
SHAPE_MATCH_NUM_THREADS=8 ./build/refinement_trajectory data trajectory-budget4 4
SHAPE_MATCH_NUM_THREADS=8 ./build/refinement_trajectory data trajectory-radius3 30 3
python3 scripts/compare_gt.py trajectory-default.csv --reference data/reference/halcon_least_squares.csv
```

输出前缀产生 .csv 和 .trace.csv，同名文件会覆盖。trace_id仅用于同次运行分组，多线程下编号不保证跨运行一致。
iteration记录每轮更新前的位姿；begin/iteration的score保留候选种子值，只有end重新评分，不能当作逐轮得分曲线。
当前轨迹工具仅配置默认nearest_point；不保证嵌套连续精定位模式的trace_id语义。
诊断源码包含真实search.cpp，静态库提供其余实现；SHAPE_MATCH_TRACE_REFINEMENT仅在诊断翻译单元定义。
