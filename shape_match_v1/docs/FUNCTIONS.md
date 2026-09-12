# 函数、阶段与实现思路

所有接口完整声明位于 `include/shape_match/shape_match.hpp` 与 `src/internal.hpp`。私有实现函数直接定义在相应 cpp 中。下列名称对应实际源码，没有保留空壳阶段函数。

前期计划中的若干细分函数在实现时进行了合并，例如 BuildNormalEquations/求解仍独立，ApplyPoseIncrement 和 EvaluateRefinement 则位于 RefineInstance 内；不为满足函数数量而引入无用抽象。

## P0：类型、参数与状态

| 函数/类型 | 源码 | 功能与实现 |
|---|---|---|
| HTuple 构造、Append、AppendValue | types.cpp | 异构序列，保留整数/浮点/字符串/句柄身份 |
| HTuple::At / D / I / S / H | types.cpp | 带边界和类型检查的取值；数值 D 可接收整数 |
| HObject::FromGray / FromBytes | types.cpp | 深拷贝输入与掩码，拒绝非有限数和尺寸不匹配 |
| HObject::FromContours / GetContours | types.cpp | 封装浮点轮廓集合 |
| HObject::GetImage / Count | types.cpp | 只读图像获取和对象数量 |
| CreateGenericShapeModel | model.cpp | 原子唯一 ID、模型句柄、默认参数 |
| MakeDefaultParams | model.cpp | 集中默认参数定义 |
| ValidateParams | model.cpp | 参数类型、范围、组合合法性及资源限制 |
| NeedsTraining | model.cpp | 标识影响训练特征的参数 |
| SetGenericShapeModelParam | model.cpp | 在临时参数表上修改与验证，成功后一次提交 |
| GetGenericShapeModelParam | model.cpp | 设置值与有效值分离；查询训练状态 |
| ResolveModel / ResolveResult | types.cpp | 类型安全句柄转换 |
| SnapshotModels | types.cpp | 锁内获取稳定参数和训练数据快照 |
| ClearShapeModel | model.cpp | 显式失效所有模型句柄别名 |

模型使用 shared_ptr 管理数据，并使用模型互斥锁避免设置/训练/清理的数据竞争。搜索获得只读快照后不再持有模型锁，因此参数变更不会半途影响一次搜索。相同模型并发调用行为采用本工程快照契约；不声称等于 HALCON 内部锁策略。

## P1：模板建模

| 函数 | 输入 → 输出 | 实现思路 |
|---|---|---|
| TrainGenericShapeModel | HObject, ModelID → 更新模型 | 训练入口，失败不覆盖旧数据 |
| EstimateNoise | Image → 噪声量级 | 二阶差分绝对值中位数，减少边缘对噪声估计的影响 |
| BuildModel | Image, Params → TrainedData | 重心、自动参数、细轮廓、半径、矩形和各层特征 |
| Smooth | Image, sigma → Image | 可分离高斯，边界复制 |
| Downsample | Image → Image | 平滑后按偶数位置采样，映射为前一级 2x/2y |
| ComputeGradients | Image → GradientField | 中心差分和幅值 |
| ExtractSubpixelContours | Image, GradientField, 阈值 → Features | 法线非极大抑制、二次插值、连通滞后筛选 |
| TraceContours | Features → ContourSet | 邻接追踪并在分支拆分，用于轮廓显示 |
| SelectModelFeatures | Features, 最大点数 → Features | 最远点采样，保留轮廓空间覆盖 |
| MinimumRectangle | Features → 四角点 | 凸包边方向枚举最小面积矩形 |
| GetGenericShapeModelObject | ModelID, contours → HObject | 输出相对模型参考点的像素中心坐标 |

边缘点在沿梯度方向的局部幅值峰上拟合二次曲线。设两侧和中心幅值为 a、b、c，则偏移：

```text
delta = 0.5*(a-c)/(a-2*b+c)
```

仅在有效峰曲率下使用，并限制在半像素范围内。该方法可能出现亚像素相位偏差，测试集用于实测它而不是假设其无偏。

模板掩码外的数据不被置黑；特征提取检查邻域有效性。紧贴掩码边界的真实边缘可能被排除，应用应为模板保留背景边距。此处仍需针对不同成像边界进一步验证。

## P2：粗搜索与跟踪

| 函数 | 输入 → 输出 | 实现思路 |
|---|---|---|
| FindGenericShapeModel | SearchImage, ModelIDs → ResultID, Count | 完整调度，零检出也返回有效结果 |
| BuildSearchPyramid | Image, 层数, 对比度 → SearchPyramid | 共享金字塔和边缘辅助图 |
| BuildEdgeField | Image, GradientField → EdgeField | 亚像素边缘加近似最近点标签图 |
| EvaluatePose | Features, EdgeField, Pose → Score | 距离衰减和法线相似度，安全得分上界剪枝 |
| SearchCoarsestLevel | ModelSnapshot, 金字塔 → Candidates | 全图步进扫描角度/尺度网格，局部峰提取 |
| MergeCandidates | Candidates, 合并距离, 上限 → Candidates | 位置、角度、尺度邻近性去重及候选束限制 |
| ImproveCoordinate | 模型/图像/初始位姿 → Pose | 对位置、角度和尺度作逐坐标邻域优化 |
| TrackToFinerLevel | 上层 Candidates → 下层 Candidates | 位移乘二，局部优化并保留候选 |
| WithinBounds | ModelSnapshot, Pose → bool | 周期角度与尺度范围判断 |
| RectangleOverlap | 两实例 → overlap | 凸多边形裁剪，按较小面积归一化 |
| FinalizeMatches | 全部 Match → 最终 Match | 分数排序、重叠、数量和严格范围处理 |

基础得分为：

```text
score = sum(exp(-distance_i^2/(2*sigma^2)) * orientation_i) / N
```

use_polarity 采用正向法线一致性；ignore_global_polarity 分别累计整体正、负极性取较大者；ignore_local_polarity 对每一点取绝对值。没有对应点时贡献 0，不能通过只除以可见点数把少量匹配抬高到 1。

安全剪枝：处理 k 个特征后，即使所有剩余点得满分仍达不到阈值，则立即退出。候选束限制和粗层阈值是另一类启发式，存在漏检风险；不能将其误称为严格无损剪枝。

## P3：亚像素优化

| 函数 | 功能 | 实现思路 |
|---|---|---|
| Bilinear / CubicSample | 灰度插值工具 | 独立工具；当前优化直接使用提取后的边缘坐标 |
| InterpolateScorePeak | 得分峰插值模式 | 每个自由度局部二次拟合，非凹峰不更新 |
| BuildCorrespondences | 建立对应边缘 | 预测点附近的 Voronoi 标签、距离和方向一致性筛选 |
| BuildNormalEquations | 构造 H、g 和 loss | 对点到法线残差累计双精度雅可比 |
| SolvePoseIncrement | 求解位姿增量 | 带阻尼的小型线性系统，主元检查 |
| RefineInstance | 完整精定位 | 对应更新、LM 步长限制、固定对应下的下降检查和提前收敛 |

内部位姿为 `(tx, ty, theta, log(scale))`。设 `a=sR*p`，图像边缘法线为 `(nx,ny)`，残差为 `n·(a+t-q)`。雅可比：

```text
J = [nx, ny, -nx*a_y + ny*a_x, nx*a_x + ny*a_y]
```

尺度固定时使用前三个自由度。解采用 double，不通过 float 累积正规方程。对应点不足、系统退化或无法继续降低残差时保留当前候选，再以最终得分决定是否返回。尚未对退化程度输出独立置信度字段，避免将非官方字段混入 HALCON 查询命名空间。

本实现没有添加未验证的灰度直接配准作为隐藏的高精度分支。高档模式仅增加最小二乘迭代预算。后续如加入直接灰度配准，应作为独立实验路径进行对照。

## P4：结果查询

| 函数 | 功能 | 实现思路 |
|---|---|---|
| OutputOrigin | 用户原点位置 | 将原点偏移按实例旋转和尺度变换 |
| ResultTransform | 2×3 变换矩阵 | 集中处理行列、角度和半像素转换 |
| ResolveMatchSelector | 选择器解析 | 选择集合求并、去重、保留分数顺序 |
| GetGenericShapeModelResult | 字段查询 | 模型标识、位姿、分数和变换矩阵 |
| GetGenericShapeModelResultObject | 轮廓输出 | 变换真实模型轮廓，返回浮点对象 |
| GetSearchDiagnostics | 阶段耗时 | 本工程独立诊断接口，不冒充 HALCON 字典 |

每个候选和匹配实例都保存 ModelSnapshot，身份不从最终排序索引推断。输出参数保持本次搜索的快照，后续模型被修改或释放不改变旧结果。

## P5：评测、I/O 与性能

| 函数/程序 | 功能 | 实现位置 |
|---|---|---|
| ReadPgm / WritePgm / Token | 带校验的 PGM I/O | types.cpp |
| Render / SignedDistance | 连续几何独立真值生成 | tests/synthetic.hpp |
| Check / Throws / test_match main | 契约、类型、生命周期与几何回归 | tests/test_match.cpp |
| accuracy_bench main | 30 个相位/姿态组合的误差导出 | tests/accuracy_bench.cpp |
| performance_bench main | 单进程预热后重复完整搜索 | tests/performance_bench.cpp |
| convert | PNG 与 alpha 无缩放转 PGM | scripts/prepare_samples.py |
| summarize / report | 对照、误差统计与实际轮廓 SVG | scripts/report.py |
| build.sh | 无 CMake 的独立构建入口 | scripts/build.sh |
| run_validation.sh | 编译、回归、样例、精度和报告 | scripts/run_validation.sh |

实际已经使用：共享金字塔、安全评分上界、稀疏搜索特征、候选束、精定位的稠密点集、参数快照与连续像素数组。当前没有宣称实现手写 SIMD、跨实例线程池或预变换特征缓存；这些明确列入优化文档。
