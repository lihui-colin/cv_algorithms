# 基于图像查询的任意目标高精度定位技术方案

## 1. 项目目标

构建一套面向工业视觉场景的 **Visual Query Universal Locator**：

> 用户仅提供一张或少量目标模板图，不需要针对目标类别重新训练，即可在场景图像中发现所有目标实例，并输出高精度二维位姿。

系统输入：

\[
(Q,I)
\]

其中：

- \(Q\)：Query Image / Template Image
- \(I\)：待检测 Scene Image

系统最终输出：

\[
\mathcal O=
\{
(x_i,y_i,\theta_i,s_i,c_i)
\}_{i=1}^{N}
\]

其中：

- \(x_i,y_i\)：目标中心位置
- \(\theta_i\)：旋转角
- \(s_i\)：尺度
- \(c_i\)：最终置信度
- \(N\)：场景中的目标实例数

对于允许更复杂形变的场景，可进一步输出：

\[
A_i\in\mathbb R^{2\times3}
\]

或：

\[
H_i\in\mathbb R^{3\times3}
\]

分别表示 affine 或 homography。

---

## 2. 核心设计思想

整个问题不由单一深度网络端到端解决，而是拆解成四个不同性质的问题：

\[
\boxed{
\text{Robust Instance Discovery}
}
\]

\[
\boxed{
\text{Robust Dense Correspondence}
}
\]

\[
\boxed{
\text{Geometrically-Constrained Pose Estimation}
}
\]

\[
\boxed{
\text{Subpixel Image-Space Refinement}
}
\]

对应当前建议技术路线：

\[
\boxed{
\text{DINOv-like}
\rightarrow
\text{RoMa-like}
\rightarrow
\text{Geometry}
\rightarrow
\text{Subpixel Optimization}
}
\]

核心原则：

\[
\boxed{
\text{深度学习负责找到，}
\quad
\text{对应关系负责对上，}
\quad
\text{几何负责算准，}
\quad
\text{图像优化负责做到亚像素。}
}
\]

---

## 3. 总体架构

```text
                    Query Image
                         │
                         │
                         ▼
                 Query Preprocess
                         │
                         ▼
              Visual Query Embedding
                         │
                         │
Scene Image ─────────────┼─────────────────────┐
                         │                     │
                         ▼                     │
        ┌───────────────────────────┐          │
        │ 1. Instance Discovery     │          │
        │                           │          │
        │ DINOv-like Visual Query   │          │
        │ Detector                  │          │
        └─────────────┬─────────────┘          │
                      │                        │
               ROI1...ROIN                    │
                      │                        │
                      ▼                        │
        ┌───────────────────────────┐          │
        │ 2. Dense Correspondence   │          │
        │                           │          │
        │ RoMa-like Matcher         │          │
        └─────────────┬─────────────┘          │
                      │                        │
          Dense Correspondences               │
                      │                        │
                      ▼                        │
        ┌───────────────────────────┐          │
        │ 3. Geometry              │          │
        │                           │          │
        │ RANSAC + Pose Model       │          │
        └─────────────┬─────────────┘          │
                      │                        │
                Initial Pose                  │
                      │                        │
                      ▼                        │
        ┌───────────────────────────┐          │
        │ 4. Subpixel Refinement    │◄─────────┘
        │                           │
        │ Gradient / Edge / ECC /   │
        │ Distance Transform        │
        └─────────────┬─────────────┘
                      │
                      ▼
              Final Refined Pose
                      │
                      ▼
              Verification Layer
                      │
                      ▼
          x, y, θ, scale, confidence
```

---

## 4. 模块一：Robust Instance Discovery

### 4.1 目标

解决：

\[
\boxed{
\text{场景中有没有 Query？在哪里？有几个？}
}
\]

这一模块不负责高精度定位。

设计目标是：

\[
\boxed{\text{High Recall}}
\]

即：

> 可以容忍定位框偏差，但不能轻易漏掉真实目标。

### 4.2 输入

```python
QueryImage
SceneImage
OptionalQueryMask
OptionalQueryBox
```

推荐 Query 不直接使用整张图片，而允许用户指定：

- box
- mask
- polygon

用于告诉系统：

> Query 图中真正需要搜索的是哪个区域。

### 4.3 输出

```python
Candidate {
    bbox
    coarse_mask
    detector_score
    query_similarity
}
```

一个 Query 可以产生多个候选实例。

### 4.4 推荐实现

第一版：

\[
\boxed{\text{DINOv-like Visual Prompt Detector}}
\]

架构：

```text
Query Image
    │
    ▼
Visual Prompt Encoder
    │
    ▼
Query Embedding
    │
    │
    ▼
Target Feature Pyramid
    │
    ▼
Prompt-conditioned Decoder
    │
    ▼
Object Queries
    │
    ▼
Candidate Boxes
```

### 4.5 工业场景关键改造

通用 Visual Prompt Detector 往往偏：

\[
\text{semantic similarity}
\]

而工业定位更需要：

\[
\text{instance identity}
\]

因此训练目标不能只有：

\[
\text{same category}=positive
\]

而应该引入：

\[
\boxed{
\text{same physical design}=positive
}
\]

以及大量：

\[
\boxed{
\text{similar part}=hard\ negative
}
\]

---

## 5. 模块二：Robust Dense Correspondence

### 5.1 目标

对于每个候选区域建立：

\[
Q\leftrightarrow ROI_i
\]

的二维对应关系。

输出：

\[
\mathcal M_i=
\{
(p_j^Q,p_j^I,c_j)
\}_{j=1}^{M}
\]

其中：

- \(p_j^Q\)：Query 中的点
- \(p_j^I\)：Scene 中对应点
- \(c_j\)：匹配 confidence

### 5.2 推荐实现

第一版：

\[
\boxed{\text{RoMa-like Dense Matcher}}
\]

也可预留替换接口：

- RoMa
- LoFTR
- LightGlue
- 自研 coarse-to-fine matcher

### 5.3 这一模块同时承担 Verification

如果模块一检测到了错误候选，Matcher 通常表现为：

- 有效匹配点数量低
- confidence 低
- spatial coverage 差
- geometry consistency 差

因此这一层不仅是 matcher，也是：

\[
\boxed{\text{fine-grained candidate verifier}}
\]

### 5.4 不能只看 Match Count

需要至少计算：

#### Match 数量

\[
N_{match}
\]

#### 平均 confidence

\[
\bar c
\]

#### 空间覆盖率

\[
R_{coverage}
\]

#### 局部分布均匀程度

例如将 Query 分成：

\[
4\times4
\]

网格，统计有多少网格具有有效 correspondence。

---

## 6. 模块三：Geometrically-Constrained Pose Estimation

### 6.1 目标

将 noisy correspondences：

\[
p_i^I=T(p_i^Q)+\epsilon_i
\]

转化成唯一、物理合理的目标位姿：

\[
T^*
\]

### 6.2 几何模型

根据具体工业场景选择自由度。

#### Translation

\[
p'=p+t
\]

适用于：

- 相机固定
- 目标不旋转

#### SE(2)

\[
p'=Rp+t
\]

输出：

\[
x,y,\theta
\]

适用于很多工业定位场景。

#### Similarity Transform

\[
p'=sRp+t
\]

输出：

\[
x,y,\theta,s
\]

适用于存在尺度变化的情况。

#### Affine

\[
p'=Ap+t
\]

适用于：

- 轻微视角变化
- 非严格正视成像

#### Homography

\[
p'\sim Hp
\]

适用于明显透视变化。

### 6.3 重要原则

\[
\boxed{
\text{Minimum Sufficient Geometry}
}
\]

即：

> 能用 SE(2)，绝不用 Homography。

因为自由度越高：

\[
\text{DoF}\uparrow
\]

模型解释错误 correspondence 的能力也越强。

### 6.4 鲁棒估计

推荐：

\[
\boxed{\text{RANSAC}}
\]

流程：

```text
Dense Matches
    │
    ▼
Confidence filtering
    │
    ▼
RANSAC
    │
    ├── Inliers
    └── Outliers
    │
    ▼
Least-Squares Refit
    │
    ▼
Initial Pose
```

输出：

```python
PoseEstimate {
    transform
    center
    angle
    scale

    num_matches
    num_inliers
    inlier_ratio
    reprojection_error
}
```

---

## 7. 模块四：Subpixel Image-Space Refinement

### 7.1 为什么必须独立存在

深度网络的 feature map 往往只有原始图像：

\[
1/8,\ 1/16
\]

分辨率。

Correspondence 网络更擅长：

\[
\boxed{\text{找对}}
\]

但不能天然保证：

\[
\boxed{0.05\sim0.1\ pixel}
\]

工业级定位精度。

因此最后必须重新回到：

\[
\boxed{\text{原始图像信号}}
\]

---

## 8. 亚像素 Refinement 推荐三条路线

### Route A：Gradient Alignment

优化：

\[
\theta^*
=
\arg\min_\theta
\sum_x
\rho
\left[
I(T(x;\theta))-Q(x)
\right]
\]

可采用：

- Gauss-Newton
- Levenberg-Marquardt
- inverse compositional optimization

适合：

- 外观稳定
- 灰度关系比较稳定

### Route B：Edge / Contour Alignment

从模板获得 contour：

\[
C=\{p_i,n_i\}
\]

目标图建立：

\[
D_I(x,y)
\]

即 edge distance field。

优化：

\[
E(T)=
\sum_{p_i\in C}
\rho
\left[
D_I(T(p_i))
\right]
\]

进一步可加入 orientation：

\[
E=
E_{distance}
+
\lambda E_{orientation}
\]

这一条与传统 Shape Matching 最接近。

### Route C：ECC

优化图像相关性：

\[
\max_T
ECC(Q,I\circ T)
\]

适合：

- patch 外观稳定
- 纹理较明显

可作为补充 refinement。

---

## 9. Verification：贯穿全系统

工业系统不能仅仅输出一个 pose。

必须能够回答：

\[
\boxed{\text{这次定位可靠吗？}}
\]

因此建议增加逻辑上的 Verification Layer。

它不是第五个主算法模块，而是横跨四层。

最终 confidence：

\[
C=
f(
C_{det},
C_{match},
R_{coverage},
R_{inlier},
E_{reproj},
E_{subpixel}
)
\]

示例：

```text
Detector score        0.94
Matcher confidence    0.88
Coverage              0.81
RANSAC inlier ratio   0.91
Reprojection error    0.21 px
Subpixel residual     0.06 px
```

最终：

```text
MATCH_VALID
Confidence = 0.97
```

如果：

```text
Detector score = 0.98
Inlier ratio   = 0.22
```

则必须：

```text
MATCH_REJECTED
```

而不是盲目相信 Detector。

---

## 10. 推荐的数据接口

### Stage 1

```python
class Candidate:
    bbox
    mask
    detector_score
    query_similarity
```

### Stage 2

```python
class MatchSet:
    query_points
    target_points
    confidence

    match_count
    spatial_coverage
```

### Stage 3

```python
class PoseEstimate:
    x
    y
    theta
    scale

    transform
    inlier_ratio
    reprojection_error
```

### Stage 4

```python
class RefinedPose:
    x
    y
    theta
    scale

    residual
    confidence
    covariance
```

---

## 11. 最终建议输出 Pose Covariance

系统不要只输出：

\[
x=512.42
\]

最好还能输出：

\[
x=512.42\pm0.03
\]

即估计：

\[
\Sigma_{pose}
\]

这使系统从：

\[
\boxed{\text{定位算法}}
\]

进一步变成：

\[
\boxed{\text{可度量的视觉测量系统}}
\]

---

## 12. Template Library

工业落地时建议增加独立的 Template Manager。

对于一个零件，不只存一张模板：

```text
Part A
├── frontal
├── angle +10°
├── angle -10°
├── bright
├── dark
├── batch 01
└── batch 02
```

这些 exemplar：

\[
Q_1,\dots,Q_K
\]

可以形成：

\[
E_{part}
=
Aggregate
(E(Q_1),...,E(Q_K))
\]

得到稳定的：

\[
\boxed{\text{Visual Query Prototype}}
\]

运行过程中：

```text
模板管理阶段
      │
      ▼
Query Embedding Cache
      │
      ▼
在线检测
```

避免每次重复编码 Query。

---

## 13. 训练方案

如果第一阶段需要自研或 fine-tune，可以大量使用合成数据。

已有一个真实模板：

\[
Q
\]

对其做：

- rotation
- scale
- illumination
- blur
- noise
- occlusion
- perspective
- background compositing

生成：

\[
Q'
\]

然后随机粘贴到 scene：

\[
I=B\oplus Q'
\]

变换矩阵：

\[
T_{GT}
\]

天然已知。

因此同时可以生成：

- bounding box GT
- mask GT
- correspondence GT
- pose GT

几乎不需要额外人工标注。

---

## 14. Hard Negative 是整个系统训练的重点

工业对象最大问题往往不是：

> 找不到差异巨大的对象。

而是：

> 区分非常像的两个零件。

因此训练集应该专门构造：

```text
Query:
Part A

Positive:
Part A rotated
Part A darker
Part A partially occluded

Hard Negatives:
Part B — only hole diameter differs
Part C — one screw hole moved
Part D — one tooth missing
Part E — different connector pin count
```

优化目标：

\[
L=
L_{detection}
+
\lambda_1L_{instance}
+
\lambda_2L_{hard-negative}
+
\lambda_3L_{geometry}
\]

---

## 15. 运行时 Pipeline

```text
1. Load Query
        │
2. Encode Query
        │
3. Detect Candidates
        │
4. NMS / candidate filtering
        │
5. Crop ROI
        │
6. Dense Correspondence
        │
7. Match filtering
        │
8. Geometry RANSAC
        │
9. Geometry validation
        │
10. Subpixel refinement
        │
11. Final verification
        │
12. Return poses
```

---

## 16. 建议加入 Early Exit

不是所有候选都需要跑完整 Pipeline。

例如：

```text
Detector score < threshold
        ↓
Reject
```

或者：

```text
Matcher inlier potential too low
        ↓
Reject
```

或者：

```text
Geometry residual > threshold
        ↓
Reject
```

这可以显著减少 RoMa 和亚像素 refinement 的计算量。

---

## 17. 性能优化策略

计算量大致满足：

\[
C_{total}
=
C_{detector}
+
N_{candidate}C_{matcher}
+
N_{valid}C_{geometry}
+
N_{valid}C_{refine}
\]

最大的变量通常是：

\[
N_{candidate}
\]

因此需要控制：

- top-K candidates
- detector threshold
- ROI resize
- matcher coarse resolution

---

## 18. 推荐两级 Matcher

未来如果 RoMa 太重，可以设计：

```text
Candidate ROI
      │
      ▼
Lightweight Matcher
      │
    pass?
    /   \
  no     yes
  │       │
reject   RoMa
          │
          ▼
       Geometry
```

形成：

\[
\boxed{
\text{cheap verification}
\rightarrow
\text{expensive precise matching}
}
\]

---

## 19. 评价体系

不能只使用 mAP。

整个系统应该拆层评估。

### Module 1

#### Instance Recall

\[
Recall@IoU_{loose}
\]

例如：

\[
IoU>0.3
\]

即可视为成功进入下一阶段。

核心指标：

\[
\boxed{Recall}
\]

### Module 2

#### Match Precision

#### Inlier Ratio

\[
R_{inlier}
=
\frac{N_{inlier}}{N_{match}}
\]

#### Spatial Coverage

### Module 3

#### Translation Error

\[
E_t=
\sqrt{
(\hat x-x)^2+
(\hat y-y)^2
}
\]

#### Rotation Error

\[
E_\theta=
|\hat\theta-\theta|
\]

#### Scale Error

\[
E_s=
|\hat s-s|
\]

### Module 4

#### Subpixel Position Error

重点统计：

- mean
- median
- P95
- P99
- max

不能只报告平均误差。

---

## 20. 工业场景最终 KPI

第一阶段 MVP 可设：

### Detection

\[
Recall>99.5\%
\]

### Geometry

\[
P95\ Translation<1px
\]

### Refinement

最终目标：

\[
P95<0.1\sim0.2px
\]

以及：

\[
P95\ angle\ error<0.1^\circ
\]

实际指标需要结合：

- 相机分辨率
- FOV
- 零件尺寸
- 景深
- SNR
- 光学系统

重新定义。

---

## 21. MVP 建议

### Phase 1：证明四模块闭环

不训练任何新模型。

直接：

\[
\boxed{
\text{DINOv}
\rightarrow
\text{RoMa}
\rightarrow
\text{OpenCV Geometry}
\rightarrow
\text{ECC / Edge Refinement}
}
\]

目标：

> 证明任意模板能够完成从全图搜索到亚像素定位的完整链路。

### Phase 2：替换 Instance Discovery

针对工业对象：

- small object
- dense objects
- similar objects
- low texture

做专门 Visual Query Detector。

重点：

\[
\boxed{\text{instance-aware visual embedding}}
\]

### Phase 3：优化 Matcher

研究：

- RoMa 裁剪
- Tiny RoMa
- coarse-to-fine matching
- local matcher
- TensorRT/NPU 部署

将 matcher latency 降下来。

### Phase 4：工业级 Subpixel

重点投入：

\[
\boxed{
\text{Contour + Gradient + Robust Optimization}
}
\]

形成真正的亚像素测量模块。

---

## 22. 最终产品形态

用户使用：

```python
model = VisualLocator()

template = model.create_template(
    image=query,
    roi=query_roi,
)

results = model.locate(
    image=scene,
    template=template
)
```

输出：

```python
[
    {
        "x": 812.37,
        "y": 431.82,
        "angle": 31.42,
        "scale": 0.998,

        "confidence": 0.976,
        "position_sigma": 0.04,

        "detector_score": 0.93,
        "inlier_ratio": 0.87,
        "reprojection_error": 0.19,
        "subpixel_residual": 0.06,
    }
]
```

用户体验最终应该非常接近：

```text
create_shape_model()
find_shape_model()
```

但模型建立过程不再依赖：

\[
\text{handcrafted edge template}
\]

而是：

\[
\boxed{
\text{Visual Prompt + Learned Correspondence + Geometry}
}
\]

---

## 23. 与传统 Shape Matching 的关系

传统：

\[
\text{Edge}
\rightarrow
\text{Orientation}
\rightarrow
\text{Pyramid Search}
\rightarrow
\text{Pose}
\rightarrow
\text{Subpixel}
\]

本方案：

\[
\text{Visual Query}
\rightarrow
\text{Learned Instance Discovery}
\rightarrow
\text{Dense Correspondence}
\rightarrow
\text{Geometry}
\rightarrow
\text{Subpixel}
\]

两者实际上非常相似，只是前三步由：

\[
\text{handcrafted feature}
\]

升级成：

\[
\boxed{\text{learned visual representation}}
\]

而最后的几何约束与亚像素优化仍然保留经典机器视觉的优势。

---

## 24. 技术方案最终定义

整个项目可以正式定义成：

\[
\boxed{
\textbf{Image-Query Universal High-Precision Locator}
}
\]

核心四层架构：

\[
\boxed{
\begin{aligned}
1.&\ \textbf{Robust Instance Discovery}\\
2.&\ \textbf{Robust Dense Correspondence}\\
3.&\ \textbf{Geometrically-Constrained Pose Estimation}\\
4.&\ \textbf{Subpixel Image-Space Refinement}
\end{aligned}
}
\]

系统思想不是试图训练一个“万能神经网络”，而是让每种方法解决自己最擅长的问题：

\[
\boxed{
\text{DL解决鲁棒性}
}
\]

\[
\boxed{
\text{Matching解决对应关系}
}
\]

\[
\boxed{
\text{Geometry解决正确性}
}
\]

\[
\boxed{
\text{Optimization解决精度}
}
\]

最终目标是把传统工业 Shape Matching 从：

\[
\boxed{
\text{基于人工边缘特征的模板定位}
}
\]

升级为：

\[
\boxed{
\text{基于 Visual Query 的通用高精度目标定位}
}
\]

并同时兼顾：

**任意目标、无需类别训练、多实例、旋转尺度鲁棒、遮挡鲁棒、相似零件区分以及亚像素工业精度。**
