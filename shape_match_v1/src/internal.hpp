#pragma once
#include "shape_match/shape_match.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <set>

namespace shape_match::detail {
constexpr double pi = 3.14159265358979323846;
using Clock = std::chrono::steady_clock;
inline double Elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
[[noreturn]] inline void Fail(ErrorCode code, const std::string &message) {
    throw HException(code, message);
}
inline void Require(bool ok, ErrorCode code, const std::string &message) {
    if (!ok)
        Fail(code, message);
}
inline void Output(const void *p) {
    Require(p != nullptr, ErrorCode::Value, "Null output pointer");
}
inline double Sq(double x) {
    return x * x;
}
inline double WrapAngle(double a) {
    return std::remainder(a, 2 * pi);
}
// Configuration is captured once; diagnostics and scratch allocations use the
// same worker count as the process-wide executor.
size_t SearchWorkerCount(size_t work_items);
void ParallelFor(size_t work_items, const std::function<void(size_t, size_t, size_t)> &function);
using Params = std::map<std::string, HValue>;
double Number(const HValue &value);
std::string String(const HValue &value);
bool IsAuto(const HValue &value);
double Num(const Params &p, const std::string &name);
std::string Str(const Params &p, const std::string &name);
bool Bool(const Params &p, const std::string &name);
/// P0: 生成版本化默认参数；自动参数保留 "auto"，训练后另存实际值。
Params MakeDefaultParams(uint64_t id);
/// P0: 校验类型、取值和参数间约束；不支持项明确抛出异常。
void ValidateParams(const Params &params);
/// P0: 判断参数是否影响模型特征；搜索范围与输出原点不强制重训练。
bool NeedsTraining(const std::string &name);

struct HandleBase {
    enum class Kind { Model, Result };
    explicit HandleBase(Kind k) : kind(k) {}
    virtual ~HandleBase() = default;
    Kind kind;
};
// Internal Cartesian convention: x=column, y=row, positive theta clockwise.
// Conversion to public HALCON convention is confined to result.cpp.
struct Vec {
    double x = 0, y = 0;
    Vec operator+(Vec b) const { return {x + b.x, y + b.y}; }
    Vec operator-(Vec b) const { return {x - b.x, y - b.y}; }
    Vec operator*(double s) const { return {x * s, y * s}; }
};
inline double Dot(Vec a, Vec b) {
    return a.x * b.x + a.y * b.y;
}
inline Vec Rotate(Vec a, double t) {
    double c = std::cos(t), s = std::sin(t);
    return {c * a.x - s * a.y, s * a.x + c * a.y};
}
struct Pose {
    double x = 0, y = 0, theta = 0, scale = 1;
};
struct EdgeFeature {
    Vec p, n;
    double strength = 0, weight = 1;
    int contour = -1;
};
using Features = std::vector<EdgeFeature>;
struct Correspondence {
    Vec model, image, normal;
};
struct GradientField {
    int width = 0, height = 0;
    std::vector<float> gx, gy, mag;
};
struct EdgeField {
    int width = 0, height = 0;
    Features edges;
    std::vector<int> nearest;
};
struct PyramidLevel {
    Image image;
    GradientField gradient;
    EdgeField field;
    std::shared_ptr<const GradientField> precise_gradient;
};
using SearchPyramid = std::vector<PyramidLevel>;
// Experimental continuous observations; never used by coarse candidate generation.
std::vector<Correspondence> ContinuousCorrespondences(const Features &features,
    const PyramidLevel &image, const Pose &pose, const std::string &metric,
    double radius, const std::string &method);
double RefinementSupportLoss(const std::vector<Correspondence> &pairs, const Pose &pose,
                             size_t model_points, double radius);
struct ModelLevel {
    Features features;
    double factor = 1;
};
struct TrainedData {
    Image image;
    Vec origin;
    Features dense;
    Features gaussian_dense;
    ContourSet contours; // absolute template pixel-center coordinates
    std::vector<ModelLevel> levels;
    std::array<Vec, 4> rectangle{}; // min-area rectangle, relative to training origin
    double radius = 1;
    Params effective;
};
struct ShapeModel : HandleBase {
    ShapeModel(uint64_t value)
        : HandleBase(Kind::Model), id(value), params(MakeDefaultParams(value)) {}
    uint64_t id;
    std::mutex mutex;
    bool alive = true, needs_training = true;
    Params params;
    std::shared_ptr<const TrainedData> trained;
};
struct ModelSnapshot {
    uint64_t id;
    Params params;
    std::shared_ptr<const TrainedData> data;
    HHandle handle;
};
struct Candidate {
    Pose pose;
    double score = 0;
};
struct Match {
    ModelSnapshot model;
    Pose pose;
    double score = 0;
};
struct MatchResult : HandleBase {
    MatchResult() : HandleBase(Kind::Result) {}
    std::vector<Match> matches;
    std::vector<ModelSnapshot> models; // supports valid identifier with zero matches
    SearchDiagnostics diagnostics;
};
/// P0: 从异构元组中取出具有类型身份的模型句柄；存活状态在锁内校验。
std::shared_ptr<ShapeModel> ResolveModel(const HTuple &tuple, size_t i = 0);
/// P4: 获取不可变搜索结果；结果独立于随后对模型参数的修改。
std::shared_ptr<MatchResult> ResolveResult(const HTuple &tuple);
/// P0/P2: 在模型锁内捕获参数/训练数据快照，并验证模型标识唯一。
std::vector<ModelSnapshot> SnapshotModels(const HTuple &tuple);

/// P1: 双线性采样；只用于内部灰度观测，越界点采用边界复制。
float Bilinear(const Image &image, double x, double y);
/// P3: 三次插值采样工具；当前最小二乘路径使用提取后的边缘点。
float CubicSample(const Image &image, double x, double y);
/// P1/P2: 可分离高斯滤波，保留域元数据而不把掩码外强制置黑。
Image Smooth(const Image &image, double sigma);
/// P1/P2: 抗混叠后按 (2x,2y) 采样；坐标映射明确可追溯。
Image Downsample(const Image &image);
/// P1/P2: 中心差分梯度及幅值；输入已经过平滑。
GradientField ComputeGradients(const Image &image);
GradientField GaussianGradients(const Image &image, double sigma = 0.8);
Features GaussianModelFeatures(const Image &image, double low, double high, int min_size);
/// P1/P2: 法线非极大值抑制、二次峰插值、连通滞后阈值。
Features ExtractSubpixelContours(const Image &image, const GradientField &gradient, double low,
                                 double high, int min_size);
/// P1: 邻接轮廓追踪，用于显示；不依赖显示轮廓顺序求解位姿。
ContourSet TraceContours(Features &features);
/// P1: 最远点采样保留空间覆盖；O(NK)，只发生在建模/候选准备阶段。
Features SelectModelFeatures(const Features &features, size_t maximum);
/// P2: 亚像素边缘及近似 Voronoi 标签图，供搜索/对应查询使用。
EdgeField BuildEdgeField(const Image &image, const GradientField &gradient, double contrast);
/// P1: 凸包边方向枚举得到最小面积外接矩形，不使用轴对齐替代。
std::array<Vec, 4> MinimumRectangle(const Features &features);
/// P2: 旋转矩形裁剪；交集面积除以较小矩形面积，与 IoU 定义不同。
double RectangleOverlap(const Match &a, const Match &b);
/// P1: 完整训练流程；失败时不覆盖上一次训练数据。
std::shared_ptr<TrainedData> BuildModel(const Image &image, const Params &params);
/// P2: 多模型共享搜索图金字塔；Domain 仅限制候选参考位置。
SearchPyramid BuildSearchPyramid(const Image &image, int count, double contrast, bool precise = false);
/// P2: 高斯距离衰减乘法线一致性，支持三种灰度极性语义和安全上界剪枝。
double EvaluatePose(const Features &features, const EdgeField &field, const Pose &pose,
                    const std::string &metric, double sigma, double min_score = 0);
/// P2: 遍历全图位置/角度/尺度，提取局部峰并进入有限候选束。
std::vector<Candidate> SearchCoarsestLevel(const ModelSnapshot &model, const SearchPyramid &pyramid,
                                           int level, SearchDiagnostics &diag);
/// P2: 按层映射候选位置，局部坐标下降并合并邻近候选。
std::vector<Candidate> TrackToFinerLevel(const ModelSnapshot &model, const SearchPyramid &pyramid,
                                         std::vector<Candidate> candidates, int level,
                                         SearchDiagnostics &diag);
/// P3: 按 subpixel 分派量化/插值/最小二乘路径，重新计算最终得分。
Candidate RefineInstance(const ModelSnapshot &model, const PyramidLevel &image,
                         Candidate candidate, bool baseline_prepared = false);
/// P3: 各自由度的局部二次得分峰插值，并限制非凹峰与步长。
Pose InterpolateScorePeak(const ModelSnapshot &model, const PyramidLevel &image, Pose pose);
/// P2/P3: 检查周期角度区间与等比尺度；容差用于非严格搜索细化。
bool WithinBounds(const ModelSnapshot &model, const Pose &pose, double tolerance = 0);
/// P2/P4: 按分数排序，模型内/跨模型重叠抑制，数量限制与严格边界过滤。
std::vector<Match> FinalizeMatches(std::vector<Match> matches,
                                   const std::vector<ModelSnapshot> &models);
/// P4: 应用用户设置的模型原点偏移，输出内部像素中心坐标。
Vec OutputOrigin(const Match &match);
/// P4: 构造供 XLD 边缘中心坐标变换使用的 2x3 行列矩阵。
std::array<double, 6> ResultTransform(const Match &match);
/// P4: 选择器求并集并按现有分数顺序输出索引，不重复返回同一实例。
std::vector<size_t> ResolveMatchSelector(const MatchResult &result, const HTuple &selector);
} // namespace shape_match::detail
