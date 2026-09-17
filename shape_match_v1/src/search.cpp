#include "internal.hpp"
#include <limits>

namespace shape_match::detail {
static double SpatialWeight(double exponent) {
    constexpr int table_size = 4096;
    constexpr double maximum = 9.0;
    static const std::array<float, table_size + 1> table = [] {
        std::array<float, table_size + 1> values{};
        for (int i = 0; i <= table_size; ++i)
            values[size_t(i)] = float(std::exp(-maximum * i / table_size));
        return values;
    }();
    const double coordinate = std::clamp(exponent, 0.0, maximum) * (table_size / maximum);
    const int index = std::min(table_size - 1, int(coordinate));
    const double fraction = coordinate - index;
    return table[size_t(index)] + fraction * (table[size_t(index + 1)] - table[size_t(index)]);
}

struct ScoringPoint {
    double x, y, nx, ny;
    int pixel_x, pixel_y;
};
struct CoarseModelCache {
    int level = 0;
    std::array<double, 6> key{};
    std::vector<std::pair<double, double>> transforms;
    std::vector<std::vector<ScoringPoint>> prepared;
};

// Both search paths share distance, polarity, normalization and pruning.
// The point accessor is inlined; it either transforms a model point or reads
// the transform cached for the coarse angle/scale grid.
template <class PointAt>
static double ScorePoints(size_t count, const EdgeField &im, const std::string &metric,
                          double sigma, double min_score, PointAt point_at) {
    double positive = 0, negative = 0;
    const bool global = metric == "ignore_global_polarity";
    const bool local = metric == "ignore_local_polarity";
    const double inv = 0.5 / (sigma * sigma);
    for (size_t index = 0; index < count; ++index) {
        const auto point = point_at(index);
        if (point.pixel_x >= 0 && point.pixel_y >= 0 && point.pixel_x < im.width &&
            point.pixel_y < im.height) {
            const int id = im.nearest[size_t(point.pixel_y) * im.width + point.pixel_x];
            if (id >= 0) {
                const auto &edge = im.edges[size_t(id)];
                const double dist = Sq(point.x - edge.p.x) + Sq(point.y - edge.p.y);
                if (dist < 18 * sigma * sigma) {
                    const double dot = point.nx * edge.n.x + point.ny * edge.n.y;
                    const double spatial = SpatialWeight(dist * inv);
                    if (local)
                        positive += std::abs(dot) * spatial;
                    else {
                        positive += std::max(0.0, dot) * spatial;
                        if (global)
                            negative += std::max(0.0, -dot) * spatial;
                    }
                }
            }
        }
        // Missing points contribute zero; the remaining points each have an
        // upper bound of one, including for global polarity reversal.
        if (min_score > 0 &&
            (std::max(positive, negative) + count - (index + 1)) < min_score * count)
            return 0;
    }
    return count ? std::clamp(std::max(positive, negative) / count, 0.0, 1.0) : 0;
}

double EvaluatePose(const Features &features, const EdgeField &im, const Pose &pose,
                    const std::string &metric, double sigma, double min_score) {
    const double c = std::cos(pose.theta), s = std::sin(pose.theta);
    return ScorePoints(features.size(), im, metric, sigma, min_score, [&](size_t index) {
        const auto &feature = features[index];
        const double x = pose.x + pose.scale * (c * feature.p.x - s * feature.p.y);
        const double y = pose.y + pose.scale * (s * feature.p.x + c * feature.p.y);
        return ScoringPoint{x,
                            y,
                            c * feature.n.x - s * feature.n.y,
                            s * feature.n.x + c * feature.n.y,
                            ScorePixel(x, im.width),
                            ScorePixel(y, im.height)};
    });
}

// Worker-local cache for an immutable feature set. Only translations reuse it; neither scene
// observations nor pixel indices are cached across poses.
class TranslationScoreCache {
  public:
    explicit TranslationScoreCache(const Features &features) : features_(features) {}
    double Score(const EdgeField &field, const Pose &pose, const std::string &metric, double sigma) {
        if (!valid_ || theta_ != pose.theta || scale_ != pose.scale) {
            const double c = std::cos(pose.theta), s = std::sin(pose.theta);
            points_.resize(features_.size());
            for (size_t i = 0; i < features_.size(); ++i) {
                const auto &f = features_[i];
                points_[i] = {pose.scale * (c * f.p.x - s * f.p.y),
                              pose.scale * (s * f.p.x + c * f.p.y),
                              c * f.n.x - s * f.n.y, s * f.n.x + c * f.n.y, 0, 0};
            }
            theta_ = pose.theta;
            scale_ = pose.scale;
            valid_ = true;
        }
        return ScorePoints(points_.size(), field, metric, sigma, 0, [&](size_t i) {
            const auto &p = points_[i];
            const double x = pose.x + p.x, y = pose.y + p.y;
            return ScoringPoint{x, y, p.nx, p.ny, ScorePixel(x, field.width), ScorePixel(y, field.height)};
        });
    }
  private:
    const Features &features_;
    std::vector<ScoringPoint> points_;
    double theta_ = 0, scale_ = 1;
    bool valid_ = false;
};

static std::vector<ScoringPoint> PrepareCoarsePoints(const Features &features, double angle,
                                                     double scale) {
    const double c = std::cos(angle), s = std::sin(angle);
    std::vector<ScoringPoint> out;
    out.reserve(features.size());
    for (const auto &feature : features) {
        const double x = scale * (c * feature.p.x - s * feature.p.y);
        const double y = scale * (s * feature.p.x + c * feature.p.y);
        out.push_back({x, y, c * feature.n.x - s * feature.n.y, s * feature.n.x + c * feature.n.y,
                       int(std::lround(x)), int(std::lround(y))});
    }
    return out;
}

static double EvaluatePreparedCoarse(const std::vector<ScoringPoint> &points, const EdgeField &im,
                                     int origin_x, int origin_y, const std::string &metric,
                                     double sigma, double min_score) {
    return ScorePoints(points.size(), im, metric, sigma, min_score, [&](size_t index) {
        const auto &p = points[index];
        return ScoringPoint{origin_x + p.x, origin_y + p.y,       p.nx,
                            p.ny,           origin_x + p.pixel_x, origin_y + p.pixel_y};
    });
}
static double ScaleMin(const ModelSnapshot &m) {
    return IsAuto(m.params.at("restrict_iso_scale_min")) ? Num(m.params, "iso_scale_min")
                                                         : Num(m.params, "restrict_iso_scale_min");
}
static double ScaleMax(const ModelSnapshot &m) {
    return IsAuto(m.params.at("restrict_iso_scale_max")) ? Num(m.params, "iso_scale_max")
                                                         : Num(m.params, "restrict_iso_scale_max");
}
static double InternalAngle(const Pose &p, const ModelSnapshot &m) {
    double a = -p.theta, start = Num(m.params, "angle_start");
    return start + std::fmod(std::fmod(a - start, 2 * pi) + 2 * pi, 2 * pi);
}
bool WithinBounds(const ModelSnapshot &m, const Pose &pose, double tolerance) {
    double a = InternalAngle(pose, m), start = Num(m.params, "angle_start"),
           end = Num(m.params, "angle_end");
    bool angle = a <= end + tolerance || a >= start + 2 * pi - tolerance;
    return angle && pose.scale >= ScaleMin(m) - tolerance && pose.scale <= ScaleMax(m) + tolerance;
}
static std::vector<Candidate> MergeCandidates(std::vector<Candidate> candidates, double distance,
                                              size_t cap) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto &a, const auto &b) { return a.score > b.score; });
    std::vector<Candidate> selected;
    // Keep several orientation hypotheses at each center. This is deliberately a
    // bounded beam (documented), not an exhaustive guarantee for arbitrarily many targets.
    for (const auto &c : candidates) {
        bool duplicate = false;
        for (const auto &k : selected) {
            if (Sq(c.pose.x - k.pose.x) + Sq(c.pose.y - k.pose.y) < distance * distance &&
                std::abs(WrapAngle(c.pose.theta - k.pose.theta)) < 0.20 &&
                std::abs(std::log(c.pose.scale / k.pose.scale)) < 0.12) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            selected.push_back(c);
            if (selected.size() >= cap)
                break;
        }
    }
    return selected;
}

static std::vector<Candidate> MergeCandidatesByScale(std::vector<Candidate> candidates,
                                                     double minimum_scale, double maximum_scale,
                                                     double distance, size_t cap) {
    if (maximum_scale <= minimum_scale || cap < 4)
        return MergeCandidates(std::move(candidates), distance, cap);
    constexpr int bins = 4;
    std::vector<Candidate> selected;
    selected.reserve(cap);
    const size_t per_bin = cap / bins;
    for (int bin = 0; bin < bins; ++bin) {
        const double low = minimum_scale + (maximum_scale - minimum_scale) * bin / bins;
        const double high = minimum_scale + (maximum_scale - minimum_scale) * (bin + 1) / bins;
        std::vector<Candidate> subset;
        for (const auto &candidate : candidates)
            if (candidate.pose.scale >= low &&
                (bin + 1 == bins ? candidate.pose.scale <= high : candidate.pose.scale < high))
                subset.push_back(candidate);
        auto merged = MergeCandidates(std::move(subset), distance, per_bin);
        selected.insert(selected.end(), merged.begin(), merged.end());
    }
    return selected;
}
static std::shared_ptr<const CoarseModelCache> GetCoarseModelCache(const ModelSnapshot &m,
                                                                 int level) {
    const auto &ml = m.data->levels[level];
    double start = Num(m.params, "angle_start"), end = Num(m.params, "angle_end");
    const double low = ScaleMin(m), high = ScaleMax(m);
    const std::array<double, 6> key{start, end, low, high, Num(m.params, "angle_step"),
                                  Num(m.params, "iso_scale_step")};
    std::lock_guard<std::mutex> lock(m.data->coarse_cache.mutex);
    if (m.data->coarse_cache.value && m.data->coarse_cache.value->level == level &&
        m.data->coarse_cache.value->key == key)
        return m.data->coarse_cache.value;
    double da =
        std::min(0.20, std::max(Num(m.params, "angle_step"), ml.factor * 0.8 / m.data->radius));
    double ds = std::max(Num(m.params, "iso_scale_step"), ml.factor * 0.75 / m.data->radius);
    int na = std::max(1, int(std::ceil((end - start) / da))),
        ns = std::max(1, int(std::ceil((high - low) / ds)));
    Require(int64_t(na + 1) * (ns + 1) < 1000000, ErrorCode::Value,
            "Angle/scale grid exceeds implementation resource envelope");
    auto cache = std::make_shared<CoarseModelCache>();
    cache->level = level;
    cache->key = key;
    auto &transforms = cache->transforms;
    for (int ai = 0; ai <= na; ++ai) {
        if (end == start && ai > 0)
            break;
        if (end - start >= 2 * pi - 1e-8 && ai == na)
            break;
        const double angle = start + (end - start) * ai / na;
        for (int si = 0; si <= ns; ++si) {
            if (high == low && si > 0)
                break;
            const double scale = low + (high - low) * si / ns;
            transforms.push_back({angle, scale});
        }
    }
    constexpr size_t max_bytes = 8 * 1024 * 1024;
    const size_t bytes = transforms.capacity() * sizeof(transforms[0]) + transforms.size() *
        (sizeof(std::vector<ScoringPoint>) + ml.coarse_features.size() * sizeof(ScoringPoint));
    if (bytes <= max_bytes) {
        cache->prepared.reserve(transforms.size());
        for (const auto &t : transforms)
            cache->prepared.push_back(PrepareCoarsePoints(ml.coarse_features, -t.first, t.second));
        m.data->coarse_cache.value = cache;
    }
    // Oversized grids use the original per-transform scratch path, not a retained cache.
    return cache;
}

std::vector<Candidate> SearchCoarsestLevel(const ModelSnapshot &m, const SearchPyramid &pyramid,
                                           int level, SearchDiagnostics &diag) {
    const auto &field = pyramid[level].field;
    const auto &coarse = m.data->levels[level].coarse_features;
    const auto cache = GetCoarseModelCache(m, level);
    const auto &transforms = cache->transforms;
    double threshold =
        std::max(0.15, Num(m.params, "min_score") * (0.62 + 0.16 * Num(m.params, "greediness")));
    const std::string metric = Str(m.params, "metric");
    const double fast_threshold = std::max(0.10, threshold * 0.55);
    constexpr int stride = 8;
    const int w = (field.width + stride - 1) / stride;
    const int h = (field.height + stride - 1) / stride;
    std::vector<std::vector<Candidate>> transform_candidates(transforms.size());
    ParallelFor(transforms.size(), [&](size_t begin, size_t finish, size_t) {
        std::vector<float> scores(size_t(w) * h);
        for (size_t job = begin; job < finish; ++job) {
            const double angle = transforms[job].first;
            const double scale = transforms[job].second;
            const auto scratch = cache->prepared.empty() ? PrepareCoarsePoints(coarse, -angle, scale)
                                                          : std::vector<ScoringPoint>{};
            const auto &prepared = cache->prepared.empty() ? scratch : cache->prepared[job];
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    scores[size_t(y) * w + x] = float(EvaluatePreparedCoarse(
                        prepared, field, x * stride, y * stride, metric, 5.0, fast_threshold));
            auto &peaks = transform_candidates[job];
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const double score = scores[size_t(y) * w + x];
                    if (score < fast_threshold)
                        continue;
                    bool peak = true;
                    for (int dy = -1; dy <= 1 && peak; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int xx = x + dx, yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= w || yy >= h || (dx == 0 && dy == 0))
                                continue;
                            if (scores[size_t(yy) * w + xx] > score) {
                                peak = false;
                                break;
                            }
                        }
                    if (peak)
                        peaks.push_back(
                            {{double(x * stride), double(y * stride), -angle, scale}, score});
                }
        }
    });
    std::vector<Candidate> candidates;
    for (auto &peaks : transform_candidates) {
        candidates.insert(candidates.end(), std::make_move_iterator(peaks.begin()),
                          std::make_move_iterator(peaks.end()));
        if (candidates.size() > 4000)
            candidates = MergeCandidates(std::move(candidates), 8.0, 256);
    }
    diag.evaluated_poses += transforms.size() * size_t(w) * h;
    auto selected =
        MergeCandidatesByScale(std::move(candidates), ScaleMin(m), ScaleMax(m), 8.0, 80);
    diag.coarse_candidates += selected.size();
    return selected;
}
static Candidate ImproveCoordinate(const ModelSnapshot &m, const Features &features,
                              const EdgeField &field, Pose pose, int level, int iterations,
                              SearchDiagnostics &diag, TranslationScoreCache *workspace = nullptr) {
    double factor = double(1 << level);
    double step[4] = {1.0, 1.0,
                      std::max(Num(m.params, "angle_step"), 0.8 * factor / m.data->radius),
                      std::max(Num(m.params, "iso_scale_step"), 0.7 * factor / m.data->radius)};
    double sigma = level == 0 ? 0.8 : 1.0;
    const std::string metric = Str(m.params, "metric");
    // Search parameters are immutable for this candidate. Resolve the variant
    // map once, not for every coordinate trial.
    const double scale_min = ScaleMin(m), scale_max = ScaleMax(m);
    const double angle_start = Num(m.params, "angle_start"), angle_end = Num(m.params, "angle_end");
    auto within_bounds = [&](const Pose &p) {
        const double a = angle_start +
            std::fmod(std::fmod(-p.theta - angle_start, 2 * pi) + 2 * pi, 2 * pi);
        return (a <= angle_end + .05 || a >= angle_start + 2 * pi - .05) &&
               p.scale >= scale_min - .05 && p.scale <= scale_max + .05;
    };
    TranslationScoreCache local_cache(features);
    auto &translation_cache = workspace ? *workspace : local_cache;
    auto score = [&](const Pose &p, bool translation_only = false) {
        ++diag.evaluated_poses;
        return translation_only ? translation_cache.Score(field, p, metric, sigma)
                                : EvaluatePose(features, field, p, metric, sigma);
    };
    double best = score(pose);
    for (int pass = 0; pass < iterations; ++pass) {
        bool improved = false;
        for (int k = 0; k < 4; ++k) {
            if (k == 3 && scale_min == scale_max)
                continue;
            Pose accepted = pose;
            for (int sign : {-1, 1}) {
                Pose test = pose;
                if (k == 0)
                    test.x += sign * step[k];
                if (k == 1)
                    test.y += sign * step[k];
                if (k == 2)
                    test.theta += sign * step[k];
                if (k == 3)
                    test.scale += sign * step[k];
                if (test.scale <= 0 || !within_bounds(test))
                    continue;
                double v = score(test, k < 2);
                if (v > best) {
                    best = v;
                    accepted = test;
                    improved = true;
                }
            }
            pose = accepted;
        }
        if (!improved || pass % 3 == 2)
            for (double &v : step)
                v *= 0.5;
    }
    return {pose, best};
}
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
static thread_local std::vector<TrackingTrace> tracking_traces;
std::vector<TrackingTrace> TakeTrackingTraces() {
    auto result = std::move(tracking_traces);
    tracking_traces.clear();
    return result;
}
#endif
static void ImproveCandidatesParallel(const ModelSnapshot &m, const Features &features,
                                      const EdgeField &field, std::vector<Candidate> &candidates,
                                      int level, int iterations, SearchDiagnostics &diag) {
    const size_t workers = SearchWorkerCount(candidates.size());
    std::vector<size_t> evaluations(workers);
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
    TrackingTrace trace;
    trace.level = level;
#endif
    ParallelFor(candidates.size(), [&](size_t begin, size_t end, size_t worker) {
        SearchDiagnostics local;
        TranslationScoreCache workspace(features);
        for (size_t index = begin; index < end; ++index) {
            auto &candidate = candidates[index];
            candidate =
                ImproveCoordinate(m, features, field, candidate.pose, level, iterations, local, &workspace);
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
            ++trace.workers[worker].candidates;
#endif
        }
        evaluations[worker] = local.evaluated_poses;
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
        trace.workers[worker].evaluations = local.evaluated_poses;
#endif
    }
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
    , &trace
#endif
    );
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
    if (tracking_traces.size() < 512)
        tracking_traces.push_back(std::move(trace));
#endif
    diag.evaluated_poses += std::accumulate(evaluations.begin(), evaluations.end(), size_t(0));
}

std::vector<Candidate> TrackToFinerLevel(const ModelSnapshot &m, const SearchPyramid &pyramid,
                                         std::vector<Candidate> candidates, int level,
                                         SearchDiagnostics &diag) {
    const auto &f = m.data->levels[level].features;
    const auto &field = pyramid[level].field;
    for (auto &c : candidates) {
        c.pose.x *= 2;
        c.pose.y *= 2;
    }
    ImproveCandidatesParallel(m, f, field, candidates, level, 4, diag);
    double threshold = Num(m.params, "min_score") * (level == 0 ? 0.85 : 0.68);
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](const auto &c) { return c.score < threshold; }),
                     candidates.end());
    return MergeCandidates(std::move(candidates), level == 0 ? 3.0 : 2.0, level == 0 ? 192 : 384);
}

/// Nearest-point lookup is accelerated by a Voronoi label map; neighborhood labels
/// are checked to reduce point switching near corners and intersecting edges.
static std::vector<Correspondence> BuildCorrespondences(const Features &features,
                                                        const EdgeField &field, const Pose &pose,
                                                        const std::string &metric, double radius) {
    std::vector<Correspondence> out;
    out.reserve(features.size());
    // For ignore_global_polarity, choose one sign for the whole instance.
    double polarity = 1;
    if (metric == "ignore_global_polarity") {
        double sum = 0;
        for (const auto &f : features) {
            Vec p = Rotate(f.p, pose.theta) * pose.scale + Vec{pose.x, pose.y};
            int x = int(std::lround(p.x)), y = int(std::lround(p.y));
            if (x >= 0 && y >= 0 && x < field.width && y < field.height) {
                int id = field.nearest[size_t(y) * field.width + x];
                if (id >= 0)
                    sum += Dot(Rotate(f.n, pose.theta), field.edges[id].n);
            }
        }
        polarity = sum < 0 ? -1 : 1;
    }
    for (const auto &f : features) {
        Vec p = Rotate(f.p, pose.theta) * pose.scale + Vec{pose.x, pose.y},
            n = Rotate(f.n, pose.theta);
        int x = int(std::lround(p.x)), y = int(std::lround(p.y));
        int best = -1;
        double cost = radius * radius;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= field.width || yy >= field.height)
                    continue;
                int id = field.nearest[size_t(yy) * field.width + xx];
                if (id < 0)
                    continue;
                const auto &e = field.edges[id];
                double dot = Dot(n, e.n);
                if (metric == "ignore_local_polarity")
                    dot = std::abs(dot);
                else
                    dot *= polarity;
                if (dot < 0.65)
                    continue;
                double d = Dot(p - e.p, p - e.p);
                if (d < cost) {
                    cost = d;
                    best = id;
                }
            }
        if (best >= 0) {
            const auto &e = field.edges[best];
            out.push_back({f.p, e.p, e.n});
        }
    }
    return out;
}
struct NormalEquations {
    double h[4][4]{};
    double g[4]{};
    double loss = 0;
    size_t count = 0;
};
static NormalEquations BuildNormalEquations(const std::vector<Correspondence> &pairs,
                                            const Pose &pose) {
    NormalEquations e;
    for (const auto &p : pairs) {
        Vec a = Rotate(p.model, pose.theta) * pose.scale, position = a + Vec{pose.x, pose.y};
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
        if (TracePointToPointResidual()) {
            const Vec d = position - p.image;
            const double residuals[2] = {d.x, d.y};
            const double jacobian[2][4] = {{1, 0, -a.y, a.x}, {0, 1, a.x, a.y}};
            for (int axis = 0; axis < 2; ++axis)
                for (int r = 0; r < 4; ++r) {
                    e.g[r] += jacobian[axis][r] * residuals[axis];
                    for (int c = 0; c < 4; ++c)
                        e.h[r][c] += jacobian[axis][r] * jacobian[axis][c];
                }
            e.loss += Dot(d, d);
            ++e.count;
            continue;
        }
#endif
        double residual = Dot(p.normal, position - p.image);
        // Point-to-normal Jacobian for (tx,ty,theta,log(scale)).
        double j[4] = {p.normal.x, p.normal.y, -p.normal.x * a.y + p.normal.y * a.x,
                       Dot(p.normal, a)};
        for (int r = 0; r < 4; ++r) {
            e.g[r] += j[r] * residual;
            for (int c = 0; c < 4; ++c)
                e.h[r][c] += j[r] * j[c];
        }
        e.loss += residual * residual;
        ++e.count;
    }
    return e;
}
static bool SolvePoseIncrement(NormalEquations e, double damping, bool scale_free,
                               std::array<double, 4> &delta, bool normalize = false) {
    int n = scale_free ? 4 : 3;
    double units[4] = {1, 1, 1, 1};
    if (normalize) {
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(e.h[i][i]) || e.h[i][i] < 1e-12)
                return false;
            units[i] = std::sqrt(e.h[i][i]);
        }
        for (int i = 0; i < n; ++i) {
            e.g[i] /= units[i];
            for (int j = 0; j < n; ++j)
                e.h[i][j] /= units[i] * units[j];
        }
    }
    double a[4][5]{};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            a[i][j] = e.h[i][j];
        a[i][i] += damping * std::max(1.0, e.h[i][i]);
        a[i][n] = -e.g[i];
    }
    // Pivoting on the small dense system; double accumulation preserves precision.
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(a[i][k]) > std::abs(a[pivot][k]))
                pivot = i;
        if (std::abs(a[pivot][k]) < 1e-10)
            return false;
        for (int j = k; j <= n; ++j)
            std::swap(a[pivot][j], a[k][j]);
        double v = a[k][k];
        for (int j = k; j <= n; ++j)
            a[k][j] /= v;
        for (int i = 0; i < n; ++i)
            if (i != k) {
                double q = a[i][k];
                for (int j = k; j <= n; ++j)
                    a[i][j] -= q * a[k][j];
            }
    }
    delta = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i)
        delta[i] = a[i][n] / units[i];
    return true;
}
Pose InterpolateScorePeak(const ModelSnapshot &m, const PyramidLevel &image, Pose pose) {
    const auto &f = m.data->levels[0].features;
    double steps[4] = {0.5, 0.5, 0.5 / m.data->radius, 0.5 / m.data->radius};
    for (int pass = 0; pass < 3; ++pass)
        for (int k = 0; k < 4; ++k) {
            if (k == 3 && ScaleMin(m) == ScaleMax(m))
                continue;
            auto shift = [&](Pose p, double d) {
                if (k == 0)
                    p.x += d;
                if (k == 1)
                    p.y += d;
                if (k == 2)
                    p.theta += d;
                if (k == 3)
                    p.scale += d;
                return p;
            };
            double left = EvaluatePose(f, image.field, shift(pose, -steps[k]),
                                       Str(m.params, "metric"), 0.8),
                   mid = EvaluatePose(f, image.field, pose, Str(m.params, "metric"), 0.8),
                   right = EvaluatePose(f, image.field, shift(pose, steps[k]),
                                        Str(m.params, "metric"), 0.8);
            double den = left - 2 * mid + right;
            if (den < -1e-10) {
                double d = std::clamp(0.5 * (left - right) / den, -1.0, 1.0) * steps[k];
                Pose trial = shift(pose, d);
                if (WithinBounds(m, trial, 0.05))
                    pose = trial;
            }
        }
    return pose;
}
Candidate RefineInstance(const ModelSnapshot &m, const PyramidLevel &image, Candidate candidate,
                         bool baseline_prepared) {
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
    candidate = TraceRefinementSeed(m, image, candidate);
    TraceRefinement(m, candidate, -1, "begin");
#endif
    auto mode = Str(m.params, "subpixel");
    const bool extra_refinement = Str(m.params, "refinement_method") != "nearest_point" &&
                                  mode.find("least_squares") == 0;
    Candidate baseline = candidate;
    if (extra_refinement && !baseline_prepared) {
        auto legacy = m;
        legacy.params["refinement_method"] = std::string("nearest_point");
        legacy.params["refinement_radius"] = 1.5;
        baseline = RefineInstance(legacy, image, candidate);
        candidate = baseline;
    }
    Pose pose = candidate.pose;
    if (mode == "none") {
        pose.x = std::round(pose.x);
        pose.y = std::round(pose.y);
        double step = Num(m.params, "angle_step"), start = Num(m.params, "angle_start");
        pose.theta = -(start + std::round((-pose.theta - start) / step) * step);
    } else if (mode == "interpolation")
        pose = InterpolateScorePeak(m, image, pose);
    else {
        int iterations = mode == "least_squares" ? 10 : mode == "least_squares_high" ? 20 : 30;
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
        iterations = TraceIterationBudget(iterations);
#endif
        bool scale_free = ScaleMin(m) != ScaleMax(m);
        double damping = 1e-4;
        // Dense points reduce discretization bias, while search retains sparse points.
        const auto method = Str(m.params, "refinement_method");
        const auto &all = method == "gradient_gaussian" ? m.data->gaussian_dense : m.data->dense;
        const auto subset = all.size() > 1200 ? SelectModelFeatures(all, 1200) : Features{};
        const auto &dense = all.size() <= 1200 ? all : subset;
        Require(dense.size() >= 12, ErrorCode::Geometry, "Insufficient refinement model points");
        const auto metric = Str(m.params, "metric");
        const bool continuous = method != "nearest_point";
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
        std::vector<Correspondence> frozen_pairs;
#endif
        for (int iteration = 0; iteration < iterations; ++iteration) {
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
            TraceRefinement(m, {pose, candidate.score}, iteration, "iteration");
#endif
            double radius = iteration < 3 ? std::max(3.0, Num(m.params, "refinement_radius"))
                                                : Num(m.params, "refinement_radius");
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
            radius = TraceCorrespondenceRadius(radius);
#endif
            auto correspond = [&](const Pose &p) {
                return continuous ? ContinuousCorrespondences(dense, image, p, metric, radius, method)
                                  : BuildCorrespondences(dense, image.field, p, metric, radius);
            };
            auto pairs = correspond(pose);
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
            if (update_policy == "frozen") {
                if (iteration == 0)
                    frozen_pairs = pairs;
                pairs = frozen_pairs;
            }
#endif
            if (pairs.size() < 12 || pairs.size() < dense.size() / 5)
                break;
            auto equations = BuildNormalEquations(pairs, pose);
            std::array<double, 4> step{};
            if (!SolvePoseIncrement(equations, damping, scale_free, step, continuous))
                break;
            double translation = std::hypot(step[0], step[1]);
            if (translation > 1) {
                step[0] /= translation;
                step[1] /= translation;
            }
            step[2] = std::clamp(step[2], -0.05, 0.05);
            step[3] = std::clamp(step[3], -0.03, 0.03);
            Pose trial{pose.x + step[0], pose.y + step[1], pose.theta + step[2],
                       pose.scale * std::exp(step[3])};
            if (!WithinBounds(m, trial, 0.05)) {
                damping *= 10;
                if (damping > 1e6)
                    break;
                continue;
            }
            double loss = BuildNormalEquations(pairs, trial).loss;
            bool accepted = loss <= equations.loss;
            if (continuous && accepted) {
                // Compare on a fixed model support as well as frozen correspondences.
                // Dropping difficult points must not manufacture a lower objective.
                auto trial_pairs = correspond(trial);
                accepted = RefinementSupportLoss(trial_pairs, trial, dense.size(), radius) <=
                           RefinementSupportLoss(pairs, pose, dense.size(), radius) + 1e-12 * dense.size();
            }
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
            const auto diagnostic_trial_pairs = correspond(trial);
            const auto diagnostic_current_pairs = correspond(pose);
            const double reassociated_loss = BuildNormalEquations(diagnostic_trial_pairs, trial).loss;
            const double support_before = BuildNormalEquations(diagnostic_current_pairs, pose).loss +
                (dense.size() - diagnostic_current_pairs.size()) * radius * radius;
            const double support_after = reassociated_loss +
                (dense.size() - diagnostic_trial_pairs.size()) * radius * radius;
            if (update_policy == "support")
                accepted = accepted && support_after <= support_before + 1e-12 * dense.size();
            TraceUpdate(iteration, radius, pairs.size(), diagnostic_trial_pairs.size(), equations.loss,
                        loss, reassociated_loss, support_before, support_after, damping,
                        std::max({std::hypot(step[0], step[1]), std::abs(step[2]) * m.data->radius,
                                  std::abs(step[3]) * m.data->radius}), accepted);
#endif
            if (accepted) {
                pose = trial;
                damping = std::max(1e-8, damping / 3);
                if (std::hypot(step[0], step[1]) < 1e-5 &&
                    std::abs(step[2]) * m.data->radius < 1e-5 &&
                    std::abs(step[3]) * m.data->radius < 1e-5)
                    break;
            } else {
                damping *= 10;
                if (damping > 1e6)
                    break;
            }
        }
    }
    candidate.pose = pose;
    candidate.score =
        EvaluatePose(m.data->levels[0].features, image.field, pose, Str(m.params, "metric"), 0.8);
#ifdef SHAPE_MATCH_TRACE_REFINEMENT
    TraceRefinement(m, candidate, -1, "end");
#endif
    // Experimental observations must not turn an accepted baseline match into
    // a below-threshold detection. Keep the baseline pose, not a stale score.
    if (extra_refinement && candidate.score < Num(m.params, "min_score") &&
        baseline.score >= Num(m.params, "min_score"))
        return baseline;
    return candidate;
}

std::vector<Match> FinalizeMatches(std::vector<Match> matches,
                                   const std::vector<ModelSnapshot> &models) {
    std::stable_sort(matches.begin(), matches.end(),
                     [](const Match &a, const Match &b) { return a.score > b.score; });
    bool global = false;
    double global_overlap = 1;
    for (const auto &m : models) {
        global |= Bool(m.params, "max_overlap_global_enable");
        global_overlap = std::min(global_overlap, Num(m.params, "max_overlap"));
    }
    std::vector<Match> out;
    std::map<uint64_t, int> count;
    for (auto &m : matches) {
        if (m.score < Num(m.model.params, "min_score"))
            continue;
        bool overlap = false;
        for (const auto &k : out)
            if (global || m.model.id == k.model.id) {
                double limit = global ? global_overlap : Num(m.model.params, "max_overlap");
                if (RectangleOverlap(m, k) > limit + 1e-9) {
                    overlap = true;
                    break;
                }
            }
        if (overlap)
            continue;
        const auto &value = m.model.params.at("num_matches");
        int64_t limit = std::holds_alternative<int64_t>(value) ? std::get<int64_t>(value) : 0;
        if (limit > 0 && count[m.model.id] >= limit)
            continue;
        ++count[m.model.id]; // Strict parameter-space boundaries apply AFTER num_matches.
        if (Bool(m.model.params, "strict_boundaries") && !WithinBounds(m.model, m.pose, 1e-10))
            continue;
        out.push_back(std::move(m));
    }
    return out;
}
} // namespace shape_match::detail

namespace shape_match {
void FindGenericShapeModel(const HObject &object, const HTuple &ids, HTuple *result_id,
                           HTuple *count) {
    using namespace detail;
    Output(result_id);
    Output(count);
    Require(result_id != count, ErrorCode::Value, "Output tuples must not alias");
    auto start = Clock::now();
    Require(object.Type() == HObject::Kind::Images && object.Count() == 1, ErrorCode::Unsupported,
            "This release accepts one search image shared by all models");
    const auto &image = object.GetImage();
    auto models = SnapshotModels(ids);
    auto result = std::make_shared<MatchResult>();
    result->models = models;
    int levels = 1;
    bool precise = false;
    double contrast = 255;
    for (const auto &m : models) {
        levels = std::max(levels, int(m.data->levels.size()));
        contrast = std::min(contrast, Num(m.params, "min_contrast"));
        precise |= Str(m.params, "refinement_method") == "gradient_gaussian" &&
                   Str(m.params, "subpixel").find("least_squares") == 0 &&
                   Num(m.params, "pyramid_level_lowest") == 1;
    }
    auto t = Clock::now();
    auto pyramid = BuildSearchPyramid(image, levels, contrast, precise);
    result->diagnostics.pyramid_ms = Elapsed(t);
    std::vector<Match> matches;
    bool allow_border = false;
    for (const auto &m : models)
        allow_border |= Str(m.params, "border_shape_models") == "true";
    for (const auto &m : models) {
        // Lower pyramid levels must use each model's own min_contrast. Reuse when equal.
        int top = int(m.data->levels.size()) - 1;
        if (!IsAuto(m.params.at("pyramid_level_highest")))
            top = std::min(top, int(Num(m.params, "pyramid_level_highest")) - 1);
        const SearchPyramid *working = &pyramid;
        SearchPyramid specific;
        if (Num(m.params, "min_contrast") > contrast) {
            t = Clock::now();
            specific = BuildSearchPyramid(image, levels, Num(m.params, "min_contrast"));
            specific[top] = pyramid[top];
            specific[0].precise_gradient = pyramid[0].precise_gradient;
            working = &specific;
            result->diagnostics.pyramid_ms += Elapsed(t);
        }
        int lowest = int(Num(m.params, "pyramid_level_lowest")) - 1;
        Require(lowest <= top, ErrorCode::Value, "Lowest search level exceeds highest");
        t = Clock::now();
        auto candidates = SearchCoarsestLevel(m, *working, top, result->diagnostics);
        result->diagnostics.top_level_ms += Elapsed(t);
        t = Clock::now();
        ImproveCandidatesParallel(m, m.data->levels[top].features, (*working)[top].field,
                                  candidates, top, 9, result->diagnostics);
        // The broad coarse probe deliberately over-generates hypotheses.  Once
        // they have been optimized and rescored with the complete top-level
        // model, keep a compact deterministic beam for the expensive fine
        // levels.
        candidates = MergeCandidates(std::move(candidates), 4, 48);
        for (int l = top - 1; l >= lowest; --l)
            candidates =
                TrackToFinerLevel(m, *working, std::move(candidates), l, result->diagnostics);
        result->diagnostics.tracking_ms += Elapsed(t);
        t = Clock::now();
        const bool prepare_baseline = lowest == 0 && Str(m.params, "refinement_method") != "nearest_point" &&
                                      Str(m.params, "subpixel").find("least_squares") == 0;
        if (prepare_baseline) {
            auto legacy = m;
            legacy.params["refinement_method"] = std::string("nearest_point");
            legacy.params["refinement_radius"] = 1.5;
            ParallelFor(candidates.size(), [&](size_t begin, size_t end, size_t) {
                for (size_t i = begin; i < end; ++i)
                    candidates[i] = RefineInstance(legacy, (*working)[0], candidates[i]);
            });
            // Merge converged duplicates before the more expensive observations;
            // distinct angular symmetry branches remain separate.
            const size_t capacity = candidates.size();
            candidates = MergeCandidates(std::move(candidates), 1.0, capacity);
        }
        ParallelFor(candidates.size(), [&](size_t begin, size_t end, size_t) {
            for (size_t index = begin; index < end; ++index)
                if (lowest == 0)
                    candidates[index] = RefineInstance(m, (*working)[0], candidates[index], prepare_baseline);
        });
        for (auto c : candidates) {
            double factor = double(1 << lowest);
            c.pose.x *= factor;
            c.pose.y *= factor;
            // A non-default lower stopping level deliberately skips fine refinement.
            int x = int(std::lround(c.pose.x)), y = int(std::lround(c.pose.y));
            if (x < 0 || y < 0 || x >= image.width || y >= image.height ||
                !image.domain[size_t(y) * image.width + x])
                continue;
            if (!allow_border) {
                bool valid = true;
                for (const auto &f : m.data->dense) {
                    Vec p = Rotate(f.p, c.pose.theta) * c.pose.scale + Vec{c.pose.x, c.pose.y};
                    if (p.x < 0 || p.y < 0 || p.x > image.width - 1 || p.y > image.height - 1) {
                        valid = false;
                        break;
                    }
                }
                if (!valid)
                    continue;
            }
            matches.push_back({m, c.pose, c.score});
            ++result->diagnostics.refined_candidates;
        }
        result->diagnostics.refinement_ms += Elapsed(t);
    }
    result->matches = FinalizeMatches(std::move(matches), models);
    result->diagnostics.total_ms = Elapsed(start);
    *count = HTuple(int64_t(result->matches.size()));
    *result_id = HTuple(HHandle{result});
}
SearchDiagnostics GetSearchDiagnostics(const HTuple &id) {
    return detail::ResolveResult(id)->diagnostics;
}
size_t GetSearchThreadCount() {
    return detail::SearchWorkerCount(std::numeric_limits<size_t>::max());
}
} // namespace shape_match
