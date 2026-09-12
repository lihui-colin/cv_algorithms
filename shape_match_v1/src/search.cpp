#include "internal.hpp"
#include <limits>

namespace shape_match::detail {
/// Per-point distance/orientation score; NOT HALCON's proprietary score formula.
/// Missing points contribute zero, so partial overlap cannot normalize to score=1.
double EvaluatePose(const Features &f, const EdgeField &im, const Pose &pose,
                    const std::string &metric, double sigma, double min_score) {
    double c = std::cos(pose.theta), s = std::sin(pose.theta), positive = 0, negative = 0;
    const bool global = metric == "ignore_global_polarity",
               local = metric == "ignore_local_polarity";
    double inv = 0.5 / (sigma * sigma);
    size_t seen = 0;
    for (const auto &v : f) {
        double x = pose.x + pose.scale * (c * v.p.x - s * v.p.y),
               y = pose.y + pose.scale * (s * v.p.x + c * v.p.y);
        int ix = int(std::lround(x)), iy = int(std::lround(y));
        if (ix >= 0 && iy >= 0 && ix < im.width && iy < im.height) {
            int id = im.nearest[size_t(iy) * im.width + ix];
            if (id >= 0) {
                const auto &q = im.edges[id];
                double dist = Sq(x - q.p.x) + Sq(y - q.p.y);
                if (dist < 18 * sigma * sigma) {
                    double dot = (c * v.n.x - s * v.n.y) * q.n.x + (s * v.n.x + c * v.n.y) * q.n.y;
                    double spatial = std::exp(-dist * inv);
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
        ++seen;
        // Mathematical upper bound, valid also for global polarity reversal.
        if (min_score > 0 &&
            (std::max(positive, negative) + f.size() - seen) < min_score * f.size())
            return 0;
    }
    return f.empty() ? 0 : std::clamp(std::max(positive, negative) / f.size(), 0.0, 1.0);
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
std::vector<Candidate> SearchCoarsestLevel(const ModelSnapshot &m, const SearchPyramid &pyramid,
                                           int level, SearchDiagnostics &diag) {
    const auto &field = pyramid[level].field;
    const auto &ml = m.data->levels[level];
    double start = Num(m.params, "angle_start"), end = Num(m.params, "angle_end");
    double da =
        std::min(0.20, std::max(Num(m.params, "angle_step"), ml.factor * 0.8 / m.data->radius));
    double ds = std::max(Num(m.params, "iso_scale_step"), ml.factor * 0.75 / m.data->radius);
    int na = std::max(1, int(std::ceil((end - start) / da))),
        ns = std::max(1, int(std::ceil((ScaleMax(m) - ScaleMin(m)) / ds)));
    Require(int64_t(na + 1) * (ns + 1) < 1000000, ErrorCode::Value,
            "Angle/scale grid exceeds implementation resource envelope");
    Features coarse = SelectModelFeatures(ml.features, 48);
    double threshold =
        std::max(0.15, Num(m.params, "min_score") * (0.62 + 0.16 * Num(m.params, "greediness")));
    std::vector<Candidate> candidates;
    for (int ai = 0; ai <= na; ++ai) {
        if (end == start && ai > 0)
            break;
        if (end - start >= 2 * pi - 1e-8 && ai == na)
            break;
        double angle = start + (end - start) * ai / na;
        for (int si = 0; si <= ns; ++si) {
            if (ScaleMax(m) == ScaleMin(m) && si > 0)
                break;
            double scale = ScaleMin(m) + (ScaleMax(m) - ScaleMin(m)) * si / ns;
            // Grid stride 2 at coarse level; sigma=1.25 tolerates nearest-grid displacement.
            int stride = 2, w = (field.width + stride - 1) / stride,
                h = (field.height + stride - 1) / stride;
            std::vector<float> scores(size_t(w) * h);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    Pose p{double(x * stride), double(y * stride), -angle, scale};
                    scores[size_t(y) * w + x] = float(
                        EvaluatePose(coarse, field, p, Str(m.params, "metric"), 1.25, threshold));
                    ++diag.evaluated_poses;
                }
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    double score = scores[size_t(y) * w + x];
                    if (score < threshold)
                        continue;
                    bool peak = true;
                    for (int dy = -1; dy <= 1 && peak; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int xx = x + dx, yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= w || yy >= h || (dx == 0 && dy == 0))
                                continue;
                            if (scores[size_t(yy) * w + xx] > score) {
                                peak = false;
                                break;
                            }
                        }
                    if (peak)
                        candidates.push_back(
                            {{double(x * stride), double(y * stride), -angle, scale}, score});
                }
            if (candidates.size() > 12000)
                candidates = MergeCandidates(std::move(candidates), 2.0, 4096);
        }
    }
    auto selected = MergeCandidates(std::move(candidates), 2.0, 512);
    diag.coarse_candidates += selected.size();
    return selected;
}
static Pose ImproveCoordinate(const ModelSnapshot &m, const Features &features,
                              const EdgeField &field, Pose pose, int level, int iterations,
                              SearchDiagnostics &diag) {
    double factor = double(1 << level);
    double step[4] = {1.0, 1.0,
                      std::max(Num(m.params, "angle_step"), 0.8 * factor / m.data->radius),
                      std::max(Num(m.params, "iso_scale_step"), 0.7 * factor / m.data->radius)};
    double sigma = level == 0 ? 0.8 : 1.0;
    auto score = [&](const Pose &p) {
        ++diag.evaluated_poses;
        return EvaluatePose(features, field, p, Str(m.params, "metric"), sigma);
    };
    double best = score(pose);
    for (int pass = 0; pass < iterations; ++pass) {
        bool improved = false;
        for (int k = 0; k < 4; ++k) {
            if (k == 3 && ScaleMin(m) == ScaleMax(m))
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
                if (test.scale <= 0 || !WithinBounds(m, test, 0.05))
                    continue;
                double v = score(test);
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
    return pose;
}
std::vector<Candidate> TrackToFinerLevel(const ModelSnapshot &m, const SearchPyramid &pyramid,
                                         std::vector<Candidate> candidates, int level,
                                         SearchDiagnostics &diag) {
    const auto &f = m.data->levels[level].features;
    const auto &field = pyramid[level].field;
    for (auto &c : candidates) {
        c.pose.x *= 2;
        c.pose.y *= 2;
        c.pose = ImproveCoordinate(m, f, field, c.pose, level, 9, diag);
        c.score = EvaluatePose(f, field, c.pose, Str(m.params, "metric"), level == 0 ? 0.8 : 1.0);
    }
    double threshold = Num(m.params, "min_score") * 0.68;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](const auto &c) { return c.score < threshold; }),
                     candidates.end());
    return MergeCandidates(std::move(candidates), level == 0 ? 3.0 : 2.0, level == 0 ? 192 : 384);
}

struct Correspondence {
    Vec model, image, normal;
};
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
                               std::array<double, 4> &delta) {
    int n = scale_free ? 4 : 3;
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
        delta[i] = a[i][n];
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
Candidate RefineInstance(const ModelSnapshot &m, const PyramidLevel &image, Candidate candidate) {
    auto mode = Str(m.params, "subpixel");
    Pose pose = candidate.pose;
    if (mode == "none") {
        pose.x = std::round(pose.x);
        pose.y = std::round(pose.y);
        double step = Num(m.params, "angle_step"), start = Num(m.params, "angle_start");
        pose.theta = -(start + std::round((-pose.theta - start) / step) * step);
    } else if (mode == "interpolation")
        pose = InterpolateScorePeak(m, image, pose);
    else {
        int iterations = mode == "least_squares" ? 10 : mode == "least_squares_high" ? 25 : 50;
        bool scale_free = ScaleMin(m) != ScaleMax(m);
        double damping = 1e-4;
        // Dense points reduce discretization bias, while search retains sparse points.
        Features dense = SelectModelFeatures(m.data->dense, 1200);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            auto pairs = BuildCorrespondences(dense, image.field, pose, Str(m.params, "metric"),
                                              iteration < 3 ? 3.0 : 1.5);
            if (pairs.size() < 12 || pairs.size() < dense.size() / 5)
                break;
            auto equations = BuildNormalEquations(pairs, pose);
            std::array<double, 4> step{};
            if (!SolvePoseIncrement(equations, damping, scale_free, step))
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
            if (loss <= equations.loss) {
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
    double contrast = 255;
    for (const auto &m : models) {
        levels = std::max(levels, int(m.data->levels.size()));
        contrast = std::min(contrast, Num(m.params, "min_contrast"));
    }
    auto t = Clock::now();
    auto pyramid = BuildSearchPyramid(image, levels, contrast);
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
            working = &specific;
            result->diagnostics.pyramid_ms += Elapsed(t);
        }
        int lowest = int(Num(m.params, "pyramid_level_lowest")) - 1;
        Require(lowest <= top, ErrorCode::Value, "Lowest search level exceeds highest");
        t = Clock::now();
        auto candidates = SearchCoarsestLevel(m, *working, top, result->diagnostics);
        result->diagnostics.top_level_ms += Elapsed(t);
        t = Clock::now();
        for (auto &c : candidates) {
            c.pose = ImproveCoordinate(m, m.data->levels[top].features, (*working)[top].field,
                                       c.pose, top, 9, result->diagnostics);
            c.score = EvaluatePose(m.data->levels[top].features, (*working)[top].field, c.pose,
                                   Str(m.params, "metric"), 1.0);
        }
        candidates = MergeCandidates(std::move(candidates), 2, 384);
        for (int l = top - 1; l >= lowest; --l)
            candidates =
                TrackToFinerLevel(m, *working, std::move(candidates), l, result->diagnostics);
        result->diagnostics.tracking_ms += Elapsed(t);
        t = Clock::now();
        for (auto c : candidates) {
            double factor = double(1 << lowest);
            c.pose.x *= factor;
            c.pose.y *= factor;
            if (lowest == 0)
                c = RefineInstance(m, (*working)[0], c);
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
} // namespace shape_match
