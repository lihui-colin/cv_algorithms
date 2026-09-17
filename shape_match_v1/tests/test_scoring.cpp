#include "../src/search.cpp"
#include <iostream>
#include <random>
#include <future>

using namespace shape_match::detail;

// Frozen pre-optimization scalar oracle; keep lround and summation order.
static double Reference(const Features &features, const EdgeField &im, const Pose &pose,
                        const std::string &metric, double sigma, double minimum) {
    static const auto table = [] {
        std::array<float, 4097> t{};
        for (int i = 0; i <= 4096; ++i)
            t[i] = float(std::exp(-9.0 * i / 4096));
        return t;
    }();
    double positive = 0, negative = 0;
    const double c = std::cos(pose.theta), s = std::sin(pose.theta), inv = .5 / (sigma * sigma);
    for (size_t i = 0; i < features.size(); ++i) {
        const auto &f = features[i];
        const double x = pose.x + pose.scale * (c * f.p.x - s * f.p.y);
        const double y = pose.y + pose.scale * (s * f.p.x + c * f.p.y);
        const int ix = int(std::lround(x)), iy = int(std::lround(y));
        if (ix >= 0 && iy >= 0 && ix < im.width && iy < im.height) {
            const int id = im.nearest[size_t(iy) * im.width + ix];
            if (id >= 0) {
                const auto &e = im.edges[id];
                const double dist = Sq(x - e.p.x) + Sq(y - e.p.y);
                if (dist < 18 * sigma * sigma) {
                    const double dot =
                        (c * f.n.x - s * f.n.y) * e.n.x + (s * f.n.x + c * f.n.y) * e.n.y;
                    const double coordinate = std::clamp(dist * inv, 0.0, 9.0) * (4096 / 9.0);
                    const int k = std::min(4095, int(coordinate));
                    const double weight = table[k] + (coordinate - k) * (table[k + 1] - table[k]);
                    if (metric == "ignore_local_polarity")
                        positive += std::abs(dot) * weight;
                    else {
                        positive += std::max(0.0, dot) * weight;
                        if (metric == "ignore_global_polarity")
                            negative += std::max(0.0, -dot) * weight;
                    }
                }
            }
        }
        if (minimum > 0 &&
            (std::max(positive, negative) + features.size() - (i + 1)) < minimum * features.size())
            return 0;
    }
    return features.empty() ? 0
                            : std::clamp(std::max(positive, negative) / features.size(), 0.0, 1.0);
}

int main() {
    EdgeField field;
    field.width = field.height = 64;
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) {
            const double angle = (x + y) * .3;
            field.nearest.push_back((x + y) % 11 ? int(field.edges.size()) : -1);
            field.edges.push_back({{x + .1, y - .2}, {std::cos(angle), std::sin(angle)}, 1, 1, 0});
        }
    Features features;
    for (int i = 0; i < 120; ++i) {
        double angle = i * .1;
        features.push_back({{15 * std::cos(angle), 13 * std::sin(angle)},
                            {std::cos(angle), std::sin(angle)},
                            1,
                            1,
                            0});
    }
    std::mt19937 random(1729);
    std::uniform_real_distribution<double> position(-20, 84), angle(-4, 4), scale(.5, 1.5);
    std::vector<Pose> poses;
    for (int i = 0; i < 1000; ++i)
        poses.push_back({position(random), position(random), angle(random), scale(random)});
    for (const auto &metric : {"use_polarity", "ignore_local_polarity", "ignore_global_polarity"})
        for (double sigma : {.8, 1., 5.})
            for (double minimum : {0., .3, .7})
                for (const auto &pose : poses) {
                    const double a = Reference(features, field, pose, metric, sigma, minimum);
                    const double b = EvaluatePose(features, field, pose, metric, sigma, minimum);
                    if (a != b) {
                        std::cerr << "Score mismatch: " << a << " " << b << '\n';
                        return 1;
                    }
                }
    TranslationScoreCache cache(features);
    for (const auto &metric : {"use_polarity", "ignore_local_polarity", "ignore_global_polarity"})
        for (const auto &base : poses)
            for (int trial = 0; trial < 4; ++trial) {
                auto pose = base;
                pose.x += .25 * trial;
                pose.y -= .125 * trial;
                if (cache.Score(field, pose, metric, .8) !=
                    EvaluatePose(features, field, pose, metric, .8)) {
                    std::cerr << "Translation cache mismatch\n";
                    return 1;
                }
            }
    ModelSnapshot model;
    model.params = MakeDefaultParams(1);
    model.params["angle_start"] = -pi;
    model.params["angle_end"] = pi;
    model.params["angle_step"] = .05;
    model.params["iso_scale_step"] = .05;
    model.params["iso_scale_min"] = .5;
    model.params["iso_scale_max"] = 1.5;
    model.params["metric"] = std::string("use_polarity");
    auto trained = std::make_shared<TrainedData>();
    trained->radius = 20;
    model.data = trained;
    trained->levels.push_back({features, 1, SelectModelFeatures(features, 12)});
    trained->levels.push_back({features, 2, SelectModelFeatures(features, 12)});
    auto check = [](bool ok) {
        if (!ok)
            throw std::runtime_error("Coarse model cache regression");
    };
    const auto initial = GetCoarseModelCache(model, 0);
    check(GetCoarseModelCache(model, 0) == initial);
    check(initial->prepared.size() == initial->transforms.size());
    for (size_t i = 0; i < initial->transforms.size(); ++i) {
        const auto t = initial->transforms[i];
        const auto reference = PrepareCoarsePoints(SelectModelFeatures(features, 12), -t.first, t.second);
        const auto &cached = initial->prepared[i];
        check(cached.size() == reference.size());
        for (size_t j = 0; j < cached.size(); ++j) {
            const auto &a = cached[j], &b = reference[j];
            check(a.x == b.x && a.y == b.y && a.nx == b.nx && a.ny == b.ny &&
                  a.pixel_x == b.pixel_x && a.pixel_y == b.pixel_y);
        }
    }
    for (const auto &key : {"angle_start", "angle_end", "angle_step", "iso_scale_step",
                           "iso_scale_min", "iso_scale_max", "restrict_iso_scale_min",
                           "restrict_iso_scale_max"}) {
        auto changed = model;
        changed.params[key] = std::string(key) == "restrict_iso_scale_min" ? .7 :
                              std::string(key) == "restrict_iso_scale_max" ? 1.2 :
                              Num(model.params, key) + .01;
        const auto updated = GetCoarseModelCache(changed, 0);
        check(updated != initial && updated->key != initial->key);
        check(GetCoarseModelCache(changed, 0) == updated);
    }
    check(GetCoarseModelCache(model, 1)->level == 1);
    auto clone = std::make_shared<TrainedData>(*trained);
    check(!clone->coarse_cache.value);
    // Concurrent searches publish complete immutable entries, including on cache misses.
    trained->coarse_cache.value.reset();
    std::vector<std::future<std::shared_ptr<const CoarseModelCache>>> pending;
    for (int i = 0; i < 4; ++i)
        pending.push_back(std::async(std::launch::async, [&] { return GetCoarseModelCache(model, 0); }));
    const auto shared = pending.front().get();
    for (size_t i = 1; i < pending.size(); ++i)
        check(pending[i].get() == shared);
    auto wide = model;
    wide.data = clone;
    clone->radius = 10000;
    wide.params["angle_step"] = .001;
    wide.params["iso_scale_step"] = .1;
    const auto oversized = GetCoarseModelCache(wide, 0);
    check(oversized->prepared.empty() && !clone->coarse_cache.value);
    std::cout << "PASS: coarse cache identity, exact transforms, invalidation, copy, concurrency and cap\n";
    for (int iterations : {0, 4, 9})
        for (size_t i = 0; i < 20; ++i) {
            shape_match::SearchDiagnostics diagnostics;
            const auto result = ImproveCoordinate(model, features, field, poses[i], 0, iterations, diagnostics);
            if (result.score != EvaluatePose(features, field, result.pose, "use_polarity", .8)) {
                std::cerr << "Reused final score mismatch\n";
                return 1;
            }
        }
    for (size_t count : {0, 1, 3, 15, 16, 17, 31, 80, 193})
        for (int iterations : {0, 4, 9}) {
            std::vector<Candidate> original;
            for (size_t i = 0; i < count; ++i)
                original.push_back({poses[i % poses.size()], .123});
            auto expected = original;
            shape_match::SearchDiagnostics reference_diag;
            for (auto &candidate : expected)
                candidate = ImproveCoordinate(model, features, field, candidate.pose, 0, iterations,
                                              reference_diag);
            for (int repeat = 0; repeat < 4; ++repeat) {
                auto actual = original;
                shape_match::SearchDiagnostics actual_diag;
                ImproveCandidatesParallel(model, features, field, actual, 0, iterations, actual_diag);
                check(actual_diag.evaluated_poses == reference_diag.evaluated_poses);
                check(actual.size() == expected.size());
                for (size_t i = 0; i < actual.size(); ++i) {
                    const auto &a = actual[i], &b = expected[i];
                    check(a.pose.x == b.pose.x && a.pose.y == b.pose.y &&
                          a.pose.theta == b.pose.theta && a.pose.scale == b.pose.scale && a.score == b.score);
                }
            }
        }
    std::cout << "PASS: serial/parallel candidates, order, scores and evaluation counts\n";
    volatile double sink = 0;
    for (int round = 0; round < 3; ++round) {
        for (bool optimized : {false, true}) {
            const auto start = Clock::now();
            for (int repeat = 0; repeat < 100; ++repeat)
                for (const auto &pose : poses)
                    sink = sink + (optimized
                                       ? EvaluatePose(features, field, pose, "use_polarity", .8)
                                       : Reference(features, field, pose, "use_polarity", .8, 0));
            std::cout << (optimized ? "optimized_ms=" : "reference_ms=") << Elapsed(start) << '\n';
        }
    }
    for (bool cached : {false, true}) {
        const auto start = Clock::now();
        for (int repeat = 0; repeat < 30; ++repeat)
            for (const auto &base : poses)
                for (int trial = 0; trial < 4; ++trial) {
                    auto pose = base;
                    pose.x += .25 * trial;
                    pose.y -= .125 * trial;
                    sink = sink + (cached ? cache.Score(field, pose, "use_polarity", .8)
                                          : EvaluatePose(features, field, pose, "use_polarity", .8));
                }
        std::cout << (cached ? "translation_cached_ms=" : "translation_direct_ms=")
                  << Elapsed(start) << '\n';
    }
    std::cout << "PASS: 27000 scalar scores, 12000 cached scores and 60 final-score reuse checks\n";
}
