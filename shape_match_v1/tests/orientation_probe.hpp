#pragma once
#include "normal_profile_probe.hpp"

namespace orientation_probe {
using namespace shape_match;
using namespace shape_match::detail;
// Fixed support: absent/weak gradients contribute zero, never shrink denominator.
inline double Score(const Features &features, const GradientField &g, Pose p) {
    if (features.empty())
        return 0;
    double sum = 0;
    for (const auto &f : features) {
        Vec q = Rotate(f.p, p.theta) * p.scale + Vec{p.x, p.y};
        Vec v = profile_probe::Sample(g, q);
        double magnitude = std::hypot(v.x, v.y);
        if (magnitude >= 3)
            sum += Dot(Rotate(f.n, p.theta), v) * (1 / magnitude);
    }
    return sum / features.size();
}
inline Pose Refine(const ModelSnapshot &model, const Features &features, const GradientField &g,
                   Pose seed) {
    Pose pose = seed;
    double best = Score(features, g, pose);
    // Parameter steps correspond approximately to the same motion at model radius.
    for (double step = 0.5; step >= 0.00001; step *= 0.5) {
        for (int sweep = 0; sweep < 100; ++sweep) {
            bool improved = false;
            for (int axis = 0; axis < 4; ++axis) {
                Pose selected = pose;
                double selected_score = best;
                for (int sign : {-1, 1}) {
                    Pose trial = pose;
                    double d = sign * step;
                    if (axis == 0)
                        trial.x += d;
                    if (axis == 1)
                        trial.y += d;
                    if (axis == 2)
                        trial.theta += d / model.data->radius;
                    if (axis == 3)
                        trial.scale *= std::exp(d / model.data->radius);
                    // Local diagnosis, not an unconstrained new global detector.
                    if (std::hypot(trial.x - seed.x, trial.y - seed.y) > 2 ||
                        std::abs(trial.theta - seed.theta) > 0.05 ||
                        std::abs(std::log(trial.scale / seed.scale)) > 0.05 ||
                        !WithinBounds(model, trial, 0.05))
                        continue;
                    double value = Score(features, g, trial);
                    if (value > selected_score + 1e-12) {
                        selected = trial;
                        selected_score = value;
                        improved = true;
                    }
                }
                pose = selected;
                best = selected_score;
            }
            if (!improved)
                break;
        }
    }
    return pose;
}
inline void Run(const HTuple &result, const HObject &image, const std::string &output) {
    auto pyramid = BuildSearchPyramid(image.GetImage(), 1, 3);
    auto gaussian = GaussianGradients(image.GetImage(), 0.8);
    std::ofstream scores(output + "/orientation_scores.csv");
    scores << "method,index,initial_score,final_score,translation_from_seed\n"
           << std::setprecision(17);
    for (bool gauss : {false, true})
        for (bool matched : {false, true}) {
            std::string name = std::string("orientation_") + (gauss ? "gaussian" : "central") +
                               (matched ? "_matched" : "_original");
            std::ofstream out(output + "/" + name + ".csv");
            out << "index,model,row,column,angle_deg,scale_row,scale_column,score\n"
                << std::setprecision(17);
            size_t index = 0;
            for (auto match : ResolveResult(result)->matches) {
                Features features = match.model.data->dense;
                if (matched) {
                    const auto &data = *match.model.data;
                    auto g = gauss ? GaussianGradients(data.image, 0.8)
                                   : BuildSearchPyramid(data.image, 1, 3)[0].gradient;
                    // Keep point coordinates fixed; match template/scene direction sampling.
                    for (auto &f : features) {
                        Vec v = profile_probe::Sample(g, f.p + data.origin);
                        double length = std::hypot(v.x, v.y);
                        if (length >= 3)
                            f.n = v * (1 / length);
                    }
                }
                const auto &g = gauss ? gaussian : pyramid[0].gradient;
                Pose seed = match.pose;
                double initial = Score(features, g, seed);
                match.pose = Refine(match.model, features, g, seed);
                double score = Score(features, g, match.pose);
                auto h = ResultTransform(match);
                out << index << ',' << Str(match.model.params, "model_identifier") << ',' << h[2]
                    << ',' << h[5] << ',' << -match.pose.theta * 180 / 3.14159265358979323846 << ','
                    << match.pose.scale << ',' << match.pose.scale << ',' << score << '\n';
                scores << name << ',' << index++ << ',' << initial << ',' << score << ','
                       << std::hypot(match.pose.x - seed.x, match.pose.y - seed.y) << '\n';
            }
            out.flush();
            if (!out)
                throw std::runtime_error("Cannot write orientation results");
        }
    scores.flush();
    if (!scores)
        throw std::runtime_error("Cannot write orientation scores");
}
} // namespace orientation_probe
