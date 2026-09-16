#pragma once
#include "../src/internal.hpp"
#include <fstream>
#include <iomanip>

// Diagnostic only: original detections seed every run; no reference poses used.
namespace profile_probe {
using namespace shape_match;
using namespace shape_match::detail;
struct Options {
    double radius;
    bool nearest, model_normal;
};
inline Vec Sample(const GradientField &g, Vec p) {
    int x = int(std::floor(p.x)), y = int(std::floor(p.y));
    if (x < 0 || y < 0 || x + 1 >= g.width || y + 1 >= g.height)
        return {};
    double dx = p.x - x, dy = p.y - y;
    auto get = [&](const std::vector<float> &v) {
        size_t i = size_t(y) * g.width + x;
        return (1 - dy) * ((1 - dx) * v[i] + dx * v[i + 1]) +
               dy * ((1 - dx) * v[i + g.width] + dx * v[i + g.width + 1]);
    };
    return {get(g.gx), get(g.gy)};
}
inline std::vector<Correspondence> Pairs(const Features &features, const GradientField &g,
                                         const Pose &pose, Options option) {
    std::vector<Correspondence> pairs;
    int steps = int(std::ceil(option.radius / 0.25));
    double spacing = option.radius / steps;
    for (const auto &f : features) {
        Vec p = Rotate(f.p, pose.theta) * pose.scale + Vec{pose.x, pose.y},
            n = Rotate(f.n, pose.theta);
        if (p.x - option.radius < 1 || p.y - option.radius < 1 ||
            p.x + option.radius >= g.width - 2 || p.y + option.radius >= g.height - 2)
            continue;
        std::vector<double> values(size_t(2 * steps + 1));
        for (int i = -steps; i <= steps; ++i)
            values[size_t(i + steps)] = Dot(n, Sample(g, p + n * (i * spacing)));
        double best = 1e100, offset = 0;
        bool found = false;
        for (int i = 1; i < 2 * steps; ++i) {
            double l = values[size_t(i - 1)], m = values[size_t(i)], r = values[size_t(i + 1)];
            if (m < 3 || m <= l || m < r || l - 2 * m + r >= -1e-8)
                continue;
            double at =
                (i - steps + std::clamp(0.5 * (l - r) / (l - 2 * m + r), -0.5, 0.5)) * spacing;
            Vec v = Sample(g, p + n * at);
            double length = std::hypot(v.x, v.y);
            if (length < 3 || Dot(n, v) / length < 0.65)
                continue;
            double cost = option.nearest ? std::abs(at) : -m;
            if (cost < best) {
                best = cost;
                offset = at;
                found = true;
            }
        }
        if (!found)
            continue;
        Vec q = p + n * offset, v = Sample(g, q);
        double length = std::hypot(v.x, v.y);
        pairs.push_back({f.p, q, option.model_normal ? n : v * (1 / length)});
    }
    return pairs;
}
inline Pose Refine(const ModelSnapshot &model, const GradientField &g, Pose pose, Options option) {
    const auto &features = model.data->dense;
    double damping = 1e-4;
    for (int iteration = 0; iteration < 30; ++iteration) {
        auto pairs = Pairs(features, g, pose, option);
        if (pairs.size() < 12 || pairs.size() < features.size() / 5)
            break;
        double h[4][5]{}, unit[4]{};
        for (const auto &pair : pairs) {
            Vec a = Rotate(pair.model, pose.theta) * pose.scale;
            double residual = Dot(pair.normal, a + Vec{pose.x, pose.y} - pair.image);
            double j[] = {pair.normal.x, pair.normal.y, -pair.normal.x * a.y + pair.normal.y * a.x,
                          Dot(pair.normal, a)};
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 4; ++k)
                    h[i][k] += j[i] * j[k];
                h[i][4] -= j[i] * residual;
            }
        }
        bool valid = true;
        for (int i = 0; i < 4; ++i) {
            unit[i] = std::sqrt(h[i][i]);
            valid &= unit[i] > 1e-8;
        }
        if (!valid)
            break;
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k)
                h[i][k] /= unit[i] * unit[k];
            h[i][4] /= unit[i];
            h[i][i] += damping;
        }
        for (int k = 0; k < 4; ++k) {
            int pivot = k;
            for (int i = k + 1; i < 4; ++i)
                if (std::abs(h[i][k]) > std::abs(h[pivot][k]))
                    pivot = i;
            if (std::abs(h[pivot][k]) < 1e-12) {
                valid = false;
                break;
            }
            for (int j = k; j < 5; ++j)
                std::swap(h[pivot][j], h[k][j]);
            double d = h[k][k];
            for (int j = k; j < 5; ++j)
                h[k][j] /= d;
            for (int i = 0; i < 4; ++i)
                if (i != k) {
                    double a = h[i][k];
                    for (int j = k; j < 5; ++j)
                        h[i][j] -= a * h[k][j];
                }
        }
        if (!valid)
            break;
        double step[4];
        for (int i = 0; i < 4; ++i)
            step[i] = h[i][4] / unit[i];
        double t = std::hypot(step[0], step[1]);
        if (t > 0.5) {
            step[0] *= 0.5 / t;
            step[1] *= 0.5 / t;
        }
        step[2] = std::clamp(step[2], -0.02, 0.02);
        step[3] = std::clamp(step[3], -0.02, 0.02);
        Pose trial{pose.x + step[0], pose.y + step[1], pose.theta + step[2],
                   pose.scale * std::exp(step[3])};
        auto next = Pairs(features, g, trial, option);
        if (WithinBounds(model, trial, 0.05) &&
            RefinementSupportLoss(next, trial, features.size(), option.radius) <=
                RefinementSupportLoss(pairs, pose, features.size(), option.radius)) {
            pose = trial;
            damping = std::max(1e-8, damping / 3);
            if (t < 1e-6 && std::abs(step[2]) + std::abs(step[3]) < 1e-7)
                break;
        } else {
            damping *= 10;
            if (damping > 1e6)
                break;
        }
    }
    return pose;
}
inline void Run(const HTuple &result, const HObject &image, const std::string &output) {
    auto central = BuildSearchPyramid(image.GetImage(), 1, 3)[0];
    auto gaussian = GaussianGradients(image.GetImage(), 0.8);
    for (bool symmetric : {false, true})
        for (bool gauss : {false, true})
            for (double radius : {1.0, 2.0})
                for (bool nearest : {false, true})
                    for (bool model_normal : {false, true}) {
                        std::string name = std::string("profile_") +
                                           (gauss ? "gaussian" : "central") + "_r" +
                                           std::to_string(int(radius)) + "_" +
                                           (nearest ? "nearest" : "strongest") + "_" +
                                           (model_normal ? "model" : "scene") +
                                           (symmetric ? "_matched_template" : "");
                        std::ofstream out(output + "/" + name + ".csv");
                        out << "index,model,row,column,angle_deg,scale_row,scale_column,score\n"
                            << std::setprecision(17);
                        size_t index = 0;
                        for (auto match : ResolveResult(result)->matches) {
                            if (symmetric) {
                                auto trained = std::make_shared<TrainedData>(*match.model.data);
                                auto template_gradient =
                                    gauss ? GaussianGradients(trained->image, 0.8)
                                          : BuildSearchPyramid(trained->image, 1, 3)[0].gradient;
                                auto pairs = Pairs(trained->dense, template_gradient,
                                                   {trained->origin.x, trained->origin.y, 0, 1},
                                                   {1, true, false});
                                // Keep the domain-defined origin and feature count fixed.
                                for (auto &f : trained->dense)
                                    for (const auto &pair : pairs)
                                        if (Dot(f.p - pair.model, f.p - pair.model) < 1e-16) {
                                            f.p = pair.image - trained->origin;
                                            f.n = pair.normal;
                                            break;
                                        }
                                match.model.data = trained;
                            }
                            match.pose = Refine(match.model, gauss ? gaussian : central.gradient,
                                                match.pose, {radius, nearest, model_normal});
                            auto h = ResultTransform(match);
                            double score =
                                EvaluatePose(match.model.data->levels[0].features, central.field,
                                             match.pose, "use_polarity", 0.8);
                            out << index++ << ',' << Str(match.model.params, "model_identifier")
                                << ',' << h[2] << ',' << h[5] << ','
                                << -match.pose.theta * 180 / 3.14159265358979323846 << ','
                                << match.pose.scale << ',' << match.pose.scale << ',' << score
                                << '\n';
                        }
                        out.flush();
                        if (!out)
                            throw std::runtime_error("Cannot write profile diagnostic");
                    }
}
} // namespace profile_probe
