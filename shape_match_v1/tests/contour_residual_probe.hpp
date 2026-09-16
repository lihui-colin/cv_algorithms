#pragma once
#include "normal_profile_probe.hpp"
#include <numeric>
#include <sstream>

namespace contour_residual_probe {
using namespace shape_match;
using namespace shape_match::detail;

inline std::vector<size_t> Components(const Features &features) {
    std::vector<size_t> parent(features.size());
    std::iota(parent.begin(), parent.end(), size_t(0));
    auto root = [&](size_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (size_t i = 0; i < features.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (Dot(features[i].p - features[j].p, features[i].p - features[j].p) < 3.3)
                parent[root(i)] = root(j);
    for (size_t i = 0; i < features.size(); ++i)
        parent[i] = root(i);
    return parent;
}

inline void Run(const HTuple &result, const HObject &image, const std::string &output,
                const std::string &reference = "") {
    struct Reference {
        std::string name;
        double row, column, angle, scale;
    };
    std::vector<Reference> references;
    if (!reference.empty()) {
        std::ifstream in(reference);
        if (!in)
            throw std::runtime_error("Cannot read reference for residual evaluation");
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            Reference r;
            int index;
            double scale_column, score;
            if (!(row >> index >> r.name >> r.row >> r.column >> r.angle >> r.scale >>
                  scale_column >> score) ||
                !std::isfinite(r.row + r.column + r.angle + r.scale) || r.scale <= 0 ||
                r.scale != scale_column)
                throw std::runtime_error("Invalid residual reference row");
            references.push_back(r);
        }
    }
    auto pyramid = BuildSearchPyramid(image.GetImage(), 1, 3);
    const auto &field = pyramid[0].field;
    auto gaussian = GaussianGradients(image.GetImage(), 0.8);
    std::ofstream out(output + (reference.empty() ? "/contour_residuals.csv"
                                                  : "/contour_residuals_at_halcon.csv"));
    out << "prediction_index,model,method,component,point,model_x,model_y,valid,residual,j_tx,j_ty,"
           "j_theta,j_logscale,scene_x,scene_y\n"
        << std::setprecision(17);
    size_t index = 0;
    for (auto match : ResolveResult(result)->matches) {
        if (!reference.empty()) {
            auto h = ResultTransform(match);
            int found = 0;
            for (const auto &r : references)
                if (r.name == Str(match.model.params, "model_identifier") &&
                    std::hypot(h[2] - r.row, h[5] - r.column) < 3) {
                    ++found;
                    double theta = -r.angle * 3.14159265358979323846 / 180;
                    double c = r.scale * std::cos(theta), s = r.scale * std::sin(theta);
                    match.pose = {r.column - .5 + .5 * (c - s), r.row - .5 + .5 * (c + s), theta,
                                  r.scale};
                }
            if (found != 1)
                throw std::runtime_error("Ambiguous residual reference association");
            // Evaluation only: do not optimize from or return this reference pose.
        }
        const auto &features = match.model.data->dense;
        auto components = Components(features);
        for (const auto &method : {std::string("nearest_1.5"), std::string("profile_gaussian_2")})
            for (size_t i = 0; i < features.size(); ++i) {
                const auto &f = features[i];
                Vec a = Rotate(f.p, match.pose.theta) * match.pose.scale;
                Vec p = a + Vec{match.pose.x, match.pose.y}, normal = Rotate(f.n, match.pose.theta);
                std::vector<Correspondence> pairs;
                if (method == "profile_gaussian_2")
                    pairs =
                        profile_probe::Pairs(Features{f}, gaussian, match.pose, {2, true, false});
                else {
                    int x = int(std::lround(p.x)), y = int(std::lround(p.y));
                    double best = 2.25;
                    int id = -1;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int xx = x + dx, yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= field.width || yy >= field.height)
                                continue;
                            int candidate = field.nearest[size_t(yy) * field.width + xx];
                            if (candidate < 0)
                                continue;
                            const auto &e = field.edges[size_t(candidate)];
                            double cost = Dot(p - e.p, p - e.p);
                            if (Dot(normal, e.n) >= 0.65 && cost < best) {
                                best = cost;
                                id = candidate;
                            }
                        }
                    if (id >= 0)
                        pairs.push_back(
                            {f.p, field.edges[size_t(id)].p, field.edges[size_t(id)].n});
                }
                out << index << ',' << Str(match.model.params, "model_identifier") << ',' << method
                    << ',' << components[i] << ',' << i << ',' << f.p.x << ',' << f.p.y << ','
                    << !pairs.empty();
                if (!pairs.empty()) {
                    const auto &pair = pairs[0];
                    out << ',' << Dot(pair.normal, p - pair.image) << ',' << pair.normal.x << ','
                        << pair.normal.y << ',' << -pair.normal.x * a.y + pair.normal.y * a.x << ','
                        << Dot(pair.normal, a) << ',' << pair.image.x << ',' << pair.image.y;
                } else
                    out << ",,,,,,,";
                out << '\n';
            }
        ++index;
    }
    out.flush();
    if (!out)
        throw std::runtime_error("Cannot write contour residuals");
}
} // namespace contour_residual_probe
