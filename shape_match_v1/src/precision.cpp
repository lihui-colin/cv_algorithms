#include "internal.hpp"

namespace shape_match::detail {
GradientField GaussianGradients(const Image &image, double sigma) {
    Require(std::isfinite(sigma) && sigma >= 0.3 && sigma <= 3, ErrorCode::Value,
            "Gaussian derivative sigma must be 0.3..3 pixels");
    int radius = int(std::ceil(3 * sigma));
    std::vector<double> smooth(size_t(2 * radius + 1)), derivative(smooth.size());
    double sum = 0, moment = 0;
    for (int k = -radius; k <= radius; ++k) {
        double v = std::exp(-0.5 * Sq(k / sigma));
        smooth[size_t(k + radius)] = v;
        derivative[size_t(k + radius)] = k * v;
        sum += v;
        moment += k * k * v;
    }
    for (size_t i = 0; i < smooth.size(); ++i) {
        smooth[i] /= sum;
        derivative[i] /= moment; // Unit response to a linear ramp, zero to a constant.
    }
    std::vector<float> horizontal(image.pixels.size()), dx(image.pixels.size());
    GradientField out;
    out.width = image.width;
    out.height = image.height;
    out.gx.resize(image.pixels.size());
    out.gy.resize(image.pixels.size());
    out.mag.resize(image.pixels.size());
    ParallelFor(size_t(image.height), [&](size_t begin, size_t end, size_t) {
        for (int y = int(begin); y < int(end); ++y)
            for (int x = 0; x < image.width; ++x) {
                double s = 0, d = 0;
                for (int k = -radius; k <= radius; ++k) {
                    double v = image(y, std::clamp(x + k, 0, image.width - 1));
                    s += smooth[size_t(k + radius)] * v;
                    d += derivative[size_t(k + radius)] * v;
                }
                horizontal[size_t(y) * image.width + x] = float(s);
                dx[size_t(y) * image.width + x] = float(d);
            }
    });
    ParallelFor(size_t(image.height), [&](size_t begin, size_t end, size_t) {
        for (int y = int(begin); y < int(end); ++y)
            for (int x = 0; x < image.width; ++x) {
                double gx = 0, gy = 0;
                for (int k = -radius; k <= radius; ++k) {
                    size_t at = size_t(std::clamp(y + k, 0, image.height - 1)) * image.width + x;
                    gx += smooth[size_t(k + radius)] * dx[at];
                    gy += derivative[size_t(k + radius)] * horizontal[at];
                }
                size_t at = size_t(y) * image.width + x;
                out.gx[at] = float(gx);
                out.gy[at] = float(gy);
                out.mag[at] = float(std::hypot(gx, gy));
            }
    });
    return out;
}
namespace {
double Cubic(double a, double b, double c, double d, double t) {
    return b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
}

// The caller guarantees a complete 4x4 stencil. Never invent edges by clamping
// an out-of-image query to a border pixel.
double Sample(const std::vector<float> &values, int width, Vec p) {
    int x = int(std::floor(p.x)), y = int(std::floor(p.y));
    double rows[4];
    for (int j = -1; j <= 2; ++j) {
        const float *row = values.data() + size_t(y + j) * width + x;
        rows[j + 1] = Cubic(row[-1], row[0], row[1], row[2], p.x - x);
    }
    return Cubic(rows[0], rows[1], rows[2], rows[3], p.y - y);
}

double Agreement(Vec a, Vec b, const std::string &metric, double polarity) {
    double dot = Dot(a, b);
    return metric == "ignore_local_polarity" ? std::abs(dot) : polarity * dot;
}

struct ProfileCache {
    int x = -1, y = -1;
    double samples[4][4]{};
    bool logarithmic = false;
};
double LogProfileSample(const GradientField &g, Vec p, Vec normal, const std::string &metric,
                        double polarity, ProfileCache &cache) {
    int x = int(std::floor(p.x)), y = int(std::floor(p.y));
    if (cache.x != x || cache.y != y) {
        cache.x = x;
        cache.y = y;
        double minimum = 1e30, maximum = 0;
        for (int j = -1; j <= 2; ++j)
            for (int i = -1; i <= 2; ++i) {
                size_t at = size_t(y + j) * g.width + x + i;
                double response = Agreement(normal, {g.gx[at], g.gy[at]}, metric, polarity);
                cache.samples[j + 1][i + 1] = response;
                minimum = std::min(minimum, response);
                maximum = std::max(maximum, response);
            }
        cache.logarithmic = minimum >= std::max(1e-5, 0.01 * maximum);
        if (cache.logarithmic)
            for (auto &row : cache.samples)
                for (double &value : row)
                    value = std::log(value);
    }
    // Do not interpolate a log floor at junctions or polarity changes.
    if (!cache.logarithmic)
        return std::log(
            std::max(1e-5, Agreement(normal, {Sample(g.gx, g.width, p), Sample(g.gy, g.width, p)},
                                     metric, polarity)));
    double rows[4];
    for (int j = -1; j <= 2; ++j) {
        const auto &values = cache.samples[j + 1];
        rows[j + 1] = Cubic(values[0], values[1], values[2], values[3], p.x - x);
    }
    return Cubic(rows[0], rows[1], rows[2], rows[3], p.y - y);
}

bool Profile(const GradientField &g, Vec p, Vec normal, double radius, const std::string &metric,
             double polarity, Vec &point, Vec &direction) {
    if (p.x - radius < 1 || p.y - radius < 1 || p.x + radius >= g.width - 2 ||
        p.y + radius >= g.height - 2)
        return false;
    auto gradient = [&](Vec q) {
        if (q.x < 1 || q.y < 1 || q.x >= g.width - 2 || q.y >= g.height - 2)
            return Vec{};
        return Vec{Sample(g.gx, g.width, q), Sample(g.gy, g.width, q)};
    };
    // A Gaussian-blurred step has a quadratic log-gradient profile. Interpolating
    // raw gradient amplitudes introduces a substantial fractional-pixel peak bias.
    ProfileCache cache;
    auto value = [&](double t) {
        Vec q = p + normal * t;
        if (q.x < 1 || q.y < 1 || q.x >= g.width - 2 || q.y >= g.height - 2)
            return std::log(1e-5);
        return LogProfileSample(g, q, normal, metric, polarity, cache);
    };
    int steps = int(std::ceil(radius / 0.25)), peak = -steps;
    double spacing = radius / steps, best = -1e30;
    for (int i = -steps; i <= steps; ++i) {
        double response = value(i * spacing);
        if (response > best) {
            best = response;
            peak = i;
        }
    }
    // A boundary maximum is not a localized edge. Thresholding the response is
    // done against the model's min_contrast by the caller's nearest-edge gate.
    if (std::abs(peak) == steps || best <= std::log(1e-5))
        return false;
    double t = peak * spacing, left = value(t - spacing), right = value(t + spacing);
    double curvature = left - 2 * best + right;
    if (curvature >= -1e-8)
        return false;
    t += std::clamp(0.5 * (left - right) / curvature, -1.0, 1.0) * spacing;
    point = p + normal * t;
    direction = gradient(point);
    double norm = std::hypot(direction.x, direction.y);
    if (norm < 1e-5)
        return false;
    direction = direction * (1 / norm);
    return Agreement(normal, direction, metric, polarity) >= 0.65;
}
} // namespace

Features GaussianModelFeatures(const Image &image, double low, double high, int min_size) {
    auto g = GaussianGradients(image);
    auto features = ExtractSubpixelContours(image, g, low, high, min_size);
    for (auto &f : features) {
        Vec point, normal;
        if (Profile(g, f.p, f.n, 0.75, "use_polarity", 1, point, normal)) {
            f.p = point;
            f.n = normal;
        }
    }
    return features;
}

std::vector<Correspondence> ContinuousCorrespondences(const Features &features,
                                                      const PyramidLevel &image, const Pose &pose,
                                                      const std::string &metric, double radius,
                                                      const std::string &method) {
    const auto &field = image.field;
    double polarity = 1;
    if (metric == "ignore_global_polarity") {
        double sum = 0;
        for (const auto &f : features) {
            Vec p = Rotate(f.p, pose.theta) * pose.scale + Vec{pose.x, pose.y};
            int x = int(std::lround(p.x)), y = int(std::lround(p.y));
            if (x < 0 || y < 0 || x >= field.width || y >= field.height)
                continue;
            int id = field.nearest[size_t(y) * field.width + x];
            if (id >= 0)
                sum += Dot(Rotate(f.n, pose.theta), field.edges[size_t(id)].n);
        }
        polarity = sum < 0 ? -1 : 1;
    }
    std::vector<Correspondence> out;
    out.reserve(features.size());
    for (const auto &f : features) {
        Vec p = Rotate(f.p, pose.theta) * pose.scale + Vec{pose.x, pose.y};
        Vec n = Rotate(f.n, pose.theta), best_point, best_normal;
        int x = int(std::lround(p.x)), y = int(std::lround(p.y));
        std::array<int, 25> ids{};
        size_t count = 0;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= field.width || yy >= field.height)
                    continue;
                int id = field.nearest[size_t(yy) * field.width + xx];
                if (id >= 0 &&
                    std::find(ids.begin(), ids.begin() + count, id) == ids.begin() + count)
                    ids[count++] = id;
            }
        double best = radius * radius;
        bool found = false;
        for (size_t i = 0; i < count; ++i) {
            const auto &a = field.edges[size_t(ids[i])];
            auto consider = [&](Vec q, Vec normal) {
                double cost = Dot(p - q, p - q);
                if (cost < best && Agreement(n, normal, metric, polarity) >= 0.65) {
                    best = cost;
                    best_point = q;
                    best_normal = normal;
                    found = true;
                }
            };
            consider(a.p, a.n); // Isolated points/endpoints retain a bounded fallback.
            if (method != "contour")
                continue;
            for (size_t j = i + 1; j < count; ++j) {
                const auto &b = field.edges[size_t(ids[j])];
                Vec d = b.p - a.p;
                double length2 = Dot(d, d);
                // Only join short, locally tangent, consistently oriented edge pieces.
                if (length2 < 0.1 || length2 > 5 || Dot(a.n, b.n) < 0.9 ||
                    Sq(Dot(d, a.n)) > 0.2 * length2 || Sq(Dot(d, b.n)) > 0.2 * length2)
                    continue;
                double t = std::clamp(Dot(p - a.p, d) / length2, 0.0, 1.0);
                Vec normal = a.n * (1 - t) + b.n * t;
                normal = normal * (1 / std::hypot(normal.x, normal.y));
                consider(a.p + d * t, normal);
            }
        }
        if (!found)
            continue;
        if (method == "gradient" || method == "gradient_gaussian") {
            Vec point, normal;
            Require(method != "gradient_gaussian" || bool(image.precise_gradient),
                    ErrorCode::Geometry, "Gaussian refinement gradient was not prepared");
            const auto &gradient =
                method == "gradient_gaussian" ? *image.precise_gradient : image.gradient;
            if (!Profile(gradient, p, n, radius, metric, polarity, point, normal) ||
                Dot(point - best_point, point - best_point) > 2.25)
                continue;
            best_point = point;
            best_normal = normal;
        }
        out.push_back({f.p, best_point, best_normal});
    }
    return out;
}

double RefinementSupportLoss(const std::vector<Correspondence> &pairs, const Pose &pose,
                             size_t model_points, double radius) {
    Require(pairs.size() <= model_points, ErrorCode::Geometry, "Invalid refinement support");
    double cap = radius * radius, loss = (model_points - pairs.size()) * cap;
    for (const auto &pair : pairs) {
        Vec p = Rotate(pair.model, pose.theta) * pose.scale + Vec{pose.x, pose.y};
        loss += std::min(cap, Sq(Dot(pair.normal, p - pair.image)));
    }
    return loss;
}
} // namespace shape_match::detail
