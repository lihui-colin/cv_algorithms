#include "internal.hpp"
#include <limits>
#include <queue>
#include <unordered_map>

namespace shape_match::detail {
static float SampleArray(const std::vector<float> &a, int w, int h, double x, double y) {
    x = std::clamp(x, 0.0, double(w - 1));
    y = std::clamp(y, 0.0, double(h - 1));
    int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    double dx = x - x0, dy = y - y0;
    return float((1 - dy) * ((1 - dx) * a[size_t(y0) * w + x0] + dx * a[size_t(y0) * w + x1]) +
                 dy * ((1 - dx) * a[size_t(y1) * w + x0] + dx * a[size_t(y1) * w + x1]));
}
float Bilinear(const Image &im, double x, double y) {
    return SampleArray(im.pixels, im.width, im.height, x, y);
}
static double Cubic(double a, double b, double c, double d, double t) {
    return b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
}
float CubicSample(const Image &im, double x, double y) {
    int ix = int(std::floor(x)), iy = int(std::floor(y));
    double row[4];
    for (int j = -1; j <= 2; ++j) {
        double v[4];
        for (int i = -1; i <= 2; ++i)
            v[i + 1] =
                im(std::clamp(iy + j, 0, im.height - 1), std::clamp(ix + i, 0, im.width - 1));
        row[j + 1] = Cubic(v[0], v[1], v[2], v[3], x - ix);
    }
    return float(Cubic(row[0], row[1], row[2], row[3], y - iy));
}
Image Smooth(const Image &im, double sigma) {
    int radius = std::max(1, int(std::ceil(3 * sigma)));
    std::vector<double> kernel(2 * radius + 1);
    double sum = 0;
    for (int i = -radius; i <= radius; ++i)
        sum += (kernel[i + radius] = std::exp(-0.5 * Sq(i / sigma)));
    for (auto &v : kernel)
        v /= sum;
    Image tmp = im, out = im;
    // Domain is metadata: filtering accesses original neighboring pixels; it never pads
    // transparent pixels with black. Edge extraction later checks the valid support.
    for (int y = 0; y < im.height; ++y)
        for (int x = 0; x < im.width; ++x) {
            double v = 0;
            for (int k = -radius; k <= radius; ++k)
                v += kernel[k + radius] * im(y, std::clamp(x + k, 0, im.width - 1));
            tmp.pixels[size_t(y) * im.width + x] = float(v);
        }
    for (int y = 0; y < im.height; ++y)
        for (int x = 0; x < im.width; ++x) {
            double v = 0;
            for (int k = -radius; k <= radius; ++k)
                v += kernel[k + radius] * tmp(std::clamp(y + k, 0, im.height - 1), x);
            out.pixels[size_t(y) * im.width + x] = float(v);
        }
    return out;
}
Image Downsample(const Image &im) {
    auto filtered = Smooth(im, 1.0);
    Image out;
    out.width = (im.width + 1) / 2;
    out.height = (im.height + 1) / 2;
    out.pixels.resize(size_t(out.width) * out.height);
    out.domain.resize(out.pixels.size());
    // Explicit sampling map: level point (x,y) corresponds to previous (2*x,2*y).
    for (int y = 0; y < out.height; ++y)
        for (int x = 0; x < out.width; ++x) {
            size_t k = size_t(y) * out.width + x, old = size_t(2 * y) * im.width + 2 * x;
            out.pixels[k] = filtered.pixels[old];
            out.domain[k] = im.domain[old];
        }
    return out;
}
GradientField ComputeGradients(const Image &im) {
    GradientField out;
    out.width = im.width;
    out.height = im.height;
    out.gx.resize(im.pixels.size());
    out.gy.resize(im.pixels.size());
    out.mag.resize(im.pixels.size());
    for (int y = 1; y < im.height - 1; ++y)
        for (int x = 1; x < im.width - 1; ++x) {
            size_t i = size_t(y) * im.width + x;
            float gx = (im(y, x + 1) - im(y, x - 1)) * 0.5f,
                  gy = (im(y + 1, x) - im(y - 1, x)) * 0.5f;
            out.gx[i] = gx;
            out.gy[i] = gy;
            out.mag[i] = std::hypot(gx, gy);
        }
    return out;
}
ContourSet TraceContours(Features &features) {
    // Grid adjacency only builds display polylines/component labels; optimization uses
    // independent points and does not depend on this ordering.
    std::map<std::pair<int, int>, std::vector<int>> cells;
    for (size_t i = 0; i < features.size(); ++i)
        cells[{int(std::lround(features[i].p.x)), int(std::lround(features[i].p.y))}].push_back(
            int(i));
    std::vector<std::vector<int>> adjacent(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        int x = int(std::lround(features[i].p.x)), y = int(std::lround(features[i].p.y));
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                auto it = cells.find({x + dx, y + dy});
                if (it == cells.end())
                    continue;
                for (int j : it->second)
                    if (int(i) != j &&
                        Dot(features[i].p - features[j].p, features[i].p - features[j].p) < 3.3)
                        adjacent[i].push_back(j);
            }
    }
    std::vector<char> visited(features.size());
    ContourSet out;
    // Split at branches rather than draw long spurious jumps across separate contours.
    for (size_t seed = 0; seed < features.size(); ++seed)
        if (!visited[seed]) {
            int at = int(seed);
            Contour c;
            while (at >= 0 && !visited[at]) {
                visited[at] = 1;
                auto &f = features[at];
                f.contour = int(out.size());
                c.push_back({f.p.y, f.p.x});
                int next = -1;
                double best = 1e30;
                for (int j : adjacent[at])
                    if (!visited[j]) {
                        double d = Dot(f.p - features[j].p, f.p - features[j].p);
                        if (d < best) {
                            best = d;
                            next = j;
                        }
                    }
                at = next;
            }
            if (c.size() > 2 &&
                std::hypot(c.front().row - c.back().row, c.front().column - c.back().column) < 1.8)
                c.push_back(c.front());
            out.push_back(std::move(c));
        }
    return out;
}
Features ExtractSubpixelContours(const Image &im, const GradientField &g, double low, double high,
                                 int min_size) {
    Features candidates;
    std::vector<int> grid(im.pixels.size(), -1);
    for (int y = 3; y < im.height - 3; ++y)
        for (int x = 3; x < im.width - 3; ++x) {
            size_t i = size_t(y) * im.width + x;
            double m = g.mag[i];
            if (m < std::max(1e-5, low))
                continue;
            bool valid = true;
            for (int dy = -2; dy <= 2 && valid; ++dy)
                for (int dx = -2; dx <= 2; ++dx)
                    if (!im.domain[size_t(y + dy) * im.width + x + dx]) {
                        valid = false;
                        break;
                    }
            if (!valid)
                continue;
            Vec n{g.gx[i] / m, g.gy[i] / m};
            double left = SampleArray(g.mag, g.width, g.height, x - n.x, y - n.y),
                   right = SampleArray(g.mag, g.width, g.height, x + n.x, y + n.y);
            if (m < left || m <= right)
                continue;
            double den = left - 2 * m + right;
            double delta =
                std::abs(den) > 1e-8 ? std::clamp(0.5 * (left - right) / den, -0.5, 0.5) : 0;
            grid[i] = int(candidates.size());
            candidates.push_back({{x + delta * n.x, y + delta * n.y}, n, m, 1, -1});
        }
    // Hysteresis operates on connected weak/strong edge components.
    std::vector<char> seen(candidates.size());
    Features kept;
    for (size_t seed = 0; seed < candidates.size(); ++seed)
        if (!seen[seed]) {
            std::vector<int> component{int(seed)};
            seen[seed] = 1;
            bool strong = false;
            for (size_t k = 0; k < component.size(); ++k) {
                auto &f = candidates[component[k]];
                strong |= f.strength >= high;
                int x = int(std::lround(f.p.x)), y = int(std::lround(f.p.y));
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= im.width || yy >= im.height)
                            continue;
                        int j = grid[size_t(yy) * im.width + xx];
                        if (j >= 0 && !seen[j]) {
                            seen[j] = 1;
                            component.push_back(j);
                        }
                    }
            }
            if (strong && int(component.size()) >= min_size)
                for (int j : component)
                    kept.push_back(candidates[j]);
        }
    return kept;
}
Features SelectModelFeatures(const Features &f, size_t maximum) {
    if (f.size() <= maximum)
        return f;
    Features out;
    out.reserve(maximum);
    // Farthest-point spatial sampling keeps short distinctive pieces and both holes
    // and outer edges; selected weights remain uniform for reproducible scoring.
    std::vector<double> distance(f.size(), 1e30);
    size_t index = 0;
    for (size_t k = 0; k < maximum; ++k) {
        out.push_back(f[index]);
        size_t next = 0;
        double best = -1;
        for (size_t i = 0; i < f.size(); ++i) {
            Vec d = f[i].p - f[index].p;
            distance[i] = std::min(distance[i], Dot(d, d));
            if (distance[i] > best) {
                best = distance[i];
                next = i;
            }
        }
        index = next;
    }
    return out;
}
EdgeField BuildEdgeField(const Image &im, const GradientField &g, double contrast) {
    EdgeField field;
    field.width = im.width;
    field.height = im.height;
    field.edges = ExtractSubpixelContours(im, g, contrast, contrast, 1);
    field.nearest.assign(im.pixels.size(), -1);
    for (size_t i = 0; i < field.edges.size(); ++i) {
        auto p = field.edges[i].p;
        int x = int(std::lround(p.x)), y = int(std::lround(p.y));
        field.nearest[size_t(y) * im.width + x] = int(i);
    }
    // Four directional sweeps propagate nearest edge labels with exact Euclidean
    // comparisons. This is a fast approximate Voronoi map, not an exact EDT.
    auto relax = [&](int x, int y, int nx, int ny) {
        if (nx < 0 || ny < 0 || nx >= im.width || ny >= im.height)
            return;
        int j = field.nearest[size_t(ny) * im.width + nx];
        if (j < 0)
            return;
        auto &at = field.nearest[size_t(y) * im.width + x];
        Vec p{double(x), double(y)};
        if (at < 0 || Dot(field.edges[j].p - p, field.edges[j].p - p) <
                          Dot(field.edges[at].p - p, field.edges[at].p - p))
            at = j;
    };
    for (int pass = 0; pass < 2; ++pass) {
        for (int y = 0; y < im.height; ++y)
            for (int x = 0; x < im.width; ++x) {
                relax(x, y, x - 1, y);
                relax(x, y, x - 1, y - 1);
                relax(x, y, x, y - 1);
                relax(x, y, x + 1, y - 1);
            }
        for (int y = im.height - 1; y >= 0; --y)
            for (int x = im.width - 1; x >= 0; --x) {
                relax(x, y, x + 1, y);
                relax(x, y, x + 1, y + 1);
                relax(x, y, x, y + 1);
                relax(x, y, x - 1, y + 1);
            }
    }
    return field;
}
static double Cross(Vec a, Vec b) {
    return a.x * b.y - a.y * b.x;
}
std::array<Vec, 4> MinimumRectangle(const Features &features) {
    std::vector<Vec> p;
    for (const auto &f : features)
        p.push_back(f.p);
    std::sort(p.begin(), p.end(), [](Vec a, Vec b) { return a.x == b.x ? a.y < b.y : a.x < b.x; });
    std::vector<Vec> hull;
    for (auto v : p) {
        while (hull.size() > 1 && Cross(hull.back() - hull[hull.size() - 2], v - hull.back()) <= 0)
            hull.pop_back();
        hull.push_back(v);
    }
    size_t lower = hull.size();
    for (int i = int(p.size()) - 2; i >= 0; --i) {
        auto v = p[i];
        while (hull.size() > lower &&
               Cross(hull.back() - hull[hull.size() - 2], v - hull.back()) <= 0)
            hull.pop_back();
        hull.push_back(v);
    }
    if (hull.size() > 1)
        hull.pop_back();
    Require(hull.size() >= 3, ErrorCode::Geometry, "Degenerate model bounding rectangle");
    double best = 1e100;
    std::array<Vec, 4> result{};
    for (size_t i = 0; i < hull.size(); ++i) {
        Vec v = hull[(i + 1) % hull.size()] - hull[i];
        double angle = std::atan2(v.y, v.x), xmin = 1e100, xmax = -1e100, ymin = 1e100,
               ymax = -1e100;
        for (auto h : hull) {
            auto q = Rotate(h, -angle);
            xmin = std::min(xmin, q.x);
            xmax = std::max(xmax, q.x);
            ymin = std::min(ymin, q.y);
            ymax = std::max(ymax, q.y);
        }
        double area = (xmax - xmin) * (ymax - ymin);
        if (area < best) {
            best = area;
            result = {Rotate({xmin, ymin}, angle), Rotate({xmax, ymin}, angle),
                      Rotate({xmax, ymax}, angle), Rotate({xmin, ymax}, angle)};
        }
    }
    return result;
}
static std::vector<Vec> Rectangle(const Match &m) {
    std::vector<Vec> out;
    for (auto p : m.model.data->rectangle)
        out.push_back(Rotate(p, m.pose.theta) * m.pose.scale + Vec{m.pose.x, m.pose.y});
    return out;
}
static double Area(const std::vector<Vec> &p) {
    if (p.size() < 3)
        return 0;
    double a = 0;
    for (size_t i = 0; i < p.size(); ++i)
        a += Cross(p[i], p[(i + 1) % p.size()]);
    return std::abs(a) * 0.5;
}
double RectangleOverlap(const Match &a, const Match &b) {
    auto subject = Rectangle(a), clip = Rectangle(b);
    double denominator = std::min(Area(subject), Area(clip));
    if (denominator <= 0)
        return 0;
    for (size_t k = 0; k < clip.size() && !subject.empty(); ++k) {
        Vec p = clip[k], q = clip[(k + 1) % clip.size()], direction = q - p;
        std::vector<Vec> output;
        for (size_t j = 0; j < subject.size(); ++j) {
            Vec s = subject[j], e = subject[(j + 1) % subject.size()];
            double ds = Cross(direction, s - p), de = Cross(direction, e - p);
            bool ins = ds >= -1e-9, ine = de >= -1e-9;
            if (ins != ine) {
                double t = ds / (ds - de);
                output.push_back(s + (e - s) * t);
            }
            if (ine)
                output.push_back(e);
        }
        subject = std::move(output);
    }
    return std::clamp(Area(subject) / denominator, 0.0, 1.0);
}
SearchPyramid BuildSearchPyramid(const Image &image, int count, double contrast) {
    SearchPyramid out;
    Image current = image;
    // Search domain limits candidate origins, not pixels used to observe contours.
    std::fill(current.domain.begin(), current.domain.end(), 1);
    for (int l = 0; l < count; ++l) {
        Image filtered = Smooth(current, 0.8);
        auto gradient = ComputeGradients(filtered);
        auto field = BuildEdgeField(filtered, gradient, contrast);
        out.push_back({current, std::move(gradient), std::move(field)});
        if (l + 1 < count)
            current = Downsample(current);
    }
    return out;
}
} // namespace shape_match::detail
