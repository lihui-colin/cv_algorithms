#pragma once
#include "shape_match/shape_match.hpp"
#include <algorithm>
#include <cmath>
// Independent continuous geometry renderer; never warps matcher-generated edge maps.
// Pixel integration is approximated by fixed supersampling. Shape has an asymmetric
// outer outline plus an off-center round hole, giving translation/angle/scale constraints.
inline double SignedDistance(double x, double y, double hole_dx = 0, double hole_dy = 0) {
    double bx = std::abs(x) - 15, by = std::abs(y) - 11;
    double box = std::hypot(std::max(bx, 0.0), std::max(by, 0.0)) + std::min(std::max(bx, by), 0.0);
    double tab = std::hypot(x - 13, y + 7) - 6;
    double hole = std::hypot(x + 5 - hole_dx, y - 1 - hole_dy) - 5;
    return std::max(std::min(box, tab), -hole);
}
inline shape_match::HObject Render(int width, int height, double cx, double cy, double angle = 0,
                                   double scale = 1, int samples = 4, double noise = 0,
                                   double blur = 0.9, double hole_dx = 0, double hole_dy = 0) {
    std::vector<float> pixels(size_t(width) * height);
    uint32_t rng = 1731;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double sum = 0;
            for (int j = 0; j < samples; ++j)
                for (int i = 0; i < samples; ++i) {
                    double xx = x + (i + 0.5) / samples - 0.5 - cx,
                           yy = y + (j + 0.5) / samples - 0.5 - cy;
                    double qx = (std::cos(angle) * xx + std::sin(angle) * yy) / scale,
                           qy = (-std::sin(angle) * xx + std::cos(angle) * yy) / scale;
                    // Analytic blurred signed-distance edge; PSF sigma fixed in image pixels.
                    double d = SignedDistance(qx, qy, hole_dx, hole_dy) * scale;
                    sum += 25 + 205 * 0.5 * (1 + std::erf(d / (blur * std::sqrt(2.0))));
                }
            rng = 1664525 * rng + 1013904223;
            double n = noise * (double(rng) / 4294967295.0 - 0.5) * std::sqrt(12.0);
            pixels[size_t(y) * width + x] =
                float(std::clamp(sum / (samples * samples) + n, 0.0, 255.0));
        }
    return shape_match::HObject::FromGray(width, height, pixels);
}
