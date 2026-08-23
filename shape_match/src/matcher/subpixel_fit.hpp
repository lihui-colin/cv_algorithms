#pragma once

#include <algorithm>
#include <cmath>

namespace openshape::detail {

struct SubpixelOffset {
  double x = 0.0;
  double y = 0.0;
};

// Fit a quadratic surface to a symmetric 3x3 score stencil and return its
// stationary point in stencil coordinates. The fallback handles flat or
// saddle-shaped stencils without quantizing the result back to an integer.
inline SubpixelOffset fit_quadratic_peak(const double score[3][3]) {
  const double center = score[1][1];
  const double a = 0.5 * (score[1][0] + score[1][2] - 2.0 * center);
  const double b = 0.5 * (score[0][1] + score[2][1] - 2.0 * center);
  const double c = 0.25 * (score[2][2] - score[0][2] - score[2][0] + score[0][0]);
  const double d = 0.5 * (score[1][2] - score[1][0]);
  const double e = 0.5 * (score[2][1] - score[0][1]);
  const double determinant = 4.0 * a * b - c * c;
  SubpixelOffset offset;
  if (determinant > 1e-12 && a < 0.0 && b < 0.0) {
    offset.x = (c * e - 2.0 * b * d) / determinant;
    offset.y = (c * d - 2.0 * a * e) / determinant;
  } else {
    const double denom_x = score[1][0] - 2.0 * center + score[1][2];
    const double denom_y = score[0][1] - 2.0 * center + score[2][1];
    if (std::abs(denom_x) > 1e-12)
      offset.x = 0.5 * (score[1][0] - score[1][2]) / denom_x;
    if (std::abs(denom_y) > 1e-12)
      offset.y = 0.5 * (score[0][1] - score[2][1]) / denom_y;
  }
  if (!std::isfinite(offset.x)) offset.x = 0.0;
  if (!std::isfinite(offset.y)) offset.y = 0.0;
  offset.x = std::clamp(offset.x, -0.5, 0.5);
  offset.y = std::clamp(offset.y, -0.5, 0.5);
  return offset;
}

}  // namespace openshape::detail
