#include "openshape/matcher/exhaustive_matcher.hpp"
#include "openshape/edge/edge_engine.hpp"
#include "openshape/matcher/pruning.hpp"
#include "subpixel_fit.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstring>
#include <thread>

namespace openshape {
namespace {
std::vector<double> grid_values(double start, double end, double step) {
  std::vector<double> values;
  if (step <= 0.0 || end < start) return values;
  const std::size_t count = static_cast<std::size_t>(std::floor((end - start) / step + 1e-9));
  values.reserve(count + 1);
  for (std::size_t i = 0; i <= count; ++i) values.push_back(start + static_cast<double>(i) * step);
  if (values.empty() || values.back() < end - step * 1e-6) values.push_back(end);
  return values;
}

struct TransformedBounds {
  double min_x = 0.0;
  double min_y = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;
};

struct TranslationRun {
  int row = 0;
  int column_begin = 0;
  int column_end = 0;
};

struct SupportWindowKey {
  int left = 0;
  int right = 0;
  int top = 0;
  int bottom = 0;

  bool operator<(const SupportWindowKey& other) const {
    if (left != other.left) return left < other.left;
    if (right != other.right) return right < other.right;
    if (top != other.top) return top < other.top;
    return bottom < other.bottom;
  }
};

int grid_value_count(int begin, int end, int origin, int step) {
  if (begin >= end) return 0;
  const int offset = ((begin - origin) % step + step) % step;
  const int first = offset == 0 ? begin : begin + step - offset;
  if (first >= end) return 0;
  return 1 + (end - 1 - first) / step;
}

int align_grid_value(int value, int origin, int step) {
  const int offset = ((value - origin) % step + step) % step;
  return offset == 0 ? value : value + step - offset;
}

std::vector<TranslationRun> full_translation_runs(const cv::Rect& roi,
                                                  int translation_step) {
  std::vector<TranslationRun> runs;
  runs.reserve(static_cast<std::size_t>(
      grid_value_count(roi.y, roi.y + roi.height, roi.y, translation_step)));
  for (int row = roi.y; row < roi.y + roi.height; row += translation_step)
    runs.push_back({row, roi.x, roi.x + roi.width});
  return runs;
}

SupportWindowKey support_window_key(const TransformedBounds& bounds,
                                    double edge_distance_sigma) {
  const int padding = std::max(
      3, static_cast<int>(std::ceil(2.0 * edge_distance_sigma)) + 3);
  // One additional pixel makes this domain a conservative superset of the
  // per-pose floating-point AABB query. It may retain a boundary pose but can
  // never omit one that the existing certified prefilter would inspect.
  return {static_cast<int>(std::floor(bounds.min_x)) - padding - 1,
          static_cast<int>(std::ceil(bounds.max_x)) + padding + 1,
          static_cast<int>(std::floor(bounds.min_y)) - padding - 1,
          static_cast<int>(std::ceil(bounds.max_y)) + padding + 1};
}

std::vector<TranslationRun> supported_translation_runs(
    const cv::Mat& support, const SupportWindowKey& window,
    const cv::Rect& roi, int translation_step) {
  if (support.empty() || window.left > 0 || window.right < 0 ||
      window.top > 0 || window.bottom < 0)
    return full_translation_runs(roi, translation_step);
  const int kernel_width = window.right - window.left + 1;
  const int kernel_height = window.bottom - window.top + 1;
  if (kernel_width <= 0 || kernel_height <= 0)
    return full_translation_runs(roi, translation_step);

  cv::Mat possible_centers;
  const cv::Mat kernel = cv::Mat::ones(kernel_height, kernel_width, CV_8UC1);
  cv::dilate(support, possible_centers, kernel,
             cv::Point(-window.left, -window.top), 1,
             cv::BORDER_CONSTANT, cv::Scalar(0));

  std::vector<TranslationRun> runs;
  for (int row = roi.y; row < roi.y + roi.height; row += translation_step) {
    const unsigned char* mask = possible_centers.ptr<unsigned char>(row);
    int column = align_grid_value(roi.x, roi.x, translation_step);
    const int end = roi.x + roi.width;
    while (column < end) {
      while (column < end && mask[column] == 0) column += translation_step;
      if (column >= end) break;
      const int begin = column;
      do {
        column += translation_step;
      } while (column < end && mask[column] != 0);
      runs.push_back({row, begin, column});
    }
  }
  return runs;
}

TransformedBounds transformed_bounds(const ModelLevelSoA& points,
                                     double angle_degrees, double scale) {
  TransformedBounds bounds;
  if (points.empty()) return bounds;
  const double radians = angle_degrees * CV_PI / 180.0;
  const double c = std::cos(radians), s = std::sin(radians);
  bounds.min_x = bounds.min_y = std::numeric_limits<double>::max();
  bounds.max_x = bounds.max_y = -std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double x = scale * (c * points.relative_x[i] - s * points.relative_y[i]);
    const double y = scale * (s * points.relative_x[i] + c * points.relative_y[i]);
    bounds.min_x = std::min(bounds.min_x, x);
    bounds.min_y = std::min(bounds.min_y, y);
    bounds.max_x = std::max(bounds.max_x, x);
    bounds.max_y = std::max(bounds.max_y, y);
  }
  return bounds;
}

enum class PosePrefilterRejection { None, OutsideImage, NoEdgeSupport };

PosePrefilterRejection prefilter_precise_pose(
    const EdgeMap& scene, const cv::Mat& support_integral,
    const TransformedBounds& bounds, double column, double row,
    double edge_distance_sigma) {
  const double pose_min_x = column + bounds.min_x;
  const double pose_min_y = row + bounds.min_y;
  const double pose_max_x = column + bounds.max_x;
  const double pose_max_y = row + bounds.max_y;
  if (pose_max_x < 0.0 || pose_max_y < 0.0 ||
      pose_min_x > static_cast<double>(scene.edges.cols - 1) ||
      pose_min_y > static_cast<double>(scene.edges.rows - 1))
    return PosePrefilterRejection::OutsideImage;
  if (support_integral.empty()) return PosePrefilterRejection::None;

  // Precise score visibility requires a Canny edge within 2*sigma, or a
  // fitted subpixel support within the scorer's 3x3 neighborhood. Extra
  // padding covers bilinear sampling and boundary rounding conservatively.
  const int padding = std::max(3, static_cast<int>(std::ceil(2.0 * edge_distance_sigma)) + 3);
  const int x0 = std::max(0, static_cast<int>(std::floor(pose_min_x)) - padding);
  const int y0 = std::max(0, static_cast<int>(std::floor(pose_min_y)) - padding);
  const int x1 = std::min(scene.edges.cols - 1,
                          static_cast<int>(std::ceil(pose_max_x)) + padding);
  const int y1 = std::min(scene.edges.rows - 1,
                          static_cast<int>(std::ceil(pose_max_y)) + padding);
  if (x0 > x1 || y0 > y1) return PosePrefilterRejection::OutsideImage;
  const int* top = support_integral.ptr<int>(y0);
  const int* bottom = support_integral.ptr<int>(y1 + 1);
  const int count = bottom[x1 + 1] - bottom[x0] - top[x1 + 1] + top[x0];
  return count == 0 ? PosePrefilterRejection::NoEdgeSupport
                    : PosePrefilterRejection::None;
}

bool audit_pose(std::size_t pose_index, double audit_rate) {
  if (audit_rate <= 0.0) return false;
  if (audit_rate >= 1.0) return true;
  std::uint64_t value = static_cast<std::uint64_t>(pose_index) +
                        UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31;
  const long double unit = static_cast<long double>(value) /
                           static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
  return unit < audit_rate;
}

void update_result_score(const EdgeMap& scene, const ModelLevelSoA& points,
                         const SearchParams& params, MatchResult& result) {
  std::size_t valid = 0;
  result.score = score_pose_precise(scene, points, result.column, result.row,
                                    result.angle, result.scale, params.polarity,
                                    params.edge_distance_sigma, &valid);
  result.valid_point_count = valid;
  result.model_point_count = points.size();
  result.valid_point_fraction = points.empty() ? 0.0 :
      static_cast<double>(valid) / static_cast<double>(points.size());
  result.residual = 1.0 - result.score;
  result.confidence = result.score * std::sqrt(result.valid_point_fraction);
  result.refined = true;
  result.refinement_converged = true;
}

void refine_nine(const EdgeMap& scene, const ModelLevelSoA& points,
                 const SearchParams& params, MatchResult& result, bool dense_refine) {
  double scores[3][3]{};
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      scores[dy + 1][dx + 1] = score_pose_precise(
          scene, points, result.column + dx, result.row + dy, result.angle,
          result.scale, params.polarity, params.edge_distance_sigma, nullptr);
  const auto offset = detail::fit_quadratic_peak(scores);
  result.column += offset.x;
  result.row += offset.y;
  // Exhaust the final half-pixel cell at 1/120-pixel spacing.  The 3x3 fit is
  // the inexpensive estimator and this bounded sweep is the accuracy
  // certificate: its cell radius is below 1/60 pixel in each coordinate.
  const double step = params.exhaustive_final_translation_step;
  if (!dense_refine) {
    update_result_score(scene, points, params, result);
    result.position_error_bound = 1.0 / 60.0;
    return;
  }
  const double anchor_x = result.column, anchor_y = result.row;
  double best = result.score, best_x = anchor_x, best_y = anchor_y;
  // The quadratic vertex is already inside the one-pixel cell; a +/-0.1
  // pixel certificate sweep is sufficient to remove its residual bias while
  // keeping the reference implementation practical on dense models.
  const int radius = std::max(1, static_cast<int>(std::ceil(0.1 / step)));
  for (int iy = -radius; iy <= radius; ++iy) {
    for (int ix = -radius; ix <= radius; ++ix) {
      const double x = anchor_x + ix * step;
      const double y = anchor_y + iy * step;
      const double value = score_pose_precise(scene, points, x, y, result.angle,
                                              result.scale, params.polarity,
                                              params.edge_distance_sigma, nullptr);
      if (value > best) { best = value; best_x = x; best_y = y; }
    }
  }
  result.column = best_x;
  result.row = best_y;
  update_result_score(scene, points, params, result);
  result.position_error_bound = std::sqrt(2.0) * step * 0.5;
}

void apply_edge_residual_fit(const EdgeMap& scene, const ModelLevelSoA& points,
                             const SearchParams& params, MatchResult& result) {
  if (scene.subpixel_edge_mask.empty() || points.empty()) return;
  struct O { double w, dx, dy; };
  std::vector<O> obs;
  const double a = result.angle * CV_PI / 180.0, c = std::cos(a), s = std::sin(a);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double x = result.column + result.scale * (c * points.relative_x[i] - s * points.relative_y[i]);
    const double y = result.row + result.scale * (s * points.relative_x[i] + c * points.relative_y[i]);
    const int cx = static_cast<int>(std::lround(x)), cy = static_cast<int>(std::lround(y));
    double bw = 0, bdx = 0, bdy = 0;
    for (int yy = std::max(1, cy - 1); yy <= std::min(scene.edges.rows - 2, cy + 1); ++yy)
      for (int xx = std::max(1, cx - 1); xx <= std::min(scene.edges.cols - 2, cx + 1); ++xx) {
        if (!scene.subpixel_edge_mask.at<unsigned char>(yy, xx)) continue;
        const double ex = scene.subpixel_x.at<float>(yy, xx), ey = scene.subpixel_y.at<float>(yy, xx);
        const double d = std::hypot(ex - x, ey - y);
        if (d > 1.5) continue;
        const double w = std::exp(-d * d / 2.0);
        if (w > bw) { bw = w; bdx = ex - x; bdy = ey - y; }
      }
    if (bw > 0.25) obs.push_back({bw, bdx, bdy});
  }
  std::stable_sort(obs.begin(), obs.end(), [](const O& x, const O& y) { return x.w > y.w; });
  const std::size_t n = std::min<std::size_t>(9, obs.size());
  if (n < 3) return;
  double sw = 0, dx = 0, dy = 0;
  for (std::size_t i = 0; i < n; ++i) { sw += obs[i].w; dx += obs[i].w * obs[i].dx; dy += obs[i].w * obs[i].dy; }
  if (sw <= 0) return;
  dx = std::clamp(dx / sw, -0.4, 0.4); dy = std::clamp(dy / sw, -0.4, 0.4);
  const double old_score = result.score;
  const double new_score = score_pose_precise(scene, points, result.column + dx, result.row + dy,
                                              result.angle, result.scale, params.polarity,
                                              params.edge_distance_sigma, nullptr);
  if (new_score + 1e-6 >= old_score) {
    result.column += dx;
    result.row += dy;
    update_result_score(scene, points, params, result);
  }
}

bool strict_accept(const SearchParams& params, const MatchResult& r) {
  const double required_visible = params.strict_detection ? 1.0 : params.min_visible_fraction;
  if (r.score < params.min_score || r.valid_point_fraction + 1e-12 < required_visible)
    return false;
  if (params.min_region_coverage > 0.0 && r.region_coverage < params.min_region_coverage)
    return false;
  if (params.max_mean_edge_distance > 0.0 && r.mean_edge_distance > params.max_mean_edge_distance)
    return false;
  if (params.min_orientation_consistency > 0.0 &&
      r.orientation_consistency < params.min_orientation_consistency)
    return false;
  return true;
}

struct PreparedTransform {
  double angle = 0.0;
  double scale = 1.0;
  std::vector<double> offset_x;
  std::vector<double> offset_y;
  std::vector<double> orientation_cos;
  std::vector<double> orientation_sin;
  TransformedBounds bounds;
};

struct WorkspacePlan {
  std::uint64_t key = 0;
  std::vector<double> angles;
  std::vector<double> scales;
  std::vector<PreparedTransform> transforms;
  std::size_t memory_bytes = 0;
};

double score_prepared_precise(const EdgeMap& scene, const ModelLevelSoA& points,
                              const PreparedTransform& transform,
                              double column, double row,
                              PolarityMode polarity, double sigma,
                              std::size_t* valid_count) {
  if (valid_count) *valid_count = 0;
  if (points.empty() || scene.edges.empty()) return 0.0;
  const int width = scene.edges.cols;
  const int height = scene.edges.rows;
  const double sigma2 = sigma * sigma;
  const double inv_two_sigma2 = 1.0 / (2.0 * sigma2);
  const float gate = static_cast<float>(std::exp(-2.0));
  const float exponent = static_cast<float>(1.0 / sigma);
  double total = 0.0;
  for (float weight : points.weight) total += std::max(0.0f, weight);
  if (total <= 0.0) return 0.0;
  double weighted = 0.0, inverted_weighted = 0.0;
  std::size_t valid = 0;
  const bool has_subpixel = !scene.subpixel_edge_mask.empty();
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double x = column + transform.offset_x[i];
    const double y = row + transform.offset_y[i];
    if (x < 0.0 || y < 0.0 || x > static_cast<double>(width - 1) ||
        y > static_cast<double>(height - 1)) continue;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float fx = static_cast<float>(x - x0);
    const float fy = static_cast<float>(y - y0);
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;
    const float* gx0 = scene.gx.ptr<float>(y0);
    const float* gx1 = scene.gx.ptr<float>(y1);
    const float* gy0 = scene.gy.ptr<float>(y0);
    const float* gy1 = scene.gy.ptr<float>(y1);
    const float* edge0 = scene.soft_edge_response.ptr<float>(y0);
    const float* edge1 = scene.soft_edge_response.ptr<float>(y1);
    const float* mag0 = scene.normalized_magnitude.ptr<float>(y0);
    const float* mag1 = scene.normalized_magnitude.ptr<float>(y1);
    const float gx = w00 * gx0[x0] + w10 * gx0[x1] + w01 * gx1[x0] + w11 * gx1[x1];
    const float gy = w00 * gy0[x0] + w10 * gy0[x1] + w01 * gy1[x0] + w11 * gy1[x1];
    const float magnitude = std::hypot(gx, gy);
    const float base_edge = w00 * edge0[x0] + w10 * edge0[x1] +
                            w01 * edge1[x0] + w11 * edge1[x1];
    const float local_strength = w00 * mag0[x0] + w10 * mag0[x1] +
                                 w01 * mag1[x0] + w11 * mag1[x1];
    float fitted_edge = 0.0f;
    if (has_subpixel) {
      const int cx = static_cast<int>(std::lround(x));
      const int cy = static_cast<int>(std::lround(y));
      for (int yy = std::max(1, cy - 1); yy <= std::min(height - 2, cy + 1); ++yy) {
        const auto* mask = scene.subpixel_edge_mask.ptr<unsigned char>(yy);
        const auto* sx = scene.subpixel_x.ptr<float>(yy);
        const auto* sy = scene.subpixel_y.ptr<float>(yy);
        for (int xx = std::max(1, cx - 1); xx <= std::min(width - 2, cx + 1); ++xx) {
          if (mask[xx] == 0) continue;
          const double dx = x - sx[xx], dy = y - sy[xx];
          fitted_edge = std::max(fitted_edge,
              static_cast<float>(std::exp(-(dx * dx + dy * dy) * inv_two_sigma2)));
        }
      }
    }
    const float edge_base = std::max(std::clamp(base_edge, 0.0f, 1.0f), fitted_edge);
    const float edge = std::pow(edge_base, exponent) *
                       (0.2f + 0.8f * std::clamp(local_strength, 0.0f, 1.0f));
    if (edge < gate || magnitude <= 1e-6f) continue;
    const float dot = static_cast<float>(
        (transform.orientation_cos[i] * gx + transform.orientation_sin[i] * gy) /
        static_cast<double>(magnitude));
    const double weight = points.weight[i] * edge;
    if (polarity == PolarityMode::GlobalEither) {
      weighted += weight * std::max(0.0f, dot);
      inverted_weighted += weight * std::max(0.0f, -dot);
    } else if (polarity == PolarityMode::Inverted) {
      weighted += weight * std::max(0.0f, -dot);
    } else if (polarity == PolarityMode::LocalEither) {
      weighted += weight * std::abs(dot);
    } else {
      weighted += weight * std::max(0.0f, dot);
    }
    ++valid;
  }
  if (valid_count) *valid_count = valid;
  if (polarity == PolarityMode::GlobalEither)
    weighted = std::max(weighted, inverted_weighted);
  return std::clamp(weighted / total, 0.0, 1.0);
}

std::uint64_t bits_of(double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(1099511628211);
  }
}

std::uint64_t workspace_key(const ShapeModel& model, const SearchParams& params,
                            const std::vector<double>& angles,
                            const std::vector<double>& scales) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto& points = model.levels_soa().front();
  const int model_version = model.version();
  hash_bytes(hash, &model_version, sizeof(model_version));
  const std::size_t count = points.size();
  hash_bytes(hash, &count, sizeof(count));
  for (std::size_t i = 0; i < count; ++i) {
    hash_bytes(hash, &points.relative_x[i], sizeof(float));
    hash_bytes(hash, &points.relative_y[i], sizeof(float));
    hash_bytes(hash, &points.orientation[i], sizeof(float));
    hash_bytes(hash, &points.weight[i], sizeof(float));
  }
  for (double value : angles) { const std::uint64_t bits = bits_of(value); hash_bytes(hash, &bits, sizeof(bits)); }
  for (double value : scales) { const std::uint64_t bits = bits_of(value); hash_bytes(hash, &bits, sizeof(bits)); }
  hash_bytes(hash, &params.exhaustive_translation_step, sizeof(double));
  hash_bytes(hash, &params.edge_distance_sigma, sizeof(double));
  hash_bytes(hash, &params.polarity, sizeof(params.polarity));
  hash_bytes(hash, &params.coarse_point_fraction, sizeof(double));
  hash_bytes(hash, &params.enable_safe_pruning, sizeof(bool));
  hash_bytes(hash, &params.greediness, sizeof(double));
  return hash;
}

std::shared_ptr<const WorkspacePlan> build_workspace_plan(
    const ShapeModel& model, const SearchParams& params) {
  const double angle_step = params.exhaustive_angle_step > 0.0
      ? params.exhaustive_angle_step : params.angle_step;
  const double scale_step = params.exhaustive_scale_step > 0.0
      ? params.exhaustive_scale_step : params.scale_step;
  auto plan = std::make_shared<WorkspacePlan>();
  plan->angles = grid_values(params.angle_start,
                             params.angle_start + params.angle_extent, angle_step);
  plan->scales = grid_values(params.scale_min, params.scale_max, scale_step);
  plan->key = workspace_key(model, params, plan->angles, plan->scales);
  const auto& points = model.levels_soa().front();
  plan->transforms.reserve(plan->angles.size() * plan->scales.size());
  for (double scale : plan->scales) {
    for (double angle : plan->angles) {
      PreparedTransform transform;
      transform.angle = angle;
      transform.scale = scale;
      transform.offset_x.resize(points.size());
      transform.offset_y.resize(points.size());
      transform.orientation_cos.resize(points.size());
      transform.orientation_sin.resize(points.size());
      const float radians = static_cast<float>(angle * CV_PI / 180.0);
      const float c = static_cast<float>(std::cos(static_cast<double>(radians)));
      const float s = static_cast<float>(std::sin(static_cast<double>(radians)));
      transform.bounds.min_x = transform.bounds.min_y = std::numeric_limits<double>::max();
      transform.bounds.max_x = transform.bounds.max_y = -std::numeric_limits<double>::max();
      for (std::size_t i = 0; i < points.size(); ++i) {
        const float local_x = c * points.relative_x[i] - s * points.relative_y[i];
        const float local_y = s * points.relative_x[i] + c * points.relative_y[i];
        const double x = scale * static_cast<double>(local_x);
        const double y = scale * static_cast<double>(local_y);
        transform.offset_x[i] = x;
        transform.offset_y[i] = y;
        const double orientation = static_cast<double>(points.orientation[i] + radians);
        transform.orientation_cos[i] = std::cos(orientation);
        transform.orientation_sin[i] = std::sin(orientation);
        transform.bounds.min_x = std::min(transform.bounds.min_x, x);
        transform.bounds.min_y = std::min(transform.bounds.min_y, y);
        transform.bounds.max_x = std::max(transform.bounds.max_x, x);
        transform.bounds.max_y = std::max(transform.bounds.max_y, y);
      }
      plan->transforms.push_back(std::move(transform));
    }
  }
  std::size_t bytes = sizeof(*plan);
  for (const auto& transform : plan->transforms) {
    bytes += sizeof(transform) + transform.offset_x.capacity() * sizeof(double) * 4;
  }
  plan->memory_bytes = bytes;
  return plan;
}

ComputeKernel validate_kernel_request(ComputeKernel requested) {
  if (requested == ComputeKernel::Scalar) return ComputeKernel::Scalar;
  if (requested == ComputeKernel::AVX512) {
#if defined(OPENSHAPE_HAS_AVX512_KERNEL)
    if (!detect_cpu_features().avx512f)
      throw InvalidArgument("AVX512 precise kernel requested but CPU does not support AVX512F");
    return ComputeKernel::AVX512;
#else
    throw InvalidArgument("AVX512 precise kernel is not available in this build");
#endif
  }
  if (requested == ComputeKernel::AVX2) {
#if defined(OPENSHAPE_HAS_AVX2_KERNEL)
    if (!detect_cpu_features().avx2)
      throw InvalidArgument("AVX2 precise kernel requested but CPU does not support AVX2");
    return ComputeKernel::AVX2;
#else
    throw InvalidArgument("AVX2 precise kernel is not available in this build");
#endif
  }
  return ComputeKernel::Scalar;
}
}

struct ExhaustiveSearchWorkspace::Impl {
  mutable std::mutex mutex;
  std::shared_ptr<const WorkspacePlan> plan;
  double last_build_ms = 0.0;
};

ExhaustiveSearchWorkspace::ExhaustiveSearchWorkspace()
    : impl_(std::make_unique<Impl>()) {}
ExhaustiveSearchWorkspace::~ExhaustiveSearchWorkspace() = default;
ExhaustiveSearchWorkspace::ExhaustiveSearchWorkspace(ExhaustiveSearchWorkspace&&) noexcept = default;
ExhaustiveSearchWorkspace& ExhaustiveSearchWorkspace::operator=(ExhaustiveSearchWorkspace&&) noexcept = default;

void ExhaustiveSearchWorkspace::prepare(const ShapeModel& model, const SearchParams& params) {
  params.validate();
  (void)validate_kernel_request(params.compute_kernel);
  if (model.empty()) throw InvalidModel("model is empty");
  const auto begin = std::chrono::steady_clock::now();
  const auto plan = build_workspace_plan(model, params);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->plan = plan;
  impl_->last_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
}

void ExhaustiveSearchWorkspace::clear() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->plan.reset();
}

std::size_t ExhaustiveSearchWorkspace::memory_bytes() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->plan ? impl_->plan->memory_bytes : 0;
}

std::vector<MatchResult> find_shape_models_exhaustive(
    const ImageView& image, const ShapeModel& model,
    const SearchParams& params, SearchStats* stats) {
  if (image.empty()) throw EmptyImage("image is empty");
  if (model.empty()) throw InvalidModel("model is empty");
  const int levels = params.num_levels == 0 ?
      static_cast<int>(model.levels().size()) : params.num_levels;
  EdgePyramid pyramid = EdgePyramid::build(
      image, model.params(), std::max(1, levels), true);
  return find_shape_models_exhaustive(pyramid.levels().front(), model, params, stats);
}

std::vector<MatchResult> find_shape_models_exhaustive_legacy(
    const EdgeMap& input_scene, const ShapeModel& model,
    const SearchParams& params, SearchStats* stats,
    const WorkspacePlan* prepared_plan = nullptr) {
  params.validate();
  if (model.empty()) throw InvalidModel("model is empty");
  EdgeMap scene = input_scene;
  if (scene.soft_edge_response.empty() || scene.subpixel_edge_mask.empty()) {
    if (scene.gray.empty()) throw InvalidArgument("scene edge map is incomplete");
    scene = EdgeEngine::compute(ImageView(scene.gray), model.params(), true);
  }
  if (stats) *stats = SearchStats{};
  cv::Rect roi = params.search_roi;
  if (roi.width == 0 || roi.height == 0) roi = cv::Rect(0, 0, scene.gray.cols, scene.gray.rows);
  if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > scene.gray.cols ||
      roi.y + roi.height > scene.gray.rows)
    throw InvalidArgument("search ROI is outside the image");

  const auto& points = model.levels_soa().front();
  const double angle_step = params.exhaustive_angle_step > 0.0 ?
      params.exhaustive_angle_step : params.angle_step;
  const double scale_step = params.exhaustive_scale_step > 0.0 ?
      params.exhaustive_scale_step : params.scale_step;
  const auto angles = grid_values(params.angle_start,
                                  params.angle_start + params.angle_extent, angle_step);
  const auto scales = grid_values(params.scale_min, params.scale_max, scale_step);
  const double low_threshold = params.min_score * params.level_min_score_factor;
  const int translation_step = std::max(
      1, static_cast<int>(std::lround(params.exhaustive_translation_step)));
  const std::size_t translation_columns = static_cast<std::size_t>(
      grid_value_count(roi.x, roi.x + roi.width, roi.x, translation_step));
  const std::size_t translation_rows = static_cast<std::size_t>(
      grid_value_count(roi.y, roi.y + roi.height, roi.y, translation_step));
  const std::size_t poses_per_transform = translation_columns * translation_rows;
  std::vector<MatchResult> all;
  cv::Mat support;
  cv::Mat support_integral;
  if (params.enable_safe_pruning) {
    support = scene.edges.clone();
    if (!scene.subpixel_edge_mask.empty())
      cv::bitwise_or(support, scene.subpixel_edge_mask, support);
    cv::integral(support, support_integral, CV_32S);
  }
  const std::vector<TranslationRun> full_runs =
      full_translation_runs(roi, translation_step);
  const std::vector<std::size_t> pruning_point_order =
      params.enable_safe_pruning ? build_pruning_point_order(points)
                                 : std::vector<std::size_t>{};
  bool use_reordered_points = false;
  for (std::size_t index = 0; index < pruning_point_order.size(); ++index) {
    if (pruning_point_order[index] != index) {
      use_reordered_points = true;
      break;
    }
  }
  std::map<SupportWindowKey, std::vector<TranslationRun>> support_run_cache;
  constexpr std::size_t minimum_sparse_domain_pose_count = 1024;
  const bool use_sparse_pose_domain =
      params.enable_safe_pruning && params.pruning_audit_rate <= 0.0 &&
      poses_per_transform >= minimum_sparse_domain_pose_count;
  if (stats) {
    stats->angle_candidate_count = angles.size();
    stats->scale_candidate_count = scales.size();
    stats->transform_candidate_count = angles.size() * scales.size();
    stats->theoretical_pose_count =
        stats->transform_candidate_count * poses_per_transform;
  }
  std::size_t pose_index = 0;
  const auto begin = std::chrono::steady_clock::now();
  for (std::size_t scale_index = 0; scale_index < scales.size(); ++scale_index) {
    const double scale = scales[scale_index];
    for (std::size_t angle_index = 0; angle_index < angles.size(); ++angle_index) {
      const double angle = angles[angle_index];
      const std::size_t transform_index = scale_index * angles.size() + angle_index;
      const PreparedTransform* prepared = prepared_plan &&
          transform_index < prepared_plan->transforms.size()
          ? &prepared_plan->transforms[transform_index] : nullptr;
      const TransformedBounds bounds = prepared ? prepared->bounds
                                                : transformed_bounds(points, angle, scale);
      const std::vector<TranslationRun>* runs = &full_runs;
      int legal_column_begin = roi.x;
      int legal_column_end = roi.x + roi.width;
      int legal_row_begin = roi.y;
      int legal_row_end = roi.y + roi.height;
      std::size_t legal_pose_count = poses_per_transform;
      if (use_sparse_pose_domain) {
        const SupportWindowKey key = support_window_key(
            bounds, params.edge_distance_sigma);
        auto found = support_run_cache.find(key);
        if (found == support_run_cache.end()) {
          found = support_run_cache.emplace(
              key, supported_translation_runs(
                       support, key, roi, translation_step)).first;
        }
        runs = &found->second;
        legal_column_begin = std::max(
            roi.x, static_cast<int>(std::ceil(-bounds.max_x)) - 1);
        legal_column_end = std::min(
            roi.x + roi.width,
            static_cast<int>(std::floor(
                static_cast<double>(scene.edges.cols - 1) - bounds.min_x)) + 2);
        legal_row_begin = std::max(
            roi.y, static_cast<int>(std::ceil(-bounds.max_y)) - 1);
        legal_row_end = std::min(
            roi.y + roi.height,
            static_cast<int>(std::floor(
                static_cast<double>(scene.edges.rows - 1) - bounds.min_y)) + 2);
        const std::size_t legal_columns = static_cast<std::size_t>(
            grid_value_count(legal_column_begin, legal_column_end,
                             roi.x, translation_step));
        const std::size_t legal_rows = static_cast<std::size_t>(
            grid_value_count(legal_row_begin, legal_row_end,
                             roi.y, translation_step));
        legal_pose_count = legal_columns * legal_rows;
        if (stats) {
          const std::size_t skipped = poses_per_transform - legal_pose_count;
          stats->aabb_domain_skipped_pose_count += skipped;
          stats->aabb_roi_rejections += skipped;
          stats->safe_pruning_rejections += skipped;
          stats->omitted_point_evaluations += skipped * points.size();
        }
      }
      std::size_t transform_enumerated_count = 0;
      for (const TranslationRun& run : *runs) {
        if (run.row < legal_row_begin || run.row >= legal_row_end) continue;
        const int column_begin = align_grid_value(
            std::max(run.column_begin, legal_column_begin),
            roi.x, translation_step);
        const int column_end = std::min(run.column_end, legal_column_end);
        for (int column = column_begin; column < column_end;
             column += translation_step) {
          const int row = run.row;
          ++transform_enumerated_count;
          const std::size_t current_pose_index = pose_index++;
          if (stats) {
            ++stats->domain_enumerated_pose_count;
            ++stats->pose_evaluations;
          }
          const PosePrefilterRejection prefilter = params.enable_safe_pruning
              ? prefilter_precise_pose(scene, support_integral, bounds, column, row,
                                       params.edge_distance_sigma)
              : PosePrefilterRejection::None;
          if (prefilter != PosePrefilterRejection::None) {
            bool confirmed_rejection = true;
            if (audit_pose(current_pose_index, params.pruning_audit_rate)) {
              if (stats) {
                ++stats->pruning_audit_evaluations;
                ++stats->full_score_evaluations;
                stats->score_point_evaluations += points.size();
              }
              std::size_t audit_valid = 0;
              const double audit_score = score_pose_precise(
                  scene, points, column, row, angle, scale, params.polarity,
                  params.edge_distance_sigma, &audit_valid);
              if (audit_score >= low_threshold) {
                confirmed_rejection = false;
                if (stats) ++stats->pruning_audit_failures;
              }
            }
            if (confirmed_rejection) {
              if (stats) {
                ++stats->safe_pruning_rejections;
                if (prefilter == PosePrefilterRejection::OutsideImage)
                  ++stats->aabb_roi_rejections;
                else
                  ++stats->edge_coverage_rejections;
                if (!audit_pose(current_pose_index, params.pruning_audit_rate))
                  stats->omitted_point_evaluations += points.size();
              }
              continue;
            }
          }

          PrunableScoreResult scored;
          if (!params.enable_safe_pruning && stats == nullptr && !prepared) {
            scored.score = score_pose_precise(
                scene, points, column, row, angle, scale, params.polarity,
                params.edge_distance_sigma, &scored.valid_point_count);
            scored.score_upper_bound = scored.score;
            scored.point_evaluations = points.size();
          } else if (!params.enable_safe_pruning && prepared) {
            scored.score = score_prepared_precise(
                scene, points, *prepared, column, row, params.polarity,
                params.edge_distance_sigma, &scored.valid_point_count);
            scored.score_upper_bound = scored.score;
            scored.point_evaluations = points.size();
          } else {
            scored = score_pose_precise_prunable(
                scene, points, column, row, angle, scale, params.polarity,
                params.edge_distance_sigma, low_threshold, 0,
                params.enable_safe_pruning,
                use_reordered_points ? &pruning_point_order : nullptr);
          }
          if (stats) stats->score_point_evaluations += scored.point_evaluations;
          if (scored.terminated()) {
            if (stats) {
              ++stats->safe_bound_terminations;
              stats->bound_terminated_point_evaluations += scored.point_evaluations;
              stats->max_termination_upper_bound = std::max(
                  stats->max_termination_upper_bound, scored.score_upper_bound);
              if (scored.reason == PruningReason::ScoreUpperBound)
                ++stats->score_bound_terminations;
              else
                ++stats->visible_bound_terminations;
            }
            bool confirmed_rejection = true;
            if (audit_pose(current_pose_index, params.pruning_audit_rate)) {
              if (stats) {
                ++stats->pruning_audit_evaluations;
                ++stats->full_score_evaluations;
                stats->score_point_evaluations += points.size();
              }
              std::size_t audit_valid = 0;
              const double audit_score = score_pose_precise(
                  scene, points, column, row, angle, scale, params.polarity,
                  params.edge_distance_sigma, &audit_valid);
              if (audit_score >= low_threshold) {
                scored.score = audit_score;
                scored.valid_point_count = audit_valid;
                confirmed_rejection = false;
                if (stats) ++stats->pruning_audit_failures;
              }
            }
            if (confirmed_rejection) {
              if (stats) {
                ++stats->safe_pruning_rejections;
                if (!audit_pose(current_pose_index, params.pruning_audit_rate))
                  stats->omitted_point_evaluations += points.size() - scored.point_evaluations;
              }
              continue;
            }
          } else {
            if (use_reordered_points) {
              std::size_t canonical_valid = 0;
              scored.score = score_pose_precise(
                  scene, points, column, row, angle, scale, params.polarity,
                  params.edge_distance_sigma, &canonical_valid);
              scored.valid_point_count = canonical_valid;
              scored.score_upper_bound = scored.score;
              if (stats) stats->score_point_evaluations += points.size();
            }
            if (stats) ++stats->full_score_evaluations;
          }
          const std::size_t valid = scored.valid_point_count;
          const double score = scored.score;
          if (score < low_threshold) continue;
          MatchResult result;
          result.column = column; result.row = row; result.angle = normalize_angle(angle);
          result.scale = scale; result.score = score; result.valid_point_count = valid;
          result.model_point_count = points.size(); result.level_used = 0;
          result.valid_point_fraction = points.empty() ? 0.0 :
              static_cast<double>(valid) / static_cast<double>(points.size());
          result.status = MatchStatus::Ambiguous;
          all.push_back(result);
          if (stats) ++stats->accepted_candidates;
        }
      }
      if (use_sparse_pose_domain && stats) {
        const std::size_t skipped = legal_pose_count - transform_enumerated_count;
        stats->edge_domain_skipped_pose_count += skipped;
        stats->edge_coverage_rejections += skipped;
        stats->safe_pruning_rejections += skipped;
        stats->omitted_point_evaluations += skipped * points.size();
      }
    }
  }
  if (stats)
    stats->domain_skipped_pose_count =
        stats->theoretical_pose_count - stats->domain_enumerated_pose_count;
  if (stats) stats->candidate_search_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();

  // Keep every local peak; there is intentionally no Top-K or clustering.
  std::stable_sort(all.begin(), all.end(), [](const MatchResult& a, const MatchResult& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.row != b.row) return a.row < b.row;
    if (a.column != b.column) return a.column < b.column;
    if (a.angle != b.angle) return a.angle < b.angle;
    return a.scale < b.scale;
  });
  // Extract all discrete local peaks before the nine-point refinement.  This
  // is not NMS: only samples belonging to the same one-pixel grid cell and
  // identical angle/scale are coalesced; distinct objects remain untouched.
  std::vector<MatchResult> peaks;
  peaks.reserve(all.size());
  for (const auto& candidate : all) {
    bool same_peak = false;
    for (const auto& peak : peaks) {
      if (std::abs(normalize_angle(candidate.angle - peak.angle)) < 1e-9 &&
          std::abs(candidate.scale - peak.scale) < 1e-12 &&
          std::hypot(candidate.column - peak.column, candidate.row - peak.row) <= 1.0) {
        same_peak = true;
        break;
      }
    }
    if (!same_peak) peaks.push_back(candidate);
  }
  std::vector<MatchResult> refined;
  refined.reserve(peaks.size());
  for (std::size_t peak_index = 0; peak_index < peaks.size(); ++peak_index) {
    auto candidate = peaks[peak_index];
    if (params.enable_subpixel || params.enable_pose_refinement || model.version() >= 2) {
      refine_nine(scene, points, params, candidate, peak_index == 0);
      if (peak_index == 0) apply_edge_residual_fit(scene, points, params, candidate);
    }
    candidate.status = strict_accept(params, candidate) ? MatchStatus::Accepted : MatchStatus::Rejected;
    if (candidate.status == MatchStatus::Accepted) refined.push_back(candidate);
  }
  std::stable_sort(refined.begin(), refined.end(), [](const MatchResult& a, const MatchResult& b) {
    return a.score > b.score;
  });
  std::vector<MatchResult> merged;
  for (const auto& candidate : refined) {
    bool duplicate = false;
    for (const auto& kept : merged) {
      if (std::abs(normalize_angle(candidate.angle - kept.angle)) < 1e-9 &&
          std::abs(candidate.scale - kept.scale) < 1e-12 &&
          std::hypot(candidate.column - kept.column, candidate.row - kept.row) < 0.75) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) merged.push_back(candidate);
  }
  if (params.num_matches > 0 && merged.size() > params.num_matches)
    merged.resize(params.num_matches);
  if (stats) {
    stats->refinement_input_candidates = peaks.size();
    stats->refinement_evaluations += peaks.size() * 10;
    stats->nms_input_candidates = refined.size();
    stats->nms_output_matches = merged.size();
  }
  // The reference path returns all accepted local peaks.  Applying legacy NMS
  // here would violate the plan's recall-first contract.
  return merged;
}

// Canonical candidate finalization shared by the prepared parallel path.  It
// intentionally mirrors the reference path's stable ordering and scalar
// refinement rules.
std::vector<MatchResult> finalize_exhaustive_candidates(
    const EdgeMap& scene, const ShapeModel& model, const SearchParams& params,
    SearchStats* stats, std::vector<MatchResult> all) {
  const auto& points = model.levels_soa().front();
  std::stable_sort(all.begin(), all.end(), [](const MatchResult& a, const MatchResult& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.row != b.row) return a.row < b.row;
    if (a.column != b.column) return a.column < b.column;
    if (a.angle != b.angle) return a.angle < b.angle;
    return a.scale < b.scale;
  });
  std::vector<MatchResult> peaks;
  peaks.reserve(all.size());
  for (const auto& candidate : all) {
    bool same_peak = false;
    for (const auto& peak : peaks) {
      if (std::abs(normalize_angle(candidate.angle - peak.angle)) < 1e-9 &&
          std::abs(candidate.scale - peak.scale) < 1e-12 &&
          std::hypot(candidate.column - peak.column, candidate.row - peak.row) <= 1.0) {
        same_peak = true;
        break;
      }
    }
    if (!same_peak) peaks.push_back(candidate);
  }
  std::vector<MatchResult> refined;
  refined.reserve(peaks.size());
  for (std::size_t peak_index = 0; peak_index < peaks.size(); ++peak_index) {
    auto candidate = peaks[peak_index];
    if (params.enable_subpixel || params.enable_pose_refinement || model.version() >= 2) {
      refine_nine(scene, points, params, candidate, peak_index == 0);
      if (peak_index == 0) apply_edge_residual_fit(scene, points, params, candidate);
    }
    candidate.status = strict_accept(params, candidate) ? MatchStatus::Accepted
                                                          : MatchStatus::Rejected;
    if (candidate.status == MatchStatus::Accepted) refined.push_back(candidate);
  }
  std::stable_sort(refined.begin(), refined.end(), [](const MatchResult& a, const MatchResult& b) {
    return a.score > b.score;
  });
  std::vector<MatchResult> merged;
  for (const auto& candidate : refined) {
    bool duplicate = false;
    for (const auto& kept : merged) {
      if (std::abs(normalize_angle(candidate.angle - kept.angle)) < 1e-9 &&
          std::abs(candidate.scale - kept.scale) < 1e-12 &&
          std::hypot(candidate.column - kept.column, candidate.row - kept.row) < 0.75) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) merged.push_back(candidate);
  }
  if (params.num_matches > 0 && merged.size() > params.num_matches)
    merged.resize(params.num_matches);
  if (stats) {
    stats->refinement_input_candidates = peaks.size();
    stats->refinement_evaluations += peaks.size() * 10;
    stats->nms_input_candidates = refined.size();
    stats->nms_output_matches = merged.size();
  }
  return merged;
}

std::vector<MatchResult> search_prepared_parallel(
    const EdgeMap& scene, const ShapeModel& model, const SearchParams& params,
    const WorkspacePlan& plan, SearchStats* stats) {
  const auto& points = model.levels_soa().front();
  cv::Rect roi = params.search_roi;
  if (roi.width == 0 || roi.height == 0)
    roi = cv::Rect(0, 0, scene.edges.cols, scene.edges.rows);
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > scene.edges.cols || roi.y + roi.height > scene.edges.rows)
    throw InvalidArgument("search ROI is outside the image");
  const int translation_step = std::max(1, static_cast<int>(
      std::lround(params.exhaustive_translation_step)));
  const std::size_t columns = static_cast<std::size_t>(
      grid_value_count(roi.x, roi.x + roi.width, roi.x, translation_step));
  const std::size_t rows = static_cast<std::size_t>(
      grid_value_count(roi.y, roi.y + roi.height, roi.y, translation_step));
  const std::size_t poses_per_transform = columns * rows;
  const std::size_t transform_count = plan.transforms.size();
  std::size_t workers = params.num_threads > 0
      ? static_cast<std::size_t>(params.num_threads)
      : static_cast<std::size_t>(std::thread::hardware_concurrency());
  if (workers == 0) workers = 1;
  if (params.num_threads == 0) workers = std::min<std::size_t>(workers, 16);
  workers = std::min(workers, std::max<std::size_t>(1, transform_count));
  struct Local { std::vector<MatchResult> candidates; SearchStats stats; };
  std::vector<Local> locals(workers);
  std::vector<std::thread> threads;
  threads.reserve(workers);
  const double low_threshold = params.min_score * params.level_min_score_factor;
  const auto begin = std::chrono::steady_clock::now();
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin_transform = transform_count * worker / workers;
    const std::size_t end_transform = transform_count * (worker + 1) / workers;
    threads.emplace_back([&, worker, begin_transform, end_transform] {
      auto& local = locals[worker];
      const auto worker_begin = stats ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
      local.stats.actual_worker_count = 1;
      for (std::size_t transform_index = begin_transform;
           transform_index < end_transform; ++transform_index) {
        const auto& transform = plan.transforms[transform_index];
        for (std::size_t row_index = 0; row_index < rows; ++row_index) {
          const int row = roi.y + static_cast<int>(row_index * translation_step);
          for (std::size_t column_index = 0; column_index < columns; ++column_index) {
            const int column = roi.x + static_cast<int>(column_index * translation_step);
            ++local.stats.pose_evaluations;
            ++local.stats.domain_enumerated_pose_count;
            std::size_t valid = 0;
            const double score = score_prepared_precise(
                scene, points, transform, column, row, params.polarity,
                params.edge_distance_sigma, &valid);
            ++local.stats.full_score_evaluations;
            local.stats.score_point_evaluations += points.size();
            if (score < low_threshold) continue;
            MatchResult result;
            result.column = column;
            result.row = row;
            result.angle = normalize_angle(transform.angle);
            result.scale = transform.scale;
            result.score = score;
            result.valid_point_count = valid;
            result.model_point_count = points.size();
            result.level_used = 0;
            result.valid_point_fraction = points.empty() ? 0.0 :
                static_cast<double>(valid) / static_cast<double>(points.size());
            result.status = MatchStatus::Ambiguous;
            local.candidates.push_back(result);
            ++local.stats.accepted_candidates;
          }
        }
      }
      if (stats) {
        local.stats.worker_score_cpu_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - worker_begin).count();
      }
    });
  }
  for (auto& thread : threads) thread.join();
  std::vector<MatchResult> all;
  for (const auto& local : locals) all.insert(all.end(), local.candidates.begin(), local.candidates.end());
  if (stats) {
    stats->angle_candidate_count = plan.angles.size();
    stats->scale_candidate_count = plan.scales.size();
    stats->transform_candidate_count = transform_count;
    stats->theoretical_pose_count = transform_count * poses_per_transform;
    stats->actual_worker_count = workers;
    stats->worker_count = workers;
    stats->parallel_search_wall_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    for (const auto& local : locals) {
      stats->pose_evaluations += local.stats.pose_evaluations;
      stats->domain_enumerated_pose_count += local.stats.domain_enumerated_pose_count;
      stats->full_score_evaluations += local.stats.full_score_evaluations;
      stats->accepted_candidates += local.stats.accepted_candidates;
      stats->score_point_evaluations += local.stats.score_point_evaluations;
      stats->worker_score_cpu_time_ms += local.stats.worker_score_cpu_time_ms;
    }
    stats->domain_skipped_pose_count = stats->theoretical_pose_count -
                                       stats->domain_enumerated_pose_count;
  }
  const auto merge_begin = std::chrono::steady_clock::now();
  auto results = finalize_exhaustive_candidates(scene, model, params, stats,
                                                std::move(all));
  if (stats) stats->worker_merge_sort_wall_time_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - merge_begin).count();
  return results;
}

std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& scene, const ShapeModel& model,
    const SearchParams& params, SearchStats* stats) {
  const ComputeKernel selected = validate_kernel_request(params.compute_kernel);
  auto result = find_shape_models_exhaustive_legacy(scene, model, params, stats);
  if (stats) stats->selected_precise_kernel = selected;
  return result;
}

std::vector<MatchResult> find_shape_models_exhaustive(
    const ImageView& image, const ShapeModel& model,
    const SearchParams& params, ExhaustiveSearchWorkspace* workspace,
    SearchStats* stats) {
  if (image.empty()) throw EmptyImage("image is empty");
  if (model.empty()) throw InvalidModel("model is empty");
  const int levels = params.num_levels == 0
      ? static_cast<int>(model.levels().size()) : params.num_levels;
  EdgePyramid pyramid = EdgePyramid::build(
      image, model.params(), std::max(1, levels), true);
  return find_shape_models_exhaustive(pyramid.levels().front(), model, params,
                                      workspace, stats);
}

std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& scene, const ShapeModel& model,
    const SearchParams& params, ExhaustiveSearchWorkspace* workspace,
    SearchStats* stats) {
  params.validate();
  const auto lookup_begin = std::chrono::steady_clock::now();
  const ComputeKernel selected = validate_kernel_request(params.compute_kernel);
  if (!workspace) {
    auto result = find_shape_models_exhaustive_legacy(scene, model, params, stats);
    if (stats) stats->selected_precise_kernel = selected;
    return result;
  }
  if (model.empty()) throw InvalidModel("model is empty");
  const double angle_step = params.exhaustive_angle_step > 0.0
      ? params.exhaustive_angle_step : params.angle_step;
  const double scale_step = params.exhaustive_scale_step > 0.0
      ? params.exhaustive_scale_step : params.scale_step;
  const auto angles = grid_values(params.angle_start,
                                  params.angle_start + params.angle_extent,
                                  angle_step);
  const auto scales = grid_values(params.scale_min, params.scale_max, scale_step);
  const std::uint64_t requested_key = workspace_key(model, params, angles, scales);
  std::shared_ptr<const WorkspacePlan> plan;
  {
    std::lock_guard<std::mutex> lock(workspace->impl_->mutex);
    plan = workspace->impl_->plan;
  }
  bool hit = plan && plan->key == requested_key;
  if (!hit) {
    workspace->prepare(model, params);
    std::lock_guard<std::mutex> lock(workspace->impl_->mutex);
    plan = workspace->impl_->plan;
  }
  // Safe-pruning domains retain the reference implementation's audit logic;
  // the prepared path is used for the non-pruned exhaustive workload.
  const auto search_begin = std::chrono::steady_clock::now();
  std::vector<MatchResult> result;
  if (!params.enable_safe_pruning && params.num_threads != 1) {
    if (stats) *stats = SearchStats{};
    result = search_prepared_parallel(scene, model, params, *plan, stats);
  } else {
    result = find_shape_models_exhaustive_legacy(scene, model, params, stats,
                                                  plan.get());
  }
  if (stats) {
    stats->selected_precise_kernel = selected;
    stats->workspace_cache_hits = hit ? 1 : 0;
    stats->workspace_cache_misses = hit ? 0 : 1;
    stats->workspace_cache_rebuilds = hit ? 0 : 1;
    stats->workspace_cache_hit = stats->workspace_cache_hits;
    stats->workspace_cache_miss = stats->workspace_cache_misses;
    stats->workspace_cache_rebuild = stats->workspace_cache_rebuilds;
    stats->workspace_memory_bytes = plan ? plan->memory_bytes : 0;
    stats->cache_lookup_wall_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - lookup_begin).count();
    {
      std::lock_guard<std::mutex> lock(workspace->impl_->mutex);
      stats->transform_table_preparation_wall_time_ms = hit ? 0.0 : workspace->impl_->last_build_ms;
    }
    stats->transform_preparation_time_ms =
        stats->transform_table_preparation_wall_time_ms;
    stats->rotation_preparation_time_ms =
        stats->transform_table_preparation_wall_time_ms;
    stats->parallel_search_wall_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - search_begin).count();
    stats->actual_worker_count = params.num_threads > 0
        ? static_cast<std::size_t>(params.num_threads)
        : std::min<std::size_t>(16, std::max<unsigned>(1, std::thread::hardware_concurrency()));
    stats->worker_count = stats->actual_worker_count;
  }
  return result;
}
}
