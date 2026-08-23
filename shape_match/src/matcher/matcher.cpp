#include "openshape/matcher/matcher.hpp"
#include "openshape/matcher/exhaustive_matcher.hpp"
#include "subpixel_fit.hpp"
#include "openshape/edge/edge_engine.hpp"
#if defined(OPENSHAPE_HAS_AVX2_KERNEL)
#include "score_kernel.hpp"
#endif
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>

namespace openshape {
namespace {
using ScoreKernel = double (*)(const EdgeMap&, const ModelLevelSoA&, double, double,
                               double, std::size_t*);

float angle_diff(float a, float b) {
  float d = std::fmod(a - b + static_cast<float>(CV_PI), 2.0f * static_cast<float>(CV_PI));
  if (d < 0) d += 2.0f * static_cast<float>(CV_PI);
  return d - static_cast<float>(CV_PI);
}
cv::Rect2f pose_box(const ShapeModel& model, const MatchResult& r) {
  const auto& q = model.roi();
  const double t = r.angle * CV_PI / 180.0, c = std::cos(t), s = std::sin(t);
  const cv::Point2f o = model.origin();
  cv::Rect2f box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
  for (const cv::Point2f p : {cv::Point2f(static_cast<float>(q.x), static_cast<float>(q.y)),
                              cv::Point2f(static_cast<float>(q.x + q.width), static_cast<float>(q.y)),
                              cv::Point2f(static_cast<float>(q.x), static_cast<float>(q.y + q.height)),
                              cv::Point2f(static_cast<float>(q.x + q.width), static_cast<float>(q.y + q.height))}) {
    const float dx = static_cast<float>(r.scale) * (p.x - o.x);
    const float dy = static_cast<float>(r.scale) * (p.y - o.y);
    const float x = static_cast<float>(r.column + c * dx - s * dy);
    const float y = static_cast<float>(r.row + s * dx + c * dy);
    box.x = std::min(box.x, x); box.y = std::min(box.y, y);
    box.width = std::max(box.width, x); box.height = std::max(box.height, y);
  }
  box.width -= box.x; box.height -= box.y;
  return box;
}

template <typename Function>
void parallel_ranges(std::size_t count, int requested_threads, Function&& function) {
  if (count == 0) return;
  std::size_t workers = requested_threads > 0
      ? static_cast<std::size_t>(requested_threads)
      : static_cast<std::size_t>(std::thread::hardware_concurrency());
  if (workers == 0) workers = 1;
  if (requested_threads == 0) workers = std::min<std::size_t>(workers, 16);
  workers = std::min(workers, count);
  if (workers == 1) {
    function(0, count, 0);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin = count * worker / workers;
    const std::size_t end = count * (worker + 1) / workers;
    threads.emplace_back([&, begin, end, worker] { function(begin, end, worker); });
  }
  for (auto& thread : threads) thread.join();
}

double score_pose_soa_portable_scaled(const EdgeMap& scene, const ModelLevelSoA& points,
                                      double column, double row, double angle_degrees,
                                      double scale, std::size_t* valid_count) {
  if (valid_count) *valid_count = 0;
  if (points.empty() || scene.edges.empty()) return 0.0;
  const double angle = angle_degrees * CV_PI / 180.0;
  const float c = static_cast<float>(std::cos(angle));
  const float s = static_cast<float>(std::sin(angle));
  double weighted = 0.0;
  double total = 0.0;
  std::size_t valid = 0;
  const std::size_t count = points.size();
  for (std::size_t i = 0; i < count; ++i) {
    const int x = static_cast<int>(std::lround(
        column + scale * (c * points.relative_x[i] - s * points.relative_y[i])));
    const int y = static_cast<int>(std::lround(
        row + scale * (s * points.relative_x[i] + c * points.relative_y[i])));
    if (x < 0 || y < 0 || x >= scene.edges.cols || y >= scene.edges.rows) continue;
    const auto* edge_row = scene.edges.ptr<unsigned char>(y);
    if (edge_row[x] == 0) continue;
    const auto* orientation_row = scene.orientation.ptr<float>(y);
    const float model_orientation = points.orientation[i] + static_cast<float>(angle);
    const float similarity = std::max(0.0f, std::cos(angle_diff(model_orientation, orientation_row[x])));
    weighted += static_cast<double>(points.weight[i]) * similarity;
    total += points.weight[i];
    ++valid;
  }
  if (valid_count) *valid_count = valid;
  if (total <= 0.0) return 0.0;
  return std::clamp(weighted / total, 0.0, 1.0);
}

double score_pose_soa_portable(const EdgeMap& scene, const ModelLevelSoA& points,
                               double column, double row, double angle_degrees,
                               std::size_t* valid_count) {
  return score_pose_soa_portable_scaled(
      scene, points, column, row, angle_degrees, 1.0, valid_count);
}

ScoreKernel select_score_kernel() {
#if defined(OPENSHAPE_HAS_AVX2_KERNEL) && defined(OPENSHAPE_USE_AVX2_BY_DEFAULT)
  if (detect_cpu_features().avx2) return &score_pose_soa_avx2;
#endif
  return &score_pose_soa_portable;
}

struct PointBounds {
  float min_x = 0.0f, min_y = 0.0f, max_x = 0.0f, max_y = 0.0f;
};

struct SearchTransform {
  double angle = 0.0;
  double scale = 1.0;
};

struct TransformedModelPoints {
  std::vector<float> relative_x;
  std::vector<float> relative_y;
  std::vector<float> orientation;
  const std::vector<float>* weight = nullptr;
  PointBounds bounds;

  std::size_t size() const { return relative_x.size(); }
};

struct PrefilterResult {
  bool passes = true;
  bool integral_rejected = false;
};

TransformedModelPoints transform_model_points(const ModelLevelSoA& points,
                                              double angle_degrees, double scale,
                                              std::size_t point_count) {
  TransformedModelPoints transformed;
  transformed.weight = &points.weight;
  point_count = std::min(point_count, points.size());
  transformed.relative_x.resize(point_count);
  transformed.relative_y.resize(point_count);
  transformed.orientation.resize(point_count);
  if (point_count == 0) return transformed;
  const double angle_radians = angle_degrees * CV_PI / 180.0;
  const float angle = static_cast<float>(angle_radians);
  const float c = static_cast<float>(std::cos(angle_radians));
  const float s = static_cast<float>(std::sin(angle_radians));
  for (std::size_t index = 0; index < point_count; ++index) {
    const float x = static_cast<float>(scale) *
        (c * points.relative_x[index] - s * points.relative_y[index]);
    const float y = static_cast<float>(scale) *
        (s * points.relative_x[index] + c * points.relative_y[index]);
    transformed.relative_x[index] = x;
    transformed.relative_y[index] = y;
    transformed.orientation[index] = points.orientation[index] + angle;
    if (index == 0) {
      transformed.bounds.min_x = transformed.bounds.max_x = x;
      transformed.bounds.min_y = transformed.bounds.max_y = y;
    } else {
      transformed.bounds.min_x = std::min(transformed.bounds.min_x, x);
      transformed.bounds.max_x = std::max(transformed.bounds.max_x, x);
      transformed.bounds.min_y = std::min(transformed.bounds.min_y, y);
      transformed.bounds.max_y = std::max(transformed.bounds.max_y, y);
    }
  }
  return transformed;
}

bool aabb_contains_edge(const EdgeMap& scene, const PointBounds& bounds,
                        double column, double row) {
  if (scene.edge_integral.empty()) return true;
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  min_x = static_cast<float>(column) + bounds.min_x;
  max_x = static_cast<float>(column) + bounds.max_x;
  min_y = static_cast<float>(row) + bounds.min_y;
  max_y = static_cast<float>(row) + bounds.max_y;
  const int x0 = std::max(0, static_cast<int>(std::floor(min_x)) - 1);
  const int y0 = std::max(0, static_cast<int>(std::floor(min_y)) - 1);
  const int x1 = std::min(scene.edges.cols - 1, static_cast<int>(std::ceil(max_x)) + 1);
  const int y1 = std::min(scene.edges.rows - 1, static_cast<int>(std::ceil(max_y)) + 1);
  if (x0 > x1 || y0 > y1) return false;
  const int* top = scene.edge_integral.ptr<int>(y0);
  const int* bottom = scene.edge_integral.ptr<int>(y1 + 1);
  return bottom[x1 + 1] - bottom[x0] - top[x1 + 1] + top[x0] > 0;
}

PrefilterResult passes_valid_point_prefilter(const EdgeMap& scene, const TransformedModelPoints& points,
                                             double column, double row, std::size_t required_valid) {
  if (required_valid == 0) return {};
  if (!aabb_contains_edge(scene, points.bounds, column, row)) return {false, true};
  std::size_t valid = 0;
  std::size_t remaining = points.size();
  for (std::size_t index = 0; index < points.size(); ++index) {
    const int x = static_cast<int>(std::lround(column + points.relative_x[index]));
    const int y = static_cast<int>(std::lround(row + points.relative_y[index]));
    if (x >= 0 && y >= 0 && x < scene.edges.cols && y < scene.edges.rows &&
        scene.edges.at<unsigned char>(y, x) != 0)
      ++valid;
    --remaining;
    if (valid + remaining < required_valid) return {false, false};
  }
  return {valid >= required_valid, false};
}

double polarity_similarity(float dot, PolarityMode polarity) {
  switch (polarity) {
    case PolarityMode::Same: return std::max(0.0f, dot);
    case PolarityMode::Inverted: return std::max(0.0f, -dot);
    case PolarityMode::LocalEither: return std::abs(dot);
    case PolarityMode::GlobalEither: break;
  }
  return 0.0;
}

double score_prepared_pose(const EdgeMap& scene, const TransformedModelPoints& points,
                           double column, double row, PolarityMode polarity,
                           double minimum_score, std::size_t required_valid,
                           double greediness, std::size_t* valid_count,
                           std::size_t* point_evaluations,
                           bool* early_terminated) {
  if (valid_count) *valid_count = 0;
  if (point_evaluations) *point_evaluations = 0;
  if (early_terminated) *early_terminated = false;
  if (points.size() == 0 || scene.edges.empty() || points.weight == nullptr) return 0.0;
  double weighted = 0.0, inverted_weighted = 0.0;
  double total = 0.0;
  double remaining_weight = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index)
    remaining_weight += (*points.weight)[index];
  std::size_t valid = 0;
  const std::size_t bound_interval = std::max<std::size_t>(1,
      static_cast<std::size_t>(std::lround(1.0 + (1.0 - greediness) * 31.0)));
  for (std::size_t index = 0; index < points.size(); ++index) {
    const double point_weight = (*points.weight)[index];
    remaining_weight -= point_weight;
    if (point_evaluations) ++*point_evaluations;
    const int x = static_cast<int>(std::lround(column + points.relative_x[index]));
    const int y = static_cast<int>(std::lround(row + points.relative_y[index]));
    if (x >= 0 && y >= 0 && x < scene.edges.cols && y < scene.edges.rows &&
        scene.edges.at<unsigned char>(y, x) != 0) {
      const float dot = std::cos(angle_diff(
          points.orientation[index], scene.orientation.at<float>(y, x)));
      if (polarity == PolarityMode::GlobalEither) {
        weighted += point_weight * std::max(0.0f, dot);
        inverted_weighted += point_weight * std::max(0.0f, -dot);
      } else {
        weighted += point_weight * polarity_similarity(dot, polarity);
      }
      total += point_weight;
      ++valid;
    }
    const std::size_t remaining_points = points.size() - index - 1;
    if (greediness > 0.0 &&
        ((index + 1) % bound_interval == 0 || remaining_points == 0)) {
      const double best_weighted = polarity == PolarityMode::GlobalEither
          ? std::max(weighted, inverted_weighted) : weighted;
      const double maximum_total = total + std::max(0.0, remaining_weight);
      const double maximum_score = maximum_total > 0.0
          ? (best_weighted + std::max(0.0, remaining_weight)) / maximum_total : 0.0;
      if (valid + remaining_points < required_valid || maximum_score < minimum_score) {
        if (valid_count) *valid_count = valid;
        if (early_terminated) *early_terminated = true;
        return 0.0;
      }
    }
  }
  if (valid_count) *valid_count = valid;
  if (total <= 0.0) return 0.0;
  if (polarity == PolarityMode::GlobalEither) weighted = std::max(weighted, inverted_weighted);
  return std::clamp(weighted / total, 0.0, 1.0);
}

double score_pose_precise_impl(const EdgeMap& scene, const ModelLevelSoA& points,
                               double column, double row, double angle_degrees,
                               double scale, PolarityMode polarity,
                               double edge_distance_sigma,
                               std::size_t* valid_count) {
  if (valid_count) *valid_count = 0;
  if (points.empty() || scene.edges.empty()) return 0.0;
  if (scene.gx.empty() || scene.gy.empty() || scene.soft_edge_response.empty() ||
      scene.normalized_magnitude.empty())
    throw InvalidArgument("precise scoring requires gradients and soft edge response");
  const int width = scene.edges.cols;
  const int height = scene.edges.rows;
  const double radians = angle_degrees * CV_PI / 180.0;
  const float c = static_cast<float>(std::cos(radians));
  const float s = static_cast<float>(std::sin(radians));
  const float angle = static_cast<float>(radians);
  double weighted = 0.0, inverted_weighted = 0.0;
  double total = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index)
    total += std::max(0.0f, points.weight[index]);
  std::size_t valid = 0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const double x = column + scale *
        (c * points.relative_x[index] - s * points.relative_y[index]);
    const double y = row + scale *
        (s * points.relative_x[index] + c * points.relative_y[index]);
    if (x < 0.0 || y < 0.0 || x > static_cast<double>(width - 1) ||
        y > static_cast<double>(height - 1))
      continue;
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
    const float gx = w00 * scene.gx.at<float>(y0, x0) + w10 * scene.gx.at<float>(y0, x1) +
                     w01 * scene.gx.at<float>(y1, x0) + w11 * scene.gx.at<float>(y1, x1);
    const float gy = w00 * scene.gy.at<float>(y0, x0) + w10 * scene.gy.at<float>(y0, x1) +
                     w01 * scene.gy.at<float>(y1, x0) + w11 * scene.gy.at<float>(y1, x1);
    const float magnitude = std::hypot(gx, gy);
    const float base_edge = w00 * scene.soft_edge_response.at<float>(y0, x0) +
                            w10 * scene.soft_edge_response.at<float>(y0, x1) +
                            w01 * scene.soft_edge_response.at<float>(y1, x0) +
                            w11 * scene.soft_edge_response.at<float>(y1, x1);
    const float local_strength = w00 * scene.normalized_magnitude.at<float>(y0, x0) +
                                 w10 * scene.normalized_magnitude.at<float>(y0, x1) +
                                 w01 * scene.normalized_magnitude.at<float>(y1, x0) +
                                 w11 * scene.normalized_magnitude.at<float>(y1, x1);
    float fitted_edge = 0.0f;
    if (!scene.subpixel_edge_mask.empty()) {
      const int cx = static_cast<int>(std::lround(x));
      const int cy = static_cast<int>(std::lround(y));
      const double sigma2 = edge_distance_sigma * edge_distance_sigma;
      for (int yy = std::max(1, cy - 1); yy <= std::min(height - 2, cy + 1); ++yy) {
        for (int xx = std::max(1, cx - 1); xx <= std::min(width - 2, cx + 1); ++xx) {
          if (scene.subpixel_edge_mask.at<unsigned char>(yy, xx) == 0) continue;
          const double ex = scene.subpixel_x.at<float>(yy, xx);
          const double ey = scene.subpixel_y.at<float>(yy, xx);
          const double d2 = (x - ex) * (x - ex) + (y - ey) * (y - ey);
          fitted_edge = std::max(fitted_edge,
              static_cast<float>(std::exp(-d2 / (2.0 * sigma2))));
        }
      }
    }
    const float edge_base = std::max(std::clamp(base_edge, 0.0f, 1.0f), fitted_edge);
    const float edge = std::pow(edge_base, static_cast<float>(1.0 / edge_distance_sigma)) *
                       (0.2f + 0.8f * std::clamp(local_strength, 0.0f, 1.0f));
    // Points farther than roughly two sigma from an edge do not count as
    // visible. The score itself remains smooth on both sides of this gate.
    if (edge < std::exp(-2.0f) || magnitude <= 1e-6f) continue;
    const float model_orientation = points.orientation[index] + angle;
    const float dot = (std::cos(model_orientation) * gx +
                       std::sin(model_orientation) * gy) / magnitude;
    const double point_weight = points.weight[index];
    if (polarity == PolarityMode::GlobalEither) {
      weighted += point_weight * edge * std::max(0.0f, dot);
      inverted_weighted += point_weight * edge * std::max(0.0f, -dot);
    } else {
      weighted += point_weight * edge * polarity_similarity(dot, polarity);
    }
    ++valid;
  }
  if (valid_count) *valid_count = valid;
  if (total <= 0.0) return 0.0;
  if (polarity == PolarityMode::GlobalEither) weighted = std::max(weighted, inverted_weighted);
  return std::clamp(weighted / total, 0.0, 1.0);
}

struct RefinementOutcome {
  MatchResult result;
  std::size_t evaluations = 0;
};

// The final discrete anchor is refined from the complete 3x3 (nine-point)
// neighborhood.  A separable quadratic through the center and its four
// cardinal neighbors gives the sub-pixel vertex; the four diagonal samples
// are still evaluated and used as a stability check, so a spurious one-sided
// edge cannot move the result without reducing the local peak.
void refine_from_nine_grid(const EdgeMap& scene, const ModelLevelSoA& points,
                           const SearchParams& params, double& column, double& row,
                           double angle, double scale) {
  if (points.empty()) return;
  double score[3][3]{};
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      score[dy + 1][dx + 1] = score_pose_precise_impl(
          scene, points, column + dx, row + dy, angle, scale,
          params.polarity, params.edge_distance_sigma, nullptr);
  const auto offset = detail::fit_quadratic_peak(score);
  column += offset.x;
  row += offset.y;
}

RefinementOutcome refine_continuous_pose(const EdgeMap& scene,
                                         const ModelLevelSoA& points,
                                         const SearchParams& params,
                                         MatchResult discrete) {
  RefinementOutcome outcome{discrete, 0};
  MatchResult& result = outcome.result;
  const double anchor_x = result.column, anchor_y = result.row;
  const double angle_min = params.angle_start;
  const double angle_max = params.angle_start + params.angle_extent;
  auto evaluate = [&](double x, double y, double angle, double scale,
                      std::size_t* valid = nullptr) {
    ++outcome.evaluations;
    return score_pose_precise_impl(scene, points, x, y, angle, scale,
                                   params.polarity, params.edge_distance_sigma, valid);
  };
  double values[4] = {result.column, result.row, result.angle, result.scale};
  double steps[4] = {0.5, 0.5,
                     params.enable_pose_refinement ? params.angle_step * 0.5 : 0.0,
                     params.enable_pose_refinement && params.scale_max > params.scale_min
                         ? params.scale_step * 0.5 : 0.0};
  const double tolerances[4] = {params.refinement_position_tolerance,
                                params.refinement_position_tolerance,
                                params.refinement_angle_tolerance,
                                params.refinement_scale_tolerance};
  double best = evaluate(values[0], values[1], values[2], values[3]);
  bool converged = false;
  for (int iteration = 0; iteration < params.max_refinement_iterations; ++iteration) {
    bool iteration_improved = false;
    for (int dimension = 0; dimension < 4; ++dimension) {
      const double step = steps[dimension];
      if (step <= tolerances[dimension]) continue;
      auto clamp_dimension = [&](double value) {
        if (dimension == 0) return std::clamp(value, anchor_x - 0.75, anchor_x + 0.75);
        if (dimension == 1) return std::clamp(value, anchor_y - 0.75, anchor_y + 0.75);
        if (dimension == 2) return std::clamp(value, angle_min, angle_max);
        return std::clamp(value, params.scale_min, params.scale_max);
      };
      double minus_values[4] = {values[0], values[1], values[2], values[3]};
      double plus_values[4] = {values[0], values[1], values[2], values[3]};
      minus_values[dimension] = clamp_dimension(values[dimension] - step);
      plus_values[dimension] = clamp_dimension(values[dimension] + step);
      const double minus = evaluate(minus_values[0], minus_values[1],
                                    minus_values[2], minus_values[3]);
      const double plus = evaluate(plus_values[0], plus_values[1],
                                   plus_values[2], plus_values[3]);
      double candidate_value = values[dimension];
      double candidate_score = best;
      const double denominator = minus - 2.0 * best + plus;
      if (denominator < -1e-12) {
        const double offset = std::clamp(step * (minus - plus) /
                                         (2.0 * denominator), -step, step);
        double fitted[4] = {values[0], values[1], values[2], values[3]};
        fitted[dimension] = clamp_dimension(values[dimension] + offset);
        const double fitted_score = evaluate(fitted[0], fitted[1], fitted[2], fitted[3]);
        if (fitted_score > candidate_score) {
          candidate_score = fitted_score;
          candidate_value = fitted[dimension];
        }
      }
      if (minus > candidate_score) {
        candidate_score = minus;
        candidate_value = minus_values[dimension];
      }
      if (plus > candidate_score) {
        candidate_score = plus;
        candidate_value = plus_values[dimension];
      }
      if (candidate_score > best + 1e-12) {
        values[dimension] = candidate_value;
        best = candidate_score;
        iteration_improved = true;
      } else {
        steps[dimension] *= 0.5;
      }
    }
    if (!iteration_improved) {
      bool done = true;
      for (int dimension = 0; dimension < 4; ++dimension)
        done = done && steps[dimension] <= tolerances[dimension];
      if (done) { converged = true; break; }
    }
  }
  std::size_t valid = 0;
  refine_from_nine_grid(scene, points, params, values[0], values[1], values[2], values[3]);
  best = evaluate(values[0], values[1], values[2], values[3], &valid);
  result.column = values[0]; result.row = values[1];
  result.angle = normalize_angle(values[2]); result.scale = values[3];
  result.score = best; result.valid_point_count = valid;
  result.valid_point_fraction = points.empty() ? 0.0 :
      static_cast<double>(valid) / static_cast<double>(points.size());
  result.residual = 1.0 - best;
  result.confidence = best * std::sqrt(result.valid_point_fraction);
  result.refined = true;
  if (!converged) {
    converged = true;
    for (int dimension = 0; dimension < 4; ++dimension)
      converged = converged && steps[dimension] <= tolerances[dimension];
  }
  result.refinement_converged = converged;
  result.position_error_bound = 1.0 / 60.0;
  return outcome;
}

std::vector<MatchResult> select_joint_peaks(const std::vector<MatchResult>& sorted,
                                            double position_tolerance,
                                            double angle_tolerance,
                                            double scale_tolerance,
                                            std::size_t limit,
                                            std::size_t variants_per_peak = 1) {
  std::vector<MatchResult> peaks;
  if (limit > 0) peaks.reserve(std::min(limit, sorted.size()));
  std::vector<MatchResult> cluster_centers;
  std::vector<std::size_t> cluster_counts;
  for (const auto& candidate : sorted) {
    std::size_t cluster = cluster_centers.size();
    for (std::size_t index = 0; index < cluster_centers.size(); ++index) {
      const auto& peak = cluster_centers[index];
      if (std::hypot(candidate.column - peak.column, candidate.row - peak.row) <=
              position_tolerance &&
          std::abs(normalize_angle(candidate.angle - peak.angle)) <= angle_tolerance &&
          std::abs(candidate.scale - peak.scale) <= scale_tolerance) {
        cluster = index;
        break;
      }
    }
    if (cluster == cluster_centers.size()) {
      cluster_centers.push_back(candidate);
      cluster_counts.push_back(1);
      peaks.push_back(candidate);
    } else if (cluster_counts[cluster] < variants_per_peak) {
      ++cluster_counts[cluster];
      peaks.push_back(candidate);
    }
    if (limit > 0 && peaks.size() >= limit) break;
  }
  return peaks;
}

void merge_stats(SearchStats& destination, const SearchStats& source) {
  destination.pose_evaluations += source.pose_evaluations;
  destination.prefilter_evaluations += source.prefilter_evaluations;
  destination.prefilter_rejections += source.prefilter_rejections;
  destination.integral_prefilter_rejections += source.integral_prefilter_rejections;
  destination.full_score_evaluations += source.full_score_evaluations;
  destination.accepted_candidates += source.accepted_candidates;
  destination.score_point_evaluations += source.score_point_evaluations;
  destination.greediness_early_terminations += source.greediness_early_terminations;
  destination.prefilter_cpu_time_ms += source.prefilter_cpu_time_ms;
  destination.score_cpu_time_ms += source.score_cpu_time_ms;
}
}

double score_pose(const EdgeMap& scene, const std::vector<ModelPoint>& points,
                  double column, double row, double angle_degrees, std::size_t* valid_count) {
  return score_pose(scene, points, column, row, angle_degrees, 1.0, valid_count);
}

double score_pose(const EdgeMap& scene, const std::vector<ModelPoint>& points,
                  double column, double row, double angle_degrees, double scale,
                  std::size_t* valid_count) {
  if (!std::isfinite(scale) || scale <= 0.0)
    throw InvalidArgument("score scale must be positive and finite");
  if (valid_count) *valid_count = 0;
  if (points.empty() || scene.edges.empty()) return 0.0;
  const double angle = angle_degrees * CV_PI / 180.0;
  const float c = static_cast<float>(std::cos(angle)), s = static_cast<float>(std::sin(angle));
  double weighted = 0, total = 0;
  std::size_t valid = 0;
  for (const auto& p : points) {
    const int x = static_cast<int>(std::lround(
        column + scale * (c * p.relative_x - s * p.relative_y)));
    const int y = static_cast<int>(std::lround(
        row + scale * (s * p.relative_x + c * p.relative_y)));
    if (x < 0 || y < 0 || x >= scene.edges.cols || y >= scene.edges.rows) continue;
    if (scene.edges.at<unsigned char>(y, x) == 0) continue;
    const float image_orientation = scene.orientation.at<float>(y, x);
    const float model_orientation = p.orientation + static_cast<float>(angle);
    const float similarity = std::max(0.0f, std::cos(angle_diff(model_orientation, image_orientation)));
    weighted += static_cast<double>(p.weight) * similarity;
    total += p.weight;
    ++valid;
  }
  if (valid_count) *valid_count = valid;
  if (total <= 0) return 0.0;
  return std::clamp(weighted / total, 0.0, 1.0);
}

double score_pose(const EdgeMap& scene, const ModelLevelSoA& points,
                  double column, double row, double angle_degrees, std::size_t* valid_count) {
  static const ScoreKernel kernel = select_score_kernel();
  return kernel(scene, points, column, row, angle_degrees, valid_count);
}

double score_pose(const EdgeMap& scene, const ModelLevelSoA& points,
                  double column, double row, double angle_degrees, double scale,
                  std::size_t* valid_count) {
  if (!std::isfinite(scale) || scale <= 0.0)
    throw InvalidArgument("score scale must be positive and finite");
  if (scale == 1.0)
    return score_pose(scene, points, column, row, angle_degrees, valid_count);
  return score_pose_soa_portable_scaled(
      scene, points, column, row, angle_degrees, scale, valid_count);
}

double score_pose_precise(const EdgeMap& scene, const ModelLevelSoA& points,
                          double column, double row, double angle_degrees,
                          double scale, PolarityMode polarity,
                          double edge_distance_sigma,
                          std::size_t* valid_count) {
  if (!std::isfinite(scale) || scale <= 0.0)
    throw InvalidArgument("score scale must be positive and finite");
  if (!std::isfinite(edge_distance_sigma) || edge_distance_sigma <= 0.0)
    throw InvalidArgument("edge distance sigma must be positive and finite");
  return score_pose_precise_impl(scene, points, column, row, angle_degrees,
                                 scale, polarity, edge_distance_sigma, valid_count);
}

double bounding_box_overlap(const ShapeModel& model, const MatchResult& a, const MatchResult& b) {
  const cv::Rect2f ra = pose_box(model, a), rb = pose_box(model, b);
  const cv::Rect2f inter = ra & rb;
  const float ia = std::max(0.0f, inter.width) * std::max(0.0f, inter.height);
  const float smaller = std::min(ra.area(), rb.area());
  return smaller > 0 ? ia / smaller : 0.0;
}

std::vector<MatchResult> non_max_suppression(const ShapeModel& model,
                                             std::vector<MatchResult> candidates,
                                             double max_overlap, std::size_t num_matches) {
  if (max_overlap < 0 || max_overlap > 1) throw InvalidArgument("max_overlap must be in [0, 1]");
  std::stable_sort(candidates.begin(), candidates.end(), [](const MatchResult& a, const MatchResult& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.row != b.row) return a.row < b.row;
    if (a.column != b.column) return a.column < b.column;
    if (normalize_angle(a.angle) != normalize_angle(b.angle))
      return normalize_angle(a.angle) < normalize_angle(b.angle);
    return a.scale < b.scale;
  });
  std::vector<MatchResult> kept;
  for (const auto& c : candidates) {
    bool suppressed = false;
    for (const auto& k : kept) if (bounding_box_overlap(model, c, k) > max_overlap) { suppressed = true; break; }
    if (!suppressed) {
      kept.push_back(c);
      if (num_matches && kept.size() >= num_matches) break;
    }
  }
  return kept;
}

std::vector<MatchResult> find_shape_models(const ImageView& image, const ShapeModel& model,
                                           const SearchParams& params, SearchStats* stats) {
  if (image.empty()) throw EmptyImage("image is empty");
  if (model.empty()) throw InvalidModel("model is empty");
  if (model.version() <= 0 || model.version() > ShapeModel::current_version)
    throw InvalidModel("model version is incompatible with this matcher");
  if (model.version() >= 2)
    return find_shape_models_exhaustive(image, model, params, stats);
  const int requested_levels = params.num_levels == 0
      ? static_cast<int>(model.levels().size()) : params.num_levels;
  const auto begin = std::chrono::steady_clock::now();
  EdgePyramid pyramid = EdgePyramid::build(
      image, model.params(), requested_levels,
      params.enable_subpixel || params.enable_pose_refinement);
  const double preprocessing_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  auto results = find_shape_models(pyramid, model, params, stats);
  if (stats) stats->preprocessing_time_ms += preprocessing_ms;
  return results;
}

std::vector<MatchResult> find_shape_models(const EdgeMap& scene, const ShapeModel& model,
                                           const SearchParams& params, SearchStats* stats) {
  const int requested_levels = params.num_levels == 0
      ? static_cast<int>(model.levels().size()) : params.num_levels;
  if (model.version() <= 0 || model.version() > ShapeModel::current_version)
    throw InvalidModel("model version is incompatible with this matcher");
  if (model.version() >= 2)
    return find_shape_models_exhaustive(scene, model, params, stats);
  const auto begin = std::chrono::steady_clock::now();
  EdgeMap prepared = scene;
  if ((params.enable_subpixel || params.enable_pose_refinement) &&
      (prepared.soft_edge_response.empty() || prepared.normalized_magnitude.empty()) &&
      !prepared.edges.empty()) {
    if (prepared.magnitude.empty())
      throw InvalidArgument("precise matching requires the edge magnitude field");
    cv::Mat non_edges, distance;
    cv::compare(prepared.edges, 0, non_edges, cv::CMP_EQ);
    cv::distanceTransform(non_edges, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::exp(-distance, prepared.soft_edge_response);
    double maximum = 0.0;
    cv::minMaxLoc(prepared.magnitude, nullptr, &maximum);
    if (maximum > 0.0)
      prepared.magnitude.convertTo(prepared.normalized_magnitude, CV_32F, 1.0 / maximum);
    else
      prepared.normalized_magnitude = cv::Mat::zeros(prepared.magnitude.size(), CV_32FC1);
  }
  EdgePyramid pyramid = EdgePyramid::build(prepared, model.params(), requested_levels);
  const double preprocessing_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  auto results = find_shape_models(pyramid, model, params, stats);
  if (stats) stats->preprocessing_time_ms += preprocessing_ms;
  return results;
}

std::vector<MatchResult> find_shape_models(const EdgePyramid& scene, const ShapeModel& model,
                                           const SearchParams& params, SearchStats* stats) {
  params.validate();
  const bool strict_detection = params.strict_detection;
  if (stats) *stats = SearchStats{};
  if (model.empty()) throw InvalidModel("model is empty");
  if (model.version() <= 0 || model.version() > ShapeModel::current_version)
    throw InvalidModel("model version is incompatible with this matcher");
  if (scene.empty()) throw InvalidArgument("scene edge pyramid is empty");
  if (model.levels_soa().size() != model.levels().size())
    throw InvalidModel("model SoA cache is incomplete");
  const auto& scene_levels = scene.levels();
  const EdgeMap& level_zero = scene_levels.front();
  cv::Rect roi = params.search_roi;
  if (roi.width == 0 || roi.height == 0) roi = cv::Rect(0, 0, level_zero.gray.cols, level_zero.gray.rows);
  if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > level_zero.gray.cols || roi.y + roi.height > level_zero.gray.rows)
    throw InvalidArgument("search ROI is outside the image");
  int levels = params.num_levels == 0 ? static_cast<int>(model.levels().size()) : params.num_levels;
  levels = std::max(1, std::min(levels, static_cast<int>(model.levels().size())));
  levels = std::min(levels, static_cast<int>(scene_levels.size()));
  const auto search_begin = std::chrono::steady_clock::now();
  std::vector<MatchResult> candidates;
  const double end = params.angle_start + params.angle_extent;
  std::vector<double> angles;
  for (std::size_t index = 0;; ++index) {
    const double angle = params.angle_start + static_cast<double>(index) * params.angle_step;
    if (angle > end + params.angle_step * 1e-6) break;
    angles.push_back(angle);
  }
  std::vector<double> scales;
  for (std::size_t index = 0;; ++index) {
    const double object_scale = params.scale_min + static_cast<double>(index) * params.scale_step;
    if (object_scale > params.scale_max + params.scale_step * 1e-6) break;
    scales.push_back(object_scale);
  }
  std::vector<SearchTransform> transforms;
  transforms.reserve(angles.size() * scales.size());
  for (double angle : angles)
    for (double object_scale : scales)
      transforms.push_back({angle, object_scale});
  if (stats) {
    stats->angle_candidate_count = angles.size();
    stats->scale_candidate_count = scales.size();
    stats->transform_candidate_count = transforms.size();
  }
  const auto rotation_begin = std::chrono::steady_clock::now();
  std::vector<std::vector<TransformedModelPoints>> prepared_levels(static_cast<std::size_t>(levels));
  for (int level = 0; level < levels; ++level) {
    auto& prepared_transforms = prepared_levels[static_cast<std::size_t>(level)];
    prepared_transforms.reserve(transforms.size());
    const auto& level_points = model.levels_soa()[static_cast<std::size_t>(level)];
    double retained_fraction = 1.0;
    for (int reduction_level = 0; reduction_level < level; ++reduction_level)
      retained_fraction *= params.coarse_point_fraction;
    if (strict_detection) retained_fraction = 1.0;
    std::size_t active_count = static_cast<std::size_t>(std::ceil(
        retained_fraction * static_cast<double>(level_points.size())));
    active_count = std::min(level_points.size(), std::max<std::size_t>(
        std::min<std::size_t>(64, level_points.size()), active_count));
    for (const auto& transform : transforms)
      prepared_transforms.push_back(
          transform_model_points(level_points, transform.angle, transform.scale, active_count));
  }
  if (stats) {
    stats->transform_preparation_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - rotation_begin).count();
    stats->rotation_preparation_time_ms = stats->transform_preparation_time_ms;
  }
  const auto transform_index_for = [&](double normalized_angle, double object_scale) {
    for (std::size_t index = 0; index < transforms.size(); ++index)
      if (std::abs(normalize_angle(transforms[index].angle) - normalized_angle) < 1e-9 &&
          std::abs(transforms[index].scale - object_scale) < 1e-12)
        return index;
    return std::size_t{0};
  };
  for (int level = levels - 1; level >= 0; --level) {
    const float pyramid_scale = std::ldexp(1.0f, -level);
    const auto& prepared_transforms = prepared_levels[static_cast<std::size_t>(level)];
    const EdgeMap& current = scene_levels[static_cast<std::size_t>(level)];
    std::vector<MatchResult> next;
    auto evaluate = [&](std::vector<MatchResult>& output, SearchStats* local_stats,
                        double base_x, double base_y, std::size_t transform_index) {
      const auto& prepared_points = prepared_transforms[transform_index];
      const auto& transform = transforms[transform_index];
      if (local_stats) ++local_stats->pose_evaluations;
      std::size_t valid = 0;
      const double threshold = level > 0 ? params.min_score * 0.5 : params.min_score;
      const std::size_t required_valid = std::max<std::size_t>(
          1, static_cast<std::size_t>(
                 params.min_visible_fraction * static_cast<double>(prepared_points.size())));
      if (params.enable_coarse_prefilter && level == levels - 1) {
        if (local_stats) ++local_stats->prefilter_evaluations;
        PrefilterResult prefilter;
        if (local_stats) {
          const auto prefilter_begin = std::chrono::steady_clock::now();
          prefilter = passes_valid_point_prefilter(
              current, prepared_points, base_x * pyramid_scale, base_y * pyramid_scale, required_valid);
          local_stats->prefilter_cpu_time_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - prefilter_begin).count();
        } else {
          prefilter = passes_valid_point_prefilter(
              current, prepared_points, base_x * pyramid_scale, base_y * pyramid_scale, required_valid);
        }
        if (!prefilter.passes) {
          if (local_stats) {
            ++local_stats->prefilter_rejections;
            if (prefilter.integral_rejected) ++local_stats->integral_prefilter_rejections;
          }
          return;
        }
      }
      if (local_stats) ++local_stats->full_score_evaluations;
      double score = 0.0;
      std::size_t score_points = 0;
      bool early_terminated = false;
      if (local_stats) {
        const auto score_begin = std::chrono::steady_clock::now();
        score = score_prepared_pose(
            current, prepared_points, base_x * pyramid_scale, base_y * pyramid_scale,
            params.polarity, threshold, required_valid,
            strict_detection ? 0.0 : params.greediness,
            &valid, &score_points, &early_terminated);
        local_stats->score_cpu_time_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - score_begin).count();
        local_stats->score_point_evaluations += score_points;
        if (early_terminated) ++local_stats->greediness_early_terminations;
      } else {
        score = score_prepared_pose(
            current, prepared_points, base_x * pyramid_scale, base_y * pyramid_scale,
            params.polarity, threshold, required_valid,
            strict_detection ? 0.0 : params.greediness,
            &valid, nullptr, nullptr);
      }
      if (valid < required_valid || score < threshold) return;
      MatchResult r;
      r.row = base_y;
      r.column = base_x;
      r.angle = normalize_angle(transform.angle);
      r.scale = transform.scale;
      r.score = score;
      r.valid_point_count = valid; r.model_point_count = model.points().size(); r.level_used = level;
      r.refined = level < levels - 1;
      r.valid_point_fraction = prepared_points.size() == 0 ? 0.0 :
          static_cast<double>(valid) / static_cast<double>(prepared_points.size());
      r.residual = 1.0 - score;
      r.confidence = score * std::sqrt(r.valid_point_fraction);
      output.push_back(r);
      if (local_stats) ++local_stats->accepted_candidates;
    };
    if (level == levels - 1) {
      const int min_x = static_cast<int>(std::floor(roi.x * pyramid_scale));
      const int min_y = static_cast<int>(std::floor(roi.y * pyramid_scale));
      const int max_x = static_cast<int>(std::ceil((roi.x + roi.width) * pyramid_scale));
      const int max_y = static_cast<int>(std::ceil((roi.y + roi.height) * pyramid_scale));
      const std::size_t row_count = static_cast<std::size_t>(std::max(0, max_y - min_y));
      const std::size_t job_count = transforms.size() * row_count;
      std::vector<std::vector<MatchResult>> worker_results;
      const std::size_t worker_count = params.num_threads > 0
          ? static_cast<std::size_t>(params.num_threads)
          : std::min<std::size_t>(16, static_cast<std::size_t>(std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency()));
      worker_results.resize(std::min(worker_count, std::max<std::size_t>(1, job_count)));
      std::vector<SearchStats> worker_stats(worker_results.size());
      parallel_ranges(job_count, params.num_threads, [&](std::size_t begin, std::size_t finish, std::size_t worker) {
        auto& output = worker_results[worker];
        SearchStats* local_stats = stats ? &worker_stats[worker] : nullptr;
        for (std::size_t job = begin; job < finish; ++job) {
          const std::size_t transform_index = row_count == 0 ? 0 : job / row_count;
          const int y = min_y + static_cast<int>(row_count == 0 ? 0 : job % row_count);
          for (int x = min_x; x < max_x; ++x)
            evaluate(output, local_stats, x / pyramid_scale, y / pyramid_scale, transform_index);
        }
      });
      for (auto& output : worker_results)
        next.insert(next.end(), output.begin(), output.end());
      if (stats) for (const auto& local : worker_stats) merge_stats(*stats, local);
    } else {
      const double refinement_step = static_cast<double>(1 << level);
      const std::vector<MatchResult> previous_candidates = candidates;
      candidates.clear();
      std::vector<std::vector<MatchResult>> worker_results;
      const std::size_t worker_count = params.num_threads > 0
          ? static_cast<std::size_t>(params.num_threads)
          : std::min<std::size_t>(16, static_cast<std::size_t>(std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency()));
      worker_results.resize(std::min(worker_count, std::max<std::size_t>(1, previous_candidates.size())));
      std::vector<SearchStats> worker_stats(worker_results.size());
      parallel_ranges(previous_candidates.size(), params.num_threads, [&](std::size_t begin, std::size_t finish, std::size_t worker) {
        auto& output = worker_results[worker];
        SearchStats* local_stats = stats ? &worker_stats[worker] : nullptr;
        for (std::size_t index = begin; index < finish; ++index) {
          const auto& previous = previous_candidates[index];
          const std::size_t transform_index = transform_index_for(previous.angle, previous.scale);
          for (int dy = -params.refinement_radius; dy <= params.refinement_radius; ++dy)
            for (int dx = -params.refinement_radius; dx <= params.refinement_radius; ++dx)
              evaluate(output, local_stats,
                       previous.column + dx * refinement_step,
                       previous.row + dy * refinement_step,
                       transform_index);
        }
      });
      for (auto& output : worker_results)
        next.insert(next.end(), output.begin(), output.end());
      if (stats) for (const auto& local : worker_stats) merge_stats(*stats, local);
    }
    const auto deterministic_order = [](const MatchResult& a, const MatchResult& b) {
      if (a.score != b.score) return a.score > b.score;
      if (a.row != b.row) return a.row < b.row;
      if (a.column != b.column) return a.column < b.column;
      if (a.angle != b.angle) return a.angle < b.angle;
      return a.scale < b.scale;
    };
    if (params.deterministic)
      std::stable_sort(next.begin(), next.end(), deterministic_order);
    else
      std::sort(next.begin(), next.end(), [](const MatchResult& a, const MatchResult& b) {
        return a.score > b.score;
      });
    if (stats) stats->peak_candidate_count = std::max(stats->peak_candidate_count, next.size());
    if (level > 0) {
      if (params.enable_candidate_clustering && !strict_detection) {
        if (stats) stats->peak_input_candidates += next.size();
        next = select_joint_peaks(
            next, params.peak_position_tolerance * static_cast<double>(1 << level),
            params.peak_angle_tolerance > 0.0 ? params.peak_angle_tolerance
                                              : params.angle_step * 0.49,
            params.peak_scale_tolerance > 0.0 ? params.peak_scale_tolerance
                                              : params.scale_step * 0.49,
            params.coarse_candidate_limit);
        if (stats) stats->peak_output_candidates += next.size();
      } else if (!strict_detection && next.size() > params.coarse_candidate_limit) {
        next.resize(params.coarse_candidate_limit);
      }
    }
    candidates = std::move(next);
  }
  if (stats) stats->candidate_search_time_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - search_begin).count();
  if ((params.enable_subpixel || params.enable_pose_refinement) && !candidates.empty()) {
    // The last search level is always level zero, hence its coordinates are
    // in the original image and can be refined without pyramid rescaling.
    const EdgeMap& finest = scene_levels.front();
    const auto refinement_begin = std::chrono::steady_clock::now();
    if (params.refinement_candidate_limit > 0 && !strict_detection) {
      if (stats) stats->peak_input_candidates += candidates.size();
      candidates = select_joint_peaks(
          candidates, params.peak_position_tolerance,
          params.peak_angle_tolerance > 0.0 ? params.peak_angle_tolerance
                                            : params.angle_step * 1.01,
          params.peak_scale_tolerance > 0.0 ? params.peak_scale_tolerance
                                            : params.scale_step * 1.01,
          params.refinement_candidate_limit, params.refinement_variants_per_peak);
      if (stats) stats->peak_output_candidates += candidates.size();
    }
    if (stats) stats->refinement_input_candidates = candidates.size();
    std::vector<MatchResult> refined_candidates;
    refined_candidates.reserve(candidates.size());
    const auto& finest_points = model.levels_soa().front();
    const std::size_t required_valid = std::max<std::size_t>(
        1, static_cast<std::size_t>(
               params.min_visible_fraction * static_cast<double>(finest_points.size())));
    for (const auto& candidate : candidates) {
      RefinementOutcome outcome = refine_continuous_pose(
          finest, finest_points, params, candidate);
      if (stats) stats->refinement_evaluations += outcome.evaluations;
      if (outcome.result.score >= params.min_score &&
          outcome.result.valid_point_count >= required_valid) {
        if (stats && outcome.result.refinement_converged) ++stats->refinement_converged;
        refined_candidates.push_back(outcome.result);
      } else {
        // Low-confidence optimization must not manufacture a match. Retain a
        // valid discrete pose as the documented safe fallback.
        if (candidate.score >= params.min_score && candidate.valid_point_count >= required_valid) {
          MatchResult fallback = candidate;
          fallback.refinement_converged = false;
          refined_candidates.push_back(fallback);
          if (stats) ++stats->refinement_fallbacks;
        }
      }
    }
    candidates = std::move(refined_candidates);
    if (stats) stats->continuous_refinement_time_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - refinement_begin).count();
  }
  // Every discrete result is verified against the complete finest-level
  // model. Reduced coarse points and early bounds are discovery mechanisms;
  // they are never allowed to create a final false positive.
  if (!candidates.empty() && !params.enable_subpixel && !params.enable_pose_refinement) {
    const auto& full_points = model.levels_soa().front();
    const std::size_t required_full = std::max<std::size_t>(
        1, static_cast<std::size_t>(
               params.min_visible_fraction * static_cast<double>(full_points.size())));
    std::vector<MatchResult> verified;
    verified.reserve(candidates.size());
    for (auto candidate : candidates) {
      const std::size_t transform_index = transform_index_for(candidate.angle, candidate.scale);
      const auto& transform = transforms[transform_index];
      const auto full_transformed = transform_model_points(
          full_points, transform.angle, transform.scale, full_points.size());
      std::size_t valid = 0;
      const double score = score_prepared_pose(
          scene_levels.front(), full_transformed, candidate.column, candidate.row,
          params.polarity, params.min_score, required_full, 0.0, &valid, nullptr, nullptr);
      if (valid >= required_full && score >= params.min_score) {
        candidate.score = score;
        candidate.valid_point_count = valid;
        candidate.valid_point_fraction = full_points.empty() ? 0.0 :
            static_cast<double>(valid) / static_cast<double>(full_points.size());
        candidate.residual = 1.0 - score;
        candidate.confidence = score * std::sqrt(candidate.valid_point_fraction);
        verified.push_back(candidate);
      }
    }
    candidates = std::move(verified);
  }
  if (stats) {
    stats->nms_input_candidates = candidates.size();
  }
  const auto nms_begin = std::chrono::steady_clock::now();
  auto results = non_max_suppression(model, std::move(candidates), params.max_overlap, params.num_matches);
  if (stats) {
    stats->nms_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - nms_begin).count();
    stats->nms_output_matches = results.size();
  }
  return results;
}

void draw_match_results(cv::Mat& image, const ShapeModel& model,
                        const std::vector<MatchResult>& results,
                        const cv::Scalar& color, int thickness) {
  if (image.empty()) throw EmptyImage("output image is empty");
  if (model.empty()) throw InvalidModel("model is empty");
  if (thickness <= 0) throw InvalidArgument("drawing thickness must be positive");
  const cv::Rect roi = model.roi();
  const cv::Point2f origin = model.origin();
  for (const auto& result : results) {
    const double radians = result.angle * CV_PI / 180.0;
    const float c = static_cast<float>(std::cos(radians));
    const float s = static_cast<float>(std::sin(radians));
    std::vector<cv::Point> polygon;
    polygon.reserve(4);
    for (const cv::Point2f corner : {
             cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y)),
             cv::Point2f(static_cast<float>(roi.x + roi.width), static_cast<float>(roi.y)),
             cv::Point2f(static_cast<float>(roi.x + roi.width), static_cast<float>(roi.y + roi.height)),
             cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y + roi.height))}) {
      const float x = static_cast<float>(result.scale) * (corner.x - origin.x);
      const float y = static_cast<float>(result.scale) * (corner.y - origin.y);
      polygon.emplace_back(cvRound(result.column + c * x - s * y),
                           cvRound(result.row + s * x + c * y));
    }
    cv::polylines(image, polygon, true, color, thickness, cv::LINE_AA);
    cv::drawMarker(image, cv::Point(cvRound(result.column), cvRound(result.row)),
                   color, cv::MARKER_CROSS, 10, thickness, cv::LINE_AA);
  }
}
}
