#include "openshape/matcher/pruning.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace openshape {
namespace {

double polarity_similarity(float dot, PolarityMode polarity) {
  switch (polarity) {
    case PolarityMode::Same: return std::max(0.0f, dot);
    case PolarityMode::Inverted: return std::max(0.0f, -dot);
    case PolarityMode::LocalEither: return std::abs(dot);
    case PolarityMode::GlobalEither: break;
  }
  return 0.0;
}

}  // namespace

std::vector<std::size_t> build_pruning_point_order(
    const ModelLevelSoA& points) {
  std::vector<std::size_t> sorted(points.size());
  std::iota(sorted.begin(), sorted.end(), std::size_t{0});
  if (points.size() < 2) return sorted;
  const bool uniform_weight = std::all_of(
      points.weight.begin() + 1, points.weight.end(),
      [&](float weight) { return weight == points.weight.front(); });
  if (uniform_weight) return sorted;
  std::stable_sort(sorted.begin(), sorted.end(), [&](std::size_t left,
                                                     std::size_t right) {
    if (points.weight[left] != points.weight[right])
      return points.weight[left] > points.weight[right];
    return left < right;
  });

  std::vector<std::size_t> order;
  order.reserve(points.size());

  std::size_t group_begin = 0;
  while (group_begin < sorted.size()) {
    std::size_t group_end = group_begin + 1;
    while (group_end < sorted.size() &&
           points.weight[sorted[group_end]] == points.weight[sorted[group_begin]])
      ++group_end;
    constexpr std::size_t lanes = 4;
    const std::size_t group_size = group_end - group_begin;
    const std::size_t lane_size = (group_size + lanes - 1) / lanes;
    for (std::size_t offset = 0; offset < lane_size; ++offset) {
      for (std::size_t lane = 0; lane < lanes; ++lane) {
        const std::size_t cursor = group_begin + lane * lane_size + offset;
        if (cursor < group_end) order.push_back(sorted[cursor]);
      }
    }
    group_begin = group_end;
  }
  return order;
}

PrunableScoreResult score_pose_precise_prunable(
    const EdgeMap& scene, const ModelLevelSoA& points,
    double column, double row, double angle_degrees, double scale,
    PolarityMode polarity, double edge_distance_sigma,
    double minimum_score, std::size_t required_visible_count,
    bool enable_pruning,
    const std::vector<std::size_t>* point_order,
    PrunableScoreWorkspace* workspace) {
  if (!std::isfinite(scale) || scale <= 0.0)
    throw InvalidArgument("score scale must be positive and finite");
  if (!std::isfinite(edge_distance_sigma) || edge_distance_sigma <= 0.0)
    throw InvalidArgument("edge distance sigma must be positive and finite");
  if (!std::isfinite(minimum_score) || minimum_score < 0.0 || minimum_score > 1.0)
    throw InvalidArgument("minimum pruning score must be in [0, 1]");

  PrunableScoreResult result;
  if (points.empty() || scene.edges.empty()) {
    result.score_upper_bound = 0.0;
    return result;
  }
  if (scene.gx.empty() || scene.gy.empty() || scene.soft_edge_response.empty() ||
      scene.normalized_magnitude.empty())
    throw InvalidArgument("precise scoring requires gradients and soft edge response");

  const int width = scene.edges.cols;
  const int height = scene.edges.rows;
  const double radians = angle_degrees * CV_PI / 180.0;
  const float c = static_cast<float>(std::cos(radians));
  const float s = static_cast<float>(std::sin(radians));
  const float angle = static_cast<float>(radians);
  double weighted = 0.0;
  double inverted_weighted = 0.0;
  double total_weight = 0.0;
  for (float weight : points.weight)
    total_weight += std::max(0.0f, weight);
  double remaining_weight = total_weight;
  const std::size_t bound_interval = 1;

  if (point_order != nullptr && point_order->size() != points.size())
    throw InvalidArgument("pruning point order size does not match model points");
  if (point_order != nullptr && workspace != nullptr) {
    workspace->weighted_contribution.resize(points.size());
    if (polarity == PolarityMode::GlobalEither)
      workspace->inverted_contribution.resize(points.size());
  }
  for (std::size_t evaluation_index = 0;
       evaluation_index < points.size(); ++evaluation_index) {
    const std::size_t index = point_order == nullptr
        ? evaluation_index : (*point_order)[evaluation_index];
    if (index >= points.size())
      throw InvalidArgument("pruning point order contains an invalid index");
    const double point_weight = std::max(0.0f, points.weight[index]);
    if (point_order != nullptr && workspace != nullptr) {
      workspace->weighted_contribution[index] = 0.0;
      if (polarity == PolarityMode::GlobalEither)
        workspace->inverted_contribution[index] = 0.0;
    }
    remaining_weight = std::max(0.0, remaining_weight - point_weight);
    ++result.point_evaluations;
    const double x = column + scale *
        (c * points.relative_x[index] - s * points.relative_y[index]);
    const double y = row + scale *
        (s * points.relative_x[index] + c * points.relative_y[index]);
    if (x >= 0.0 && y >= 0.0 && x <= static_cast<double>(width - 1) &&
        y <= static_cast<double>(height - 1)) {
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
      const float gx = w00 * scene.gx.at<float>(y0, x0) +
                       w10 * scene.gx.at<float>(y0, x1) +
                       w01 * scene.gx.at<float>(y1, x0) +
                       w11 * scene.gx.at<float>(y1, x1);
      const float gy = w00 * scene.gy.at<float>(y0, x0) +
                       w10 * scene.gy.at<float>(y0, x1) +
                       w01 * scene.gy.at<float>(y1, x0) +
                       w11 * scene.gy.at<float>(y1, x1);
      const float magnitude = std::hypot(gx, gy);
      const float base_edge = w00 * scene.soft_edge_response.at<float>(y0, x0) +
                              w10 * scene.soft_edge_response.at<float>(y0, x1) +
                              w01 * scene.soft_edge_response.at<float>(y1, x0) +
                              w11 * scene.soft_edge_response.at<float>(y1, x1);
      const float local_strength =
          w00 * scene.normalized_magnitude.at<float>(y0, x0) +
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
            fitted_edge = std::max(
                fitted_edge, static_cast<float>(std::exp(-d2 / (2.0 * sigma2))));
          }
        }
      }
      const float edge_base = std::max(std::clamp(base_edge, 0.0f, 1.0f), fitted_edge);
      const float edge =
          std::pow(edge_base, static_cast<float>(1.0 / edge_distance_sigma)) *
          (0.2f + 0.8f * std::clamp(local_strength, 0.0f, 1.0f));
      if (edge >= std::exp(-2.0f) && magnitude > 1e-6f) {
        const float model_orientation = points.orientation[index] + angle;
        const float dot = (std::cos(model_orientation) * gx +
                           std::sin(model_orientation) * gy) / magnitude;
        if (polarity == PolarityMode::GlobalEither) {
          const double positive = point_weight * edge * std::max(0.0f, dot);
          const double negative = point_weight * edge * std::max(0.0f, -dot);
          weighted += positive;
          inverted_weighted += negative;
          if (point_order != nullptr && workspace != nullptr) {
            workspace->weighted_contribution[index] = positive;
            workspace->inverted_contribution[index] = negative;
          }
        } else {
          const double contribution =
              point_weight * edge * polarity_similarity(dot, polarity);
          weighted += contribution;
          if (point_order != nullptr && workspace != nullptr)
            workspace->weighted_contribution[index] = contribution;
        }
        ++result.valid_point_count;
      }
    }

    const std::size_t remaining_points = points.size() - evaluation_index - 1;
    if (enable_pruning &&
        ((evaluation_index + 1) % bound_interval == 0 || remaining_points == 0)) {
      const double best_weighted = polarity == PolarityMode::GlobalEither
          ? std::max(weighted, inverted_weighted) : weighted;
      result.score_upper_bound = total_weight > 0.0
          ? std::clamp((best_weighted + remaining_weight) / total_weight, 0.0, 1.0)
          : 0.0;
      if (required_visible_count > 0 &&
          result.valid_point_count + remaining_points < required_visible_count) {
        result.reason = PruningReason::VisibleUpperBound;
        return result;
      }
      if (result.score_upper_bound + 1e-12 < minimum_score) {
        result.reason = PruningReason::ScoreUpperBound;
        return result;
      }
    }
  }

  if (point_order != nullptr && workspace != nullptr) {
    weighted = 0.0;
    inverted_weighted = 0.0;
    for (std::size_t index = 0; index < points.size(); ++index) {
      weighted += workspace->weighted_contribution[index];
      if (polarity == PolarityMode::GlobalEither)
        inverted_weighted += workspace->inverted_contribution[index];
    }
  }
  const double best_weighted = polarity == PolarityMode::GlobalEither
      ? std::max(weighted, inverted_weighted) : weighted;
  result.score = total_weight > 0.0
      ? std::clamp(best_weighted / total_weight, 0.0, 1.0) : 0.0;
  result.score_upper_bound = result.score;
  return result;
}

}  // namespace openshape
