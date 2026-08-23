#include "openshape/matcher/exhaustive_matcher.hpp"
#include "openshape/edge/edge_engine.hpp"
#include "subpixel_fit.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>

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

std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& input_scene, const ShapeModel& model,
    const SearchParams& params, SearchStats* stats) {
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
  std::vector<MatchResult> all;
  const auto begin = std::chrono::steady_clock::now();
  for (double scale : scales) {
    for (double angle : angles) {
      for (int row = roi.y; row < roi.y + roi.height; row +=
           std::max(1, static_cast<int>(std::lround(params.exhaustive_translation_step)))) {
        for (int column = roi.x; column < roi.x + roi.width; column +=
             std::max(1, static_cast<int>(std::lround(params.exhaustive_translation_step)))) {
          if (stats) { ++stats->pose_evaluations; ++stats->full_score_evaluations; }
          std::size_t valid = 0;
          const double score = score_pose_precise(scene, points, column, row, angle, scale,
                                                  params.polarity, params.edge_distance_sigma, &valid);
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
    }
  }
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
}
