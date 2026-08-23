#include "openshape/openshape.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Distribution {
  double p50 = 0.0;
  double p95 = 0.0;
};

Distribution summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](double fraction) {
    const double index = fraction * static_cast<double>(samples.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(index);
    const std::size_t upper = std::min(lower + 1, samples.size() - 1);
    return samples[lower] + (index - static_cast<double>(lower)) *
                                (samples[upper] - samples[lower]);
  };
  return {percentile(0.50), percentile(0.95)};
}

bool same_results(const std::vector<openshape::MatchResult>& left,
                  const std::vector<openshape::MatchResult>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (left[i].row != right[i].row || left[i].column != right[i].column ||
        left[i].angle != right[i].angle || left[i].scale != right[i].scale ||
        left[i].score != right[i].score)
      return false;
  }
  return true;
}

cv::Mat make_template() {
  cv::Mat image = cv::Mat::zeros(96, 96, CV_8UC1);
  cv::rectangle(image, cv::Rect(32, 35, 32, 26), cv::Scalar(255), 2);
  cv::line(image, cv::Point(34, 58), cv::Point(61, 38), cv::Scalar(255), 2);
  return image;
}

cv::Mat make_scene() {
  cv::Mat image = make_template();
  cv::line(image, cv::Point(4, 8), cv::Point(18, 8), cv::Scalar(96), 1);
  cv::line(image, cv::Point(76, 82), cv::Point(91, 82), cv::Scalar(96), 1);
  return image;
}

struct RunResult {
  Distribution time;
  openshape::SearchStats stats;
  std::vector<openshape::MatchResult> matches;
};

RunResult measure(const openshape::EdgeMap& scene,
                  const openshape::ShapeModel& model,
                  const openshape::SearchParams& params,
                  int iterations) {
  RunResult result;
  for (int i = 0; i < 2; ++i)
    result.matches = openshape::find_shape_models_exhaustive(scene, model, params);
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    result.matches = openshape::find_shape_models_exhaustive(
        scene, model, params, &result.stats);
    samples.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count());
  }
  result.time = summarize(std::move(samples));
  return result;
}

void print_run(const char* name, const RunResult& run) {
  std::cout << name
            << "_p50_ms=" << run.time.p50
            << ' ' << name << "_p95_ms=" << run.time.p95
            << ' ' << name << "_poses=" << run.stats.pose_evaluations
            << ' ' << name << "_theoretical_poses=" << run.stats.theoretical_pose_count
            << ' ' << name << "_domain_poses=" << run.stats.domain_enumerated_pose_count
            << ' ' << name << "_domain_skipped=" << run.stats.domain_skipped_pose_count
            << ' ' << name << "_aabb_domain_skipped="
            << run.stats.aabb_domain_skipped_pose_count
            << ' ' << name << "_edge_domain_skipped="
            << run.stats.edge_domain_skipped_pose_count
            << ' ' << name << "_full_scores=" << run.stats.full_score_evaluations
            << ' ' << name << "_score_points=" << run.stats.score_point_evaluations
            << ' ' << name << "_safe_rejections=" << run.stats.safe_pruning_rejections
            << ' ' << name << "_edge_rejections=" << run.stats.edge_coverage_rejections
            << ' ' << name << "_bound_rejections=" << run.stats.safe_bound_terminations
            << ' ' << name << "_omitted_points=" << run.stats.omitted_point_evaluations;
}

}  // namespace

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::max(1, std::stoi(argv[1])) : 30;
  const bool wide_domain = argc > 2 &&
      (std::string(argv[2]) == "wide" ||
       std::string(argv[2]) == "wide-optimized");
  const bool optimized_only = argc > 2 &&
      std::string(argv[2]) == "wide-optimized";
  openshape::ShapeModelParams model_params;
  model_params.min_gradient_magnitude = 1.0;
  model_params.min_model_points = 4;
  model_params.max_model_points = 160;
  model_params.high_precision_point_spacing = 0.5;
  model_params.num_levels = 1;
  model_params.model_version = 2;
  const auto model = openshape::create_shape_model(make_template(), model_params);
  const auto scene = openshape::EdgeEngine::compute(make_scene(), model.params(), true);

  openshape::SearchParams baseline_params;
  baseline_params.angle_start = -5.0;
  baseline_params.angle_extent = 10.0;
  baseline_params.angle_step = 5.0;
  baseline_params.min_score = 0.75;
  baseline_params.level_min_score_factor = 1.0;
  baseline_params.num_matches = 0;
  if (wide_domain) {
    baseline_params.angle_start = -15.0;
    baseline_params.angle_extent = 30.0;
    baseline_params.angle_step = 0.1;
    baseline_params.scale_min = 0.8;
    baseline_params.scale_max = 1.2;
    baseline_params.scale_step = 0.05;
    baseline_params.search_roi = cv::Rect(40, 40, 17, 17);
  }
  openshape::SearchParams optimized_params = baseline_params;
  optimized_params.enable_safe_pruning = true;

  if (optimized_only) {
    const RunResult optimized = measure(scene, model, optimized_params, iterations);
    std::cout << std::fixed << std::setprecision(3)
              << "scenario=wide-domain-pruning iterations=" << iterations
              << " model_points=" << model.size() << ' ';
    print_run("optimized", optimized);
    std::cout << '\n';
    return 0;
  }

  const RunResult baseline = measure(scene, model, baseline_params, iterations);
  const RunResult optimized = measure(scene, model, optimized_params, iterations);
  const bool equivalent = same_results(baseline.matches, optimized.matches);
  const double point_reduction = baseline.stats.score_point_evaluations == 0 ? 0.0 :
      1.0 - static_cast<double>(optimized.stats.score_point_evaluations) /
                static_cast<double>(baseline.stats.score_point_evaluations);
  const double speedup = optimized.time.p50 > 0.0
      ? baseline.time.p50 / optimized.time.p50 : 0.0;

  std::cout << std::fixed << std::setprecision(3)
            << "scenario=" << (wide_domain ? "wide-domain-pruning" : "synthetic-pruning")
            << " iterations=" << iterations
            << " model_points=" << model.size() << ' ';
  print_run("baseline", baseline);
  std::cout << ' ';
  print_run("optimized", optimized);
  std::cout << " point_reduction=" << point_reduction
            << " p50_speedup=" << speedup
            << " equivalent=" << (equivalent ? 1 : 0) << '\n';
  return equivalent ? 0 : 2;
}
