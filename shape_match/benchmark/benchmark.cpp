#include "openshape/openshape.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
struct Distribution {
  double p50 = 0.0, p95 = 0.0, p99 = 0.0;
};

Distribution summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](double p) {
    const double index = p * static_cast<double>(samples.size() - 1);
    const auto lower = static_cast<std::size_t>(index);
    const auto upper = std::min(lower + 1, samples.size() - 1);
    const double fraction = index - static_cast<double>(lower);
    return samples[lower] + fraction * (samples[upper] - samples[lower]);
  };
  return {percentile(0.50), percentile(0.95), percentile(0.99)};
}

template <typename Function>
Distribution measure(int warmup, int iterations, Function&& operation) {
  for (int i = 0; i < warmup; ++i) operation();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    operation();
    samples.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count());
  }
  return summarize(std::move(samples));
}

bool same_result(const std::vector<openshape::MatchResult>& a,
                 const std::vector<openshape::MatchResult>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].row != b[i].row || a[i].column != b[i].column ||
        a[i].angle != b[i].angle || a[i].scale != b[i].scale ||
        a[i].score != b[i].score)
      return false;
  }
  return true;
}

cv::Mat make_scene(int width, int height, int targets, bool clutter, double target_scale) {
  cv::Mat scene = cv::Mat::zeros(height, width, CV_8UC1);
  for (int target = 0; target < targets; ++target) {
    const int x = width * (target + 1) / (targets + 1);
    const int y = height * (target % 2 + 1) / 3;
    cv::circle(scene, {x, y}, cvRound(35.0 * target_scale), cv::Scalar(255), 3);
  }
  if (clutter) {
    for (int x = 40; x < width; x += 80)
      cv::line(scene, {x, 0}, {x, height - 1}, cv::Scalar(96), 1);
    for (int y = 50; y < height; y += 100)
      cv::line(scene, {0, y}, {width - 1, y}, cv::Scalar(96), 1);
  }
  return scene;
}

void print_distribution(const char* name, const Distribution& value) {
  std::cout << ' ' << name << "_p50_ms=" << value.p50
            << ' ' << name << "_p95_ms=" << value.p95
            << ' ' << name << "_p99_ms=" << value.p99;
}

long peak_rss_kb() {
#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmHWM:") {
      long value = 0;
      status >> value;
      return value;
    }
    std::string rest;
    std::getline(status, rest);
  }
#endif
  return -1;
}

bool run_case(const std::string& name, int width, int height, int threads,
              int targets, bool clutter, double angle_start, double angle_extent,
              double angle_step, double target_scale, double scale_min,
              double scale_max, double scale_step, int iterations,
              const openshape::ShapeModel& model, bool deterministic = true,
              bool optimized_pipeline = false) {
  const cv::Mat scene = make_scene(width, height, targets, clutter, target_scale);
  openshape::SearchParams filtered;
  filtered.min_score = 0.5;
  filtered.num_matches = 1;
  filtered.num_threads = threads;
  filtered.deterministic = deterministic;
  if (optimized_pipeline) filtered.enable_fast_pipeline();
  filtered.angle_start = angle_start;
  filtered.angle_extent = angle_extent;
  filtered.angle_step = angle_step;
  filtered.scale_min = scale_min;
  filtered.scale_max = scale_max;
  filtered.scale_step = scale_step;
  openshape::SearchParams unfiltered = filtered;
  unfiltered.enable_coarse_prefilter = false;
  const auto prepared = openshape::EdgePyramid::build(
      openshape::ImageView(scene), model.params(), static_cast<int>(model.levels().size()));

  std::vector<openshape::MatchResult> filtered_results;
  std::vector<openshape::MatchResult> unfiltered_results;
  openshape::SearchStats filtered_stats;
  openshape::SearchStats unfiltered_stats;
  const Distribution full = measure(2, iterations, [&] {
    filtered_results = openshape::find_shape_models(scene, model, filtered);
  });
  const Distribution prepared_filtered = measure(2, iterations, [&] {
    filtered_results = openshape::find_shape_models(prepared, model, filtered, &filtered_stats);
  });
  const Distribution prepared_unfiltered = measure(1, iterations, [&] {
    unfiltered_results = openshape::find_shape_models(prepared, model, unfiltered, &unfiltered_stats);
  });
  const bool equivalent = same_result(filtered_results, unfiltered_results);
  const double rejection_ratio = filtered_stats.prefilter_evaluations == 0 ? 0.0 :
      static_cast<double>(filtered_stats.prefilter_rejections) /
      static_cast<double>(filtered_stats.prefilter_evaluations);
  std::cout << std::fixed << std::setprecision(3)
            << "scenario=" << name << " image=" << width << 'x' << height
            << " threads=" << threads << " targets=" << targets
            << " deterministic=" << (deterministic ? 1 : 0)
            << " optimized=" << (optimized_pipeline ? 1 : 0)
            << " clutter=" << (clutter ? 1 : 0)
            << " angles=" << angle_start << ':' << angle_extent << ':' << angle_step
            << " scales=" << scale_min << ':' << scale_max << ':' << scale_step
            << " transforms=" << filtered_stats.transform_candidate_count
            << " target_scale=" << target_scale
            << " iterations=" << iterations;
  print_distribution("full", full);
  print_distribution("filtered", prepared_filtered);
  print_distribution("unfiltered", prepared_unfiltered);
  std::cout << " poses=" << filtered_stats.pose_evaluations
            << " prefilter_rejected=" << filtered_stats.prefilter_rejections
            << " integral_rejected=" << filtered_stats.integral_prefilter_rejections
            << " rejection_ratio=" << rejection_ratio
            << " full_scores=" << filtered_stats.full_score_evaluations
            << " unfiltered_full_scores=" << unfiltered_stats.full_score_evaluations
            << " transform_prepare_ms=" << filtered_stats.transform_preparation_time_ms
            << " prefilter_cpu_ms=" << filtered_stats.prefilter_cpu_time_ms
            << " score_cpu_ms=" << filtered_stats.score_cpu_time_ms
            << " refine_ms=" << filtered_stats.continuous_refinement_time_ms
            << " nms_ms=" << filtered_stats.nms_time_ms
            << " peak_candidates=" << filtered_stats.peak_candidate_count
            << " accepted_candidates=" << filtered_stats.accepted_candidates
            << " score_points=" << filtered_stats.score_point_evaluations
            << " early_terminated=" << filtered_stats.greediness_early_terminations
            << " peaks_in=" << filtered_stats.peak_input_candidates
            << " peaks_out=" << filtered_stats.peak_output_candidates
            << " refine_candidates=" << filtered_stats.refinement_input_candidates
            << " peak_rss_kb=" << peak_rss_kb()
            << " detected_scale=" << (filtered_results.empty() ? 0.0 : filtered_results.front().scale)
            << " equivalent=" << (equivalent ? 1 : 0) << '\n';
  return equivalent;
}
}

int main(int argc, char** argv) {
  int iterations = 30;
  if (argc > 1) iterations = std::max(1, std::stoi(argv[1]));
  cv::Mat templ = cv::Mat::zeros(120, 120, CV_8UC1);
  cv::circle(templ, {60, 60}, 35, cv::Scalar(255), 3);
  openshape::ShapeModelParams model_params;
  model_params.min_gradient_magnitude = 1;
  model_params.min_model_points = 4;
  const auto model = openshape::create_shape_model(templ, model_params);

  std::cout << "cpu=" << openshape::cpu_features_string()
            << " hardware_threads=" << std::thread::hardware_concurrency()
            << " compiler=" << __VERSION__
            << " opencv=" << CV_VERSION
#if defined(NDEBUG)
            << " build_type=release"
#else
            << " build_type=debug"
#endif
            << " direct_score_kernel=" << openshape::active_score_kernel_name()
            << " matcher_score_kernel=" << openshape::active_matcher_score_kernel_name()
            << " model_points=" << model.size()
            << " levels=" << model.levels().size() << '\n';
  bool success = true;
  for (const auto& size : {cv::Size(640, 400), cv::Size(1280, 1024), cv::Size(2000, 2000)}) {
    success &= run_case("sparse", size.width, size.height, 1, 1, false,
                        0, 0, 1, 1.0, 1.0, 1.0, 0.05, iterations, model);
    success &= run_case("sparse", size.width, size.height, 4, 1, false,
                        0, 0, 1, 1.0, 1.0, 1.0, 0.05, iterations, model);
    success &= run_case("sparse", size.width, size.height, 8, 1, false,
                        0, 0, 1, 1.0, 1.0, 1.0, 0.05, iterations, model);
    for (int threads : {1, 4, 8})
      success &= run_case("sparse-fast", size.width, size.height, threads, 1, false,
                          0, 0, 1, 1.0, 1.0, 1.0, 0.05, iterations, model, false);
  }
  success &= run_case("multi-angle", 640, 400, 4, 2, false,
                      -20, 40, 5, 1.0, 1.0, 1.0, 0.05, iterations, model);
  success &= run_case("scale", 640, 400, 4, 1, false,
                      0, 0, 1, 1.25, 0.75, 1.25, 0.125, iterations, model);
  success &= run_case("angle-scale", 640, 400, 4, 2, true,
                      -10, 20, 5, 1.25, 0.75, 1.25, 0.125, iterations, model);
  success &= run_case("angle-scale-optimized", 640, 400, 4, 2, true,
                      -10, 20, 5, 1.25, 0.75, 1.25, 0.125, iterations, model,
                      true, true);
  success &= run_case("clutter", 640, 400, 4, 2, true,
                      -10, 20, 5, 1.0, 1.0, 1.0, 0.05, iterations, model);
  return success ? 0 : 2;
}
