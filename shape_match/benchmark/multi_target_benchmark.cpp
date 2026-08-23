#include "openshape/openshape.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

double percentile(std::vector<double> samples, double fraction) {
  std::sort(samples.begin(), samples.end());
  const double index = fraction * static_cast<double>(samples.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(index);
  const std::size_t upper = std::min(lower + 1, samples.size() - 1);
  return samples[lower] + (index - static_cast<double>(lower)) *
                              (samples[upper] - samples[lower]);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: " << argv[0]
              << " TEMPLATE_IMAGE TARGET_IMAGE [ITERATIONS]\n";
    return 2;
  }
  const int iterations = argc == 4 ? std::max(1, std::stoi(argv[3])) : 30;
  const cv::Mat templ = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  const cv::Mat image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  if (templ.empty() || image.empty()) {
    std::cerr << "unable to load benchmark images\n";
    return 3;
  }

  openshape::ShapeModelParams model_params;
  model_params.model_version = 2;
  model_params.num_levels = 5;
  model_params.min_gradient_magnitude = 8.0;
  model_params.min_model_points = 20;
  model_params.max_model_points = 160;
  model_params.high_precision_point_spacing = 0.5;
  const auto model = openshape::create_shape_model(templ, model_params);

  openshape::SearchParams params;
  params.enable_pyramid_candidate_search = true;
  params.max_pyramid_levels = 5;
  params.num_levels = 5;
  params.angle_start = -60.0;
  params.angle_extent = 120.0;
  params.angle_step = 0.5;
  // Exercise the production multi-scale candidate path.  The grid contains
  // 0.80, 0.85, ..., 1.20 (nine discrete scales) and is propagated through
  // the scene/model pyramids before level-0 verification.
  params.scale_min = 0.8;
  params.scale_max = 1.2;
  params.scale_step = 0.05;
  params.min_score = 0.45;
  params.level_min_score_factor = 0.25;
  params.num_matches = 8;
  params.max_overlap = 0.35;
  params.strict_detection = false;
  params.num_threads = 4;

  openshape::SearchStats stats;
  std::vector<openshape::MatchResult> results;
  for (int iteration = 0; iteration < 5; ++iteration)
    results = openshape::find_shape_models(
        openshape::ImageView(image), model, params, &stats);

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    results = openshape::find_shape_models(
        openshape::ImageView(image), model, params, &stats);
    samples.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count());
  }

  std::cout << std::fixed << std::setprecision(3)
            << "iterations=" << iterations
            << " model_points=" << model.size()
            << " levels=" << model.levels().size()
            << " results=" << results.size()
            << " p50_ms=" << percentile(samples, 0.50)
            << " p95_ms=" << percentile(samples, 0.95)
            << " min_ms=" << *std::min_element(samples.begin(), samples.end())
            << " max_ms=" << *std::max_element(samples.begin(), samples.end())
            << " poses=" << stats.pose_evaluations
            << " score_points=" << stats.score_point_evaluations
            << " preprocessing_ms=" << stats.preprocessing_time_ms
            << " search_ms=" << stats.candidate_search_time_ms
            << " score_cpu_ms=" << stats.score_cpu_time_ms
            << " full_scores=" << stats.full_score_evaluations
            << " nms_input=" << stats.nms_input_candidates << '\n';
  for (std::size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    std::cout << index << " row=" << result.row
              << " col=" << result.column
              << " angle=" << result.angle
              << " scale=" << result.scale
              << " score=" << result.score
              << " valid=" << result.valid_point_count
              << '/' << result.model_point_count << '\n';
  }
  return 0;
}
