#include "openshape/openshape.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <string>

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

cv::Mat make_template() {
  cv::Mat image = cv::Mat::zeros(72, 72, CV_8UC1);
  cv::rectangle(image, cv::Rect(18, 22, 36, 24), cv::Scalar(255), 2);
  cv::line(image, cv::Point(18, 34), cv::Point(36, 52), cv::Scalar(255), 2);
  return image;
}

openshape::ShapeModel make_model(const cv::Mat& image) {
  openshape::ShapeModelParams params;
  params.min_gradient_magnitude = 1;
  params.min_model_points = 8;
  params.max_model_points = 250;
  params.min_point_distance = 2;
  params.num_levels = 1;
  return openshape::create_shape_model(image, params);
}

openshape::SearchParams search_params() {
  openshape::SearchParams params;
  params.num_levels = 1;
  params.min_score = 0.45;
  params.max_overlap = 0.25;
  params.num_matches = 1;
  return params;
}

void draw_rotated_target(cv::Mat& scene, const cv::Point2f& center, float angle) {
  const cv::RotatedRect rectangle(center, cv::Size2f(36, 24), angle);
  cv::Point2f points[4];
  rectangle.points(points);
  for (int i = 0; i < 4; ++i)
    cv::line(scene, points[i], points[(i + 1) % 4], cv::Scalar(255), 2);
}

void place_transformed_template(const cv::Mat& templ, cv::Mat& scene,
                                const cv::Point2f& center, double angle, double scale) {
  const cv::Point2f source_center(templ.cols * 0.5f, templ.rows * 0.5f);
  cv::Mat transform = cv::getRotationMatrix2D(source_center, -angle, scale);
  transform.at<double>(0, 2) += center.x - source_center.x;
  transform.at<double>(1, 2) += center.y - source_center.y;
  cv::warpAffine(templ, scene, transform, scene.size(), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar(0));
}

void test_translation_and_noise() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat scene = cv::Mat::zeros(180, 220, CV_8UC1);
  templ(cv::Rect(0, 0, templ.cols, templ.rows)).copyTo(scene(cv::Rect(74, 58, templ.cols, templ.rows)));
  cv::RNG rng(12345);
  cv::Mat noise(scene.size(), CV_8UC1);
  rng.fill(noise, cv::RNG::UNIFORM, 0, 8);
  scene += noise;

  auto params = search_params();
  const auto results = openshape::find_shape_models(scene, model, params);
  check(!results.empty(), "translation with deterministic noise is detected");
  if (!results.empty()) {
    check(std::abs(results.front().column - 110.0) <= 4.0, "translation column is within tolerance");
    check(std::abs(results.front().row - 94.0) <= 4.0, "translation row is within tolerance");
    check(results.front().score >= params.min_score, "translation score meets threshold");
  }
}

void test_rotation() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat scene = cv::Mat::zeros(180, 220, CV_8UC1);
  draw_rotated_target(scene, cv::Point2f(110, 92), 14.0f);
  cv::line(scene, cv::Point(94, 92), cv::Point(110, 108), cv::Scalar(255), 2);

  auto params = search_params();
  params.angle_start = -25;
  params.angle_extent = 50;
  params.angle_step = 2;
  const auto results = openshape::find_shape_models(scene, model, params);
  check(!results.empty(), "rotated target is detected");
  if (!results.empty()) {
    check(std::abs(results.front().angle - 14.0) <= 5.0 || std::abs(results.front().angle + 14.0) <= 5.0,
          "rotation angle is within tolerance");
    check(results.front().score >= params.min_score, "rotation score meets threshold");
  }
}

void test_multiple_and_negative() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat scene = cv::Mat::zeros(180, 300, CV_8UC1);
  templ.copyTo(scene(cv::Rect(20, 50, templ.cols, templ.rows)));
  templ.copyTo(scene(cv::Rect(190, 90, templ.cols, templ.rows)));
  auto params = search_params();
  params.num_matches = 0;
  const auto results = openshape::find_shape_models(scene, model, params);
  check(results.size() >= 2, "multiple instances produce at least two detections");

  cv::Mat background = cv::Mat::zeros(scene.size(), CV_8UC1);
  params.num_matches = 5;
  params.min_score = 0.7;
  const auto negative = openshape::find_shape_models(background, model, params);
  check(negative.empty(), "pure background produces no false positive");
}


void test_scale_search() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  for (double expected_scale : {0.75, 1.25}) {
    cv::Mat scene = cv::Mat::zeros(190, 230, CV_8UC1);
    const cv::Point2f center(116, 94);
    place_transformed_template(templ, scene, center, 0.0, expected_scale);
    auto params = search_params();
    params.min_score = 0.35;
    params.scale_min = 0.65;
    params.scale_max = 1.35;
    params.scale_step = 0.10;
    params.search_roi = cv::Rect(104, 82, 25, 25);
    const auto results = openshape::find_shape_models(scene, model, params);
    check(!results.empty(), "scaled target is detected");
    if (!results.empty()) {
      check(std::abs(results.front().column - center.x) <= 3.0 &&
                std::abs(results.front().row - center.y) <= 3.0,
            "scaled target position is within tolerance");
      check(std::abs(results.front().scale - expected_scale) <= params.scale_step + 1e-9,
            "detected scale is within one discrete search step");
    }
  }
}

void test_multiscale_range() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);

  for (double expected_scale : {0.8, 1.2}) {
    cv::Mat scene = cv::Mat::zeros(220, 260, CV_8UC1);
    const cv::Point2f center(130, 110);
    place_transformed_template(templ, scene, center, 0.0, expected_scale);
    auto params = search_params();
    params.scale_min = 0.8;
    params.scale_max = 1.2;
    params.scale_step = 0.05;
    params.min_score = 0.30;
    params.search_roi = cv::Rect(112, 92, 36, 36);
    openshape::SearchStats stats;
    const auto results = openshape::find_shape_models(scene, model, params, &stats);
    check(!results.empty(), "multi-scale search detects boundary scale target");
    check(stats.scale_candidate_count == 9 && stats.transform_candidate_count ==
              stats.angle_candidate_count * 9,
          "multi-scale search enumerates the complete 0.8-1.2 grid");
    if (!results.empty()) {
      check(std::abs(results.front().column - center.x) <= 4.0 &&
                std::abs(results.front().row - center.y) <= 4.0,
            "multi-scale target position is within tolerance");
      check(std::abs(results.front().scale - expected_scale) <= params.scale_step + 1e-9,
            "multi-scale result preserves the selected scale");
    }
  }
}

void test_combined_rotation_and_scale() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat scene = cv::Mat::zeros(200, 240, CV_8UC1);
  const cv::Point2f center(122, 98);
  place_transformed_template(templ, scene, center, 14.0, 1.25);
  auto params = search_params();
  params.min_score = 0.35;
  params.angle_start = 8.0;
  params.angle_extent = 12.0;
  params.angle_step = 2.0;
  params.scale_min = 1.0;
  params.scale_max = 1.5;
  params.scale_step = 0.125;
  params.search_roi = cv::Rect(110, 86, 25, 25);
  const auto results = openshape::find_shape_models(scene, model, params);
  check(!results.empty(), "combined rotation and scale target is detected");
  if (!results.empty()) {
    check(std::abs(results.front().column - center.x) <= 3.0 &&
              std::abs(results.front().row - center.y) <= 3.0,
          "combined transform position is within tolerance");
    check(std::abs(results.front().angle - 14.0) <= params.angle_step + 1e-9,
          "combined transform angle is within one discrete step");
    check(std::abs(results.front().scale - 1.25) <= params.scale_step + 1e-9,
          "combined transform scale is within one discrete step");
  }
}

void test_multiple_scales_and_negative() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat scene = cv::Mat::zeros(190, 330, CV_8UC1);
  place_transformed_template(templ, scene, cv::Point2f(82, 92), 0.0, 0.75);
  cv::Mat second = cv::Mat::zeros(scene.size(), scene.type());
  place_transformed_template(templ, second, cv::Point2f(248, 96), 0.0, 1.25);
  cv::max(scene, second, scene);
  auto params = search_params();
  params.min_score = 0.35;
  params.num_matches = 0;
  params.max_overlap = 0.2;
  params.scale_min = 0.75;
  params.scale_max = 1.25;
  params.scale_step = 0.25;
  const auto results = openshape::find_shape_models(scene, model, params);
  bool found_small = false, found_large = false;
  for (const auto& result : results) {
    if (std::abs(result.column - 82.0) <= 4.0 && std::abs(result.row - 92.0) <= 4.0 &&
        std::abs(result.scale - 0.75) <= params.scale_step)
      found_small = true;
    if (std::abs(result.column - 248.0) <= 4.0 && std::abs(result.row - 96.0) <= 4.0 &&
        std::abs(result.scale - 1.25) <= params.scale_step)
      found_large = true;
  }
  check(found_small && found_large, "multiple instances at different scales are detected");

  cv::Mat background = cv::Mat::zeros(scene.size(), CV_8UC1);
  params.num_matches = 5;
  params.min_score = 0.7;
  check(openshape::find_shape_models(background, model, params).empty(),
        "scale search keeps a pure background negative");
}

void test_subpixel_refinement() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat scene = cv::Mat::zeros(180, 220, CV_8UC1);
  const cv::Point2f center(110.35f, 93.65f);
  place_transformed_template(templ, scene, center, 0.0, 1.0);
  auto params = search_params();
  params.min_score = 0.25;
  params.search_roi = cv::Rect(102, 85, 18, 18);
  const auto discrete = openshape::find_shape_models(scene, model, params);
  params.enable_subpixel = true;
  const auto refined = openshape::find_shape_models(scene, model, params);
  check(!discrete.empty() && !refined.empty(), "subpixel target is detected");
  if (!discrete.empty() && !refined.empty()) {
    const double discrete_error = std::hypot(discrete.front().column - center.x,
                                             discrete.front().row - center.y);
    const double refined_error = std::hypot(refined.front().column - center.x,
                                            refined.front().row - center.y);
    check(refined.front().refined, "subpixel result is marked refined");
    check(std::isfinite(refined.front().column) && std::isfinite(refined.front().row),
          "subpixel result coordinates are finite");
    check(refined_error <= discrete_error + 0.25,
          "subpixel refinement does not worsen the discrete position materially");
  }
}

void test_joint_pose_refinement() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat scene = cv::Mat::zeros(210, 250, CV_8UC1);
  const cv::Point2f center(124.35f, 101.65f);
  const double expected_angle = 7.35;
  const double expected_scale = 1.073;
  place_transformed_template(templ, scene, center, expected_angle, expected_scale);
  auto params = search_params();
  params.min_score = 0.25;
  params.angle_start = 5.0;
  params.angle_extent = 5.0;
  params.angle_step = 1.0;
  params.scale_min = 1.0;
  params.scale_max = 1.12;
  params.scale_step = 0.03;
  params.search_roi = cv::Rect(117, 94, 16, 16);
  const auto discrete = openshape::find_shape_models(scene, model, params);
  params.enable_pose_refinement = true;
  params.strict_detection = false;
  params.max_refinement_iterations = 20;
  openshape::SearchStats stats;
  const auto refined = openshape::find_shape_models(scene, model, params, &stats);
  check(!discrete.empty() && !refined.empty(), "fractional angle/scale target is detected");
  if (!discrete.empty() && !refined.empty()) {
    const auto pose_error = [&](const openshape::MatchResult& result) {
      return std::hypot(result.column - center.x, result.row - center.y) +
             0.1 * std::abs(result.angle - expected_angle) +
             10.0 * std::abs(result.scale - expected_scale);
    };
    check(refined.front().refined && refined.front().valid_point_fraction > 0.0 &&
              refined.front().confidence > 0.0 && refined.front().residual >= 0.0,
          "joint refinement reports convergence quality fields");
    check(pose_error(refined.front()) <= pose_error(discrete.front()) + 0.15,
          "joint refinement does not materially worsen the four-dimensional pose");
    check(stats.refinement_evaluations > 0 && stats.continuous_refinement_time_ms >= 0.0,
          "joint refinement statistics report evaluations and timing");
    check(stats.refinement_input_candidates > 0 &&
              stats.refinement_input_candidates <= params.refinement_candidate_limit &&
              stats.peak_input_candidates >= stats.peak_output_candidates,
          "joint peak clustering bounds the number of continuously refined candidates");
  }
}

void test_polarity_and_occlusion() {
  const cv::Mat templ = make_template();
  const auto model = make_model(templ);
  cv::Mat inverted_template;
  cv::bitwise_not(templ, inverted_template);
  cv::Mat inverted_scene(180, 220, CV_8UC1, cv::Scalar(255));
  inverted_template.copyTo(inverted_scene(cv::Rect(74, 58, templ.cols, templ.rows)));
  auto params = search_params();
  params.search_roi = cv::Rect(104, 88, 14, 14);
  params.min_score = 0.35;
  params.polarity = openshape::PolarityMode::GlobalEither;
  const auto inverted = openshape::find_shape_models(inverted_scene, model, params);
  check(!inverted.empty(), "global polarity mode detects a contrast-inverted target");

  cv::Mat occluded = cv::Mat::zeros(180, 220, CV_8UC1);
  templ.copyTo(occluded(cv::Rect(74, 58, templ.cols, templ.rows)));
  cv::rectangle(occluded, cv::Rect(110, 58, 36, 72), cv::Scalar(0), cv::FILLED);
  params.polarity = openshape::PolarityMode::Same;
  params.min_score = 0.25;
  params.min_visible_fraction = 0.20;
  const auto tolerant = openshape::find_shape_models(occluded, model, params);
  params.min_visible_fraction = 0.90;
  const auto strict = openshape::find_shape_models(occluded, model, params);
  check(!tolerant.empty(), "configurable visible fraction tolerates partial occlusion");
  check(strict.empty(), "strict visible fraction rejects excessive occlusion");
}
}

int main() {
  test_translation_and_noise();
  test_rotation();
  test_multiple_and_negative();
  test_scale_search();
  test_multiscale_range();
  test_combined_rotation_and_scale();
  test_multiple_scales_and_negative();
  test_subpixel_refinement();
  test_joint_pose_refinement();
  test_polarity_and_occlusion();
  if (failures != 0) {
    std::cerr << failures << " integration assertion(s) failed\n";
    return 1;
  }
  std::cout << "all integration assertions passed\n";
  return 0;
}
