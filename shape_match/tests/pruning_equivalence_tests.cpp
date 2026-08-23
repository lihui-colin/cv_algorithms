#include "openshape/openshape.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

cv::Mat rectangle_image() {
  cv::Mat image = cv::Mat::zeros(80, 80, CV_8UC1);
  cv::rectangle(image, cv::Rect(25, 28, 30, 24), cv::Scalar(255), 2);
  return image;
}

cv::Mat asymmetric_template_image() {
  cv::Mat image = cv::Mat::zeros(80, 80, CV_8UC1);
  cv::rectangle(image, cv::Rect(22, 25, 35, 26), cv::Scalar(255), 2);
  cv::line(image, cv::Point(24, 48), cv::Point(54, 28), cv::Scalar(255), 2);
  return image;
}

cv::Mat precision_regression_image() {
  cv::Mat image = cv::Mat::zeros(96, 96, CV_8UC1);
  cv::rectangle(image, cv::Rect(32, 35, 32, 26), cv::Scalar(255), 2);
  cv::line(image, cv::Point(34, 58), cv::Point(61, 38), cv::Scalar(255), 2);
  return image;
}

void add_rotated_target(cv::Mat& scene, cv::Point center, double angle) {
  const cv::Mat source = asymmetric_template_image();
  const cv::Mat transform = cv::getRotationMatrix2D(
      cv::Point2f(40.0f, 40.0f), angle, 1.0);
  cv::Mat rotated;
  cv::warpAffine(source, rotated, transform, source.size(), cv::INTER_NEAREST,
                 cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::Mat destination = scene(cv::Rect(
      center.x - source.cols / 2, center.y - source.rows / 2,
      source.cols, source.rows));
  cv::max(destination, rotated, destination);
}

bool contains_position(const std::vector<openshape::MatchResult>& results,
                       cv::Point2d expected, double tolerance) {
  for (const auto& result : results)
    if (std::hypot(result.column - expected.x, result.row - expected.y) <= tolerance)
      return true;
  return false;
}

openshape::ShapeModel make_model() {
  openshape::ShapeModelParams params;
  params.min_gradient_magnitude = 1.0;
  params.min_model_points = 4;
  params.max_model_points = 300;
  params.min_point_distance = 1.0;
  params.num_levels = 1;
  params.model_version = 2;
  return openshape::create_shape_model(rectangle_image(), params);
}

bool same_results(const std::vector<openshape::MatchResult>& left,
                  const std::vector<openshape::MatchResult>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (std::abs(left[i].score - right[i].score) > 1e-12 ||
        std::abs(left[i].row - right[i].row) > 1e-12 ||
        std::abs(left[i].column - right[i].column) > 1e-12 ||
        std::abs(left[i].angle - right[i].angle) > 1e-12 ||
        std::abs(left[i].scale - right[i].scale) > 1e-12)
      return false;
  }
  return true;
}

void test_prunable_score_matches_complete_score() {
  using namespace openshape;
  const auto model = make_model();
  const EdgeMap scene = EdgeEngine::compute(rectangle_image(), model.params(), true);
  for (PolarityMode polarity : {PolarityMode::Same, PolarityMode::Inverted,
                                PolarityMode::GlobalEither, PolarityMode::LocalEither})
    for (double column : {8.0, 31.25, 40.0, 73.0})
      for (double row : {9.0, 40.0, 68.5})
        for (double angle : {-17.0, 0.0, 21.0})
          for (double scale : {0.8, 1.0, 1.2}) {
            std::size_t valid = 0;
            const double complete = score_pose_precise(
                scene, model.levels_soa().front(), column, row, angle, scale,
                polarity, 1.0, &valid);
            const auto unpruned = score_pose_precise_prunable(
                scene, model.levels_soa().front(), column, row, angle, scale,
                polarity, 1.0, 0.0, 0, false);
            check(!unpruned.terminated() &&
                      unpruned.point_evaluations == model.size(),
                  "disabled bounded scorer evaluates the complete model");
            check(std::abs(complete - unpruned.score) <= 1e-12 &&
                      valid == unpruned.valid_point_count,
                  "bounded scorer preserves complete score and visibility semantics");

            const auto pruned = score_pose_precise_prunable(
                scene, model.levels_soa().front(), column, row, angle, scale,
                polarity, 1.0, 0.65, 0, true);
            if (pruned.terminated())
              check(complete + 1e-12 < 0.65,
                    "every score-bound certificate is confirmed by complete scoring");
            else
              check(std::abs(complete - pruned.score) <= 1e-12,
                    "non-terminated prunable score remains complete");
          }
}

void test_visible_and_score_certificates() {
  using namespace openshape;
  const auto model = make_model();
  const cv::Mat blank = cv::Mat::zeros(80, 80, CV_8UC1);
  const EdgeMap scene = EdgeEngine::compute(blank, model.params(), true);
  const auto score_rejection = score_pose_precise_prunable(
      scene, model.levels_soa().front(), 40.0, 40.0, 0.0, 1.0,
      PolarityMode::Same, 1.0, 0.8, 0, true);
  check(score_rejection.reason == PruningReason::ScoreUpperBound &&
            score_rejection.score_upper_bound < 0.8 &&
            score_rejection.point_evaluations < model.size(),
        "score upper bound rejects a blank pose before all points are evaluated");
  const auto visible_rejection = score_pose_precise_prunable(
      scene, model.levels_soa().front(), 40.0, 40.0, 0.0, 1.0,
      PolarityMode::Same, 1.0, 0.0, model.size(), true);
  check(visible_rejection.reason == PruningReason::VisibleUpperBound &&
            visible_rejection.point_evaluations < model.size(),
        "visible-count upper bound produces an independent rejection certificate");
}

openshape::SearchParams exhaustive_params() {
  openshape::SearchParams params;
  params.angle_start = 0.0;
  params.angle_extent = 0.0;
  params.scale_min = 1.0;
  params.scale_max = 1.0;
  params.min_score = 0.75;
  params.level_min_score_factor = 1.0;
  params.search_roi = cv::Rect(32, 32, 17, 17);
  params.num_matches = 0;
  params.exhaustive_translation_step = 1.0;
  return params;
}

void test_exhaustive_equivalence_and_audit() {
  using namespace openshape;
  const auto model = make_model();
  const EdgeMap scene = EdgeEngine::compute(rectangle_image(), model.params(), true);
  SearchParams baseline_params = exhaustive_params();
  SearchStats baseline_stats;
  const auto baseline = find_shape_models_exhaustive(
      scene, model, baseline_params, &baseline_stats);

  SearchParams optimized_params = baseline_params;
  optimized_params.enable_safe_pruning = true;
  SearchStats optimized_stats;
  const auto optimized = find_shape_models_exhaustive(
      scene, model, optimized_params, &optimized_stats);
  check(same_results(baseline, optimized),
        "safe pruning preserves exhaustive accepted poses and scores");
  check(optimized_stats.theoretical_pose_count == baseline_stats.pose_evaluations &&
            optimized_stats.domain_enumerated_pose_count ==
                optimized_stats.pose_evaluations &&
            optimized_stats.domain_enumerated_pose_count <=
                optimized_stats.theoretical_pose_count,
        "pose-domain statistics distinguish theoretical and enumerated poses");
  check(optimized_stats.safe_pruning_rejections > 0 &&
            optimized_stats.omitted_point_evaluations > 0 &&
            optimized_stats.full_score_evaluations < baseline_stats.full_score_evaluations &&
            optimized_stats.score_point_evaluations < baseline_stats.score_point_evaluations,
        "safe pruning records rejected poses and reduces model-point evaluations");

  optimized_params.pruning_audit_rate = 1.0;
  SearchStats scene_audit_stats;
  const auto scene_audited = find_shape_models_exhaustive(
      scene, model, optimized_params, &scene_audit_stats);
  check(same_results(baseline, scene_audited) &&
            scene_audit_stats.pruning_audit_evaluations > 0 &&
            scene_audit_stats.pruning_audit_failures == 0,
        "100% audit confirms pose-level and point-level rejection on a target scene");

  const cv::Mat blank = cv::Mat::zeros(80, 80, CV_8UC1);
  const EdgeMap blank_scene = EdgeEngine::compute(blank, model.params(), true);
  SearchParams blank_domain_params = optimized_params;
  blank_domain_params.search_roi = cv::Rect{};
  blank_domain_params.pruning_audit_rate = 0.0;
  SearchStats blank_domain_stats;
  const auto domain_pruned_blank = find_shape_models_exhaustive(
      blank_scene, model, blank_domain_params, &blank_domain_stats);
  check(domain_pruned_blank.empty() && blank_domain_stats.pose_evaluations == 0 &&
            blank_domain_stats.domain_enumerated_pose_count == 0 &&
            blank_domain_stats.domain_skipped_pose_count ==
                blank_domain_stats.theoretical_pose_count,
        "blank support map skips the complete translation pose domain");

  optimized_params.pruning_audit_rate = 1.0;
  SearchStats audit_stats;
  const auto audited = find_shape_models_exhaustive(
      blank_scene, model, optimized_params, &audit_stats);
  check(audited.empty() && audit_stats.pruning_audit_evaluations > 0 &&
            audit_stats.pruning_audit_failures == 0,
        "100% audit fully rescored every sampled rejection without finding a false reject");
}

void test_coarse_to_fine_v2_candidate_search() {
  using namespace openshape;
  ShapeModelParams model_params;
  model_params.min_gradient_magnitude = 1.0;
  model_params.min_model_points = 4;
  model_params.max_model_points = 160;
  model_params.min_point_distance = 1.0;
  model_params.num_levels = 5;
  model_params.model_version = 2;
  const auto model = create_shape_model(rectangle_image(), model_params);
  SearchParams params;
  params.enable_pyramid_candidate_search = true;
  params.max_pyramid_levels = 5;
  params.num_levels = 5;
  params.angle_start = -15.0;
  params.angle_extent = 30.0;
  params.angle_step = 0.1;
  params.scale_min = 1.0;
  params.scale_max = 1.0;
  params.search_roi = cv::Rect(32, 32, 17, 17);
  params.min_score = 0.65;
  params.level_min_score_factor = 0.25;
  params.strict_detection = false;
  params.num_threads = 1;
  SearchStats single_stats;
  const auto single_results = find_shape_models(
      ImageView(rectangle_image()), model, params, &single_stats);
  params.num_threads = 4;
  SearchStats threaded_stats;
  const auto threaded_results = find_shape_models(
      ImageView(rectangle_image()), model, params, &threaded_stats);
  check(model.levels().size() == 5 && !threaded_results.empty(),
        "v2 pyramid candidate search retains the target across five levels");
  check(same_results(single_results, threaded_results),
        "v2 pyramid candidate search is deterministic across one and four threads");
  check(threaded_stats.pose_evaluations < (301u * 17u * 17u) / 2u,
        "hierarchical angle and position propagation removes over half of the single-level poses");
  check(single_stats.pose_evaluations == threaded_stats.pose_evaluations,
        "v2 pyramid candidate pose count is independent of worker count");

  ShapeModelParams precision_model_params = model_params;
  precision_model_params.high_precision_point_spacing = 0.5;
  const cv::Mat precision_image = precision_regression_image();
  const auto precision_model = create_shape_model(
      precision_image, precision_model_params);
  SearchParams precision_params = params;
  precision_params.search_roi = cv::Rect(40, 40, 17, 17);
  precision_params.min_score = 0.75;
  precision_params.num_matches = 1;
  precision_params.num_threads = 4;
  const auto precision_results = find_shape_models(
      ImageView(precision_image), precision_model, precision_params);
  check(precision_results.size() == 1 &&
            precision_results.front().row == 48.0 &&
            precision_results.front().column == 48.0 &&
            std::abs(precision_results.front().angle + 0.9) <= 1e-12 &&
            precision_results.front().score >= 0.869,
        "single-target precise shortlist preserves the established pose and score");
}

openshape::SearchParams multi_target_params() {
  openshape::SearchParams params;
  params.enable_pyramid_candidate_search = true;
  params.max_pyramid_levels = 5;
  params.num_levels = 5;
  params.angle_start = -35.0;
  params.angle_extent = 70.0;
  params.angle_step = 1.0;
  params.scale_min = 1.0;
  params.scale_max = 1.0;
  params.min_score = 0.35;
  params.level_min_score_factor = 0.25;
  params.num_matches = 8;
  params.max_overlap = 0.35;
  params.strict_detection = false;
  return params;
}

void test_synthetic_multi_target_recall() {
  using namespace openshape;
  ShapeModelParams model_params;
  model_params.model_version = 2;
  model_params.num_levels = 5;
  model_params.min_gradient_magnitude = 1.0;
  model_params.min_model_points = 4;
  model_params.max_model_points = 160;
  model_params.high_precision_point_spacing = 0.5;
  const auto model = create_shape_model(asymmetric_template_image(), model_params);

  cv::Mat single_scene = cv::Mat::zeros(150, 280, CV_8UC1);
  add_rotated_target(single_scene, cv::Point(80, 75), 0.0);
  SearchParams single_params = multi_target_params();
  single_params.num_matches = 1;
  const auto single_results = find_shape_models(
      ImageView(single_scene), model, single_params);
  check(single_results.size() == 1 &&
            contains_position(single_results, cv::Point2d(80, 75), 3.0),
        "single-target pyramid result remains centered on the original target");

  cv::Mat separated_scene = cv::Mat::zeros(150, 280, CV_8UC1);
  add_rotated_target(separated_scene, cv::Point(80, 75), 0.0);
  add_rotated_target(separated_scene, cv::Point(200, 75), 25.0);
  SearchParams params = multi_target_params();
  params.num_threads = 1;
  SearchStats single_thread_stats;
  const auto separated_single_thread = find_shape_models(
      ImageView(separated_scene), model, params, &single_thread_stats);
  params.num_threads = 4;
  SearchStats four_thread_stats;
  const auto separated_four_threads = find_shape_models(
      ImageView(separated_scene), model, params, &four_thread_stats);
  check(separated_four_threads.size() == 2 &&
            contains_position(separated_four_threads, cv::Point2d(80, 75), 5.0) &&
            contains_position(separated_four_threads, cv::Point2d(200, 75), 12.0),
        "spatial peak quotas retain two separated rotated targets without duplicates");
  check(same_results(separated_single_thread, separated_four_threads) &&
            single_thread_stats.pose_evaluations == four_thread_stats.pose_evaluations,
        "multi-target results and pose counts are deterministic across one and four threads");

  cv::Mat neighboring_scene = cv::Mat::zeros(150, 280, CV_8UC1);
  add_rotated_target(neighboring_scene, cv::Point(80, 75), 0.0);
  add_rotated_target(neighboring_scene, cv::Point(140, 75), 25.0);
  const auto neighboring_results = find_shape_models(
      ImageView(neighboring_scene), model, params);
  check(neighboring_results.size() == 2 &&
            contains_position(neighboring_results, cv::Point2d(80, 75), 12.0) &&
            contains_position(neighboring_results, cv::Point2d(140, 75), 15.0),
        "position clustering keeps neighboring targets as independent peaks");
}

void test_invariant_image_multi_target_recall() {
  using namespace openshape;
  const std::filesystem::path data_dir =
      std::filesystem::path(__FILE__).parent_path().parent_path() / "data";
  const cv::Mat templ = cv::imread(
      (data_dir / "invariant_template_1.jpg").string(), cv::IMREAD_GRAYSCALE);
  const cv::Mat image = cv::imread(
      (data_dir / "invariant_image_1.png").string(), cv::IMREAD_GRAYSCALE);
  check(!templ.empty() && !image.empty(),
        "invariant multi-target regression data is available");
  if (templ.empty() || image.empty()) return;

  ShapeModelParams model_params;
  model_params.model_version = 2;
  model_params.num_levels = 5;
  model_params.min_gradient_magnitude = 8.0;
  model_params.min_model_points = 20;
  model_params.max_model_points = 160;
  model_params.high_precision_point_spacing = 0.5;
  const auto model = create_shape_model(templ, model_params);
  SearchParams params;
  params.enable_pyramid_candidate_search = true;
  params.max_pyramid_levels = 5;
  params.num_levels = 5;
  params.angle_start = -60.0;
  params.angle_extent = 120.0;
  params.angle_step = 0.5;
  params.scale_min = 1.0;
  params.scale_max = 1.0;
  params.min_score = 0.45;
  params.level_min_score_factor = 0.25;
  params.num_matches = 8;
  params.max_overlap = 0.35;
  params.strict_detection = false;
  params.num_threads = 1;
  const auto single_thread_results = find_shape_models(
      ImageView(image), model, params);
  params.num_threads = 4;
  const auto four_thread_results = find_shape_models(
      ImageView(image), model, params);
  check(four_thread_results.size() == 2 &&
            contains_position(four_thread_results, cv::Point2d(251, 167), 12.0) &&
            contains_position(four_thread_results, cv::Point2d(420, 141), 12.0),
        "invariant_image_1 returns both spatially distinct metal targets");
  check(same_results(single_thread_results, four_thread_results),
        "real-image multi-target results are deterministic across one and four threads");
}

}  // namespace

int main() {
  test_prunable_score_matches_complete_score();
  test_visible_and_score_certificates();
  test_exhaustive_equivalence_and_audit();
  test_coarse_to_fine_v2_candidate_search();
  test_synthetic_multi_target_recall();
  test_invariant_image_multi_target_recall();
  if (failures != 0) {
    std::cerr << failures << " pruning equivalence checks failed\n";
    return 1;
  }
  std::cout << "pruning equivalence checks passed\n";
  return 0;
}
