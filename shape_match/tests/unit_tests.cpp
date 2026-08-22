#include "openshape/openshape.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <stdexcept>
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

template <typename Exception, typename Function>
void check_throws(Function&& function, const std::string& message) {
  try {
    function();
    check(false, message + " (no exception)");
  } catch (const Exception&) {
  } catch (const std::exception& e) {
    check(false, message + " (wrong exception: " + e.what() + ")");
  }
}

cv::Mat rectangle_image(int width = 80, int height = 80) {
  cv::Mat image = cv::Mat::zeros(height, width, CV_8UC1);
  cv::rectangle(image, cv::Rect(20, 20, 40, 30), cv::Scalar(255), 2);
  return image;
}

openshape::ShapeModel make_model() {
  openshape::ShapeModelParams params;
  params.min_gradient_magnitude = 1;
  params.min_model_points = 4;
  params.max_model_points = 200;
  params.min_point_distance = 2;
  params.num_levels = 3;
  return openshape::create_shape_model(rectangle_image(), params);
}

void test_parameters_and_angles() {
  using namespace openshape;
  check(normalize_angle(0) == 0, "zero angle remains zero");
  check(normalize_angle(180) == -180, "180 degrees normalizes to -180");
  check(std::abs(normalize_angle(-181) - 179) < 1e-9, "negative angle wraps correctly");

  ShapeModelParams model;
  model.validate();
  model.canny_high_threshold = model.canny_low_threshold;
  check_throws<InvalidArgument>([&] { model.validate(); }, "invalid Canny threshold is rejected");

  SearchParams search;
  search.validate();
  check(search.enable_coarse_prefilter, "coarse valid-point prefilter is enabled by default");
  check(search.scale_min == 1.0 && search.scale_max == 1.0 && search.scale_step == 0.05,
        "scale search defaults preserve fixed-scale matching");
  search.min_score = 1.1;
  check_throws<InvalidArgument>([&] { search.validate(); }, "score outside [0,1] is rejected");
  search.min_score = 0.5;
  search.enable_subpixel = true;
  search.validate();
  search.enable_subpixel = false;
  search.enable_pose_refinement = true;
  search.polarity = PolarityMode::GlobalEither;
  search.min_visible_fraction = 0.4;
  search.validate();
  search.enable_pose_refinement = false;
  search.greediness = 1.0;
  search.coarse_point_fraction = 0.5;
  search.enable_candidate_clustering = true;
  search.refinement_candidate_limit = 8;
  search.validate();
  search.greediness = 1.1;
  check_throws<InvalidArgument>([&] { search.validate(); }, "greediness above one is rejected");
  search.greediness = 0.0;
  search.coarse_point_fraction = 0.0;
  check_throws<InvalidArgument>([&] { search.validate(); }, "zero coarse point fraction is rejected");
  search.coarse_point_fraction = 1.0;
  search.enable_candidate_clustering = false;
  search.refinement_variants_per_peak = 0;
  check_throws<InvalidArgument>([&] { search.validate(); },
                                "zero refinement variants per peak is rejected");
  search.refinement_variants_per_peak = 3;
  search.enable_fast_pipeline();
  check(search.greediness == 0.9 && search.coarse_point_fraction == 0.75 &&
            search.enable_candidate_clustering && search.refinement_candidate_limit == 32 &&
            !search.strict_detection,
        "fast pipeline helper selects the conservative optimized defaults");
  search.min_visible_fraction = 0.0;
  check_throws<InvalidArgument>([&] { search.validate(); }, "zero visible fraction is rejected");
  search.min_visible_fraction = 0.25;
  search.scale_min = 0.0;
  check_throws<InvalidArgument>([&] { search.validate(); }, "zero minimum scale is rejected");
  search.scale_min = -0.5;
  check_throws<InvalidArgument>([&] { search.validate(); }, "negative minimum scale is rejected");
  search.scale_min = 1.1;
  search.scale_max = 1.0;
  check_throws<InvalidArgument>([&] { search.validate(); }, "reversed scale range is rejected");
  search.scale_min = 1.0;
  search.scale_max = 1.0;
  search.scale_step = -0.1;
  check_throws<InvalidArgument>([&] { search.validate(); }, "non-positive scale step is rejected");
}

void test_image_and_edges() {
  using namespace openshape;
  cv::Mat gray = rectangle_image();
  cv::Mat bgr, bgra;
  cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
  cv::cvtColor(gray, bgra, cv::COLOR_GRAY2BGRA);
  check(EdgeEngine::to_gray(ImageView(gray)).channels() == 1, "gray input remains gray");
  check(EdgeEngine::to_gray(ImageView(bgr)).channels() == 1, "BGR input converts to gray");
  check(EdgeEngine::to_gray(ImageView(bgra)).channels() == 1, "BGRA input converts to gray");
  cv::Mat sixteen = cv::Mat::zeros(10, 10, CV_16UC1);
  check_throws<InvalidArgument>([&] { EdgeEngine::to_gray(ImageView(sixteen)); }, "non-8-bit input is rejected");
  cv::Mat two_channel(10, 10, CV_8UC2);
  check_throws<InvalidArgument>([&] { EdgeEngine::to_gray(ImageView(two_channel)); }, "unsupported channel count is rejected");

  ShapeModelParams params;
  params.min_gradient_magnitude = 1;
  const EdgeMap map = EdgeEngine::compute(ImageView(gray), params, true);
  check(map.gray.type() == CV_8UC1, "edge gray image type is CV_8UC1");
  check(map.gx.type() == CV_32FC1 && map.gy.type() == CV_32FC1, "Sobel gradients are CV_32F");
  check(map.magnitude.type() == CV_32FC1 && map.orientation.type() == CV_32FC1, "magnitude and orientation are CV_32F");
  check(map.edges.type() == CV_8UC1, "Canny mask is CV_8UC1");
  check(map.soft_edge_response.type() == CV_32FC1 &&
            map.soft_edge_response.size() == map.edges.size(),
        "soft edge response is a full-size float field");
  check(map.normalized_magnitude.type() == CV_32FC1,
        "continuous edge strength is normalized into a float field");
  check(map.edge_integral.type() == CV_32SC1 &&
            map.edge_integral.size() == cv::Size(map.edges.cols + 1, map.edges.rows + 1),
        "Canny summed-area table has the expected type and padded size");
  double min_value = 0, max_value = 0;
  cv::minMaxLoc(map.magnitude, &min_value, &max_value);
  check(min_value >= 0 && max_value > 0, "gradient magnitude is non-negative and non-zero");
}

void test_precise_score_polarity() {
  using namespace openshape;
  const cv::Mat templ = rectangle_image();
  const auto model = make_model();
  cv::Mat inverted;
  cv::bitwise_not(templ, inverted);
  const EdgeMap normal = EdgeEngine::compute(templ, model.params(), true);
  const EdgeMap reversed = EdgeEngine::compute(inverted, model.params(), true);
  std::size_t normal_valid = 0, reversed_valid = 0;
  const double normal_score = score_pose_precise(
      normal, model.levels_soa()[0], 40, 40, 0, 1.0,
      PolarityMode::Same, 1.0, &normal_valid);
  const double signed_reversed = score_pose_precise(
      reversed, model.levels_soa()[0], 40, 40, 0, 1.0,
      PolarityMode::Same, 1.0, &reversed_valid);
  const double global_either = score_pose_precise(
      reversed, model.levels_soa()[0], 40, 40, 0, 1.0,
      PolarityMode::GlobalEither, 1.0, nullptr);
  const double local_either = score_pose_precise(
      reversed, model.levels_soa()[0], 40, 40, 0, 1.0,
      PolarityMode::LocalEither, 1.0, nullptr);
  check(normal_valid > 0 && reversed_valid > 0 && normal_score > 0.8,
        "precise score recognizes aligned soft edges");
  check(global_either > signed_reversed + 0.4 && local_either > signed_reversed + 0.4,
        "global and local polarity modes recover an inverted target");
  check_throws<InvalidArgument>(
      [&] { score_pose_precise(normal, model.levels_soa()[0], 40, 40, 0, 1, PolarityMode::Same, 0); },
      "precise score rejects zero edge sigma");
}

void test_model_persistence() {
  using namespace openshape;
  const ShapeModel model = make_model();
  const std::string path = "/tmp/openshape-model-roundtrip.osm";
  const std::string corrupt_path = "/tmp/openshape-model-corrupt.osm";
  model.save(path);
  const ShapeModel loaded = ShapeModel::load(path);
  check(loaded.version() == model.version() && loaded.roi() == model.roi() &&
            loaded.origin() == model.origin() && loaded.size() == model.size() &&
            loaded.levels().size() == model.levels().size(),
        "versioned model round-trip preserves metadata and pyramid shape");
  bool points_equal = loaded.points().size() == model.points().size();
  for (std::size_t i = 0; points_equal && i < model.points().size(); ++i) {
    const auto& a = loaded.points()[i];
    const auto& b = model.points()[i];
    points_equal = a.relative_x == b.relative_x && a.relative_y == b.relative_y &&
                   a.orientation == b.orientation && a.weight == b.weight && a.level == b.level;
  }
  check(points_equal, "versioned model round-trip is bit-exact for model points");

  std::ifstream input(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  if (!bytes.empty()) bytes.back() ^= 0x1;
  std::ofstream corrupt(corrupt_path, std::ios::binary | std::ios::trunc);
  corrupt.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  corrupt.close();
  check_throws<InvalidModel>([&] { ShapeModel::load(corrupt_path); },
                             "model checksum rejects a corrupted file");
  std::remove(path.c_str());
  std::remove(corrupt_path.c_str());
}

void test_pyramids_and_model() {
  using namespace openshape;
  const cv::Mat image = rectangle_image(81, 65);
  const auto pyramid = ImagePyramid::build(ImageView(image), 3);
  check(pyramid.levels().size() == 3, "explicit image pyramid level count");
  check(pyramid.levels()[0].size() == cv::Size(81, 65), "pyramid level zero size");
  check(pyramid.levels()[1].size() == cv::Size(41, 33), "pyramid odd-size downsample");
  check(pyramid.levels()[2].size() == cv::Size(21, 17), "pyramid second downsample");

  const auto model = make_model();
  check(!model.empty() && model.size() >= 4, "model has valid points");
  check(model.levels().size() == 3, "model pyramid level count");
  check(model.levels()[1].size() == model.size(), "model level preserves point count");
  check(std::abs(model.levels()[1][0].relative_x - model.points()[0].relative_x * 0.5f) < 1e-5f,
        "model point coordinates scale by one half");
  check(model.levels_soa().size() == model.levels().size(), "SoA cache has one level per AoS level");
  if (!model.levels_soa().empty()) {
    check(model.levels_soa()[0].size() == model.levels()[0].size(), "SoA cache preserves point count");
    check(std::abs(model.levels_soa()[0].relative_x[0] - model.levels()[0][0].relative_x) < 1e-6f,
          "SoA cache preserves point coordinates");
  }

  ShapeModelParams magnitude_params = make_model().params();
  magnitude_params.model_point_sampling = "magnitude";
  const auto magnitude_model = ShapeModelBuilder::create(ImageView(image), magnitude_params);
  bool uniform_weights = true;
  for (std::size_t i = 1; i < model.points().size(); ++i)
    uniform_weights = uniform_weights && model.points()[i].weight == model.points()[0].weight;
  bool magnitude_varies = false;
  for (std::size_t i = 1; i < magnitude_model.points().size(); ++i)
    magnitude_varies = magnitude_varies ||
        magnitude_model.points()[i].weight != magnitude_model.points()[0].weight;
  check(uniform_weights, "uniform model sampling uses equal point confidence weights");
  check(magnitude_varies, "magnitude model sampling preserves gradient confidence weights");

  ShapeModelParams custom;
  custom.roi = cv::Rect(10, 10, 60, 60);
  custom.origin = cv::Point2f(10, 10);
  custom.min_gradient_magnitude = 1;
  custom.min_model_points = 1;
  custom.max_model_points = 200;
  const auto shifted = ShapeModelBuilder::create(ImageView(rectangle_image()), custom);
  check(std::abs(shifted.points()[0].relative_x - (model.points()[0].relative_x + 30)) < 100,
        "custom origin produces relative coordinates");

  ShapeModelParams low_texture;
  low_texture.min_gradient_magnitude = 1;
  low_texture.min_model_points = 4;
  check_throws<InvalidModel>([&] { ShapeModelBuilder::create(ImageView(cv::Mat::zeros(40, 40, CV_8UC1)), low_texture); },
                             "low-texture model is rejected");

  EdgeMap prepared = EdgeEngine::compute(ImageView(image), make_model().params());
  const auto edge_pyramid = EdgePyramid::build(prepared, make_model().params(), 3);
  check(edge_pyramid.levels().size() == 3, "edge pyramid preserves explicit level count");
  check(edge_pyramid.levels()[1].gray.size() == cv::Size(41, 33),
        "edge pyramid downsample size matches image pyramid");
}

void test_score_and_nms() {
  using namespace openshape;
  EdgeMap scene;
  scene.edges = cv::Mat::ones(20, 20, CV_8UC1) * 255;
  scene.orientation = cv::Mat::zeros(20, 20, CV_32FC1);
  std::vector<ModelPoint> points{{0, 0, 0, 1, 0}, {2, 0, 0, 1, 0}, {0, 2, 0, 1, 0}};
  std::size_t valid = 0;
  check(std::abs(score_pose(scene, points, 5, 5, 0, &valid) - 1.0) < 1e-9 && valid == 3,
        "aligned score is one");
  points[0].orientation = static_cast<float>(CV_PI);
  check(score_pose(scene, points, 5, 5, 0, nullptr) < 0.7, "opposite direction lowers score");
  check(score_pose(scene, points, -5, -5, 0, &valid) == 0 && valid == 0, "out-of-bounds score is zero");
  ModelLevelSoA soa;
  for (const auto& point : points) {
    soa.relative_x.push_back(point.relative_x);
    soa.relative_y.push_back(point.relative_y);
    soa.orientation.push_back(point.orientation);
    soa.weight.push_back(point.weight);
  }
  const double aos_score = score_pose(scene, points, 5, 5, 17, &valid);
  std::size_t soa_valid = 0;
  const double soa_score = score_pose(scene, soa, 5, 5, 17, &soa_valid);
  check(std::abs(aos_score - soa_score) < 1e-7 && valid == soa_valid,
        "SoA score kernel matches AoS scalar baseline");
  std::size_t scaled_aos_valid = 0, scaled_soa_valid = 0;
  const double legacy_scale_score = score_pose(scene, points, 5, 5, 17, &scaled_aos_valid);
  const double explicit_scale_score = score_pose(scene, points, 5, 5, 17, 1.0, &scaled_soa_valid);
  check(legacy_scale_score == explicit_scale_score && scaled_aos_valid == scaled_soa_valid,
        "explicit unit-scale AoS score is identical to the legacy overload");
  const double legacy_soa_scale_score = score_pose(scene, soa, 5, 5, 17, &scaled_aos_valid);
  const double explicit_soa_scale_score = score_pose(scene, soa, 5, 5, 17, 1.0, &scaled_soa_valid);
  check(legacy_soa_scale_score == explicit_soa_scale_score && scaled_aos_valid == scaled_soa_valid,
        "explicit unit-scale SoA score is identical to the legacy overload");

  EdgeMap scaled_scene;
  scaled_scene.edges = cv::Mat::zeros(30, 30, CV_8UC1);
  scaled_scene.orientation = cv::Mat::zeros(30, 30, CV_32FC1);
  std::vector<ModelPoint> scaled_points{{0, 0, 0, 1, 0}, {4, 0, 0, 1, 0}, {0, 4, 0, 1, 0}};
  scaled_scene.edges.at<unsigned char>(10, 10) = 255;
  scaled_scene.edges.at<unsigned char>(10, 16) = 255;
  scaled_scene.edges.at<unsigned char>(16, 10) = 255;
  ModelLevelSoA scaled_soa;
  for (const auto& point : scaled_points) {
    scaled_soa.relative_x.push_back(point.relative_x);
    scaled_soa.relative_y.push_back(point.relative_y);
    scaled_soa.orientation.push_back(point.orientation);
    scaled_soa.weight.push_back(point.weight);
  }
  const double scaled_aos_score = score_pose(
      scaled_scene, scaled_points, 10, 10, 0, 1.5, &scaled_aos_valid);
  const double scaled_soa_score = score_pose(
      scaled_scene, scaled_soa, 10, 10, 0, 1.5, &scaled_soa_valid);
  check(scaled_aos_score == 1.0 && scaled_soa_score == 1.0 &&
            scaled_aos_valid == 3 && scaled_soa_valid == 3,
        "scale-aware AoS and SoA scores sample transformed model coordinates");
  check_throws<InvalidArgument>(
      [&] { score_pose(scaled_scene, scaled_points, 10, 10, 0, 0.0); },
      "scale-aware score rejects non-positive scale");

  std::vector<ModelPoint> many_points;
  ModelLevelSoA many_soa;
  for (int i = 0; i < 16; ++i) {
    const ModelPoint point{static_cast<float>(i % 4), static_cast<float>(i / 4),
                          static_cast<float>((i % 5) * 0.11), 0.5f + 0.03f * i, 0};
    many_points.push_back(point);
    many_soa.relative_x.push_back(point.relative_x);
    many_soa.relative_y.push_back(point.relative_y);
    many_soa.orientation.push_back(point.orientation);
    many_soa.weight.push_back(point.weight);
  }
  std::size_t many_aos_valid = 0, many_soa_valid = 0;
  const double many_aos_score = score_pose(scene, many_points, 5.25, 5.75, 23, &many_aos_valid);
  const double many_soa_score = score_pose(scene, many_soa, 5.25, 5.75, 23, &many_soa_valid);
  check(std::abs(many_aos_score - many_soa_score) < 5e-4 && many_aos_valid == many_soa_valid,
        "SIMD-dispatched score matches scalar baseline for a full vector batch");

  const auto model = make_model();
  std::vector<MatchResult> candidates{
      {40, 40, 0, 1, 0.90, 10, 10, 0, false},
      {40, 40, 0, 1, 0.80, 10, 10, 0, false},
      {40, 100, 0, 1, 0.85, 10, 10, 0, false}};
  const auto kept = non_max_suppression(model, candidates, 0.5, 0);
  check(kept.size() == 2, "NMS suppresses overlapping lower score candidate");
  check(kept[0].score >= kept[1].score, "NMS results are sorted by score");
  const auto limited = non_max_suppression(model, candidates, 0.5, 1);
  check(limited.size() == 1, "NMS applies num_matches after suppression");
  check_throws<InvalidArgument>([&] { non_max_suppression(model, candidates, 1.1); }, "invalid NMS overlap is rejected");
  MatchResult small;
  small.row = 40;
  small.column = 40;
  small.scale = 0.5;
  MatchResult large = small;
  large.column = 80;
  large.scale = 1.0;
  check(std::abs(bounding_box_overlap(model, small, large) - 0.5) < 1e-6,
        "bounding-box overlap uses each result scale");
  MatchResult high_scale = small;
  high_scale.scale = 1.2;
  high_scale.score = 0.9;
  MatchResult low_scale = high_scale;
  low_scale.scale = 0.8;
  const auto scale_sorted = non_max_suppression(model, {high_scale, low_scale}, 1.0, 0);
  check(scale_sorted.size() == 2 && scale_sorted.front().scale == 0.8,
        "NMS uses scale as the final stable tie-break");

  cv::Mat visualization = cv::Mat::zeros(140, 160, CV_8UC3);
  draw_match_results(visualization, model, kept);
  check(cv::countNonZero(visualization.reshape(1)) > 0,
        "drawing interface renders model boxes and origins");
  cv::Mat scaled_visualization = cv::Mat::zeros(140, 160, CV_8UC3);
  MatchResult scaled_result;
  scaled_result.row = 70;
  scaled_result.column = 70;
  scaled_result.scale = 0.5;
  draw_match_results(scaled_visualization, model, {scaled_result});
  check(cv::countNonZero(scaled_visualization(cv::Rect(47, 47, 7, 7)).reshape(1)) > 0,
        "drawing scales the model ROI around the result origin");
  check_throws<InvalidArgument>([&] { draw_match_results(visualization, model, kept, {}, 0); },
                                "drawing rejects non-positive thickness");
}

void test_matching_determinism() {
  using namespace openshape;
  const auto model = make_model();
  cv::Mat scene = cv::Mat::zeros(160, 180, CV_8UC1);
  cv::rectangle(scene, cv::Rect(70, 60, 40, 30), cv::Scalar(255), 2);
  SearchParams params;
  params.num_levels = 1;
  params.min_score = 0.5;
  params.num_matches = 1;
  const auto first = find_shape_models(ImageView(scene), model, params);
  const auto second = find_shape_models(ImageView(scene), model, params);
  SearchParams explicit_fixed = params;
  explicit_fixed.scale_min = 1.0;
  explicit_fixed.scale_max = 1.0;
  explicit_fixed.scale_step = 0.25;
  const auto explicit_fixed_results = find_shape_models(ImageView(scene), model, explicit_fixed);
  check(!first.empty(), "synthetic translation produces a match");
  check(first.size() == second.size(), "repeated matching has stable result count");
  if (!first.empty() && !second.empty()) {
    check(first.front().row == second.front().row && first.front().column == second.front().column &&
              first.front().angle == second.front().angle && first.front().scale == 1.0 &&
              second.front().scale == 1.0 && first.front().score == second.front().score,
          "repeated matching has stable result values");
  }
  check(first.size() == explicit_fixed_results.size(),
        "default and explicit fixed-scale searches have identical result counts");
  if (!first.empty() && !explicit_fixed_results.empty()) {
    check(first.front().row == explicit_fixed_results.front().row &&
              first.front().column == explicit_fixed_results.front().column &&
              first.front().angle == explicit_fixed_results.front().angle &&
              first.front().scale == explicit_fixed_results.front().scale &&
              first.front().score == explicit_fixed_results.front().score,
          "default and explicit fixed-scale searches are exactly compatible");
  }
  SearchParams pyramid_params = params;
  pyramid_params.num_levels = 0;
  const auto pyramid_results = find_shape_models(ImageView(scene), model, pyramid_params);
  check(!pyramid_results.empty(), "default pyramid search produces a match");
  SearchParams threaded = pyramid_params;
  threaded.angle_start = -8;
  threaded.angle_extent = 16;
  threaded.angle_step = 2;
  threaded.scale_min = 0.9;
  threaded.scale_max = 1.1;
  threaded.scale_step = 0.1;
  threaded.num_threads = 4;
  const auto threaded_results = find_shape_models(ImageView(scene), model, threaded);
  SearchParams fast = threaded;
  fast.deterministic = false;
  const auto fast_results = find_shape_models(ImageView(scene), model, fast);
  check(!fast_results.empty() && !threaded_results.empty() &&
            fast_results.front().score == threaded_results.front().score,
        "fast scheduling mode preserves the best score");
  threaded.num_threads = 1;
  SearchStats scale_stats;
  const auto single_results = find_shape_models(ImageView(scene), model, threaded, &scale_stats);
  check(scale_stats.angle_candidate_count == 9 && scale_stats.scale_candidate_count == 3 &&
            scale_stats.transform_candidate_count == 27,
        "search statistics report angle, scale, and joint transform counts");
  check(scale_stats.transform_preparation_time_ms == scale_stats.rotation_preparation_time_ms,
        "legacy rotation preparation timing mirrors scale-aware transform preparation");
  check(threaded_results.size() == single_results.size(), "threaded and single-threaded result counts match");
  if (!threaded_results.empty() && !single_results.empty()) {
    check(threaded_results.front().row == single_results.front().row &&
              threaded_results.front().column == single_results.front().column &&
              threaded_results.front().angle == single_results.front().angle &&
              threaded_results.front().scale == single_results.front().scale &&
              threaded_results.front().score == single_results.front().score,
          "threaded and single-threaded top result is deterministic");
  }
  SearchParams multilevel_unfiltered = threaded;
  multilevel_unfiltered.enable_coarse_prefilter = false;
  const auto multilevel_unfiltered_results = find_shape_models(ImageView(scene), model, multilevel_unfiltered);
  check(single_results.size() == multilevel_unfiltered_results.size(),
        "multilevel coarse prefilter preserves result count");
  if (!single_results.empty() && !multilevel_unfiltered_results.empty()) {
    check(single_results.front().row == multilevel_unfiltered_results.front().row &&
              single_results.front().column == multilevel_unfiltered_results.front().column &&
              single_results.front().angle == multilevel_unfiltered_results.front().angle &&
              single_results.front().scale == multilevel_unfiltered_results.front().scale &&
              single_results.front().score == multilevel_unfiltered_results.front().score,
          "multilevel coarse prefilter preserves the top result exactly");
  }
  const EdgeMap precomputed = EdgeEngine::compute(ImageView(scene), model.params());
  const auto reused_results = find_shape_models(precomputed, model, threaded);
  check(reused_results.size() == single_results.size(), "precomputed edge map preserves result count");
  if (!reused_results.empty() && !single_results.empty()) {
    check(reused_results.front().row == single_results.front().row &&
              reused_results.front().column == single_results.front().column &&
              reused_results.front().angle == single_results.front().angle &&
              reused_results.front().scale == single_results.front().scale &&
              reused_results.front().score == single_results.front().score,
          "precomputed edge map preserves top result");
  }
  if (!single_results.empty()) {
    std::size_t baseline_valid = 0;
    const double baseline_score = score_pose(precomputed, model.levels_soa()[0],
                                             single_results.front().column,
                                             single_results.front().row,
                                             single_results.front().angle,
                                             single_results.front().scale,
                                             &baseline_valid);
    check(std::abs(baseline_score - single_results.front().score) < 1e-7 &&
              baseline_valid == single_results.front().valid_point_count,
          "angle-prepared matcher score matches the public scalar/SoA score contract");
  }
  EdgeMap incomplete;
  check_throws<InvalidArgument>([&] { find_shape_models(incomplete, model, params); },
                                "incomplete precomputed edge map is rejected");
  const auto prepared_pyramid = EdgePyramid::build(precomputed, model.params(),
                                                   static_cast<int>(model.levels().size()));
  const auto prepared_pyramid_results = find_shape_models(prepared_pyramid, model, threaded);
  check(prepared_pyramid_results.size() == single_results.size(), "prepared edge pyramid preserves result count");
  if (!prepared_pyramid_results.empty() && !single_results.empty()) {
    check(prepared_pyramid_results.front().row == single_results.front().row &&
              prepared_pyramid_results.front().column == single_results.front().column &&
              prepared_pyramid_results.front().angle == single_results.front().angle &&
              prepared_pyramid_results.front().scale == single_results.front().scale &&
              prepared_pyramid_results.front().score == single_results.front().score,
          "prepared edge pyramid preserves top result");
  }
  check(!cpu_features_string().empty(), "CPU feature summary is available");
  const std::string kernel_name = active_score_kernel_name();
  check(kernel_name == "portable-soa" || kernel_name == "avx2",
        "active score kernel is a supported portable or AVX2 implementation");
  if (kernel_name == "avx2")
    check(detect_cpu_features().avx2, "AVX2 kernel is selected only on AVX2-capable runtime");
  check(std::string(active_matcher_score_kernel_name()) == "portable-rotated-soa",
        "matcher reports its angle-prepared portable score kernel");

  SearchParams filtered = params;
  filtered.num_levels = 1;
  filtered.enable_coarse_prefilter = true;
  SearchStats filtered_stats;
  const auto filtered_results = find_shape_models(precomputed, model, filtered, &filtered_stats);
  SearchParams unfiltered = filtered;
  unfiltered.enable_coarse_prefilter = false;
  SearchStats unfiltered_stats;
  const auto unfiltered_results = find_shape_models(precomputed, model, unfiltered, &unfiltered_stats);
  check(filtered_results.size() == unfiltered_results.size(),
        "coarse prefilter preserves result count");
  if (!filtered_results.empty() && !unfiltered_results.empty()) {
    check(filtered_results.front().row == unfiltered_results.front().row &&
              filtered_results.front().column == unfiltered_results.front().column &&
              filtered_results.front().angle == unfiltered_results.front().angle &&
              filtered_results.front().scale == unfiltered_results.front().scale &&
              filtered_results.front().score == unfiltered_results.front().score,
          "coarse prefilter preserves the top result exactly");
  }
  check(filtered_stats.prefilter_evaluations == filtered_stats.pose_evaluations,
        "single-level global search prefilters every pose");
  check(filtered_stats.prefilter_rejections > 0 &&
            filtered_stats.full_score_evaluations < filtered_stats.pose_evaluations,
        "coarse prefilter rejects poses before the full score kernel");
  check(filtered_stats.integral_prefilter_rejections > 0,
        "summed-area prefilter rejects edge-free transformed model regions");
  check(unfiltered_stats.prefilter_evaluations == 0 &&
            unfiltered_stats.full_score_evaluations == unfiltered_stats.pose_evaluations,
        "disabled prefilter sends every pose to the full score kernel");

  SearchParams greedy = unfiltered;
  greedy.greediness = 0.95;
  greedy.strict_detection = false;
  SearchStats greedy_stats;
  const auto greedy_results = find_shape_models(precomputed, model, greedy, &greedy_stats);
  check(greedy_results.size() == unfiltered_results.size(),
        "safe greediness pruning preserves result count");
  if (!greedy_results.empty() && !unfiltered_results.empty())
    check(greedy_results.front().row == unfiltered_results.front().row &&
              greedy_results.front().column == unfiltered_results.front().column &&
              greedy_results.front().score == unfiltered_results.front().score,
          "safe greediness pruning preserves the best result exactly");
  check(greedy_stats.greediness_early_terminations > 0 &&
            greedy_stats.score_point_evaluations < unfiltered_stats.score_point_evaluations,
        "greediness rejects poor poses before evaluating every model point");

  SearchParams optimized = threaded;
  optimized.num_levels = 0;
  optimized.greediness = 0.9;
  optimized.strict_detection = false;
  optimized.coarse_point_fraction = 0.75;
  optimized.enable_candidate_clustering = true;
  SearchStats optimized_stats;
  const auto optimized_results = find_shape_models(scene, model, optimized, &optimized_stats);
  check(!optimized_results.empty(), "clustered reduced-point pyramid search retains the target");
  check(optimized_stats.peak_input_candidates >= optimized_stats.peak_output_candidates,
        "candidate clustering reports non-expanding peak reduction");
  check(filtered_stats.nms_output_matches == filtered_results.size() &&
            filtered_stats.candidate_search_time_ms >= 0 && filtered_stats.nms_time_ms >= 0,
        "search statistics report final results and phase timings");

  ShapeModelParams incompatible_params = model.params();
  incompatible_params.model_version = ShapeModel::current_version + 1;
  const auto incompatible_model = create_shape_model(rectangle_image(), incompatible_params);
  check_throws<InvalidModel>([&] { find_shape_models(scene, incompatible_model, params); },
                             "matcher rejects an incompatible model version");
}
}

int main() {
  test_parameters_and_angles();
  test_image_and_edges();
  test_pyramids_and_model();
  test_precise_score_polarity();
  test_model_persistence();
  test_score_and_nms();
  test_matching_determinism();
  if (failures != 0) {
    std::cerr << failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all unit assertions passed\n";
  return 0;
}
