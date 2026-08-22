#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include "openshape/core/error.hpp"

namespace openshape {

inline double normalize_angle(double degrees) {
  double a = std::fmod(degrees + 180.0, 360.0);
  if (a < 0) a += 360.0;
  return a - 180.0;
}

struct ImageView {
  ImageView() = default;
  explicit ImageView(const cv::Mat& image) : mat_(image) {}
  ImageView(const cv::Mat& image, const cv::Rect& roi) {
    if (image.empty()) throw EmptyImage("image is empty");
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        roi.x + roi.width > image.cols || roi.y + roi.height > image.rows)
      throw InvalidArgument("ROI is outside the image");
    mat_ = image(roi);
  }
  const cv::Mat& mat() const { return mat_; }
  bool empty() const { return mat_.empty(); }
  int rows() const { return mat_.rows; }
  int cols() const { return mat_.cols; }
private:
  cv::Mat mat_;
};

class Image {
public:
  Image() = default;
  explicit Image(const cv::Mat& image) : mat_(image.clone()) {}
  static Image read(const std::string& path);
  const cv::Mat& mat() const { return mat_; }
  cv::Mat& mat() { return mat_; }
  ImageView view() const { return ImageView(mat_); }
  bool empty() const { return mat_.empty(); }
private:
  cv::Mat mat_;
};

struct EdgePoint {
  float x = 0, y = 0;
  float magnitude = 0;
  float orientation = 0;
};

struct ModelPoint {
  float relative_x = 0, relative_y = 0;
  float orientation = 0;
  float weight = 0;
  int level = 0;
};

// Gradient-polarity policy used by the high-accuracy matcher.  The historical
// behavior is Same, so adding this field does not change existing searches.
enum class PolarityMode {
  Same,
  Inverted,
  GlobalEither,
  LocalEither
};

struct ShapeModelParams {
  cv::Rect roi{};
  cv::Point2f origin{-1.f, -1.f};
  double canny_low_threshold = 50.0;
  double canny_high_threshold = 150.0;
  int canny_aperture_size = 3;
  bool canny_l2_gradient = true;
  int sobel_kernel_size = 3;
  int sobel_border_type = cv::BORDER_REFLECT_101;
  int gaussian_kernel_size = 1;
  double gaussian_sigma = 0.0;
  double min_gradient_magnitude = 10.0;
  std::size_t min_model_points = 4;
  std::size_t max_model_points = 1000;
  double min_point_distance = 2.0;
  int num_levels = 0;
  std::string model_point_sampling = "uniform";
  int model_version = 1;
  void validate() const;
};

struct SearchParams {
  double angle_start = 0.0;
  double angle_extent = 0.0;
  double angle_step = 1.0;
  double scale_min = 1.0;
  double scale_max = 1.0;
  double scale_step = 0.05;
  int num_levels = 0;
  double min_score = 0.5;
  std::size_t num_matches = 1;
  double max_overlap = 0.5;
  cv::Rect search_roi{};
  std::size_t coarse_candidate_limit = 64;
  int refinement_radius = 2;
  int num_threads = 0;
  bool deterministic = true;
  // At the global/coarsest level, reject poses that cannot reach the minimum
  // valid Canny-point count before running the orientation score kernel.
  bool enable_coarse_prefilter = true;
  // Refine final-level row/column coordinates with a continuous local score.
  // Disabled by default to preserve the historical integer-pixel search.
  bool enable_subpixel = false;
  // Jointly refine column, row, angle, and scale with the continuous edge
  // objective.  This is an opt-in, versioned behavior extension.
  bool enable_pose_refinement = false;
  // Correctness-first mode. When enabled, all model points are retained,
  // candidate clustering/early termination are disabled, and final poses are
  // revalidated against the complete model. This is the default.
  bool strict_detection = true;
  // Safe score-upper-bound pruning. Zero disables it; values near one check
  // the bound more frequently and trade a little branch overhead for faster
  // rejection of poor orientation matches.
  double greediness = 0.0;
  // Fraction of the strongest model points retained per additional coarse
  // pyramid level. One preserves the historical full-point behavior.
  double coarse_point_fraction = 1.0;
  bool enable_candidate_clustering = false;
  double peak_position_tolerance = 2.0;
  double peak_angle_tolerance = 0.0;
  double peak_scale_tolerance = 0.0;
  // Maximum clustered final candidates sent to continuous optimization.
  // Zero means unlimited.
  std::size_t refinement_candidate_limit = 32;
  std::size_t refinement_variants_per_peak = 3;
  PolarityMode polarity = PolarityMode::Same;
  // Minimum fraction of model points that must have image support.  The
  // default is identical to the historical fixed points.size()/4 rule.
  double min_visible_fraction = 0.25;
  // Soft-edge falloff in pixels for continuous scoring.
  double edge_distance_sigma = 1.0;
  int max_refinement_iterations = 12;
  double refinement_position_tolerance = 1e-3;
  double refinement_angle_tolerance = 1e-3;
  double refinement_scale_tolerance = 1e-5;
  void enable_fast_pipeline();
  void validate() const;
};

struct SearchStats {
  std::size_t angle_candidate_count = 0;
  std::size_t scale_candidate_count = 0;
  std::size_t transform_candidate_count = 0;
  std::size_t pose_evaluations = 0;
  std::size_t prefilter_evaluations = 0;
  std::size_t prefilter_rejections = 0;
  std::size_t integral_prefilter_rejections = 0;
  std::size_t full_score_evaluations = 0;
  std::size_t accepted_candidates = 0;
  std::size_t nms_input_candidates = 0;
  std::size_t nms_output_matches = 0;
  double preprocessing_time_ms = 0.0;
  double transform_preparation_time_ms = 0.0;
  // Compatibility field retained from 0.1; equal to
  // transform_preparation_time_ms in scale-aware searches.
  double rotation_preparation_time_ms = 0.0;
  double prefilter_cpu_time_ms = 0.0;
  double score_cpu_time_ms = 0.0;
  double continuous_refinement_time_ms = 0.0;
  double candidate_search_time_ms = 0.0;
  double nms_time_ms = 0.0;
  std::size_t peak_candidate_count = 0;
  std::size_t refinement_evaluations = 0;
  std::size_t refinement_converged = 0;
  std::size_t refinement_fallbacks = 0;
  std::size_t score_point_evaluations = 0;
  std::size_t greediness_early_terminations = 0;
  std::size_t peak_input_candidates = 0;
  std::size_t peak_output_candidates = 0;
  std::size_t refinement_input_candidates = 0;
};

struct Pose { float column = 0, row = 0, angle = 0, scale = 1; };

struct MatchResult {
  double row = 0, column = 0, angle = 0, scale = 1.0, score = 0;
  std::size_t valid_point_count = 0, model_point_count = 0;
  int level_used = 0;
  bool refined = false;
  bool refinement_converged = false;
  double residual = 1.0;
  double confidence = 0.0;
  double valid_point_fraction = 0.0;
};

}
