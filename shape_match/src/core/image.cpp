#include "openshape/core/types.hpp"
#include <opencv2/imgcodecs.hpp>

namespace openshape {
Image Image::read(const std::string& path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (image.empty()) throw EmptyImage("unable to read image: " + path);
  return Image(image);
}

void ShapeModelParams::validate() const {
  if (canny_low_threshold < 0 || canny_high_threshold <= canny_low_threshold)
    throw InvalidArgument("Canny thresholds must satisfy 0 <= low < high");
  if (canny_aperture_size != 3 && canny_aperture_size != 5 && canny_aperture_size != 7)
    throw InvalidArgument("Canny aperture size must be 3, 5, or 7");
  if (sobel_kernel_size != 1 && sobel_kernel_size != 3 && sobel_kernel_size != 5 && sobel_kernel_size != 7)
    throw InvalidArgument("Sobel kernel size must be 1, 3, 5, or 7");
  if (gaussian_kernel_size < 1 || (gaussian_kernel_size > 1 && gaussian_kernel_size % 2 == 0) || gaussian_sigma < 0)
    throw InvalidArgument("Gaussian kernel must be 1 or a positive odd number and sigma non-negative");
  if (min_model_points == 0 || max_model_points < min_model_points)
    throw InvalidArgument("invalid model point limits");
  if (min_point_distance < 0 || min_gradient_magnitude < 0)
    throw InvalidArgument("minimum distances and magnitudes must be non-negative");
  if (num_levels < 0) throw InvalidArgument("num_levels cannot be negative");
  if (model_version <= 0) throw InvalidArgument("model_version must be positive");
  if (model_point_sampling != "uniform" && model_point_sampling != "magnitude")
    throw InvalidArgument("model_point_sampling must be uniform or magnitude");
}

void SearchParams::validate() const {
  if (angle_extent < 0 || angle_step <= 0) throw InvalidArgument("invalid angle range or step");
  if (!std::isfinite(scale_min) || !std::isfinite(scale_max) || !std::isfinite(scale_step) ||
      scale_min <= 0 || scale_max < scale_min || scale_step <= 0)
    throw InvalidArgument("scale range must satisfy 0 < scale_min <= scale_max and scale_step > 0");
  if (min_score < 0 || min_score > 1) throw InvalidArgument("min_score must be in [0, 1]");
  if (max_overlap < 0 || max_overlap > 1) throw InvalidArgument("max_overlap must be in [0, 1]");
  if (coarse_candidate_limit == 0) throw InvalidArgument("coarse_candidate_limit must be positive");
  if (refinement_variants_per_peak == 0)
    throw InvalidArgument("refinement_variants_per_peak must be positive");
  if (refinement_radius < 0 || num_threads < 0) throw InvalidArgument("invalid refinement radius or thread count");
  if (num_levels < 0) throw InvalidArgument("num_levels cannot be negative");
  if (!std::isfinite(min_visible_fraction) || min_visible_fraction <= 0 || min_visible_fraction > 1)
    throw InvalidArgument("min_visible_fraction must be in (0, 1]");
  if (!std::isfinite(greediness) || greediness < 0 || greediness > 1)
    throw InvalidArgument("greediness must be in [0, 1]");
  if (!std::isfinite(coarse_point_fraction) || coarse_point_fraction <= 0 ||
      coarse_point_fraction > 1)
    throw InvalidArgument("coarse_point_fraction must be in (0, 1]");
  if (!std::isfinite(peak_position_tolerance) || peak_position_tolerance < 0 ||
      !std::isfinite(peak_angle_tolerance) || peak_angle_tolerance < 0 ||
      !std::isfinite(peak_scale_tolerance) || peak_scale_tolerance < 0)
    throw InvalidArgument("peak tolerances must be finite and non-negative");
  if (!std::isfinite(edge_distance_sigma) || edge_distance_sigma <= 0)
    throw InvalidArgument("edge_distance_sigma must be positive and finite");
  if (max_refinement_iterations <= 0)
    throw InvalidArgument("max_refinement_iterations must be positive");
  if (!std::isfinite(refinement_position_tolerance) || refinement_position_tolerance <= 0 ||
      !std::isfinite(refinement_angle_tolerance) || refinement_angle_tolerance <= 0 ||
      !std::isfinite(refinement_scale_tolerance) || refinement_scale_tolerance <= 0)
    throw InvalidArgument("refinement tolerances must be positive and finite");
}

void SearchParams::enable_fast_pipeline() {
  strict_detection = false;
  greediness = 0.9;
  coarse_point_fraction = 0.75;
  enable_candidate_clustering = true;
  refinement_candidate_limit = 32;
}
}
