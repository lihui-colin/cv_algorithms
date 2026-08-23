#include "openshape/edge/edge_engine.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace openshape {
cv::Mat EdgeEngine::to_gray(const ImageView& image) {
  if (image.empty()) throw EmptyImage("image is empty");
  const cv::Mat& src = image.mat();
  if (src.depth() != CV_8U) throw InvalidArgument("image must have 8-bit depth");
  cv::Mat gray;
  if (src.channels() == 1) gray = src.clone();
  else if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
  else throw InvalidArgument("image must have 1, 3, or 4 channels");
  return gray;
}

EdgeMap EdgeEngine::compute(const ImageView& image, const ShapeModelParams& params,
                            bool continuous_fields) {
  params.validate();
  EdgeMap result;
  result.gray = to_gray(image);
  if (params.gaussian_kernel_size > 1)
    cv::GaussianBlur(result.gray, result.gray,
                     cv::Size(params.gaussian_kernel_size, params.gaussian_kernel_size),
                     params.gaussian_sigma, params.gaussian_sigma, params.sobel_border_type);
  cv::Sobel(result.gray, result.gx, CV_32F, 1, 0, params.sobel_kernel_size, 1.0, 0.0, params.sobel_border_type);
  cv::Sobel(result.gray, result.gy, CV_32F, 0, 1, params.sobel_kernel_size, 1.0, 0.0, params.sobel_border_type);
  cv::magnitude(result.gx, result.gy, result.magnitude);
  cv::phase(result.gx, result.gy, result.orientation, true);
  result.orientation *= static_cast<float>(CV_PI / 180.0);
  cv::Canny(result.gray, result.edges, params.canny_low_threshold, params.canny_high_threshold,
            params.canny_aperture_size, params.canny_l2_gradient);
  if (continuous_fields) {
    cv::Mat non_edges;
    cv::compare(result.edges, 0, non_edges, cv::CMP_EQ);
    cv::Mat distance;
    cv::distanceTransform(non_edges, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::exp(-distance, result.soft_edge_response);
    double maximum = 0.0;
    cv::minMaxLoc(result.magnitude, nullptr, &maximum);
    if (maximum > 0.0)
      result.magnitude.convertTo(result.normalized_magnitude, CV_32F, 1.0 / maximum);
    else
      result.normalized_magnitude = cv::Mat::zeros(result.magnitude.size(), CV_32FC1);

    // Fit the peak of the gradient magnitude along its local normal.  The
    // three-sample parabolic fit is inexpensive, deterministic and provides a
    // useful sub-pixel contour location even when Canny itself is quantized.
    result.subpixel_x = cv::Mat::zeros(result.gray.size(), CV_32FC1);
    result.subpixel_y = cv::Mat::zeros(result.gray.size(), CV_32FC1);
    result.fit_residual = cv::Mat::zeros(result.gray.size(), CV_32FC1);
    result.edge_polarity = cv::Mat::zeros(result.gray.size(), CV_32FC1);
    result.subpixel_edge_mask = cv::Mat::zeros(result.gray.size(), CV_8UC1);
    result.continuous_distance = distance.clone();
    for (int y = 1; y + 1 < result.gray.rows; ++y) {
      for (int x = 1; x + 1 < result.gray.cols; ++x) {
        if (result.edges.at<unsigned char>(y, x) == 0 &&
            result.magnitude.at<float>(y, x) < params.min_gradient_magnitude)
          continue;
        const float gx = result.gx.at<float>(y, x);
        const float gy = result.gy.at<float>(y, x);
        const float mag = result.magnitude.at<float>(y, x);
        if (!(mag > 1e-6f) || !std::isfinite(mag)) continue;
        const float nx = gx / mag, ny = gy / mag;
        auto sample_mag = [&](double sx, double sy) {
          const int ix = std::clamp(static_cast<int>(std::lround(sx)), 0, result.gray.cols - 1);
          const int iy = std::clamp(static_cast<int>(std::lround(sy)), 0, result.gray.rows - 1);
          return static_cast<double>(result.magnitude.at<float>(iy, ix));
        };
        const double center = mag;
        const double before = sample_mag(x - nx, y - ny);
        const double after = sample_mag(x + nx, y + ny);
        const double denominator = before - 2.0 * center + after;
        double offset = 0.0;
        if (denominator < -1e-9)
          offset = std::clamp(0.5 * (before - after) / denominator, -0.5, 0.5);
        const double fitted = center + 0.5 * offset * (before - after);
        const double residual = std::abs(center - fitted) /
            std::max(1.0, std::abs(center));
        if (!std::isfinite(residual) || residual > params.max_fit_residual) continue;
        result.subpixel_x.at<float>(y, x) = static_cast<float>(x + offset * nx);
        result.subpixel_y.at<float>(y, x) = static_cast<float>(y + offset * ny);
        result.fit_residual.at<float>(y, x) = static_cast<float>(residual);
        result.edge_polarity.at<float>(y, x) = gx * nx + gy * ny >= 0 ? 1.0f : -1.0f;
        result.subpixel_edge_mask.at<unsigned char>(y, x) = 255;
      }
    }
  }
  cv::integral(result.edges, result.edge_integral, CV_32S);
  return result;
}
}
