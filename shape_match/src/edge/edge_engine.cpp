#include "openshape/edge/edge_engine.hpp"
#include <opencv2/imgproc.hpp>

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
  }
  cv::integral(result.edges, result.edge_integral, CV_32S);
  return result;
}
}
