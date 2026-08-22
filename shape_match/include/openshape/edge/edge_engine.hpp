#pragma once
#include "openshape/core/types.hpp"

namespace openshape {
struct EdgeMap {
  cv::Mat gray, gx, gy, magnitude, orientation, edges;
  // CV_32FC1 response in [0,1], equal to one on a Canny edge and decaying
  // smoothly with Euclidean distance. Used only by opt-in precise scoring.
  cv::Mat soft_edge_response, normalized_magnitude;
  // CV_32SC1 summed-area table of the Canny mask, with one-pixel top/left
  // padding as produced by cv::integral.
  cv::Mat edge_integral;
};

class EdgeEngine {
public:
  static cv::Mat to_gray(const ImageView& image);
  static EdgeMap compute(const ImageView& image, const ShapeModelParams& params,
                         bool continuous_fields = false);
  static EdgeMap compute(const cv::Mat& image, const ShapeModelParams& params,
                         bool continuous_fields = false) {
    return compute(ImageView(image), params, continuous_fields);
  }
};
}
