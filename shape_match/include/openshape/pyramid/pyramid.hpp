#pragma once
#include "openshape/core/types.hpp"
#include "openshape/edge/edge_engine.hpp"
#include "openshape/model/shape_model.hpp"

namespace openshape {
class ImagePyramid {
public:
  static ImagePyramid build(const ImageView& image, int num_levels = 0);
  const std::vector<cv::Mat>& levels() const { return levels_; }
private:
  std::vector<cv::Mat> levels_;
};

class ModelPyramid {
public:
  static ModelPyramid build(const ShapeModel& model, int num_levels = 0);
  const std::vector<std::vector<ModelPoint>>& levels() const { return levels_; }
private:
  std::vector<std::vector<ModelPoint>> levels_;
};

// Prepared scene edge/gradient pyramid. Matrix storage is immutable after
// construction, so one instance can safely be shared by concurrent searches.
class EdgePyramid {
public:
  static EdgePyramid build(const ImageView& image, const ShapeModelParams& params,
                           int num_levels = 0, bool continuous_fields = false,
                           bool subpixel_fields = true);
  static EdgePyramid build(const EdgeMap& level_zero, const ShapeModelParams& params,
                           int num_levels = 0);
  const std::vector<EdgeMap>& levels() const { return levels_; }
  bool empty() const { return levels_.empty(); }
private:
  std::vector<EdgeMap> levels_;
};
}
