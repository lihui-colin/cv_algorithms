#pragma once
#include "openshape/core/types.hpp"
#include "openshape/edge/edge_engine.hpp"

namespace openshape {

// Stable public ModelPoint storage is kept as an AoS for API compatibility.
// The matcher consumes this cache to avoid repeatedly de-interleaving points
// in the hot score loop.
struct ModelLevelSoA {
  std::vector<float> relative_x;
  std::vector<float> relative_y;
  std::vector<float> orientation;
  std::vector<float> weight;
  std::vector<float> fit_residual;
  std::vector<int> polarity;
  std::vector<int> region;
  int level = 0;

  std::size_t size() const { return relative_x.size(); }
  bool empty() const { return relative_x.empty(); }
};

class ShapeModel {
public:
  static constexpr int current_version = 2;
  ShapeModel() = default;
  const std::vector<ModelPoint>& points() const { return points_; }
  const std::vector<std::vector<ModelPoint>>& levels() const { return levels_; }
  const std::vector<ModelLevelSoA>& levels_soa() const { return levels_soa_; }
  const cv::Rect& roi() const { return roi_; }
  cv::Point2f origin() const { return origin_; }
  int version() const { return version_; }
  const ShapeModelParams& params() const { return params_; }
  bool empty() const { return points_.empty(); }
  std::size_t size() const { return points_.size(); }
  void save(const std::string& path) const;
  static ShapeModel load(const std::string& path);
private:
  friend class ShapeModelBuilder;
  friend class ModelPyramid;
  cv::Rect roi_{};
  cv::Point2f origin_{};
  int version_ = current_version;
  ShapeModelParams params_{};
  std::vector<ModelPoint> points_;
  std::vector<std::vector<ModelPoint>> levels_;
  std::vector<ModelLevelSoA> levels_soa_;
};

class ShapeModelBuilder {
public:
  static ShapeModel create(const ImageView& image, const ShapeModelParams& params = {});
  static ShapeModel create(const cv::Mat& image, const ShapeModelParams& params = {}) {
    return create(ImageView(image), params);
  }
};

inline ShapeModel create_shape_model(const ImageView& image, const ShapeModelParams& params = {}) {
  return ShapeModelBuilder::create(image, params);
}

inline ShapeModel create_shape_model(const cv::Mat& image, const ShapeModelParams& params = {}) {
  return ShapeModelBuilder::create(image, params);
}
}
