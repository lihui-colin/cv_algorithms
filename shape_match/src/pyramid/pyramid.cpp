#include "openshape/pyramid/pyramid.hpp"
#include "openshape/edge/edge_engine.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace openshape {
namespace {
int auto_levels(const cv::Size& size) {
  int levels = 1; int w = size.width, h = size.height;
  while (w >= 32 && h >= 32 && levels < 5) { w = (w + 1) / 2; h = (h + 1) / 2; ++levels; }
  return levels;
}

void validate_edge_map(const EdgeMap& map) {
  if (map.gray.empty() || map.edges.empty() || map.orientation.empty())
    throw InvalidArgument("scene edge map is incomplete");
  if (map.gray.type() != CV_8UC1 || map.edges.type() != CV_8UC1 ||
      map.orientation.type() != CV_32FC1 || map.gray.size() != map.edges.size() ||
      map.gray.size() != map.orientation.size())
    throw InvalidArgument("scene edge map has incompatible matrix types or sizes");
}
}

ImagePyramid ImagePyramid::build(const ImageView& image, int num_levels) {
  cv::Mat gray = EdgeEngine::to_gray(image);
  if (num_levels < 0) throw InvalidArgument("num_levels cannot be negative");
  const int count = num_levels == 0 ? auto_levels(gray.size()) : num_levels;
  if (count < 1) throw InvalidArgument("num_levels must be at least 1");
  ImagePyramid pyramid; pyramid.levels_.push_back(gray);
  for (int i = 1; i < count; ++i) {
    if (pyramid.levels_.back().cols < 2 || pyramid.levels_.back().rows < 2) break;
    cv::Mat next; cv::pyrDown(pyramid.levels_.back(), next); pyramid.levels_.push_back(next);
  }
  return pyramid;
}

ModelPyramid ModelPyramid::build(const ShapeModel& model, int num_levels) {
  if (model.empty()) throw InvalidModel("cannot build a pyramid from an empty model");
  if (num_levels < 0) throw InvalidArgument("num_levels cannot be negative");
  int count = num_levels;
  if (count == 0) count = model.params_.num_levels;
  if (count == 0) {
    count = 1;
    int w = model.roi_.width, h = model.roi_.height;
    while (w >= 32 && h >= 32 && count < 5) {
      w = (w + 1) / 2; h = (h + 1) / 2; ++count;
    }
  }
  ModelPyramid pyramid; pyramid.levels_.reserve(count);
  for (int level = 0; level < count; ++level) {
    const float scale = std::ldexp(1.0f, -level);
    std::vector<ModelPoint> points; points.reserve(model.points_.size());
    for (const auto& p : model.points_) {
      ModelPoint q = p; q.relative_x *= scale; q.relative_y *= scale; q.level = level; points.push_back(q);
    }
    pyramid.levels_.push_back(std::move(points));
  }
  return pyramid;
}

EdgePyramid EdgePyramid::build(const ImageView& image, const ShapeModelParams& params,
                               int num_levels, bool continuous_fields,
                               bool subpixel_fields) {
  params.validate();
  EdgeMap level_zero = EdgeEngine::compute(
      image, params, continuous_fields, subpixel_fields);
  return build(level_zero, params, num_levels);
}

EdgePyramid EdgePyramid::build(const EdgeMap& level_zero, const ShapeModelParams& params,
                               int num_levels) {
  params.validate();
  validate_edge_map(level_zero);
  if (num_levels < 0) throw InvalidArgument("num_levels cannot be negative");
  int count = num_levels;
  if (count == 0) count = params.num_levels;
  if (count == 0) count = auto_levels(level_zero.gray.size());
  if (count < 1) throw InvalidArgument("num_levels must be at least 1");
  EdgePyramid pyramid;
  pyramid.levels_.reserve(count);
  pyramid.levels_.push_back(level_zero);
  for (int level = 1; level < count; ++level) {
    const cv::Mat& previous_gray = pyramid.levels_.back().gray;
    if (previous_gray.cols < 2 || previous_gray.rows < 2) break;
    cv::Mat down;
    cv::pyrDown(previous_gray, down);
    pyramid.levels_.push_back(EdgeEngine::compute(
        ImageView(down), params, false, false));
  }
  return pyramid;
}
}
