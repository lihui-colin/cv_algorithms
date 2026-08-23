#pragma once
#include "openshape/core/cpu_features.hpp"
#include "openshape/core/types.hpp"
#include "openshape/model/shape_model.hpp"
#include "openshape/pyramid/pyramid.hpp"

namespace openshape {
class ExhaustiveSearchWorkspace;
std::vector<MatchResult> find_shape_models(const ImageView& image,
                                           const ShapeModel& model,
                                           const SearchParams& params = {},
                                           SearchStats* stats = nullptr);
inline std::vector<MatchResult> find_shape_models(const cv::Mat& image,
                                                  const ShapeModel& model,
                                                  const SearchParams& params = {},
                                                  SearchStats* stats = nullptr) {
  return find_shape_models(ImageView(image), model, params, stats);
}
double score_pose(const EdgeMap& scene, const std::vector<ModelPoint>& points,
                  double column, double row, double angle_degrees,
                  std::size_t* valid_count = nullptr);
double score_pose(const EdgeMap& scene, const std::vector<ModelPoint>& points,
                  double column, double row, double angle_degrees, double scale,
                  std::size_t* valid_count = nullptr);
double score_pose(const EdgeMap& scene, const ModelLevelSoA& points,
                  double column, double row, double angle_degrees,
                  std::size_t* valid_count = nullptr);
double score_pose(const EdgeMap& scene, const ModelLevelSoA& points,
                  double column, double row, double angle_degrees, double scale,
                  std::size_t* valid_count = nullptr);
// Smooth, bilinearly sampled score used by continuous pose refinement.
// Build its EdgeMap with EdgeEngine::compute(..., true).
double score_pose_precise(const EdgeMap& scene, const ModelLevelSoA& points,
                          double column, double row, double angle_degrees,
                          double scale = 1.0,
                          PolarityMode polarity = PolarityMode::Same,
                          double edge_distance_sigma = 1.0,
                          std::size_t* valid_count = nullptr);
// Reuse an already computed edge map when matching several models or running
// repeated searches over the same image. EdgeMap must come from EdgeEngine.
std::vector<MatchResult> find_shape_models(const EdgeMap& scene,
                                           const ShapeModel& model,
                                           const SearchParams& params = {},
                                           SearchStats* stats = nullptr);
std::vector<MatchResult> find_shape_models(const ImageView& image,
                                           const ShapeModel& model,
                                           const SearchParams& params,
                                           ExhaustiveSearchWorkspace* workspace,
                                           SearchStats* stats = nullptr);
std::vector<MatchResult> find_shape_models(const EdgeMap& scene,
                                           const ShapeModel& model,
                                           const SearchParams& params,
                                           ExhaustiveSearchWorkspace* workspace,
                                           SearchStats* stats = nullptr);
std::vector<MatchResult> find_shape_models(const EdgePyramid& scene,
                                           const ShapeModel& model,
                                           const SearchParams& params = {},
                                           SearchStats* stats = nullptr);
inline std::vector<MatchResult> find_shape_models(const cv::Mat& image,
                                                  const ShapeModel& model,
                                                  const SearchParams& params,
                                                  ExhaustiveSearchWorkspace* workspace,
                                                  SearchStats* stats = nullptr) {
  return find_shape_models(ImageView(image), model, params, workspace, stats);
}
double bounding_box_overlap(const ShapeModel& model, const MatchResult& a,
                            const MatchResult& b);
std::vector<MatchResult> non_max_suppression(const ShapeModel& model,
                                             std::vector<MatchResult> candidates,
                                             double max_overlap,
                                             std::size_t num_matches = 0);
void draw_match_results(cv::Mat& image, const ShapeModel& model,
                        const std::vector<MatchResult>& results,
                        const cv::Scalar& color = cv::Scalar(0, 255, 0),
                        int thickness = 2);
}
