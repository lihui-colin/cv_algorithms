#pragma once

#include "openshape/matcher/matcher.hpp"

namespace openshape {

using ExhaustiveSearchParams = SearchParams;

// Correctness-first reference search.  The translation/angle/scale grids are
// enumerated deterministically, every candidate is scored with the complete
// fitted model, and only the final 3x3 neighborhood refinement is sub-pixel.
std::vector<MatchResult> find_shape_models_exhaustive(
    const ImageView& image, const ShapeModel& model,
    const SearchParams& params = {}, SearchStats* stats = nullptr);

std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& scene, const ShapeModel& model,
    const SearchParams& params = {}, SearchStats* stats = nullptr);

class ExhaustiveMatcher {
public:
  static std::vector<MatchResult> match(const ImageView& image,
                                        const ShapeModel& model,
                                        const SearchParams& params = {},
                                        SearchStats* stats = nullptr) {
    return find_shape_models_exhaustive(image, model, params, stats);
  }
  static std::vector<MatchResult> match(const EdgeMap& scene,
                                        const ShapeModel& model,
                                        const SearchParams& params = {},
                                        SearchStats* stats = nullptr) {
    return find_shape_models_exhaustive(scene, model, params, stats);
  }
};

inline std::vector<MatchResult> find_shape_models_exhaustive(
    const cv::Mat& image, const ShapeModel& model,
    const SearchParams& params = {}, SearchStats* stats = nullptr) {
  return find_shape_models_exhaustive(ImageView(image), model, params, stats);
}

}  // namespace openshape
