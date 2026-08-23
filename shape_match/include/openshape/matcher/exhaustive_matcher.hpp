#pragma once

#include "openshape/matcher/matcher.hpp"

namespace openshape {

using ExhaustiveSearchParams = SearchParams;

// Reusable, immutable preparation for exhaustive searches.  The implementation
// is intentionally opaque so ISA-specific tables and worker scratch remain an
// ABI-private detail of the library.
class ExhaustiveSearchWorkspace {
public:
  ExhaustiveSearchWorkspace();
  ~ExhaustiveSearchWorkspace();
  ExhaustiveSearchWorkspace(ExhaustiveSearchWorkspace&&) noexcept;
  ExhaustiveSearchWorkspace& operator=(ExhaustiveSearchWorkspace&&) noexcept;
  ExhaustiveSearchWorkspace(const ExhaustiveSearchWorkspace&) = delete;
  ExhaustiveSearchWorkspace& operator=(const ExhaustiveSearchWorkspace&) = delete;

  void prepare(const ShapeModel& model, const SearchParams& params);
  void clear();
  std::size_t memory_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend std::vector<MatchResult> find_shape_models_exhaustive(
      const EdgeMap&, const ShapeModel&, const SearchParams&,
      ExhaustiveSearchWorkspace*, SearchStats*);
};

// Correctness-first reference search.  The translation/angle/scale grids are
// enumerated deterministically, every candidate is scored with the complete
// fitted model, and only the final 3x3 neighborhood refinement is sub-pixel.
std::vector<MatchResult> find_shape_models_exhaustive(
    const ImageView& image, const ShapeModel& model,
    const SearchParams& params = {}, SearchStats* stats = nullptr);

std::vector<MatchResult> find_shape_models_exhaustive(
    const ImageView& image, const ShapeModel& model,
    const SearchParams& params, ExhaustiveSearchWorkspace* workspace,
    SearchStats* stats = nullptr);

inline std::vector<MatchResult> find_shape_models_exhaustive(
    const ImageView& image, const ShapeModel& model,
    ExhaustiveSearchWorkspace& workspace, const SearchParams& params = {},
    SearchStats* stats = nullptr) {
  return find_shape_models_exhaustive(image, model, params, &workspace, stats);
}

std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& scene, const ShapeModel& model,
    const SearchParams& params = {}, SearchStats* stats = nullptr);

// Workspace-aware overload.  The workspace may be shared by concurrent calls
// after prepare() returns; each call owns its result and worker scratch.
std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& scene, const ShapeModel& model,
    const SearchParams& params, ExhaustiveSearchWorkspace* workspace,
    SearchStats* stats = nullptr);

inline std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& scene, const ShapeModel& model,
    ExhaustiveSearchWorkspace& workspace, const SearchParams& params = {},
    SearchStats* stats = nullptr) {
  return find_shape_models_exhaustive(scene, model, params, &workspace, stats);
}

inline std::vector<MatchResult> find_shape_models_exhaustive(
    const EdgeMap& scene, const ShapeModel& model,
    ExhaustiveSearchWorkspace* workspace, const SearchParams& params = {},
    SearchStats* stats = nullptr) {
  return find_shape_models_exhaustive(scene, model, params, workspace, stats);
}

class ExhaustiveMatcher {
public:
  static std::vector<MatchResult> match(const ImageView& image,
                                        const ShapeModel& model,
                                        const SearchParams& params = {},
                                        SearchStats* stats = nullptr) {
    return find_shape_models_exhaustive(image, model, params, stats);
  }
  static std::vector<MatchResult> match(const ImageView& image,
                                        const ShapeModel& model,
                                        const SearchParams& params,
                                        ExhaustiveSearchWorkspace* workspace,
                                        SearchStats* stats = nullptr) {
    return find_shape_models_exhaustive(image, model, params, workspace, stats);
  }
  static std::vector<MatchResult> match(const EdgeMap& scene,
                                        const ShapeModel& model,
                                        const SearchParams& params = {},
                                        SearchStats* stats = nullptr) {
    return find_shape_models_exhaustive(scene, model, params, stats);
  }
  static std::vector<MatchResult> match(const EdgeMap& scene,
                                        const ShapeModel& model,
                                        const SearchParams& params,
                                        ExhaustiveSearchWorkspace* workspace,
                                        SearchStats* stats = nullptr) {
    return find_shape_models_exhaustive(scene, model, params, workspace, stats);
  }
};

inline std::vector<MatchResult> find_shape_models_exhaustive(
    const cv::Mat& image, const ShapeModel& model,
    const SearchParams& params = {}, SearchStats* stats = nullptr) {
  return find_shape_models_exhaustive(ImageView(image), model, params, stats);
}

inline std::vector<MatchResult> find_shape_models_exhaustive(
    const cv::Mat& image, const ShapeModel& model,
    const SearchParams& params, ExhaustiveSearchWorkspace* workspace,
    SearchStats* stats = nullptr) {
  return find_shape_models_exhaustive(ImageView(image), model, params,
                                      workspace, stats);
}

}  // namespace openshape
