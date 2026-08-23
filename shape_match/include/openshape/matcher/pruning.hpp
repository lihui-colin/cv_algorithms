#pragma once

#include "openshape/core/types.hpp"
#include "openshape/model/shape_model.hpp"

namespace openshape {

enum class PruningReason {
  None,
  ScoreUpperBound,
  VisibleUpperBound,
};

struct PrunableScoreResult {
  double score = 0.0;
  double score_upper_bound = 1.0;
  std::size_t valid_point_count = 0;
  std::size_t point_evaluations = 0;
  PruningReason reason = PruningReason::None;

  bool terminated() const { return reason != PruningReason::None; }
};

struct PrunableScoreWorkspace {
  std::vector<double> weighted_contribution;
  std::vector<double> inverted_contribution;
};

// Builds a deterministic evaluation permutation for non-uniform models.
// Stronger weights are evaluated first and equal-weight groups are interleaved
// without removing points. Uniform models retain their cache-friendly order.
std::vector<std::size_t> build_pruning_point_order(
    const ModelLevelSoA& points);

// Precise scorer with certified early rejection. When enable_pruning is
// false, this is exactly the complete precise score used by the exhaustive
// reference matcher. A terminated result is a rejection certificate, not a
// partial score that may be accepted.
PrunableScoreResult score_pose_precise_prunable(
    const EdgeMap& scene, const ModelLevelSoA& points,
    double column, double row, double angle_degrees, double scale,
    PolarityMode polarity, double edge_distance_sigma,
    double minimum_score, std::size_t required_visible_count,
    bool enable_pruning,
    const std::vector<std::size_t>* point_order = nullptr,
    PrunableScoreWorkspace* workspace = nullptr);

}  // namespace openshape
