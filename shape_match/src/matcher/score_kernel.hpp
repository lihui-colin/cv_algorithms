#pragma once
#include "openshape/core/types.hpp"
#include "openshape/model/shape_model.hpp"

namespace openshape {
double score_pose_soa_avx2(const EdgeMap& scene, const ModelLevelSoA& points,
                           double column, double row, double angle_degrees,
                           std::size_t* valid_count);
}
