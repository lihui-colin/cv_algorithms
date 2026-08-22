#include "score_kernel.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace openshape {
namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

float angle_diff_scalar(float a, float b) {
  float d = std::fmod(a - b + kPi, kTwoPi);
  if (d < 0.0f) d += kTwoPi;
  return d - kPi;
}

float score_single(const EdgeMap& scene, const ModelLevelSoA& points, std::size_t index,
                   double column, double row, double angle, std::size_t* valid) {
  const float c = static_cast<float>(std::cos(angle));
  const float s = static_cast<float>(std::sin(angle));
  const int x = static_cast<int>(std::lround(column + c * points.relative_x[index] -
                                             s * points.relative_y[index]));
  const int y = static_cast<int>(std::lround(row + s * points.relative_x[index] +
                                             c * points.relative_y[index]));
  if (x < 0 || y < 0 || x >= scene.edges.cols || y >= scene.edges.rows) return 0.0f;
  if (scene.edges.at<unsigned char>(y, x) == 0) return 0.0f;
  const float difference = angle_diff_scalar(points.orientation[index] + static_cast<float>(angle),
                                             scene.orientation.at<float>(y, x));
  if (valid) ++*valid;
  return std::max(0.0f, std::cos(difference));
}

float horizontal_sum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehdup_ps(sum));
  sum = _mm_add_ss(sum, _mm_movehl_ps(_mm_setzero_ps(), sum));
  return _mm_cvtss_f32(sum);
}

__m256 cosine_positive(__m256 difference) {
  const __m256 z = _mm256_mul_ps(difference, difference);
  // Taylor polynomial through x^12. On [-pi, pi] its error is below the
  // score regression tolerance, while avoiding a platform-specific cosf ABI.
  __m256 result = _mm256_set1_ps(1.0f / 479001600.0f);
  result = _mm256_add_ps(_mm256_set1_ps(-1.0f / 3628800.0f), _mm256_mul_ps(z, result));
  result = _mm256_add_ps(_mm256_set1_ps(1.0f / 40320.0f), _mm256_mul_ps(z, result));
  result = _mm256_add_ps(_mm256_set1_ps(-1.0f / 720.0f), _mm256_mul_ps(z, result));
  result = _mm256_add_ps(_mm256_set1_ps(1.0f / 24.0f), _mm256_mul_ps(z, result));
  result = _mm256_add_ps(_mm256_set1_ps(-0.5f), _mm256_mul_ps(z, result));
  result = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(z, result));
  return _mm256_max_ps(result, _mm256_setzero_ps());
}
}

double score_pose_soa_avx2(const EdgeMap& scene, const ModelLevelSoA& points,
                           double column, double row, double angle_degrees,
                           std::size_t* valid_count) {
  if (valid_count) *valid_count = 0;
  if (points.empty() || scene.edges.empty()) return 0.0;
  const double angle = angle_degrees * CV_PI / 180.0;
  const float angle_f = static_cast<float>(angle);
  const float c = static_cast<float>(std::cos(angle));
  const float s = static_cast<float>(std::sin(angle));
  double weighted_scalar = 0.0;
  double total_scalar = 0.0;
  std::size_t valid = 0;
  const std::size_t count = points.size();
  const __m256 column_v = _mm256_set1_ps(static_cast<float>(column));
  const __m256 row_v = _mm256_set1_ps(static_cast<float>(row));
  const __m256 c_v = _mm256_set1_ps(c);
  const __m256 s_v = _mm256_set1_ps(s);
  const __m256 angle_v = _mm256_set1_ps(angle_f);
  const __m256 pi_v = _mm256_set1_ps(kPi);
  const __m256 two_pi_v = _mm256_set1_ps(kTwoPi);
  const int edge_step = static_cast<int>(scene.edges.step);
  const int orientation_step = static_cast<int>(scene.orientation.step);

  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 x_model = _mm256_loadu_ps(points.relative_x.data() + index);
    const __m256 y_model = _mm256_loadu_ps(points.relative_y.data() + index);
    const __m256 x_value = _mm256_add_ps(column_v,
        _mm256_sub_ps(_mm256_mul_ps(c_v, x_model), _mm256_mul_ps(s_v, y_model)));
    const __m256 y_value = _mm256_add_ps(row_v,
        _mm256_add_ps(_mm256_mul_ps(s_v, x_model), _mm256_mul_ps(c_v, y_model)));
    alignas(32) float vector_xf[8], vector_yf[8];
    alignas(32) int vector_x[8], vector_y[8];
    _mm256_store_ps(vector_xf, x_value);
    _mm256_store_ps(vector_yf, y_value);
    _mm256_store_si256(reinterpret_cast<__m256i*>(vector_x), _mm256_cvttps_epi32(
        _mm256_add_ps(x_value, _mm256_or_ps(_mm256_set1_ps(0.5f),
                                             _mm256_and_ps(x_value, _mm256_set1_ps(-0.0f))))));
    _mm256_store_si256(reinterpret_cast<__m256i*>(vector_y), _mm256_cvttps_epi32(
        _mm256_add_ps(y_value, _mm256_or_ps(_mm256_set1_ps(0.5f),
                                             _mm256_and_ps(y_value, _mm256_set1_ps(-0.0f))))));

    alignas(32) int gather_index[8], mask_value[8];
    for (int lane = 0; lane < 8; ++lane) {
      const std::size_t point_index = index + static_cast<std::size_t>(lane);
      const float x_fraction = std::fabs(vector_xf[lane] - std::trunc(vector_xf[lane]));
      const float y_fraction = std::fabs(vector_yf[lane] - std::trunc(vector_yf[lane]));
      const bool near_rounding_boundary = std::fabs(x_fraction - 0.5f) < 1e-4f ||
                                          std::fabs(y_fraction - 0.5f) < 1e-4f;
      int sample_x = vector_x[lane];
      int sample_y = vector_y[lane];
      bool in_bounds = sample_x >= 0 && sample_y >= 0 &&
                       sample_x < scene.edges.cols && sample_y < scene.edges.rows;
      bool edge_hit = in_bounds && scene.edges.at<unsigned char>(sample_y, sample_x) != 0;
      bool vector_lane = in_bounds && edge_hit;
      if (near_rounding_boundary) {
        const int scalar_x = static_cast<int>(std::lround(column + c * points.relative_x[point_index] -
                                                          s * points.relative_y[point_index]));
        const int scalar_y = static_cast<int>(std::lround(row + s * points.relative_x[point_index] +
                                                          c * points.relative_y[point_index]));
        sample_x = scalar_x;
        sample_y = scalar_y;
        in_bounds = scalar_x >= 0 && scalar_y >= 0 &&
                    scalar_x < scene.edges.cols && scalar_y < scene.edges.rows;
        edge_hit = in_bounds && scene.edges.at<unsigned char>(scalar_y, scalar_x) != 0;
        vector_lane = in_bounds && edge_hit && scalar_x == vector_x[lane] && scalar_y == vector_y[lane];
      }
      if (vector_lane) {
        gather_index[lane] = sample_y * orientation_step + sample_x * static_cast<int>(sizeof(float));
        mask_value[lane] = -1;
      } else {
        gather_index[lane] = 0;
        mask_value[lane] = 0;
        if (edge_hit) {
          const float similarity = score_single(scene, points, point_index, column, row, angle, &valid);
          weighted_scalar += static_cast<double>(points.weight[point_index]) * similarity;
          total_scalar += points.weight[point_index];
        }
      }
      if (vector_lane) ++valid;
    }

    const __m256i indices = _mm256_load_si256(reinterpret_cast<const __m256i*>(gather_index));
    const __m256 mask = _mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(mask_value)));
    const __m256 image_orientation = _mm256_mask_i32gather_ps(_mm256_setzero_ps(),
        scene.orientation.ptr<float>(), indices, mask, 1);
    __m256 difference = _mm256_sub_ps(_mm256_add_ps(_mm256_loadu_ps(points.orientation.data() + index), angle_v),
                                      image_orientation);
    difference = _mm256_sub_ps(difference,
        _mm256_mul_ps(two_pi_v, _mm256_floor_ps(_mm256_div_ps(_mm256_add_ps(difference, pi_v), two_pi_v))));
    const __m256 similarity = cosine_positive(difference);
    const __m256 weights = _mm256_and_ps(_mm256_loadu_ps(points.weight.data() + index), mask);
    weighted_scalar += static_cast<double>(horizontal_sum(_mm256_mul_ps(weights, similarity)));
    total_scalar += static_cast<double>(horizontal_sum(weights));
  }
  for (; index < count; ++index) {
    const float similarity = score_single(scene, points, index, column, row, angle, &valid);
    const int x = static_cast<int>(std::lround(column + c * points.relative_x[index] - s * points.relative_y[index]));
    const int y = static_cast<int>(std::lround(row + s * points.relative_x[index] + c * points.relative_y[index]));
    if (x < 0 || y < 0 || x >= scene.edges.cols || y >= scene.edges.rows || scene.edges.at<unsigned char>(y, x) == 0)
      continue;
    weighted_scalar += static_cast<double>(points.weight[index]) * similarity;
    total_scalar += points.weight[index];
  }
  if (valid_count) *valid_count = valid;
  if (total_scalar <= 0.0) return 0.0;
  return std::clamp(weighted_scalar / total_scalar, 0.0, 1.0);
}

}
