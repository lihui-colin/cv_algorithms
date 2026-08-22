#include "openshape/model/shape_model.hpp"
#include "openshape/pyramid/pyramid.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <type_traits>

namespace openshape {
namespace {
constexpr unsigned char kModelMagic[8] = {'O', 'S', 'M', 'M', 'D', 'L', '1', 0};
constexpr std::uint32_t kFileFormatVersion = 1;

class BufferWriter {
public:
  template <typename T>
  void scalar(T value) {
    static_assert(std::is_arithmetic<T>::value, "arithmetic type required");
    using Bits = typename std::conditional<sizeof(T) == 8, std::uint64_t,
                 typename std::conditional<sizeof(T) == 4, std::uint32_t,
                 typename std::conditional<sizeof(T) == 2, std::uint16_t,
                                           std::uint8_t>::type>::type>::type;
    Bits bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    for (std::size_t i = 0; i < sizeof(T); ++i)
      data.push_back(static_cast<unsigned char>((bits >> (i * 8)) & 0xffu));
  }
  void boolean(bool value) { scalar<std::uint8_t>(value ? 1 : 0); }
  void string(const std::string& value) {
    scalar<std::uint32_t>(static_cast<std::uint32_t>(value.size()));
    data.insert(data.end(), value.begin(), value.end());
  }
  std::vector<unsigned char> data;
};

class BufferReader {
public:
  explicit BufferReader(const std::vector<unsigned char>& bytes) : bytes_(bytes) {}
  template <typename T>
  T scalar() {
    if (offset_ + sizeof(T) > bytes_.size()) throw InvalidModel("truncated model file");
    using Bits = typename std::conditional<sizeof(T) == 8, std::uint64_t,
                 typename std::conditional<sizeof(T) == 4, std::uint32_t,
                 typename std::conditional<sizeof(T) == 2, std::uint16_t,
                                           std::uint8_t>::type>::type>::type;
    Bits bits = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
      bits |= static_cast<Bits>(bytes_[offset_++]) << (i * 8);
    T value{};
    std::memcpy(&value, &bits, sizeof(T));
    return value;
  }
  bool boolean() {
    const auto value = scalar<std::uint8_t>();
    if (value > 1) throw InvalidModel("invalid boolean in model file");
    return value != 0;
  }
  std::string string() {
    const auto size = scalar<std::uint32_t>();
    if (offset_ + size > bytes_.size()) throw InvalidModel("truncated model string");
    std::string value(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                      bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + size));
    offset_ += size;
    return value;
  }
  bool finished() const { return offset_ == bytes_.size(); }
private:
  const std::vector<unsigned char>& bytes_;
  std::size_t offset_ = 0;
};

std::uint32_t checksum(const std::vector<unsigned char>& bytes) {
  std::uint32_t value = 2166136261u;
  for (unsigned char byte : bytes) {
    value ^= byte;
    value *= 16777619u;
  }
  return value;
}

void write_params(BufferWriter& out, const ShapeModelParams& p) {
  out.scalar<std::int32_t>(p.roi.x); out.scalar<std::int32_t>(p.roi.y);
  out.scalar<std::int32_t>(p.roi.width); out.scalar<std::int32_t>(p.roi.height);
  out.scalar<float>(p.origin.x); out.scalar<float>(p.origin.y);
  out.scalar<double>(p.canny_low_threshold); out.scalar<double>(p.canny_high_threshold);
  out.scalar<std::int32_t>(p.canny_aperture_size); out.boolean(p.canny_l2_gradient);
  out.scalar<std::int32_t>(p.sobel_kernel_size); out.scalar<std::int32_t>(p.sobel_border_type);
  out.scalar<std::int32_t>(p.gaussian_kernel_size); out.scalar<double>(p.gaussian_sigma);
  out.scalar<double>(p.min_gradient_magnitude);
  out.scalar<std::uint64_t>(p.min_model_points); out.scalar<std::uint64_t>(p.max_model_points);
  out.scalar<double>(p.min_point_distance); out.scalar<std::int32_t>(p.num_levels);
  out.string(p.model_point_sampling); out.scalar<std::int32_t>(p.model_version);
}

ShapeModelParams read_params(BufferReader& in) {
  ShapeModelParams p;
  p.roi.x = in.scalar<std::int32_t>(); p.roi.y = in.scalar<std::int32_t>();
  p.roi.width = in.scalar<std::int32_t>(); p.roi.height = in.scalar<std::int32_t>();
  p.origin.x = in.scalar<float>(); p.origin.y = in.scalar<float>();
  p.canny_low_threshold = in.scalar<double>(); p.canny_high_threshold = in.scalar<double>();
  p.canny_aperture_size = in.scalar<std::int32_t>(); p.canny_l2_gradient = in.boolean();
  p.sobel_kernel_size = in.scalar<std::int32_t>(); p.sobel_border_type = in.scalar<std::int32_t>();
  p.gaussian_kernel_size = in.scalar<std::int32_t>(); p.gaussian_sigma = in.scalar<double>();
  p.min_gradient_magnitude = in.scalar<double>();
  p.min_model_points = static_cast<std::size_t>(in.scalar<std::uint64_t>());
  p.max_model_points = static_cast<std::size_t>(in.scalar<std::uint64_t>());
  p.min_point_distance = in.scalar<double>(); p.num_levels = in.scalar<std::int32_t>();
  p.model_point_sampling = in.string(); p.model_version = in.scalar<std::int32_t>();
  p.validate();
  return p;
}

void write_points(BufferWriter& out, const std::vector<ModelPoint>& points) {
  out.scalar<std::uint64_t>(points.size());
  for (const auto& point : points) {
    out.scalar<float>(point.relative_x); out.scalar<float>(point.relative_y);
    out.scalar<float>(point.orientation); out.scalar<float>(point.weight);
    out.scalar<std::int32_t>(point.level);
  }
}

std::vector<ModelPoint> read_points(BufferReader& in) {
  const auto count = in.scalar<std::uint64_t>();
  if (count > 10000000u) throw InvalidModel("model point count is unreasonable");
  std::vector<ModelPoint> points(static_cast<std::size_t>(count));
  for (auto& point : points) {
    point.relative_x = in.scalar<float>(); point.relative_y = in.scalar<float>();
    point.orientation = in.scalar<float>(); point.weight = in.scalar<float>();
    point.level = in.scalar<std::int32_t>();
    if (!std::isfinite(point.relative_x) || !std::isfinite(point.relative_y) ||
        !std::isfinite(point.orientation) || !std::isfinite(point.weight) || point.weight < 0)
      throw InvalidModel("model contains invalid point data");
  }
  return points;
}

ModelLevelSoA make_soa(const std::vector<ModelPoint>& points) {
  ModelLevelSoA soa;
  soa.level = points.empty() ? 0 : points.front().level;
  soa.relative_x.reserve(points.size()); soa.relative_y.reserve(points.size());
  soa.orientation.reserve(points.size()); soa.weight.reserve(points.size());
  for (const auto& point : points) {
    soa.relative_x.push_back(point.relative_x); soa.relative_y.push_back(point.relative_y);
    soa.orientation.push_back(point.orientation); soa.weight.push_back(point.weight);
  }
  return soa;
}
}

ShapeModel ShapeModelBuilder::create(const ImageView& image, const ShapeModelParams& params) {
  params.validate();
  EdgeMap map = EdgeEngine::compute(image, params);
  cv::Rect roi = params.roi;
  if (roi.width == 0 || roi.height == 0) roi = cv::Rect(0, 0, map.gray.cols, map.gray.rows);
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x + roi.width > map.gray.cols || roi.y + roi.height > map.gray.rows)
    throw InvalidArgument("model ROI is outside the image");
  cv::Point2f origin = params.origin;
  if (origin.x < 0 && origin.y < 0) origin = cv::Point2f(roi.x + roi.width * 0.5f, roi.y + roi.height * 0.5f);
  else if (origin.x < 0 || origin.y < 0) throw InvalidArgument("origin must specify both coordinates");
  if (origin.x < roi.x || origin.x > roi.x + roi.width || origin.y < roi.y || origin.y > roi.y + roi.height)
    throw InvalidArgument("model origin must lie in the ROI");

  std::vector<EdgePoint> candidates;
  candidates.reserve(static_cast<std::size_t>(roi.area() / 4));
  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      if (map.edges.at<unsigned char>(y, x) == 0) continue;
      const float m = map.magnitude.at<float>(y, x);
      if (m < params.min_gradient_magnitude) continue;
      candidates.push_back({static_cast<float>(x), static_cast<float>(y), m, map.orientation.at<float>(y, x)});
    }
  }
  if (params.model_point_sampling == "magnitude") {
    std::stable_sort(candidates.begin(), candidates.end(), [](const EdgePoint& a, const EdgePoint& b) {
      return a.magnitude > b.magnitude;
    });
  } else {
    // Deterministic scan order plus the minimum-distance Poisson selection
    // gives spatially uniform contour coverage. The old implementation always
    // sorted by magnitude, which over-represented a few bright highlights and
    // made textured objects prone to high-score false positives.
    std::stable_sort(candidates.begin(), candidates.end(), [](const EdgePoint& a, const EdgePoint& b) {
      if (a.y != b.y) return a.y < b.y;
      if (a.x != b.x) return a.x < b.x;
      return a.magnitude > b.magnitude;
    });
  }
  const float min_dist2 = static_cast<float>(params.min_point_distance * params.min_point_distance);
  std::vector<EdgePoint> selected;
  selected.reserve(std::min(params.max_model_points, candidates.size()));
  for (const auto& c : candidates) {
    bool far_enough = true;
    for (const auto& s : selected) {
      const float dx = c.x - s.x, dy = c.y - s.y;
      if (dx * dx + dy * dy < min_dist2) { far_enough = false; break; }
    }
    if (far_enough) selected.push_back(c);
    if (selected.size() >= params.max_model_points) break;
  }
  if (selected.size() < params.min_model_points)
    throw InvalidModel("template contains fewer than min_model_points valid edge points");

  float max_mag = 0;
  for (const auto& p : selected) max_mag = std::max(max_mag, p.magnitude);
  ShapeModel model;
  model.roi_ = roi; model.origin_ = origin; model.version_ = params.model_version; model.params_ = params;
  model.points_.reserve(selected.size());
  for (const auto& p : selected) {
    const float point_weight = params.model_point_sampling == "uniform"
        ? 1.0f : (max_mag > 0 ? p.magnitude / max_mag : 0.f);
    model.points_.push_back({p.x - origin.x, p.y - origin.y, p.orientation,
                             point_weight, 0});
  }
  model.levels_ = ModelPyramid::build(model, params.num_levels).levels();
  model.levels_soa_.reserve(model.levels_.size());
  for (const auto& level_points : model.levels_) {
    ModelLevelSoA soa;
    soa.level = level_points.empty() ? 0 : level_points.front().level;
    soa.relative_x.reserve(level_points.size());
    soa.relative_y.reserve(level_points.size());
    soa.orientation.reserve(level_points.size());
    soa.weight.reserve(level_points.size());
    for (const auto& point : level_points) {
      soa.relative_x.push_back(point.relative_x);
      soa.relative_y.push_back(point.relative_y);
      soa.orientation.push_back(point.orientation);
      soa.weight.push_back(point.weight);
    }
    model.levels_soa_.push_back(std::move(soa));
  }
  return model;
}

void ShapeModel::save(const std::string& path) const {
  if (empty()) throw InvalidModel("cannot save an empty model");
  if (path.empty()) throw InvalidArgument("model path is empty");
  BufferWriter payload;
  payload.scalar<std::int32_t>(version_);
  payload.scalar<std::int32_t>(roi_.x); payload.scalar<std::int32_t>(roi_.y);
  payload.scalar<std::int32_t>(roi_.width); payload.scalar<std::int32_t>(roi_.height);
  payload.scalar<float>(origin_.x); payload.scalar<float>(origin_.y);
  write_params(payload, params_);
  write_points(payload, points_);
  payload.scalar<std::uint32_t>(static_cast<std::uint32_t>(levels_.size()));
  for (const auto& level : levels_) write_points(payload, level);

  BufferWriter file;
  file.data.insert(file.data.end(), std::begin(kModelMagic), std::end(kModelMagic));
  file.scalar<std::uint32_t>(kFileFormatVersion);
  file.scalar<std::uint64_t>(payload.data.size());
  file.scalar<std::uint32_t>(checksum(payload.data));
  file.data.insert(file.data.end(), payload.data.begin(), payload.data.end());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw InvalidModel("unable to open model file for writing: " + path);
  output.write(reinterpret_cast<const char*>(file.data.data()),
               static_cast<std::streamsize>(file.data.size()));
  if (!output) throw InvalidModel("failed to write model file: " + path);
}

ShapeModel ShapeModel::load(const std::string& path) {
  if (path.empty()) throw InvalidArgument("model path is empty");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw InvalidModel("unable to open model file: " + path);
  std::vector<unsigned char> file((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  constexpr std::size_t header_size = 8 + 4 + 8 + 4;
  if (file.size() < header_size ||
      !std::equal(std::begin(kModelMagic), std::end(kModelMagic), file.begin()))
    throw InvalidModel("unrecognized or truncated model file");
  std::vector<unsigned char> header_bytes(file.begin() + 8, file.begin() + header_size);
  BufferReader header(header_bytes);
  const auto format_version = header.scalar<std::uint32_t>();
  const auto payload_size = header.scalar<std::uint64_t>();
  const auto expected_checksum = header.scalar<std::uint32_t>();
  if (format_version != kFileFormatVersion)
    throw InvalidModel("unsupported model file format version");
  if (payload_size != file.size() - header_size)
    throw InvalidModel("model file size does not match its header");
  std::vector<unsigned char> payload(file.begin() + header_size, file.end());
  if (checksum(payload) != expected_checksum) throw InvalidModel("model checksum mismatch");

  BufferReader in(payload);
  ShapeModel model;
  model.version_ = in.scalar<std::int32_t>();
  if (model.version_ <= 0 || model.version_ > current_version)
    throw InvalidModel("model version is incompatible with this library");
  model.roi_.x = in.scalar<std::int32_t>(); model.roi_.y = in.scalar<std::int32_t>();
  model.roi_.width = in.scalar<std::int32_t>(); model.roi_.height = in.scalar<std::int32_t>();
  model.origin_.x = in.scalar<float>(); model.origin_.y = in.scalar<float>();
  model.params_ = read_params(in);
  model.points_ = read_points(in);
  const auto level_count = in.scalar<std::uint32_t>();
  if (level_count == 0 || level_count > 32) throw InvalidModel("invalid model pyramid level count");
  model.levels_.reserve(level_count); model.levels_soa_.reserve(level_count);
  for (std::uint32_t level = 0; level < level_count; ++level) {
    auto points = read_points(in);
    if (points.size() != model.points_.size())
      throw InvalidModel("model pyramid point count mismatch");
    model.levels_soa_.push_back(make_soa(points));
    model.levels_.push_back(std::move(points));
  }
  if (!in.finished()) throw InvalidModel("model file contains trailing payload data");
  if (model.empty() || model.roi_.width <= 0 || model.roi_.height <= 0 ||
      !std::isfinite(model.origin_.x) || !std::isfinite(model.origin_.y))
    throw InvalidModel("model metadata is invalid");
  return model;
}
}
