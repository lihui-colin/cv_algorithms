#pragma once
/** @file HALCON 24.11-style procedural surface, independently implemented.
 * The namespace and runtime types belong to this project; no HALCON ABI claim.
 * Only documented implemented parameters are accepted. See docs/API.md.
 */
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace shape_match {

/// Project error categories, deliberately NOT fabricated HALCON error numbers.
enum class ErrorCode { Type, Value, Handle, Untrained, Unsupported, Image, Geometry };
class HException : public std::runtime_error {
  public:
    HException(ErrorCode code, const std::string &message)
        : std::runtime_error(message), code_(code) {}
    ErrorCode Code() const noexcept { return code_; }

  private:
    ErrorCode code_;
};

namespace detail {
struct HandleBase;
struct Access;
} // namespace detail
/// Typed, reference-counted handle. ClearShapeModel invalidates all aliases.
struct HHandle {
    std::shared_ptr<detail::HandleBase> state;
    bool operator==(const HHandle &other) const noexcept { return state == other.state; }
};
using HValue = std::variant<int64_t, double, std::string, HHandle>;

/// Small heterogeneous tuple. Indexing is zero-based and range checked.
class HTuple {
  public:
    HTuple() = default;
    HTuple(int v) : values_{int64_t(v)} {}
    HTuple(int64_t v) : values_{v} {}
    HTuple(double v) : values_{v} {}
    HTuple(const char *v) : values_{std::string(v)} {}
    HTuple(std::string v) : values_{std::move(v)} {}
    HTuple(HHandle v) : values_{std::move(v)} {}
    HTuple(std::initializer_list<HTuple> values);
    size_t Length() const noexcept { return values_.size(); }
    bool Empty() const noexcept { return values_.empty(); }
    const HValue &At(size_t i) const;
    double D(size_t i = 0) const;
    int64_t I(size_t i = 0) const;
    const std::string &S(size_t i = 0) const;
    HHandle H(size_t i = 0) const;
    HTuple &Append(const HTuple &other);
    HTuple &AppendValue(HValue value);

  private:
    std::vector<HValue> values_;
};

/// Pixel-center coordinates: top-left pixel is (row=0,column=0).
struct Point {
    double row = 0, column = 0;
};
using Contour = std::vector<Point>;
using ContourSet = std::vector<Contour>;

/// Image owns float pixels in gray range [0,255] and a byte domain (0/1).
struct Image {
    int width = 0, height = 0;
    std::vector<float> pixels;
    std::vector<uint8_t> domain;
    float operator()(int row, int column) const { return pixels[size_t(row) * width + column]; }
};

/// Only image tuples and XLD-like contour objects needed by matching are stored.
class HObject {
  public:
    enum class Kind { Empty, Images, Contours };
    Kind Type() const noexcept { return kind_; }
    /// Application adapter, NOT a HALCON operator. Makes an owned image copy.
    static HObject FromGray(int width, int height, const std::vector<float> &pixels,
                            const std::vector<uint8_t> &domain = {});
    static HObject FromBytes(int width, int height, const std::vector<uint8_t> &pixels,
                             const std::vector<uint8_t> &domain = {});
    static HObject FromContours(ContourSet contours);
    const Image &GetImage(size_t i = 0) const;
    const ContourSet &GetContours() const;
    size_t Count() const noexcept;

  private:
    Kind kind_ = Kind::Empty;
    std::vector<std::shared_ptr<const Image>> images_;
    ContourSet contours_;
    friend struct detail::Access;
};

/// Create untrained model with default parameters and unique identifier.
void CreateGenericShapeModel(HTuple *ModelID);
/// Transactionally set parameters; relevant changes mark the model untrained.
void SetGenericShapeModelParam(const HTuple &ModelID, const HTuple &GenParamName,
                               const HTuple &GenParamValue);
/// Query effective parameters, *_param original values, and needs_training.
void GetGenericShapeModelParam(const HTuple &ModelID, const HTuple &GenParamName,
                               HTuple *GenParamValue);
/// Train one grayscale image template with its domain. XLD input is not yet supported.
void TrainGenericShapeModel(const HObject &Template, const HTuple &ModelID);
/// Query "contours" relative to the selected model origin, pixel-center convention.
void GetGenericShapeModelObject(HObject *Object, const HTuple &ModelID, const HTuple &GenParamName);
/// Full image search, one or multiple trained models; returns a result handle even if empty.
void FindGenericShapeModel(const HObject &SearchImage, const HTuple &ModelID, HTuple *MatchResultID,
                           HTuple *NumMatchResult);
/// Query scalar fields. Selector: index, all, best, identifier, model handle, or their tuple.
void GetGenericShapeModelResult(const HTuple &MatchResultID, const HTuple &MatchSelector,
                                const HTuple &GenParamName, HTuple *GenParamValue);
/// Query "contours" already placed in the search image, pixel-center convention.
void GetGenericShapeModelResultObject(HObject *Objects, const HTuple &MatchResultID,
                                      const HTuple &MatchSelector, const HTuple &GenParamName);
/// Invalidate one or multiple model handles. Result snapshots remain readable.
void ClearShapeModel(const HTuple &ModelID);

/// Application I/O helpers (P5 PGM only). PNG conversion lives in scripts/prepare_samples.py.
HObject ReadPgm(const std::string &path, const std::string &mask_path = "");
void WritePgm(const HObject &image, const std::string &path);
/// Runtime diagnostics are deliberately outside the HALCON-compatible parameter namespace.
struct SearchDiagnostics {
    double pyramid_ms = 0, top_level_ms = 0, tracking_ms = 0, refinement_ms = 0, total_ms = 0;
    size_t evaluated_poses = 0, coarse_candidates = 0, refined_candidates = 0;
};
SearchDiagnostics GetSearchDiagnostics(const HTuple &MatchResultID);
} // namespace shape_match
