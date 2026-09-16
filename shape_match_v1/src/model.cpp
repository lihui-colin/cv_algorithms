#include "internal.hpp"
#include <limits>

namespace shape_match::detail {
Params MakeDefaultParams(uint64_t id) {
    return {{"model_identifier", std::string("model_") + std::to_string(id)},
            {"contrast_low", std::string("auto")},
            {"contrast_high", std::string("auto")},
            {"min_size", std::string("auto")},
            {"metric", std::string("auto")},
            {"optimization", std::string("auto")},
            {"num_levels", std::string("auto")},
            {"angle_step", std::string("auto")},
            {"iso_scale_step", std::string("auto")},
            {"iso_scale_min", 1.0},
            {"iso_scale_max", 1.0},
            {"angle_start", 0.0},
            {"angle_end", 2 * pi},
            {"restrict_iso_scale_min", std::string("auto")},
            {"restrict_iso_scale_max", std::string("auto")},
            {"min_score", 0.5},
            {"num_matches", std::string("all")},
            {"max_overlap", 0.5},
            {"max_overlap_global_enable", std::string("false")},
            {"min_contrast", std::string("auto")},
            {"greediness", 0.9},
            {"strict_boundaries", std::string("false")},
            {"subpixel", std::string("least_squares")},
            {"refinement_method", std::string("nearest_point")},
            {"refinement_radius", 1.5},
            {"pyramid_level_highest", std::string("auto")},
            {"pyramid_level_lowest", int64_t(1)},
            {"pyramid_level_robust_tracking", std::string("false")},
            {"border_shape_models", std::string("system")},
            {"origin_row", 0.0},
            {"origin_column", 0.0},
            {"prepare_contours_for_visualization", std::string("true")}};
}
bool NeedsTraining(const std::string &k) {
    static const std::set<std::string> keys = {
        "contrast_low", "contrast_high", "min_size",       "metric",        "optimization",
        "num_levels",   "angle_step",    "iso_scale_step", "iso_scale_min", "iso_scale_max"};
    return keys.count(k) != 0;
}
void ValidateParams(const Params &p) {
    auto range = [&](const std::string &k, double low, double high, bool allow_auto = false) {
        if (allow_auto && IsAuto(p.at(k)))
            return;
        double v = Num(p, k);
        Require(std::isfinite(v) && v >= low && v <= high, ErrorCode::Value,
                "Out-of-range parameter: " + k);
    };
    auto integer = [&](const std::string &k, int low, int high, bool automatic = false) {
        if (automatic && IsAuto(p.at(k)))
            return;
        Require(std::holds_alternative<int64_t>(p.at(k)), ErrorCode::Type,
                "Integer required: " + k);
        range(k, low, high);
    };
    auto one_of = [&](const std::string &k, std::initializer_list<const char *> values) {
        auto v = Str(p, k);
        bool found = false;
        for (auto x : values)
            found |= v == x;
        Require(found, ErrorCode::Unsupported, "Unsupported value for " + k + ": " + v);
    };
    auto id = Str(p, "model_identifier");
    Require(!id.empty() && id != "all" && id != "best", ErrorCode::Value,
            "Reserved or empty model identifier");
    for (auto k : {"contrast_low", "contrast_high", "min_contrast"})
        range(k, 0, 255, true);
    if (!IsAuto(p.at("contrast_low")) && !IsAuto(p.at("contrast_high")))
        Require(Num(p, "contrast_low") <= Num(p, "contrast_high"), ErrorCode::Value,
                "contrast_low exceeds contrast_high");
    integer("min_size", 0, 1000000, true);
    integer("num_levels", 1, 8, true);
    range("angle_step", 1e-6, 1e6, true);
    range("iso_scale_step", 1e-6, 1e6, true);
    // Implementation resource envelope is documented; never silently clamp scales.
    range("iso_scale_min", 0.05, 20);
    range("iso_scale_max", 0.05, 20);
    Require(Num(p, "iso_scale_min") <= Num(p, "iso_scale_max"), ErrorCode::Value,
            "Invalid scale interval");
    range("angle_start", -1e6, 1e6);
    range("angle_end", -1e6, 1e6);
    Require(Num(p, "angle_end") >= Num(p, "angle_start") &&
                Num(p, "angle_end") - Num(p, "angle_start") <= 2 * pi + 1e-8,
            ErrorCode::Value, "Angle interval must be ordered and at most 2*pi");
    for (auto k : {"restrict_iso_scale_min", "restrict_iso_scale_max"})
        range(k, 0.05, 20, true);
    double lo = IsAuto(p.at("restrict_iso_scale_min")) ? Num(p, "iso_scale_min")
                                                       : Num(p, "restrict_iso_scale_min");
    double hi = IsAuto(p.at("restrict_iso_scale_max")) ? Num(p, "iso_scale_max")
                                                       : Num(p, "restrict_iso_scale_max");
    Require(lo >= Num(p, "iso_scale_min") && hi <= Num(p, "iso_scale_max") && lo <= hi,
            ErrorCode::Value, "Search scale restriction exceeds trained interval");
    for (auto k : {"min_score", "max_overlap", "greediness"})
        range(k, 0, 1);
    if (auto s = std::get_if<std::string>(&p.at("num_matches")))
        Require(*s == "all", ErrorCode::Value, "num_matches string must be all");
    else
        integer("num_matches", 0, 1000000);
    one_of("metric", {"auto", "use_polarity", "ignore_global_polarity", "ignore_local_polarity"});
    one_of("optimization", {"auto", "none", "point_reduction_low", "point_reduction_medium",
                            "point_reduction_high"});
    one_of("subpixel", {"none", "interpolation", "least_squares", "least_squares_high",
                        "least_squares_very_high"});
    one_of("refinement_method", {"nearest_point", "contour", "gradient", "gradient_gaussian"});
    range("refinement_radius", 0.5, 10.0);
    for (auto k :
         {"strict_boundaries", "max_overlap_global_enable", "prepare_contours_for_visualization"})
        one_of(k, {"true", "false"});
    one_of("border_shape_models", {"system", "true", "false"});
    one_of("pyramid_level_robust_tracking",
           {"false"}); // Explicitly reject unimplemented robust fallback.
    integer("pyramid_level_highest", 1, 8, true);
    integer("pyramid_level_lowest", 1, 8);
    for (auto k : {"origin_row", "origin_column"})
        range(k, -1e6, 1e6);
}

/// Estimate noise from second differences; unlike HALCON, estimation is public here.
static double EstimateNoise(const Image &im) {
    std::vector<double> d;
    d.reserve(im.pixels.size() / 4);
    for (int y = 1; y < im.height - 1; y += 2)
        for (int x = 1; x < im.width - 1; x += 2) {
            size_t i = size_t(y) * im.width + x;
            if (im.domain[i] && im.domain[i - 1] && im.domain[i + 1])
                d.push_back(std::abs(im.pixels[i - 1] - 2 * im.pixels[i] + im.pixels[i + 1]));
        }
    if (d.empty())
        return 0;
    auto mid = d.begin() + d.size() / 2;
    std::nth_element(d.begin(), mid, d.end());
    return *mid / (0.67448975 * std::sqrt(6.0));
}

std::shared_ptr<TrainedData> BuildModel(const Image &im, const Params &params) {
    Require(im.width >= 12 && im.height >= 12, ErrorCode::Image, "Template must be at least 12x12");
    auto data = std::make_shared<TrainedData>();
    data->image = im;
    data->effective = params;
    double count = 0;
    for (int y = 0; y < im.height; ++y)
        for (int x = 0; x < im.width; ++x)
            if (im.domain[size_t(y) * im.width + x]) {
                data->origin.x += x;
                data->origin.y += y;
                ++count;
            }
    Require(count >= 16, ErrorCode::Image, "Template domain too small");
    data->origin = data->origin * (1 / count);
    double noise = EstimateNoise(im);
    auto &p = data->effective;
    bool la = IsAuto(p["contrast_low"]), ha = IsAuto(p["contrast_high"]);
    if (la && ha) {
        p["contrast_low"] = std::max(5.0, noise * 2.5);
        p["contrast_high"] = std::max(10.0, noise * 5);
    } else if (la)
        p["contrast_low"] = p["contrast_high"];
    else if (ha)
        p["contrast_high"] = p["contrast_low"];
    if (IsAuto(p["min_contrast"]))
        p["min_contrast"] = std::min(Num(p, "contrast_low"), std::max(3.0, noise * 2));
    if (IsAuto(p["metric"]))
        p["metric"] = std::string("use_polarity");
    if (IsAuto(p["min_size"]))
        p["min_size"] = int64_t(4);
    if (IsAuto(p["optimization"]))
        p["optimization"] = std::string("point_reduction_medium");
    Image filtered = Smooth(im, 0.8);
    auto grad = ComputeGradients(filtered);
    auto absolute = ExtractSubpixelContours(filtered, grad, Num(p, "contrast_low"),
                                            Num(p, "contrast_high"), int(Num(p, "min_size")));
    Require(absolute.size() >= 12, ErrorCode::Geometry, "Insufficient template edge features");
    data->contours = TraceContours(absolute);
    data->dense = absolute;
    for (auto &f : data->dense) {
        f.p = f.p - data->origin;
        data->radius = std::max(data->radius, std::sqrt(Dot(f.p, f.p)));
    }
    data->rectangle = MinimumRectangle(data->dense);
    data->gaussian_dense = GaussianModelFeatures(im, Num(p, "contrast_low"),
                                                 Num(p, "contrast_high"), int(Num(p, "min_size")));
    for (auto &f : data->gaussian_dense)
        f.p = f.p - data->origin;
    if (IsAuto(p["num_levels"]))
        p["num_levels"] = int64_t(
            std::clamp(int(std::floor(std::log2(std::max(1.0, data->radius / 9)))) + 1, 1, 5));
    if (IsAuto(p["angle_step"]))
        p["angle_step"] = std::min(0.20, 0.8 / data->radius);
    else
        p["angle_step"] = std::min(0.20, Num(p, "angle_step"));
    if (IsAuto(p["iso_scale_step"]))
        p["iso_scale_step"] = 0.8 / data->radius;
    int count_levels = int(Num(p, "num_levels"));
    Image current = im;
    for (int l = 0; l < count_levels; ++l) {
        Require(current.width >= 5 && current.height >= 5, ErrorCode::Geometry,
                "Too many pyramid levels");
        double factor = double(1 << l);
        Features features;
        if (l == 0)
            features = data->dense;
        else {
            auto smooth = Smooth(current, 0.8);
            auto g = ComputeGradients(smooth);
            features =
                ExtractSubpixelContours(smooth, g, Num(p, "contrast_low"), Num(p, "contrast_high"),
                                        std::max(1, int(Num(p, "min_size")) / int(factor)));
            for (auto &f : features)
                f.p = f.p - data->origin * (1 / factor);
        }
        Require(features.size() >= 4, ErrorCode::Geometry,
                "Insufficient edge features on pyramid level " + std::to_string(l + 1));
        size_t maximum = Str(p, "optimization") == "none"                   ? features.size()
                         : Str(p, "optimization") == "point_reduction_low"  ? 192
                         : Str(p, "optimization") == "point_reduction_high" ? 48
                                                                            : 96;
        data->levels.push_back({SelectModelFeatures(features, maximum), factor});
        if (l + 1 < count_levels)
            current = Downsample(current);
    }
    return data;
}
} // namespace shape_match::detail

namespace shape_match {
void CreateGenericShapeModel(HTuple *id) {
    detail::Output(id);
    static std::atomic<uint64_t> next{1};
    *id = HTuple(HHandle{std::make_shared<detail::ShapeModel>(next.fetch_add(1))});
}
void SetGenericShapeModelParam(const HTuple &ids, const HTuple &names, const HTuple &values) {
    using namespace detail;
    Require(ids.Length() == 1, ErrorCode::Unsupported,
            "Parameter setter currently accepts one model handle");
    Require(!names.Empty() && names.Length() == values.Length(), ErrorCode::Value,
            "Parameter names and values must have equal nonzero length");
    auto m = ResolveModel(ids);
    std::lock_guard<std::mutex> lock(m->mutex);
    Require(m->alive, ErrorCode::Handle, "Model cleared");
    Params updated = m->params;
    bool dirty = false;
    for (size_t i = 0; i < names.Length(); ++i) {
        auto k = names.S(i);
        Require(updated.count(k), ErrorCode::Unsupported, "Unsupported parameter: " + k);
        updated[k] = values.At(i);
        dirty |= NeedsTraining(k) || (k == "min_contrast" && IsAuto(values.At(i)));
    }
    ValidateParams(updated);
    m->params = std::move(updated);
    m->needs_training |= dirty;
}
void GetGenericShapeModelParam(const HTuple &ids, const HTuple &names, HTuple *value) {
    using namespace detail;
    Output(value);
    Require(ids.Length() == 1 && !names.Empty(), ErrorCode::Value,
            "Expected one model and parameter names");
    auto m = ResolveModel(ids);
    std::lock_guard<std::mutex> lock(m->mutex);
    Require(m->alive, ErrorCode::Handle, "Model cleared");
    HTuple out;
    for (size_t i = 0; i < names.Length(); ++i) {
        auto k = names.S(i);
        if (k == "needs_training") {
            out.Append(m->needs_training ? "true" : "false");
            continue;
        }
        if (k == "scale_type") {
            out.Append(Num(m->params, "iso_scale_min") == 1 && Num(m->params, "iso_scale_max") == 1
                           ? "none"
                           : "isotropic");
            continue;
        }
        if (k == "has_samples") {
            out.Append("false");
            continue;
        }
        bool raw = k.size() > 6 && k.substr(k.size() - 6) == "_param";
        if (raw)
            k.resize(k.size() - 6);
        Require(m->params.count(k), ErrorCode::Unsupported, "Unsupported parameter query: " + k);
        auto v = m->params.at(k);
        if (!raw && m->trained && !m->needs_training && (NeedsTraining(k) || IsAuto(v)))
            v = m->trained->effective.at(k);
        out.AppendValue(v);
    }
    *value = std::move(out);
}
void TrainGenericShapeModel(const HObject &object, const HTuple &ids) {
    using namespace detail;
    Require(ids.Length() == 1 && object.Count() == 1, ErrorCode::Value,
            "Training needs one model and one image");
    auto m = ResolveModel(ids);
    std::lock_guard<std::mutex> lock(m->mutex);
    Require(m->alive, ErrorCode::Handle, "Model cleared");
    auto data = BuildModel(object.GetImage(), m->params);
    m->trained = std::move(data);
    m->needs_training = false;
}
void GetGenericShapeModelObject(HObject *object, const HTuple &ids, const HTuple &names) {
    using namespace detail;
    Output(object);
    Require(ids.Length() == 1 && names.Length() == 1 && names.S() == "contours",
            ErrorCode::Unsupported, "Only model contours are supported");
    auto m = ResolveModel(ids);
    std::lock_guard<std::mutex> lock(m->mutex);
    Require(m->alive, ErrorCode::Handle, "Model cleared");
    Require(m->trained && !m->needs_training, ErrorCode::Untrained, "Train model first");
    Require(Bool(m->params, "prepare_contours_for_visualization"), ErrorCode::Value,
            "Contour visualization is disabled");
    ContourSet contours = m->trained->contours;
    Vec o = m->trained->origin + Vec{Num(m->params, "origin_column"), Num(m->params, "origin_row")};
    for (auto &c : contours)
        for (auto &p : c) {
            p.row -= o.y;
            p.column -= o.x;
        }
    *object = HObject::FromContours(std::move(contours));
}
void ClearShapeModel(const HTuple &ids) {
    using namespace detail;
    Require(!ids.Empty(), ErrorCode::Value, "No model handles");
    for (size_t i = 0; i < ids.Length(); ++i) {
        auto m = ResolveModel(ids, i);
        std::lock_guard<std::mutex> lock(m->mutex);
        Require(m->alive, ErrorCode::Handle, "Model already cleared");
        m->alive = false;
        m->trained.reset();
    }
}
} // namespace shape_match
