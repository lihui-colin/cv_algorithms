#include "internal.hpp"

namespace shape_match::detail {
Vec OutputOrigin(const Match &m) {
    Vec offset{Num(m.model.params, "origin_column"), Num(m.model.params, "origin_row")};
    return Vec{m.pose.x, m.pose.y} + Rotate(offset, m.pose.theta) * m.pose.scale;
}
std::array<double, 6> ResultTransform(const Match &m) {
    // XLD convention: add (0.5,0.5), apply matrix in row-column coordinates,
    // then subtract (0.5,0.5). This matrix maps the model-origin-relative contour.
    double c = m.pose.scale * std::cos(m.pose.theta), s = m.pose.scale * std::sin(m.pose.theta);
    auto o = OutputOrigin(m);
    return {c, s, o.y + 0.5 - 0.5 * (c + s), -s, c, o.x + 0.5 - 0.5 * (c - s)};
}
std::vector<size_t> ResolveMatchSelector(const MatchResult &result, const HTuple &selector) {
    Require(!selector.Empty(), ErrorCode::Value, "Empty result selector");
    std::vector<bool> selected(result.matches.size());
    for (size_t k = 0; k < selector.Length(); ++k) {
        const auto &value = selector.At(k);
        if (auto i = std::get_if<int64_t>(&value)) {
            Require(*i >= 0 && uint64_t(*i) < selected.size(), ErrorCode::Value,
                    "Result index out of range");
            selected[size_t(*i)] = true;
        } else if (auto s = std::get_if<std::string>(&value)) {
            if (*s == "all")
                std::fill(selected.begin(), selected.end(), true);
            else if (*s == "best") {
                if (!selected.empty())
                    selected[0] = true;
            } else {
                bool known = false;
                for (const auto &m : result.models)
                    known |= Str(m.params, "model_identifier") == *s;
                Require(known, ErrorCode::Value, "Unknown model identifier selector: " + *s);
                for (size_t i = 0; i < result.matches.size(); ++i)
                    if (Str(result.matches[i].model.params, "model_identifier") == *s)
                        selected[i] = true;
            }
        } else if (auto h = std::get_if<HHandle>(&value)) {
            bool known = false;
            for (const auto &m : result.models)
                known |= m.handle == *h;
            Require(known, ErrorCode::Handle, "Model handle does not belong to this result");
            for (size_t i = 0; i < result.matches.size(); ++i)
                if (result.matches[i].model.handle == *h)
                    selected[i] = true;
        } else
            Fail(ErrorCode::Type, "Selector must be integer, string, or typed model handle");
    }
    std::vector<size_t> out;
    for (size_t i = 0; i < selected.size(); ++i)
        if (selected[i])
            out.push_back(i);
    return out;
}
} // namespace shape_match::detail

namespace shape_match {
void GetGenericShapeModelResult(const HTuple &id, const HTuple &selector, const HTuple &name,
                                HTuple *values) {
    using namespace detail;
    Output(values);
    Require(name.Length() == 1, ErrorCode::Value, "One result field per call");
    auto result = ResolveResult(id);
    auto indices = ResolveMatchSelector(*result, selector);
    auto k = name.S();
    HTuple out;
    const std::set<std::string> allowed = {
        "num_match_result", "model_identifier", "row",   "column",    "angle",
        "scale_row",        "scale_column",     "score", "hom_mat_2d"};
    Require(allowed.count(k), ErrorCode::Unsupported, "Unsupported result field: " + k);
    if (k == "num_match_result") {
        *values = HTuple(int64_t(indices.size()));
        return;
    }
    for (size_t index : indices) {
        const auto &m = result->matches[index];
        if (k == "model_identifier")
            out.Append(Str(m.model.params, "model_identifier"));
        // Matching pose values describe the edge-centered contour transformation;
        // they are its translation, not pixel-centered origin + an unconditional 0.5.
        else if (k == "row")
            out.Append(ResultTransform(m)[2]);
        else if (k == "column")
            out.Append(ResultTransform(m)[5]);
        else if (k == "angle")
            out.Append(WrapAngle(-m.pose.theta));
        else if (k == "scale_row" || k == "scale_column")
            out.Append(m.pose.scale);
        else if (k == "score")
            out.Append(m.score);
        else if (k == "hom_mat_2d")
            for (double v : ResultTransform(m))
                out.Append(v);
    }
    *values = std::move(out);
}
void GetGenericShapeModelResultObject(HObject *object, const HTuple &id, const HTuple &selector,
                                      const HTuple &name) {
    using namespace detail;
    Output(object);
    Require(name.Length() == 1 && name.S() == "contours", ErrorCode::Unsupported,
            "Only matched contours are supported");
    auto result = ResolveResult(id);
    auto selected = ResolveMatchSelector(*result, selector);
    ContourSet out;
    for (size_t i : selected) {
        const auto &m = result->matches[i];
        Require(Bool(m.model.params, "prepare_contours_for_visualization"), ErrorCode::Value,
                "Contour visualization disabled");
        for (const auto &contour : m.model.data->contours) {
            Contour transformed;
            transformed.reserve(contour.size());
            for (const auto &p : contour) {
                Vec q = Rotate(Vec{p.column, p.row} - m.model.data->origin, m.pose.theta) *
                            m.pose.scale +
                        Vec{m.pose.x, m.pose.y};
                transformed.push_back({q.y, q.x});
            }
            out.push_back(std::move(transformed));
        }
    }
    *object = HObject::FromContours(std::move(out));
}
} // namespace shape_match
