#include "shape_match/shape_match.hpp"
#include "synthetic.hpp"
#include <cmath>
#include <functional>
#include <iostream>
using namespace shape_match;
int checks = 0;
void Check(bool ok, const char *message) {
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}
void Throws(const std::function<void()> &f, ErrorCode code) {
    bool caught = false;
    try {
        f();
    } catch (const HException &e) {
        caught = e.Code() == code;
    }
    Check(caught, "Expected typed exception");
}
int main() {
    try {
        HTuple tuple{1, 2.5, "ring"};
        Check(tuple.Length() == 3 && tuple.I() == 1 && tuple.S(2) == "ring", "Tuple types");
        Throws([&] { tuple.H(); }, ErrorCode::Type);
        Throws([&] { tuple.At(3); }, ErrorCode::Value);
        HTuple model, v;
        CreateGenericShapeModel(&model);
        GetGenericShapeModelParam(model, "needs_training", &v);
        Check(v.S() == "true", "Initial state");
        HTuple r, n;
        auto im = Render(96, 88, 51.27, 44.63);
        Throws([&] { FindGenericShapeModel(im, model, &r, &n); }, ErrorCode::Untrained);
        Throws([&] { SetGenericShapeModelParam(model, "made_up_parameter", 3); },
               ErrorCode::Unsupported);
        Throws([&] { SetGenericShapeModelParam(model, {"min_score", "max_overlap"}, {0.8, 2.0}); },
               ErrorCode::Value);
        GetGenericShapeModelParam(model, "min_score", &v);
        Check(v.D() == 0.5, "Setter transaction rollback");
        SetGenericShapeModelParam(model, {"model_identifier", "num_levels"}, {"test", 2});
        TrainGenericShapeModel(Render(64, 64, 31.5, 31.5), model);
        SetGenericShapeModelParam(model, {"angle_start", "angle_end", "min_score", "subpixel"},
                                  {-0.2, 0.2, 0.75, "least_squares_very_high"});
        GetGenericShapeModelParam(model, "needs_training", &v);
        Check(v.S() == "false", "Search params must not retrain");
        FindGenericShapeModel(im, model, &r, &n);
        Check(n.I() == 1, "Exactly one asymmetric target");
        HTuple rows, cols;
        GetGenericShapeModelResult(r, "all", "row", &rows);
        GetGenericShapeModelResult(r, "best", "column", &cols);
        Check(std::hypot(rows.D() - 44.63, cols.D() - 51.27) < 0.15,
              "Subpixel translation and edge-centered transform");
        GetGenericShapeModelResult(r, {"all", "best", 0, "test", model}, "num_match_result", &v);
        Check(v.I() == 1, "Selector de-duplication");
        Throws([&] { GetGenericShapeModelResult(r, 9, "score", &v); }, ErrorCode::Value);
        Throws([&] { GetGenericShapeModelResult(r, 0.0, "score", &v); }, ErrorCode::Type);
        Throws([&] { GetGenericShapeModelResult(r, "bogus", "score", &v); }, ErrorCode::Value);
        HObject local, placed;
        GetGenericShapeModelObject(&local, model, "contours");
        GetGenericShapeModelResultObject(&placed, r, 0, "contours");
        HTuple matrix;
        GetGenericShapeModelResult(r, 0, "hom_mat_2d", &matrix);
        Check(local.GetContours().size() == placed.GetContours().size(), "Contour object count");
        for (size_t i = 0; i < local.GetContours().size(); ++i)
            for (size_t j = 0; j < local.GetContours()[i].size(); ++j) {
                auto p = local.GetContours()[i][j], q = placed.GetContours()[i][j];
                double rr = matrix.D(0) * (p.row + 0.5) + matrix.D(1) * (p.column + 0.5) +
                            matrix.D(2) - 0.5;
                double cc = matrix.D(3) * (p.row + 0.5) + matrix.D(4) * (p.column + 0.5) +
                            matrix.D(5) - 0.5;
                Check(std::hypot(rr - q.row, cc - q.column) < 1e-9, "Contour matrix convention");
            }
        // Same pixels but a domain excluding the object origin must yield no result.
        auto raw = im.GetImage();
        std::vector<uint8_t> domain(raw.pixels.size(), 0);
        for (int y = 0; y < 10; ++y)
            for (int x = 0; x < 10; ++x)
                domain[size_t(y) * raw.width + x] = 1;
        HTuple empty;
        FindGenericShapeModel(HObject::FromGray(raw.width, raw.height, raw.pixels, domain), model,
                              &empty, &n);
        Check(n.I() == 0, "Search origin domain");
        GetGenericShapeModelResult(empty, "all", "row", &v);
        Check(v.Empty(), "Empty numeric tuple");
        GetGenericShapeModelResult(empty, "test", "num_match_result", &v);
        Check(v.I() == 0, "Known model with no matches");
        // Changed output origin must shift the reported transform, not the actual contour.
        SetGenericShapeModelParam(model, {"origin_row", "origin_column"}, {3.0, -2.0});
        HTuple shifted;
        FindGenericShapeModel(im, model, &shifted, &n);
        HTuple shifted_r, shifted_c, old_a;
        GetGenericShapeModelResult(shifted, 0, "row", &shifted_r);
        GetGenericShapeModelResult(shifted, 0, "column", &shifted_c);
        GetGenericShapeModelResult(r, 0, "angle", &old_a);
        double theta = -old_a.D();
        Check(std::abs((shifted_r.D() - rows.D()) - (std::sin(theta) * -2 + std::cos(theta) * 3)) <
                  1e-6,
              "Origin row transformed with pose");
        Check(std::abs((shifted_c.D() - cols.D()) - (std::cos(theta) * -2 - std::sin(theta) * 3)) <
                  1e-6,
              "Origin column transformed with pose");
        SetGenericShapeModelParam(model, {"origin_row", "origin_column"}, {0.0, 0.0});
        auto left = Render(160, 96, 40.2, 43.3), right = Render(160, 96, 119.7, 48.1);
        std::vector<float> pair = left.GetImage().pixels;
        for (size_t i = 0; i < pair.size(); ++i)
            pair[i] = std::min(pair[i], right.GetImage().pixels[i]);
        FindGenericShapeModel(HObject::FromGray(160, 96, pair), model, &shifted, &n);
        Check(n.I() == 2, "Two instances on full image");
        GetGenericShapeModelResult(shifted, {1, 0, 1}, "row", &v);
        Check(v.Length() == 2, "Multi-index selection sorts and de-duplicates");
        SetGenericShapeModelParam(model, "num_matches", 1);
        FindGenericShapeModel(HObject::FromGray(160, 96, pair), model, &shifted, &n);
        Check(n.I() == 1, "num_matches limit");
        auto blank = HObject::FromGray(96, 88, std::vector<float>(96 * 88, 230));
        FindGenericShapeModel(blank, model, &empty, &n);
        Check(n.I() == 0, "Uniform negative scene");
        HTuple duplicate;
        duplicate.Append(model).Append(model);
        Throws([&] { FindGenericShapeModel(im, duplicate, &shifted, &n); }, ErrorCode::Value);
        SetGenericShapeModelParam(model, "subpixel", "none");
        FindGenericShapeModel(im, model, &shifted, &n);
        Check(n.I() == 1, "Pixel-only mode");
        SetGenericShapeModelParam(model, "subpixel", "interpolation");
        FindGenericShapeModel(im, model, &shifted, &n);
        Check(n.I() == 1, "Score interpolation mode");
        SetGenericShapeModelParam(model, "subpixel", "least_squares_high");
        FindGenericShapeModel(im, model, &shifted, &n);
        Check(n.I() == 1, "High least-squares mode");
        // Global and local polarity behavior are checked on a contrast-inverted scene.
        SetGenericShapeModelParam(model, "metric", "ignore_global_polarity");
        TrainGenericShapeModel(Render(64, 64, 31.5, 31.5), model);
        auto inverted = im.GetImage().pixels;
        for (auto &f : inverted)
            f = 255 - f;
        FindGenericShapeModel(HObject::FromGray(96, 88, inverted), model, &shifted, &n);
        Check(n.I() == 1, "Global polarity reversal");
        SetGenericShapeModelParam(model, "metric", "ignore_local_polarity");
        TrainGenericShapeModel(Render(64, 64, 31.5, 31.5), model);
        FindGenericShapeModel(HObject::FromGray(96, 88, inverted), model, &shifted, &n);
        Check(n.I() == 1, "Local polarity reversal");
        HTuple alias = model;
        ClearShapeModel(model);
        Throws([&] { GetGenericShapeModelParam(alias, "min_score", &v); }, ErrorCode::Handle);
        GetGenericShapeModelResult(r, 0, "model_identifier", &v);
        Check(v.S() == "test", "Result snapshot survives model clear");
        CreateGenericShapeModel(&model);
        TrainGenericShapeModel(Render(64, 64, 31.5, 31.5), model);
        SetGenericShapeModelParam(model, "contrast_low", 8);
        GetGenericShapeModelParam(model, "needs_training", &v);
        Check(v.S() == "true", "Training parameter invalidation");
        ClearShapeModel(model);
        std::cout << "PASS " << checks << " assertions\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
