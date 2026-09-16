#include "../src/internal.hpp"
#include "synthetic.hpp"
#include <iostream>

using namespace shape_match;
using namespace shape_match::detail;
static int checks = 0;
static void Check(bool condition, const char *message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}

static PyramidLevel LineField(bool reverse) {
    PyramidLevel image;
    auto &g = image.gradient;
    auto &field = image.field;
    g.width = field.width = 32;
    g.height = field.height = 24;
    g.gx.resize(768);
    g.gy.resize(768);
    g.mag.resize(768);
    field.nearest.resize(768);
    double sign = reverse ? -1 : 1;
    field.edges = {{{12.25, 9}, {sign, 0}, 100}, {{12.25, 11}, {sign, 0}, 100}};
    for (int y = 0; y < 24; ++y)
        for (int x = 0; x < 32; ++x) {
            size_t at = size_t(y) * 32 + x;
            g.mag[at] = float(100 * std::exp(-0.5 * Sq((x - 12.25) / 1.1)));
            g.gx[at] = float(sign * g.mag[at]);
            field.nearest[at] = y <= 10 ? 0 : 1;
        }
    return image;
}

int main() {
    try {
        std::vector<float> ramp(32 * 24);
        for (int y = 0; y < 24; ++y)
            for (int x = 0; x < 32; ++x)
                ramp[size_t(y) * 32 + x] = float(20 + 3 * x + 2 * y);
        auto object = HObject::FromGray(32, 24, ramp);
        auto g = GaussianGradients(object.GetImage());
        for (int y = 4; y < 20; ++y)
            for (int x = 4; x < 28; ++x) {
                size_t at = size_t(y) * 32 + x;
                Check(std::abs(g.gx[at] - 3) < 1e-5 && std::abs(g.gy[at] - 2) < 1e-5,
                      "Gaussian derivative normalization/sign");
            }
        auto constant = object.GetImage();
        std::fill(constant.pixels.begin(), constant.pixels.end(), 100);
        auto flat = GaussianGradients(constant);
        for (float value : flat.mag)
            Check(value < 1e-6, "Constant image has no Gaussian edges");

        for (double angle : {0.0, -0.13, pi / 2})
            for (double scale : {0.8, 1.0, 1.4}) {
                Match match;
                match.pose = {100.3, 80.7, angle, scale};
                match.model.params = {{"origin_row", 3.0}, {"origin_column", -2.0}};
                auto matrix = ResultTransform(match);
                Vec local{9.3, -7.1};
                Vec expected = Rotate(local + Vec{-2, 3}, angle) * scale + Vec{100.3, 80.7};
                Vec actual{matrix[3] * (local.y + 0.5) + matrix[4] * (local.x + 0.5) + matrix[5] - 0.5,
                           matrix[0] * (local.y + 0.5) + matrix[1] * (local.x + 0.5) + matrix[2] - 0.5};
                Check(std::hypot(expected.x - actual.x, expected.y - actual.y) < 1e-10,
                      "Independent edge/pixel-center transform and origin shift");
            }

        Features model{{{0, 0}, {1, 0}, 100}};
        Pose pose{12.8, 10, 0, 1};
        for (bool reverse : {false, true}) {
            auto image = LineField(reverse);
            for (const std::string method : {"contour", "gradient"}) {
                for (const std::string metric :
                     {"use_polarity", "ignore_global_polarity", "ignore_local_polarity"}) {
                    auto pairs = ContinuousCorrespondences(model, image, pose, metric, 1.5, method);
                    if (reverse && metric == "use_polarity") {
                        Check(pairs.empty(), "Polarity reversal must be rejected");
                        continue;
                    }
                    Check(pairs.size() == 1, "Continuous line correspondence");
                    Check(std::abs(pairs[0].image.x - 12.25) < (method == "contour" ? 1e-9 : 0.04),
                          "Subpixel line location");
                    Check(std::abs(pairs[0].image.y - 10) < 1e-9, "Continuous tangent coordinate");
                    Check(std::abs(std::hypot(pairs[0].normal.x, pairs[0].normal.y) - 1) < 1e-9,
                          "Unit correspondence normal");
                    Check(RefinementSupportLoss({}, pose, 1, 1.5) >=
                              RefinementSupportLoss(pairs, pose, 1, 1.5),
                          "Dropping an observation cannot lower capped support loss");
                }
                auto empty = ContinuousCorrespondences(model, image, {-20, -20, 0, 1},
                                                       "use_polarity", 1.5, method);
                Check(empty.empty(), "Outside image must not clamp to an edge");
            }
        }

        HTuple id;
        CreateGenericShapeModel(&id);
        SetGenericShapeModelParam(
            id, {"num_levels", "angle_start", "angle_end", "min_score", "num_matches"},
            {2, -0.2, 0.2, 0.5, 1});
        TrainGenericShapeModel(Render(64, 64, 31.5, 31.5, 0, 1, 8), id);
        bool rejected = false;
        try {
            SetGenericShapeModelParam(id, "refinement_radius", 0.0);
        } catch (const HException &) {
            rejected = true;
        }
        Check(rejected, "Invalid refinement radius rejected");
        rejected = false;
        try {
            SetGenericShapeModelParam(id, "refinement_method", "unknown");
        } catch (const HException &) {
            rejected = true;
        }
        Check(rejected, "Invalid refinement method rejected");
        for (const std::string method :
             {"nearest_point", "contour", "gradient", "gradient_gaussian"}) {
            SetGenericShapeModelParam(id, {"refinement_method", "subpixel"},
                                      {method, "least_squares_very_high"});
            HTuple state;
            GetGenericShapeModelParam(id, "needs_training", &state);
            Check(state.S() == "false", "Refinement selection does not invalidate training");
            HTuple result, count, row, column, scale;
            FindGenericShapeModel(Render(104, 96, 51.23, 44.67, 0.08, 1, 8), id, &result, &count);
            Check(count.I() == 1, "All refinement methods retain clean target");
            GetGenericShapeModelResult(result, 0, "row", &row);
            GetGenericShapeModelResult(result, 0, "column", &column);
            GetGenericShapeModelResult(result, 0, "scale_row", &scale);
            double truth_r = 44.67 + 0.5 - 0.5 * (std::cos(0.08) + std::sin(0.08));
            double truth_c = 51.23 + 0.5 - 0.5 * (std::cos(0.08) - std::sin(0.08));
            Check(std::hypot(row.D() - truth_r, column.D() - truth_c) < 0.08,
                  "Rotated fixed-scale precision");
            Check(scale.D() == 1, "Fixed scale must stay fixed");
        }
        ClearShapeModel(id);

        // Same valid pixels in cropped and full-size template coordinate systems.
        HTuple cropped, full;
        CreateGenericShapeModel(&cropped); CreateGenericShapeModel(&full);
        auto big = Render(104, 96, 51.5, 44.5, 0, 1, 8).GetImage();
        std::fill(big.domain.begin(), big.domain.end(), 0);
        for (int y = 13; y < 77; ++y)
            for (int x = 20; x < 84; ++x)
                big.domain[size_t(y) * big.width + x] = 1;
        for (const auto &model : {cropped, full})
            SetGenericShapeModelParam(model, {"num_levels", "iso_scale_min", "iso_scale_max",
                "angle_start", "angle_end", "min_score", "num_matches", "subpixel"},
                {2, 0.9, 1.1, -0.2, 0.2, 0.5, 1, "least_squares_very_high"});
        TrainGenericShapeModel(Render(64, 64, 31.5, 31.5, 0, 1, 8), cropped);
        TrainGenericShapeModel(HObject::FromGray(big.width, big.height, big.pixels, big.domain), full);
        HTuple result_c, result_f, count_c, count_f, rc, rf, cc, cf;
        auto scene = Render(104, 96, 52.3, 45.8, -0.11, 1.04, 8);
        FindGenericShapeModel(scene, cropped, &result_c, &count_c);
        FindGenericShapeModel(scene, full, &result_f, &count_f);
        Check(count_c.I() == 1 && count_f.I() == 1, "Cropped/full-domain template recall");
        GetGenericShapeModelResult(result_c, 0, "row", &rc);
        GetGenericShapeModelResult(result_c, 0, "column", &cc);
        GetGenericShapeModelResult(result_f, 0, "row", &rf);
        GetGenericShapeModelResult(result_f, 0, "column", &cf);
        Check(std::hypot(rc.D() - rf.D(), cc.D() - cf.D()) < 1e-4, "Crop coordinate invariance");
        ClearShapeModel(cropped); ClearShapeModel(full);
        std::cout << "PASS " << checks << " precision assertions\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
