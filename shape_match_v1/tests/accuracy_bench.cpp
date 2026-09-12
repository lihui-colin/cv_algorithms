#include "synthetic.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace shape_match;
int main(int argc, char **argv) {
    try {
        std::string output = argc > 1 ? argv[1] : "accuracy.csv";
        std::ofstream out(output);
        if (!out)
            throw std::runtime_error("Cannot write accuracy CSV");
        HTuple model;
        CreateGenericShapeModel(&model);
        SetGenericShapeModelParam(model, {"num_levels", "iso_scale_min", "iso_scale_max"},
                                  {2, 0.85, 1.15});
        TrainGenericShapeModel(Render(64, 64, 31.5, 31.5, 0, 1, 8), model);
        SetGenericShapeModelParam(
            model, {"angle_start", "angle_end", "subpixel", "min_score", "num_matches"},
            {-0.25, 0.25, "least_squares_very_high", 0.75, 1});
        out << std::setprecision(17)
            << "case,kind,truth_row,truth_column,truth_angle,truth_scale,found,row,column,angle,"
               "scale,position_error,angle_error,scale_error,total_ms\n";
        int found = 0;
        double sq = 0, max_error = 0;
        for (int i = 0; i < 30; ++i) {
            bool mixed = i >= 20;
            double x = 51 + (i % 5) * 0.2 + 0.03, y = 44 + (i / 5 % 4) * 0.25 + 0.07,
                   a = mixed ? (i - 25) * 0.037 : 0, s = mixed ? 0.90 + (i - 20) * 0.02 : 1;
            // Ground truth uses the translation of T(+.5) * T(center) * A * T(-.5),
            // exactly the convention needed to transform pixel-centered model XLDs.
            double truth_r = y + 0.5 - 0.5 * s * (std::cos(a) + std::sin(a));
            double truth_c = x + 0.5 - 0.5 * s * (std::cos(a) - std::sin(a));
            HTuple result, n;
            FindGenericShapeModel(Render(104, 96, x, y, a, s, 8), model, &result, &n);
            out << i << ',' << (mixed ? "pose" : "translation") << ',' << truth_r << ',' << truth_c
                << ',' << -a << ',' << s << ',' << n.I();
            if (n.I()) {
                HTuple r, c, angle, scale;
                GetGenericShapeModelResult(result, 0, "row", &r);
                GetGenericShapeModelResult(result, 0, "column", &c);
                GetGenericShapeModelResult(result, 0, "angle", &angle);
                GetGenericShapeModelResult(result, 0, "scale_row", &scale);
                double error = std::hypot(r.D() - truth_r, c.D() - truth_c);
                out << ',' << r.D() << ',' << c.D() << ',' << angle.D() << ',' << scale.D() << ','
                    << error << ',' << std::remainder(angle.D() + a, 2 * 3.14159265358979323846)
                    << ',' << scale.D() - s;
                found++;
                sq += error * error;
                max_error = std::max(max_error, error);
            } else
                out << ",,,,,,,";
            out << ',' << GetSearchDiagnostics(result).total_ms << '\n';
        }
        ClearShapeModel(model);
        std::cout << "found=" << found << "/30 rmse=" << (found ? std::sqrt(sq / found) : 0)
                  << " max=" << max_error << " (pixel)\n";
        return found == 30 ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
