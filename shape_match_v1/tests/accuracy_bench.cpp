#include "synthetic.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
using namespace shape_match;
int main(int argc, char **argv) {
    try {
        std::string output = argc > 1 ? argv[1] : "accuracy.csv";
        std::ofstream out(output);
        if (!out)
            throw std::runtime_error("Cannot write accuracy CSV");
        HTuple model;
        CreateGenericShapeModel(&model);
        SetGenericShapeModelParam(model, {"refinement_method", "refinement_radius"},
                                  {argc > 2 ? argv[2] : "nearest_point", argc > 3 ? std::stod(argv[3]) : 1.5});
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
        const bool extended = argc > 4 && std::string(argv[4]) == "extended";
        const int cases = extended ? 70 : 30;
        std::map<std::string, std::pair<double, int>> groups;
        double sq = 0, max_error = 0;
        for (int i = 0; i < cases; ++i) {
            bool mixed = i >= 20;
            double x = 51 + (i % 5) * 0.2 + 0.03, y = 44 + (i / 5 % 4) * 0.25 + 0.07,
                   a = mixed ? (i - 25) * 0.037 : 0, s = mixed ? 0.90 + (i - 20) * 0.02 : 1;
            std::string kind = mixed ? "pose" : "translation";
            double noise = 0, blur = 0.9, hole_dx = 0;
            if (i >= 30) {
                a = (i % 7 - 3) * 0.037;
                s = 0.9 + (i % 5) * 0.04;
                if (i < 40) { kind = "noise"; noise = 1 + (i % 5); }
                else if (i < 50) { kind = "blur"; blur = 0.65 + (i % 5) * 0.15; }
                else if (i < 60) { kind = "occlusion"; }
                else { kind = "deformation"; hole_dx = (i % 5 - 2) * 0.25; }
            }
            // Ground truth uses the translation of T(+.5) * T(center) * A * T(-.5),
            // exactly the convention needed to transform pixel-centered model XLDs.
            double truth_r = y + 0.5 - 0.5 * s * (std::cos(a) + std::sin(a));
            double truth_c = x + 0.5 - 0.5 * s * (std::cos(a) - std::sin(a));
            HTuple result, n;
            auto scene = Render(104, 96, x, y, a, s, 8, noise, blur, hole_dx);
            if (kind == "occlusion") {
                auto raw = scene.GetImage();
                for (int yy = int(y) - 16; yy < int(y) - 6; ++yy)
                    for (int xx = int(x) - 3; xx < int(x) + 3; ++xx)
                        raw.pixels[size_t(yy) * raw.width + xx] = 230;
                scene = HObject::FromGray(raw.width, raw.height, raw.pixels, raw.domain);
            }
            FindGenericShapeModel(scene, model, &result, &n);
            out << i << ',' << kind << ',' << truth_r << ',' << truth_c
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
                groups[kind].first += error * error;
                groups[kind].second++;
            } else
                out << ",,,,,,,";
            out << ',' << GetSearchDiagnostics(result).total_ms << '\n';
        }
        ClearShapeModel(model);
        std::cout << "found=" << found << '/' << cases << " rmse=" << (found ? std::sqrt(sq / found) : 0)
                  << " max=" << max_error << " (pixel)\n";
        bool passed = found == cases;
        for (const auto &group : groups) {
            double rmse = std::sqrt(group.second.first / group.second.second);
            double limit = group.first == "translation" || group.first == "pose" ? 1.0 / 30 :
                           group.first == "occlusion" || group.first == "deformation" ? 0.3 : 0.1;
            passed &= rmse <= limit;
            std::cout << group.first << " rmse=" << rmse << " limit=" << limit << '\n';
        }
        return passed ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
