#include "shape_match/shape_match.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace shape_match;

/// In-process warm-cache benchmark. Training and file I/O occur before timing.
int main(int argc, char **argv) {
    try {
        const int iterations = argc > 1 ? std::stoi(argv[1]) : 5;
        if (iterations < 1 || iterations > 1000)
            throw std::runtime_error("iterations must be 1..1000");
        const std::string output = argc > 2 ? argv[2] : "results/benchmark.json";
        HTuple ring, nut, result, count;
        CreateGenericShapeModel(&ring);
        CreateGenericShapeModel(&nut);
        for (const auto &model : {ring, nut})
            SetGenericShapeModelParam(model, {"refinement_method", "refinement_radius"},
                                      {argc > 3 ? argv[3] : "nearest_point", argc > 4 ? std::stod(argv[4]) : 1.5});
        SetGenericShapeModelParam(ring, {"model_identifier", "iso_scale_min", "iso_scale_max"},
                                  {"ring", 0.8, 1.2});
        SetGenericShapeModelParam(nut, {"model_identifier", "iso_scale_min", "iso_scale_max"},
                                  {"nut", 0.6, 1.4});
        TrainGenericShapeModel(ReadPgm("data/template_ring.pgm", "data/template_ring_mask.pgm"),
                               ring);
        TrainGenericShapeModel(ReadPgm("data/template_nut.pgm", "data/template_nut_mask.pgm"), nut);
        constexpr double rad = 3.14159265358979323846 / 180;
        SetGenericShapeModelParam(ring, {"angle_start", "angle_end", "subpixel"},
                                  {-22.5 * rad, 22.5 * rad, "least_squares_very_high"});
        SetGenericShapeModelParam(nut, {"angle_start", "angle_end", "subpixel"},
                                  {-30 * rad, 60 * rad, "least_squares_very_high"});
        const HTuple models{ring, nut};
        const auto image = ReadPgm("data/original.pgm");
        FindGenericShapeModel(image, models, &result, &count); // warm-up
        std::vector<double> elapsed;
        for (int i = 0; i < iterations; ++i) {
            FindGenericShapeModel(image, models, &result, &count);
            if (count.I() != 7)
                throw std::runtime_error("Detection changed during performance run");
            elapsed.push_back(GetSearchDiagnostics(result).total_ms);
        }
        auto ordered = elapsed;
        std::sort(ordered.begin(), ordered.end());
        double median = ordered.size() % 2
                            ? ordered[ordered.size() / 2]
                            : (ordered[ordered.size() / 2 - 1] + ordered[ordered.size() / 2]) / 2;
        const double p95 =
            ordered[std::min(ordered.size() - 1, size_t(std::ceil(ordered.size() * 0.95)) - 1)];
        std::ofstream out(output);
        if (!out)
            throw std::runtime_error("Cannot write benchmark JSON");
        out << std::setprecision(12) << "{\"iterations\":" << iterations
            << ",\"warmup\":1,\"threads\":" << GetSearchThreadCount() << ",\"median_ms\":" << median
            << ",\"p95_ms\":" << p95 << ",\"min_ms\":" << ordered.front()
            << ",\"max_ms\":" << ordered.back() << ",\"samples_ms\":[";
        for (size_t i = 0; i < elapsed.size(); ++i) {
            if (i)
                out << ',';
            out << elapsed[i];
        }
        out << "]}\n";
        ClearShapeModel(models);
        std::cout << "median_ms=" << median << " p95_ms=" << p95
                  << " threads=" << GetSearchThreadCount() << " instances=7 runs=" << iterations
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
