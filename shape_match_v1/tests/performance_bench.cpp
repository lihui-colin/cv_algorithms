#include "shape_match/shape_match.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
#include "../src/internal.hpp"
#endif

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
        constexpr int warmup = 20;
        double cold_ms = 0;
        for (int i = 0; i < warmup; ++i) {
            FindGenericShapeModel(image, models, &result, &count);
            if (i == 0)
                cold_ms = GetSearchDiagnostics(result).total_ms;
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
            detail::TakeTrackingTraces();
#endif
        }
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
        std::vector<std::vector<detail::TrackingTrace>> tracking;
#endif
        std::vector<double> elapsed;
        std::vector<SearchDiagnostics> stages;
        std::vector<std::string> resources;
        for (int i = 0; i < iterations; ++i) {
            FindGenericShapeModel(image, models, &result, &count);
            if (count.I() != 7)
                throw std::runtime_error("Detection changed during performance run");
            elapsed.push_back(GetSearchDiagnostics(result).total_ms);
            stages.push_back(GetSearchDiagnostics(result));
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
            tracking.push_back(detail::TakeTrackingTraces());
#endif
            std::ifstream status("/proc/self/status");
            std::string line, resource;
            while (std::getline(status, line))
                if (line.find("VmRSS:") == 0 || line.find("Threads:") == 0)
                    resource += line + " ";
            resources.push_back(resource);
        }
        auto ordered = elapsed;
        std::sort(ordered.begin(), ordered.end());
        double median = ordered.size() % 2
                            ? ordered[ordered.size() / 2]
                            : (ordered[ordered.size() / 2 - 1] + ordered[ordered.size() / 2]) / 2;
        const double p95 =
            ordered[std::min(ordered.size() - 1, size_t(std::ceil(ordered.size() * 0.95)) - 1)];
        const double p99 = ordered[size_t(std::ceil(ordered.size() * .99)) - 1];
        std::ofstream out(output);
        if (!out)
            throw std::runtime_error("Cannot write benchmark JSON");
        out << std::setprecision(12) << "{\"iterations\":" << iterations
            << ",\"warmup\":" << warmup << ",\"threads\":" << GetSearchThreadCount() << ",\"median_ms\":" << median
            << ",\"p95_ms\":" << p95 << ",\"p99_ms\":" << p99 << ",\"cold_ms\":" << cold_ms
            << ",\"min_ms\":" << ordered.front()
            << ",\"max_ms\":" << ordered.back() << ",\"samples_ms\":[";
        for (size_t i = 0; i < elapsed.size(); ++i) {
            if (i)
                out << ',';
            out << elapsed[i];
        }
        out << "]}\n";
        std::ofstream detail(output + ".csv");
        detail << "iteration,total_ms,pyramid_ms,coarse_ms,tracking_ms,refinement_ms,evaluated_poses,resources\n"
               << std::setprecision(12);
        for (size_t i = 0; i < stages.size(); ++i) {
            const auto &d = stages[i];
            detail << i << ',' << d.total_ms << ',' << d.pyramid_ms << ',' << d.top_level_ms << ','
                   << d.tracking_ms << ',' << d.refinement_ms << ',' << d.evaluated_poses << ','
                   << resources[i] << '\n';
        }
        if (!detail || !out)
            throw std::runtime_error("Cannot write benchmark results");
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
        std::ofstream trace_out(output + ".tracking.csv");
        trace_out << "frame,batch,level,worker,dispatch_ms,start_ms,end_ms,completed_ms,cpu_ms,candidates,evaluations\n"
                  << std::setprecision(12);
        for (size_t frame = 0; frame < tracking.size(); ++frame)
            for (size_t batch = 0; batch < tracking[frame].size(); ++batch) {
                const auto &t = tracking[frame][batch];
                for (size_t worker = 0; worker < t.workers.size(); ++worker) {
                    const auto &w = t.workers[worker];
                    trace_out << frame << ',' << batch << ',' << t.level << ','
                              << worker << ',' << t.dispatch_ms << ',' << w.start_ms << ','
                              << w.end_ms << ',' << t.completed_ms << ',' << w.cpu_ms << ','
                              << w.candidates << ',' << w.evaluations << '\n';
                }
            }
        if (!trace_out)
            throw std::runtime_error("Cannot write tracking diagnostics");
#endif
        ClearShapeModel(models);
        std::cout << "median_ms=" << median << " p95_ms=" << p95 << " p99_ms=" << p99
                  << " threads=" << GetSearchThreadCount() << " instances=7 runs=" << iterations
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
