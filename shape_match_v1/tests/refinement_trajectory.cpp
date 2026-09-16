// Compile the real search implementation with diagnostics only in this executable.
// The production library does not define SHAPE_MATCH_TRACE_REFINEMENT.
#include "../src/internal.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdlib>

namespace shape_match::detail {
struct TraceRow {
    size_t id;
    ModelSnapshot model;
    Candidate candidate;
    int iteration;
    std::string event;
};
static std::mutex trace_mutex;
static std::vector<TraceRow> trace_rows;
static size_t next_trace_id = 0;
static thread_local size_t trace_id = 0;
static int diagnostic_budget = -1;
static double diagnostic_radius = 0;
static std::string seed_mode;
static bool point_to_point = false;
static std::string update_policy = "native";
static std::vector<std::string> update_rows;
static void TraceUpdate(int iteration, double radius, size_t count, size_t trial_count,
                        double before, double frozen, double reassociated, double support_before,
                        double support_after, double damping, double step_px, bool accepted) {
    std::ostringstream row;
    row << std::setprecision(17) << trace_id << ',' << iteration << ',' << radius << ',' << count
        << ',' << trial_count << ',' << before << ',' << frozen << ',' << reassociated << ','
        << support_before << ',' << support_after << ',' << damping << ',' << step_px << ',' << accepted;
    std::lock_guard<std::mutex> lock(trace_mutex);
    update_rows.push_back(row.str());
}
static bool TracePointToPointResidual() {
    return point_to_point;
}
struct Seed {
    std::string model;
    double row, column;
    Pose pose;
};
static std::vector<Seed> diagnostic_seeds;
static Candidate TraceRefinementSeed(const ModelSnapshot &model, const PyramidLevel &image,
                                     Candidate candidate) {
    if (seed_mode.empty())
        return candidate;
    if (seed_mode == "score_interpolation")
        candidate.pose = InterpolateScorePeak(model, image, candidate.pose);
    else {
        const auto h = ResultTransform({model, candidate.pose, candidate.score});
        const Seed *selected = nullptr;
        for (const auto &seed : diagnostic_seeds)
            if (seed.model == Str(model.params, "model_identifier") &&
                std::hypot(seed.row - h[2], seed.column - h[5]) < 3) {
                if (selected)
                    throw std::runtime_error("Ambiguous diagnostic seed association");
                selected = &seed;
            }
        if (selected)
            candidate.pose = selected->pose;
    }
    candidate.score = EvaluatePose(model.data->levels[0].features, image.field, candidate.pose,
                                   Str(model.params, "metric"), 0.8);
    return candidate;
}
static double TraceCorrespondenceRadius(double original) {
    return diagnostic_radius > 0 ? diagnostic_radius : original;
}
static int TraceIterationBudget(int original) {
    return diagnostic_budget < 0 ? original : diagnostic_budget;
}
static void TraceRefinement(const ModelSnapshot &m, Candidate c, int iteration, const char *event) {
    std::lock_guard<std::mutex> lock(trace_mutex);
    if (std::string(event) == "begin")
        trace_id = next_trace_id++;
    trace_rows.push_back({trace_id, m, c, iteration, event});
}
} // namespace shape_match::detail
#define SHAPE_MATCH_TRACE_REFINEMENT
#include "../src/search.cpp"

int main(int argc, char **argv) {
    using namespace shape_match;
    using namespace shape_match::detail;
    try {
        if (const char *policy = std::getenv("SHAPE_MATCH_TRACE_UPDATE"))
            update_policy = policy;
        if (update_policy != "native" && update_policy != "frozen" && update_policy != "support")
            throw std::runtime_error("Unknown diagnostic update policy");
        if (argc < 3 || argc > 7)
            throw std::runtime_error("Usage: refinement_trajectory DATA OUTPUT_PREFIX "
                                     "[BUDGET [RADIUS_OR_0 [native|score_interpolation|SEED_CSV "
                                     "[point_to_point]]]]");
        std::string data = argv[1], prefix = argv[2];
        if (argc >= 4) {
            diagnostic_budget = std::stoi(argv[3]);
            if (diagnostic_budget < 0 || diagnostic_budget > 100)
                throw std::runtime_error("Budget must be 0..100");
        }
        if (argc >= 5) {
            diagnostic_radius = std::stod(argv[4]);
            if (!std::isfinite(diagnostic_radius) ||
                (diagnostic_radius != 0 && diagnostic_radius < 0.5) || diagnostic_radius > 10)
                throw std::runtime_error("Radius must be 0 (native schedule) or 0.5..10");
        }
        if (argc >= 6) {
            seed_mode = argv[5];
            if (seed_mode == "native")
                seed_mode.clear();
            if (!seed_mode.empty() && seed_mode != "score_interpolation") {
                std::ifstream in(seed_mode);
                if (!in)
                    throw std::runtime_error("Cannot read diagnostic seeds");
                std::string line;
                std::getline(in, line);
                while (std::getline(in, line)) {
                    std::replace(line.begin(), line.end(), ',', ' ');
                    std::istringstream row(line);
                    Seed seed;
                    int index;
                    double angle, scale, scale_column, score;
                    if (!(row >> index >> seed.model >> seed.row >> seed.column >> angle >> scale >>
                          scale_column >> score) ||
                        !std::isfinite(seed.row + seed.column + angle + scale) || scale <= 0 ||
                        scale != scale_column)
                        throw std::runtime_error("Invalid diagnostic seed row");
                    double theta = -angle * pi / 180, c = scale * std::cos(theta),
                           s = scale * std::sin(theta);
                    seed.pose = {seed.column - .5 + .5 * (c - s), seed.row - .5 + .5 * (c + s),
                                 theta, scale};
                    diagnostic_seeds.push_back(seed);
                }
                if (diagnostic_seeds.empty())
                    throw std::runtime_error("Empty diagnostic seeds");
                std::cerr << "DIAGNOSTIC: imported poses; not a standalone detector evaluation\n";
            }
        }
        if (argc == 7) {
            if (std::string(argv[6]) != "point_to_point")
                throw std::runtime_error("Unknown residual mode");
            point_to_point = true;
        }
        HTuple ring, nut, result, count;
        CreateGenericShapeModel(&ring);
        CreateGenericShapeModel(&nut);
        for (int i = 0; i < 2; ++i) {
            auto model = i ? nut : ring;
            std::string name = i ? "nut" : "ring";
            SetGenericShapeModelParam(model, {"model_identifier", "iso_scale_min", "iso_scale_max"},
                                      {name, i ? 0.6 : 0.8, i ? 1.4 : 1.2});
            TrainGenericShapeModel(ReadPgm(data + "/template_" + name + ".pgm",
                                           data + "/template_" + name + "_mask.pgm"),
                                   model);
            SetGenericShapeModelParam(model, {"angle_start", "angle_end", "subpixel"},
                                      {(i ? -30 : -22.5) * pi / 180, (i ? 60 : 22.5) * pi / 180,
                                       "least_squares_very_high"});
        }
        FindGenericShapeModel(ReadPgm(data + "/original.pgm"), {ring, nut}, &result, &count);
        std::ofstream out(prefix + ".csv"), trace(prefix + ".trace.csv");
        out << "index,model,row,column,angle_deg,scale_row,scale_column,score\n"
            << std::setprecision(17);
        trace << "trace_id,model,event,iteration,row,column,angle_deg,scale,score\n"
              << std::setprecision(17);
        size_t index = 0;
        for (const auto &m : ResolveResult(result)->matches) {
            auto h = ResultTransform(m);
            out << index++ << ',' << Str(m.model.params, "model_identifier") << ',' << h[2] << ','
                << h[5] << ',' << -m.pose.theta * 180 / pi << ',' << m.pose.scale << ','
                << m.pose.scale << ',' << m.score << '\n';
        }
        for (const auto &r : trace_rows) {
            Match m{r.model, r.candidate.pose, r.candidate.score};
            auto h = ResultTransform(m);
            trace << r.id << ',' << Str(m.model.params, "model_identifier") << ',' << r.event << ','
                  << r.iteration << ',' << h[2] << ',' << h[5] << ',' << -m.pose.theta * 180 / pi
                  << ',' << m.pose.scale << ',' << m.score << '\n';
        }
        out.flush();
        trace.flush();
        std::ofstream updates(prefix + ".updates.csv");
        updates << "trace_id,iteration,radius,count,trial_count,loss_before,frozen_trial_loss,"
                   "reassociated_trial_loss,support_before,support_after,damping,step_px,accepted\n";
        for (const auto &row : update_rows)
            updates << row << '\n';
        updates.flush();
        if (!out || !trace || !updates)
            throw std::runtime_error("Cannot write trajectory output");
        ClearShapeModel({ring, nut});
        std::cout << "matches=" << count.I() << " traces=" << next_trace_id << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
