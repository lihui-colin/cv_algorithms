#include "../src/internal.hpp"
#include "contour_residual_probe.hpp"
#include "normal_profile_probe.hpp"
#include "orientation_probe.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

using namespace shape_match;
using namespace shape_match::detail;

static Features ReadContourPositions(const std::string &path, const Features &original,
                                     bool project) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot read " + path);
    std::string line;
    std::getline(in, line);
    Features imported;
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int contour, point;
        Vec p;
        if (!(row >> contour >> point >> p.y >> p.x) || !std::isfinite(p.x) || !std::isfinite(p.y))
            throw std::runtime_error("Invalid contour CSV");
        if (original.empty())
            throw std::runtime_error("Empty original model");
        auto near =
            std::min_element(original.begin(), original.end(), [&](const auto &a, const auto &b) {
                return Dot(a.p - p, a.p - p) < Dot(b.p - p, b.p - p);
            });
        // Export lacks normals/weights. Reuse C++ normals, not HALCON internals.
        auto f = *near;
        f.p = p;
        f.contour = contour;
        imported.push_back(f);
    }
    if (imported.size() < 12)
        throw std::runtime_error("Insufficient imported points");
    if (!project)
        return imported;
    auto projected = original;
    for (auto &f : projected) {
        double best = 1e100;
        Vec q = f.p;
        for (size_t i = 1; i < imported.size(); ++i) {
            const auto &a = imported[i - 1], &b = imported[i];
            if (a.contour != b.contour)
                continue;
            Vec v = b.p - a.p;
            double length2 = Dot(v, v);
            if (length2 < 1e-16)
                continue;
            Vec p = a.p + v * std::clamp(Dot(f.p - a.p, v) / length2, 0.0, 1.0);
            double d = Dot(f.p - p, f.p - p);
            if (d < best) {
                best = d;
                q = p;
            }
        }
        f.p = q;
    }
    return projected;
}

static void ContourExperiment(const HTuple &result, const HObject &image, const std::string &data,
                              const std::string &output) {
    for (bool project : {false, true}) {
        std::ofstream out(output +
                          (project ? "/halcon_projected_refine.csv" : "/halcon_points_refine.csv"));
        out << "index,model,row,column,angle_deg,scale_row,scale_column,score\n"
            << std::setprecision(17);
        size_t index = 0;
        for (auto match : ResolveResult(result)->matches) {
            auto name = Str(match.model.params, "model_identifier");
            auto trained = std::make_shared<TrainedData>(*match.model.data);
            trained->dense = ReadContourPositions(
                data + "/reference/template_" + name + "_contours.csv", trained->dense, project);
            match.model.data = trained;
            auto pyramid =
                BuildSearchPyramid(image.GetImage(), 1, Num(match.model.params, "min_contrast"));
            auto refined = RefineInstance(match.model, pyramid[0], {match.pose, match.score});
            match.pose = refined.pose;
            auto h = ResultTransform(match);
            out << index++ << ',' << name << ',' << h[2] << ',' << h[5] << ','
                << -match.pose.theta * 180 / 3.14159265358979323846 << ',' << match.pose.scale
                << ',' << match.pose.scale << ',' << refined.score << '\n';
        }
        out.flush();
        if (!out)
            throw std::runtime_error("Cannot write contour diagnostic");
    }
}

// Offline parameter ablation. Reads exported model parameters, never GT poses.
static std::map<std::string, std::string> ReadParameters(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot read " + path);
    std::map<std::string, std::string> values;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string name, value;
        if (!std::getline(row, name, ',') || !std::getline(row, value, ','))
            throw std::runtime_error("Invalid parameter CSV: " + path);
        if (!values.emplace(name, value).second)
            throw std::runtime_error("Duplicate parameter: " + name);
    }
    return values;
}

int main(int argc, char **argv) {
    try {
        if (argc != 3 && !(argc == 4 && std::string(argv[3]) == "--evaluate-reference"))
            throw std::runtime_error(
                "Usage: halcon_parameter_probe DATA NEW_OUTPUT_DIR [--evaluate-reference]");
        const std::string data = argv[1], output = argv[2];
        if (!std::filesystem::create_directory(output))
            throw std::runtime_error("Use a new output directory");
        const auto image = ReadPgm(data + "/original.pgm");
        const std::vector<std::string> names{"ring", "nut"};
        const std::vector<std::vector<std::string>> changes{
            {},
            {"contrast_low", "contrast_high"},
            {"min_size"},
            {"num_levels"},
            {"optimization"},
            {"angle_step"},
            {"contrast_low", "contrast_high", "min_size", "num_levels", "optimization",
             "angle_step"},
            {"contrast_low", "contrast_high", "min_size", "optimization", "angle_step"}};
        const std::vector<std::string> labels{"baseline", "contrast",         "min_size",
                                              "levels",   "optimization",     "angle_step",
                                              "all",      "all_except_levels"};
        std::ofstream stats(output + "/runs.csv");
        stats << "experiment,count,search_ms\n" << std::setprecision(17);
        for (size_t experiment = 0; experiment < changes.size(); ++experiment) {
            HTuple ring, nut, result, count;
            CreateGenericShapeModel(&ring);
            CreateGenericShapeModel(&nut);
            const std::vector<HTuple> handles{ring, nut};
            for (size_t i = 0; i < names.size(); ++i) {
                const auto &model = handles[i];
                auto parameters =
                    ReadParameters(data + "/reference/template_" + names[i] + "_parameters.csv");
                SetGenericShapeModelParam(model,
                                          {"model_identifier", "iso_scale_min", "iso_scale_max"},
                                          {names[i], i == 0 ? 0.8 : 0.6, i == 0 ? 1.2 : 1.4});
                for (const auto &key : changes[experiment]) {
                    const auto &value = parameters.at(key);
                    if (key == "optimization")
                        SetGenericShapeModelParam(model, key, value);
                    else if (key == "min_size" || key == "num_levels")
                        SetGenericShapeModelParam(model, key, std::stoi(value));
                    else
                        SetGenericShapeModelParam(model, key, std::stod(value));
                }
                TrainGenericShapeModel(ReadPgm(data + "/template_" + names[i] + ".pgm",
                                               data + "/template_" + names[i] + "_mask.pgm"),
                                       model);
                constexpr double rad = 3.14159265358979323846 / 180;
                SetGenericShapeModelParam(model, {"angle_start", "angle_end", "subpixel"},
                                          {i == 0 ? -22.5 * rad : -30 * rad,
                                           i == 0 ? 22.5 * rad : 60 * rad,
                                           "least_squares_very_high"});
            }
            FindGenericShapeModel(image, {ring, nut}, &result, &count);
            if (experiment == 0)
                ContourExperiment(result, image, data, output);
            if (experiment == 0)
                profile_probe::Run(result, image, output);
            if (experiment == 0)
                orientation_probe::Run(result, image, output);
            if (experiment == 0)
                contour_residual_probe::Run(result, image, output);
            if (experiment == 0 && argc == 4)
                contour_residual_probe::Run(result, image, output,
                                            data + "/reference/matching_results.csv");
            std::ofstream out(output + "/" + labels[experiment] + ".csv");
            out << "index,model,row,column,angle_deg,scale_row,scale_column,score\n"
                << std::setprecision(17);
            for (int i = 0; i < count.I(); ++i) {
                HTuple name;
                GetGenericShapeModelResult(result, i, "model_identifier", &name);
                out << i << ',' << name.S();
                for (const auto &field :
                     {"row", "column", "angle", "scale_row", "scale_column", "score"}) {
                    HTuple value;
                    GetGenericShapeModelResult(result, i, field, &value);
                    out << ','
                        << (std::string(field) == "angle" ? value.D() * 180 / 3.14159265358979323846
                                                          : value.D());
                }
                out << '\n';
            }
            out.flush();
            if (!out)
                throw std::runtime_error("Cannot write results");
            stats << labels[experiment] << ',' << count.I() << ','
                  << GetSearchDiagnostics(result).total_ms << '\n';
            std::cout << labels[experiment] << " count=" << count.I() << '\n';
            ClearShapeModel({ring, nut});
        }
        stats.flush();
        if (!stats)
            throw std::runtime_error("Cannot write run summary");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
