// Offline diagnosis only: GT is never passed to FindGenericShapeModel.
// Reference-seeded refinement deliberately tests the local objective, not recall.
#include "../src/internal.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace shape_match;
using namespace shape_match::detail;

struct Reference {
    int index;
    std::string model;
    double row, column, angle, scale;
};

static std::vector<Reference> ReadReferences(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot read reference CSV: " + path);
    std::vector<Reference> references;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        if (line.empty())
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        Reference r{};
        double column_scale, score;
        if (!(row >> r.index >> r.model >> r.row >> r.column >> r.angle >> r.scale >>
              column_scale >> score) ||
            !std::isfinite(r.row) || !std::isfinite(r.column) || !std::isfinite(r.angle) ||
            !std::isfinite(r.scale) || r.scale <= 0 || column_scale != r.scale)
            throw std::runtime_error("Invalid isotropic reference CSV row");
        // Bundled sample's historical export-label correction; see data/PROVENANCE.md.
        r.model = r.index == 1 || r.index == 2 || r.index == 5 ? "ring" : "nut";
        references.push_back(r);
    }
    if (references.size() != 7)
        throw std::runtime_error("This diagnostic expects the bundled seven-object sample");
    return references;
}

struct Observation {
    size_t pairs = 0, disagreements = 0;
    double squared_residual = 0;
};

// Geometric connected components, not the fragmented visualization polyline IDs.
// Each component is fitted independently for diagnosis, never used to select a detection.
static void WriteGroups(std::ostream &out, const Match &match, const Reference &ref,
                        const PyramidLevel &image) {
    const auto &features = match.model.data->dense;
    std::vector<size_t> parent(features.size());
    std::iota(parent.begin(), parent.end(), size_t(0));
    auto root = [&](size_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (size_t i = 0; i < features.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (Dot(features[i].p - features[j].p, features[i].p - features[j].p) < 3.3)
                parent[root(i)] = root(j);
    std::map<size_t, Features> groups;
    for (size_t i = 0; i < features.size(); ++i)
        groups[root(i)].push_back(features[i]);
    for (const auto &group : groups) {
        if (group.second.size() < 12)
            continue;
        auto data = std::make_shared<TrainedData>(*match.model.data);
        data->dense = group.second;
        Match local = match;
        local.model.data = data;
        local.pose = RefineInstance(local.model, image, {match.pose, match.score}).pose;
        auto matrix = ResultTransform(local);
        Vec center;
        double min_x = 1e30, min_y = 1e30, max_x = -1e30, max_y = -1e30;
        for (const auto &f : group.second) {
            center = center + f.p;
            min_x = std::min(min_x, f.p.x);
            max_x = std::max(max_x, f.p.x);
            min_y = std::min(min_y, f.p.y);
            max_y = std::max(max_y, f.p.y);
        }
        center = center * (1.0 / group.second.size());
        out << ref.index << ',' << ref.model << ',' << group.first << ',' << group.second.size()
            << ',' << center.y << ',' << center.x << ',' << max_y - min_y << ',' << max_x - min_x
            << ',' << matrix[2] << ',' << matrix[5] << ',' << -local.pose.theta * 180 / pi << ','
            << local.pose.scale << ',' << std::hypot(matrix[2] - ref.row, matrix[5] - ref.column)
            << '\n';
    }
}

// Independently audit the production 3x3 Voronoi-label query against all scene
// edges. The O(model_points * scene_points) oracle is never used in matching.
static Observation Observe(const Match &match, const EdgeField &field) {
    Observation out;
    for (const auto &f : match.model.data->dense) {
        Vec p = Rotate(f.p, match.pose.theta) * match.pose.scale + Vec{match.pose.x, match.pose.y};
        Vec n = Rotate(f.n, match.pose.theta);
        auto consider = [&](int id, double &cost, int &best) {
            if (id < 0)
                return;
            const auto &e = field.edges[size_t(id)];
            double distance = Dot(p - e.p, p - e.p);
            if (Dot(n, e.n) >= 0.65 && distance < cost) {
                cost = distance;
                best = id;
            }
        };
        int x = int(std::lround(p.x)), y = int(std::lround(p.y)), nearest = -1, exact = -1;
        double cost = 1.5 * 1.5, exact_cost = cost;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (x + dx >= 0 && y + dy >= 0 && x + dx < field.width && y + dy < field.height)
                    consider(field.nearest[size_t(y + dy) * field.width + x + dx], cost, nearest);
        for (size_t i = 0; i < field.edges.size(); ++i)
            consider(int(i), exact_cost, exact);
        // Equal-distance ties with identical residuals need not have identical IDs.
        if (nearest != exact && (nearest < 0 || exact < 0 || std::abs(cost - exact_cost) > 1e-12))
            ++out.disagreements;
        if (nearest >= 0) {
            const auto &e = field.edges[size_t(nearest)];
            out.squared_residual += Sq(Dot(e.n, p - e.p));
            ++out.pairs;
        }
    }
    return out;
}

int main(int argc, char **argv) {
    try {
        if (argc > 3)
            throw std::runtime_error("Usage: refinement_probe [data_directory] [output.csv]");
        const std::string data = argc > 1 ? argv[1] : "data";
        const std::string output = argc > 2 ? argv[2] : "refinement_probe.csv";
        auto references = ReadReferences(data + "/reference/matching_results.csv");
        HTuple models;
        for (const std::string name : {"ring", "nut"}) {
            const bool ring = name == "ring";
            HTuple model;
            CreateGenericShapeModel(&model);
            SetGenericShapeModelParam(model, {"model_identifier", "iso_scale_min", "iso_scale_max"},
                                      {name, ring ? 0.8 : 0.6, ring ? 1.2 : 1.4});
            TrainGenericShapeModel(ReadPgm(data + "/template_" + name + ".pgm",
                                           data + "/template_" + name + "_mask.pgm"),
                                   model);
            SetGenericShapeModelParam(model, {"angle_start", "angle_end", "subpixel"},
                                      {(ring ? -22.5 : -30.0) * pi / 180,
                                       (ring ? 22.5 : 60.0) * pi / 180, "least_squares_very_high"});
            models.Append(model);
        }
        auto image = ReadPgm(data + "/original.pgm");
        HTuple result, count;
        FindGenericShapeModel(image, models, &result, &count);
        const auto matches = ResolveResult(result)->matches;
        if (matches.size() != references.size())
            throw std::runtime_error("Sample recall changed: expected exactly seven detections");
        std::ofstream csv(output);
        if (!csv)
            throw std::runtime_error("Cannot write diagnostic CSV: " + output);
        std::ofstream groups(output + ".groups.csv");
        if (!groups)
            throw std::runtime_error("Cannot write component diagnostic CSV");
        groups << std::setprecision(17)
               << "reference_index,model,component,points,center_row,center_column,height,width,"
                  "fitted_row,fitted_column,fitted_angle_deg,fitted_scale,position_error_px\n";
        csv << std::setprecision(17)
            << "mode,reference_index,model,row,column,angle_deg,scale,position_error_px,"
               "edge_rms_px,pairs,model_points,nearest_disagreements\n";
        std::set<size_t> used;
        std::map<std::string, double> errors;
        for (const auto &r : references) {
            size_t index = matches.size();
            double distance = 3;
            for (size_t i = 0; i < matches.size(); ++i) {
                if (used.count(i) || Str(matches[i].model.params, "model_identifier") != r.model)
                    continue;
                auto h = ResultTransform(matches[i]);
                double d = std::hypot(h[2] - r.row, h[5] - r.column);
                if (d < distance) {
                    index = i;
                    distance = d;
                }
            }
            if (index == matches.size())
                throw std::runtime_error("Unassociated reference: " + std::to_string(r.index));
            used.insert(index);
            Match match = matches[index];
            auto pyramid =
                BuildSearchPyramid(image.GetImage(), 1, Num(match.model.params, "min_contrast"));
            WriteGroups(groups, match, r, pyramid[0]);
            auto emit = [&](const std::string &mode, Pose pose) {
                Match m = match;
                m.pose = pose;
                auto h = ResultTransform(m);
                auto observation = Observe(m, pyramid[0].field);
                if (!observation.pairs)
                    throw std::runtime_error("No refinement correspondences");
                double error = std::hypot(h[2] - r.row, h[5] - r.column);
                errors[mode] += error * error;
                csv << mode << ',' << r.index << ',' << r.model << ',' << h[2] << ',' << h[5] << ','
                    << -pose.theta * 180 / pi << ',' << pose.scale << ',' << error << ','
                    << std::sqrt(observation.squared_residual / observation.pairs) << ','
                    << observation.pairs << ',' << m.model.data->dense.size() << ','
                    << observation.disagreements << '\n';
            };
            emit("baseline", match.pose);
            Candidate repeated{match.pose, match.score};
            for (int i = 0; i < 4; ++i)
                repeated = RefineInstance(match.model, pyramid[0], repeated);
            emit("extra_120_budget", repeated.pose);
            Pose reference{0, 0, -r.angle * pi / 180, r.scale};
            double c = reference.scale * std::cos(reference.theta),
                   s = reference.scale * std::sin(reference.theta);
            // Inverse of ResultTransform for the default (zero-offset) model origin.
            reference.x = r.column - 0.5 + 0.5 * (c - s);
            reference.y = r.row - 0.5 + 0.5 * (c + s);
            emit("reference", reference);
            emit("reference_refined", RefineInstance(match.model, pyramid[0], {reference, 1}).pose);
        }
        csv.flush();
        groups.flush();
        if (!csv || !groups)
            throw std::runtime_error("Failed writing diagnostic CSV");
        ClearShapeModel(models);
        for (const auto &entry : errors)
            std::cout << entry.first
                      << " position_rmse_px=" << std::sqrt(entry.second / references.size())
                      << '\n';
        std::cout
            << "GT-seeded modes diagnose the objective; they are not detection accuracy tests.\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
