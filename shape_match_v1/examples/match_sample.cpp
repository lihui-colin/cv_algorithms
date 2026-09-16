#include "shape_match/shape_match.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace shape_match;
struct TemplateSpec {
    std::string name, path, mask;
    double min_scale, max_scale, start, end;
};
// JSON escaping is shared by identifiers and string metadata, never interpolated raw.
static std::string Json(const std::string &s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else
            out << c;
    }
    out << '"';
    return out.str();
}
static std::string Csv(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += '"';
        out += c;
    }
    return out + '"';
}
static void WriteDiagnostics(const std::string &output, const HTuple &models,
                             const HTuple &result, const std::vector<TemplateSpec> &templates) {
    std::ofstream params(output + "/model_params.csv"), contours(output + "/model_contours.csv"),
        transforms(output + "/matching_transforms.csv");
    if (!params || !contours || !transforms)
        throw std::runtime_error("Cannot create alignment diagnostics");
    params << std::setprecision(17) << "model,parameter,value\n";
    contours << std::setprecision(17) << "model,contour,point,row,column\n";
    transforms << std::setprecision(17) << "index,model,m00,m01,m02,m10,m11,m12\n";
    for (size_t i = 0; i < models.Length(); ++i) {
        HTuple id(models.H(i));
        const auto &spec = templates[i];
        for (const char *key : {"subpixel", "refinement_method", "refinement_radius", "metric",
             "contrast_low", "contrast_high", "min_contrast", "min_size", "optimization",
             "num_levels", "angle_start", "angle_end", "angle_step", "iso_scale_min",
             "iso_scale_max", "iso_scale_step", "origin_row", "origin_column",
             "pyramid_level_lowest", "pyramid_level_highest", "greediness", "min_score"}) {
            HTuple value;
            GetGenericShapeModelParam(id, key, &value);
            params << Csv(spec.name) << ',' << key << ',';
            if (const auto *s = std::get_if<std::string>(&value.At(0)))
                params << Csv(*s);
            else
                params << value.D();
            params << '\n';
        }
        auto object = ReadPgm(spec.path, spec.mask);
        const auto &im = object.GetImage();
        double row = 0, column = 0, pixels = 0;
        for (int y = 0; y < im.height; ++y)
            for (int x = 0; x < im.width; ++x)
                if (im.domain[size_t(y) * im.width + x]) {
                    row += y; column += x; ++pixels;
                }
        params << Csv(spec.name) << ",template_domain_row," << row / pixels << '\n'
               << Csv(spec.name) << ",template_domain_column," << column / pixels << '\n'
               << Csv(spec.name) << ",template_width," << im.width << '\n'
               << Csv(spec.name) << ",template_height," << im.height << '\n';
        HObject model_contours;
        GetGenericShapeModelObject(&model_contours, id, "contours");
        const auto &lines = model_contours.GetContours();
        for (size_t c = 0; c < lines.size(); ++c)
            for (size_t p = 0; p < lines[c].size(); ++p)
                contours << Csv(spec.name) << ',' << c << ',' << p << ',' << lines[c][p].row
                         << ',' << lines[c][p].column << '\n';
    }
    HTuple count;
    GetGenericShapeModelResult(result, "all", "num_match_result", &count);
    for (int64_t i = 0; i < count.I(); ++i) {
        HTuple name, matrix;
        GetGenericShapeModelResult(result, i, "model_identifier", &name);
        GetGenericShapeModelResult(result, i, "hom_mat_2d", &matrix);
        transforms << i << ',' << Csv(name.S());
        for (size_t j = 0; j < matrix.Length(); ++j)
            transforms << ',' << matrix.D(j);
        transforms << '\n';
    }
    params.flush(); contours.flush(); transforms.flush();
    if (!params || !contours || !transforms)
        throw std::runtime_error("Failed writing alignment diagnostics");
}
int main(int argc, char **argv) {
    try {
        std::string image = "data/original.pgm", output = "results",
                    mode = "least_squares_very_high", refinement = "nearest_point";
        std::vector<TemplateSpec> templates;
        double min_score = 0.5;
        double refinement_radius = 1.5;
        bool diagnostics = false;
        int levels = 0, matches = 0;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&]() {
                if (++i >= argc)
                    throw std::runtime_error("Missing value for " + arg);
                return std::string(argv[i]);
            };
            if (arg == "--image")
                image = next();
            else if (arg == "--output-dir")
                output = next();
            else if (arg == "--min-score")
                min_score = std::stod(next());
            else if (arg == "--subpixel")
                mode = next();
            else if (arg == "--refinement-method")
                refinement = next();
            else if (arg == "--refinement-radius")
                refinement_radius = std::stod(next());
            else if (arg == "--diagnostics")
                diagnostics = true;
            else if (arg == "--num-levels")
                levels = std::stoi(next());
            else if (arg == "--num-matches")
                matches = std::stoi(next());
            else if (arg == "--template") {
                TemplateSpec t;
                t.name = next();
                t.path = next();
                t.mask = next();
                if (t.mask == "-")
                    t.mask.clear();
                t.min_scale = std::stod(next());
                t.max_scale = std::stod(next());
                t.start = std::stod(next());
                t.end = std::stod(next());
                templates.push_back(t);
            } else if (arg == "--help") {
                std::cout << "match_sample [--image PGM] [--template NAME PGM MASK_OR_- SCALE_MIN "
                             "SCALE_MAX ANGLE_START_DEG ANGLE_END_DEG]... [--output-dir DIR] "
                             "[--min-score 0.5] [--subpixel least_squares_very_high] [--num-levels "
                             "N] [--num-matches N] [--refinement-method nearest_point|contour|gradient|gradient_gaussian] "
                             "[--refinement-radius 1.5] [--diagnostics]\nNo --template: use bundled ring/nut templates. "
                             "No ROI preselection.\n";
                return 0;
            } else
                throw std::runtime_error("Unknown option: " + arg);
        }
        if (templates.empty())
            templates = {
                {"ring", "data/template_ring.pgm", "data/template_ring_mask.pgm", 0.8, 1.2, -22.5,
                 22.5},
                {"nut", "data/template_nut.pgm", "data/template_nut_mask.pgm", 0.6, 1.4, -30, 60}};
        HTuple models;
        for (const auto &spec : templates) {
            HTuple id;
            CreateGenericShapeModel(&id);
            SetGenericShapeModelParam(id, {"model_identifier", "iso_scale_min", "iso_scale_max"},
                                      {spec.name, spec.min_scale, spec.max_scale});
            if (levels)
                SetGenericShapeModelParam(id, "num_levels", levels);
            auto object = ReadPgm(spec.path, spec.mask);
            TrainGenericShapeModel(object, id);
            SetGenericShapeModelParam(id, {"refinement_method", "refinement_radius"},
                                      {refinement, refinement_radius});
            SetGenericShapeModelParam(
                id, {"angle_start", "angle_end", "min_score", "subpixel", "num_matches"},
                {spec.start * 3.14159265358979323846 / 180, spec.end * 3.14159265358979323846 / 180,
                 min_score, mode, matches});
            models.Append(id);
        }
        HTuple result, count;
        FindGenericShapeModel(ReadPgm(image), models, &result, &count);
        std::filesystem::create_directories(output);
        if (diagnostics)
            WriteDiagnostics(output, models, result, templates);
        std::ofstream csv(output + "/matching_results.csv"), json(output + "/contours.json"),
            timing(output + "/timing.json");
        if (!csv || !json || !timing)
            throw std::runtime_error("Cannot create output files");
        csv << std::setprecision(17)
            << "index,model,row,column,angle_deg,scale_row,scale_column,score\n";
        json << std::setprecision(17) << "[\n";
        for (int64_t i = 0; i < count.I(); ++i) {
            HTuple name, r, c, a, sr, sc, score;
            GetGenericShapeModelResult(result, i, "model_identifier", &name);
            GetGenericShapeModelResult(result, i, "row", &r);
            GetGenericShapeModelResult(result, i, "column", &c);
            GetGenericShapeModelResult(result, i, "angle", &a);
            GetGenericShapeModelResult(result, i, "scale_row", &sr);
            GetGenericShapeModelResult(result, i, "scale_column", &sc);
            GetGenericShapeModelResult(result, i, "score", &score);
            csv << i << ',' << Csv(name.S()) << ',' << r.D() << ',' << c.D() << ','
                << a.D() * 180 / 3.14159265358979323846 << ',' << sr.D() << ',' << sc.D() << ','
                << score.D() << '\n';
            std::cout << i << ' ' << name.S() << " row=" << r.D() << " col=" << c.D()
                      << " angle=" << a.D() * 180 / 3.14159265358979323846 << " scale=" << sr.D()
                      << " score=" << score.D() << '\n';
            HObject contours;
            GetGenericShapeModelResultObject(&contours, result, i, "contours");
            if (i)
                json << ",\n";
            json << "{\"model\":" << Json(name.S()) << ",\"contours\":[";
            bool first = true;
            for (const auto &line : contours.GetContours()) {
                if (!first)
                    json << ',';
                first = false;
                json << '[';
                for (size_t j = 0; j < line.size(); ++j) {
                    if (j)
                        json << ',';
                    json << '[' << line[j].row << ',' << line[j].column << ']';
                }
                json << ']';
            }
            json << "]}";
        }
        json << "\n]\n";
        auto d = GetSearchDiagnostics(result);
        timing << std::setprecision(10) << "{\"threads\":" << GetSearchThreadCount()
               << ",\"pyramid_ms\":" << d.pyramid_ms << ",\"top_level_ms\":" << d.top_level_ms
               << ",\"tracking_ms\":" << d.tracking_ms << ",\"refinement_ms\":" << d.refinement_ms
               << ",\"total_ms\":" << d.total_ms << ",\"evaluated_poses\":" << d.evaluated_poses
               << ",\"coarse_candidates\":" << d.coarse_candidates
               << ",\"refined_candidates\":" << d.refined_candidates << "}\n";
        ClearShapeModel(models);
        std::cout << "matches=" << count.I() << " total_ms=" << d.total_ms << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
