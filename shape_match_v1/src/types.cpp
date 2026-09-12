#include "internal.hpp"
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

namespace shape_match {
HTuple::HTuple(std::initializer_list<HTuple> values) {
    for (const auto &x : values)
        Append(x);
}
const HValue &HTuple::At(size_t i) const {
    detail::Require(i < values_.size(), ErrorCode::Value, "Tuple index out of range");
    return values_[i];
}
double HTuple::D(size_t i) const {
    return detail::Number(At(i));
}
int64_t HTuple::I(size_t i) const {
    auto p = std::get_if<int64_t>(&At(i));
    detail::Require(p, ErrorCode::Type, "Expected integer");
    return *p;
}
const std::string &HTuple::S(size_t i) const {
    auto p = std::get_if<std::string>(&At(i));
    detail::Require(p, ErrorCode::Type, "Expected string");
    return *p;
}
HHandle HTuple::H(size_t i) const {
    auto p = std::get_if<HHandle>(&At(i));
    detail::Require(p, ErrorCode::Type, "Expected typed handle, not an integer");
    return *p;
}
HTuple &HTuple::Append(const HTuple &other) {
    auto copy = other.values_;
    values_.insert(values_.end(), copy.begin(), copy.end());
    return *this;
}
HTuple &HTuple::AppendValue(HValue v) {
    values_.push_back(std::move(v));
    return *this;
}
HObject HObject::FromGray(int width, int height, const std::vector<float> &pixels,
                          const std::vector<uint8_t> &mask) {
    detail::Require(width > 0 && height > 0, ErrorCode::Image, "Image dimensions must be positive");
    size_t size = size_t(width) * size_t(height);
    detail::Require(pixels.size() == size && (mask.empty() || mask.size() == size),
                    ErrorCode::Image, "Image/mask size mismatch");
    for (float v : pixels)
        detail::Require(std::isfinite(v) && v >= 0 && v <= 255, ErrorCode::Image,
                        "Gray values must be finite in [0,255]");
    auto image = std::make_shared<Image>();
    image->width = width;
    image->height = height;
    image->pixels = pixels;
    image->domain = mask.empty() ? std::vector<uint8_t>(size, 1) : mask;
    for (auto &v : image->domain)
        v = v ? 1 : 0;
    HObject object;
    object.kind_ = Kind::Images;
    object.images_.push_back(std::move(image));
    return object;
}
HObject HObject::FromBytes(int w, int h, const std::vector<uint8_t> &p,
                           const std::vector<uint8_t> &d) {
    return FromGray(w, h, std::vector<float>(p.begin(), p.end()), d);
}
HObject HObject::FromContours(ContourSet c) {
    HObject o;
    o.kind_ = Kind::Contours;
    o.contours_ = std::move(c);
    return o;
}
const Image &HObject::GetImage(size_t i) const {
    detail::Require(kind_ == Kind::Images && i < images_.size(), ErrorCode::Type,
                    "Expected image object");
    return *images_[i];
}
const ContourSet &HObject::GetContours() const {
    detail::Require(kind_ == Kind::Contours, ErrorCode::Type, "Expected contour object");
    return contours_;
}
size_t HObject::Count() const noexcept {
    return kind_ == Kind::Images ? images_.size() : contours_.size();
}
// PGM parser intentionally handles comments and CRLF, but does not guess file formats.
static std::string Token(std::istream &in) {
    std::string token;
    char c;
    while (in.get(c)) {
        if (c == '#') {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(c))) {
            token += c;
            break;
        }
    }
    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (c == '\r' && in.peek() == '\n')
                in.get();
            break;
        }
        token += c;
    }
    detail::Require(!token.empty(), ErrorCode::Image, "Truncated PGM header");
    return token;
}
HObject ReadPgm(const std::string &path, const std::string &mask_path) {
    std::ifstream in(path, std::ios::binary);
    detail::Require(bool(in), ErrorCode::Image, "Cannot open " + path);
    detail::Require(Token(in) == "P5", ErrorCode::Image, "Only binary P5 PGM is supported");
    int w = std::stoi(Token(in)), h = std::stoi(Token(in)), max = std::stoi(Token(in));
    detail::Require(w > 0 && h > 0 && max == 255 && size_t(w) * h < 1000000000, ErrorCode::Image,
                    "Invalid PGM dimensions or range");
    std::vector<uint8_t> pixels(size_t(w) * h);
    in.read(reinterpret_cast<char *>(pixels.data()), std::streamsize(pixels.size()));
    detail::Require(in.gcount() == std::streamsize(pixels.size()), ErrorCode::Image,
                    "Truncated PGM pixels");
    std::vector<uint8_t> mask;
    if (!mask_path.empty()) {
        auto m = ReadPgm(mask_path);
        const auto &im = m.GetImage();
        detail::Require(im.width == w && im.height == h, ErrorCode::Image,
                        "Mask dimensions differ");
        for (float v : im.pixels)
            mask.push_back(v > 0);
    }
    return HObject::FromBytes(w, h, pixels, mask);
}
void WritePgm(const HObject &object, const std::string &path) {
    const auto &image = object.GetImage();
    std::ofstream out(path, std::ios::binary);
    detail::Require(bool(out), ErrorCode::Image, "Cannot write " + path);
    out << "P5\n" << image.width << ' ' << image.height << "\n255\n";
    for (float v : image.pixels)
        out.put(static_cast<char>(std::lround(v)));
    detail::Require(bool(out), ErrorCode::Image, "PGM write failed");
}
} // namespace shape_match

namespace shape_match::detail {
double Number(const HValue &v) {
    if (auto p = std::get_if<double>(&v))
        return *p;
    if (auto p = std::get_if<int64_t>(&v))
        return double(*p);
    Fail(ErrorCode::Type, "Expected numeric parameter");
}
std::string String(const HValue &v) {
    if (auto p = std::get_if<std::string>(&v))
        return *p;
    Fail(ErrorCode::Type, "Expected string parameter");
}
bool IsAuto(const HValue &v) {
    auto p = std::get_if<std::string>(&v);
    return p && *p == "auto";
}
double Num(const Params &p, const std::string &k) {
    return Number(p.at(k));
}
std::string Str(const Params &p, const std::string &k) {
    return String(p.at(k));
}
bool Bool(const Params &p, const std::string &k) {
    return Str(p, k) == "true";
}
std::shared_ptr<ShapeModel> ResolveModel(const HTuple &tuple, size_t i) {
    auto h = tuple.H(i);
    Require(h.state && h.state->kind == HandleBase::Kind::Model, ErrorCode::Handle,
            "Not a model handle");
    return std::static_pointer_cast<ShapeModel>(h.state);
}
std::shared_ptr<MatchResult> ResolveResult(const HTuple &tuple) {
    Require(tuple.Length() == 1, ErrorCode::Value, "Expected one result handle");
    auto h = tuple.H();
    Require(h.state && h.state->kind == HandleBase::Kind::Result, ErrorCode::Handle,
            "Not a result handle");
    return std::static_pointer_cast<MatchResult>(h.state);
}
std::vector<ModelSnapshot> SnapshotModels(const HTuple &tuple) {
    Require(!tuple.Empty(), ErrorCode::Value, "No models supplied");
    std::vector<ModelSnapshot> out;
    std::set<std::string> identifiers;
    for (size_t i = 0; i < tuple.Length(); ++i) {
        auto m = ResolveModel(tuple, i);
        std::lock_guard<std::mutex> guard(m->mutex);
        Require(m->alive, ErrorCode::Handle, "Model was cleared");
        Require(!m->needs_training && bool(m->trained), ErrorCode::Untrained,
                "Train model before find");
        Require(identifiers.insert(Str(m->params, "model_identifier")).second, ErrorCode::Value,
                "Duplicate model identifiers in search");
        Params p = m->trained->effective;
        for (const auto &kv : m->params) {
            if (!NeedsTraining(kv.first) && !(kv.first == "min_contrast" && IsAuto(kv.second)))
                p[kv.first] = kv.second;
        }
        out.push_back({m->id, std::move(p), m->trained, tuple.H(i)});
    }
    return out;
}
} // namespace shape_match::detail
