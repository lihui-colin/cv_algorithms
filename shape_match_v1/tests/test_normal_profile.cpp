#include "contour_residual_probe.hpp"
#include "orientation_probe.hpp"
#include <iostream>

int main() {
    using namespace profile_probe;
    try {
        GradientField g;
        g.width = g.height = 40;
        g.gx.resize(1600);
        g.gy.resize(1600);
        g.mag.resize(1600);
        for (int y = 0; y < 40; ++y)
            for (int x = 0; x < 40; ++x)
                g.gx[size_t(y) * 40 + x] =
                    float(6 * std::exp(-2 * Sq(x - 16.0)) + 12 * std::exp(-2 * Sq(x - 19.0)));
        Features f{{{0, 0}, {1, 0}, 10, 1, 0}};
        auto near = Pairs(f, g, {16.2, 20, 0, 1}, {4, true, true});
        auto strong = Pairs(f, g, {16.2, 20, 0, 1}, {4, false, false});
        auto check = [](bool value) {
            if (!value)
                throw std::runtime_error("Profile test failed");
        };
        check(near.size() == 1 && strong.size() == 1);
        check(near[0].image.x < 17 && strong[0].image.x > 18);
        check(std::abs(near[0].image.y - 20) < 1e-12);
        check(std::abs(Dot(strong[0].normal, strong[0].normal) - 1) < 1e-12);
        check(Pairs(f, g, {0.5, 20, 0, 1}, {4, true, true}).empty());
        f[0].n = {-1, 0};
        check(Pairs(f, g, {16.2, 20, 0, 1}, {4, true, true}).empty());
        std::fill(g.gx.begin(), g.gx.end(), 0);
        check(Pairs(f, g, {16.2, 20, 0, 1}, {4, true, true}).empty());
        Features components{{{0, 0}, {1, 0}, 1, 1, 0},
                            {{1, 0}, {1, 0}, 1, 1, 0},
                            {{10, 0}, {1, 0}, 1, 1, 0},
                            {{11, 0}, {1, 0}, 1, 1, 0}};
        auto labels = contour_residual_probe::Components(components);
        check(labels[0] == labels[1] && labels[2] == labels[3] && labels[0] != labels[2]);
        std::fill(g.gx.begin(), g.gx.end(), 10);
        f[0].n = {1, 0};
        check(std::abs(orientation_probe::Score(f, g, {20, 20, 0, 1}) - 1) < 1e-12);
        f[0].n = {-1, 0};
        check(std::abs(orientation_probe::Score(f, g, {20, 20, 0, 1}) + 1) < 1e-12);
        check(orientation_probe::Score(f, g, {-2, 20, 0, 1}) == 0);
        f.push_back({{100, 0}, {1, 0}, 1, 1, 0});
        check(std::abs(orientation_probe::Score(f, g, {20, 20, 0, 1}) + 0.5) < 1e-12);
        std::cout << "PASS 12 profile/component/orientation checks\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
