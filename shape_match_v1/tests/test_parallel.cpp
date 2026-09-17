#include "../src/internal.hpp"
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <thread>

using namespace shape_match::detail;

static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void CheckCoverage(size_t count) {
    std::vector<int> visits(count);
    std::vector<size_t> sizes(SearchWorkerCount(count));
    TrackingTrace trace;
    ParallelFor(count, [&](size_t begin, size_t end, size_t worker) {
        Check(begin <= end && end <= count && worker < sizes.size(), "Invalid worker range");
        sizes[worker] = end - begin;
        for (size_t i = begin; i < end; ++i)
            ++visits[i];
    }, &trace);
    Check(std::all_of(visits.begin(), visits.end(), [](int n) { return n == 1; }),
          "Work must be visited exactly once");
    Check(std::accumulate(sizes.begin(), sizes.end(), size_t(0)) == count, "Missing work");
#ifdef SHAPE_MATCH_TRACKING_DIAGNOSTICS
    if (count)
        for (const auto &w : trace.workers) {
            Check(w.start_ms >= trace.dispatch_ms && w.end_ms >= w.start_ms &&
                  trace.completed_ms >= w.end_ms, "Invalid trace timestamps");
#ifdef __linux__
            Check(w.cpu_ms >= 0, "Missing thread CPU timing");
#endif
        }
#endif
}

static void CheckFilters() {
    for (int width : {1, 2, 3, 5, 6, 7, 8, 17, 64, 127})
        for (int height : {1, 3, 18, 35})
            for (double sigma : {.5, .8, 1., 1.4, 3.}) {
                shape_match::Image im;
                im.width = width; im.height = height;
                for (int i = 0; i < width * height; ++i) {
                    im.pixels.push_back(float((i * 73 + 19) % 256));
                    im.domain.push_back(i % 5 != 0);
                }
                const int radius = std::max(1, int(std::ceil(3 * sigma)));
                std::vector<double> kernel(size_t(2 * radius + 1));
                double sum = 0;
                for (int k = -radius; k <= radius; ++k)
                    sum += (kernel[k + radius] = std::exp(-.5 * Sq(k / sigma)));
                for (auto &v : kernel) v /= sum;
                auto tmp = im, expected = im;
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        double v = 0;
                        for (int k = -radius; k <= radius; ++k)
                            v += kernel[k + radius] * im(y, std::clamp(x+k, 0, width-1));
                        tmp.pixels[size_t(y)*width+x] = float(v);
                    }
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        double v = 0;
                        for (int k = -radius; k <= radius; ++k)
                            v += kernel[k + radius] * tmp(std::clamp(y+k, 0, height-1), x);
                        expected.pixels[size_t(y)*width+x] = float(v);
                    }
                const auto actual = Smooth(im, sigma);
                Check(actual.pixels == expected.pixels && actual.domain == im.domain,
                      "Optimized smoothing differs from scalar reference");
                if (sigma == 1.) {
                    const auto down = Downsample(im);
                    for (int y = 0; y < down.height; ++y)
                        for (int x = 0; x < down.width; ++x)
                            Check(down(y,x) == expected(2*y,2*x) &&
                                  down.domain[size_t(y)*down.width+x] == im.domain[size_t(2*y)*width+2*x],
                                  "Fused downsampling differs from filter then decimate");
                }
            }
}

static void CheckScorePixels() {
    for (int extent : {1, 2, 17, 640, 4096}) {
        auto check = [&](double x) {
            const long rounded = std::lround(x);
            const int expected = rounded >= 0 && rounded < extent ? int(rounded) : -1;
            Check(ScorePixel(x, extent) == expected, "Score pixel differs from lround");
        };
        for (int i = -2; i <= extent + 2; ++i) {
            check(double(i));
            const double tie = i + .5;
            check(tie);
            check(std::nextafter(tie, -std::numeric_limits<double>::infinity()));
            check(std::nextafter(tie, std::numeric_limits<double>::infinity()));
        }
        for (int i = -10000; i < 10000; ++i)
            check(i * .071239);
        Check(ScorePixel(std::numeric_limits<double>::infinity(), extent) == -1,
              "Infinite pixel coordinate accepted");
        Check(ScorePixel(std::numeric_limits<double>::quiet_NaN(), extent) == -1,
              "NaN pixel coordinate accepted");
    }
}

int main() {
    try {
        CheckFilters();
        CheckScorePixels();
        ParallelFor(0, [](size_t, size_t, size_t) { throw std::runtime_error("Empty loop ran"); });
        for (size_t count : {1, 2, 3, 7, 33, 257})
            CheckCoverage(count);

        // Exercise exceptions on both the submitting thread and a pool worker.
        for (size_t failing = 0; failing < std::min<size_t>(2, SearchWorkerCount(257)); ++failing) {
            bool caught = false;
            try {
                ParallelFor(257, [=](size_t, size_t, size_t worker) {
                    if (worker == failing)
                        throw std::runtime_error("worker failure");
                });
            } catch (const std::runtime_error &error) {
                caught = std::string(error.what()) == "worker failure";
            }
            Check(caught, "Worker exception must reach caller");
            CheckCoverage(257);
        }

        std::atomic<size_t> nested{0};
        ParallelFor(17, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i)
                ParallelFor(11, [&](size_t first, size_t last, size_t) { nested += last - first; });
        });
        Check(nested == 17 * 11, "Nested work lost");

        std::vector<std::future<void>> callers;
        for (int caller = 0; caller < 4; ++caller)
            callers.push_back(std::async(std::launch::async, [] {
                for (int run = 0; run < 20; ++run)
                    CheckCoverage(257);
            }));
        for (auto &caller : callers)
            caller.get();

        // Runtime environment changes must not resize scratch arrays relative
        // to the already-created pool.
        const size_t workers = SearchWorkerCount(257);
#if defined(_WIN32)
        _putenv_s("SHAPE_MATCH_NUM_THREADS", "1");
#else
        setenv("SHAPE_MATCH_NUM_THREADS", "1", 1);
#endif
        Check(SearchWorkerCount(257) == workers, "Worker configuration changed after startup");
        CheckCoverage(257);
        std::cout << "PASS parallel ranges, exceptions, nesting and concurrent callers\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
