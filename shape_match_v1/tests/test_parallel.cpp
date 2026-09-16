#include "../src/internal.hpp"
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>

using namespace shape_match::detail;

static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void CheckCoverage(size_t count) {
    std::vector<int> visits(count);
    std::vector<size_t> sizes(SearchWorkerCount(count));
    ParallelFor(count, [&](size_t begin, size_t end, size_t worker) {
        Check(begin <= end && end <= count && worker < sizes.size(), "Invalid worker range");
        sizes[worker] = end - begin;
        for (size_t i = begin; i < end; ++i)
            ++visits[i];
    });
    Check(std::all_of(visits.begin(), visits.end(), [](int n) { return n == 1; }),
          "Work must be visited exactly once");
    Check(std::accumulate(sizes.begin(), sizes.end(), size_t(0)) == count, "Missing work");
}

int main() {
    try {
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
