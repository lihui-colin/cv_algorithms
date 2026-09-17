#include "internal.hpp"
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <string_view>
#include <thread>

namespace shape_match::detail {
namespace {
using RangeTask = std::function<void(size_t, size_t, size_t)>;
thread_local bool executing = false;

size_t ConfiguredWorkers() {
    static const size_t count = [] {
        const unsigned available = std::max(1u, std::thread::hardware_concurrency());
        const char *value = std::getenv("SHAPE_MATCH_NUM_THREADS");
        if (!value)
            // Small-image stages submit many short jobs. A conservative default
            // avoids oversubscription; larger workloads can opt in explicitly.
            return size_t(std::min(4u, available));
        const std::string_view text(value);
        unsigned requested = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), requested);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !requested)
            return size_t(1);
        return size_t(std::min(requested, available));
    }();
    return count;
}

class Executor {
  public:
    Executor() {
        try {
            for (size_t worker = 1; worker < ConfiguredWorkers(); ++worker)
                threads_.emplace_back([this, worker] { Worker(worker); });
        } catch (...) {
            Stop();
            throw;
        }
    }
    ~Executor() { Stop(); }

    void Run(size_t count, const RangeTask &task) {
        // Serialize submissions, but let nested parallel loops run on their
        // calling worker so they cannot deadlock waiting for this same pool.
        std::unique_lock<std::mutex> serial(submission_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            count_ = count;
            active_ = SearchWorkerCount(count);
            remaining_ = active_ - 1;
            error_ = nullptr;
            ++generation_;
        }
        ready_.notify_all();
        Invoke(0);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
        task_ = {};
        if (error_)
            std::rethrow_exception(error_);
    }

  private:
    void Invoke(size_t worker) noexcept {
        executing = true;
        try {
            task_(count_ * worker / active_, count_ * (worker + 1) / active_, worker);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_)
                error_ = std::current_exception();
        }
        executing = false;
    }

    void Worker(size_t worker) {
        size_t observed = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || generation_ != observed; });
            if (stopping_)
                return;
            observed = generation_;
            if (worker >= active_)
                continue;
            lock.unlock();
            Invoke(worker);
            lock.lock();
            if (--remaining_ == 0)
                done_.notify_one();
        }
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto &thread : threads_)
            thread.join();
    }

    size_t count_ = 0, active_ = 1, remaining_ = 0, generation_ = 0;
    bool stopping_ = false;
    std::mutex mutex_, submission_;
    std::condition_variable ready_, done_;
    RangeTask task_;
    std::exception_ptr error_;
    std::vector<std::thread> threads_;
};
} // namespace

size_t SearchWorkerCount(size_t work_items) {
    return std::min(ConfiguredWorkers(), std::max<size_t>(1, work_items));
}

void ParallelFor(size_t work_items, const RangeTask &function) {
    if (!work_items)
        return;
    if (executing || SearchWorkerCount(work_items) == 1) {
        function(0, work_items, 0);
        return;
    }
    static Executor executor;
    executor.Run(work_items, function);
}
} // namespace shape_match::detail
