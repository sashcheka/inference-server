#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace inference {

struct MetricsSnapshot {
    std::uint64_t requests_total = 0;
    std::uint64_t requests_failed = 0;
    std::uint64_t requests_rejected = 0;
    std::uint64_t batches_total = 0;
    std::size_t queue_depth = 0;
    double average_batch_size = 0.0;
    double average_queue_wait_ms = 0.0;
    double average_inference_ms = 0.0;
    double average_end_to_end_ms = 0.0;
};

class Metrics {
   public:
    void record_request();
    void record_failure();
    void record_rejection();
    void record_batch(std::size_t batch_size, std::chrono::steady_clock::duration queue_wait_total,
                      std::chrono::steady_clock::duration inference_duration);
    void record_end_to_end(std::chrono::steady_clock::duration duration);

    [[nodiscard]] MetricsSnapshot snapshot(std::size_t queue_depth) const;

   private:
    mutable std::mutex mutex_;
    MetricsSnapshot totals_;
    std::uint64_t queue_wait_samples_ = 0;
    std::uint64_t end_to_end_samples_ = 0;
    double batch_size_sum_ = 0.0;
    double queue_wait_ms_sum_ = 0.0;
    double inference_ms_sum_ = 0.0;
    double end_to_end_ms_sum_ = 0.0;
};

}  // namespace inference
