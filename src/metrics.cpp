#include "inference/metrics.hpp"

#include <chrono>

namespace inference {
namespace {

using Milliseconds = std::chrono::duration<double, std::milli>;

[[nodiscard]] double to_milliseconds(std::chrono::steady_clock::duration duration) {
    return Milliseconds(duration).count();
}

}  // namespace

void Metrics::record_request() {
    const std::lock_guard lock(mutex_);
    ++totals_.requests_total;
}

void Metrics::record_failure() {
    const std::lock_guard lock(mutex_);
    ++totals_.requests_failed;
}

void Metrics::record_rejection() {
    const std::lock_guard lock(mutex_);
    ++totals_.requests_rejected;
}

void Metrics::record_batch(std::size_t batch_size,
                           std::chrono::steady_clock::duration queue_wait_total,
                           std::chrono::steady_clock::duration inference_duration) {
    const std::lock_guard lock(mutex_);
    ++totals_.batches_total;
    batch_size_sum_ += static_cast<double>(batch_size);
    queue_wait_samples_ += static_cast<std::uint64_t>(batch_size);
    queue_wait_ms_sum_ += to_milliseconds(queue_wait_total);
    inference_ms_sum_ += to_milliseconds(inference_duration);
}

void Metrics::record_end_to_end(std::chrono::steady_clock::duration duration) {
    const std::lock_guard lock(mutex_);
    ++end_to_end_samples_;
    end_to_end_ms_sum_ += to_milliseconds(duration);
}

MetricsSnapshot Metrics::snapshot(std::size_t queue_depth) const {
    const std::lock_guard lock(mutex_);
    auto result = totals_;
    result.queue_depth = queue_depth;
    result.queue_wait_samples = queue_wait_samples_;
    result.end_to_end_samples = end_to_end_samples_;
    if (result.batches_total > 0) {
        const auto count = static_cast<double>(result.batches_total);
        result.average_batch_size = batch_size_sum_ / count;
        result.average_inference_ms = inference_ms_sum_ / count;
    }
    if (queue_wait_samples_ > 0) {
        result.average_queue_wait_ms =
            queue_wait_ms_sum_ / static_cast<double>(queue_wait_samples_);
    }
    if (end_to_end_samples_ > 0) {
        result.average_end_to_end_ms =
            end_to_end_ms_sum_ / static_cast<double>(end_to_end_samples_);
    }
    return result;
}

}  // namespace inference
