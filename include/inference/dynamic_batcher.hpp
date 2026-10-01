#pragma once

#include <chrono>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "inference/bounded_queue.hpp"
#include "inference/metrics.hpp"
#include "inference/tensor.hpp"

namespace inference {

struct BatchingOptions {
    std::size_t queue_capacity = 64;
    std::size_t max_batch_size = 8;
    std::chrono::milliseconds max_batch_delay{2};
};

// Requests must have matching input tensor layouts; dimension zero is combined into one batch.
class DynamicBatcher {
   public:
    using RequestResult = std::vector<Tensor>;
    using BatchInference = std::function<RequestResult(const std::vector<Tensor>&)>;
    using ResultFuture = std::future<RequestResult>;

    // metrics is non-owning and must outlive this batcher.
    DynamicBatcher(BatchingOptions options, BatchInference infer_batch, Metrics* metrics = nullptr);
    ~DynamicBatcher();

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    [[nodiscard]] std::optional<ResultFuture> try_submit(std::vector<Tensor> inputs);
    void close();

    [[nodiscard]] std::size_t queue_depth() const { return queue_.size(); }
    [[nodiscard]] std::size_t queue_capacity() const noexcept { return queue_.capacity(); }
    [[nodiscard]] const BatchingOptions& options() const noexcept { return options_; }

   private:
    struct Request {
        std::vector<Tensor> inputs;
        std::promise<RequestResult> result;
        std::chrono::steady_clock::time_point accepted_at;
    };

    void run();
    void execute(std::vector<Request>& requests);
    [[nodiscard]] static std::vector<Tensor> combine_inputs(const std::vector<Request>& requests);
    [[nodiscard]] static std::vector<RequestResult> split_outputs(
        const std::vector<Tensor>& outputs, std::size_t batch_size);

    BatchingOptions options_;
    BoundedQueue<Request> queue_;
    BatchInference infer_batch_;
    Metrics* metrics_;
    std::jthread worker_;
    std::once_flag close_once_;
};

}  // namespace inference
