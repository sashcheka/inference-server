#include "inference/dynamic_batcher.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inference {

DynamicBatcher::DynamicBatcher(BatchingOptions options, BatchInference infer_batch,
                               Metrics* metrics)
    : options_(options),
      queue_(options.queue_capacity),
      infer_batch_(std::move(infer_batch)),
      metrics_(metrics) {
    constexpr std::size_t max_supported_batch_size = 1024;
    constexpr auto max_supported_delay = std::chrono::milliseconds{60000};
    if (options_.max_batch_size == 0 || options_.max_batch_size > max_supported_batch_size) {
        throw std::invalid_argument("max_batch_size must be between 1 and 1024");
    }
    if (options_.max_batch_delay < std::chrono::milliseconds::zero() ||
        options_.max_batch_delay > max_supported_delay) {
        throw std::invalid_argument("max_batch_delay must be between 0 and 60000 milliseconds");
    }
    if (!infer_batch_) {
        throw std::invalid_argument("dynamic batcher requires an inference function");
    }

    worker_ = std::jthread([this] { run(); });
}

DynamicBatcher::~DynamicBatcher() { close(); }

std::optional<DynamicBatcher::ResultFuture> DynamicBatcher::try_submit(std::vector<Tensor> inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("a batched request must contain at least one input tensor");
    }
    for (const auto& input : inputs) {
        if (input.shape().empty() || input.shape().front() != 1) {
            throw std::invalid_argument("each request tensor must have a batch dimension of 1");
        }
    }

    Request request{
        .inputs = std::move(inputs), .result = {}, .accepted_at = std::chrono::steady_clock::now()};
    auto future = request.result.get_future();
    if (!queue_.try_push(std::move(request))) {
        return std::nullopt;
    }
    return future;
}

void DynamicBatcher::close() {
    std::call_once(close_once_, [this] {
        queue_.close();
        if (worker_.joinable()) {
            worker_.join();
        }
    });
}

void DynamicBatcher::run() {
    while (auto first = queue_.pop()) {
        const auto deadline = first->accepted_at + options_.max_batch_delay;
        std::vector<Request> requests;
        requests.push_back(std::move(*first));

        while (requests.size() < options_.max_batch_size) {
            auto next = queue_.pop_until(deadline);
            if (!next) {
                break;
            }
            requests.push_back(std::move(*next));
        }
        execute(requests);
    }
}

std::vector<Tensor> DynamicBatcher::combine_inputs(const std::vector<Request>& requests) {
    if (requests.empty()) {
        throw std::logic_error("cannot combine an empty request batch");
    }

    const auto input_count = requests.front().inputs.size();
    std::vector<Tensor> batched_inputs;
    batched_inputs.reserve(input_count);

    for (std::size_t input_index = 0; input_index < input_count; ++input_index) {
        const auto& first = requests.front().inputs[input_index];
        auto shape = first.shape();
        std::size_t total_bytes = 0;
        std::vector<std::byte> data;

        for (const auto& request : requests) {
            if (request.inputs.size() != input_count) {
                throw std::invalid_argument("requests in a batch have different input counts");
            }
            const auto& input = request.inputs[input_index];
            if (input.element_type() != first.element_type() || input.shape() != first.shape()) {
                throw std::invalid_argument(
                    "requests in a batch have incompatible tensor metadata");
            }
            if (input.byte_size() > data.max_size() - total_bytes) {
                throw std::overflow_error("batched input tensor is too large");
            }
            total_bytes += input.byte_size();
        }

        shape.front() = static_cast<std::int64_t>(requests.size());
        data.reserve(total_bytes);
        for (const auto& request : requests) {
            const auto& input = request.inputs[input_index];
            data.insert(data.end(), input.data().begin(), input.data().end());
        }
        batched_inputs.emplace_back(first.element_type(), std::move(shape), std::move(data));
    }
    return batched_inputs;
}

std::vector<DynamicBatcher::RequestResult> DynamicBatcher::split_outputs(
    const std::vector<Tensor>& outputs, std::size_t batch_size) {
    if (outputs.empty()) {
        throw std::runtime_error("model returned no output tensors");
    }

    std::vector<RequestResult> results(batch_size);
    for (const auto& output : outputs) {
        if (output.shape().empty() ||
            output.shape().front() != static_cast<std::int64_t>(batch_size) ||
            output.byte_size() % batch_size != 0) {
            throw std::runtime_error("model output does not match the executed batch size");
        }

        auto sample_shape = output.shape();
        sample_shape.front() = 1;
        const auto bytes_per_request = output.byte_size() / batch_size;
        for (std::size_t request_index = 0; request_index < batch_size; ++request_index) {
            std::vector<std::byte> sample_data(bytes_per_request);
            const auto byte_offset = request_index * bytes_per_request;
            if (bytes_per_request != 0) {
                std::memcpy(sample_data.data(), output.data().data() + byte_offset,
                            bytes_per_request);
            }
            results[request_index].emplace_back(output.element_type(), sample_shape,
                                                std::move(sample_data));
        }
    }
    return results;
}

void DynamicBatcher::execute(std::vector<Request>& requests) {
    std::vector<RequestResult> results;
    try {
        auto batched_inputs = combine_inputs(requests);
        auto queue_wait_total = std::chrono::steady_clock::duration::zero();
        for (const auto& request : requests) {
            queue_wait_total += std::chrono::steady_clock::now() - request.accepted_at;
        }

        std::vector<Tensor> batched_outputs;
        const auto inference_start = std::chrono::steady_clock::now();
        try {
            batched_outputs = infer_batch_(batched_inputs);
        } catch (...) {
            if (metrics_ != nullptr) {
                metrics_->record_batch(requests.size(), queue_wait_total,
                                       std::chrono::steady_clock::now() - inference_start);
            }
            throw;
        }
        if (metrics_ != nullptr) {
            metrics_->record_batch(requests.size(), queue_wait_total,
                                   std::chrono::steady_clock::now() - inference_start);
        }
        results = split_outputs(batched_outputs, requests.size());
    } catch (...) {
        const auto error = std::current_exception();
        for (auto& request : requests) {
            request.result.set_exception(error);
        }
        return;
    }

    for (std::size_t index = 0; index < requests.size(); ++index) {
        requests[index].result.set_value(std::move(results[index]));
    }
}

}  // namespace inference
