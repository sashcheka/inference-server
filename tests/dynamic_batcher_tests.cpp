#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "inference/dynamic_batcher.hpp"

namespace {

[[nodiscard]] inference::Tensor scalar_input(float value) {
    return inference::Tensor::from_float32({1, 1}, {value});
}

[[nodiscard]] float scalar_output(const inference::DynamicBatcher::RequestResult& result) {
    return result.front().float32_values().front();
}

TEST(DynamicBatcherTest, ExecutesSingleRequestAfterBatchDelay) {
    std::vector<std::int64_t> observed_batch_sizes;
    inference::DynamicBatcher batcher(
        {.queue_capacity = 4,
         .max_batch_size = 8,
         .max_batch_delay = std::chrono::milliseconds{10}},
        [&observed_batch_sizes](const std::vector<inference::Tensor>& inputs) {
            observed_batch_sizes.push_back(inputs.front().shape().front());
            return inputs;
        });

    auto future = batcher.try_submit({scalar_input(2.0F)});
    ASSERT_TRUE(future.has_value());
    ASSERT_EQ(future->wait_for(std::chrono::seconds{1}), std::future_status::ready);
    EXPECT_FLOAT_EQ(scalar_output(future->get()), 2.0F);
    EXPECT_EQ(observed_batch_sizes, (std::vector<std::int64_t>{1}));
}

TEST(DynamicBatcherTest, ExecutesMaxSizeBatchAndPreservesRequestMapping) {
    std::vector<std::int64_t> observed_batch_sizes;
    inference::DynamicBatcher batcher(
        {.queue_capacity = 8, .max_batch_size = 3, .max_batch_delay = std::chrono::seconds{5}},
        [&observed_batch_sizes](const std::vector<inference::Tensor>& inputs) {
            observed_batch_sizes.push_back(inputs.front().shape().front());
            auto values = inputs.front().float32_values();
            for (auto& value : values) {
                value += 100.0F;
            }
            return std::vector<inference::Tensor>{
                inference::Tensor::from_float32(inputs.front().shape(), values)};
        });

    auto first = batcher.try_submit({scalar_input(1.0F)});
    auto second = batcher.try_submit({scalar_input(2.0F)});
    auto third = batcher.try_submit({scalar_input(3.0F)});
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(third.has_value());

    EXPECT_FLOAT_EQ(scalar_output(first->get()), 101.0F);
    EXPECT_FLOAT_EQ(scalar_output(second->get()), 102.0F);
    EXPECT_FLOAT_EQ(scalar_output(third->get()), 103.0F);
    EXPECT_EQ(observed_batch_sizes, (std::vector<std::int64_t>{3}));
}

TEST(DynamicBatcherTest, RejectsRequestWhenBoundedQueueIsFull) {
    std::promise<void> inference_started;
    auto started = inference_started.get_future();
    std::promise<void> release_inference;
    const auto release = release_inference.get_future().share();
    std::atomic<unsigned int> calls{0};
    inference::DynamicBatcher batcher(
        {.queue_capacity = 1, .max_batch_size = 1, .max_batch_delay = std::chrono::milliseconds{0}},
        [&](const std::vector<inference::Tensor>& inputs) {
            if (calls.fetch_add(1, std::memory_order_relaxed) == 0) {
                inference_started.set_value();
            }
            release.wait();
            return inputs;
        });

    auto first = batcher.try_submit({scalar_input(1.0F)});
    ASSERT_TRUE(first.has_value());
    const auto entered = started.wait_for(std::chrono::seconds{5}) == std::future_status::ready;
    if (!entered) {
        release_inference.set_value();
        FAIL() << "inference worker did not start";
        return;
    }

    auto second = batcher.try_submit({scalar_input(2.0F)});
    auto rejected = batcher.try_submit({scalar_input(3.0F)});
    release_inference.set_value();

    EXPECT_TRUE(second.has_value());
    EXPECT_FALSE(rejected.has_value());
    EXPECT_FLOAT_EQ(scalar_output(first->get()), 1.0F);
    if (second) {
        EXPECT_FLOAT_EQ(scalar_output(second->get()), 2.0F);
    }
}

TEST(DynamicBatcherTest, PropagatesInferenceErrorsToEveryRequest) {
    inference::DynamicBatcher batcher(
        {.queue_capacity = 4, .max_batch_size = 2, .max_batch_delay = std::chrono::milliseconds{0}},
        [](const std::vector<inference::Tensor>&) -> std::vector<inference::Tensor> {
            throw std::runtime_error("test inference error");
        });

    auto future = batcher.try_submit({scalar_input(1.0F)});
    ASSERT_TRUE(future.has_value());
    EXPECT_THROW(static_cast<void>(future->get()), std::runtime_error);
}

TEST(DynamicBatcherTest, RecordsBatchAndQueueMeasurements) {
    inference::Metrics metrics;
    inference::DynamicBatcher batcher(
        {.queue_capacity = 2, .max_batch_size = 1, .max_batch_delay = std::chrono::milliseconds{0}},
        [](const std::vector<inference::Tensor>& inputs) { return inputs; }, &metrics);

    auto future = batcher.try_submit({scalar_input(1.0F)});
    ASSERT_TRUE(future.has_value());
    static_cast<void>(future->get());

    const auto snapshot = metrics.snapshot(batcher.queue_depth());
    EXPECT_EQ(snapshot.batches_total, 1);
    EXPECT_DOUBLE_EQ(snapshot.average_batch_size, 1.0);
    EXPECT_GE(snapshot.average_queue_wait_ms, 0.0);
    EXPECT_GE(snapshot.average_inference_ms, 0.0);
}

TEST(DynamicBatcherTest, CloseDrainsAcceptedRequestsBeforeReturning) {
    std::vector<std::int64_t> observed_batch_sizes;
    inference::DynamicBatcher batcher(
        {.queue_capacity = 4, .max_batch_size = 4, .max_batch_delay = std::chrono::seconds{30}},
        [&observed_batch_sizes](const std::vector<inference::Tensor>& inputs) {
            observed_batch_sizes.push_back(inputs.front().shape().front());
            return inputs;
        });

    auto first = batcher.try_submit({scalar_input(1.0F)});
    auto second = batcher.try_submit({scalar_input(2.0F)});
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    batcher.close();

    EXPECT_FLOAT_EQ(scalar_output(first->get()), 1.0F);
    EXPECT_FLOAT_EQ(scalar_output(second->get()), 2.0F);
    EXPECT_EQ(observed_batch_sizes, (std::vector<std::int64_t>{2}));
    EXPECT_FALSE(batcher.try_submit({scalar_input(3.0F)}).has_value());
}

TEST(DynamicBatcherTest, RejectsTensorsWithoutSingleItemBatchDimension) {
    inference::DynamicBatcher batcher(
        {.queue_capacity = 2, .max_batch_size = 2, .max_batch_delay = std::chrono::milliseconds{0}},
        [](const std::vector<inference::Tensor>& inputs) { return inputs; });

    EXPECT_THROW(static_cast<void>(
                     batcher.try_submit({inference::Tensor::from_float32({2, 1}, {1.0F, 2.0F})})),
                 std::invalid_argument);
}

}  // namespace
