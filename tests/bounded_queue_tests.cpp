#include "inference/bounded_queue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

namespace {

TEST(BoundedQueueTest, RejectsZeroCapacity) {
    EXPECT_THROW((inference::BoundedQueue<int>(0)), std::invalid_argument);
}

TEST(BoundedQueueTest, EnforcesCapacityAndPreservesFifoOrder) {
    inference::BoundedQueue<int> queue(2);

    EXPECT_EQ(queue.capacity(), 2);
    EXPECT_TRUE(queue.try_push(10));
    EXPECT_TRUE(queue.try_push(20));
    EXPECT_FALSE(queue.try_push(30));
    EXPECT_EQ(queue.size(), 2);
    EXPECT_EQ(queue.pop(), 10);
    EXPECT_EQ(queue.pop(), 20);
    EXPECT_FALSE(queue.try_pop().has_value());
}

TEST(BoundedQueueTest, CloseDrainsAcceptedItemsAndRejectsNewItems) {
    inference::BoundedQueue<int> queue(2);
    ASSERT_TRUE(queue.try_push(42));

    queue.close();

    EXPECT_TRUE(queue.closed());
    EXPECT_FALSE(queue.try_push(99));
    EXPECT_EQ(queue.pop(), 42);
    EXPECT_FALSE(queue.pop().has_value());
}

TEST(BoundedQueueTest, CloseWakesWaitingConsumer) {
    inference::BoundedQueue<int> queue(1);
    std::promise<void> consumer_started;
    auto started = consumer_started.get_future();
    std::atomic<bool> returned_empty{false};

    std::thread consumer([&] {
        consumer_started.set_value();
        returned_empty.store(!queue.pop().has_value(), std::memory_order_release);
    });

    started.wait();
    queue.close();
    consumer.join();

    EXPECT_TRUE(returned_empty.load(std::memory_order_acquire));
}

TEST(BoundedQueueTest, SupportsConcurrentProducers) {
    constexpr std::size_t producer_count = 4;
    constexpr std::size_t values_per_producer = 100;
    constexpr std::size_t total_values = producer_count * values_per_producer;
    inference::BoundedQueue<int> queue(total_values);
    std::atomic<bool> push_failed{false};
    std::vector<std::thread> producers;
    producers.reserve(producer_count);

    for (std::size_t producer = 0; producer < producer_count; ++producer) {
        producers.emplace_back([&, producer] {
            for (std::size_t index = 0; index < values_per_producer; ++index) {
                const auto value = static_cast<int>(producer * values_per_producer + index);
                if (!queue.try_push(value)) {
                    push_failed.store(true, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }
    queue.close();

    std::vector<bool> seen(total_values, false);
    std::size_t received = 0;
    while (const auto value = queue.pop()) {
        ASSERT_GE(*value, 0);
        ASSERT_LT(static_cast<std::size_t>(*value), total_values);
        const auto index = static_cast<std::size_t>(*value);
        EXPECT_FALSE(seen[index]);
        seen[index] = true;
        ++received;
    }

    EXPECT_FALSE(push_failed.load(std::memory_order_relaxed));
    EXPECT_EQ(received, total_values);
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
}

}  // namespace
