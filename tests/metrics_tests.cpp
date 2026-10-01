#include <gtest/gtest.h>

#include <chrono>

#include "inference/metrics.hpp"

namespace {

TEST(MetricsTest, EmptySnapshotHasZeroCountersAndCurrentQueueDepth) {
    inference::Metrics metrics;

    const auto snapshot = metrics.snapshot(3);

    EXPECT_EQ(snapshot.requests_total, 0);
    EXPECT_EQ(snapshot.requests_failed, 0);
    EXPECT_EQ(snapshot.requests_rejected, 0);
    EXPECT_EQ(snapshot.batches_total, 0);
    EXPECT_EQ(snapshot.queue_depth, 3);
    EXPECT_DOUBLE_EQ(snapshot.average_batch_size, 0.0);
}

TEST(MetricsTest, AggregatesRequestBatchAndLatencyMeasurements) {
    using namespace std::chrono_literals;
    inference::Metrics metrics;
    metrics.record_request();
    metrics.record_request();
    metrics.record_failure();
    metrics.record_rejection();
    metrics.record_batch(2, 6ms, 4ms);
    metrics.record_batch(4, 12ms, 8ms);
    metrics.record_end_to_end(10ms);
    metrics.record_end_to_end(20ms);

    const auto snapshot = metrics.snapshot(5);

    EXPECT_EQ(snapshot.requests_total, 2);
    EXPECT_EQ(snapshot.requests_failed, 1);
    EXPECT_EQ(snapshot.requests_rejected, 1);
    EXPECT_EQ(snapshot.batches_total, 2);
    EXPECT_EQ(snapshot.queue_wait_samples, 6);
    EXPECT_EQ(snapshot.end_to_end_samples, 2);
    EXPECT_EQ(snapshot.queue_depth, 5);
    EXPECT_DOUBLE_EQ(snapshot.average_batch_size, 3.0);
    EXPECT_DOUBLE_EQ(snapshot.average_queue_wait_ms, 3.0);
    EXPECT_DOUBLE_EQ(snapshot.average_inference_ms, 6.0);
    EXPECT_DOUBLE_EQ(snapshot.average_end_to_end_ms, 15.0);
}

}  // namespace
