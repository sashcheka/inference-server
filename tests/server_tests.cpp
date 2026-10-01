#include <gtest/gtest.h>
#include <httplib.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "inference/server.hpp"

namespace {

constexpr std::array<unsigned char, 68> kOnePixelPng = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
    0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x04, 0x00, 0x00,
    0x00, 0xb5, 0x1c, 0x0c, 0x02, 0x00, 0x00, 0x00, 0x0b, 0x49, 0x44, 0x41, 0x54, 0x78,
    0xda, 0x63, 0xfc, 0xff, 0x1f, 0x00, 0x03, 0x03, 0x02, 0x00, 0xef, 0xa3, 0x44, 0x2f,
    0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

[[nodiscard]] std::string png_body() {
    return {reinterpret_cast<const char*>(kOnePixelPng.data()), kOnePixelPng.size()};
}

[[nodiscard]] inference::ImageClassificationPipeline make_pipeline() {
    inference::ImageClassificationConfig config;
    config.width = 2;
    config.height = 2;
    config.resize_shorter = 2;
    return inference::ImageClassificationPipeline(std::move(config), {"zero", "one", "two"});
}

[[nodiscard]] inference::ModelMetadata make_metadata() {
    return {.inputs = {{.name = "images",
                        .element_type = inference::TensorElementType::Float32,
                        .shape = {-1, 3, 2, 2}}},
            .outputs = {{.name = "scores",
                         .element_type = inference::TensorElementType::Float32,
                         .shape = {-1, 3}}},
            .supports_dynamic_batch = true};
}

[[nodiscard]] std::vector<inference::Tensor> class_scores(
    const std::vector<inference::Tensor>& inputs) {
    const auto batch_size = static_cast<std::size_t>(inputs.front().shape().front());
    std::vector<float> values(batch_size * 3, 0.0F);
    for (std::size_t row = 0; row < batch_size; ++row) {
        values[row * 3 + 1] = 1.0F;
        values[row * 3 + 2] = 2.0F;
    }
    return {inference::Tensor::from_float32({static_cast<std::int64_t>(batch_size), 3}, values)};
}

class TestServer {
   public:
    TestServer(const inference::ImageClassificationPipeline& pipeline,
               inference::DynamicBatcher& batcher, inference::Metrics& metrics)
        : server_({.bind_address = "127.0.0.1", .port = 8080, .http_threads = 4},
                  {.name = "test-model",
                   .provider = inference::ExecutionProvider::Cpu,
                   .metadata = make_metadata()},
                  pipeline, batcher, metrics) {
        port_ = server_.bind_to_any_port();
        if (port_ <= 0) {
            throw std::runtime_error("could not bind test server to an available port");
        }
        listener_ = std::thread([this] {
            static_cast<void>(server_.listen_after_bind());
            listener_finished_.store(true, std::memory_order_release);
        });
        if (!wait_until_ready()) {
            stop();
            throw std::runtime_error("test HTTP server did not start");
        }
    }

    ~TestServer() { stop(); }

    TestServer(const TestServer&) = delete;
    TestServer& operator=(const TestServer&) = delete;

    [[nodiscard]] int port() const noexcept { return port_; }

    void stop() {
        if (listener_.joinable()) {
            server_.stop();
            listener_.join();
        }
    }

    [[nodiscard]] bool listener_finished() const noexcept {
        return listener_finished_.load(std::memory_order_acquire);
    }

   private:
    [[nodiscard]] bool wait_until_ready() const {
        httplib::Client client("127.0.0.1", port_);
        client.set_connection_timeout(std::chrono::milliseconds{100});
        for (int attempt = 0; attempt < 100; ++attempt) {
            if (client.Get("/health")) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }
        return false;
    }

    inference::InferenceServer server_;
    int port_ = 0;
    std::thread listener_;
    std::atomic<bool> listener_finished_{false};
};

[[nodiscard]] int post_image(int port) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds{2});
    client.set_read_timeout(std::chrono::seconds{5});
    const auto result = client.Post("/predict", png_body(), "image/png");
    return result ? result->status : 0;
}

[[nodiscard]] bool wait_for_queued_request(int port) {
    httplib::Client client("127.0.0.1", port);
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (const auto response = client.Get("/metrics")) {
            const auto metrics = nlohmann::json::parse(response->body);
            if (metrics.at("requests_total").get<std::uint64_t>() >= 2 &&
                metrics.at("queue_depth").get<std::size_t>() == 1) {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    return false;
}

TEST(InferenceServerTest, ExposesHealthModelAndMetricsEndpoints) {
    auto pipeline = make_pipeline();
    inference::Metrics metrics;
    inference::DynamicBatcher batcher(
        {.queue_capacity = 4, .max_batch_size = 1, .max_batch_delay = std::chrono::milliseconds{0}},
        class_scores, &metrics);
    TestServer server(pipeline, batcher, metrics);
    httplib::Client client("127.0.0.1", server.port());

    const auto health = client.Get("/health");
    const auto model = client.Get("/model");
    const auto metrics_response = client.Get("/metrics");

    ASSERT_TRUE(health);
    ASSERT_TRUE(model);
    ASSERT_TRUE(metrics_response);
    EXPECT_EQ(health->status, 200);
    EXPECT_EQ(nlohmann::json::parse(health->body).at("status"), "ok");
    const auto model_json = nlohmann::json::parse(model->body);
    EXPECT_EQ(model_json.at("name"), "test-model");
    EXPECT_EQ(model_json.at("provider"), "cpu");
    EXPECT_EQ(model_json.at("inputs").at(0).at("shape"), (nlohmann::json::array({-1, 3, 2, 2})));
    EXPECT_EQ(metrics_response->status, 200);
}

TEST(InferenceServerTest, PredictsAndRejectsUnsupportedRequestBodies) {
    auto pipeline = make_pipeline();
    inference::Metrics metrics;
    inference::DynamicBatcher batcher(
        {.queue_capacity = 4, .max_batch_size = 1, .max_batch_delay = std::chrono::milliseconds{0}},
        class_scores, &metrics);
    TestServer server(pipeline, batcher, metrics);
    httplib::Client client("127.0.0.1", server.port());

    const auto prediction = client.Post("/predict", png_body(), "image/png");
    const auto unsupported = client.Post("/predict", "not an image", "text/plain");
    const auto invalid_image = client.Post("/predict", "invalid", "image/png");

    ASSERT_TRUE(prediction);
    ASSERT_TRUE(unsupported);
    ASSERT_TRUE(invalid_image);
    EXPECT_EQ(prediction->status, 200);
    const auto prediction_json = nlohmann::json::parse(prediction->body);
    EXPECT_EQ(prediction_json.at("predictions").at(0).at("class_id"), 2);
    EXPECT_EQ(prediction_json.at("predictions").at(0).at("label"), "two");
    EXPECT_EQ(unsupported->status, 415);
    EXPECT_EQ(invalid_image->status, 400);
}

TEST(InferenceServerTest, RejectsOverloadAndStopsCleanly) {
    auto pipeline = make_pipeline();
    inference::Metrics metrics;
    std::promise<void> inference_started;
    auto started = inference_started.get_future();
    std::promise<void> release_inference;
    const auto release = release_inference.get_future().share();
    std::atomic<bool> first_call{true};
    inference::DynamicBatcher batcher(
        {.queue_capacity = 1, .max_batch_size = 1, .max_batch_delay = std::chrono::milliseconds{0}},
        [&](const std::vector<inference::Tensor>& inputs) {
            if (first_call.exchange(false, std::memory_order_acq_rel)) {
                inference_started.set_value();
            }
            release.wait();
            return class_scores(inputs);
        },
        &metrics);
    TestServer server(pipeline, batcher, metrics);
    std::promise<int> first_status;
    auto first_response = first_status.get_future();
    std::thread first_request([&] { first_status.set_value(post_image(server.port())); });

    const auto first_inference_started =
        started.wait_for(std::chrono::seconds{2}) == std::future_status::ready;
    std::promise<int> second_status;
    auto second_response = second_status.get_future();
    std::thread second_request;
    auto second_is_queued = false;
    auto overloaded_status = 0;
    if (first_inference_started) {
        second_request = std::thread([&] { second_status.set_value(post_image(server.port())); });
        second_is_queued = wait_for_queued_request(server.port());
        if (second_is_queued) {
            overloaded_status = post_image(server.port());
        }
    }
    release_inference.set_value();
    first_request.join();
    if (second_request.joinable()) {
        second_request.join();
    }
    server.stop();

    EXPECT_TRUE(first_inference_started);
    EXPECT_TRUE(second_is_queued);
    EXPECT_EQ(overloaded_status, 503);
    EXPECT_EQ(first_response.get(), 200);
    if (second_is_queued) {
        EXPECT_EQ(second_response.get(), 200);
    }
    EXPECT_TRUE(server.listener_finished());
    const auto snapshot = metrics.snapshot(batcher.queue_depth());
    EXPECT_EQ(snapshot.requests_rejected, 1);
}

}  // namespace
