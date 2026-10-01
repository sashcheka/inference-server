#include "inference/server.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace inference {
namespace {

using Json = nlohmann::json;

[[nodiscard]] Json tensor_metadata_json(const TensorMetadata& metadata) {
    return Json{{"name", metadata.name},
                {"element_type", to_string(metadata.element_type)},
                {"shape", metadata.shape}};
}

void write_json(httplib::Response& response, int status, const Json& body) {
    response.status = status;
    response.set_content(body.dump(), "application/json");
}

[[nodiscard]] std::string normalized_content_type(std::string value) {
    const auto semicolon = value.find(';');
    if (semicolon != std::string::npos) {
        value.resize(semicolon);
    }
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char character) {
                          return std::isspace(character) != 0;
                      }).base();
    if (first >= last) {
        return {};
    }
    value = std::string(first, last);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

class RequestLatencyRecorder {
   public:
    explicit RequestLatencyRecorder(Metrics& metrics)
        : metrics_(metrics), started_at_(std::chrono::steady_clock::now()) {}

    ~RequestLatencyRecorder() {
        metrics_.record_end_to_end(std::chrono::steady_clock::now() - started_at_);
    }

   private:
    Metrics& metrics_;
    std::chrono::steady_clock::time_point started_at_;
};

}  // namespace

InferenceServer::InferenceServer(ServerOptions options, ServedModelInfo model,
                                 const ImageClassificationPipeline& pipeline,
                                 DynamicBatcher& batcher, Metrics& metrics)
    : options_(std::move(options)),
      model_(std::move(model)),
      pipeline_(pipeline),
      batcher_(batcher),
      metrics_(metrics) {
    if (options_.port < 1 || options_.port > 65535 || options_.bind_address.empty()) {
        throw std::invalid_argument("server bind address and port are invalid");
    }
    if (options_.http_threads == 0 || options_.http_threads > 64 ||
        options_.max_pending_http_requests == 0 || options_.max_request_bytes == 0) {
        throw std::invalid_argument("HTTP thread, queue, and payload limits must be positive");
    }
    if (model_.name.empty()) {
        throw std::invalid_argument("served model name is required");
    }
    pipeline_.validate_model(model_.metadata);

    const auto thread_count = options_.http_threads;
    const auto pending_limit = options_.max_pending_http_requests;
    server_.new_task_queue = [thread_count, pending_limit] {
        return new httplib::ThreadPool(thread_count, thread_count, pending_limit);
    };
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

bool InferenceServer::listen() { return server_.listen(options_.bind_address, options_.port); }

int InferenceServer::bind_to_any_port() { return server_.bind_to_any_port(options_.bind_address); }

bool InferenceServer::listen_after_bind() { return server_.listen_after_bind(); }

void InferenceServer::stop() { server_.stop(); }

void InferenceServer::register_routes() {
    server_.Get("/health", [this](const httplib::Request&, httplib::Response& response) {
        write_json(response, 200,
                   Json{{"status", "ok"},
                        {"queue_depth", batcher_.queue_depth()},
                        {"queue_capacity", batcher_.queue_capacity()}});
    });

    server_.Get("/model", [this](const httplib::Request&, httplib::Response& response) {
        Json inputs = Json::array();
        for (const auto& input : model_.metadata.inputs) {
            inputs.push_back(tensor_metadata_json(input));
        }
        Json outputs = Json::array();
        for (const auto& output : model_.metadata.outputs) {
            outputs.push_back(tensor_metadata_json(output));
        }
        write_json(response, 200,
                   Json{{"name", model_.name},
                        {"provider", to_string(model_.provider)},
                        {"supports_dynamic_batch", model_.metadata.supports_dynamic_batch},
                        {"inputs", std::move(inputs)},
                        {"outputs", std::move(outputs)}});
    });

    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& response) {
        const auto snapshot = metrics_.snapshot(batcher_.queue_depth());
        write_json(response, 200,
                   Json{{"requests_total", snapshot.requests_total},
                        {"requests_failed", snapshot.requests_failed},
                        {"requests_rejected", snapshot.requests_rejected},
                        {"queue_depth", snapshot.queue_depth},
                        {"batches_total", snapshot.batches_total},
                        {"queue_wait_samples", snapshot.queue_wait_samples},
                        {"end_to_end_samples", snapshot.end_to_end_samples},
                        {"average_batch_size", snapshot.average_batch_size},
                        {"average_queue_wait_ms", snapshot.average_queue_wait_ms},
                        {"average_inference_ms", snapshot.average_inference_ms},
                        {"average_end_to_end_ms", snapshot.average_end_to_end_ms}});
    });

    server_.Post("/predict", [this](const httplib::Request& request, httplib::Response& response) {
        handle_predict(request, response);
    });

    server_.set_error_handler([](const httplib::Request&, httplib::Response& response) {
        if (response.status == 413) {
            write_json(response, 413, Json{{"error", "request body exceeds configured limit"}});
        }
    });
}

void InferenceServer::handle_predict(const httplib::Request& request, httplib::Response& response) {
    metrics_.record_request();
    const RequestLatencyRecorder latency(metrics_);

    const auto content_type = normalized_content_type(request.get_header_value("Content-Type"));
    if (content_type != "image/jpeg" && content_type != "image/png") {
        metrics_.record_failure();
        write_json(response, 415, Json{{"error", "Content-Type must be image/jpeg or image/png"}});
        return;
    }
    if (request.body.empty()) {
        metrics_.record_failure();
        write_json(response, 400, Json{{"error", "request body must contain an image"}});
        return;
    }

    try {
        const auto* data = reinterpret_cast<const std::byte*>(request.body.data());
        const std::span<const std::byte> encoded(data, request.body.size());
        auto image = pipeline_.preprocess(encoded);
        auto future = batcher_.try_submit({std::move(image)});
        if (!future) {
            metrics_.record_rejection();
            response.set_header("Retry-After", "1");
            write_json(response, 503, Json{{"error", "inference queue is full"}});
            return;
        }

        auto outputs = future->get();
        if (outputs.size() != 1) {
            throw std::runtime_error("classification model must return one output tensor");
        }
        const auto predictions = pipeline_.postprocess(outputs.front());
        Json results = Json::array();
        for (const auto& prediction : predictions) {
            results.push_back(Json{{"class_id", prediction.class_id},
                                   {"label", prediction.label},
                                   {"score", prediction.score}});
        }
        write_json(response, 200, Json{{"predictions", std::move(results)}});
    } catch (const std::invalid_argument& error) {
        metrics_.record_failure();
        write_json(response, 400, Json{{"error", error.what()}});
    } catch (const std::exception& error) {
        metrics_.record_failure();
        write_json(response, 500, Json{{"error", error.what()}});
    } catch (...) {
        metrics_.record_failure();
        write_json(response, 500, Json{{"error", "unknown inference error"}});
    }
}

}  // namespace inference
