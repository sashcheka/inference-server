#pragma once

#include <httplib.h>

#include <cstddef>
#include <string>

#include "inference/dynamic_batcher.hpp"
#include "inference/execution_provider.hpp"
#include "inference/image_classifier.hpp"
#include "inference/metrics.hpp"
#include "inference/model_metadata.hpp"

namespace inference {

struct ServerOptions {
    std::string bind_address = "127.0.0.1";
    int port = 8080;
    std::size_t http_threads = 4;
    std::size_t max_pending_http_requests = 64;
    std::size_t max_request_bytes = 10 * 1024 * 1024;
};

struct ServedModelInfo {
    std::string name;
    ExecutionProvider provider = ExecutionProvider::Cpu;
    ModelMetadata metadata;
};

class InferenceServer {
   public:
    InferenceServer(ServerOptions options, ServedModelInfo model,
                    const ImageClassificationPipeline& pipeline, DynamicBatcher& batcher,
                    Metrics& metrics);

    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;

    [[nodiscard]] bool listen();
    [[nodiscard]] int bind_to_any_port();
    [[nodiscard]] bool listen_after_bind();
    void stop();

   private:
    void register_routes();
    void handle_predict(const httplib::Request& request, httplib::Response& response);

    ServerOptions options_;
    ServedModelInfo model_;
    const ImageClassificationPipeline& pipeline_;
    DynamicBatcher& batcher_;
    Metrics& metrics_;
    httplib::Server server_;
};

}  // namespace inference
