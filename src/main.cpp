#include <CLI/CLI.hpp>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "inference/dynamic_batcher.hpp"
#include "inference/execution_provider.hpp"
#include "inference/image_classifier.hpp"
#include "inference/metrics.hpp"
#include "inference/model_runner.hpp"
#include "inference/server.hpp"

namespace {

volatile std::sig_atomic_t shutdown_signal = 0;

void handle_signal(int signal_number) { shutdown_signal = signal_number; }

int run_server(const std::filesystem::path& model_path, const std::filesystem::path& config_path,
               const std::string& provider_name, const std::string& bind_address, int port,
               int threads, std::size_t max_batch_size, int max_batch_delay_ms,
               std::size_t queue_capacity, std::size_t http_threads) {
    const auto provider = inference::parse_execution_provider(provider_name);
    inference::ModelRunner runner({.model_path = model_path,
                                   .provider = provider,
                                   .intra_op_threads = threads,
                                   .inter_op_threads = 1});
    auto pipeline = inference::ImageClassificationPipeline::from_config(config_path);
    pipeline.validate_model(runner.metadata());
    if (!runner.metadata().supports_dynamic_batch && max_batch_size != 1) {
        throw std::invalid_argument(
            "model does not support dynamic batches; use --max-batch-size 1");
    }

    inference::Metrics metrics;
    inference::DynamicBatcher batcher(
        {.queue_capacity = queue_capacity,
         .max_batch_size = max_batch_size,
         .max_batch_delay = std::chrono::milliseconds{max_batch_delay_ms}},
        [&runner](const std::vector<inference::Tensor>& inputs) { return runner.infer(inputs); },
        &metrics);
    inference::InferenceServer server(
        {.bind_address = bind_address, .port = port, .http_threads = http_threads},
        {.name = model_path.filename().string(),
         .provider = provider,
         .metadata = runner.metadata()},
        pipeline, batcher, metrics);

    shutdown_signal = 0;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::jthread signal_watcher([&server](std::stop_token stop_token) {
        while (!stop_token.stop_requested() && shutdown_signal == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
        }
        if (shutdown_signal != 0) {
            server.stop();
        }
    });

    std::cout << "Listening on http://" << bind_address << ':' << port << " using "
              << inference::to_string(provider) << " provider; press Ctrl-C to stop.\n";
    const auto started = server.listen();
    signal_watcher.request_stop();
    if (signal_watcher.joinable()) {
        signal_watcher.join();
    }
    batcher.close();
    if (!started) {
        std::cerr << "Could not start the HTTP server on " << bind_address << ':' << port << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model_path;
    std::filesystem::path config_path = "config/resnet18.json";
    std::string provider = "cpu";
    std::string bind_address = "127.0.0.1";
    int port = 8080;
    int threads = 2;
    int max_batch_delay_ms = 2;
    std::size_t max_batch_size = 8;
    std::size_t queue_capacity = 64;
    std::size_t http_threads = 4;

    CLI::App app{"Small ONNX Runtime image-classification server"};
    app.add_option("--model", model_path, "ONNX model file")->required();
    app.add_option("--config", config_path, "Image classification config")
        ->default_val(config_path.string());
    app.add_option("--provider", provider, "Execution provider: cpu or coreml")
        ->default_val(provider);
    app.add_option("--bind", bind_address, "Address to listen on")->default_val(bind_address);
    app.add_option("--port", port, "HTTP port")->check(CLI::Range(1, 65535));
    app.add_option("--threads", threads, "ONNX Runtime intra-op threads")->check(CLI::Range(1, 64));
    app.add_option("--max-batch-size", max_batch_size, "Maximum inference batch size")
        ->check(CLI::Range(std::size_t{1}, std::size_t{1024}));
    app.add_option("--max-batch-delay-ms", max_batch_delay_ms, "Maximum batch collection delay")
        ->check(CLI::Range(0, 60000));
    app.add_option("--queue-capacity", queue_capacity, "Maximum queued inference requests")
        ->check(CLI::Range(std::size_t{1}, std::size_t{1000000}));
    app.add_option("--http-threads", http_threads, "HTTP handler thread count")
        ->check(CLI::Range(std::size_t{1}, std::size_t{64}));

    try {
        app.parse(argc, argv);
        return run_server(model_path, config_path, provider, bind_address, port, threads,
                          max_batch_size, max_batch_delay_ms, queue_capacity, http_threads);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    } catch (const std::exception& error) {
        std::cerr << "inference-server: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
