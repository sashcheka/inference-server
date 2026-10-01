#include <httplib.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <latch>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct WorkerResult {
    std::vector<double> latency_ms;
    std::size_t succeeded = 0;
    std::size_t failed = 0;
};

struct MetricWindow {
    std::uint64_t requests = 0;
    std::uint64_t failures = 0;
    std::uint64_t rejections = 0;
    std::uint64_t batches = 0;
    double average_batch_size = 0.0;
    double average_queue_wait_ms = 0.0;
    double average_inference_ms = 0.0;
    double average_end_to_end_ms = 0.0;
};

[[nodiscard]] std::string read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("could not open image file: " + filename);
    }
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (data.empty()) {
        throw std::runtime_error("image file is empty: " + filename);
    }
    constexpr std::size_t max_image_bytes = 10 * 1024 * 1024;
    if (data.size() > max_image_bytes) {
        throw std::runtime_error("image exceeds the server's 10 MiB request limit");
    }
    return data;
}

[[nodiscard]] std::string content_type_for(const std::string& filename) {
    auto extension = std::filesystem::path(filename).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (extension == ".jpg" || extension == ".jpeg") {
        return "image/jpeg";
    }
    if (extension == ".png") {
        return "image/png";
    }
    throw std::invalid_argument("image extension must be .jpg, .jpeg, or .png");
}

[[nodiscard]] Json fetch_metrics(const std::string& url) {
    httplib::Client client(url);
    client.set_connection_timeout(std::chrono::seconds{2});
    client.set_read_timeout(std::chrono::seconds{2});
    const auto response = client.Get("/metrics");
    if (!response || response->status != 200) {
        throw std::runtime_error("could not read server metrics from " + url + "/metrics");
    }
    return Json::parse(response->body);
}

[[nodiscard]] double average_delta(const Json& before, const Json& after, const char* average_name,
                                   const char* count_name) {
    const auto before_count = before.at(count_name).get<std::uint64_t>();
    const auto after_count = after.at(count_name).get<std::uint64_t>();
    if (after_count < before_count) {
        return 0.0;
    }
    const auto count_delta = after_count - before_count;
    if (count_delta == 0) {
        return 0.0;
    }
    const auto before_sum =
        before.at(average_name).get<double>() * static_cast<double>(before_count);
    const auto after_sum = after.at(average_name).get<double>() * static_cast<double>(after_count);
    return (after_sum - before_sum) / static_cast<double>(count_delta);
}

[[nodiscard]] MetricWindow metric_window(const Json& before, const Json& after) {
    MetricWindow result;
    const auto before_requests = before.at("requests_total").get<std::uint64_t>();
    const auto after_requests = after.at("requests_total").get<std::uint64_t>();
    const auto before_failures = before.at("requests_failed").get<std::uint64_t>();
    const auto after_failures = after.at("requests_failed").get<std::uint64_t>();
    const auto before_rejections = before.at("requests_rejected").get<std::uint64_t>();
    const auto after_rejections = after.at("requests_rejected").get<std::uint64_t>();
    const auto before_batches = before.at("batches_total").get<std::uint64_t>();
    const auto after_batches = after.at("batches_total").get<std::uint64_t>();
    if (after_requests < before_requests || after_failures < before_failures ||
        after_rejections < before_rejections || after_batches < before_batches) {
        return result;
    }

    result.requests = after_requests - before_requests;
    result.failures = after_failures - before_failures;
    result.rejections = after_rejections - before_rejections;
    result.batches = after_batches - before_batches;
    result.average_batch_size = average_delta(before, after, "average_batch_size", "batches_total");
    result.average_queue_wait_ms =
        average_delta(before, after, "average_queue_wait_ms", "queue_wait_samples");
    result.average_inference_ms =
        average_delta(before, after, "average_inference_ms", "batches_total");
    result.average_end_to_end_ms =
        average_delta(before, after, "average_end_to_end_ms", "end_to_end_samples");
    return result;
}

[[nodiscard]] double percentile(const std::vector<double>& sorted_values, double fraction) {
    if (sorted_values.empty()) {
        return 0.0;
    }
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted_values.size())) - 1.0);
    return sorted_values[std::min(index, sorted_values.size() - 1)];
}

int run_benchmark(const std::string& url, const std::string& image_path, int concurrency,
                  int duration_seconds) {
    const auto image = read_file(image_path);
    const auto content_type = content_type_for(image_path);
    const auto before_metrics = fetch_metrics(url);
    std::vector<WorkerResult> worker_results(static_cast<std::size_t>(concurrency));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(concurrency));
    std::latch ready(static_cast<std::ptrdiff_t>(concurrency));
    std::latch start_gate(1);
    Clock::time_point deadline;
    for (int worker_index = 0; worker_index < concurrency; ++worker_index) {
        workers.emplace_back([&, worker_index] {
            auto& result = worker_results[static_cast<std::size_t>(worker_index)];
            ready.count_down();
            start_gate.wait();
            try {
                httplib::Client client(url);
                client.set_connection_timeout(std::chrono::seconds{2});
                client.set_read_timeout(std::chrono::seconds{30});
                client.set_write_timeout(std::chrono::seconds{5});
                while (Clock::now() < deadline) {
                    const auto request_start = Clock::now();
                    const auto response = client.Post("/predict", image, content_type);
                    const auto request_end = Clock::now();
                    result.latency_ms.push_back(
                        std::chrono::duration<double, std::milli>(request_end - request_start)
                            .count());
                    if (response && response->status == 200) {
                        ++result.succeeded;
                    } else {
                        ++result.failed;
                    }
                }
            } catch (const std::exception&) {
                ++result.failed;
            }
        });
    }
    ready.wait();
    const auto start = Clock::now();
    deadline = start + std::chrono::seconds{duration_seconds};
    start_gate.count_down();
    for (auto& worker : workers) {
        worker.join();
    }
    const auto elapsed = std::chrono::duration<double>(deadline - start).count();
    const auto after_metrics = fetch_metrics(url);
    const auto server_metrics = metric_window(before_metrics, after_metrics);

    WorkerResult total;
    for (auto& result : worker_results) {
        total.succeeded += result.succeeded;
        total.failed += result.failed;
        total.latency_ms.insert(total.latency_ms.end(), result.latency_ms.begin(),
                                result.latency_ms.end());
    }
    const auto request_count = total.succeeded + total.failed;
    double mean_latency_ms = 0.0;
    for (const auto latency : total.latency_ms) {
        mean_latency_ms += latency;
    }
    if (!total.latency_ms.empty()) {
        mean_latency_ms /= static_cast<double>(total.latency_ms.size());
    }
    auto sorted_latency = total.latency_ms;
    std::sort(sorted_latency.begin(), sorted_latency.end());

    std::cout << "Benchmark: " << url << " using " << image_path << '\n'
              << "Duration: " << elapsed << " s; concurrency: " << concurrency << '\n'
              << "Requests: " << request_count << " (" << total.succeeded << " successful, "
              << total.failed << " failed); throughput: "
              << (elapsed > 0.0 ? static_cast<double>(request_count) / elapsed : 0.0)
              << " requests/s\n"
              << "End-to-end latency (ms): mean=" << mean_latency_ms
              << " p50=" << percentile(sorted_latency, 0.50)
              << " p95=" << percentile(sorted_latency, 0.95)
              << " p99=" << percentile(sorted_latency, 0.99) << '\n'
              << "Server interval: requests=" << server_metrics.requests
              << " failures=" << server_metrics.failures
              << " rejections=" << server_metrics.rejections
              << " batches=" << server_metrics.batches
              << " average_batch_size=" << server_metrics.average_batch_size
              << " queue_wait_ms=" << server_metrics.average_queue_wait_ms
              << " inference_ms=" << server_metrics.average_inference_ms
              << " server_end_to_end_ms=" << server_metrics.average_end_to_end_ms << '\n';
    return total.failed == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string url = "http://127.0.0.1:8080";
    std::string image_path;
    int concurrency = 4;
    int duration_seconds = 10;

    CLI::App app{"Small HTTP load test for inference-server"};
    app.add_option("--url", url, "Base URL of a running inference-server")->default_val(url);
    app.add_option("--image", image_path, "JPEG or PNG file to send")->required();
    app.add_option("--concurrency", concurrency, "Concurrent clients")->check(CLI::Range(1, 64));
    app.add_option("--duration", duration_seconds, "Benchmark duration in seconds")
        ->check(CLI::Range(1, 600));

    try {
        app.parse(argc, argv);
        return run_benchmark(url, image_path, concurrency, duration_seconds);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    } catch (const std::exception& error) {
        std::cerr << "benchmark: " << error.what() << '\n';
        return 1;
    }
}
