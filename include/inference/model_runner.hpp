#pragma once

#include "inference/execution_provider.hpp"
#include "inference/model_metadata.hpp"
#include "inference/tensor.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace inference {

struct ModelRunnerOptions {
    std::filesystem::path model_path;
    ExecutionProvider provider = ExecutionProvider::Cpu;
    int intra_op_threads = 2;
    int inter_op_threads = 1;
};

class ModelRunner {
public:
    explicit ModelRunner(ModelRunnerOptions options);

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    [[nodiscard]] const ModelMetadata& metadata() const noexcept { return metadata_; }
    [[nodiscard]] ExecutionProvider provider() const noexcept { return options_.provider; }
    [[nodiscard]] std::vector<Tensor> infer(const std::vector<Tensor>& inputs);

private:
    [[nodiscard]] ModelMetadata inspect_model() const;
    void validate_inputs(const std::vector<Tensor>& inputs) const;

    ModelRunnerOptions options_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    ModelMetadata metadata_;
};

}  // namespace inference
