#include "inference/model_runner.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::array<unsigned char, 130> kSquareModel = {
    0x08, 0x03, 0x12, 0x06, 0x63, 0x68, 0x65, 0x6e, 0x74, 0x61, 0x3a, 0x70, 0x0a, 0x15,
    0x0a, 0x01, 0x58, 0x0a, 0x01, 0x57, 0x12, 0x01, 0x59, 0x1a, 0x05, 0x6d, 0x75, 0x6c,
    0x5f, 0x31, 0x22, 0x03, 0x4d, 0x75, 0x6c, 0x12, 0x08, 0x6d, 0x75, 0x6c, 0x20, 0x74,
    0x65, 0x73, 0x74, 0x2a, 0x23, 0x08, 0x03, 0x08, 0x02, 0x10, 0x01, 0x22, 0x18, 0x00,
    0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80,
    0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x42, 0x01, 0x57, 0x5a, 0x13,
    0x0a, 0x01, 0x58, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08,
    0x03, 0x0a, 0x02, 0x08, 0x02, 0x62, 0x13, 0x0a, 0x01, 0x59, 0x12, 0x0e, 0x0a, 0x0c,
    0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x03, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x04,
    0x0a, 0x00, 0x10, 0x07,
};

class TemporaryModel {
public:
    TemporaryModel() {
        static std::atomic<unsigned int> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("inference_server_model_" + std::to_string(stamp) + "_" +
                 std::to_string(sequence.fetch_add(1)) + ".onnx");
        std::ofstream file(path_, std::ios::binary);
        file.write(reinterpret_cast<const char*>(kSquareModel.data()),
                   static_cast<std::streamsize>(kSquareModel.size()));
        if (!file) {
            throw std::runtime_error("could not write ONNX test model");
        }
    }

    ~TemporaryModel() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

TEST(ModelRunnerTest, LoadsModelInspectsMetadataAndRunsInference) {
    const TemporaryModel model;
    inference::ModelRunner runner({.model_path = model.path(),
                                   .provider = inference::ExecutionProvider::Cpu,
                                   .intra_op_threads = 1,
                                   .inter_op_threads = 1});

    const auto& metadata = runner.metadata();
    ASSERT_EQ(metadata.inputs.size(), 1);
    ASSERT_EQ(metadata.outputs.size(), 1);
    EXPECT_EQ(metadata.inputs.front().name, "X");
    EXPECT_EQ(metadata.outputs.front().name, "Y");
    EXPECT_EQ(metadata.inputs.front().element_type, inference::TensorElementType::Float32);
    EXPECT_EQ(metadata.inputs.front().shape, (inference::Tensor::Shape{3, 2}));
    EXPECT_FALSE(metadata.supports_dynamic_batch);

    const auto input = inference::Tensor::from_float32(
        {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const auto outputs = runner.infer({input});

    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs.front().shape(), (inference::Tensor::Shape{3, 2}));
    EXPECT_EQ(outputs.front().float32_values(),
              (std::vector<float>{1.0F, 4.0F, 9.0F, 16.0F, 25.0F, 36.0F}));
}

TEST(ModelRunnerTest, RejectsMissingModel) {
    inference::ModelRunnerOptions options;
    options.model_path = std::filesystem::temp_directory_path() / "missing-inference-model.onnx";
    EXPECT_THROW(inference::ModelRunner(std::move(options)), Ort::Exception);
}

TEST(ModelRunnerTest, RejectsIncompatibleTensorShape) {
    const TemporaryModel model;
    inference::ModelRunner runner({.model_path = model.path(),
                                   .provider = inference::ExecutionProvider::Cpu,
                                   .intra_op_threads = 1,
                                   .inter_op_threads = 1});
    const auto input = inference::Tensor::from_float32({2, 3}, {1, 2, 3, 4, 5, 6});

    EXPECT_THROW(static_cast<void>(runner.infer({input})), std::invalid_argument);
}

TEST(ExecutionProviderTest, RejectsUnknownProviderName) {
    EXPECT_THROW(static_cast<void>(inference::parse_execution_provider("unknown")),
                 std::invalid_argument);
}

TEST(ExecutionProviderTest, RecognizesReservedCudaProvider) {
    EXPECT_EQ(inference::parse_execution_provider("cuda"), inference::ExecutionProvider::Cuda);
}

}  // namespace
