#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "inference/image_classifier.hpp"

namespace {

constexpr std::array<unsigned char, 68> kOnePixelPng = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
    0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x04, 0x00, 0x00,
    0x00, 0xb5, 0x1c, 0x0c, 0x02, 0x00, 0x00, 0x00, 0x0b, 0x49, 0x44, 0x41, 0x54, 0x78,
    0xda, 0x63, 0xfc, 0xff, 0x1f, 0x00, 0x03, 0x03, 0x02, 0x00, 0xef, 0xa3, 0x44, 0x2f,
    0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

class TemporaryDirectory {
   public:
    TemporaryDirectory() {
        static std::atomic<unsigned int> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("inference_server_image_" + std::to_string(stamp) + "_" +
                 std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

   private:
    std::filesystem::path path_;
};

[[nodiscard]] std::vector<std::byte> png_bytes() {
    std::vector<std::byte> bytes;
    bytes.reserve(kOnePixelPng.size());
    for (const auto value : kOnePixelPng) {
        bytes.push_back(static_cast<std::byte>(value));
    }
    return bytes;
}

[[nodiscard]] inference::ImageClassificationPipeline make_pipeline() {
    inference::ImageClassificationConfig config;
    config.width = 2;
    config.height = 2;
    config.resize_shorter = 2;
    return inference::ImageClassificationPipeline(std::move(config), {"first", "second", "third"});
}

TEST(ImageClassifierTest, RejectsInvalidImage) {
    const auto pipeline = make_pipeline();
    const std::array<std::byte, 4> invalid{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};

    EXPECT_THROW(static_cast<void>(pipeline.preprocess(invalid)), std::invalid_argument);
}

TEST(ImageClassifierTest, PreprocessesPngIntoNormalizedNchwTensor) {
    const auto pipeline = make_pipeline();

    const auto tensor = pipeline.preprocess(png_bytes());

    EXPECT_EQ(tensor.element_type(), inference::TensorElementType::Float32);
    EXPECT_EQ(tensor.shape(), (inference::Tensor::Shape{1, 3, 2, 2}));
    EXPECT_EQ(tensor.float32_values().size(), 12);
}

TEST(ImageClassifierTest, RejectsZeroConfiguredPixelLimit) {
    inference::ImageClassificationConfig config;
    config.width = 2;
    config.height = 2;
    config.resize_shorter = 2;
    config.max_image_pixels = 0;
    EXPECT_THROW(inference::ImageClassificationPipeline(std::move(config), {"label"}),
                 std::invalid_argument);
}

TEST(ImageClassifierTest, ReturnsHighestScoringTopKLabels) {
    const auto pipeline = make_pipeline();
    const auto scores = inference::Tensor::from_float32({1, 3}, {0.0F, 1.0F, 2.0F});

    const auto predictions = pipeline.postprocess(scores, 2);

    ASSERT_EQ(predictions.size(), 2);
    EXPECT_EQ(predictions[0].class_id, 2);
    EXPECT_EQ(predictions[0].label, "third");
    EXPECT_EQ(predictions[1].class_id, 1);
    EXPECT_GT(predictions[0].score, predictions[1].score);
    EXPECT_NEAR(predictions[0].score + predictions[1].score, 0.9100F, 0.001F);
}

TEST(ImageClassifierTest, ValidatesModelInputAndOutputMetadata) {
    const auto pipeline = make_pipeline();
    const inference::ModelMetadata metadata{
        .inputs = {{.name = "images",
                    .element_type = inference::TensorElementType::Float32,
                    .shape = {1, 3, 2, 2}}},
        .outputs = {{.name = "scores",
                     .element_type = inference::TensorElementType::Float32,
                     .shape = {1, 3}}},
        .supports_dynamic_batch = false,
    };

    EXPECT_NO_THROW(pipeline.validate_model(metadata));

    auto incompatible = metadata;
    incompatible.inputs.front().shape[1] = 1;
    EXPECT_THROW(pipeline.validate_model(incompatible), std::invalid_argument);
}

TEST(ImageClassifierTest, LoadsLabelsRelativeToConfigFile) {
    TemporaryDirectory directory;
    {
        std::ofstream labels(directory.path() / "labels.txt");
        labels << "n00000001 first label\nn00000002 second label\n";
    }
    {
        std::ofstream config(directory.path() / "classifier.json");
        config << R"({
            "width": 2,
            "height": 2,
            "resize_shorter": 2,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "labels_file": "labels.txt"
        })";
    }

    const auto pipeline =
        inference::ImageClassificationPipeline::from_config(directory.path() / "classifier.json");

    ASSERT_EQ(pipeline.labels().size(), 2);
    EXPECT_EQ(pipeline.labels()[0], "first label");
}

}  // namespace
