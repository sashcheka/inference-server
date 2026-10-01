#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "inference/model_metadata.hpp"
#include "inference/tensor.hpp"

namespace inference {

struct ImageClassificationConfig {
    std::size_t width = 224;
    std::size_t height = 224;
    std::size_t resize_shorter = 256;
    std::size_t max_image_pixels = 40000000;
    std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> standard_deviation{0.229F, 0.224F, 0.225F};
    std::filesystem::path labels_file;
};

struct Prediction {
    std::size_t class_id;
    std::string label;
    float score;
};

class ImageClassificationPipeline {
   public:
    ImageClassificationPipeline(ImageClassificationConfig config, std::vector<std::string> labels);

    [[nodiscard]] static ImageClassificationPipeline from_config(
        const std::filesystem::path& config_path);

    void validate_model(const ModelMetadata& metadata) const;
    [[nodiscard]] Tensor preprocess(std::span<const std::byte> encoded_image) const;
    [[nodiscard]] std::vector<Prediction> postprocess(const Tensor& scores,
                                                      std::size_t top_k = 5) const;

    [[nodiscard]] const ImageClassificationConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<std::string>& labels() const noexcept { return labels_; }

   private:
    ImageClassificationConfig config_;
    std::vector<std::string> labels_;
};

}  // namespace inference
