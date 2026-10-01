#include "inference/image_classifier.hpp"

#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace inference {
namespace {

using Json = nlohmann::json;

[[nodiscard]] std::vector<std::string> load_labels(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open labels file: " + path.string());
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line.size() > 10 && line.front() == 'n' && line[9] == ' ') {
            line.erase(0, 10);
        }
        labels.push_back(std::move(line));
    }
    if (labels.empty()) {
        throw std::runtime_error("labels file is empty: " + path.string());
    }
    return labels;
}

[[nodiscard]] std::array<float, 3> read_triplet(const Json& json, std::string_view key) {
    const auto values = json.at(key).get<std::vector<float>>();
    if (values.size() != 3) {
        throw std::invalid_argument("classification config field '" + std::string(key) +
                                    "' must contain three values");
    }
    return {values[0], values[1], values[2]};
}

[[nodiscard]] std::size_t scaled_dimension(std::size_t source, double scale) {
    const auto result = std::lround(static_cast<double>(source) * scale);
    if (result <= 0 || static_cast<unsigned long long>(result) >
                           static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("resized image dimension is outside supported range");
    }
    return static_cast<std::size_t>(result);
}

[[nodiscard]] float bilinear_channel(const stbi_uc* pixels, int source_width, int source_height,
                                     std::size_t resized_width, std::size_t resized_height,
                                     std::size_t crop_left, std::size_t crop_top,
                                     std::size_t target_x, std::size_t target_y,
                                     std::size_t channel) {
    const auto source_x = (static_cast<double>(crop_left + target_x) + 0.5) *
                              static_cast<double>(source_width) /
                              static_cast<double>(resized_width) -
                          0.5;
    const auto source_y = (static_cast<double>(crop_top + target_y) + 0.5) *
                              static_cast<double>(source_height) /
                              static_cast<double>(resized_height) -
                          0.5;
    const auto clamped_x = std::clamp(source_x, 0.0, static_cast<double>(source_width - 1));
    const auto clamped_y = std::clamp(source_y, 0.0, static_cast<double>(source_height - 1));
    const auto x0 = static_cast<int>(std::floor(clamped_x));
    const auto y0 = static_cast<int>(std::floor(clamped_y));
    const auto x1 = std::min(x0 + 1, source_width - 1);
    const auto y1 = std::min(y0 + 1, source_height - 1);
    const auto x_weight = static_cast<float>(clamped_x - static_cast<double>(x0));
    const auto y_weight = static_cast<float>(clamped_y - static_cast<double>(y0));
    const auto pixel_offset = [source_width, channel](int x, int y) {
        return (static_cast<std::size_t>(y) * static_cast<std::size_t>(source_width) +
                static_cast<std::size_t>(x)) *
                   3 +
               channel;
    };
    const auto top_left = static_cast<float>(pixels[pixel_offset(x0, y0)]);
    const auto top_right = static_cast<float>(pixels[pixel_offset(x1, y0)]);
    const auto bottom_left = static_cast<float>(pixels[pixel_offset(x0, y1)]);
    const auto bottom_right = static_cast<float>(pixels[pixel_offset(x1, y1)]);
    const auto top = top_left + (top_right - top_left) * x_weight;
    const auto bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
    return top + (bottom - top) * y_weight;
}

}  // namespace

ImageClassificationPipeline::ImageClassificationPipeline(ImageClassificationConfig config,
                                                         std::vector<std::string> labels)
    : config_(std::move(config)), labels_(std::move(labels)) {
    constexpr std::size_t max_output_pixels = 1024 * 1024;
    if (config_.width == 0 || config_.height == 0 ||
        config_.width > max_output_pixels / config_.height ||
        config_.resize_shorter < std::max(config_.width, config_.height)) {
        throw std::invalid_argument("image size and resize_shorter are outside supported limits");
    }
    if (config_.max_image_pixels == 0) {
        throw std::invalid_argument("max_image_pixels must be positive");
    }
    if (labels_.empty()) {
        throw std::invalid_argument("classification pipeline requires at least one label");
    }
    for (std::size_t channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(config_.mean[channel]) ||
            !std::isfinite(config_.standard_deviation[channel]) ||
            config_.standard_deviation[channel] <= 0.0F) {
            throw std::invalid_argument(
                "normalization means and standard deviations must be finite");
        }
    }
}

ImageClassificationPipeline ImageClassificationPipeline::from_config(
    const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        throw std::runtime_error("could not open classification config: " + config_path.string());
    }
    const auto json = Json::parse(file);

    ImageClassificationConfig config;
    config.width = json.at("width").get<std::size_t>();
    config.height = json.at("height").get<std::size_t>();
    config.resize_shorter = json.at("resize_shorter").get<std::size_t>();
    config.max_image_pixels = json.value("max_image_pixels", config.max_image_pixels);
    config.mean = read_triplet(json, "mean");
    config.standard_deviation = read_triplet(json, "std");
    config.labels_file = json.at("labels_file").get<std::string>();
    if (config.labels_file.is_relative()) {
        config.labels_file = config_path.parent_path() / config.labels_file;
    }
    auto labels = load_labels(config.labels_file);
    return ImageClassificationPipeline(std::move(config), std::move(labels));
}

void ImageClassificationPipeline::validate_model(const ModelMetadata& metadata) const {
    if (metadata.inputs.size() != 1 || metadata.outputs.size() != 1) {
        throw std::invalid_argument("image classification requires one model input and one output");
    }

    const auto& input = metadata.inputs.front();
    if (input.element_type != TensorElementType::Float32 || input.shape.size() != 4 ||
        input.shape[1] != 3 || (input.shape[0] != 1 && input.shape[0] >= 0) ||
        (input.shape[2] >= 0 && static_cast<std::size_t>(input.shape[2]) != config_.height) ||
        (input.shape[3] >= 0 && static_cast<std::size_t>(input.shape[3]) != config_.width)) {
        throw std::invalid_argument(
            "classification model input must be float32 NCHW with 3 channels and batch 1 or "
            "dynamic");
    }

    const auto& output = metadata.outputs.front();
    if (output.element_type != TensorElementType::Float32 || output.shape.size() != 2 ||
        (output.shape[0] != 1 && output.shape[0] >= 0) || output.shape[1] <= 0 ||
        static_cast<std::size_t>(output.shape[1]) != labels_.size()) {
        throw std::invalid_argument(
            "classification model output must be float32 [batch, class_count] matching labels");
    }
}

Tensor ImageClassificationPipeline::preprocess(std::span<const std::byte> encoded_image) const {
    if (encoded_image.empty() ||
        encoded_image.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("encoded image is empty or too large");
    }

    const auto* encoded = reinterpret_cast<const stbi_uc*>(encoded_image.data());
    const auto encoded_size = static_cast<int>(encoded_image.size());
    int source_width = 0;
    int source_height = 0;
    int source_channels = 0;
    if (stbi_info_from_memory(encoded, encoded_size, &source_width, &source_height,
                              &source_channels) == 0) {
        throw std::invalid_argument("request body is not a supported JPEG or PNG image");
    }
    (void)source_channels;

    if (source_width <= 0 || source_height <= 0 ||
        static_cast<std::size_t>(source_width) >
            config_.max_image_pixels / static_cast<std::size_t>(source_height)) {
        throw std::invalid_argument("image dimensions exceed configured safety limit");
    }

    int decoded_width = 0;
    int decoded_height = 0;
    using ImagePtr = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    ImagePtr pixels(
        stbi_load_from_memory(encoded, encoded_size, &decoded_width, &decoded_height, nullptr, 3),
        &stbi_image_free);
    if (!pixels || decoded_width != source_width || decoded_height != source_height) {
        throw std::invalid_argument("could not decode request image");
    }

    const auto shorter = static_cast<std::size_t>(std::min(source_width, source_height));
    const auto resize_scale =
        static_cast<double>(config_.resize_shorter) / static_cast<double>(shorter);
    const auto resized_width =
        scaled_dimension(static_cast<std::size_t>(source_width), resize_scale);
    const auto resized_height =
        scaled_dimension(static_cast<std::size_t>(source_height), resize_scale);
    const auto crop_left = (resized_width - config_.width) / 2;
    const auto crop_top = (resized_height - config_.height) / 2;
    const auto plane_size = config_.width * config_.height;
    std::vector<float> values(3 * plane_size);

    for (std::size_t y = 0; y < config_.height; ++y) {
        for (std::size_t x = 0; x < config_.width; ++x) {
            const auto pixel = y * config_.width + x;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const auto raw =
                    bilinear_channel(pixels.get(), source_width, source_height, resized_width,
                                     resized_height, crop_left, crop_top, x, y, channel);
                const auto normalized =
                    (raw / 255.0F - config_.mean[channel]) / config_.standard_deviation[channel];
                values[channel * plane_size + pixel] = normalized;
            }
        }
    }

    return Tensor::from_float32(
        {1, 3, static_cast<std::int64_t>(config_.height), static_cast<std::int64_t>(config_.width)},
        values);
}

std::vector<Prediction> ImageClassificationPipeline::postprocess(const Tensor& scores,
                                                                 std::size_t top_k) const {
    if (scores.element_type() != TensorElementType::Float32 || scores.shape().size() != 2 ||
        scores.shape()[0] != 1 || static_cast<std::size_t>(scores.shape()[1]) != labels_.size()) {
        throw std::invalid_argument("classification scores must have shape [1, label_count]");
    }
    if (top_k == 0) {
        throw std::invalid_argument("top_k must be greater than zero");
    }

    const auto values = scores.float32_values();
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("model returned a non-finite classification score");
    }

    const auto maximum = *std::max_element(values.begin(), values.end());
    std::vector<double> probabilities(values.size());
    double sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        probabilities[index] = std::exp(static_cast<double>(values[index] - maximum));
        sum += probabilities[index];
    }
    for (auto& probability : probabilities) {
        probability /= sum;
    }

    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    const auto result_count = std::min(top_k, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(result_count),
                      indices.end(), [&probabilities](std::size_t left, std::size_t right) {
                          return probabilities[left] > probabilities[right];
                      });

    std::vector<Prediction> predictions;
    predictions.reserve(result_count);
    for (std::size_t rank = 0; rank < result_count; ++rank) {
        const auto class_id = indices[rank];
        predictions.push_back(
            Prediction{class_id, labels_[class_id], static_cast<float>(probabilities[class_id])});
    }
    return predictions;
}

}  // namespace inference
