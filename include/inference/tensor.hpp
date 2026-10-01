#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace inference {

enum class TensorElementType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Bool,
    String,
    Unsupported,
};

[[nodiscard]] inline std::string_view to_string(TensorElementType type) noexcept {
    switch (type) {
        case TensorElementType::Float32: return "float32";
        case TensorElementType::Float64: return "float64";
        case TensorElementType::Float16: return "float16";
        case TensorElementType::BFloat16: return "bfloat16";
        case TensorElementType::Int8: return "int8";
        case TensorElementType::UInt8: return "uint8";
        case TensorElementType::Int16: return "int16";
        case TensorElementType::UInt16: return "uint16";
        case TensorElementType::Int32: return "int32";
        case TensorElementType::UInt32: return "uint32";
        case TensorElementType::Int64: return "int64";
        case TensorElementType::UInt64: return "uint64";
        case TensorElementType::Bool: return "bool";
        case TensorElementType::String: return "string";
        case TensorElementType::Unsupported: return "unsupported";
    }
    return "unsupported";
}

[[nodiscard]] inline std::size_t element_size(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float32:
        case TensorElementType::Int32:
        case TensorElementType::UInt32: return 4;
        case TensorElementType::Float64:
        case TensorElementType::Int64:
        case TensorElementType::UInt64: return 8;
        case TensorElementType::Float16:
        case TensorElementType::BFloat16:
        case TensorElementType::Int16:
        case TensorElementType::UInt16: return 2;
        case TensorElementType::Int8:
        case TensorElementType::UInt8:
        case TensorElementType::Bool: return 1;
        case TensorElementType::String:
        case TensorElementType::Unsupported:
            throw std::invalid_argument("string and unsupported tensor types are not byte tensors");
    }
    throw std::invalid_argument("unknown tensor element type");
}

class Tensor {
public:
    using Shape = std::vector<std::int64_t>;

    Tensor(TensorElementType element_type, Shape shape, std::vector<std::byte> data)
        : element_type_(element_type), shape_(std::move(shape)), data_(std::move(data)) {
        const auto expected_bytes = checked_byte_size(element_type_, shape_);
        if (data_.size() != expected_bytes) {
            throw std::invalid_argument("tensor data size does not match its shape and type");
        }
    }

    [[nodiscard]] static Tensor from_float32(Shape shape, const std::vector<float>& values) {
        const auto count = element_count(shape);
        if (values.size() != count) {
            throw std::invalid_argument("float32 value count does not match tensor shape");
        }

        std::vector<std::byte> data(values.size() * sizeof(float));
        if (!data.empty()) {
            std::memcpy(data.data(), values.data(), data.size());
        }
        return Tensor(TensorElementType::Float32, std::move(shape), std::move(data));
    }

    [[nodiscard]] TensorElementType element_type() const noexcept { return element_type_; }
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::byte>& data() const noexcept { return data_; }
    [[nodiscard]] std::size_t byte_size() const noexcept { return data_.size(); }
    [[nodiscard]] std::size_t element_count() const { return element_count(shape_); }

    [[nodiscard]] std::vector<float> float32_values() const {
        if (element_type_ != TensorElementType::Float32) {
            throw std::invalid_argument("tensor is not float32");
        }
        std::vector<float> values(element_count());
        if (!data_.empty()) {
            std::memcpy(values.data(), data_.data(), data_.size());
        }
        return values;
    }

    [[nodiscard]] static std::size_t element_count(const Shape& shape) {
        std::size_t count = 1;
        for (const auto dimension : shape) {
            if (dimension <= 0) {
                throw std::invalid_argument("tensor dimensions must be positive");
            }
            const auto unsigned_dimension = static_cast<std::size_t>(dimension);
            if (count > std::numeric_limits<std::size_t>::max() / unsigned_dimension) {
                throw std::overflow_error("tensor element count overflows size_t");
            }
            count *= unsigned_dimension;
        }
        return count;
    }

private:
    [[nodiscard]] static std::size_t checked_byte_size(TensorElementType type,
                                                        const Shape& shape) {
        const auto count = element_count(shape);
        const auto width = element_size(type);
        if (count > std::numeric_limits<std::size_t>::max() / width) {
            throw std::overflow_error("tensor byte size overflows size_t");
        }
        return count * width;
    }

    TensorElementType element_type_;
    Shape shape_;
    std::vector<std::byte> data_;
};

}  // namespace inference
