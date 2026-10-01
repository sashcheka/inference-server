#pragma once

#include "inference/tensor.hpp"

#include <string>
#include <vector>

namespace inference {

struct TensorMetadata {
    std::string name;
    TensorElementType element_type = TensorElementType::Unsupported;
    // A negative dimension denotes a dynamic dimension, as reported by ONNX Runtime.
    Tensor::Shape shape;
};

struct ModelMetadata {
    std::vector<TensorMetadata> inputs;
    std::vector<TensorMetadata> outputs;
    bool supports_dynamic_batch = false;
};

}  // namespace inference
