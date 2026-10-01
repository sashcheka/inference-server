#include "inference/model_runner.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

namespace inference {
namespace {

[[nodiscard]] TensorElementType to_element_type(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return TensorElementType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return TensorElementType::Float64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return TensorElementType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return TensorElementType::BFloat16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return TensorElementType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return TensorElementType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return TensorElementType::Int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return TensorElementType::UInt16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return TensorElementType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return TensorElementType::UInt32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return TensorElementType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return TensorElementType::UInt64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return TensorElementType::Bool;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return TensorElementType::String;
        default: return TensorElementType::Unsupported;
    }
}

[[nodiscard]] ONNXTensorElementDataType to_onnx_element_type(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case TensorElementType::Float64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case TensorElementType::Float16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case TensorElementType::BFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        case TensorElementType::Int8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case TensorElementType::UInt8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case TensorElementType::Int16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case TensorElementType::UInt16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case TensorElementType::Int32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case TensorElementType::UInt32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case TensorElementType::Int64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case TensorElementType::UInt64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        case TensorElementType::Bool: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        case TensorElementType::String:
        case TensorElementType::Unsupported:
            throw std::invalid_argument("string and unsupported tensors are not implemented");
    }
    throw std::invalid_argument("unknown tensor element type");
}

[[nodiscard]] std::vector<TensorMetadata> inspect_values(const Ort::Session& session,
                                                           bool inputs) {
    Ort::AllocatorWithDefaultOptions allocator;
    const auto count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<TensorMetadata> values;
    values.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        auto name = inputs ? session.GetInputNameAllocated(index, allocator)
                           : session.GetOutputNameAllocated(index, allocator);
        auto type_info = inputs ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
        if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
            throw std::runtime_error("model inputs and outputs must be tensors in this project");
        }

        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        values.push_back(TensorMetadata{.name = name.get(),
                                        .element_type = to_element_type(tensor_info.GetElementType()),
                                        .shape = tensor_info.GetShape()});
    }
    return values;
}

[[nodiscard]] bool has_dynamic_batch(const std::vector<TensorMetadata>& tensors) {
    return !tensors.empty() && std::all_of(tensors.begin(), tensors.end(), [](const auto& tensor) {
        return !tensor.shape.empty() && tensor.shape.front() < 0;
    });
}

}  // namespace

ModelRunner::ModelRunner(ModelRunnerOptions options)
    : options_(std::move(options)), env_(ORT_LOGGING_LEVEL_WARNING, "inference-server") {
    if (options_.model_path.empty()) {
        throw std::invalid_argument("model path is required");
    }
    if (options_.intra_op_threads < 1 || options_.inter_op_threads < 1) {
        throw std::invalid_argument("ONNX Runtime thread counts must be positive");
    }
    if (options_.provider == ExecutionProvider::Cuda) {
        throw std::invalid_argument("CUDA provider is reserved for a future Linux/NVIDIA build");
    }

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(options_.intra_op_threads);
    session_options.SetInterOpNumThreads(options_.inter_op_threads);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (options_.provider == ExecutionProvider::CoreML) {
#ifdef __APPLE__
        const auto providers = Ort::GetAvailableProviders();
        if (std::find(providers.begin(), providers.end(), "CoreMLExecutionProvider") ==
            providers.end()) {
            throw std::runtime_error(
                "CoreML was requested, but this ONNX Runtime build does not include CoreML");
        }
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0));
#else
        throw std::runtime_error("CoreML is only available in the macOS build");
#endif
    }

    session_ = std::make_unique<Ort::Session>(env_, options_.model_path.c_str(), session_options);
    metadata_ = inspect_model();
}

ModelMetadata ModelRunner::inspect_model() const {
    ModelMetadata result;
    result.inputs = inspect_values(*session_, true);
    result.outputs = inspect_values(*session_, false);
    result.supports_dynamic_batch = has_dynamic_batch(result.inputs) &&
                                    has_dynamic_batch(result.outputs);
    return result;
}

void ModelRunner::validate_inputs(const std::vector<Tensor>& inputs) const {
    if (inputs.size() != metadata_.inputs.size()) {
        throw std::invalid_argument("input tensor count does not match model metadata");
    }

    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto& actual = inputs[index];
        const auto& expected = metadata_.inputs[index];
        if (actual.element_type() != expected.element_type) {
            throw std::invalid_argument("input tensor type does not match model metadata");
        }
        if (actual.shape().size() != expected.shape.size()) {
            throw std::invalid_argument("input tensor rank does not match model metadata");
        }
        for (std::size_t dim = 0; dim < actual.shape().size(); ++dim) {
            if (expected.shape[dim] >= 0 && actual.shape()[dim] != expected.shape[dim]) {
                throw std::invalid_argument("input tensor shape does not match model metadata");
            }
        }
    }
}

std::vector<Tensor> ModelRunner::infer(const std::vector<Tensor>& inputs) {
    validate_inputs(inputs);

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_values;
    input_values.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_values.push_back(Ort::Value::CreateTensor(
            memory_info, const_cast<std::byte*>(input.data().data()), input.byte_size(),
            input.shape().data(), input.shape().size(), to_onnx_element_type(input.element_type())));
    }

    std::vector<const char*> input_names;
    input_names.reserve(metadata_.inputs.size());
    for (const auto& input : metadata_.inputs) {
        input_names.push_back(input.name.c_str());
    }
    std::vector<const char*> output_names;
    output_names.reserve(metadata_.outputs.size());
    for (const auto& output : metadata_.outputs) {
        output_names.push_back(output.name.c_str());
    }

    auto output_values = session_->Run(Ort::RunOptions{nullptr}, input_names.data(),
                                       input_values.data(), input_values.size(),
                                       output_names.data(), output_names.size());
    std::vector<Tensor> outputs;
    outputs.reserve(output_values.size());
    for (const auto& output : output_values) {
        const auto info = output.GetTensorTypeAndShapeInfo();
        const auto type = to_element_type(info.GetElementType());
        const auto size = output.GetTensorSizeInBytes();
        std::vector<std::byte> data(size);
        if (size != 0) {
            std::memcpy(data.data(), output.GetTensorRawData(), size);
        }
        outputs.emplace_back(type, info.GetShape(), std::move(data));
    }
    return outputs;
}

}  // namespace inference
