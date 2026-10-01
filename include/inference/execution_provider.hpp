#pragma once

#include <stdexcept>
#include <string_view>

namespace inference {

enum class ExecutionProvider { Cpu, CoreML, Cuda };

[[nodiscard]] inline ExecutionProvider parse_execution_provider(std::string_view name) {
    if (name == "cpu") {
        return ExecutionProvider::Cpu;
    }
    if (name == "coreml") {
        return ExecutionProvider::CoreML;
    }
    if (name == "cuda") {
        return ExecutionProvider::Cuda;
    }
    throw std::invalid_argument("provider must be one of: cpu, coreml, cuda");
}

[[nodiscard]] inline std::string_view to_string(ExecutionProvider provider) noexcept {
    switch (provider) {
        case ExecutionProvider::Cpu: return "cpu";
        case ExecutionProvider::CoreML: return "coreml";
        case ExecutionProvider::Cuda: return "cuda";
    }
    return "unknown";
}

}  // namespace inference
