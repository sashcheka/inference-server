# inference-server

A C++20 learning project for image-classification inference serving with ONNX Runtime. The goal is to make model lifetime, bounded work queues, dynamic batching, execution providers, and latency/throughput trade-offs visible in a small codebase.

The project is implemented in small milestones. The current version includes the generic ONNX runner and ImageNet image preprocessing; serving, dynamic batching, and benchmarking are being added next.

## Planned request flow

```mermaid
flowchart LR
    Client --> HTTP[HTTP handlers]
    HTTP --> Validate[Validate and preprocess image]
    Validate --> Queue[Bounded request queue]
    Queue --> Batcher[Dynamic batcher]
    Batcher --> Runner[Loaded ONNX Runtime session]
    Runner --> EP{Execution provider}
    EP --> CPU[CPU]
    EP --> CoreML[CoreML]
    EP --> FutureCUDA[Future CUDA]
    Runner --> Post[Classification postprocessing]
    Post --> Client
```

The intended server keeps one model session loaded for its lifetime. A classification adapter will handle image-specific preprocessing and labels; the model runner will remain generic over ONNX tensor inputs and outputs.

## Current milestone

The repository contains a bounded queue, a generic ONNX Runtime C++ runner, and a model-specific ImageNet classification pipeline. The runner reads tensor names, shapes, and element types from the ONNX model once at startup, then keeps the session alive for subsequent inference calls. The pipeline decodes JPEG/PNG, resizes and normalizes images into NCHW tensors, then maps logits to ImageNet labels. The queue is provider-independent; producers use `try_push`, and closing it wakes consumers and drains accepted items.

```sh
./scripts/setup_onnxruntime.sh
cmake -S . -B build
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The setup script installs the pinned ONNX Runtime 1.30.0 C/C++ package under `.deps/onnxruntime`. The macOS arm64 package includes the CoreML Execution Provider; the Linux packages use CPU execution. Set `ONNXRUNTIME_ROOT` if the runtime is installed elsewhere. FetchContent pins nlohmann/json 3.12.0 and stb_image to a commit; GoogleTest is found locally or fetched at 1.17.0.

Download model files and ImageNet labels with:

```sh
./scripts/download_models.sh
```

The model files are kept out of git and verified against SHA-256 before use. Each model's configuration lives in `config/`; preprocessing options are deliberately kept out of the generic `ModelRunner`.

## Planned dependencies

- ONNX Runtime C++ API for model loading and inference.
- `cpp-httplib` for the small HTTP server and benchmark client.
- `nlohmann/json` for model configuration and JSON responses.
- CLI11 for command-line options.
- GoogleTest for unit tests.
- `stb_image` for JPEG/PNG decoding and model-specific resizing/normalization.
- CMake and the C++ standard library for the build and concurrency primitives.

Model sources, licenses, input sizes, normalization, and checksums are documented in `models/README.md` and `scripts/download_models.sh`.
