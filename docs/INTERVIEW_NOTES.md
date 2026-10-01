# Interview notes

## Concepts to explain

1. **Inference server:** a process that keeps a trained model available and turns input requests into predictions.
2. **Model lifetime:** loading once avoids repeated disk reads, graph setup, and weight allocation on every request.
3. **ONNX:** a portable model graph format that describes operators, tensors, and weights.
4. **ONNX Runtime:** the runtime that loads an ONNX graph, optimizes it, and executes it through an Execution Provider.
5. **Execution Provider:** ONNX Runtime's backend for placing supported graph operations on a device/runtime such as CPU or CoreML.
6. **CPU inference:** portable execution; `--threads` controls the intra-op worker count, defaulting to two here.
7. **CoreML inference:** Apple's ML execution framework, selected through its ONNX Runtime provider. Provider selection alone does not prove CPU, GPU, or Neural Engine placement.
8. **CUDA:** NVIDIA's GPU platform, a future provider for a Linux/NVIDIA build. Apple Silicon does not support CUDA.
9. **Dynamic batching:** briefly collect individual compatible requests and run them together as one tensor batch.
10. **Latency and throughput:** batching can raise completed requests per second while requests wait longer to join a batch.
11. **Backpressure:** reject work when bounded capacity is exhausted instead of allowing memory and latency to grow without limit.
12. **Bounded queue:** a mutex-protected deque whose capacity is fixed; `try_push` gives producers an immediate overload decision.
13. **Mutex:** protects the deque and its closed flag so concurrent producers and the consumer see consistent state.
14. **Condition variable:** lets the worker sleep until work or shutdown, avoiding polling and busy waiting.
15. **Producer/consumer:** HTTP workers produce preprocessed requests; one inference worker consumes them and executes batches.
16. **Ownership:** queue values own their tensors and promises; the HTTP handler owns the future and waits for its result. The runner outlives the worker callback.
17. **Preprocessing:** resize/crop and channel normalization must match the model's training/export convention or predictions become unreliable.
18. **Generic runner vs specific pipeline:** ONNX tensor execution is reusable; image decoding, normalization, labels, and postprocessing depend on the model task.
19. **ResNet vs MobileNet:** both classify images, but ResNet is a deeper residual architecture while MobileNet is designed for lower compute; this repository's MobileNet export also fixes batch size to one.
20. **This implementation's boundary:** one model, one inference worker, one image per HTTP request, aggregate metrics, no CUDA, and no claim of production scale.

## Interview questions with short answers

**Why is the queue bounded, and what happens when it fills?**  It caps queued memory and makes overload explicit. `try_push` fails immediately and `/predict` returns 503 with `Retry-After`.

**Why use a mutex and condition variable? Why not a lock-free queue?**  A mutex is easy to audit for this small multi-producer queue; the condition variable parks the consumer until work arrives. A lock-free queue adds complexity and is only justified after profiling shows this queue is a bottleneck.

**Why not use one thread per request?**  That creates unbounded scheduling and memory pressure and makes access to the model harder to control. A fixed HTTP pool feeds one clear inference owner.

**What happens as batch size grows?**  It can improve device utilization and throughput, but increases wait time and tensor memory. A deadline bounds how long the first request waits.

**How is request/result mapping preserved?**  Requests are appended in queue order; input data is concatenated in that order; output slices are split by the same indices and each promise is fulfilled accordingly.

**How does shutdown work?**  Queue close wakes the worker. It drains already accepted items, then exits when the closed queue is empty; the owner joins it before destroying the model runner.

**Why keep the model loaded? Why ONNX Runtime?**  It avoids loading and initializing weights per request. ONNX Runtime provides model metadata, graph optimizations, and swappable execution providers behind a stable C++ API.

**Why does CoreML differ from CUDA?**  CoreML is Apple's ML execution framework and can choose placement internally. CUDA is NVIDIA's GPU programming/runtime stack and requires NVIDIA hardware and a compatible Linux runtime in this target setup.

**How would you add an NVIDIA server?**  Install a CUDA-enabled ONNX Runtime build, append its CUDA provider when requested, verify provider availability, and run controlled tests on NVIDIA hardware. Keep provider setup isolated from preprocessing and queue logic.

**What are likely bottlenecks, and how would you profile them?**  Decode/preprocess, queue delay, batch formation, model execution, or HTTP serialization. Compare the existing timing metrics, use a CPU profiler and system counters, then vary one setting at a time.

**How could this serve multiple models?**  Give each model an explicit loaded session and queue/batcher with routing keyed by model name, then bound memory and scheduling across them. Do not share mutable request state implicitly.

**How would NVIDIA Triton differ?**  Triton is a mature multi-model serving system with broader protocol, scheduling, and deployment features. This project makes a small subset of those mechanisms visible for learning.

**What would you check before claiming better throughput?**  Repeat on the same model, image set, provider, thread count, concurrency, and duration; report tail latency and average batch size alongside requests per second, and inspect resource use.

## Before listing this project on a resume

Be ready to walk through `BoundedQueue::try_push`/`pop`, the batch deadline in `DynamicBatcher::run`, output splitting, ONNX tensor metadata, preprocessing normalization, HTTP overload handling, and the limitations of the measured smoke runs. Do not describe CoreML as GPU or Neural Engine execution without device-placement evidence.
