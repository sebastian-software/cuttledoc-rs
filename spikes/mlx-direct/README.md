# Direct official MLX spike (#6)

This spike deliberately uses the official MLX C++ core, not `mlx-c` and not a
Rust/Node wrapper. It exposes one task-shaped C ABI rather than arrays,
operators, streams, or model-runtime objects:

```c
int32_t cuttledoc_mlx_project_audio(
    const float *audio, size_t audio_len,
    const float *weights, size_t weights_len,
    float *output, size_t output_len,
    int32_t device_kind, char **error_out);
```

The Rust example supplies one 576-sample audio frame and a `[576, 4]` float
projection matrix. The C++ shim copies both caller-owned buffers, asks MLX to
run `matmul` plus `tanh` on an explicit CPU or GPU device, evaluates the lazy
graph, and copies four scores back to Rust. This is an execution/lifecycle
probe for a representative dense audio-model block, **not** a transcription
model or a production inference API.

## Pinned upstream

- MLX tag: `v0.32.0`
- MLX commit: `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`

Clone that exact official source outside this repository, then run:

```sh
CUTTLEDOC_MLX_SOURCE_DIR=/absolute/path/to/mlx \
  bash scripts/run-mlx-direct-spike.sh
```

The script verifies the checkout commit before configuring a fresh temporary
build. It requires Xcode's Metal Toolchain (`xcrun metal`), CMake, and Rust.
It builds MLX from the supplied source with Metal enabled, builds the owned
shim, builds the Rust caller, and runs the same projection on CPU and GPU.

## Boundary rules

- The repository does not vendor MLX or add a Cargo MLX dependency.
- The ABI has Rust-owned primitive buffers and an owned error string only; no
  MLX object crosses it.
- Input/weights are copied before MLX's lazy graph can outlive the call.
- `mlx-c` can be used later only to answer a named comparison question. It is
  not this spike's integration path.

The remaining #6 work is recorded in the runtime matrix: a real model/weight
format, repeated lifecycle and memory measurements, output quality, artifact
packaging, and upgrade evidence across two official MLX releases.
