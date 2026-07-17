# Direct official MLX spike (#6)

This spike runs the complete Whisper Tiny audio frontend and encoder over the
official MLX C++ core. Rust owns the input and result; the adapter owns all MLX
arrays and exposes only a task-level C ABI:

```c
void *cuttledoc_mlx_whisper_encoder_create(
    const char *model_directory, int32_t device_kind, char **error_out);

int32_t cuttledoc_mlx_whisper_encoder_encode(
    void *handle, const float *audio, size_t audio_len,
    char **json_out, char **error_out);

void cuttledoc_mlx_whisper_encoder_destroy(void *handle);
```

The input is mono f32 PCM at 16 kHz. The adapter pads it to Whisper's 30-second
window, computes the pinned 80-bin log-Mel frontend, evaluates both
convolutions and all four self-attention/MLP encoder blocks, and returns an
owned summary of the `[1, 1500, 384]` encoder output. This is a real model path,
not yet an end-to-end transcriber. It follows the official Whisper execution
behavior: Float32 activations on CPU and Float16 on Metal.

## Pinned inputs

- MLX `v0.32.0`:
  `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`
- upgrade control MLX `v0.31.2`:
  `68cf2fddd8de5edd8ab3d926391772b2e2cedad8`
- `mlx-community/whisper-tiny`:
  `78c52ab98ca87f570bc57ad852e15ef7060f9f76`
- MLX Examples frontend/filter source:
  `796f5b53cab69a3d48a44233ce21aae889e94a08`

The model fetcher verifies every downloaded digest and extracts only the 66
encoder tensors plus `mel_80.npy`:

```sh
bash scripts/fetch-mlx-whisper-encoder-model.sh
```

Clone either exact official MLX release outside this repository, then run:

```sh
CUTTLEDOC_MLX_SOURCE_DIR=/absolute/path/to/mlx \
  bash scripts/run-mlx-direct-spike.sh
```

The runner requires Xcode's Metal Toolchain, CMake, Rust, FFmpeg, `curl`, and
`unzip`. It builds for macOS 14, statically links MLX into the shim, colocates
`mlx.metallib`, adds an executable-relative RPATH to the probe, normalizes the
real FLEURS fixture, and runs repeated CPU and GPU lifecycles. Set
`CUTTLEDOC_MLX_BUILD_DIR` to preserve the build cache.

## Boundary and lifecycle rules

- The repository neither vendors MLX nor adds an MLX Cargo dependency.
- No MLX array, operator, stream, model, or device handle crosses the C ABI.
- Model tensors are loaded from NPY on CPU because MLX does not implement the
  NPY Load primitive on GPU; GPU sessions explicitly copy owned tensors to
  Metal.
- The Metal FP16 fingerprint exactly matches the pinned official MLX Examples
  Whisper graph. The source-built CPU FP16 convolution did not; the adapter
  uses the official graph's reference-compatible FP32 CPU behavior.
- MLX's default device is process-global. The spike serializes create,
  encode, and destroy through one mutex to prevent cross-session device races.
- Destroy deletes all session-owned arrays and clears MLX's allocator cache.
- Inference is synchronous and can be cancelled only at the next task/chunk
  boundary.
- `mlx-c` remains an optional control for a named uncertainty. The real model
  path did not reveal one, so no second product dependency was added.

See [`docs/spikes/mlx-direct.md`](../../docs/spikes/mlx-direct.md) and the
[raw lifecycle evidence](../../benchmarks/raw/phase0.mlx-whisper-encoder.fleurs-en-000-1/result.json).
