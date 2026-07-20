# Direct official MLX task adapter (#6, #15)

This spike runs Whisper Tiny end to end over the official MLX C++ core. Rust
owns the input, lifecycle, and result; the adapter owns all MLX arrays,
tokenizer data, and model-specific graph code. It exposes only a task-level C
ABI:

```c
void *cuttledoc_mlx_whisper_create(
    const char *model_directory, int32_t device_kind, char **error_out);

int32_t cuttledoc_mlx_whisper_transcribe(
    void *handle, const float *audio, size_t audio_len,
    const char *language, char **json_out, char **error_out);

void cuttledoc_mlx_whisper_destroy(void *handle);
```

The input is mono f32 PCM at 16 kHz plus an explicit ISO 639-1 language code.
The bounded spike surface accepts `en`, `de`, `es`, `fr`, and `pt`; unsupported
or absent values are errors rather than an implicit English fallback. The
adapter pads audio to Whisper's 30-second window, computes the pinned 80-bin
log-Mel frontend, evaluates both convolutions and all four audio blocks, then
runs the four-layer autoregressive text decoder with the official multilingual
vocabulary and timestamp-token rules. It returns Rust-owned transcript text,
tokens, segment times, timing, and memory evidence. CPU uses the official
reference-compatible Float32 path; Metal uses Float16.

## Pinned inputs

- MLX `v0.32.0`:
  `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`
- upgrade control MLX `v0.31.2`:
  `68cf2fddd8de5edd8ab3d926391772b2e2cedad8`
- `mlx-community/whisper-tiny`:
  `78c52ab98ca87f570bc57ad852e15ef7060f9f76`
- MLX Examples frontend/filter/tokenizer and decoder reference:
  `796f5b53cab69a3d48a44233ce21aae889e94a08`

The model fetcher verifies every downloaded digest and extracts all 166 model
tensors plus `mel_80.npy` and `multilingual.tiktoken`:

```sh
bash scripts/fetch-mlx-whisper-model.sh
```

Clone either exact official MLX release outside this repository, then run:

```sh
CUTTLEDOC_MLX_SOURCE_DIR=/absolute/path/to/mlx \
  bash scripts/run-mlx-direct-spike.sh
```

Select another fixture and its matching language explicitly:

```sh
CUTTLEDOC_MLX_SOURCE_DIR=/absolute/path/to/mlx \
  CUTTLEDOC_MLX_LANGUAGE=de \
  CUTTLEDOC_MLX_FIXTURE=../cuttledoc/packages/cuttledoc/fixtures/fleurs-de-000.ogg \
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
  transcribe, and destroy through one mutex to prevent cross-session device
  races.
- Destroy deletes all session-owned arrays and clears MLX's allocator cache.
- Encoder execution is synchronous; autoregressive decoding provides a natural
  cancellation checkpoint between tokens. Long-audio chunking and streaming
  updates remain production work.
- Language choice is Rust-owned task input. Only its numeric Whisper language
  token enters the model prefix; language detection remains a separate product
  capability decision.
- `mlx-c` remains an optional control for a named uncertainty. The real model
  path did not reveal one, so no second product dependency was added.

See [`docs/spikes/mlx-direct.md`](../../docs/spikes/mlx-direct.md), the
[end-to-end benchmark record](../../benchmarks/runs/mlx-whisper-e2e.fleurs-en-000.json),
and its
[raw lifecycle evidence](../../benchmarks/raw/phase0.mlx-whisper-e2e.fleurs-en-000-1/result.json).
