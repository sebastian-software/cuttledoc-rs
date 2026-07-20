# Direct Qwen3-ASR adapter over official MLX (#17)

This spike advances Qwen3-ASR as the third first-class Apple-local ASR
candidate. It is a repository-owned task adapter over the official MLX C++
core, not a generic Rust MLX binding and not a production dependency on
`mlx-audio`.

The first committed vertical slice proves the artifact boundary:

1. `model-manifest.json` pins the original Qwen model revision, the exact
   8-bit MLX conversion, all runtime artifacts, and their SHA-256 digests.
2. `CuttledocQwen3MlxShim.cpp` opens `model.safetensors` with the official MLX
   C++ loader and validates the expected 1,005-tensor architecture, BF16 audio
   tower, and 197 affine 8-bit text modules.
3. `rust/main.rs` calls that implementation through a two-function C ABI,
   copies the JSON into Rust-owned memory, and releases the C allocation.

No MLX array, operator, device, or model object crosses the ABI. This slice does
not yet claim transcript parity.

The second vertical slice executes the repository-owned 128-bin
Whisper-compatible frontend and the complete Qwen3 Conv2d stack through the
official MLX C++ operators. It exposes only PCM input and a compact JSON
fingerprint through the C ABI. Both CPU and GPU modes are supported; the
safetensors primitive materializes on CPU and the adapter explicitly copies
the lazy arrays to Metal for GPU execution.

## Pinned inputs

- official MLX `v0.32.0`,
  `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`;
- official `Qwen/Qwen3-ASR-0.6B`,
  `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`;
- `mlx-community/Qwen3-ASR-0.6B-8bit`,
  `89e96d92ba34aca20b3e29fb10cc284097d1219f`.

The model and converted artifacts are Apache-2.0. Official MLX is MIT.
`mlx-audio` remains a disposable reference oracle under the repository
dependency policy.

## Reproduce the loader

Validate the complete local model directory first:

```sh
node scripts/validate-qwen3-mlx-model.mjs \
  --model-dir /absolute/path/to/Qwen3-ASR-0.6B-8bit
```

Configure the existing direct-MLX source build and compile only the Qwen shim:

```sh
cmake -S spikes/mlx-direct -B /private/tmp/cuttledoc-qwen3-mlx-direct-build \
  -DMLX_SOURCE_DIR=/absolute/path/to/mlx-v0.32.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build /private/tmp/cuttledoc-qwen3-mlx-direct-build \
  --target cuttledoc_qwen3_mlx_shim --parallel 2
```

Compile and run the Rust-owned probe:

```sh
rustc --edition 2024 spikes/qwen3-mlx-direct/rust/main.rs \
  -L native=/private/tmp/cuttledoc-qwen3-mlx-direct-build \
  -l dylib=cuttledoc_qwen3_mlx_shim \
  -C link-arg=-Wl,-rpath,@executable_path \
  -o /private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect

DYLD_LIBRARY_PATH=/private/tmp/cuttledoc-qwen3-mlx-direct-build \
  /private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect \
  /absolute/path/to/Qwen3-ASR-0.6B-8bit
```

Run the frontend/Conv2d parity probe on a mono 16-kHz float32 fixture:

```sh
DYLD_LIBRARY_PATH=/private/tmp/cuttledoc-qwen3-mlx-direct-build \
  /private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect \
  frontend \
  /absolute/path/to/Qwen3-ASR-0.6B-8bit \
  /absolute/path/to/fixture.f32le \
  gpu
```

Compare its JSON output with the pinned reference oracle:

```sh
node scripts/validate-qwen3-mlx-frontend.mjs \
  --oracle \
  benchmarks/oracles/qwen3-asr-0.6b.audiobook-en-2277-149874-0000.encoder.json \
  --actual /absolute/path/to/direct-frontend-result.json
```

The validator requires exact shapes, feature length, chunk boundaries, and
sample positions. It allows `2e-6` absolute error for sampled values and
`2e-5` relative error for aggregate statistics because the Transformers
reference and direct MLX path use different FFT implementations.

## Next parity gates

1. Match all 18 audio blocks and the 1,024-dimensional audio embeddings.
2. Implement tokenizer/prompt assembly, audio-token replacement, and the
   quantized 28-layer Qwen3 decoder with KV caching.
3. Require exact transcript parity on one fixture before running the
   multilingual audiobook matrix.
