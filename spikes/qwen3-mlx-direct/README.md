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

The third vertical slice extends the same boundary through sinusoidal
positions, the ragged block-attention mask, all 18 audio transformer layers,
and the final 1,024-dimensional audio features. The pinned fixture produces
202 audio tokens and the exact reference windows `[0, 104, 202]`.

The fourth vertical slice owns the Qwen byte-level BPE vocabulary and merge
table, builds the exact ASR chat prompt, dequantizes the affine 8-bit token
embeddings with official MLX, and replaces every audio-pad embedding with the
corresponding direct encoder feature. No Transformers tokenizer is required by
the direct runtime.

The fifth vertical slice executes all 28 quantized Qwen3 text layers with
grouped-query attention, RoPE, SwiGLU, tied output projection, and a
repository-owned growable KV cache. Prompt prefill and the first two greedy
decode positions match the pinned Python reference.

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

Replace `frontend` with `encoder` to run all 18 audio layers and the final
projection:

```sh
DYLD_LIBRARY_PATH=/private/tmp/cuttledoc-qwen3-mlx-direct-build \
  /private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect \
  encoder \
  /absolute/path/to/Qwen3-ASR-0.6B-8bit \
  /absolute/path/to/fixture.f32le \
  gpu
```

Run the complete tokenizer/prompt/audio-embedding merge:

```sh
DYLD_LIBRARY_PATH=/private/tmp/cuttledoc-qwen3-mlx-direct-build \
  /private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect \
  prompt \
  /absolute/path/to/Qwen3-ASR-0.6B-8bit \
  /absolute/path/to/fixture.f32le \
  en \
  gpu
```

Run decoder prompt prefill and the first two cached greedy positions:

```sh
DYLD_LIBRARY_PATH=/private/tmp/cuttledoc-qwen3-mlx-direct-build \
  /private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect \
  decoder \
  /absolute/path/to/Qwen3-ASR-0.6B-8bit \
  /absolute/path/to/fixture.f32le \
  en \
  gpu
```

Compare its JSON output with the pinned reference oracle:

```sh
node scripts/validate-qwen3-mlx-frontend.mjs \
  --oracle \
  benchmarks/oracles/qwen3-asr-0.6b.audiobook-en-2277-149874-0000.encoder.json \
  --actual /absolute/path/to/direct-frontend-result.json
```

The prompt boundary has a separate exact validator:

```sh
node scripts/validate-qwen3-mlx-prompt.mjs \
  --oracle \
  benchmarks/oracles/qwen3-asr-0.6b.audiobook-en-2277-149874-0000.prompt-en.json \
  --actual /absolute/path/to/direct-prompt-result.json
```

The decoder boundary has a separate validator for cache state, logits, and
greedy decisions:

```sh
node scripts/validate-qwen3-mlx-decoder.mjs \
  --oracle \
  benchmarks/oracles/qwen3-asr-0.6b.audiobook-en-2277-149874-0000.decoder-en.json \
  --actual /absolute/path/to/direct-decoder-result.json
```

The validator requires exact shapes, feature length, chunk boundaries, sample
positions, encoder length, and attention windows. It allows `2e-6` sampled
absolute error through `conv_out`, `1e-5` through the full encoder, and `2e-5`
relative error for aggregate statistics because the Transformers reference and
direct MLX path use different FFT implementations.

## Next parity gates

1. Require exact full-transcript parity on one fixture.
2. Run the multilingual audiobook matrix.
