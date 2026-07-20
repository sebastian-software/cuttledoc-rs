# Direct Qwen3-ASR adapter over official MLX

**Status:** Issue
[#17](https://github.com/sebastian-software/cuttledoc-rs/issues/17) is in
progress. The model-artifact boundary and the direct 128-Mel/Conv2d path are
complete. All 18 audio transformer layers also match the reference oracle;
tokenizer and decoder parity remain open.

**Runnable artifact:**
[`spikes/qwen3-mlx-direct`](../../spikes/qwen3-mlx-direct/).

## Why this path exists

Qwen3-ASR 0.6B produced the strongest measured new-model signal in the first
multilingual matrix and remains an important third candidate beside Apple
Speech and the Whisper/CoreML fallback. The accepted architecture does not
require a generic Rust port of MLX. Rust can call a repository-owned,
model-specific C++ task adapter that uses the official MLX core directly.

The Python `mlx-audio` implementation remains useful as a pinned reference
oracle. It is not needed in the product runtime and no object from it crosses
the Rust boundary.

## Milestone 1: exact model load

Commit `10641df1dd816b4e73f45697630cf8219ee9cc29` adds:

- a manifest for the exact original and converted model revisions plus nine
  SHA-256-pinned runtime files;
- official MLX safetensors support in the source build;
- a narrow inspection C ABI and Rust caller;
- checks for all 1,005 tensors, the BF16 audio tower, the affine 8-bit text
  modules, layer counts, and critical tensor shapes.

The committed raw record is
[`benchmarks/raw/phase0.qwen3-mlx-direct-model-load-1/result.json`](../../benchmarks/raw/phase0.qwen3-mlx-direct-model-load-1/result.json).

| Observation | Result |
| --- | ---: |
| MLX version | 0.32.0 |
| Model artifact | 1,006,229,426 bytes |
| Tensors | 1,005 |
| BF16 tensors | 808 |
| packed UInt32 tensors | 197 |
| affine 8-bit modules | 197 |
| mean inspection wall time, three warm filesystem-cache runs | 46.667 ms |
| maximum resident set during inspection | 17,973,248 bytes |
| shim dylib | 18,770,096 bytes |
| common MLX metallib | 130,164,152 bytes |

The small resident set is not a model-memory result. `load_safetensors` creates
lazy file-backed MLX arrays, and this milestone deliberately does not evaluate
them. It proves that the official C++ loader understands the converted
artifact and that the owned adapter sees exactly the architecture expected by
the reference implementation.

## Milestone 2: direct audio frontend and convolution stack

The task adapter now accepts mono 16-kHz float32 PCM through its C ABI,
computes the 128-bin Slaney-normalized frontend, applies all three Qwen3 Conv2d
layers and `conv_out`, and returns compact numerical fingerprints to Rust.
Safetensors loading remains CPU-bound in official MLX; GPU mode performs an
explicit MLX copy before evaluating the graph on Metal.

For the pinned 248,080-sample audiobook fixture, both direct device modes
produce the exact reference shape sequence and the exact 16-chunk layout
(fifteen 100-frame chunks plus one 50-frame chunk). The GPU parity validator
measured:

| Boundary | Shape | maximum sampled absolute error | maximum aggregate relative error |
| --- | --- | ---: | ---: |
| input features | `1 × 128 × 1550` | 2.3842e-7 | 1.5489e-6 |
| Conv2d 1 | `16 × 64 × 50 × 480` | 1.1176e-8 | 2.6301e-7 |
| Conv2d 2 | `16 × 32 × 25 × 480` | 3.4273e-7 | 4.9016e-7 |
| Conv2d 3 | `16 × 16 × 13 × 480` | 2.9802e-7 | 4.2105e-7 |
| `conv_out` | `16 × 13 × 896` | 1.0431e-6 | 3.0681e-7 |

The byte digests intentionally differ at the frontend: Transformers computes
its reference spectrogram via NumPy Float64/Complex64, whereas the owned path
uses MLX Float32/Complex64. The checked tolerances are narrow enough to expose
layout, padding, filter-bank, activation, or weight errors without pretending
that different FFT implementations must be byte-identical.

## Milestone 3: complete direct audio encoder

The direct C++ adapter now reproduces the sinusoidal positions, the Qwen
ragged block-attention layout, all 18 transformer layers, post-normalization,
and both output projections. The fixed clip produces exactly 202 tokens and
the reference attention boundaries `[0, 104, 202]`.

| Boundary | Shape | maximum sampled absolute error | maximum aggregate relative error |
| --- | --- | ---: | ---: |
| encoder input | `202 × 896` | 2.7418e-6 | 4.4689e-7 |
| encoder layer 0 | `1 × 202 × 896` | 3.3081e-6 | 1.4844e-6 |
| encoder layer 17 | `1 × 202 × 896` | 4.9323e-6 | 3.1458e-7 |
| projected audio features | `202 × 1024` | 9.5926e-8 | 1.6638e-6 |

One development parity run took 506.203 ms on Metal and reported a
1,212,305,644-byte MLX peak. The CPU ABI smoke test also completed, in
508.573 ms with a 488,174,416-byte peak. These single runs include lazy weight
materialization inside the probe and establish functional boundaries; they are
not yet the repeated end-to-end performance measurement.

## Remaining parity gates

1. Implement the Qwen tokenizer and prompt layout, replace audio-pad token
   embeddings, and run the 28-layer quantized decoder with a repository-owned
   KV cache.
2. Require exact transcript parity on one fixture before adding the direct
   adapter to the multilingual audiobook matrix.
3. Add task lifecycle, cancellation checkpoints, bounded streaming updates,
   memory measurements after materialization, and artifact pruning evidence.

The order matters: comparing encoder tensors first prevents frontend or
attention-layout errors from being misdiagnosed as generation or tokenizer
problems.
