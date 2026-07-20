# Direct Qwen3-ASR adapter over official MLX

**Status:** Issue
[#17](https://github.com/sebastian-software/cuttledoc-rs/issues/17) acceptance
criteria are complete. The repository-owned path loads the pinned model,
executes the frontend, 18-layer audio encoder, prompt, 28-layer decoder and KV
cache directly on official MLX, reproduces the full pinned transcript, has
completed the 15-fixture multilingual audiobook pilot, and now exposes a
reusable Rust-owned task lifecycle with stable error and cancellation
semantics.

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

## Milestone 4: prompt and mixed audio/text embeddings

The repository-owned adapter now parses the pinned 151,643-entry Qwen
vocabulary and 151,387 BPE merges, applies the GPT/Qwen byte mapping, and
constructs the model's ASR chat prompt without Transformers. For the English
fixture it produces exactly the reference's 220 token IDs:

- nine prefix tokens through `<|audio_start|>`;
- 202 `<|audio_pad|>` tokens at positions 9 through 210;
- nine suffix tokens ending in `language en<asr_text>`.

It then gathers the packed 8-bit embedding rows, dequantizes them with
official `mx::dequantize`, casts the direct audio features to BF16, and
replaces the 202 placeholder rows. The token sequence, audio positions, and
all 24 sampled BF16 values across token embeddings, audio features, and merged
inputs are exact matches. The largest aggregate relative difference is
5.2991e-6 and comes from the already measured frontend FFT variance.

The Metal development probe took 381.677 ms after adapter construction and
reported a 1,766,258,540-byte MLX peak. This is still a parity probe with lazy
materialization, not the final repeated decoder performance result.

## Milestone 5: quantized decoder prefill and KV cache

The adapter now executes all 28 text transformer layers directly through
official MLX. It implements the converted model's affine 8-bit linear layers,
RMS normalization, grouped-query attention, RoPE with base 1,000,000, SwiGLU
MLPs, tied quantized output projection, and a repository-owned KV cache whose
capacity grows in 256-token steps.

For the pinned English fixture, the two-step probe proves:

- the 220-token prompt is split and cached with the same prefill semantics as
  the reference generator;
- the first-layer sampled key/value cache entries are exact after prefill;
- the first token is exactly `6217` and the second exactly `10810`;
- the first top-10 candidate set overlaps 10/10 and the second 9/10;
- the layer-27 maximum sampled difference is 0.3711 in BF16 values, while the
  largest aggregate relative difference across checked tensors is 3.0532%.

The deeper hidden-state values are not byte-identical because official MLX C++
and the Python reference reach different fused BF16 execution paths. The exact
greedy decisions, exact first-layer cache samples, and bounded deeper
fingerprints make the acceptance criterion explicit instead of hiding that
numerical distinction.

The fresh Metal parity run took 354.257 ms after adapter construction and
reported a 1,918,491,576-byte MLX peak. Top-candidate fingerprinting forces the
full vocabulary logits to materialize, so this remains a development probe
rather than a production memory measurement.

## Milestone 6: exact greedy transcript

The direct adapter now continues generation for up to 256 tokens, stops on
either Qwen EOS token, and decodes normal token IDs with its repository-owned
inverse byte-level vocabulary. No Transformers or `mlx-audio` tokenizer is
present in this runtime path.

On the pinned English audiobook fixture, the direct result matches all 61
generated reference tokens and the complete decoded transcript exactly:

> Minnie's Flat, as the one-floor resident apartments were then being called,
> was in a part of West Van Buren Street inhabited by families of laborers and
> clerks, men who had come and were still coming, with the rush of population
> pouring in at the rate of fifty thousand a year.

Generation stopped on `<|im_end|>` (`151645`) with the expected cache offset of
281. The single Metal development run took 772.659 ms after adapter
construction and reported a 1,904,298,424-byte MLX peak. It includes audio
encoding, prompt construction, model prefill, 61 cached decoder steps, and
token decoding, but is not yet a repeated performance sample.

## Milestone 7: multilingual audiobook pilot

The direct adapter completed all 45 fresh-process runs in the 15-fixture
audiobook matrix: one discarded warm-up and two measured repetitions for each
of three German, English, Spanish, French, and Portuguese clips. Every run was
deterministic within its fixture and stopped on `<|im_end|>`.

| Language | Direct macro WER | Reference macro WER | Direct macro CER |
| --- | ---: | ---: | ---: |
| German | 11.84% | 9.88% | 3.41% |
| English | 7.37% | 8.20% | 1.18% |
| Spanish | 12.34% | 12.34% | 1.73% |
| French | 3.47% | 3.47% | 1.40% |
| Portuguese | 19.76% | 21.78% | 5.74% |
| **Macro** | **10.96%** | **11.14%** | **2.69%** |

Twelve of 15 final texts are exact matches with the `mlx-audio` oracle. The
other three are stable greedy-decision differences after the measured BF16
drift: the direct path is worse against the unverified dataset transcript on
one German clip, better on one English and one Portuguese clip, and slightly
better in the aggregate. Therefore the broader result establishes behavioral
parity, not a false claim of byte-identical logits or transcripts on every
input.

The mean adapter-reported inference time was 750.066 ms with a 0.0510 mean
real-time factor. The maximum process RSS was 1,110,016,000 bytes and the
maximum MLX allocation counter was 2,554,761,832 bytes. These fresh-process
runs include lazy materialization in each inference and are not directly
comparable to the reference runner's repeated inference inside one initialized
engine.

The checked raw result is
[`benchmarks/raw/phase0.qwen3-mlx-direct.audiobook-pilot-1/result.json`](../../benchmarks/raw/phase0.qwen3-mlx-direct.audiobook-pilot-1/result.json).

## Milestone 8: Rust-owned lifecycle and cancellation

Commit `825ceaaed0ca6dfd97bc7a950cb1644df288a7ce` adds the narrow task
boundary required by the spike:

- `create` owns the model arrays and the official MLX stream context;
- `transcribe` accepts caller-owned PCM and returns copied JSON;
- `cancel` is thread-safe and cooperative;
- `destroy` waits for an active call before releasing native state; and
- stable status codes distinguish success (`0`), invalid arguments (`1`),
  runtime errors (`2`), cancellation (`3`), and concurrent use (`4`).

MLX 0.32 keeps normal default streams thread-local. The adapter therefore
creates one official thread-crossing CPU stream and one Metal stream, reuses
them process-wide, and protects them with the existing runtime mutex. One
transcription may be active per handle. This avoids pretending that lazy MLX
arrays created on one Rust worker can be evaluated on another without an
explicit stream boundary.

Three complete create/destroy cycles with two calls per handle produced six of
six exact matches with the pinned 61-token transcript. The same run measured:

| Observation | Result |
| --- | ---: |
| exact repeated transcripts | 6 / 6 |
| mean create | 14.100 ms |
| mean destroy | 26.238 ms |
| mean first call per handle | 711.022 ms |
| mean second call on the same handle | 583.393 ms |
| invalid-argument status | 1 |
| concurrent-use status | 4 |
| cancelled status | 3 |
| cancel signal to returned status | 238.522 ms |

The timing is development evidence, not a production throughput benchmark.
Cancellation does not preempt an in-flight Metal kernel or synchronous MLX
graph. The recorded call observed its atomic signal immediately after decoder
prefill; subsequent generation checks it after every decoder step.

The checked raw evidence is
[`benchmarks/raw/phase0.qwen3-mlx-direct.lifecycle-1/result.json`](../../benchmarks/raw/phase0.qwen3-mlx-direct.lifecycle-1/result.json).

## Follow-up product gates

1. Put the proven handle behind the common Rust engine abstraction and its
   awaited shutdown contract.
2. Extend the corpus with held-out audiobook and professional-podcast audio,
   German first, before setting language-specific selection thresholds.
3. Measure clean-host materialization, energy, and the pruned release artifact.
4. Keep batch/final-only output explicit. Incremental audio input, word
   timestamps, and forced alignment require separate capability work.

The order matters: comparing encoder tensors first prevents frontend or
attention-layout errors from being misdiagnosed as generation or tokenizer
problems.
