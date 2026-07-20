# Direct official MLX result (#6)

**Status:** #6 and #15 complete; advance official MLX as the third first-class
inference foundation. Broader selection evidence remains in #4 and #12.

**Evidence date:** 2026-07-17.

**Runnable artifact:** [`spikes/mlx-direct`](../../spikes/mlx-direct/).

## Decision

Cuttledoc can build directly on the official MLX C++ core through a small
repository-owned, task-level adapter. The harder native boundary is manageable
and is not a reason to reject one of the strongest Apple Silicon foundations.
There is no useful Rust-port moment here: Rust owns the product contract and
lifecycle, while the adapter owns MLX's model graph and native resources.

This result advances MLX rather than adopting it as the production ASR backend
today. The real audio frontend, encoder, tokenizer, autoregressive decoder, and
timestamp-token path now produce a complete transcript through Rust. Broader
multilingual quality, long-audio chunking, incremental updates, energy, clean
cold start, cancellation plumbing, and artifact pruning remain selection or
productization work. MLX also stays available for future text generation, TTS,
and other model families without exposing a generic tensor runtime in the
public API.

`mlx-c` is not the planned product route. The direct model implementation did
not expose an uncertainty that a second C wrapper would resolve, so it was not
added. Community Rust and Node wrappers remain reference-only.

## What “Rust talks directly to MLX” means

Rust already owns and invokes this MLX path. The word “direct” describes the
upstream dependency and ownership chain, not the absence of a language
boundary:

```text
safe Rust engine proxy
  -> five-function repository-owned C ABI
  -> repository-owned C++ task/model implementation
  -> pinned official MLX C++ core
```

Rust cannot declare MLX's C++ templates, overloads, namespaces, exceptions, and
standard-library types through its stable `extern "C"` FFI. It could use
`cxx`, `autocxx`, or generated bindings, but those approaches still generate or
compile C++ glue. More importantly, exposing MLX arrays and operators to Rust
would widen the unsafe and upgrade-sensitive surface without moving product
logic into Rust.

The small `extern "C"` surface is therefore both the fastest spike path and the
intended production seam:

- Rust owns PCM input, model identity, scheduling, cancellation checkpoints,
  errors, progress, domain results, and deterministic lifecycle.
- C++ owns MLX arrays, the lazy graph, model-specific kernels, exceptions, and
  device/runtime state.
- The ABI transfers only pointers with explicit ownership, scalar values,
  caller-owned PCM, and Rust-owned result/error copies.
- A future private `cuttledoc-mlx` Rust crate can make the five unsafe calls
  safe and ergonomic without becoming a general MLX binding or port.

Replacing the handwritten declarations with generated bindings later is a
tooling choice, not an architectural change. The boundary should widen only
when an end-to-end task requires a new product-level operation, never merely to
mirror another MLX operator.

## Proven path

The Rust probe passes real 16-kHz f32 PCM through an owned C ABI. The C++
session:

1. loads all 166 pinned Float16 Whisper Tiny tensors plus the official
   multilingual tokenizer vocabulary;
2. computes the official MLX Examples log-Mel frontend with its pinned filter;
3. runs both convolutions and all four attention/MLP encoder blocks;
4. runs four autoregressive decoder blocks with timestamp-token rules and
   decodes mergeable token bytes into UTF-8 text;
5. explicitly evaluates MLX's lazy graph with the official Whisper compute
   behavior: Float32 activations on CPU and Float16 on Metal;
6. returns Rust-owned transcript text, tokens, segment timestamps, timings, and
   memory values;
7. destroys the model arrays and clears the runtime cache.

The source model is
`mlx-community/whisper-tiny@78c52ab98ca87f570bc57ad852e15ef7060f9f76`.
Its 74,418,182-byte NPZ has SHA-256
`d5a3b8671ac7aab11a2c9d0f16e7da94bad5500d785856f438c6bd44c3723944`.
The extracted model tensors plus filter occupy 74,454,976 bytes. The official
multilingual vocabulary is pinned separately with SHA-256
`b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126`.
The pinned model card records conversion from OpenAI Whisper Tiny with the
official MIT MLX Examples converter. It omits a structured license field, so a
production model manifest should preserve the MIT source/converter chain
explicitly rather than inferring metadata from the hosting repository.

## Measurements

The evidence host was an M1 Ultra with 64 GB RAM, macOS 26.5.2, Xcode 26.6,
CMake 4.4.0, and a macOS 14 deployment target. The fixture was the same
10.56-second FLEURS sample used by the legacy and Apple Speech baselines.
Each release/device combination ran three create/run/destroy sessions and two
encoder evaluations per session.

| MLX | Device | First load | Median second run | Encoder RTF | MLX peak | Process memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.31.2 | CPU FP32 | 38.9 ms | 119.1 ms | 0.0113 | 132.4 MB | 215.0 MB max RSS |
| 0.31.2 | GPU FP16 | 47.7 ms | 11.6 ms | 0.00110 | 229.0 MB | 246.8 MB peak footprint |
| 0.32.0 | CPU FP32 | 36.4 ms | 115.1 ms | 0.0109 | 132.4 MB | 213.7 MB max RSS |
| 0.32.0 | GPU FP16 | 48.8 ms | 11.8 ms | 0.00111 | 229.0 MB | 246.1 MB peak footprint |

The first GPU inference varied with the process-external Metal kernel cache:
533.0 ms in the recorded 0.31.2 process and 27.6 ms in the already primed
0.32.0 process. Later runs were 11.6–16.0 ms. These are not comparable
clean-system cold-start measurements. Encoder timing is also not
end-to-end transcription timing and cannot be compared directly with ASR WER,
first-token latency, or final-result timing.

The timing spread between the two release runs is ordinary single-host noise;
the upgrade conclusion is source/output compatibility, not a performance win.

All repeated CPU fingerprints were identical, all repeated GPU fingerprints
were identical, and the fingerprint for each device was unchanged between MLX
0.31.2 and 0.32.0. The Metal FP16 result exactly matched the pinned official
MLX Examples Whisper graph used as a test oracle. CPU FP32 matched within
last-bit reduction differences. The upgrade required zero adapter source
changes.

### End-to-end ASR measurement

The #15 extension used MLX 0.32.0 and ran three create/transcribe/destroy
lifecycles with two transcriptions each. All six CPU and all six GPU results
were identical:

> However, due to the slow communication channels, styles in the West could
> lag behind by 25 to 30 years.

After benchmark normalization this matches the fixture reference except for
the reference's singular final word `year`, giving 5.26% WER and 1.23% CER.
The decoder returned two timestamp-token segments, 0.00–7.24 seconds and
7.24–8.62 seconds. The pinned official `mlx-whisper` 0.4.3 Python reference
returned identical text and the same 7.24-second boundary; its final boundary
was 8.52 seconds. The 100-ms final-boundary difference is retained as evidence:
the C++ spike recomputes full decoder context instead of using the reference's
FP16 KV-cache path, so text equivalence does not imply last-bit timestamp-logit
equivalence.

| Device | First complete run | Median warm complete run | Warm RTF | MLX peak |
| --- | ---: | ---: | ---: | ---: |
| CPU FP32 | 625.4 ms | 599.2 ms | 0.0567 | 191.5 MB |
| GPU FP16 | 179.0 ms | 115.8 ms | 0.0110 | 285.8 MB |

The repeated-process maximum resident set sizes were 462.0 MB for CPU and
124.7 MB for GPU. Those process values include the Rust probe and dynamically
loaded shim; MLX's allocator peak is the more stable within-session measure.
Warm GPU decoder time was approximately 101–105 ms in the stable runs. This
non-cached decoder is already fast enough to prove the product boundary, but a
production backend should add a bounded KV cache before treating it as final
performance evidence.

## Concrete integration findings

- NPY `Load` evaluates only on CPU. A GPU session must materialize each tensor
  on CPU and explicitly copy it to Metal.
- The source-built CPU FP16 convolution diverged at the first convolution even
  though its log-Mel input matched the reference. MLX Whisper's supported FP32
  CPU path matched the official graph, so the adapter uses CPU FP32 and Metal
  FP16 rather than pretending the two backends share one compute dtype.
- MLX's default device is global. Although every substantial operation receives
  the intended device, operator overloads use the default; the current adapter
  serializes runtime entry with a mutex. Production concurrency needs either
  the same bounded rule or an audit that removes all default-device use.
- Destroying the session releases owned arrays, and `clear_cache()` returns
  cached allocator buffers. Three repeated lifecycle cycles completed on both
  devices.
- The release artifact needs an executable-relative RPATH and a colocated
  `mlx.metallib`; a successful static link alone is insufficient.
- With the macOS 14 deployment target, MLX 0.32.0 produced an 18,638,120-byte
  shim plus a 130,164,152-byte metallib. MLX 0.31.2 produced 18,598,072 and
  125,268,216 bytes respectively. Kernel/artifact pruning is still meaningful
  production work.
- The synchronous encoder cannot be preempted mid-graph. Cancellation is
  observable at the next chunk or decoder-step boundary.
- The task ABI now carries a complete transcript without adding any MLX array
  or operator handle. The future private Rust adapter needs safe ownership,
  typed errors, cancellation checks between decoder steps, and a worker proxy;
  it does not need a general MLX binding.

## Disposition and next proof

**Technical result:** works.

**Production boundary result:** continue. Keep the task-level C ABI, pinned
official source, explicit artifact manifest, process-serialized lifecycle, and
no public MLX types.

**ASR result:** works on the first real fixture. Do not use the 11.8-ms encoder
number as an ASR claim; the comparable warm end-to-end GPU result is about
115.8 ms. The next proof is breadth and product behavior: multilingual
fixtures, long-audio chunking, incremental result mapping, energy, clean cold
start, safe cancellation plumbing, KV-cache behavior, and artifact pruning.

The end-to-end machine-readable evidence is in
[`benchmarks/runs/mlx-whisper-e2e.fleurs-en-000.json`](../../benchmarks/runs/mlx-whisper-e2e.fleurs-en-000.json)
and its
[raw record](../../benchmarks/raw/phase0.mlx-whisper-e2e.fleurs-en-000-1/result.json).
The earlier
[encoder-only record](../../benchmarks/runs/mlx-whisper-encoder.fleurs-en-000.json)
is retained as the narrower #6 foundation measurement.
