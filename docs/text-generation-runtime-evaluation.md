# Apple-local text-generation runtime evaluation

**Status:** issue #7 complete; embedded transcript enhancement deferred from
the initial v3 scope.

**Evidence date:** 2026-07-21.

## Outcome

Official MLX is the preferred foundation for a future embedded Apple-local
text-generation backend, but Cuttledoc should not ship an embedded enhancement
model in the initial v3 release merely because the runtime works. The pinned
Qwen3 0.6B experiment proves model load, deterministic autoregressive
generation, token streaming, cooperative cancellation, and modest delivery
cost. The same run also proves why runtime viability and correction quality are
separate gates: the model violated a lexical invariant and doubled the
development fixture's content error count.

The practical decision is therefore:

1. keep raw transcription complete and usable without enhancement;
2. keep `TextGenerationEngine` independent of STT, TTS, and inference runtime;
3. allow OpenAI and Ollama adapters to be implemented independently behind
   that contract;
4. defer an embedded backend until a model and prompt pass multilingual,
   held-out transcript-enhancement gates; and
5. start that future embedded bakeoff with a narrow repository-owned adapter
   over official MLX, with llama.cpp/GGUF retained as the credible fallback.

This is an explicit scope deferral, not a rejection of MLX.

## Candidate review

The maintenance signals below are a dated snapshot, not a popularity contest.
Licensing, API width, model ownership, release discipline, platform focus,
delivery cost, and replaceability all contribute to the disposition.

| Candidate | 2026-07-21 evidence | Model and delivery implications | Disposition |
| --- | --- | --- | --- |
| [Official MLX](https://github.com/ml-explore/mlx) plus an owned model adapter | Apple-maintained MIT project; about 27.6k stars; v0.32.0 released 2026-07-07; current macOS/Metal work. [MLX-LM](https://github.com/ml-explore/mlx-lm) is the official reference implementation, with about 6.4k stars and v0.31.3. | MLX safetensors fit the existing verified model-cache design and reuse the already accepted MLX runtime foundation. Cuttledoc still owns the model graph, KV cache, task ABI, lifecycle, and upgrade tests. | Preferred future embedded foundation. `mlx-lm` and Python remain reference-only; product code talks to official MLX through a narrow owned C++/C ABI. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Mature MIT project; about 121k stars; frequent numbered builds; Apple Silicon/Metal, GGUF, a C API, CLI, and local server are first-class paths. | Strong model availability and portable delivery, but GGUF creates a second model format and cache alongside the MLX safetensors already needed by Cuttledoc. It also adds another native runtime to build, sign, test, and update. | Credible fallback. Promote if cross-platform embedded inference or GGUF distribution becomes more valuable than one Apple-local runtime foundation. |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Active MIT Rust project; about 7.5k stars; v0.9.0 released 2026-07-07; Metal, safetensors, GGUF/UQFF, Rust SDK, servers, multimodal and agent features. | Capable, but its broad runtime and feature surface are disproportionate to Cuttledoc's narrow correction task. Adoption would add a large second ownership and dependency surface instead of reusing official MLX. | Reference only. Re-evaluate only if its Rust-native lifecycle or supported model produces a measured advantage large enough to pay for the surface. |
| [Candle](https://github.com/huggingface/candle) | Established Apache-2.0/MIT Rust framework; about 20.7k stars; active Metal kernels and safetensors/GGUF support; no GitHub release stream at the snapshot. | Cuttledoc would own or closely track model-specific implementations as well as the runtime integration. Current model coverage is not a stable task-level compatibility promise. | Reference only. Useful implementation prior art, not an initial embedded dependency. |
| [Core ML](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) | Apple supports stateful models and KV-cache-style state on current platforms; the framework itself is mature and system-delivered. | Requires a maintained converted model, conversion validation, compiled artifacts, and model-specific state mapping. No selected instruction model currently gains a measured deployment advantage over MLX. | Deferred. Reconsider only for a concrete converted model that proves a size, energy, Neural Engine, or deployment benefit. |
| Ollama HTTP adapter | Existing Cuttledoc 2 behavior and a pragmatic local-process provider boundary. | Ollama owns process lifecycle and its model cache. Cuttledoc owns provider configuration, request cancellation, errors, and capability mapping, but does not embed the runtime. | Independently implementable local provider; not the embedded runtime decision. |
| OpenAI HTTP adapter | Remote model access without local native runtime or model delivery. | No local model cache. Credentials, privacy, cost, network errors, rate limits, cancellation, and opaque model revisions remain provider concerns. | Independently implementable remote provider; not coupled to Ollama or an embedded backend. |

No weak project enters a Cargo, npm, native build, or release manifest as a
result of this evaluation. The official MLX foundation was already accepted;
the experiment adds only a locked, disposable Python reference environment.

## Pinned experiment

The runnable probe is in
[`spikes/text-generation-mlx-reference`](../spikes/text-generation-mlx-reference/README.md).
It pins:

- `Qwen/Qwen3-0.6B` and the exact
  `mlx-community/Qwen3-0.6B-4bit` conversion, both Apache-2.0;
- a 351,383,618-byte verified MLX model snapshot;
- official MLX 0.32.0 and MLX-LM 0.31.3;
- greedy, non-thinking generation with seed zero;
- `surface-only-v1` by SHA-256; and
- one German professionally narrated audiobook development fixture whose
  evaluation reference is never rendered into the prompt.

The conversion identifies the official Qwen model family but does not name
the exact upstream source commit it converted. The manifest therefore pins an
observed official source revision for capability and license evidence without
claiming that it was the conversion input.

The model was intentionally small enough to make runtime integration cheap.
It is not proposed as the production correction model.

| Measurement | Result |
| --- | ---: |
| Fresh-process model load, without clearing OS/Metal caches | 428.4 ms |
| First generated token | 197.1 ms |
| Complete generation | 431.8 ms |
| Generation throughput | 270.1 tokens/s |
| Prompt / generated tokens | 269 / 61 |
| Process peak resident size | 1,028,030,464 bytes |
| MLX peak allocation | 669,108,360 bytes |
| Model snapshot | 351,383,618 bytes |
| Deterministic repeat | identical text and token IDs |
| Cooperative cancellation | stopped after four tokens in 158.6 ms; process remained usable |

The output changed `fielen` to `fiel` despite a hard no-lexical-change rule.
The external gate rejected it. On the unverified development transcript, WER
moved from one error in 39 words (2.56%) to two (5.13%). The model also failed
to add useful punctuation. This is not a population-quality result, but it is
enough to reject this exact model/prompt tuple and demonstrate that prompt
instructions alone are not a safety boundary.

The complete machine-readable result is
[`phase5.qwen3-0.6b-4bit-mlx-reference.issue7-de-audiobook-whisper-1.json`](../benchmarks/postprocessing/runs/phase5.qwen3-0.6b-4bit-mlx-reference.issue7-de-audiobook-whisper-1.json).

## Model-format and cache consequences

| Backend | Artifact ownership | Cache consequence |
| --- | --- | --- |
| Official MLX adapter | Cuttledoc manifest plus immutable MLX safetensors | Reuses the existing download, digest, rollback, and MLX runtime strategy. Model weights remain task/model-specific; “shared runtime” does not imply shared weights. |
| llama.cpp | Cuttledoc or provider-managed GGUF | A second representation is normally required even for the same source model. Do not silently keep both MLX and GGUF copies. |
| mistral.rs / Candle | Cuttledoc-managed safetensors, GGUF, or runtime-specific quantization | Format flexibility increases the compatibility and validation matrix rather than removing it. |
| Core ML | Cuttledoc-managed compiled model plus conversion provenance | Separate platform-specific artifacts, conversion toolchain, and rollback evidence. |
| Ollama | Provider-managed | Discover capability and model identity, but do not mutate or claim ownership of Ollama's cache. |
| OpenAI | Remote provider | No local artifact; retain provider/model request identity in diagnostics. |

The cache key remains `provider + model identity + immutable revision + artifact
digest + format`. A friendly model name is never enough to reuse an artifact
across runtimes.

## Quality gate before resuming embedded work

A later embedded proposal must freeze a complete
`model + revision + quantization + prompt + decoding` tuple and test it per
language and domain. German remains first, followed by English, Spanish,
French, and Portuguese. Professional podcast and audiobook recordings are the
target domain; synthetic round trips remain diagnostic only.

At minimum, the proposal must report content WER/CER, surface improvements,
edit precision, transcript-level regression rate, protected-content and
contract violations, first-token latency, throughput, load, memory, model
size, cancellation, and repeated determinism. Raw text is always retained.
No aggregate gain may hide a regression in a product-priority language cell or
a critical change to a name, number, date, unit, technical term, or negation.

## Replacement boundary

The product boundary is `TextGenerationEngine`, not MLX, GGUF, Ollama, or an
OpenAI response type. Transcript chunking, prompt selection, audit diffs,
mechanical guards, and raw/corrected result retention live in the enhancement
orchestrator above the engine. The engine only owns model/provider execution,
ordered text deltas, supported generation options, cancellation, and lifecycle.

That separation lets a future official-MLX adapter, llama.cpp adapter, Ollama
provider, and OpenAI provider coexist or replace one another without changing
STT or TTS contracts.
