# ADR-0013: Defer embedded text generation and prefer official MLX

## Status

Accepted

## Context

Cuttledoc can optionally improve transcript punctuation, capitalization,
structure, and carefully bounded recognition errors. Cuttledoc 2 used Ollama,
while v3 must also permit remote providers and may eventually embed an
Apple-local model. The embedded decision must not leak into the public task API
or be inherited accidentally from the selected speech-recognition runtime.

Issue #7 compared official MLX, llama.cpp/GGUF, mistral.rs, Candle, and Core ML
and ran a pinned multilingual instruction model on a representative German
audiobook transcript. The official MLX reference generated quickly, streamed
tokens, repeated deterministically, and cancelled cooperatively. Its Qwen3
0.6B model nevertheless violated the no-lexical-change prompt, changing
`fielen` to `fiel`; the required mechanical gate rejected the output.

## Decision

Defer an embedded transcript-enhancement backend from the initial v3 scope.
Enhancement remains optional and the complete raw transcript remains available.

Keep text generation as a separate task contract. OpenAI, Ollama, and future
embedded adapters can implement `TextGenerationEngine` independently; no
provider or runtime type crosses the stable API. Transcript chunking, prompt
selection, protected spans, external diffs, policy gates, and raw/corrected
retention belong to the enhancement orchestrator above the engine.

When embedded Apple-local text generation resumes, use official MLX through a
narrow repository-owned C++ adapter and C ABI as the leading candidate. The
adapter owns the selected model graph, KV cache, errors, lifecycle, streaming,
and cancellation while official MLX owns tensor execution and Metal kernels.
The official Python MLX-LM project remains a pinned reference only, not a
production dependency.

Retain llama.cpp/GGUF as the fallback if cross-platform embedded inference,
GGUF model availability, or a measured delivery advantage outweighs the cost
of a second native runtime and model format. Keep mistral.rs and Candle as
reference-only. Reconsider Core ML only with a selected converted model that
demonstrates a concrete deployment, energy, size, or compute advantage.

Do not select Qwen3 0.6B with `surface-only-v1` as a correction tuple. A future
model/prompt tuple must pass multilingual held-out quality and safety gates;
runtime success alone cannot promote it.

## Consequences

### Positive

- Initial v3 is not delayed by an optional model whose quality is not proven.
- MLX remains the first-class future Apple-local path instead of being rejected
  because a repository-owned model adapter requires work.
- OpenAI and Ollama can progress without waiting for an embedded decision.
- llama.cpp remains a real replacement path rather than an unused parallel
  dependency.
- Mechanical guards, not prompt obedience, define the correction safety
  boundary.

### Negative

- Initial v3 has no bundled offline transcript enhancement model.
- A future official-MLX backend still requires Cuttledoc to own and maintain a
  model-specific adapter.
- Cross-platform embedded inference may later require the additional
  llama.cpp/GGUF artifact and build path.
- Provider capabilities differ, so unsupported deterministic or structured
  options must fail explicitly rather than being silently ignored.

## Validation

The decision is backed by the
[`Apple-local text-generation evaluation`](../text-generation-runtime-evaluation.md),
the pinned
[`MLX reference probe`](../../spikes/text-generation-mlx-reference/README.md),
and its negative validator self-test. The measured 351 MB model loaded in
428.4 ms, produced its first token in 197.1 ms, generated at 270.1 tokens/s,
repeated with identical text and token IDs, and stopped cooperatively after
four tokens. The lexical gate rejected the deterministic output and the
development WER moved from 2.56% to 5.13%.

Revisit the deferral when representative, held-out professional podcast and
audiobook data demonstrate a safe model/prompt tuple in each product-priority
language, or when product scope explicitly requires embedded offline
enhancement.
