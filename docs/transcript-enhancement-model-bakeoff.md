# Transcript-enhancement model bakeoff

**Status:** in progress under issue #20.

**Plan date:** 2026-07-21.

## Question

Which model can safely improve professionally recorded podcast and audiobook
transcripts, and only then which Apple-local runtime delivers that model with
the best product tradeoff?

The experiment separates these questions deliberately:

1. compare model and prompt quality through one common pinned reference layer;
2. freeze the surviving tuple or tuples; and
3. compare those finalists through repository-owned official-MLX and Core ML
   paths.

This prevents a fast small model from winning before its corrections are useful
and prevents a stronger model from being discarded merely because its adapter
requires more work.

## Model candidates

| Priority | Source model | Role | Initial reference representation |
| ---: | --- | --- | --- |
| 1 | `google/gemma-4-E2B-it` | Primary quality candidate | Pinned 4-bit MLX conversion supported by the locked MLX-LM reference |
| 2 | `Qwen/Qwen3.5-0.8B` | Small efficiency challenger | Pinned 4-bit MLX conversion supported by the locked MLX-LM reference |
| 3 | `HuggingFaceTB/SmolLM3-3B` | Multilingual product-size compromise | Pinned 4-bit MLX conversion supported by the locked MLX-LM reference |
| Control | Ollama `gemma3n:e4b` | Historical Cuttledoc 2 behavior | Provider-managed artifact, accepted only when exact identity is recoverable |
| Negative control | `Qwen/Qwen3-0.6B` | Runtime and external-gate regression control | Existing pinned 4-bit MLX result |

The initial MLX representations make the model comparison practical on the
same host and runtime. They do not select MLX as the product boundary. Exact
source and conversion revisions, artifact files, sizes, and SHA-256 digests
belong in the candidate manifest before a run is accepted.

## Frozen candidate identity

Every result identifies the complete tuple:

`source model + source revision + conversion revision + artifact digests +
quantization + reference runtime revision + prompt id/hash + decoding settings
+ language + domain + ASR backend + supplied context`.

Results from different tuples are not merged under a friendly model name. The
reference transcript is never rendered into the inference prompt.

## Evaluation stages

### Stage A: executable development matrix

- Generalize the issue-#7 runner and validator to consume candidate manifests
  rather than hard-coded Qwen identifiers.
- Execute all three candidates on the existing German audiobook development
  fixture with deterministic, non-thinking decoding where supported.
- Run `surface-only-v1` first because its lexical invariant is mechanically
  decidable.
- Record exact output, token identifiers, external diff, parser and policy
  result, load time, first token, throughput, peak process/MLX memory, model
  size, deterministic repeat, and cooperative cancellation.
- Treat WER on the unverified development transcript as diagnostic only.

This stage rejects broken artifacts, incompatible templates, instruction
failures, and obviously unsuitable model/prompt tuples. It does not select a
production model.

### Stage B: German professional-audio selection

- Use source-grouped, human-verified professional podcast and independently
  narrated audiobook data.
- Compare the same raw Apple, Whisper, Qwen, Parakeet, and Voxtral ASR outputs
  where available.
- Tune only on development sources, freeze on validation sources, and execute
  once on unseen test works or episodes.
- Compare raw/no-op, historical, surface-only, conservative error-profile, and
  targeted-span prompts where their required product evidence exists.

The model-quality recommendation is German-first because that is the primary
product use case and because the user can directly listen to and review German
errors.

### Stage C: primary-language robustness

Repeat the frozen winner on English, French, Spanish, and Portuguese sources.
Report every language/domain cell separately. Synthetic roundtrip audio may
increase voice and pronunciation variance but remains a diagnostic control,
not acceptance gold.

### Stage D: embedded runtime comparison

Port only the quality survivor or survivors.

| Evidence | Official MLX path | Core ML path |
| --- | --- | --- |
| Boundary | Narrow repository-owned C++ adapter and C ABI | Rust-owned lifecycle directly or through a narrow repository-owned Apple shim |
| Model fidelity | Match pinned MLX reference logits/tokens within declared tolerance | Match source/reference logits or frozen text outputs after conversion |
| State | Owned graph and KV/architecture-specific cache | Explicit stateful model inputs/outputs or Core ML state |
| Compute | Record Metal execution and memory | Record actual compute plan/device; do not infer Neural Engine use |
| Product behavior | Streaming, backpressure, cancellation, reuse, cleanup, errors | Streaming, cancellation boundary, reuse, cleanup, errors |
| Delivery | Runtime/model size, load time, signing, upgrade work | Compiled artifact size, conversion provenance, load/compile time, deployment target |

Runtime selection is per model and task. A winner for transcript enhancement
does not automatically become the ASR or TTS runtime.

## Quality and safety gates

Report content and presentation separately:

- Unicode WER/CER before and after;
- punctuation, capitalization, and sentence-boundary scores;
- proposed, accepted, beneficial, neutral, and harmful lexical edits;
- edit precision and transcript-level regression rate;
- names and terminology, numbers/dates/units, negation, omissions, and
  hallucinations;
- protected-span, parser, and output-contract violations; and
- results per language, domain, source group, ASR backend, prompt, and model.

Acceptance requires:

- no critical semantic regression on held-out test data;
- no protected-content violation;
- no content-metric regression in any product-priority language/domain cell;
- a mechanically exact lexical invariant for surface-only mode; and
- enough presentation or bounded lexical improvement to justify model size and
  operational complexity.

The complete raw transcript remains available regardless of whether an
enhancement candidate is accepted.

## Commit checkpoints

1. ADR and experiment plan.
2. Generic manifest, runner, validator, and self-tests.
3. Immutable candidate pins and verified artifact metadata.
4. Stage-A machine-readable results and analysis.
5. Held-out quality results and finalist selection.
6. MLX/Core ML finalist evidence and final ADR.
