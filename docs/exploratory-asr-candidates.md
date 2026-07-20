# Exploratory Apple-local ASR candidates

**Status:** issue #12 evaluation complete.

**Evidence date:** 2026-07-20.

The mandatory benchmark selected Apple SpeechTranscriber and a Whisper
large-v3-turbo fallback. This sweep asks a different question: which stronger
current model, if any, justifies expanding the repository-owned adapter over
the official MLX core?

## Disposition

| Candidate and exact artifact | License | Current official runtime | Apple-local result |
| --- | --- | --- | --- |
| [Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2/tree/87bc52657add533cd0156b3fc1aef027280754bf) | CC-BY-4.0 | NeMo main; Linux on NVIDIA Ampere/Blackwell/Hopper | blocked: no official MLX/CoreML or accepted owned adapter |
| [Canary 1B Flash](https://huggingface.co/nvidia/canary-1b-flash/tree/2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e) | CC-BY-4.0 | NeMo; NVIDIA-oriented | blocked: Safetensors exist, but no accepted Apple runtime and no Portuguese coverage |
| [Canary-Qwen 2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b/tree/b1469e1bba1cfe140205529c79c434ca47180960) | CC-BY-4.0 | NeMo 2.5+; Linux/NVIDIA | blocked: English-only, larger, and no accepted Apple runtime |
| [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/f3d333391852ba876df169dcc9ba902d25b6ab0b) | OpenMDW-1.1 | NeMo 26.06 and Transformers 5.13+; official integration lists Linux/NVIDIA | blocked: custom-license disposition and accepted Apple runtime both missing |
| [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) | Apache-2.0 | official Transformers/vLLM Python stack | measured end to end through the repository-owned adapter over official MLX; lifecycle/cancellation accepted, advance corpus and packaging gates |
| [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | Apache-2.0 | official Transformers/vLLM Python stack | blocked until the shared 0.6B architecture has an accepted Apple boundary |
| [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/tree/3060fe34b35ba5d44202ce9ff3c097642914f8f3) | Apache-2.0 | vLLM 0.10+ or Transformers; about 9.5 GB GPU memory | blocked: no accepted Apple runtime; owned port is materially larger than Qwen 0.6B |
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/tree/2769294da9567371363522aac9bbcfdd19447add) | Apache-2.0 | official vLLM and Transformers; publisher-linked ExecuTorch, C/MPS, MLX, and Rust paths | measured through pinned 4-bit MLX reference on audiobook and FLEURS controls at 480/2,400 ms; advance held-out corpus and true-streaming gates |

The machine-readable blocked records retain exact format, revision, license,
runtime, platform, engineering cost, and blocker under
[`benchmarks/runs`](../benchmarks/runs/). The original `canary-1b` remains
CC-BY-NC-4.0, but that license is not incorrectly applied to v2, Flash, or
Canary-Qwen.

Nemotron is not rejected as “NeMo checkpoint only.” The pinned repository
contains both a NeMo checkpoint and F32 Safetensors, and the publisher documents
NeMo plus Transformers offline and cache-aware streaming paths. The remaining
gates are its custom model license and the absence of an accepted native Apple
runtime.

The Voxtral entries likewise do not rely on a stale llama.cpp claim. Mini 4B
Realtime now has official Transformers support in addition to vLLM, and its
publisher card links ExecuTorch, pure C with Metal/MPS, MLX, and Rust community
paths. The pinned MLX oracle proves Apple-local feasibility and quality; none
of those paths automatically defines Cuttledoc's accepted product boundary.

## Qwen3-ASR 0.6B reference result

The bounded ten-fixture run used:

- official `Qwen/Qwen3-ASR-0.6B` at
  `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`;
- `mlx-community/Qwen3-ASR-0.6B-8bit` at
  `89e96d92ba34aca20b3e29fb10cc284097d1219f`;
- reference-only `mlx-audio` v0.4.5 at
  `04151c6abb74b886f879a4457ccdc96761f10102`; and
- official MLX 0.32.0.

| Metric | Qwen3-ASR 0.6B reference | Mandatory comparison context |
| --- | ---: | --- |
| Macro WER / CER | 5.10% / 1.49% | Whisper 5.61% / 1.25%; Apple 7.58% / 3.35% |
| Mean warm inference / RTF | 356.7 ms / 0.0243 | Whisper 774.2 ms; Apple 189.5 ms |
| Mean first token | 148.0 ms | Apple first result 52.1 ms |
| Maximum process RSS | 1.25 GB | Whisper 2.00 GB; Apple 24.0 MB |
| MLX peak allocation | 2.15 GB | reference-runtime diagnostic, not process RSS |
| Model / reference environment | 1.01 GB / 402.6 MB | Whisper model 2.90 GB |

The exact aggregate is
[`result.json`](../benchmarks/raw/phase0.qwen3-asr-0.6b-mlx-reference.multilingual-fleurs-10-1/result.json).
It preserves two measured repetitions per fixture and every token emission.

The recorded 5.10% WER uses the immutable phase-0 punctuation-deletion
normalization. The later boundary-preserving error review reports 4.15% for
Qwen and 3.60% for Whisper because it no longer collapses `T-Rex` or `25-30`
into one token. This reinforces the need for more independent fixtures and
semantic review; it does not reverse the Qwen/MLX adapter decision.

This is strong model and platform evidence, not dependency acceptance. The
Python adapter and community conversion are an oracle for fixtures, shape
mapping, and expected text only. The product experiment must load converted
weights through repository-owned task code compiled directly against official
MLX, preserving the same Rust-facing C ABI ownership established by #6 and
#15.

The reference runtime's `stream=True` emits decoded tokens after audio
preprocessing; it is not proof of incremental audio-input streaming. Its
token-position timestamps are estimates and are recorded as `unknown`.
Official Qwen streaming is currently vLLM-only and does not return timestamps;
official word timestamps require the separate Qwen3-ForcedAligner 0.6B.

## Decision

Qwen3-ASR 0.6B is now the third candidate through a bounded repository-owned
C++ adapter over official MLX. The direct path reproduces the pinned exact
transcript and 12/15 broader audiobook texts, with deterministic EOS completion
across 45 fresh-process runs. Its reusable session then reproduced the pinned
transcript in six of six calls across three complete lifecycles and returned
stable invalid-argument, busy, and cancelled statuses. The remaining product
work is held-out target-domain coverage, common-engine integration, explicit
batch-versus-streaming capabilities, clean materialization measurements, and a
documented release artifact budget—without importing `mlx-audio` or another
community runtime.

Voxtral Realtime is now a measured third model candidate rather than blocked
future research. Its 15-clip audiobook results are language- and delay-specific:

| Delay | Macro WER | German | English | Spanish | French | Portuguese | Mean RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 480 ms | 5.84% | 2.72% | 5.41% | 7.94% | 6.68% | 6.45% | 0.223 |
| 2,400 ms | 4.25% | 0.68% | 6.06% | 5.31% | 1.74% | 7.46% | 0.225 |
| Whisper comparison | 4.49% | 2.52% | 5.95% | 1.21% | 4.77% | 8.00% | 0.0569 |

The higher delay wins macro, German, Spanish, and French quality without a
meaningful offline-runtime increase, while 480 ms is better for English and
Portuguese. This validates language-specific selection and rules out a single
global Voxtral setting. The pilot is small, uses development-exposed
unverified transcripts, and measures complete-clip inference. The next gate is
held-out professional speech plus stateful streaming latency and cancellation.

The same settings were then run on the existing ten-clip FLEURS integration
set:

| Delay | Macro WER | German | English | Spanish | French | Portuguese | Mean RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 480 ms | 5.13% | 10.32% | 5.01% | 3.68% | 0.88% | 5.74% | 0.223 |
| 2,400 ms | 3.82% | 4.76% | 5.01% | 2.70% | 0.88% | 5.74% | 0.214 |
| Qwen3-ASR reference | 5.10% | 4.76% | 7.39% | 7.61% | 0.00% | 5.74% | 0.0243 |
| Whisper comparison | 5.61% | 4.76% | 15.29% | 2.70% | 1.75% | 3.56% | 0.0564 |

The aggregate advantage at 2,400 ms and the strong French cell replicate
across sources. The individual language pattern does not: the two FLEURS
German clips expose word-boundary sensitivity, and three languages are
identical across delay settings. The boundary-review report lowers the
2,400 ms aggregate to 2.86% and its German cell to 0% by treating hyphens and
spaces as token boundaries. This is strong candidate evidence, not a
population-level European-language ranking.

True incremental input is also proven on a 13.34-second German audiobook
fixture. Four repeated 320 ms-chunk sessions are deterministic at 0% WER/CER,
first append arrives at 1.69-1.71 seconds for the 480 ms setting and 3.61
seconds for 2,400 ms, running steps stay below 202 ms, and endpoint
finalization stays below 717 ms.

The finer 80 ms control exposes a blocking queue defect in the reference
runtime: one `step()` can drain audio concurrently appended by the producer
until end-of-stream, producing observed 6.24-25.15 second executor stalls.
There is no native cancellation method; `close()` is EOS finalization. This
rejects `mlx-audio` as the product boundary while strengthening the case for a
bounded owned adapter over official MLX.

Do not broaden Qwen to 1.7B yet. For Voxtral, compare a narrow official-MLX
adapter with the pure-C/MPS path while preserving the measured 320 ms behavior
and adding cooperative cancellation. Canary and Nemotron remain useful future
evidence.
