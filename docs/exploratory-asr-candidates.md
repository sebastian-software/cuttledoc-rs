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
| [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) | Apache-2.0 | official Transformers/vLLM Python stack | measured through a pinned reference-only MLX port; advance to an owned official-MLX adapter |
| [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | Apache-2.0 | official Transformers/vLLM Python stack | blocked until the shared 0.6B architecture has an accepted Apple boundary |
| [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/tree/3060fe34b35ba5d44202ce9ff3c097642914f8f3) | Apache-2.0 | vLLM 0.10+ or Transformers; about 9.5 GB GPU memory | blocked: no accepted Apple runtime; owned port is materially larger than Qwen 0.6B |
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/tree/2769294da9567371363522aac9bbcfdd19447add) | Apache-2.0 | vLLM realtime endpoint; at least 16 GB GPU memory | blocked: community MLX path is reference-only; owned 4B port follows a smaller adapter proof |

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

The Voxtral entries likewise do not rely on a stale llama.cpp claim. Their
current publisher cards list vLLM/Transformers for Mini 3B and vLLM's realtime
endpoint for Mini 4B. Those are valid upstream runtimes, but not Cuttledoc's
native Apple boundary.

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

Advance Qwen3-ASR 0.6B as the third candidate through one bounded
repository-owned C++ adapter over official MLX. The acceptance test is not
merely “it runs”: it must reproduce the reference transcripts, remain within a
documented binary/model budget, expose cancellation between decoder steps, and
show an upgrade path without importing `mlx-audio` or another community
runtime.

Do not broaden the implementation to Qwen 1.7B or Voxtral until the smaller
adapter demonstrates that the architecture and release cost are maintainable.
Canary and Nemotron remain useful future model evidence, but neither displaces
the current product selection or the direct Qwen/MLX follow-up.
