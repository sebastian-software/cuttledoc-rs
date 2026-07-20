# Synthetic speech roundtrip benchmark

**Status:** accepted diagnostic plan; no measured TTS result yet

## Purpose

The first speech-synthesis spike uses controlled text-to-speech (TTS) output
as input to the existing speech-to-text (STT) candidates. This creates a
reproducible TTS → STT loop that is useful for:

- validating audio formats, chunk ownership, streaming, cancellation, and
  lifecycle across the Rust boundary;
- exposing pronunciation failures for names, numbers, abbreviations, German
  compounds, and embedded English technical terms;
- comparing all ASR engines against identical generated PCM; and
- separating lexical recognition errors from capitalization and punctuation
  reconstruction.

It is not target-domain evidence. Clean generated speech does not represent
speaker diversity, rooms, microphones, mastering, overlap, disfluency, or
spontaneous language in professional podcasts and audiobooks. Synthetic
scores therefore stay separate from issue #18 and cannot select a production
ASR model or satisfy a release-quality gate.

The machine-readable contract is
[`synthetic-roundtrip-plan.json`](../benchmarks/fixtures/synthetic-roundtrip-plan.json).

## Initial text source

The primary German pilot selects six 45–90 second passages from the exact revision
`268935951` of the German Wikipedia article
["Künstliche Intelligenz"](https://de.wikipedia.org/w/index.php?oldid=268935951&title=K%C3%BCnstliche_Intelligenz).
The selected sections provide technical terminology, proper names, dates,
abbreviations, compounds, parenthetical phrases, and long sentences without
inventing a benchmark text around one model's behavior.

An English control selects three passages from exact revision `1365114492` of
["Artificial intelligence"](https://en.wikipedia.org/w/index.php?oldid=1365114492&title=Artificial_intelligence).
German and English are separate result cells; aggregate WER must not obscure
language-specific behavior.

Wikipedia text is available under CC BY-SA 4.0. Materialization must preserve
the article title, exact revision link, history/authorship link, license,
verbatim-text digest, spoken-text digest, and a notice for every change made
for speech. Generated audio remains local until its attribution and
redistribution package has been reviewed. This isolation also prevents
CC BY-SA benchmark assets from being mistaken for code covered by the
repository license.

## First candidate set

| Candidate | Initial role | First boundary | Why it is included |
| --- | --- | --- | --- |
| Apple `AVSpeechSynthesizer` | system baseline | narrow Swift-to-C ABI called from Rust | fastest path to validate the synthesis lifecycle and capture an installed German voice |
| Qwen3-TTS 0.6B CustomVoice | primary open MLX candidate | pinned `mlx-audio` Python reference runner | German-capable, comparatively bounded model and a practical Apple Silicon implementation |
| Chatterbox multilingual | expressive open MLX comparator | pinned `mlx-audio` Python reference runner | German support and a different quality/prosody trade-off |
| Qwen-Audio-3.0-TTS-Plus | remote English quality ceiling | Alibaba Cloud API reference runner | current provider-voice leader; prevents the local comparison from defining quality only relative to weak baselines |

`mlx-audio` is a serious candidate, not a rejected integration shortcut. The
initial reference runner pins commit
`64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`. It gives the spike a working
Apple Silicon implementation for Qwen3-TTS and Chatterbox quickly enough to
measure model behavior. The measurements then decide between:

1. retaining the broader `mlx-audio` dependency;
2. owning a narrow adapter for the selected model directly over official MLX,
   following the direct Qwen3-ASR precedent; or
3. stopping a model whose quality or operational cost does not justify either
   integration.

The reference path must not silently define the future Rust API. It exists to
produce evidence for the repository-owned synthesis contract required by
ADR-0009.

On 2026-07-20, Artificial Analysis ranked Qwen-Audio-3.0-TTS-Plus first in its
Provider Voice Arena at 1,237 Quality Elo (±17) from 1,427 samples. That is
useful external evidence, but its scope matters: the arena evaluates provider
voices, and Alibaba currently documents the Plus model's built-in voices for
Mandarin and English. The score is therefore an English remote quality
ceiling, not evidence that the local open Qwen3-TTS model or German synthesis
has the same quality.

## Measurement and diagnosis

Every TTS candidate uses a fixed German voice and fixed generation parameters.
Its digest-checked normalized PCM is passed unchanged to Apple Speech,
Whisper, direct Qwen3-ASR over MLX, and Parakeet. Raw generated audio and raw
ASR text are retained locally.

The report keeps two text views:

- surface-form comparison preserves punctuation and capitalization to expose
  reconstruction behavior;
- lexical comparison applies one shared normalizer before WER/CER and content
  recall.

Critical-token, proper-name, and number/date/unit recall are reported beside
WER and CER. TTS timing includes cold load, time to first audio, completion
time, real-time factor, peak memory, model size, stream chunk count, and
cancellation latency. A small blinded listening pass records pronunciation
and prosody.

A roundtrip mismatch is not automatically an ASR error. It is attributed to
TTS only when the generated audio contains the wrong or missing spoken
content, to ASR when intelligible content is transcribed incorrectly across
the shared audio, or left unresolved when the evidence is ambiguous.

## Implementation sequence

1. Materialize and hash the German and English passages from both pinned
   Wikipedia revisions, including attribution and change metadata.
2. Build the Apple system-voice Rust vertical slice and validate generated
   audio ownership plus cancellation.
3. Pin the converted Qwen3-TTS artifact and build the `mlx-audio` reference
   runner at the selected commit.
4. Run the two local/system candidates through all four ASR backends, add the
   remote Qwen English ceiling when credentials are available, and produce the
   first language-aware roundtrip report.
5. Add Chatterbox only after the runner and report contract are stable.
6. Compare quality, latency, memory, model delivery, and maintenance cost;
   prototype the narrow direct official-MLX boundary for the leading open
   model.
7. Record the synthesis contract and dependency/release decision in the
   follow-up ADR required by ADR-0009.
