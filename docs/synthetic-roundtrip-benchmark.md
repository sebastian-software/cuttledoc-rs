# Synthetic speech roundtrip benchmark

**Status:** German and Spanish pass all three Qwen content cells; English
technical and native-factual pass while English dialogue exposes a repeat-and-
truncate failure; French, Portuguese, and listening review remain pending

## Purpose

The benchmark uses controlled text-to-speech (TTS) output as input to the
speech-to-text (STT) candidates. This creates a reproducible TTS → STT loop
that is useful for:

- validating audio formats, chunk ownership, streaming, cancellation, and
  lifecycle across the Rust boundary;
- exposing pronunciation failures for names, numbers, abbreviations, German
  compounds, and embedded English technical terms;
- comparing all ASR engines against identical generated PCM; and
- separating lexical recognition errors from capitalization and punctuation
  reconstruction.

The primary product workload is clean, professionally produced speech. A
multi-voice synthetic matrix is therefore useful decision support for the
default recommendation: it can compare engines under exact, repeatable input
and known reference text. It does not establish universal superiority or
release-quality real-world WER. The professional German podcast corpus remains
a separate long-form control, and users can select any supported ASR engine.

Single-voice diagnostics already checked in do not satisfy the revised matrix.
Each measured locale must cover at least three pinned voices across all three
required TTS engines. Results remain split by locale, TTS engine, and voice so
a strong or weak voice cannot disappear inside one aggregate.

The machine-readable contract is
[`synthetic-roundtrip-plan.json`](../benchmarks/fixtures/synthetic-roundtrip-plan.json).

The exact 22-passage selection is
[`synthetic-roundtrip-selection.json`](../benchmarks/fixtures/synthetic-roundtrip-selection.json).
The repository materializer retrieves ten pinned MediaWiki revisions, reads
five digest-pinned repository-authored sources, and reproduces all 22
expected text digests. It writes the full CC BY-SA passage and attribution
package to the caller-selected local output directory. Reviewed fixtures may
additionally enter `benchmarks/assets` with an explicit per-asset license and
attribution package.

```sh
node scripts/materialize-synthetic-roundtrip.mjs \
  --output-dir /absolute/path/to/cuttledoc-synthetic-roundtrip
```

## Initial text sources

The primary German pilot selects six 45–90 second passages from the exact revision
`268935951` of the German Wikipedia article
["Künstliche Intelligenz"](https://de.wikipedia.org/w/index.php?oldid=268935951&title=K%C3%BCnstliche_Intelligenz).
The selected sections provide technical terminology, proper names, dates,
abbreviations, compounds, parenthetical phrases, and long sentences without
inventing a benchmark text around one model's behavior.

The primary non-Asian locale set is `de-DE`, `en-US`, `es-419`, `fr-FR`, and
`pt-BR`. Each locale has a separate three-part calibration slice so aggregate
WER cannot obscure language- or content-specific behavior:

| Locale | Technical cell | Native-factual cell | Dialogue cell |
| --- | --- | --- | --- |
| `de-DE` | `synthetic-de-origin` | `synthetic-de-native` | `synthetic-de-dialogue` |
| `en-US` | `synthetic-en-reasoning` | `synthetic-en-native` | `synthetic-en-dialogue` |
| `es-419` | `synthetic-es-technical` | `synthetic-es-native` | `synthetic-es-dialogue` |
| `fr-FR` | `synthetic-fr-technical` | `synthetic-fr-native` | `synthetic-fr-dialogue` |
| `pt-BR` | `synthetic-pt-technical` | `synthetic-pt-native` | `synthetic-pt-dialogue` |

The technical and native-factual cells use exact, digest-pinned Wikipedia
revisions. The dialogues are repository-authored and idiomatically adapted
around the same recording scenario; they are comparable in intent but are not
claimed to have identical linguistic difficulty.

The Spanish and Portuguese source editions provide practical clean-speech
controls, not universal regional coverage. In particular, an `es-419` result
must not be generalized to every Latin American variety, and a `pt-BR` result
must not be generalized beyond the actual text and generated voice. Future
native-variety material may strengthen those cells without changing this
limitation.

The dialogue is currently synthesized by one described voice. It isolates the
effect of conversational text and punctuation; it is not a two-speaker or
speaker-diarization test.

All selected text is available under CC BY-SA 4.0. Materialization preserves
the source identity, exact revision or repository digest, history/authorship
link, license, verbatim-text digest, spoken-text digest, and any change notice.
Lossless generated audio remains local and digest-pinned. A compact Opus copy
may be committed only after its attribution, redistribution, and
lossless-versus-codec control have passed review. The dedicated asset tree and
SPDX sidecars prevent CC BY-SA media from being mistaken for code covered by
the repository MIT license.

## Calibration shortlist

| Candidate | Disposition | First boundary | Why it is included |
| --- | --- | --- | --- |
| Apple `AVSpeechSynthesizer` | required system baseline | narrow Swift-to-C ABI called from Rust | two installed voices per locale exercise the platform path without a bundled model |
| Qwen3-TTS 1.7B VoiceDesign | required open multilingual generator | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16@7d3824ab` through `mlx-audio@64e8416c` | description-pinned voices avoid third-party prompt audio and give German and English distinct controlled presentations |
| Voxtral 4B TTS 2603 BF16 | required European-language hypothesis; reference-only | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16@dd85c02a` through `mlx-audio@64e8416c` | native German and other European-language presets plus BF16 isolate model behavior from the first 4-bit diagnostic |
| KugelAudio-0-Open | bounded German challenger | `kugelaudio/kugelaudio-0-open@22d6ed9b` through `mlx-audio@64e8416c` | German/European focus justifies one short calibration, but the current MLX path exposes only an implicit default voice |
| Chatterbox Multilingual V3 | deferred | upstream reference; current MLX conversion still to verify | multi-voice comparison requires reference audio whose rights, digests, and provenance must be accepted first |
| Qwen-Audio-3.0-TTS-Plus | optional remote English ceiling | Alibaba Cloud API reference runner | provider evidence remains useful for English listening, but does not block the local or German matrix |

`mlx-audio` is a serious candidate, not a rejected integration shortcut. The
historical reference runner pins commit
`64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`. It gives the spike a working
Apple Silicon implementation for Qwen3-TTS 0.6B and Voxtral 4-bit quickly
enough to measure model behavior. Those diagnostic pins remain immutable.
The current Qwen VoiceDesign, Voxtral BF16, and KugelAudio snapshots and their
shared runtime are frozen in
[`spikes/tts-calibration`](../spikes/tts-calibration/README.md). The manifests
pin all Hub files by byte count and SHA-256; the generic fetcher rejects any
download drift. The measurements now decide between:

1. retaining the broader `mlx-audio` dependency;
2. owning a narrow adapter for the selected model directly over official MLX,
   following the direct Qwen3-ASR precedent; or
3. stopping a model whose quality or operational cost does not justify either
   integration.

The reference path must not silently define the future Rust API. It exists to
produce evidence for the repository-owned synthesis contract required by
ADR-0009.

KugelAudio remains a single-voice challenger at this checkpoint. Although its
Hub metadata names three `.pt` voice presets, those files are absent from the
pinned snapshot and `mlx-audio` 0.4.5 ignores the `voice` argument for this
model. The implicit default path remains runnable, but cannot contribute to
the required multi-voice coverage.

The dated rationale, rejected alternatives, ASR receivers, and staged
execution gate are recorded in
[`speech-engine-shortlist-2026-07.md`](speech-engine-shortlist-2026-07.md).

Voxtral is evaluated separately for synthesis and recognition. For TTS, the
official model provides 20 presets across nine languages, including native
German and French voices. The publisher paper reports a German WER reduction
from 4.08% before DPO to 0.83% after DPO, plus a 72.0% German human preference
win rate against ElevenLabs Flash v2.5. Those figures justify a local
language-specific measurement but do not substitute for the shared benchmark.
The open TTS weights and supplied voices are CC BY-NC 4.0, so they remain
reference-only; production would require the hosted Mistral API or a separate
commercial license.

On 2026-07-20, Artificial Analysis ranked Qwen-Audio-3.0-TTS-Plus first in its
Provider Voice Arena at 1,237 Quality Elo (±17) from 1,427 samples. That is
useful external evidence, but its scope matters: the arena evaluates provider
voices, and Alibaba currently documents the Plus model's built-in voices for
Mandarin and English. The score is therefore an English remote quality
ceiling, not evidence that the local open Qwen3-TTS model or German synthesis
has the same quality.

## Measurement and diagnosis

Each voice has a stable provider/model voice ID and fixed generation
parameters. The same passages are used across voices. Every artifact's
digest-checked normalized PCM is passed unchanged to Apple Speech, Whisper,
direct Qwen3-ASR over MLX, Parakeet, and direct Voxtral Realtime over MLX at
the 2,400 ms quality configuration. Raw generated audio and raw ASR text are
retained locally. A reviewed Ogg Opus copy may additionally be retained in Git
for exact cross-engine reproduction.

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

## Repository audio format and codec control

The canonical Git format for reviewed clean-speech fixtures is mono Ogg Opus
with a 64 kbit/s VBR target, the `audio` application, 20 ms frames, and 0%
expected packet loss. The original lossless generation output remains local
for listening and acoustic-quality review. Benchmark media is excluded from
product packages and carries its own attribution and license.

The first checked fixture is
[`synthetic-de-origin`](../benchmarks/assets/synthetic/de-DE/qwen3-tts-0.6b-ryan/synthetic-de-origin/manifest.json).
Its 7.44 MB lossless float PCM source becomes a 763,465-byte Opus artifact.
FFmpeg 8.1.2 with libopus 1.6.1 produced byte-identical repeated encodes. The
machine-readable
[`codec control`](../benchmarks/controls/opus-codec-qwen3-tts-de-1.json)
compares normalized lossless audio with 48, 64, and 96 kbit/s targets across
all five required ASR engines:

| Backend | Lossless WER | 48 kbit/s | 64 kbit/s | 96 kbit/s |
| --- | ---: | ---: | ---: | ---: |
| Whisper large-v3-turbo | 1.94% | 1.94% | 1.94% | 1.94% |
| Parakeet TDT 0.6B v3 | 8.74% | 9.71% | 6.80% | 8.74% |
| Direct Qwen3-ASR 0.6B/MLX | 1.94% | 2.91% | 1.94% | 1.94% |
| Direct Voxtral Realtime 4B/MLX | 3.88% | 3.88% | 3.88% | 3.88% |
| Apple Speech | 4.85% | 4.85% | 4.85% | 4.85% |

At 64 kbit/s, no backend regressed in lexical WER relative to lossless. The
remaining transcript differences are punctuation or compound-hyphen choices;
Parakeet happened to improve by two word edits. That non-monotonic improvement
is model sensitivity, not evidence that lossy audio is better. This bounded
control accepts 64 kbit/s for clean synthetic ASR fixtures; it does not claim
codec transparency for music, TTS listening tests, or every future voice and
language.

## Apple system baseline result

The first machine-readable synthesis result is
[`phase5.apple-tts.synthetic-de-origin-1`](../benchmarks/raw/phase5.apple-tts.synthetic-de-origin-1/result.json).
At source revision `cc252ac91b2b1b67573361ac9ebb96c32a76466e`, the
installed compact German voice Anna generated 55.145 seconds of native mono
22,050 Hz f32 PCM. The measured development run reached first audio at
0.963 seconds and completed synthesis in 1.437 seconds (RTF 0.0261) with a
28.2 MB peak footprint. Busy and cancelled calls returned stable statuses `4`
and `3`; cancel-to-return was 0.288 ms.

This proves the Rust/Swift lifecycle and output ownership, not perceived voice
quality. The identical binary could enumerate voices but received a platform
cancel before its first buffer in a restricted process context. Clean packaged
execution therefore remains an explicit productization gate.

## Qwen3-TTS MLX reference result

The pinned local open-model result is
[`phase5.qwen3-tts-0.6b-mlx-reference.synthetic-de-origin-1`](../benchmarks/raw/phase5.qwen3-tts-0.6b-mlx-reference.synthetic-de-origin-1/result.json).
Qwen3-TTS generated 77.52 seconds of mono 24 kHz f32 PCM through the pinned
`mlx-audio` reference at source revision
`ca4307f0f2ae475ef03e5500d05074fb0cdda943`. Loading took 4.665 seconds,
synthesis took 39.597 seconds (conventional RTF 0.511), process RSS peaked at
2.71 GB, and MLX reported 13.90 GB of peak allocated memory. Generation
stopped normally at 969 audio tokens rather than the 1,200-token limit.

The model supports German but does not provide a native German preset voice.
The first diagnostic therefore fixes the English-native `Ryan` preset speaking
German. This is a real cross-lingual limitation to review, not a reason to
discard the platform or model family.

Five ASR backends have now transcribed the exact same digest-pinned 16 kHz
audio:

| Backend | German WER | German CER | Diagnostic observation |
| --- | ---: | ---: | --- |
| Whisper large-v3-turbo | 1.94% | 0.00% | exact lexical character sequence; WER comes from splitting `Rockefeller-Stiftung` |
| Direct Qwen3-ASR 0.6B/MLX | 1.94% | 0.58% | semantically equivalent `ca.`/`circa` and `im Lauf`/`im Laufe`; names and technical terms preserved |
| Direct Voxtral Realtime 4B/MLX | 3.88% | 0.00% | lexical differences are source/reference hyphens and `ca.`/`circa`; names and technical terms preserved |
| Apple Speech | 4.85% | 0.58% | substitutions include `Dartmouth`/`Dartmoalth` and `Augmentation`/`Augumentation` |
| Parakeet TDT 0.6B v3 | 8.74% | 3.03% | language-switch errors concentrate around embedded English terms |

The spread strengthens the evidence that Qwen generated the intended German
content and demonstrates why one roundtrip score must not be treated as a TTS
quality score. All five ASR cells are complete on identical PCM. A blinded
listening review remains necessary to evaluate pronunciation and prosody and
to assign any audible residual errors.

## Qwen3-TTS 1.7B VoiceDesign calibration

The first current-model calibration is
[`qwen-de-clear-documentary`](../benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-de-clear-documentary.1/result.json).
Its separately licensed
[`64 kbit/s Opus fixture`](../benchmarks/assets/synthetic/de-DE/qwen3-tts-1.7b-voicedesign-clear/synthetic-de-origin/manifest.json)
is retained so the critical-token failure can be replayed without regenerating
the 4.52 GB model; lossless f32 remains local and digest-pinned.
The pinned 1.7B BF16 snapshot generated 52.24 seconds of mono 24 kHz f32 PCM.
Loading took 1.785 seconds and synthesis took 22.844 seconds (RTF 0.437), with
12.63 GB MLX peak allocation. Generation stopped normally at 653 of 1,200
tokens; the output is finite and contains the complete passage.

All five receivers transcribed the identical normalized f32 master:

| Backend | German WER | German CER | Diagnostic observation |
| --- | ---: | ---: | --- |
| Parakeet TDT 0.6B v3 | 3.88% | 1.30% | best aggregate score, but renders `1962` as `106 und 66` |
| Whisper large-v3-turbo | 4.85% | 4.48% | renders the year as nonsensical words around `1966` |
| Direct Voxtral Realtime 4B/MLX | 4.85% | 1.16% | renders the year as `101,600` |
| Direct Qwen3-ASR 0.6B/MLX | 6.80% | 3.90% | renders the year as `1966` plus nonsensical material |
| Apple Speech | 6.80% | 2.31% | omits the year and adds several proper-name errors |

The receivers otherwise recover the full text and broadly agree on names and
technical content. Their independent, incompatible failures at the same
`1962` position localize a probable synthesis pronunciation error. The clear
profile therefore fails the critical-token gate provisionally. The shorter
duration is faster delivery, not truncation.

The paired
[`qwen-de-warm-podcast`](../benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-de-warm-podcast.1/result.json)
run generated 56.00 seconds at RTF 0.448 and stopped normally at 700 of 1,200
tokens. Its separately licensed
[`64 kbit/s Opus fixture`](../benchmarks/assets/synthetic/de-DE/qwen3-tts-1.7b-voicedesign-warm/synthetic-de-origin/manifest.json)
provides the positive replay control. All five receivers return the complete
passage and recover `1962` exactly:

| Backend | Clear WER | Warm WER | Warm-profile observation |
| --- | ---: | ---: | --- |
| Apple Speech | 6.80% | 0.97% | one `on`/`an` error; `1962` exact |
| Whisper large-v3-turbo | 4.85% | 2.91% | three orthographic/tokenization edits; `1962` exact |
| Direct Voxtral Realtime 4B/MLX | 4.85% | 2.91% | three orthographic/tokenization edits; `1962` exact |
| Direct Qwen3-ASR 0.6B/MLX | 6.80% | 3.88% | four minor phrase-boundary edits; `1962` exact |
| Parakeet TDT 0.6B v3 | 3.88% | 5.83% | six edits concentrated around embedded English; `1962` exact |

The opposite result on the same text, model revision, seed, and five receivers
shows that the critical failure is specific to the designed voice/profile, not
the Qwen VoiceDesign family. Retain the warm German profile and reject the
clear German profile. Listening remains required for audible realization and
prosody.

The
[`qwen-en-warm-podcast`](../benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-en-warm-podcast.1/result.json)
cross-language gate generated 56.72 seconds at RTF 0.478 and stopped normally
at 709 of 1,200 tokens. All five receivers return the complete passage with
zero normalized character edits. Each reports the same 3/117 word edits
(2.56% WER) solely because the hyphenated reference token
`chains-of-thought` is spoken as three words. The separately licensed
[`64 kbit/s Opus fixture`](../benchmarks/assets/synthetic/en-US/qwen3-tts-1.7b-voicedesign-warm/synthetic-en-reasoning/manifest.json)
preserves this positive control.

The bounded Qwen content step keeps the accepted warm voice description fixed
while changing the content cell. German established the method first; the
same technical, native-factual, and dialogue structure is now frozen for the
other four primary locales before the larger Voxtral calibration begins.

The `synthetic-de-native` run is now complete. Qwen generated 54.88 seconds of
finite audio and stopped normally at 686 of 1,200 tokens. All five receivers
recover the complete passage, including the 8th century, 1440, and Johannes
Gutenberg. WER spans only 1.92–2.88%; Whisper and Voxtral have zero normalized
character edits. The remaining differences are orthography, compound
boundaries, and one receiver-specific dropped preposition rather than a shared
content failure. This cell passes the lexical content gate; listening review
remains open.

The `synthetic-de-dialogue` run generated 56.88 seconds of finite audio and
stopped normally at 711 tokens. Whisper, direct Qwen3-ASR, and Apple each make
one word edit (0.81% WER); direct Voxtral makes two (1.63%). Parakeet is the
outlier at seven edits (5.69%), including “Bagmara”, “Nine”, and an omitted
speaker attribution. The other four receivers recover those locations, so the
evidence identifies a receiver-specific Parakeet weakness rather than a shared
Qwen content error. The dialogue cell passes the lexical content gate; one
voice reads the full exchange, and listening review remains open.

The completed German cells and the English technical/native-factual cells
accept the warm Qwen profiles, while the clear German profile and English
dialogue fail for different reasons. The English dialogue reads 102 reference
words correctly, then repeats “What happens if someone turns away while
speaking?” until the 1,200-token limit and omits the final 32 reference words.
This is a shared synthesis failure: four receivers report roughly 24–25% WER,
while Whisper transcribes the audible repetitions and exceeds 200% WER.

All three Spanish generations finish normally. The technical cell ranges from
1.27% WER (Whisper) to 14.65% (Parakeet), the native-factual cell from 1.58%
to 20.53% (Qwen-ASR), and the dialogue from 0.68% for four receivers to 10.27%
for Parakeet. The high values are not shared omissions: Whisper and Voxtral
recover both factual passages almost completely, and four receivers recover
the dialogue with one edit. The similar Whisper/Voxtral rendering of “a
elegir” in the native-factual audio remains explicitly open for pronunciation
review. French and Portuguese are the next bounded Qwen step. Listening
remains a separate gate before full-matrix promotion.

## Voxtral TTS MLX reference result

The native-German Voxtral result is
[`phase5.voxtral-tts-4b-mlx-reference.synthetic-de-origin-1`](../benchmarks/raw/phase5.voxtral-tts-4b-mlx-reference.synthetic-de-origin-1/result.json).
The pinned 4-bit conversion generated 54.96 seconds of mono 24 kHz f32 PCM at
source revision `61ddf5c9715796f9309a5ef278610fe2f3de858d`. Loading took
1.866 seconds, synthesis took 18.060 seconds (conventional RTF 0.329), process
RSS peaked at 2.92 GB, and MLX reported 5.57 GB of peak allocated memory.
Generation stopped normally at 687 audio tokens. This reference path is faster
and materially smaller in MLX peak allocation than the first Qwen3-TTS run,
but the different output durations prevent a direct speech-rate conclusion.

The output RMS was only 0.0107, compared with 0.0632 for the Qwen diagnostic.
Every backend therefore also ran against a clipping-free +12 dB control. The
raw-level matrix is:

| Backend | Voxtral German WER | Voxtral German CER | Qwen German WER | Diagnostic observation |
| --- | ---: | ---: | ---: | --- |
| Whisper large-v3-turbo | 8.74% | 4.62% | 1.94% | omits the first German parenthetical and substitutes `on`/`und` plus `den Begriff`/`in den Brief` |
| Direct Qwen3-ASR 0.6B/MLX | 11.65% | 4.91% | 1.94% | additional verb, phrase, and technical-term changes |
| Apple Speech | 17.48% | 6.50% | 4.85% | several proper-name and embedded-English errors |
| Parakeet TDT 0.6B v3 | 62.14% | 44.22% | 8.74% | level-sensitive VAD truncates after the 1962 sentence |

At +12 dB, Parakeet improves to 46.60% WER and Apple to 14.56%. Whisper is
byte-for-byte unchanged at 8.74%, while direct Qwen3-ASR remains near 12% at
12.62%. Low signal level therefore explains part of the Parakeet and Apple
failure, but not the shared lexical discrepancy. On this one passage, the
pinned 4-bit MLX reference has materially worse roundtrip fidelity than
Qwen3-TTS despite using a native German voice.

This is not yet a model-family verdict. The remaining causes include 4-bit
quantization, the MLX reference implementation, the selected preset, and
stochastic generation. A listening review and second generation control are
required, followed by BF16 or hosted Voxtral output if practical. The local
artifact also remains reference-only under CC BY-NC 4.0.

## Implementation sequence

1. Materialize and hash the five primary-language slices from ten pinned
   Wikipedia revisions and five repository-authored dialogues, including
   attribution and change metadata.
   **Complete:** 22 selectors reproduce their expected SHA-256 digests.
2. Build the Apple system-voice Rust vertical slice and validate generated
   audio ownership plus cancellation.
   **Complete:** real PCM, timing, busy, cancellation, and execution-context
   behavior are recorded.
3. Pin the historical Qwen3-TTS 0.6B artifact and build the `mlx-audio` reference
   runner at the selected commit.
   **Artifact complete:** revision `6415d95f88be018ff9e46813119dc3bc12261328`,
   its 2.49 GB snapshot, every file digest, the runtime revision, and the
   cross-lingual German generation contract are frozen in
   [`model-manifest.json`](../spikes/qwen3-tts-mlx-reference/model-manifest.json).
   **Reference run complete:** real PCM, timing, resource use, termination, and
   all four historical ASR content checks are recorded. Listening is still
   open.
4. Re-evaluate current TTS and ASR engines before multiplying the matrix.
   **Complete:** the July 2026 shortlist selects Apple, Qwen VoiceDesign, and
   Voxtral BF16 as required generators, KugelAudio as a bounded German
   challenger, and five required ASR receivers including direct Voxtral
   Realtime.
   **Archive control complete:** a five-engine lossless/48/64/96 kbit/s check
   accepts 64 kbit/s Ogg Opus for reviewed clean-speech Git fixtures. The first
   CC BY-SA asset, attribution, reproduction script, and validation record are
   checked in; lossless generation PCM remains local.
5. Pin the new model/runtime artifacts and run the three content cells in each
   primary locale before the full multi-voice matrix. Generate two voices each
   for Apple, Qwen, and Voxtral, plus one German KugelAudio voice; run the five
   required ASR receivers and the two bounded ASR qualifiers. Freeze only the
   candidates that pass fidelity, stability, listening, and operational-cost
   gates.
   **In progress:** all three model/runtime snapshots are pinned. German's
   three Qwen cells pass the lexical gate, the English technical and
   native-factual cells pass, the English dialogue has a pinned repeat-and-
   truncate failure, all three Spanish cells complete, and the remaining six
   French and Portuguese Qwen cells are frozen for execution.
6. Compare quality, latency, memory, model delivery, and maintenance cost;
   prototype the narrow direct official-MLX boundary for the leading open
   model.
7. Record the synthesis contract and dependency/release decision in the
   follow-up ADR required by ADR-0009.
