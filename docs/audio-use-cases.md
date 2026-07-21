# Audio use cases and preprocessing scope

## Primary product workload

Cuttledoc primarily transcribes clean, intentionally produced speech:

- professionally recorded podcasts;
- audiobooks and other narrated long-form material; and
- comparable recordings with intelligible voices, controlled levels, and
  ordinary editing or mastering.

Backend recommendations and release benchmarks optimize for this workload.
Results are reported per language and recording scenario. They are practical
indications, not a claim that one engine is universally best. Users remain able
to select another supported engine for their material.

Meeting captures, distant microphones, strong room reverberation, overlapping
conversation, and field recordings are useful best-effort inputs, but they do
not drive the initial backend choice.

## Preprocessing boundary

The default audio path should remain conservative:

```text
source media
  -> decode and channel selection/downmix
  -> resample to the engine input format
  -> non-destructive level handling
  -> optional VAD/chunking required by the selected engine
  -> speech recognition
```

Noise suppression, dereverberation, source separation, beamforming, and speech
enhancement belong before recognition when they are needed. They must be
optional, preserve the original input, and be benchmarked as part of the exact
pipeline configuration. A cleanup model can remove information or introduce
speech-like artifacts, so applying one unconditionally can make clean recordings
worse.

Room reverberation is especially difficult to reverse after recording. The
preferred order is prevention during capture, then restrained restoration when
the source requires it, then ASR. Cuttledoc may add named preprocessing profiles
later, but a broad audio-restoration subsystem is not part of the current
transcription release gate.

## Benchmark implication

The primary reproducible comparison uses exact text synthesized with multiple
voices per language. A small, rights-reviewed professional-podcast corpus is a
real-world control for long-form behavior and gross regressions. It does not
need expensive independent gold transcription unless the project later wants
to publish defensible real-world WER claims.

Synthetic results must remain split by language, TTS engine, and voice. The
real-world control must remain visibly separate. Neither aggregate should hide
language-specific failures.
