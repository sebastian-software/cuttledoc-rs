# Attribution and provenance

## Source text

- Work: “Künstliche Intelligenz”, German Wikipedia
- Exact revision: [268935951](https://de.wikipedia.org/w/index.php?oldid=268935951&title=K%C3%BCnstliche_Intelligenz)
- Contributor history: [article history](https://de.wikipedia.org/w/index.php?title=K%C3%BCnstliche_Intelligenz&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-de-origin` in
  `benchmarks/fixtures/synthetic-roundtrip-selection.json`

The passage joins selected source material into one benchmark utterance. No
ASR-specific wording was introduced. Capitalization and punctuation are
preserved in `reference.txt`; speaking necessarily renders punctuation as
prosody rather than literal characters.

## Generated recording

- Generator: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- Model revision: `7d3824abff87e49756bb0f83fb5411de75d160c4`
- Model license: Apache-2.0
- Runtime: `Blaizzy/mlx-audio` 0.4.5
- Runtime revision: `64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`
- Runtime license: MIT
- Official compute runtime: Apple MLX 0.32.0
- Voice profile: `qwen-de-warm-podcast`, Seed 0
- Voice instruction: “A warm, conversational German-speaking voice with
  natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-de-warm-podcast.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`d47fe6927ad81229a098f6ed91bb4c80b8379a826c95eeb8d53f682edf1a2e6c`.
It was quantized to the PCM16 WAV with SHA-256
`e3b2beed47e5811daf4affc68ed80d1703f936aaadebf8d7f46c40cc3bf7d89b`
before the checked-in Ogg Opus encode. The codec conversion is a format and
compression change; it does not intentionally alter the spoken content.

All five checked ASR receivers recover the critical source token `1962` and
return the complete passage. This recording passes the lexical calibration
gate provisionally; listening review is still required before treating it as
an accepted quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.

