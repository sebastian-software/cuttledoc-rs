# Attribution and provenance

## Source text

- Work: “Buchdruck”, German Wikipedia
- Exact revision: [268791130](https://de.wikipedia.org/w/index.php?oldid=268791130&title=Buchdruck)
- Contributor history: [article history](https://de.wikipedia.org/w/index.php?title=Buchdruck&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-de-native` in
  `benchmarks/fixtures/synthetic-roundtrip-selection.json`

The benchmark uses the exact lead paragraph without an additional spoken-text
transform. Capitalization and punctuation are preserved in `reference.txt`;
speaking necessarily renders punctuation as prosody rather than literal
characters.

## Generated recording

- Generator: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- Model revision: `7d3824abff87e49756bb0f83fb5411de75d160c4`
- Model license: Apache-2.0
- Runtime: `Blaizzy/mlx-audio` 0.4.5
- Runtime revision: `64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`
- Runtime license: MIT
- Official compute runtime: Apple MLX 0.32.0
- Voice profile: `qwen-de-warm-native`, Seed 0
- Voice instruction: “A warm, conversational German-speaking voice with
  natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-de-warm-native.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`87d10c77bb42da5a20d39b3075045c41b3f24fad076cfef7b9cee07d36cecc47`.
It was quantized to the PCM16 WAV with SHA-256
`209fe8f634c34ff89c02ea8b6c6576e983444a92a7da9a3fe0df9a0b22a9dea3`
before the checked-in Ogg Opus encode. The codec conversion is a format and
compression change; it does not intentionally alter the spoken content.

All five checked ASR receivers recover the complete native-German factual
passage, including the 8th century, the year 1440, and Johannes Gutenberg.
Observed WER ranges from 1.92% to 2.88%; Whisper and Voxtral reproduce the
exact normalized character content. This recording passes the lexical content
gate; listening review is still required before treating it as a quality
exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
