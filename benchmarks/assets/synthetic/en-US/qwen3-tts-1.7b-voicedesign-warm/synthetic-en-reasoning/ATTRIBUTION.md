# Attribution and provenance

## Source text

- Work: “Artificial intelligence”, English Wikipedia
- Exact revision: [1365114492](https://en.wikipedia.org/w/index.php?oldid=1365114492&title=Artificial_intelligence)
- Contributor history: [article history](https://en.wikipedia.org/w/index.php?title=Artificial_intelligence&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-en-reasoning` in
  `benchmarks/fixtures/synthetic-roundtrip-selection.json`

The passage is a verbatim selected source paragraph. Capitalization and
punctuation are preserved in `reference.txt`; speaking necessarily renders
punctuation as prosody rather than literal characters.

## Generated recording

- Generator: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- Model revision: `7d3824abff87e49756bb0f83fb5411de75d160c4`
- Model license: Apache-2.0
- Runtime: `Blaizzy/mlx-audio` 0.4.5
- Runtime revision: `64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`
- Runtime license: MIT
- Official compute runtime: Apple MLX 0.32.0
- Voice profile: `qwen-en-warm-podcast`, Seed 0
- Voice instruction: “A warm, conversational American English voice with
  natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-en-warm-podcast.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`6ec75583ea12ac8151e047e0917511b2d3c6751375629c1e16724770325699f5`.
It was quantized to the PCM16 WAV with SHA-256
`3cf26f94bff3a37317954e77770b323ecd9350ab2e9245f7dfecb8609a305a25`
before the checked-in Ogg Opus encode. The codec conversion is a format and
compression change; it does not intentionally alter the spoken content.

All five checked ASR receivers recover the exact normalized character content.
The three reported word edits per receiver only reflect the spoken expansion
of the hyphenated reference token `chains-of-thought`. This recording passes
the lexical calibration gate; listening review is still required before
treating it as an accepted quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.

