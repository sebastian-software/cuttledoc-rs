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
- Voice profile: `qwen-de-clear-documentary`, Seed 0
- Voice instruction: “A clear, calm German-speaking voice with precise
  articulation, restrained expression, and steady documentary narration.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-de-clear-documentary.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`af4e290315377c15f0235f75e07276fa9019f74ebb66b2d08e691944d7a1b19b`.
It was quantized to the PCM16 WAV with SHA-256
`8e289ef36193a2cd1861b2ee5ceef02bac637e72a87f0c7c0316cc523c13166d`
before the checked-in Ogg Opus encode. The codec conversion is a format and
compression change; it does not intentionally alter the spoken content.

All five checked ASR receivers fail independently at the source token `1962`.
The file is retained as a reproducible calibration failure, not as an accepted
quality exemplar. Preserve that caveat when using it as decision evidence.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
