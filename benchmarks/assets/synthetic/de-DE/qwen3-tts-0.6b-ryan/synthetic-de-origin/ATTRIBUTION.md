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

- Generator: `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16`
- Model revision: `6415d95f88be018ff9e46813119dc3bc12261328`
- Model license: Apache-2.0
- Runtime: `Blaizzy/mlx-audio` 0.4.5
- Runtime revision: `64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`
- Runtime license: MIT
- Official compute runtime: Apple MLX
- Voice configuration: `Ryan`, German output, cross-lingual from an
  English-native preset
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-0.6b-mlx-reference.synthetic-de-origin-1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`ec7c19aaa1f7c9aeb1362f2a0259af81ea3599f6aa5876bb07bd833db7bad7b5`.
It was quantized to the PCM16 WAV with SHA-256
`773f28cb8970ed1425b7e20151d192008116093b80790d9f62021800b795d372`
before the checked-in Ogg Opus encode. The codec conversion is a format and
compression change; it does not intentionally alter the spoken content.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
