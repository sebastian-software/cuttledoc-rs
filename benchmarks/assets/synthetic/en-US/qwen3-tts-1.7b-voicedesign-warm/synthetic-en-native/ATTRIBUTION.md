# Attribution and provenance

## Source text

- Work: “Audiobook”, English Wikipedia
- Exact revision: `1355975145`
- Revision: [permanent link](https://en.wikipedia.org/w/index.php?oldid=1355975145&title=Audiobook)
- Source history: [page history](https://en.wikipedia.org/w/index.php?title=Audiobook&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-en-native` in
  `benchmarks/fixtures/synthetic-roundtrip-selection.json`

The passage is materialized without an additional spoken-text transform.

## Generated recording

- Generator: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- Model revision: `7d3824abff87e49756bb0f83fb5411de75d160c4`
- Model license: Apache-2.0
- Runtime: `Blaizzy/mlx-audio` 0.4.5
- Runtime revision: `64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`
- Runtime license: MIT
- Official compute runtime: Apple MLX 0.32.0
- Voice profile: `qwen-en-warm-native`, Seed 0
- Voice instruction: “A warm, conversational American English voice with
  natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-en-warm-native.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`a7065589d9794aa4d439659343ba07d89542538422cf35f14d5d91f08bffcdaf`.
It was quantized to the PCM16 WAV with SHA-256
`f40711afbf0d4c555cf572b1e38fa40bb622359209f7cf632bb988433b1703cb`
before the checked-in Ogg Opus encode.

Four ASR receivers make one lexical word edit and Apple makes four. All five
recover the complete passage, including the 1930s and 1980s. This recording
passes the native-factual lexical gate; listening review is still required
before treating it as a voice-quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
