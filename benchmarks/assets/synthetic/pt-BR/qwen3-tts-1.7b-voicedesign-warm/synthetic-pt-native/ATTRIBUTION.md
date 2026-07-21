# Attribution and provenance

## Source text

- Work: “Audiolivro”
- Exact revision: `64246519`
- Revision: [permanent link](https://pt.wikipedia.org/w/index.php?oldid=64246519&title=Audiolivro)
- Source history: [history](https://pt.wikipedia.org/w/index.php?title=Audiolivro&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-pt-native` in
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
- Voice profile: `qwen-pt-warm-native`, Seed 0
- Voice instruction: “A warm, conversational Brazilian Portuguese voice with natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-pt-warm-native.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`23127a92f1260af8692607f18b009d7dc857219524555741cc3c2a23f3ca5b66`. It was quantized to the PCM16 WAV with SHA-256
`3f41e6f91fa3bf1fa8cb69caebf0d9396546164e5120707efbf8fe2c9643f84a` before the checked-in Ogg Opus encode.

All five receivers recover the complete native-factual passage. Voxtral makes one word edit, Whisper two, Parakeet and Apple four, while Qwen-ASR alone makes twelve. Listening review remains required before treating this recording as
a voice-quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
