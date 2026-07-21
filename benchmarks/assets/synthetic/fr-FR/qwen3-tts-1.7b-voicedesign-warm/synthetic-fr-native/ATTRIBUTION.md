# Attribution and provenance

## Source text

- Work: “Livre audio”
- Exact revision: `235403403`
- Revision: [permanent link](https://fr.wikipedia.org/w/index.php?oldid=235403403&title=Livre_audio)
- Source history: [history](https://fr.wikipedia.org/w/index.php?title=Livre_audio&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-fr-native` in
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
- Voice profile: `qwen-fr-warm-native`, Seed 0
- Voice instruction: “A warm, conversational French-speaking voice with natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-fr-warm-native.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`ded4483c9ab55661e5ac26f839ec244c3672ddac630fb10890ebfd9ca78dad7f`. It was quantized to the PCM16 WAV with SHA-256
`9afa2b26bb5f322b62128a055a45a47896e08a50179098c228c7bf94279f7896` before the checked-in Ogg Opus encode.

All five receivers recover the complete native-factual passage. Voxtral is lexically exact; the other receivers make between two and six word edits. Listening review remains required before treating this recording as
a voice-quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
