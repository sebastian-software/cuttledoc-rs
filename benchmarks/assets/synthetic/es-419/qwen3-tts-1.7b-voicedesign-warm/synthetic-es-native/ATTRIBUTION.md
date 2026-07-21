# Attribution and provenance

## Source text

- Work: “Audiolibro”
- Exact revision: `174385246`
- Revision: [permanent link](https://es.wikipedia.org/w/index.php?oldid=174385246&title=Audiolibro)
- Source history: [history](https://es.wikipedia.org/w/index.php?title=Audiolibro&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-es-native` in
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
- Voice profile: `qwen-es-warm-native`, Seed 0
- Voice instruction: “A warm, conversational Latin American Spanish voice with natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-es-warm-native.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`89cb03f681fc2659b7270773ac80540f4ea234a0eeb9f6e73cac3e246b15daa2`. It was quantized to the PCM16 WAV with SHA-256
`996cad09b996e92c449d599c2bac02ef9c75e57f924174af481467eebeb5cffa` before the checked-in Ogg Opus encode.

Whisper and Voxtral recover the complete native-factual passage with three and four word edits. Their similar rendering of “a elegir” keeps that phrase open for pronunciation review; larger Qwen-ASR and Apple errors are not shared omissions. Listening review remains required before treating this recording as
a voice-quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
