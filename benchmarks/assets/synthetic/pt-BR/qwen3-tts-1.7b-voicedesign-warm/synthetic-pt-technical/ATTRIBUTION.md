# Attribution and provenance

## Source text

- Work: “Inteligência artificial”
- Exact revision: `72427477`
- Revision: [permanent link](https://pt.wikipedia.org/w/index.php?oldid=72427477&title=Intelig%C3%AAncia_artificial)
- Source history: [history](https://pt.wikipedia.org/w/index.php?title=Intelig%C3%AAncia_artificial&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-pt-technical` in
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
- Voice profile: `qwen-pt-warm-technical`, Seed 0
- Voice instruction: “A warm, conversational Brazilian Portuguese voice with natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-pt-warm-technical.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`1c1a1222e1f8a75ec5c0399ef922ebda846fd4f1a28202082c0c400538b14ef8`. It was quantized to the PCM16 WAV with SHA-256
`09c60d9d4ebd694273ac5849bc3bfa6cf17a94563eb1d9414817c28a0fe3ad47` before the checked-in Ogg Opus encode.

All five receivers garble the embedded critical term “Agentic AI” into incompatible nonsensical forms. The remaining Portuguese passage is complete, so this is a localized synthesis pronunciation failure rather than truncation. Listening review remains required before treating this recording as
a voice-quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
