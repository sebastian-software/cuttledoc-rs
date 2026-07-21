# Attribution and provenance

## Source text

- Work: “Intelligence artificielle”
- Exact revision: `237681031`
- Revision: [permanent link](https://fr.wikipedia.org/w/index.php?oldid=237681031&title=Intelligence_artificielle)
- Source history: [history](https://fr.wikipedia.org/w/index.php?title=Intelligence_artificielle&action=history)
- Source license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Passage selector and digest: `synthetic-fr-technical` in
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
- Voice profile: `qwen-fr-warm-technical`, Seed 0
- Voice instruction: “A warm, conversational French-speaking voice with natural pacing, gentle energy, and polished podcast delivery.”
- Generation record:
  `benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-fr-warm-technical.1/result.json`

The lossless generated source is mono 24 kHz float PCM with SHA-256
`1581535efc07539be822e637a4cf836191034549cb6508c4856003a76999f070`. It was quantized to the PCM16 WAV with SHA-256
`a10c5f395d91f77007007e0fda97030588133c3b3fa9d3f0feaf4a287e4c66f3` before the checked-in Ogg Opus encode.

All five receivers agree that Qwen stops after the third of five technical list items even though generation reports normal termination below the token limit. Whisper additionally repeats filler after the shared endpoint. Listening review remains required before treating this recording as
a voice-quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
