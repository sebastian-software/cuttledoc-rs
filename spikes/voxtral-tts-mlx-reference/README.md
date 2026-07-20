# Voxtral TTS MLX reference

This spike evaluates Voxtral as a European-language synthesis candidate on the
same pinned German Wikipedia passage used by the Apple and Qwen3-TTS
diagnostics.

The first candidate is the 4-bit MLX conversion of Voxtral 4B TTS 2603. It
provides native German, French, Spanish, Italian, Portuguese, and Dutch voice
presets instead of relying on cross-lingual synthesis from an English preset.
The German pilot fixes `de_female`.

This is a quality and integration reference only. The model and supplied voice
references are CC BY-NC 4.0, so the local artifact is not eligible for the
production dependency set without a separate commercial license. The hosted
Mistral API is evaluated as a distinct commercial-service path.

The reference runtime is the same pinned `mlx-audio` revision used by the
Qwen3-TTS spike. It runs over official Apple MLX and keeps the future Rust
boundary undecided.

Download and verify the 2.54 GB snapshot:

```sh
bash scripts/fetch-voxtral-tts-mlx-model.sh \
  /absolute/path/to/voxtral-4b-tts-2603-mlx-4bit
```

Create the locked Python environment:

```sh
uv sync \
  --project spikes/voxtral-tts-mlx-reference \
  --python /opt/homebrew/bin/python3.12
```

Run the fixed German diagnostic:

```sh
CUTTLEDOC_TTS_TEXT_DIR=/absolute/path/to/materialized-passages \
  bash scripts/run-voxtral-tts-mlx-reference.sh
```

The runner verifies every model artifact and the passage digest, then writes
digest-pinned f32 PCM, a PCM16 listening WAV, and a machine-readable result to
a local output directory. Generated audio remains local because both the input
passage and model require explicit attribution/license handling.
