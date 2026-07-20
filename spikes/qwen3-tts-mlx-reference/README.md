# Qwen3-TTS MLX reference

This spike runs the first open local synthesis candidate through the exact
German passage already used by the Apple system baseline. It is deliberately a
pinned Python reference implementation before a Rust product boundary is
selected.

The model, runtime, artifacts, voice, and generation parameters are frozen in
[`model-manifest.json`](./model-manifest.json). The model is German-capable,
but it has no native German preset speaker. The first result therefore fixes
the English preset `Ryan` for cross-lingual German synthesis and must report
that limitation.

The converted artifact is Apache-2.0 and the `mlx-audio` runtime is MIT. Its
model card names the official Qwen repository and `mlx-audio` 0.3.0 as the
conversion source, but does not expose the exact upstream commit used for the
conversion. The manifest keeps that provenance gap explicit instead of
inventing a source relationship.

Download and verify the 2.49 GB snapshot:

```sh
bash scripts/fetch-qwen3-tts-mlx-model.sh \
  /absolute/path/to/qwen3-tts-0.6b-customvoice-mlx-bf16
```

The fetcher downloads the exact Hugging Face revision and verifies the byte
count and SHA-256 digest of every file, including both LFS weight files.

Create the locked Python environment and run the fixed German diagnostic:

```sh
uv sync \
  --project spikes/qwen3-tts-mlx-reference \
  --python /opt/homebrew/bin/python3.12

CUTTLEDOC_TTS_TEXT_DIR=/absolute/path/to/materialized-passages \
  bash scripts/run-qwen3-tts-mlx-reference.sh
```

The runner verifies the model layout and passage digest, executes a fresh
non-streaming process, and writes digest-pinned f32 PCM, a PCM16 listening WAV,
and a machine-readable result to a local output directory. The generated audio
inherits the source passage's CC BY-SA handling and is not checked into Git.

The reference run remains development-diagnostic evidence. It can compare
local model behavior with Apple TTS, but it cannot substitute for professional
podcast or audiobook source recordings in the ASR selection gate.
