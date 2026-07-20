# Qwen3-ASR MLX reference probe

This probe measures whether a stronger current ASR model justifies extending
the repository-owned C++ adapter over official MLX. It is evidence for issue
#12, not a proposed production dependency.

The reference stack is pinned to:

- `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
  (Apache-2.0);
- `mlx-community/Qwen3-ASR-0.6B-8bit@89e96d92ba34aca20b3e29fb10cc284097d1219f`
  (Apache-2.0 conversion);
- `Blaizzy/mlx-audio` v0.4.5 at
  `04151c6abb74b886f879a4457ccdc96761f10102` (MIT); and
- official MLX 0.32.0 (MIT), supplied by the pinned reference environment.

`mlx-audio` and its Python model adapter remain reference-only under ADR-0005.
They do not enter a Cargo/npm manifest, a release build, or a stable API. The
probe emits owned JSON so the common harness can preserve transcripts, model
load, warm inference, first-token latency, process and MLX peak memory, and the
append-only token update stream.

For the repository-owned official-MLX port, the same disposable environment
can export deterministic encoder boundary fingerprints:

```sh
python spikes/qwen3-mlx-reference/export_encoder_oracle.py \
  /absolute/path/to/Qwen3-ASR-0.6B-8bit \
  /absolute/path/to/fixture.f32le
```

The checked-in first parity oracle is
[`benchmarks/oracles/qwen3-asr-0.6b.audiobook-en-2277-149874-0000.encoder.json`](../../benchmarks/oracles/qwen3-asr-0.6b.audiobook-en-2277-149874-0000.encoder.json).
It records shapes, float32 digests, and aggregate statistics from the log-Mel
input through the final audio embeddings. It is development evidence, not an
additional product dependency or quality result.

Create the disposable reference environment and model outside the repository:

```sh
UV_CACHE_DIR=/private/tmp/cuttledoc-uv-cache \
  uv venv --python 3.12 /private/tmp/cuttledoc-qwen3-asr/.venv

UV_CACHE_DIR=/private/tmp/cuttledoc-uv-cache \
  uv pip install \
    --python /private/tmp/cuttledoc-qwen3-asr/.venv/bin/python \
    mlx-audio==0.4.5

HF_HOME=/private/tmp/cuttledoc-qwen3-asr/hf-home \
  /private/tmp/cuttledoc-qwen3-asr/.venv/bin/hf download \
    mlx-community/Qwen3-ASR-0.6B-8bit \
    --revision 89e96d92ba34aca20b3e29fb10cc284097d1219f \
    --local-dir /private/tmp/cuttledoc-qwen3-asr/model
```

Run the same bounded multilingual matrix as the mandatory candidates:

```sh
CUTTLEDOC_QWEN3_PYTHON=/private/tmp/cuttledoc-qwen3-asr/.venv/bin/python \
CUTTLEDOC_QWEN3_MODEL_DIR=/private/tmp/cuttledoc-qwen3-asr/model \
  node scripts/run-phase0-asr-matrix.mjs \
    --candidate qwen3-mlx-reference \
    --repetitions 2 \
    --output /tmp/qwen3-mlx-reference.json
```

The probe reports the reference runtime's token emissions as streaming updates,
but marks their timestamps `unknown`: its token-position estimates are not
word or segment alignment. The official Qwen streaming path likewise requires
vLLM and does not return timestamps; official timestamps use a separate
Qwen3-ForcedAligner model.
