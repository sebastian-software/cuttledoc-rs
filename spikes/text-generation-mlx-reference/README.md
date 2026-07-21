# Text-generation MLX reference probe

This spike answers the runnable part of issue #7 with a deliberately small,
multilingual instruction model. It runs `mlx-community/Qwen3-0.6B-4bit` at an
exact revision through `mlx-lm` over official Apple MLX on Apple Silicon.

The Python package is a pinned reference, not a proposed production boundary.
If Cuttledoc later embeds this backend, the product path remains a narrow,
repository-owned C++ task adapter over official MLX, matching the existing MLX
architecture decision. No `mlx-lm` type belongs in the Rust or Node API.

The probe uses the German audiobook development fixture
`issue7-de-audiobook-whisper`. Only the language, domain, and raw Whisper text
are rendered into `surface-only-v1`; its unverified dataset reference remains
evaluation-only. The prompt forbids lexical changes, and the runner rejects an
output whose case-and-punctuation-insensitive word sequence differs.

Download and verify the 351 MB converted snapshot:

```sh
bash scripts/fetch-text-generation-mlx-model.sh \
  /absolute/path/to/qwen3-0.6b-4bit
```

Create the locked environment and run the probe:

```sh
uv sync \
  --project spikes/text-generation-mlx-reference \
  --python /opt/homebrew/bin/python3.12

CUTTLEDOC_TEXT_GENERATION_MODEL_DIR=/absolute/path/to/qwen3-0.6b-4bit \
  CUTTLEDOC_TEXT_GENERATION_OUTPUT=/tmp/cuttledoc-text-generation-result.json \
  bash scripts/run-text-generation-mlx-reference.sh
```

The run records cold model load, first-token latency, throughput, peak process
and MLX memory, exact output/token identifiers, mechanical gates, and a
cooperative cancellation probe. It is not a quality benchmark: one
development-exposed, unverified fixture cannot select a model or prompt.
