# Text-generation MLX reference probe

This spike answered the runnable part of issue #7 with a deliberately small,
multilingual instruction model. It now also provides the manifest-driven common
reference runner for the model-quality matrix in issue #20. The historical
default remains `mlx-community/Qwen3-0.6B-4bit` at an exact revision through
`mlx-lm` over official Apple MLX on Apple Silicon.

The Python package is a pinned reference, not a proposed production boundary.
If Cuttledoc later embeds this backend, the product path remains a narrow,
repository-owned C++ task adapter over official MLX, matching the existing MLX
architecture decision. No `mlx-lm` type belongs in the Rust or Node API.

The default probe uses the German audiobook development fixture
`issue7-de-audiobook-whisper`. Only the language, domain, and raw Whisper text
are rendered into `surface-only-v1`; its unverified dataset reference remains
evaluation-only. The prompt forbids lexical changes, and the runner rejects an
output whose case-and-punctuation-insensitive word sequence differs.

Download and verify the 351 MB converted snapshot:

```sh
bash scripts/fetch-text-generation-mlx-model.sh \
  /absolute/path/to/qwen3-0.6b-4bit-mlx
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

Select another pinned candidate without changing the runner:

```sh
CUTTLEDOC_TEXT_GENERATION_MANIFEST=/absolute/path/to/candidate.json \
  bash scripts/fetch-text-generation-mlx-model.sh /absolute/path/to/model

CUTTLEDOC_TEXT_GENERATION_MANIFEST=/absolute/path/to/candidate.json \
  CUTTLEDOC_TEXT_GENERATION_MODEL_DIR=/absolute/path/to/model \
  CUTTLEDOC_TEXT_GENERATION_OUTPUT=/tmp/candidate-result.json \
  bash scripts/run-text-generation-mlx-reference.sh
```

The manifest freezes the source and conversion identities, artifact digests,
prompt path/hash, chat-template options, decoding contract, fixture, and result
identity. MLX-LM is only the common quality-reference layer; issue #20 compares
the quality survivor through owned official-MLX and Core ML product paths.

The run records cold model load, first-token latency, throughput, peak process
and MLX memory, exact output/token identifiers, mechanical gates, and a
cooperative cancellation probe. It is not a quality benchmark: one
development-exposed, unverified fixture cannot select a model or prompt.

Validate the pinned manifest, fixture, prompt, and checked-in measurement:

```sh
node scripts/validate-text-generation-mlx-reference.mjs --self-test
```
