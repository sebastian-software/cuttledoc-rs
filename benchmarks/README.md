# Apple Silicon ASR benchmark harness

This directory is the machine-readable entry point for issue #4 and the later
#12 candidate sweep. Model and runtime identifiers are open strings, so adding
Nemotron, Qwen, Voxtral, Canary, or another candidate does not change schema
version 1.0.0.

Validate all checked-in manifests and run records:

```sh
node scripts/validate-benchmark-data.mjs --self-test
```

Materialize the exact legacy model artifacts:

```sh
node scripts/fetch-parakeet-baseline-models.mjs \
  --model-dir ~/.cache/parakeet-coreml/models \
  --vad-dir ~/.cache/parakeet-coreml/vad

node scripts/fetch-whisper-baseline-models.mjs \
  --model-dir ~/.cache/whisper-coreml/models
```

Both fetchers use immutable Hugging Face revisions and reject a directory whose
stable file-tree digest differs from the recorded baseline. They deliberately
leave multi-gigabyte artifacts outside Git.

Materialize the real Whisper Tiny encoder used by the direct MLX spike:

```sh
bash scripts/fetch-mlx-whisper-encoder-model.sh
```

That fetcher pins and verifies the converted NPZ and official MLX Examples mel
filter, then extracts the 66 encoder tensors outside Git. Run the complete
frontend/encoder on both CPU and GPU with an exact official MLX checkout:

```sh
CUTTLEDOC_MLX_SOURCE_DIR=/absolute/path/to/mlx-v0.32.0 \
  bash scripts/run-mlx-direct-spike.sh
```

Run a compatibility baseline against the sibling legacy implementations:

```sh
node scripts/run-legacy-asr-baseline.mjs \
  --backend parakeet \
  --fixture ../cuttledoc/packages/cuttledoc/fixtures/fleurs-en-000.ogg \
  --reference ../cuttledoc/packages/cuttledoc/fixtures/fleurs-en-000.txt \
  --module-dir ../parakeet-coreml \
  --model-dir ~/.cache/parakeet-coreml/models \
  --vad-dir ~/.cache/parakeet-coreml/vad \
  --repetitions 5 \
  --output /tmp/parakeet-fleurs-en-000.json
```

Replace `--backend`, `--module-dir`, and model options with the Whisper values
from its run record for that baseline. `--output` is important because native
diagnostics from a legacy addon may otherwise share stdout with the result.

A run is one immutable JSON document. `measured` records require a hashed,
provenance-audited quality fixture, complete quality/timing/resource metrics,
host RAM, and at least one raw artifact. Non-redistributable quality fixtures
must have a reproducible `local-required` acquisition path. `partial` and
`blocked` records require a precise explanation and may not silently supply
invented metrics. Raw tool output belongs below `benchmarks/raw/<run-id>/`; summaries in
`docs/model-evaluation.md` are derived evidence, not the source of truth.

## Fixture rules

- Record exact provenance, artifact license, language, normalization, and hash.
- Synthetic speech is useful for pipeline smoke tests but is not quality data.
- A local-required fixture may stay out of Git only after its expected digest
  and acquisition procedure are fixed in the manifest.
- Use the same normalized audio bytes for every candidate in a comparison.

WER and CER are lowercase, punctuation-insensitive Levenshtein distance divided
by reference word or character count. CER removes whitespace after the common
word normalization. Values in JSON are fractions, so `0.2105` is 21.05%.

## Timing and resource procedure

1. Record chip, RAM, OS build, power state, source revision, runtime/model
   revisions, quantization, and options before execution.
2. Measure cold load in a fresh process. Measure warm inference inside one
   long-lived engine so process startup is not confused with model work.
3. Preserve per-repetition raw values; the run record contains the declared
   aggregate and points back to those raw files.
4. Compute RTF as warm inference seconds divided by input-audio seconds.
5. Record first-result latency and update counts from the same ordered stream
   consumed by the contract reducer.
6. Measure peak resident memory for the complete process and record model and
   shipped binary/metallib sizes separately.

## Relative energy procedure

On the same plugged-in host, stabilize background load, use identical fixture
order, discard one warm-up, and alternate candidate order across at least five
runs. Capture CPU/GPU/ANE samples at 500 ms while the benchmark runs:

```sh
sudo powermetrics \
  --samplers cpu_power,gpu_power,ane_power \
  --sample-rate 500 \
  --format plist \
  --output-file benchmarks/raw/<run-id>/powermetrics.plist
```

Stop `powermetrics` immediately after the run and preserve its raw output.
Apple describes these power values as estimates, so use them only for relative
comparison on the same device, never as cross-device energy claims. Record an
external-meter method separately when used.
