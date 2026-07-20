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

Materialize the real Whisper Tiny model used by the direct MLX spike:

```sh
bash scripts/fetch-mlx-whisper-model.sh
```

That fetcher pins and verifies the converted NPZ plus the official MLX Examples
mel filter and multilingual tokenizer vocabulary, then extracts all 166 model
tensors outside Git. Run the end-to-end model on both CPU and GPU with an exact
official MLX checkout:

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

A single-fixture run is one immutable JSON document. `measured` records require
a hashed, provenance-audited quality fixture, complete
quality/timing/resource metrics, host RAM, and at least one raw artifact.
Non-redistributable quality fixtures must have a reproducible `local-required`
acquisition path. `partial` and `blocked` records require a precise explanation
and may not silently supply invented metrics. Raw tool output belongs below
`benchmarks/raw/<run-id>/`.
Comparable aggregate decisions belong below `benchmarks/matrices/`; the
validator derives and cross-checks their metrics against the linked raw
artifacts. `docs/model-evaluation.md` is explanatory evidence, not the source
of truth.

The validator also derives every checked-in multilingual aggregate directly
from its per-fixture results, including exploratory aggregates that have not
been promoted into a product-selection matrix.

The initial multilingual breadth set is intentionally bounded to two FLEURS
test samples in each of `en_us`, `de_de`, `es_419`, `fr_fr`, and `pt_br`.
Those ten quality fixtures are pinned in the manifest with source, reference,
and normalized-PCM digests. They expose language-path failures quickly; they
are not statistically sufficient for a final model-quality ranking.

The target benchmark design is language- and domain-aware. See
[`docs/asr-benchmark-design.md`](../docs/asr-benchmark-design.md) and the
machine-readable
[`source-candidates.json`](fixtures/source-candidates.json) registry. The
held-out execution contract is
[`target-domain-plan.json`](fixtures/target-domain-plan.json). It keeps German
podcast and audiobook cells first, rejects development-source overlap, and
requires an accepted source-specific rights review before acquisition. FLEURS
remains a short-read integration gate; audiobook and professional-podcast
cells are reported separately, with independent-source grouping and
human-verified gold transcripts.

Candidate-level rights reviews live under [`rights`](rights/) and are
validated with the rest of the benchmark data. They cannot authorize a
download or import. The exact deny-by-default workflow and the requirements for
an accepted source-group review are documented in
[`docs/source-rights-review.md`](../docs/source-rights-review.md). Once a
source-group review is accepted, import its digest-pinned local audio with:

```sh
node scripts/import-target-domain-source.mjs \
  --review benchmarks/rights/<source-group>.json \
  --source /absolute/path/to/original-audio \
  --output-dir /absolute/path/to/cuttledoc-target-domain
```

The first target-domain development input is the 15-clip multilingual
audiobook pilot. It deliberately remains outside Git as generated audio while
the repository pins dataset revisions, row metadata, source digests, and
normalized PCM digests. Materialize and verify it with:

```sh
node scripts/fetch-audiobook-pilot.mjs \
  --output-dir /absolute/path/to/cuttledoc-audiobook-pilot
```

The command rejects dataset-row, source-byte, or normalized-byte drift. It
requires network access and `ffmpeg`; set `FFMPEG=/absolute/path/to/ffmpeg`
when the binary is not on `PATH`. The dataset transcripts are not yet
human-verified surface-form gold, so the pilot may drive development and raw
lexical diagnostics but not correction-model acceptance.

Run a candidate on the materialized pilot by selecting its manifest and
fixture directory:

```sh
CUTTLEDOC_SPEECH_PROBE=/absolute/path/to/cuttledoc-speech-spike \
  CUTTLEDOC_SPEECH_LIBRARY_DIR=/absolute/path/to/build-directory \
  node scripts/run-phase0-asr-matrix.mjs \
    --candidate apple-speech \
    --manifest benchmarks/fixtures/audiobook-pilot.json \
    --fixture-dir /absolute/path/to/cuttledoc-audiobook-pilot \
    --repetitions 2 \
    --output /tmp/apple-speech-audiobook-pilot.json
```

The same manifest selection works for `whisper`, `parakeet`, and
`qwen3-mlx-reference`. Every backend receives the exact same digest-checked
float PCM; runtime-specific environment variables remain the ones documented
for the original matrix.

The checked-in development results are summarized in
[`matrices/phase0.audiobook-pilot-1.json`](matrices/phase0.audiobook-pilot-1.json).
Raw transcripts and repetitions remain under
[`raw`](raw/), and the language-specific, boundary-preserving edit report is
[`analysis/phase0.audiobook-pilot-1.errors.json`](analysis/phase0.audiobook-pilot-1.errors.json).

Historical Cuttledoc 2 transcript-correction evidence is preserved as a
digest-pinned aggregate in
[`postprocessing/cuttledoc-v2-snapshot.json`](postprocessing/cuttledoc-v2-snapshot.json).
Its TTS and real-ASR experiments, language splits, missing results, and legacy
normalization limitations are documented in
[`docs/postprocessing-evaluation.md`](../docs/postprocessing-evaluation.md).
The next prompt candidates are immutable benchmark inputs under
[`postprocessing/prompts`](postprocessing/prompts/); their hashes, edit
permissions, context fields, output contracts, and mechanical rejection gates
are pinned by the prompt manifest.

Run one candidate across the bounded set and write its immutable aggregate
outside the repository before review:

```sh
CUTTLEDOC_MLX_PROBE=/absolute/path/to/cuttledoc-mlx-whisper \
  CUTTLEDOC_MLX_MODEL_DIR=/absolute/path/to/mlx-whisper-tiny \
  node scripts/run-phase0-asr-matrix.mjs \
    --candidate mlx \
    --repetitions 2 \
    --output /tmp/mlx-multilingual.json
```

The same command accepts `apple-speech`, `parakeet`, and `whisper`. Their
required probe, module, model, and VAD paths are named in missing-environment
errors so local artifacts never become hidden repository assumptions. Every
candidate uses the same manifest order and quality normalization; the raw
aggregate retains each transcript, timing repetition, segment list, and
candidate-specific streaming behavior.

Issue #12 also supplies `qwen3-mlx-reference`. It intentionally runs the
reference-only Python port in an external disposable environment while keeping
the model, runtime, and result shape pinned. See
[`spikes/qwen3-mlx-reference`](../spikes/qwen3-mlx-reference/) for the exact
materialization and the boundary between useful MLX evidence and an accepted
product dependency.

## Fixture rules

- Record exact provenance, artifact license, language, normalization, and hash.
- Synthetic speech is useful for pipeline smoke tests but is not quality data.
- A local-required fixture may stay out of Git only after its expected digest
  and acquisition procedure are fixed in the manifest.
- Use the same normalized audio bytes for every candidate in a comparison.

WER and CER are lowercase, punctuation-insensitive Levenshtein distance divided
by reference word or character count. CER removes whitespace after the common
word normalization. Values in JSON are fractions, so `0.2105` is 21.05%.

Generate deterministic word-level edit alignments for every checked-in
multilingual aggregate:

```sh
node scripts/analyze-asr-errors.mjs \
  --output benchmarks/analysis/phase0.multilingual-fleurs-10-1.errors.json

node scripts/analyze-asr-errors.mjs \
  --check benchmarks/analysis/phase0.multilingual-fleurs-10-1.errors.json
```

The report reproduces the immutable phase-0 WER and also provides a review
view that preserves boundaries around hyphens, dashes, and slashes. This avoids
reviewing `T-Rex` as `trex` or `25-30` as `2530`. It separates substitutions,
omissions, and insertions by candidate, language, and fixture. Numeric and
negation changes receive mechanical risk hints. Semantic severity remains an
explicit review field; the script never pretends it can infer whether a lexical
difference is harmless.

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
