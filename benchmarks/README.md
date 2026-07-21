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

The first concrete selection is
[`target-domain-corpus.json`](fixtures/target-domain-corpus.json). It pins three
CC-BY-4.0 German podcast episodes, three exact ten-minute ranges, five speakers,
validation/test isolation, publisher-draft transcript digests, and deterministic
mono 16 kHz float PCM. Its gold status deliberately remains
`pending-independent-human-review`. Materialize or re-verify the normalized
audio outside Git with:

```sh
node scripts/materialize-target-domain-corpus.mjs \
  --input-dir /absolute/path/to/cuttledoc-target-domain \
  --output-dir /absolute/path/to/cuttledoc-target-domain/normalized
```

The primary clean-speech decision-support contract is
[`synthetic-roundtrip-plan.json`](fixtures/synthetic-roundtrip-plan.json)
under issue #13. It keeps German primary and reports `de-DE`, `en-US`,
`es-419`, `fr-FR`, and `pt-BR` separately, pins exact source revisions or digests, and
requires at least three pinned voices per locale across all three required TTS
engines before producing a recommendation. A bounded three-content-type
calibration runs per primary locale before the full matrix. Every generated artifact crosses the
same five ASR backends: Apple SpeechTranscriber, Whisper, direct Qwen3-ASR,
Parakeet, and direct Voxtral Realtime. Apple `AVSpeechSynthesizer` is the
system baseline; Qwen3-TTS 1.7B VoiceDesign and Voxtral TTS BF16 are the two
required MLX reference generators. KugelAudio receives one German challenger
run, while Chatterbox and the remote Qwen provider ceiling are deferred and
optional respectively. Synthetic
results may inform the default model recommendation for clean produced speech,
but do not establish universal superiority or release-quality real-world WER.
The professional German podcast corpus remains a separate optional long-form
control. See
[`docs/synthetic-roundtrip-benchmark.md`](../docs/synthetic-roundtrip-benchmark.md)
and the dated
[`docs/speech-engine-shortlist-2026-07.md`](../docs/speech-engine-shortlist-2026-07.md).
The exact model/runtime revisions, artifact digests, fixed voice profiles, and
generic verified fetch command live in
[`spikes/tts-calibration`](../spikes/tts-calibration/README.md). KugelAudio is
currently limited to an implicit default voice and does not satisfy
multi-voice coverage.

The selection contains technical, native-factual, and conversational-dialogue
cells for each primary locale. Spanish and Portuguese remain explicit regional
proxies: results apply to the pinned text and voice, not to every `es-419` or
`pt-BR` variety.
Materialize the eleven digest-pinned passage files and their CC BY-SA
attribution package outside Git with:

```sh
node scripts/materialize-synthetic-roundtrip.mjs \
  --output-dir /absolute/path/to/cuttledoc-synthetic-roundtrip
```

The command retrieves three exact MediaWiki revisions, verifies page and parent
revision metadata, reads one repository-authored source at its pinned SHA-256,
resolves fixed section/paragraph selectors, and rejects any character-count or
digest drift. The checked-in
[`synthetic-roundtrip-selection.json`](fixtures/synthetic-roundtrip-selection.json)
contains selectors and digests. Lossless generated audio remains local. The
separately licensed [`assets`](assets/README.md) tree may contain reviewed Ogg
Opus copies with complete attribution and a passed codec control; it currently
contains the codec-control Qwen fixture, paired clear and warm German Qwen
VoiceDesign fixtures, the warm English fixture, and the native-German content
control.

Validate its five-engine lossless/48/64/96 kbit/s evidence and the committed
asset hashes with:

```sh
node scripts/validate-opus-codec-control.mjs --self-test
```

The first measured TTS lifecycle record is
[`phase5.apple-tts.synthetic-de-origin-1`](raw/phase5.apple-tts.synthetic-de-origin-1/result.json).
It records the native Apple voice PCM metadata and digest, Rust/Swift
ownership, timing/resources, stable busy/cancel statuses, and the restricted
process-context control. The raw CC BY-SA-derived audio remains local.
The reviewed compact Qwen codec-control copy is committed separately under
`benchmarks/assets`; it is not a replacement for the lossless TTS artifact.

The current German Qwen VoiceDesign calibrations are the paired
[`qwen-de-clear-documentary`](raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-de-clear-documentary.1/result.json)
and
[`qwen-de-warm-podcast`](raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-de-warm-podcast.1/result.json)
runs. Both record the pinned model/profile, f32 synthesis metrics, and five ASR
receivers on one normalized PCM digest per run. All five receivers fail at
`1962` for the clear profile and recover it for the warm profile, proving the
critical failure is voice/profile-specific. The subsequent
[`qwen-en-warm-podcast`](raw/phase5.qwen3-tts-1.7b-voicedesign.qwen-en-warm-podcast.1/result.json)
run passes the English lexical gate with exact normalized character content
from all five receivers. The native-German `Buchdruck` cell subsequently
passes with complete critical-fact recovery and 1.92–2.88% WER across the five
receivers. The dialogue cell holds the accepted warm description
constant: four receivers stay at or below two word edits, while Parakeet alone
makes seven. The three German content cells are therefore lexically complete;
listening review precedes Voxtral calibration.

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

The first repository-owned Voxtral boundary over official MLX is reproduced
with:

```sh
scripts/run-voxtral-mlx-direct-spike.sh
```

Its checked-in
[`boundary record`](raw/phase0.voxtral-realtime-mlx-direct-boundary-1/result.json)
machine-checks the pinned model layout, bounded 80 ms feed/320 ms step
contract, backpressure, and cancellation. The separate
[`frontend result`](raw/phase0.voxtral-realtime-mlx-direct.frontend-480ms-1/result.json)
matches the pinned
[`Python oracle`](oracles/voxtral-realtime.audiobook-de-135_82_000105.frontend-480ms.json)
through offline-streaming padding, log-mel extraction, and both causal Conv1d
stages. The
[`encoder result`](raw/phase0.voxtral-realtime-mlx-direct.encoder-480ms-1/result.json)
then proves all 32 causal layers, repository-owned rotating cache and
sliding-window semantics, downsampling, and adapter projection against its
pinned oracle. The
[`decoder result`](raw/phase0.voxtral-realtime-mlx-direct.transcription-480ms-1/result.json)
adds delay conditioning, all 26 GQA decoder layers, decoder caches, the tied
language-model head, greedy generation, and Tekken decoding. It matches all 178
tokens and the final German text from its pinned Python oracle exactly. This is
a usable complete-buffer Rust transcription path and implementation-parity
evidence, not held-out ASR-quality or live-input streaming evidence.

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
- Synthetic speech is controlled decision-support for clean-speech model
  selection, but is not standalone release-quality or real-world evidence.
- A local-required fixture may stay out of Git only after its expected digest
  and acquisition procedure are fixed in the manifest.
- A repository media fixture requires separate licensing, attribution,
  provenance, a lossless-versus-codec control, and product-package exclusion.
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
