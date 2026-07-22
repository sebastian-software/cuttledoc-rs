# Factorial transcript-postprocessing benchmark

**Status:** design locked; voice qualification blocks execution.

**Evidence date:** 2026-07-22.

This benchmark compares transcript postprocessing under controlled variation in
language, source text, speech synthesis, voice, generation, speech recognition,
and language model. It replaces one-dimensional model screens with a paired
factorial design while retaining an explicit limit: synthetic clean speech can
select development survivors, but it cannot establish release quality on human
podcasts or audiobooks.

The machine-readable plan is
[`factorial-plan.json`](../benchmarks/postprocessing/factorial-plan.json). The
complete execution ledger is generated deterministically as
[`factorial-cells.json`](../benchmarks/postprocessing/factorial-cells.json).

## Locked primary matrix

| Axis | Locked values | Replication |
| --- | --- | ---: |
| Language | `de-DE`, `en-US`, `es-419`, `fr-FR`, `pt-BR` | 5 |
| Content | technical, native factual, dialogue | 3 |
| Independent source passages | separately pinned texts within each language/content cell | 2 |
| TTS | Apple AVSpeechSynthesizer, Qwen3-TTS 1.7B VoiceDesign, Voxtral 4B TTS BF16 | 3 |
| Voice | engine-specific voice identities within each language | 2 |
| Primary generation | engine-specific realizations described below | 2 |
| STT | Whisper large-v3-turbo, direct Qwen3-ASR 0.6B/MLX, Parakeet TDT 0.6B v3 | 3 |
| Hosted postprocessor | GPT-5.6 Sol, Gemini 3.6 Flash, GPT-5.6 Terra, Kimi K3, Claude Sonnet 5 | 5 |
| Hosted repeat | identical prompt, fixture, model, and pinned provider | 2 |

The two passages are source replication, not two excerpts from one synthesized
recording. They prevent one unusually easy paragraph from defining an entire
content type. The two voices expose speaker and presentation sensitivity. The
two generation realizations expose synthesis variance, and the two LLM repeats
expose hosted-response variance. These dimensions remain separate in every
record.

## Seed and repeat semantics

A nominal `seed=0,1` loop would not mean the same thing for all three TTS
engines. The locked contract therefore records engine semantics instead of
pretending the APIs are identical:

- Apple exposes no generation seed. Two independent calls with the same pinned
  installed voice measure host/runtime output stability.
- Qwen VoiceDesign uses the seed as part of the generated voice identity. Each
  of the two voice slots has its own fixed instruction and identity seed; two
  calls reuse that exact identity.
- Voxtral uses a pinned voice preset. Seeds 0 and 1 create the two primary
  realizations without changing the preset.
- Voxtral also receives one extra seed-0 generation for both voices on the
  first technical passage in every language. These ten audio files are a
  same-seed reproducibility control and do not enter the LLM matrix.

Every stochastic input, identity seed, repeat number, and random execution
order is stored. A `repeat` is never silently substituted for a missing source
passage or voice.

## Derived workload

| Artifact or unit | Count |
| --- | ---: |
| Passage slots | 30 |
| Voice slots | 30 |
| Primary audio artifacts | 360 |
| Voxtral same-seed control audio | 10 |
| Total STT transcripts | 1,110 |
| Six-section LLM documents | 180 |
| Document/model pairs | 900 |
| Provider requests after two repeats | 1,800 |
| Primary section-level model outputs | 5,400 |
| Section-level repeat observations | 10,800 |

Each LLM document groups the six passages for one language, TTS engine, voice,
primary realization, and STT model. The sections remain independently scored,
but the model receives realistic multi-page context. The hidden source text is
never included in the prompt.

The 10,800 section observations are not 10,800 independent samples. The same
30 source passages occur under multiple voices, realizations, recognizers, and
postprocessors. Reports therefore keep passage and source identity, publish
per-language results, use language-level macro averages instead of pooling all
words, and use source-grouped confidence intervals. Repeats estimate variance;
they do not inflate the source sample count.

## Prompt and local audit

The matrix uses
[`conservative-sections-local-diff-v2.txt`](../benchmarks/postprocessing/prompts/conservative-sections-local-diff-v2.txt).
The model returns only immutable section ids and complete corrected section
texts. It does not author an edit ledger, reasons, or confidence scores.

The repository derives the authoritative input/output lexical diff, checks
protected spans, computes section and aggregate WER, and retains every raw
response. This removes the bookkeeping failure seen when otherwise strong
models omitted or duplicated ledger entries. It does not weaken the quality
gate: unsupported changes to correct names, numbers, dates, units, terms, or
negations are counted as harmful edits and reviewed independently of aggregate
WER.

## Execution gates

The primary matrix cannot start until all of the following are true:

1. All 30 passage slots remain materialized, revision- and digest-pinned,
   rights-reviewed, and appropriate for their declared content type.
2. Every Apple host voice identifier is resolved and every Qwen/Voxtral voice
   slot passes calibration and listening review.
3. Each generated master is retained losslessly, digest-pinned, and normalized
   once. All three STT models receive exactly the same normalized PCM.
4. All raw STT transcripts and LLM responses have durable paths before remote
   execution begins.
5. A dry-run token and cost estimate based on the completed multilingual
   documents is reviewed and an explicit budget is approved.

The current selection contains 35 passages from 21 pinned sources. Thirty
passages fill the factorial cells; five earlier German and English calibration
passages remain available outside the locked matrix. Every language/content
cell now has two independently digest-pinned texts. German and English use two
technical passages already in the calibration corpus, while the second factual
passages come from different article sections and the second dialogues are new
repository sources. The Spanish, French, and Portuguese candidates are
accepted for the baseline without a mandatory native-language review. If an
execution exposes wording, regional-variety, or dialogue-idiom problems, a
future corpus revision can correct them without invalidating the pinned
baseline results.

Spanish `es-ES` and Portuguese `pt-PT` Voxtral presets remain explicitly
labeled regional proxies for the `es-419` and `pt-BR` text cells. This limits
how broadly their results may be generalized, but does not block execution.

## Remote execution and cost boundary

Every hosted candidate remains pinned to one provider, with fallback disabled,
required parameters enabled, denied data collection, and ZDR required. The
consumed one-time non-ZDR Qwen3.7 Max exception is not part of this matrix.

Using the recorded cost of the current 995-word German document as a reference,
five models and two requests cost approximately `$0.714417` per future
document. Applied to 180 documents, the provisional total is `$128.59506`.
This is a projection, not authorization: multilingual tokenization, future
document length, and provider pricing may change it.

Requests are assigned to two deterministic randomized blocks. Each block
contains every one of the 900 document/model pairs exactly once. This prevents
selectively omitting expensive or failed cells after observing results and
keeps a stopped block balanced. A partial block is not a completed benchmark,
and the second block remains required for the planned repeat analysis.

## Decision and reporting boundary

Primary metrics are raw and postprocessed WER, relative error reduction,
harmful-edit rate on correct input spans, section-regression rate, proper-name
recall, number/date/unit recall, technical-term recall, and presentation-only
changes. Results are reported by language, content type, passage, TTS engine,
voice, realization, STT model, and LLM model before any macro summary.

The synthetic matrix may remove weak or unsafe development candidates and
identify language-specific defaults. A frozen survivor still requires unseen,
human-verified professional podcast or audiobook audio before a release claim.
The user-facing product may continue to expose multiple STT engines even when
one becomes the default.

Validate the complete design and regenerate/check the ledger with:

```sh
node scripts/generate-postprocessing-factorial-matrix.mjs --check
node scripts/validate-postprocessing-factorial-plan.mjs --self-test
node spikes/text-generation-openrouter-reference/run_reference.mjs --self-test
```

## Resumable local execution

The local TTS and STT stages write large, host-specific artifacts below the
ignored `artifacts/postprocessing-factorial-local` directory by default. The
runner verifies all 30 materialized passages before creating state, records
host capabilities, resolves Apple voice identifiers from the installed voice
inventory, and treats every audio unit as an independently digest-pinned
checkpoint. A valid existing checkpoint is resumed rather than regenerated.

Initialize and inspect the local run with:

```sh
node scripts/run-postprocessing-factorial-local.mjs init
node scripts/run-postprocessing-factorial-local.mjs resolve-voices
node scripts/run-postprocessing-factorial-local.mjs status
```

Run one qualification artifact per resolved Apple voice before expanding the
full Apple slice:

```sh
node scripts/run-postprocessing-factorial-local.mjs run-apple-tts \
  --qualification-only
```

Transcribe every locally completed qualification artifact through the three
locked STT engines. Whisper engines are reused per language, Parakeet is reused
across languages, and the repository-owned direct Qwen adapter runs against the
same persisted normalized PCM digest:

```sh
node scripts/run-postprocessing-factorial-local.mjs run-stt \
  --qualification-only --backend all
node scripts/run-postprocessing-factorial-local.mjs summarize-qualification
```

The summary command keeps raw audio and transcripts in the ignored artifact
directory, but writes a compact, digest-pinned qualification report to
`benchmarks/postprocessing/qualifications`. This first gate only detects broken
or unintelligible synthesis before expansion. It deliberately retains
recognizer disagreement and does not turn a single synthetic passage into an
engine ranking or a claim about human podcast and audiobook quality.

The Qwen VoiceDesign qualification uses the same ten-cell slice. Its batch
worker verifies the pinned BF16 snapshot once, loads the MLX model once, then
checkpoints every voice independently before the shared normalization and STT
stages:

```sh
node scripts/run-postprocessing-factorial-local.mjs run-qwen-tts \
  --qualification-only
node scripts/run-postprocessing-factorial-local.mjs run-stt \
  --qualification-only --backend all
node scripts/run-postprocessing-factorial-local.mjs summarize-qualification \
  --engine qwen
```

Qwen repeats reuse the voice slot's identity seed exactly. Long inputs use the
pinned `sentence-aware-adaptive-v2` policy: it prefers sentence boundaries
after 45 words, enforces a hard 55-word limit, resets the qualified identity
seed for every chunk, and inserts 250 ms of silence between chunks. Every chunk
must finish normally below the 1,200-token ceiling. A pathological chunk that
hits the ceiling is deterministically bisected and retried down to four words;
if it still cannot finish, the complete audio unit fails instead of silently
turning missing speech into STT errors.

If a pinned Qwen identity fails that gate, the recovery plan tests the next
three seeds without changing its description. It selects the first seed that
finishes and stays below the direct-Qwen receiver threshold, rather than
choosing the lowest WER after observing all candidates:

```sh
node scripts/run-postprocessing-factorial-local.mjs run-qwen-recovery
```

The recovery result is only a proposal for a factorial-plan revision. The
selected identities still have to rerun the complete three-receiver
qualification under that new revision before the full slice can expand.

Plan revision 4 applied the bounded recovery result: `qwen-de-clear` moved
from identity seed 1 to 2 and `qwen-es-clear` from seed 1 to 3. The other eight
Qwen slots passed the first three-receiver gate. The complete revision-4
rerun then passed all ten identities without a token-limit failure. Plan
revision 5 consequently marks every Qwen slot qualified and makes all 120
Qwen audio cells ready; Apple host voices and the exact Voxtral BF16 artifact
remain separate execution gates.

Plan revision 6 also promotes the ten Apple voices that passed the local
three-receiver screen. Their exact AVSpeechSynthesizer identifiers and actual
installed locales are pinned in the plan; cross-region voices remain visible
as such instead of being relabeled as `es-419`, `fr-FR`, or `pt-BR`. A host
must expose those exact identifiers before the Apple slice can run.

After all 120 audio and 360 STT artifacts for one engine are present, create a
compact repository report while leaving raw PCM and transcripts ignored:

```sh
node scripts/run-postprocessing-factorial-local.mjs summarize-slice \
  --engine apple \
  --output-dir artifacts/postprocessing-factorial-local-plan-6
```

The report retains every cell's digests and WER/CER, publishes language,
content, voice, and recognizer strata, and separately measures whether the two
generation repeats produced identical audio and transcripts.

## Current complete local slices

Plan revision 6 has complete Apple and Qwen VoiceDesign slices. The values
below are macro mean channel WER percentages: they measure the complete
TTS-to-STT path and must not be presented as isolated STT accuracy.

| TTS source | STT receiver | Overall | German | English | Spanish | French | Portuguese |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | Whisper | 3.32 | 4.39 | 2.18 | 2.48 | 3.15 | 4.38 |
| Apple | Qwen3-ASR | 8.60 | 12.90 | 7.65 | 4.49 | 5.93 | 12.04 |
| Apple | Parakeet | 7.56 | 8.48 | 10.21 | 3.08 | 6.74 | 9.31 |
| Qwen VoiceDesign | Whisper | 2.18 | 1.91 | 2.22 | 1.29 | 2.58 | 2.92 |
| Qwen VoiceDesign | Qwen3-ASR | 7.59 | 7.59 | 5.71 | 7.28 | 4.67 | 12.69 |
| Qwen VoiceDesign | Parakeet | 7.91 | 4.65 | 5.10 | 10.63 | 7.17 | 12.02 |

Across all three receivers, Apple has 6.49% macro mean WER and Qwen has
5.89%. That aggregate hides material reversals by language and receiver, so the
language strata remain the selection evidence. Both slices produced 60/60
identical normalized repeat pairs and 180/180 identical transcript pairs.
Qwen emitted 418 successful chunks across 120 documents; two documents needed
one deterministic adaptive retry each, no document remained capped, and the
largest successful chunk used 441 of the allowed 1,200 tokens.

The runner retains both the engine-native mono `f32le` master and one derived
16 kHz mono `f32le` normalization. The normalized digest is the single input
that all STT engines must share. Qwen and Voxtral execution is enabled only
when the exact model format pinned by the factorial plan is present; an older
or differently quantized local snapshot is reported as unavailable rather
than substituted silently.

## Local LLM screen

The three pinned MLX text-generation candidates are evaluated as a local
development screen alongside, not in place of, the locked five-model hosted
matrix. The local contract is recorded in
[`local-llm-screen.json`](../benchmarks/postprocessing/local-llm-screen.json).
It applies Gemma 4 E2B, Qwen 3.5 0.8B, and SmolLM3 3B to the 120 currently
available Apple and Qwen six-section documents, twice each. That produces 720
local requests while preserving the same hidden-reference and local-diff
rules as the hosted comparison.

Materialize the six-section documents from the digest-checked STT checkpoints:

```sh
node scripts/run-postprocessing-factorial-local-llm.mjs materialize
node scripts/run-postprocessing-factorial-local-llm.mjs status
```

Run one resumable candidate or all three. The worker verifies the exact model
snapshot before loading it once for the complete selected batch, retains every
raw response, and rejects missing, reordered, or duplicate section ids:

```sh
node scripts/run-postprocessing-factorial-local-llm.mjs run \
  --candidate qwen3.5-0.8b-4bit-mlx
node scripts/run-postprocessing-factorial-local-llm.mjs run --candidate all
node scripts/run-postprocessing-factorial-local-llm.mjs summarize
```

`--locale`, `--limit`, and `--repetitions` provide bounded diagnostic slices;
they do not change the locked full-screen counts. Gemma remains
evaluation-only pending its explicit product-rights disposition, and mlx-lm
remains the quality-reference runtime rather than a product dependency. The
summary keeps strict contract compliance separate from diagnostic recovery:
for example, JSON inside a prohibited Markdown fence can be inspected for the
cause of a failure, but it never contributes to accepted postprocessed WER.
