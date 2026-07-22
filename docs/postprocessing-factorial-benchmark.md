# Factorial transcript-postprocessing benchmark

**Status:** design locked; native-language review and voice qualification block
execution.

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
2. Non-German and non-English source text and voice output have native-language
   review. Spanish `es-ES` and Portuguese `pt-PT` Voxtral presets remain
   regional proxies for the `es-419` and `pt-BR` text cells unless native review
   accepts that limitation.
3. Every Apple host voice identifier is resolved and every Qwen/Voxtral voice
   slot passes calibration and listening review.
4. Each generated master is retained losslessly, digest-pinned, and normalized
   once. All three STT models receive exactly the same normalized PCM.
5. All raw STT transcripts and LLM responses have durable paths before remote
   execution begins.
6. A dry-run token and cost estimate based on the completed multilingual
   documents is reviewed and an explicit budget is approved.

The current selection contains 35 passages from 21 pinned sources. Thirty
passages fill the factorial cells; five earlier German and English calibration
passages remain available outside the locked matrix. Every language/content
cell now has two independently digest-pinned texts. German and English use two
technical passages already in the calibration corpus, while the second factual
passages come from different article sections and the second dialogues are new
repository sources. The Spanish, French, and Portuguese candidates are
materialized but remain blocked until native-language review accepts the text,
regional variety, and dialogue idiom.

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
