# Long-form transcript postprocessing screen

**Status:** German multi-page development screen complete for nine hosted
models; the pinned Mistral endpoint remains temporarily blocked.

**Evidence date:** 2026-07-22.

The original hosted screen used one 39-word transcript with one disputed word.
It established model and output-contract capability, but it could not estimate
correction coverage, regressions, or behavior across different error classes.
This screen replaces that quality input with eight independent German sections:

- 995 normalized reference words, approximately two to three A4 pages;
- 506.7 seconds of clean synthetic speech;
- eight pinned German Apple system voices;
- documentary, technical, native factual, and dialog content; and
- 60 raw Whisper word errors, or 6.03% micro-WER.

The exact source texts come from the pinned synthetic-round-trip selection.
Each section was synthesized separately, normalized to 16 kHz mono, and
transcribed by the same `whisper-large-v3-turbo-coreml-whispercpp` backend.
The fixture retains each source revision, voice identifier, lossless and
normalized audio digest, raw transcript, and exact source reference. Generated
audio remains local-required. References are used only after inference and are
never rendered into the model prompt.

## Results

Every complete model run contains two provider-pinned requests with fallback
disabled, required parameters, and denied data collection. All use a
zero-data-retention route except the explicitly authorized, already consumed
Qwen3.7 Max exception on public synthetic text. Cost covers the two recorded
requests per run.

| Candidate | Contract | Output errors | Micro-WER | Error reduction | Sections improved / unchanged / regressed | Recorded cost |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Raw Whisper | n/a | 60 | 6.03% | n/a | n/a | n/a |
| Qwen 3.5 122B-A10B / AtlasCloud FP8 | Passed | 25 | 2.51% | 58.3% | 6 / 1 / 1 | $0.0254754 |
| GPT-5.6 Sol / Azure EU | Failed: one unreported lexical change | 5 | 0.50% | 91.7% | 6 / 2 / 0 | $0.2455266 |
| Claude Sonnet 4.6 / Bedrock EU West 1 (historical) | Passed | 31 | 3.12% | 48.3% | 5 / 3 / 0 | $0.1521399 |
| Gemini 3.6 Flash / Google Vertex Global | Failed: one unreported lexical change | 9 | 0.90% | 85.0% | 6 / 1 / 1 | $0.0716760 |
| Kimi K3 / Moonshot AI INT4 | Passed | 23 | 2.31% | 61.7% | 6 / 2 / 0 | $0.1387416 |
| GPT-5.6 Luna / Azure EU | Failed: all six lexical changes unreported | 54 | 5.43% | 10.0% | 1 / 7 / 0 | $0.0294822 |
| GPT-5.6 Terra / Azure EU | Failed: one redundant overlapping ledger entry | 20 | 2.01% | 66.7% | 6 / 2 / 0 | $0.1135068 |
| Claude Sonnet 5 / Azure US East 2 | Passed | 28 | 2.81% | 53.3% | 5 / 3 / 0 | $0.1449660 |
| Qwen3.7 Max / Alibaba, consumed non-ZDR exception | Failed: four repeated edit occurrences not enumerated | 15 | 1.51% | 75.0% | 6 / 2 / 0 | $0.044950625 |
| Mistral Small 3.2 24B / Parasail BF16 | Blocked: upstream HTTP 429 twice | — | — | — | — | not recorded |

Gemini was the only one of the nine complete long-form results whose two raw
responses were identical. The recorded cost of the nine complete two-request
results is $0.966465125. Requests from the blocked Mistral attempts, an
abandoned historical Qwen invocation, and Sonnet 5's initial token-limit
failure have no complete response record and are intentionally excluded rather
than estimated.

## Findings

GPT-5.6 Sol is the clear text-quality ceiling on this development fixture. It
restored the difficult proper-name and English-term section exactly, made no
section worse, and left only five normalized disagreements. It nevertheless
changed `Vorläufe` to `Vorläufer` without listing that lexical edit, so the
repository-owned cross-field contract correctly rejects the result. Its
remaining mismatches also show why WER is not a complete correctness label:
`circa` versus `ca.`, `universelle` versus source `universale`, and digits
versus a number word include convention choices, while retained `erzielen`
versus `erzwingen` is a substantive missed correction.

Qwen 3.5 fixed many difficult names and technical terms and reported every
lexical change. It also rewrote an already exact section from `Industrie 5.0` and
`2025` to `Industrie 4.0` and `2023`. Those two unsupported factual changes are
precisely the high-severity regression that an aggregate WER improvement would
hide. Qwen 3.5 therefore passes the mechanical ledger contract but is not a
safe quality candidate under this prompt.

Claude Sonnet 4.6 was the only original result to combine a complete edit
ledger with zero section regressions. It was substantially more conservative:
it left several recoverable names, abbreviations, spoken punctuation labels,
and the closing `Vielen Dank` hallucination untouched. It is the safest
complete original result here, not the most capable one.

The expanded current-model screen changes that comparison without producing a
winner. Gemini 3.6 Flash came closest to GPT-5.6 Sol at nine errors and repeated
byte-identically, but it changed an already correct `Industrie 5.0` to
`Industrie 4.0` based on world knowledge. It reported that harmful change, yet
omitted one other lexical change from its ledger, so both the section-regression
gate and cross-field audit caught independent problems. The Google endpoint
also rejects disabled reasoning; explicit low effort is therefore part of the
pinned candidate tuple.

Kimi K3 is the strongest new result that passed the entire contract: 23 errors,
six improved sections, and no regression. It is less accurate than GPT-5.6 Sol
and Gemini on this fixture but safer under the current bounded prompt. Claude
Sonnet 5 also passed completely with no regression, but its 28 errors are only
a modest improvement over the historical Sonnet 4.6 result's 31. Sonnet 5 is
now the active Anthropic control; Sonnet 4.6 remains in the table solely as
historical evidence, not because the old model is still preferred.

The three GPT-5.6 tiers form the expected quality ordering under the identical
Azure EU no-reasoning tuple. Sol remains the ceiling at five errors, balanced
Terra leaves 20, and cost-efficient Luna leaves 54. Terra improves six sections
without a regression and is slightly more accurate than Kimi K3. Its mechanical
failure is bookkeeping rather than a hidden output change: the final spoken
`Punkt` deletion is reported once on its own and again in the overlapping
`Punkt gute` to `Gute` edit. The ledger therefore counts 51 changes for a
locally derived 50-change output diff.

Sonnet 5 initially consumed the full 8,192-token completion budget without
emitting content; explicit low reasoning produced the complete record above.
Client-observed request durations ranged from about 0.4 seconds for Terra to
roughly 10.2 seconds for Kimi, but these remote gateway measurements are not
comparable to local runtime latency.

Qwen3.7 Max is the third-best text restorer in the complete matrix at 15 errors,
behind GPT-5.6 Sol and Gemini. It improved six sections without a regression.
Its mechanical rejection is narrower than Gemini's: `Key` to `KI` was applied
three times but reported as one correction rule, and `Keywinter` to `KI-Winter`
was applied twice but reported once. The output is therefore short by four
per-occurrence ledger entries even though those repeated edits are beneficial.
This reinforces the decision to derive the authoritative diff locally rather
than requiring models to enumerate identical occurrences.

Alibaba was the only cataloged Qwen3.7 Max endpoint and was absent from
OpenRouter's dynamic ZDR list. The repository owner explicitly authorized one
execution on this public synthetic fixture. The recorded request kept Alibaba
pinned, fallback disabled, and data collection denied; only ZDR was relaxed.
The manifest now marks the exception consumed, and the runner rejects another
execution. This is evidence for this model screen, not a general privacy-policy
change.

The audit now verifies edit completeness by applying reported, non-overlapping
lexical replacements to the normalized input in source order and comparing the
reconstruction with the normalized output. This avoids false failures caused
by non-unique Levenshtein alignments for multiword edits. Output WER and the
input-to-output diff remain independently derived.

## Decision boundary

This is development-only synthetic round-trip evidence. It is materially more
informative than the one-word probe, but it does not select a product model:

- system TTS can pronounce punctuation labels or abbreviations in ways a human
  narrator would not;
- exact source restoration can penalize benign conventions or reward source
  wording that deserves human review;
- the texts and error profile are development-exposed; and
- no result covers held-out professional podcast or audiobook audio.

The follow-up prompt now separates correction quality from audit metadata.
Cuttledoc derives the authoritative lexical diff locally from the input and
corrected text; the model returns no edit ledger, reasons, or confidence
scores. The implemented contract keeps immutable section ids and
protected-content gates and explicitly forbids unsupported changes to numbers,
dates, names, and technical terms. It is used by the
[`factorial transcript-postprocessing benchmark`](postprocessing-factorial-benchmark.md).
Only a frozen survivor should proceed to held-out human-verified professional
audio.

The fixture is
[`issue20-de-synthetic-multipage-whisper-1.json`](../benchmarks/postprocessing/fixtures/issue20-de-synthetic-multipage-whisper-1.json),
and complete outputs are retained under
[`benchmarks/postprocessing/runs`](../benchmarks/postprocessing/runs).
