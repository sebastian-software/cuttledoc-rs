# Long-form transcript postprocessing screen

**Status:** German multi-page development screen complete for three hosted
models; the pinned Mistral endpoint is temporarily blocked.

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
disabled, required parameters, denied data collection, and a zero-data-retention
route. Cost covers those two recorded requests.

| Candidate | Contract | Output errors | Micro-WER | Error reduction | Sections improved / unchanged / regressed | Recorded cost |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Raw Whisper | n/a | 60 | 6.03% | n/a | n/a | n/a |
| Qwen 3.5 122B-A10B / AtlasCloud FP8 | Passed | 25 | 2.51% | 58.3% | 6 / 1 / 1 | $0.0254754 |
| GPT-5.6 Sol / Azure EU | Failed: one unreported lexical change | 5 | 0.50% | 91.7% | 6 / 2 / 0 | $0.2455266 |
| Claude Sonnet 4.6 / Bedrock EU West 1 | Passed | 31 | 3.12% | 48.3% | 5 / 3 / 0 | $0.1521399 |
| Mistral Small 3.2 24B / Parasail BF16 | Blocked: upstream HTTP 429 twice | — | — | — | — | not recorded |

All three complete remote repeats differed. The recorded cost of the three
complete two-request results is $0.4231419. Requests from the blocked Mistral
attempts and an abandoned Qwen invocation have no complete response record and
are intentionally excluded rather than estimated.

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

Qwen fixed many difficult names and technical terms and reported every lexical
change. It also rewrote an already exact section from `Industrie 5.0` and
`2025` to `Industrie 4.0` and `2023`. Those two unsupported factual changes are
precisely the high-severity regression that an aggregate WER improvement would
hide. Qwen therefore passes the mechanical ledger contract but is not a safe
quality candidate under this prompt.

Claude was the only result to combine a complete edit ledger with zero section
regressions. It was substantially more conservative: it left several
recoverable names, abbreviations, spoken punctuation labels, and the closing
`Vielen Dank` hallucination untouched. It is the safest complete result here,
not the most capable one.

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

The next prompt iteration should separate correction quality from audit
metadata. Cuttledoc can derive the authoritative lexical diff locally from the
input and corrected text; model-supplied reasons and confidence should remain
advisory rather than make an otherwise strong correction unusable. The next
development comparison should therefore keep immutable section ids and
protected-content gates, remove the exhaustive model-authored edit-ledger
requirement, and explicitly forbid unsupported changes to numbers, dates,
names, and technical terms. Only a frozen survivor should proceed to held-out
human-verified German professional audio.

The fixture is
[`issue20-de-synthetic-multipage-whisper-1.json`](../benchmarks/postprocessing/fixtures/issue20-de-synthetic-multipage-whisper-1.json),
and complete outputs are retained under
[`benchmarks/postprocessing/runs`](../benchmarks/postprocessing/runs).
