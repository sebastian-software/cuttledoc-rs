# Language- and domain-aware ASR benchmark

**Status:** accepted evaluation direction; a development audiobook pilot is
acquired, while target-size audiobook and podcast corpora remain in progress.

**Evidence date:** 2026-07-20.

The initial ten-fixture FLEURS matrix is an integration and multilingual breadth
gate. It is not large or diverse enough to rank ASR backends for Cuttledoc's
primary workload: professionally recorded podcasts and audiobooks.

## Decision model

Results must be reported on two independent axes:

1. language (`de`, `en`, `es`, `fr`, and `pt`); and
2. source domain (`short-read`, `audiobook`, `podcast`, and a
   `professional-speech` bridge where target-domain material is missing).

There is no canonical all-language, all-domain WER. A product profile may
weight cells, for example German podcasts more strongly than French
audiobooks, but the weights must be explicit and the underlying cell results
must remain visible. Backend auto-routing by language is allowed only after a
cell has enough independent recordings to support it.

Model-only and product-pipeline quality are separate measurements:

- **aligned-clip track:** fixed gold-aligned clips isolate model recognition;
- **long-form track:** complete 5-15 minute passages include decoding,
  chunking/VAD, overlap handling, merge behavior, timestamps, and memory over
  time.

## Source plan

Machine-readable source dispositions and exact repository revisions are in
[`source-candidates.json`](../benchmarks/fixtures/source-candidates.json).

| Source | Languages | Domain | Role |
| --- | --- | --- | --- |
| Existing FLEURS fixtures | all five | short read | keep as a fast integration gate; expand within the existing source only for variance estimation, not domain coverage |
| [Multilingual LibriSpeech](https://www.openslr.org/94/) plus [LibriSpeech](https://www.openslr.org/12/) clean validation for English | all five language families | LibriVox audiobooks | first target-domain addition; CC-BY-4.0 and native-language development splits; Spanish and Portuguese regional varieties still require an audit |
| [GigaSpeech](https://github.com/SpeechColab/GigaSpeech) | English | podcasts and audiobooks | strong English target-domain candidate with professionally annotated evaluation data; keep audio local until per-source rights are accepted |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | transcribed data for English, German, Spanish, and French | professional parliamentary speech | openly licensed robustness bridge, not a podcast substitute; no transcribed Portuguese set |
| Curated Cuttledoc gold set | all five, German first | professional podcasts | required target-domain set using source-permitted or user-provided recordings and human-verified transcripts |

MLS does not by itself prove Latin-American Spanish or Brazilian Portuguese
coverage. Those product locales remain separate cells and require the curated
set or another region-pinned source.

### Acquired audiobook development pilot

Revision `audiobook-pilot-1` contains 15 hash-pinned clips: three independent
speakers and chapters in each of German, English, Spanish, French, and
Portuguese. Its normalized duration is 223.68 seconds:

| Language | Clips | Speakers/chapters | Duration |
| --- | ---: | ---: | ---: |
| German | 3 | 3/3 | 49.03 s |
| English | 3 | 3/3 | 43.88 s |
| Spanish | 3 | 3/3 | 49.90 s |
| French | 3 | 3/3 | 38.88 s |
| Portuguese | 3 | 3/3 | 41.99 s |

The selected Hugging Face MLS conversion exposes German, Spanish, French, and
Portuguese but not English. The English cells therefore use LibriSpeech clean
validation from the same LibriVox audiobook domain. Both dataset revisions,
row identities, source bytes, and normalized 16 kHz mono float PCM bytes are
pinned in
[`audiobook-pilot.json`](../benchmarks/fixtures/audiobook-pilot.json).

This set is development evidence, not the target benchmark. It falls far short
of 30 minutes per language/domain, and all clips have already influenced
benchmark design. Spanish and Portuguese carry no regional locale claim.
Dataset transcripts are useful lexical references but remain
`dataset-transcript-unverified`: they must be checked against the exact audio
before semantic error severity or postprocessing acceptance is reported.

The initial acquisition target per language and target domain is at least 30
minutes from three independent works or episodes and three speakers. Clips from
one episode are correlated: confidence intervals and train/tune/test boundaries
must group by the original work, not pretend each clip is an independent
source. Release thresholds require a later power/variance review rather than an
arbitrary clip count.

The curated podcast set must cover, where available:

- clean single-speaker studio speech;
- two or more speakers with natural turn-taking;
- proper nouns, dates, numbers, units, and domain terminology;
- incidental music or room tone; and
- at least one uninterrupted long-form passage per source.

## Gold transcript contract

Reference text must be human-verified against the exact audio range. Publisher
show notes, subtitles, book text, or another ASR transcript are starting
material only, never unquestioned gold.

Each fixture records source URL and revision, license or user authorization,
work/episode and speaker identity, language and regional variety, exact time
range, original and normalized audio hashes, and both verbatim and evaluation
text. Redistribution is denied by default unless the source license explicitly
permits it; a local-required fixture still pins acquisition and digests.

Gold text retains punctuation, capitalization, numbers, disfluencies, speaker
turns, and uncertainty annotations. A versioned normalization derives the
content-WER view without destroying the richer reference.

## Metrics and error severity

| View | Measures | Purpose |
| --- | --- | --- |
| Content | Unicode-normalized WER/CER, substitutions, insertions, deletions | lexical recognition after case and punctuation are ignored |
| Critical content | names/terms, numbers/dates/units, and negation accuracy | errors most likely to change meaning |
| Surface form | punctuation F1, sentence-boundary F1, capitalization accuracy | readability improvements invisible to current WER |
| Long form | omissions and hallucinations per hour, continuity, timestamp drift, speaker-turn preservation | full-pipeline behavior on podcasts and audiobooks |
| Operations | warm/cold latency, RTF, first result, memory, model/runtime size, energy when available | local product cost |

Every current lexical error is small enough to inspect manually. The review
taxonomy is:

1. surface-only formatting;
2. word-boundary or orthographic form;
3. benign lexical or inflectional difference;
4. critical content substitution, including names, terminology, numbers,
   units, dates, or negation;
5. omission;
6. insertion or hallucination; and
7. attribution, timing, or long-form merge failure.

Counts and example alignments are reported per language and domain. One
critical hallucination cannot be hidden by many punctuation fixes or a better
global average.

## LLM correction is a separate stage

Raw ASR output is immutable benchmark evidence. A correction model consumes a
copy and produces a second result evaluated against the same gold transcript.
It is never folded into the ASR backend score.

A correction candidate is accepted only per language/domain cell and only if:

- content WER does not regress;
- no critical-content or hallucination regression is introduced;
- surface-form measures improve when that is the claimed benefit;
- every changed word is retained as an auditable diff; and
- the raw transcript remains available as fallback.

The favorable Cuttledoc 2 Gemma/Phi/Mistral numbers came from two
TTS-generated samples per language. A separate real-ASR experiment contains
substantial model- and language-specific regressions. Those historical results
are useful hypotheses, not an acceptance result for Cuttledoc 3.

## Execution order

1. Import the historical postprocessing evidence with its exact hashes and
   normalization limitations.
2. Add deterministic word alignment and per-language/domain error reports.
3. Expand FLEURS from two to ten clips per language as the short-read variance
   set.
4. Add a compact MLS/LibriSpeech audiobook subset for all five languages.
5. Add GigaSpeech English podcast material and a legally reviewed,
   German-first curated podcast set.
6. Re-run Apple, Whisper, Qwen, and Parakeet on identical cells before defining
   language-specific routing.
7. Evaluate Gemma and other correction models only on the resulting real raw
   outputs.
