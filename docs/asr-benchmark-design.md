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
The held-out cell order, source isolation, rights gate, gold contract, and
minimum scale are frozen in
[`target-domain-plan.json`](../benchmarks/fixtures/target-domain-plan.json)
and tracked by [issue #18](https://github.com/sebastian-software/cuttledoc-rs/issues/18).

| Source | Languages | Domain | Role |
| --- | --- | --- | --- |
| Existing FLEURS fixtures | all five | short read | keep as a fast integration gate; expand within the existing source only for variance estimation, not domain coverage |
| [Multilingual LibriSpeech](https://www.openslr.org/94/) plus [LibriSpeech](https://www.openslr.org/12/) clean validation for English | all five language families | LibriVox audiobooks | first target-domain addition; CC-BY-4.0 and native-language development splits; Spanish and Portuguese regional varieties still require an audit |
| [Merkel Podcast Corpus](https://github.com/deeplsd/Merkel-Podcast-Corpus) | German | professional weekly podcasts | leading German podcast candidate; blocked because the repository exposes no explicit dataset, transcript, or audio license |
| [HUI Audio Corpus German](https://opendata.iisys.de/dataset/hui-audio-corpus-german/) | German | LibriVox audiobooks | leading independent German audiobook candidate; the generator is Apache-2.0, but generated audio and per-work text rights require separate review |
| [GigaSpeech](https://github.com/SpeechColab/GigaSpeech) | English | podcasts and audiobooks | strong English target-domain candidate with professionally annotated evaluation data; keep audio local until per-source rights are accepted |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | transcribed data for English, German, Spanish, and French | professional parliamentary speech | CC0 corpus-data robustness bridge subject to the [European Parliament legal notice](https://www.europarl.europa.eu/legal-notice/en/), not a podcast substitute; no transcribed Portuguese set |
| Curated Cuttledoc gold set | all five, German first | professional podcasts | required target-domain set using source-permitted or user-provided recordings and human-verified transcripts |

MLS does not by itself prove Latin-American Spanish or Brazilian Portuguese
coverage. Those product locales remain separate cells and require the curated
set or another region-pinned source.

A candidate does not pass the rights gate because its repository or generator
has a permissive license. Audio, transcript, derived clips, and redistribution
must each be supported by source-specific evidence. Acquisition is blocked
until an accepted review records that evidence; local-only material remains
the default even after acquisition is allowed.

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

### Development-pilot ASR results

All five candidates ran sequentially on identical digest-checked PCM with one
discarded warm-up and two measured repetitions. The immutable matrix is
[`phase0.audiobook-pilot-1.json`](../benchmarks/matrices/phase0.audiobook-pilot-1.json);
the deterministic word alignments are in
[`phase0.audiobook-pilot-1.errors.json`](../benchmarks/analysis/phase0.audiobook-pilot-1.errors.json).

| Candidate | Recorded macro WER | Boundary-review WER | Mean RTF |
| --- | ---: | ---: | ---: |
| Whisper large-v3-turbo/CoreML | 4.49% | 3.77% | 0.0569 |
| Apple SpeechTranscriber | 7.18% | 7.08% | 0.0174 |
| Parakeet TDT 0.6B/CoreML | 8.90% | 8.56% | 0.0211 |
| Qwen3-ASR 0.6B/MLX reference | 11.14% | 10.53% | 0.0305 |
| Qwen3-ASR 0.6B/direct MLX | 10.96% | pending | 0.0510 |

The boundary-review view keeps apostrophes inside words while treating
hyphens, dashes, and slashes as boundaries. It removes obvious scoring
collisions such as `one-floor` versus `one floor`; it does not judge semantic
equivalence or repair the unverified references.

| Language | Whisper | Apple | Parakeet | Qwen/MLX |
| --- | ---: | ---: | ---: | ---: |
| German | 2.52% | 3.70% | 11.50% | 9.88% |
| English | 3.81% | 4.10% | 4.28% | 4.28% |
| Spanish | 1.21% | 12.01% | 7.40% | 12.34% |
| French | 3.33% | 3.70% | 9.70% | 3.42% |
| Portuguese | 7.99% | 11.91% | 9.95% | 22.74% |

These numbers reject a single language-agnostic quality claim, but they do not
yet justify language routing. Whisper leads every language after the limited
boundary normalization, while Apple is close on German, English, and French
and remains materially faster with real incremental word-timestamp updates.
Portuguese is the weakest cell for every candidate except Parakeet relative to
its own other languages. Qwen's strong French result and weak Portuguese row
show why its direct MLX adapter should be evaluated, not selected or rejected,
on a global mean.

The direct Qwen adapter is now that evaluation: it completed all 45 runs, was
deterministic per fixture, and matched the Python reference text on 12/15
clips. Its raw WER is 11.84% German, 7.37% English, 12.34% Spanish, 3.47%
French, and 19.76% Portuguese. The three stable differences improve the
aggregate slightly but confirm that the measured BF16 execution drift can
cross a greedy decision boundary. The boundary-review analysis has not yet
been regenerated for this fifth candidate, so the matrix does not invent that
number.

Manual gold review comes next. Examples already show contractions, historical
spelling, diacritics, number forms, and proper names mixed with real content
errors. Until those are separated, the table is a development signal rather
than an accuracy claim.

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
   language-specific routing. The compact development pass is complete; the
   target-size held-out pass remains open.
7. Evaluate Gemma and other correction models only on the resulting real raw
   outputs.
