# Prompt-controlled transcript correction

**Status:** prompt candidates and evaluation contract defined; model runs
pending representative audiobook and podcast gold data.

**Evidence date:** 2026-07-20.

Prompt behavior is part of the correction candidate. A result is identified by
the complete tuple:

`model + model revision + quantization + prompt id/hash + decoding settings +
language + domain + ASR backend + supplied context`.

Changing a prompt creates a new candidate. Results from different tuples may
not be merged under a model name such as “Gemma.”

## Candidate ladder

The exact prompt text and digests are pinned by
[`manifest.json`](../benchmarks/postprocessing/prompts/manifest.json).

| Candidate | Lexical freedom | Purpose |
| --- | --- | --- |
| Historical Cuttledoc 2 | broad proofreading | reproduce the prior baseline and its regression behavior |
| Surface only | none | improve punctuation, capitalization, and structure with a mechanically enforceable lexical invariant |
| Conservative error profile | bounded contextual edits | test whether backend/language/domain error classes learned on development sources improve precision |
| Targeted spans | supplied alternatives at supplied spans only | test the safest lexical repair path when ASR confidence, alternatives, glossary checks, or deterministic validators identify a suspect span |

The surface-only mode is rejected whenever its case- and
punctuation-insensitive token sequence differs from the input. The structured
modes are rejected when JSON is invalid, an edit is missing from the audit
list, edits overlap, protected content changes, or the output contains a
lexical change outside its allowed policy.

## Context is evidence, not permission

A lexical correction model does not hear the audio. It may receive:

- language and regional variety;
- recording domain;
- an error-class profile derived from independent development recordings;
- a user or episode glossary;
- protected names, numbers, dates, units, terms, and negations; and
- backend evidence such as low-confidence spans or decoder alternatives.

The prompt never receives the gold transcript, gold alignment, or an error
example from the held-out source. An aggregate tendency such as “this backend
often joins hyphenated names” narrows permitted reasoning but does not prove
that a particular word is wrong.

Transcript content is untrusted quoted data, not an instruction channel. The
new prompt variants delimit it and explicitly reject instructions spoken in the
recording. The historical prompt remains unchanged so it can serve as an exact
control.

## Split discipline

Splits are grouped by complete source work or episode:

1. **development:** inspect errors, build error profiles, author prompts, and
   choose thresholds;
2. **validation:** select the prompt/model/decoding tuple without editing it;
3. **test:** run the frozen tuple once on unseen works and report the result.

Clips from one episode or book cannot cross these groups. The ten current
FLEURS fixtures have already influenced the design and are therefore
development data only. The 15 audiobook-pilot clips and their ASR outputs have
also now been inspected and are development-only.

## Error-conditioned prompts

The audiobook pilot provides a first concrete error profile, not acceptance
gold. Across its 15 clips, the boundary-preserving review alignment finds 23
Whisper edits, 43 Apple edits, 54 Parakeet edits, and 60 Qwen/MLX-reference
edits. The differences are highly language-specific: Whisper has two Spanish
review edits but nine Portuguese edits; Apple has five German edits but 15
Spanish edits; Qwen has four French edits but 24 Portuguese edits.

Prompt tuning therefore starts from explicit, backend/language/domain error
classes rather than “make this transcript better.” For each development
alignment:

1. human review separates reference conventions, benign orthography, and real
   content errors;
2. recurring correctable classes become bounded prompt instructions;
3. names, numbers, dates, units, terminology, and negations become protected
   spans unless product evidence marks the exact span suspect;
4. every proposed lexical change declares its matched error class and evidence;
5. unmatched edits are rejected, even when they sound stylistically better;
6. prompt gains are reported as edit precision and regression rate per
   language/backend, not only as average WER reduction.

Examples already visible in the pilot include hyphen boundaries, contractions,
historical accents and inflections, number words versus digits, and proper-name
spellings alongside genuine substitutions and omissions. A free proofreading
prompt cannot safely distinguish those without audio or external context. The
surface-only prompt handles punctuation and capitalization; the conservative
profile may address only reviewed recurring classes; targeted-spans mode is
reserved for product-available confidence, alternatives, or glossary evidence.

The dataset transcripts remain unverified, so their alignments may shape the
prompt taxonomy but cannot label an edit beneficial or harmful. That requires
human-verified verbatim and evaluation references first.

## Measurements

Every run retains raw input, corrected output, parsed edit proposals, an
external token diff, prompt/model/runtime identity, latency, tokens per second,
and parser or policy rejection.

Results are reported per language, domain, ASR backend, prompt, and model:

- Unicode content WER/CER before and after;
- surface punctuation, sentence-boundary, and capitalization scores;
- proposed, accepted, beneficial, neutral, and harmful lexical edit counts;
- edit precision and transcript-level regression rate;
- names/terms, numbers/dates/units, and negation accuracy;
- omissions and hallucinations;
- protected-span and output-contract violations; and
- latency, memory, prompt tokens, generated tokens, and model/runtime size.

An aggregate gain is insufficient. Acceptance requires no critical semantic
regression on the test set, no protected-content violation, and a non-regressing
content metric in every product-priority language/domain cell. The raw
transcript remains available even for an accepted correction stage.

The initial templates use one English instruction language to isolate the
other variables. If a model follows localized instructions more reliably, each
localized template receives a new prompt id and hash and is evaluated as a
separate candidate rather than silently replacing the English prompt.

## First experiment

1. Use Gemma 3n E4B as the historical local-model candidate and deterministic
   decoding where the runtime supports it.
2. Run raw/no-op, historical, surface-only, and conservative prompts on
   development data from real Apple, Whisper, and Qwen outputs.
3. Build suspect spans only from product-available signals; never use gold
   alignment at inference.
4. Freeze at most one surface and one lexical prompt tuple per model for
   validation.
5. Run the surviving tuple on unseen audiobook and professional-podcast
   sources, German first.
