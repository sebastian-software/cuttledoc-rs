# Target-domain gold review

**Status:** German podcast selection ready for independent human review;
German audiobook source acquisition still blocked.

**Evidence date:** 2026-07-21.

Issue [#18](https://github.com/sebastian-software/cuttledoc-rs/issues/18)
requires references that a human independently verifies against the exact audio.
Publisher transcripts and ASR output are draft material only. Automation may
prepare a review bundle, but it cannot promote its own transcript to gold.

## Frozen German podcast selection

| Split | Episode | Exact source range | Speakers in scope |
| --- | --- | ---: | --- |
| validation | Die Wissensarchitekt*innen S01E01 | 00:02:50–00:12:50 | Ulrike Wuttke, Yuliya Fadeeva, Renke Siems |
| test | Die Wissensarchitekt*innen S01E02 | 00:02:04–00:12:04 | Ulrike Wuttke, Nicola Mößner |
| test | Die Wissensarchitekt*innen S01E04 | 00:09:43–00:19:43 | Ulrike Wuttke, Björn Brembs |

The three passages total exactly 1,800,000 ms and use five distinct speakers.
Each complete episode is a source group, so no episode or derivative crosses a
split. The source audio, publisher-draft PDF, selected PCM, rights review, CC BY
attribution, and output name are pinned in
[`target-domain-corpus.json`](../benchmarks/fixtures/target-domain-corpus.json).

## Review procedure

1. Materialize the exact normalized audio with the repository command documented
   in [`benchmarks/README.md`](../benchmarks/README.md).
2. Prepare one JSON transcript per passage using
   [`target-domain-transcript.schema.json`](../benchmarks/schema/target-domain-transcript.schema.json).
   The publisher PDF and raw ASR outputs may seed this draft.
3. A reviewer other than the draft preparer listens to the complete 10-minute PCM
   and corrects every speaker turn against that exact digest.
4. Preserve verbatim wording, disfluencies, uncertainty, punctuation,
   capitalization, numbers, and speaker changes. Label names/terms,
   numbers/dates/units, negation, and uncertain spans.
5. Record the reviewer identity, UTC review time, exact-audio comparison, and all
   preserved fields in the transcript. Produce a separate review record or signed
   review export and record its SHA-256.
6. Set the corpus entry to `human-verified`, add transcript byte/digest metadata,
   and advance the cell only after all three entries validate.

The validator fails closed: pending entries must not contain reviewer or transcript
digests, while `human-verified` entries require complete reviewer evidence. Raw ASR
results must be frozen separately and may never overwrite these references.

## Remaining external gate

The podcast cell cannot claim its 30 minutes as human-verified until a person
completes the procedure above. The audiobook cell also needs three professionally
recorded German works with three speakers and explicit rights covering audio,
transcription, derived clips, and commercial benchmark use. LibriVox/HUI material
remains diagnostic because it is crowd-read; Vorleser.net remains permission-only.
