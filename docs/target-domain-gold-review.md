# Target-domain gold review

**Status:** German podcast review inputs are materialized; independent gold
review is the next transcript-enhancement selection gate.

**Evidence date:** 2026-07-23.

The pragmatic benchmark uses reproducible multi-voice synthetic speech as its
primary ASR comparison and this corpus as a real-world long-form control. The
completed hosted challenge now makes the German podcast validation passage a
required gate for selecting a transcript-enhancement model. This does not
reopen the ASR-backend decision or make optional enhancement release-blocking.
Exact WER and semantic-safety claims still require a human to verify references
against the exact audio. Publisher transcripts and ASR output remain draft
material only. Automation may prepare a review bundle, but it cannot promote
its own transcript to gold.

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
2. Generate a validation-only review bundle. The command verifies the normalized
   audio and publisher PDF digests, writes a deterministic float-WAV playback
   wrapper over the exact PCM, extracts the publisher draft as an aid, and records
   the five frozen ASR backends:

   ```sh
   node scripts/prepare-target-domain-review.mjs prepare
   ```

   Test sources remain closed unless the command receives both `--split test`
   and `--allow-test`.
3. Generate the five validation ASR drafts against the identical PCM. The
   resumable runner checkpoints each completed backend and records no quality
   score while human gold is pending. It treats backend presence and complete
   audio coverage as separate conditions:

   ```sh
   node scripts/run-target-domain-asr.mjs run \
     --output artifacts/target-domain/review/validation-asr-drafts.json
   ```

   Apple, Whisper, and Parakeet prove coverage from their final segment
   timestamp. The direct Qwen adapter uses deterministic 30-second,
   non-overlapping chunks because its repository boundary intentionally limits
   one generation to 256 tokens. The record exposes the boundary-split
   limitation and every chunk digest. Direct Voxtral receives an 8,192-token
   safety budget and must reach `audio_end` or EOS; hitting `max_tokens` leaves
   that draft incomplete. A record is `complete` only when all five backends
   satisfy their coverage condition.

4. Prepare one JSON transcript per passage using
   [`target-domain-transcript.schema.json`](../benchmarks/schema/target-domain-transcript.schema.json).
   The publisher PDF and raw ASR outputs may seed this draft.
5. A reviewer other than the draft preparer listens to the complete 10-minute PCM
   and corrects every speaker turn against that exact digest.
6. Preserve verbatim wording, disfluencies, uncertainty, punctuation,
   capitalization, numbers, and speaker changes. Label names/terms,
   numbers/dates/units, negation, and uncertain spans.
7. Record the reviewer identity, UTC review time, exact-audio comparison, and all
   preserved fields in the transcript. Produce a separate review record or signed
   review export and record its SHA-256.
8. Set the corpus entry to `human-verified`, add transcript byte/digest metadata,
   and advance the cell only after all three entries validate.

The validator fails closed: pending entries must not contain reviewer or transcript
digests, while `human-verified` entries require complete reviewer evidence. Raw ASR
results must be frozen separately and may never overwrite these references.

## Scope boundary

The podcast cell cannot claim its 30 minutes as human-verified until a person
completes the procedure above. Validation review is now required before issue
#20 can select or port an enhancement model. German audiobook acquisition,
test-split opening, and additional language/domain gold cells remain paused
until the validation configuration is frozen or a later decision explicitly
needs broader evidence.
