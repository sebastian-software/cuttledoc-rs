# Benchmark source rights review

**Status:** accepted acquisition gate; three German podcast source groups are
accepted and materialized locally. German audiobook acquisition remains blocked.

**Evidence date:** 2026-07-21.

Issue [#18](https://github.com/sebastian-software/cuttledoc-rs/issues/18)
requires professionally recorded, held-out podcast and audiobook audio. Public
availability is not permission to acquire, derive clips from, use, or
redistribute a recording. The benchmark therefore treats rights review as an
executable gate rather than a prose note.

This process records engineering evidence and enforces project policy. It is
not legal advice. Unclear evidence remains blocked until the rightsholder,
source terms, or an appropriate reviewer resolves it.

## Review levels

A candidate-level review records whether a corpus is worth pursuing and which
questions remain. It can never authorize acquisition. Current reviews include:

- `die-wissensarchitektinnen-s01e01`, `s01e02`, and `s01e04`: accepted
  source-group reviews for professionally produced German podcast episodes.
  The official feed licenses each episode under CC BY 4.0; exact originals are
  digest-pinned, and selected ranges remain pending independent gold review;

- [`merkel-podcast-corpus.json`](../benchmarks/rights/merkel-podcast-corpus.json):
  strong German podcast fit, blocked because no explicit audio/transcript grant
  was found;
- [`hui-audio-corpus-german.json`](../benchmarks/rights/hui-audio-corpus-german.json):
  strong German audiobook fit, blocked because the generator license does not
  establish rights to generated LibriVox audio and per-work text; and
- [`vorleser-net.json`](../benchmarks/rights/vorleser-net.json): strong
  professionally presented German audiobook fit, but explicitly permission
  only because the current terms prohibit the required STT/ML analysis,
  research/testing, editing, and commercial use without prior written
  permission; and
- [`gigaspeech.json`](../benchmarks/rights/gigaspeech.json): strong English
  target-domain fit, blocked until each underlying source group is reviewed.

Only a source-group review may be accepted. One review covers exactly one
complete work or episode and its derivatives. Acceptance requires:

1. accepted audio, transcript, derived-clip, and commercial benchmark-use
   decisions, each with a recognized basis and evidence URL;
2. an explicit `local-only` or `allowed` redistribution decision;
3. evidence that the complete source group is absent from FLEURS, MLS, and
   LibriSpeech development inputs;
4. the exact original artifact SHA-256, source URL, and safe local artifact
   name;
5. `manual-local-file` acquisition, so authentication, subscriptions, or user
   authorization remain outside repository automation; and
6. no unresolved blockers.

Permission or license evidence belongs in the review. Credentials, private
audio, and private authorization documents do not belong in Git.

## Import workflow

Create a new source-group review under `benchmarks/rights/`, validate the
repository, and import the exact local file:

```sh
node scripts/validate-benchmark-data.mjs --self-test

node scripts/import-target-domain-source.mjs \
  --review benchmarks/rights/<source-group>.json \
  --source /absolute/path/to/original-audio \
  --output-dir /absolute/path/to/cuttledoc-target-domain
```

The importer fails closed unless every gate above is accepted. It verifies the
source digest, copies the original bytes without normalization, and writes a
deterministic `.provenance.json` sidecar. Re-running the command verifies the
existing files rather than silently replacing them.

Original import is intentionally separate from audio normalization and gold
transcript work. The first selection is pinned in
[`target-domain-corpus.json`](../benchmarks/fixtures/target-domain-corpus.json):
three ten-minute German podcast passages, five speakers, frozen validation/test
assignments, exact original/publisher-transcript/normalized-PCM digests, and an
explicit pending independent-review state. Reproduce the local PCM with:

```sh
node scripts/materialize-target-domain-corpus.mjs \
  --input-dir /absolute/path/to/cuttledoc-target-domain \
  --output-dir /absolute/path/to/cuttledoc-target-domain/normalized
```

The materializer rechecks each accepted rights review and original digest before
invoking FFmpeg, then rejects any normalized-byte or digest drift. The publisher
transcripts are alignment aids, never unquestioned gold. See
[`target-domain-gold-review.md`](target-domain-gold-review.md) for the human gate.
