# Streaming contract reducer spike (#8)

This dependency-free Rust crate turns ADR-0008 into executable semantics before
the production workspace is scaffolded. It defines backend-neutral
`TranscriptionUpdate` values and a deterministic reducer for:

- monotonically ordered updates;
- half-open affected time ranges;
- volatile replacement and explicit revocation;
- volatile-to-final transitions;
- immutable finalized ranges; and
- atomic error handling that does not mutate state or advance sequence.

Run it with:

```sh
cargo test --manifest-path spikes/stream-contract/Cargo.toml
```

The tests consume
[`fixtures/contracts/transcription-updates.tsv`](../../fixtures/contracts/transcription-updates.tsv).
That deliberately simple format is also consumed by the Node artifact spike so
Rust and JavaScript cannot drift onto different hand-written examples.

The spike stores time in integer milliseconds to keep cross-language vectors
exact. The stable Rust API can wrap `Duration`; Node converts the same values to
seconds only at its public presentation boundary.
