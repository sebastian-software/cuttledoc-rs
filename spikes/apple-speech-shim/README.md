# Apple Speech Swift shim spike (#11)

This is a repository-owned C boundary, not a general Swift binding layer. The
complete ABI is in [`include/cuttledoc_speech_shim.h`](include/cuttledoc_speech_shim.h):

- dynamic locale/installed/reserved-asset inventory;
- create, push caller-owned mono f32 PCM, finish, cancel, and destroy;
- JSON callbacks containing ordered replace/revoke updates, volatile/final
  stability, native time ranges, word segments, and confidence;
- a compatibility file-transcription entry point retained from the bootstrap.

Swift owns `SpeechAnalyzer`, `SpeechTranscriber`, the bounded 64-chunk
`AsyncStream`, asset requests/reservations, and task cleanup. Every successful
PCM push copies and converts the samples to the signed 16-bit buffers required
by the framework, so Rust may reuse its input immediately. Calls for one
session must be serialized. Status `4` means the bounded queue rejected the
chunk and the caller may retry it.

The callback's UTF-8 pointer is borrowed only for the callback duration. The
checked-in Rust example copies it immediately into Rust-owned storage. Callback
execution may occur on a Swift concurrency worker; it is not main-thread
affine. `finish`, `cancel`, and asset-backed `create` wait at the C boundary and
should therefore run on a Rust blocking worker in an async product runtime.

Run the real FLEURS PCM stream, identity inspection, a missing-asset
installation/reservation probe, and cancellation:

```sh
bash scripts/run-apple-speech-spike.sh
```

The default fixture is pinned in the benchmark manifest and read from the
sibling `cuttledoc` checkout. Override it with
`CUTTLEDOC_SPEECH_FIXTURE=/absolute/path/to/audio`.

`destroy` is mandatory after `finish` or `cancel`. It cancels unfinished work,
releases only reservations acquired by that session, and releases the retained
Swift session object. No Swift actor, framework object, audio buffer, or
allocated result string crosses into long-lived Rust ownership.
