# Apple Speech Swift shim result (#11)

**Status:** technically feasible for one unbundled file-transcription path on
the macOS 26 Apple Silicon baseline; not a production Speech adapter or an STT
selection.

**Evidence date:** 2026-07-16.

**Evidence host:** macOS 26.5.2 (25F84), arm64; Apple Swift 6.3.3.

**Runnable artifact:** [`spikes/apple-speech-shim`](../../spikes/apple-speech-shim/).

## Question and boundary

Can Rust call the new Apple `SpeechAnalyzer`/`SpeechTranscriber` APIs through a
small Swift-owned C ABI without putting Swift or Speech framework types in the
product API?

The spike exposes only two functions: one that transcribes a UTF-8 file path
and returns an owned UTF-8 string, plus one that releases that string. Swift
owns the `SpeechAnalyzer`, `SpeechTranscriber`, asset request, asynchronous
result sequence, and error conversion. Rust owns the FFI call, copies the
string, and invokes the matching release function.

## Reproduction and observed result

```sh
bash scripts/run-apple-speech-spike.sh
```

The script compiles an unbundled Swift dynamic library and a Rust executable,
uses `say` to make a temporary AIFF fixture, and runs the Rust binary with the
library on `DYLD_LIBRARY_PATH`. On the evidence host it completed successfully
through `AssetInventory` and `SpeechAnalyzer`, returning:

```text
Cuddle Dock is testing offline speech transcription on Apple Silicon.
```

The generated prompt said “Cuttledoc”, so this also records a harmless
recognition variation; it is not a quality claim. The generated Swift library
was 91 KiB and the Rust executable 486 KiB before system frameworks, neither
of which measures a shipped application artifact.

This is evidence that an unbundled Rust executable can enter the system Speech
path on this host. It is **not** evidence of a clean-machine install: the run
does not instrument whether the required `en-US` asset was already available
or downloaded by this invocation. It also does not generalize the outcome to
all executable signing identities, locales, models, or macOS installations.

## Lifecycle findings

- `SpeechAnalyzer` is kept inside the Swift asynchronous task; no Swift actor
  or Speech object crosses the C boundary.
- The shim resolves a supported `en-US` locale and asks `AssetInventory` for
  an installation request before creating analysis. If the framework reports
  one, it calls `downloadAndInstall()`.
- The current preset is finals-oriented `.transcription`; it accumulates text
  until analysis finishes. It intentionally does **not** expose volatile
  replacement/revocation, time ranges, confidence, bounded PCM streaming,
  cancellation, or locale enumeration.
- The C call waits for the Swift task in this small experiment. That is useful
  for validating ownership and error transfer, but not a product-ready
  backpressure or cancellation design.

## Decision and remaining work

The narrow Swift C ABI is a viable repository-owned boundary for further
testing. It does not yet establish that system-managed Speech assets are a
reliable distribution dependency, nor does it select Apple Speech over CoreML
or the compatibility backend.

Before advancing #11, the shim must:

1. Return a dynamic locale/asset availability inventory and record fresh-host,
   missing-asset, and installation outcomes for the intended CLI identity.
2. Feed bounded PCM and show streamed volatile -> final updates, including
   explicit replacement/revocation behavior.
3. Request and map `audioTimeRange` and confidence into Rust-owned domain
   result records, with a documented fallback for unavailable metadata.
4. Provide a non-blocking Rust-facing lifecycle with cancellation,
   backpressure, serialized analyzer use, and deterministic cleanup.
5. Test packaging, signing/notarization, error mapping, and memory/lifecycle
   under repeated real fixtures before comparing quality and runtime cost in
   the common STT bakeoff.

The required result semantics and stop conditions are captured in
[the Apple runtime evaluation](../apple-runtime-evaluation.md).
