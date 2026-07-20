# Apple runtime evaluation

**Status:** Phase 0 decision matrix, not a production runtime selection.

**Evidence snapshot:** 2026-07-17.

**Related:** [#2](https://github.com/sebastian-software/cuttledoc-rs/issues/2),
[ADR-0003](adr/0003-staged-native-interop.md),
[ADR-0005](adr/0005-third-party-dependency-policy.md),
[ADR-0006](adr/0006-apple-runtime-and-model-selection-by-bakeoff.md),
[ADR-0007](adr/0007-apple-silicon-macos-26-baseline.md), and
[ADR-0010](adr/0010-capability-oriented-engine-ownership.md).

## Decision summary

CoreML, MLX, Metal-native runtimes, Apple system Speech, and remote APIs are
complementary *internal* task adapters. Cuttledoc does not select one universal
Apple runtime, and its public API will never expose tensors, compute units,
CoreML objects, MLX arrays, Metal objects, or foreign runtime handles.

The Phase 0 boundaries now have real evidence:

1. The CoreML lifecycle works through bounded `objc2-core-ml`, but its native
   objects require an owned worker and it has not yet proven a complete ASR
   backend.
2. Apple Speech works through the owned Swift C ABI with PCM streaming,
   volatile-to-final replacements, timestamps/confidence, asset lifecycle, and
   cancellation.
3. Official MLX works as the third first-class foundation through the owned C++
   task ABI. The full Whisper Tiny encoder matches the official reference on
   CPU and Metal across two MLX releases, and the #15 extension now produces an
   end-to-end transcript plus model timestamps through Rust.

These results settle the capability-oriented engine boundary in ADR-0010 but
do not select the initial transcription backend. Text generation remains a
separate task decision; speech synthesis is an explicit future direction, not
an inference-runtime shortcut or a Phase 0 contract.

## Selection criteria

Every candidate/model pair is scored against the same evidence record:

| Dimension | Required evidence |
| --- | --- |
| Product fit | STT or text-generation behavior, language coverage, timestamps, confidence, streaming revision behavior, and fixture quality. |
| Lifecycle | Cold/warm load, deterministic close, repeated create/run/destroy, cancellation boundary, backpressure, actor/thread requirements, and error mapping. |
| Apple execution | Requested compute policy, available/actual plan when the OS exposes it, chip/OS/model restrictions, first-result latency, real-time factor, peak memory, and measured energy procedure. |
| Model delivery | Conversion path, immutable source/revision, download size, validation, cache behavior, update/rollback, and license/provenance. |
| Distribution | `aarch64-apple-darwin` deployment target, linked frameworks, native build tools, binary size, signed/notarized artifact implications, and npm prebuild compatibility. |
| Maintenance | The dependency-policy disposition, release/CI/security signals, direct/transitive cost, named owner, and replacement boundary. |

The macOS 26 Apple Silicon product baseline makes every local candidate
available for a like-for-like test, but it does **not** promise that every
model runs on every Apple chip or on the Neural Engine. Core ML lets callers
request compute units and inspect available devices/compute plans, while its
actual allocation remains model- and OS-dependent. MLX documents CPU and GPU
devices with unified memory; it is not evidence of Neural Engine execution.
The benchmark record must therefore report requested policy and observed
diagnostics separately, never infer hardware use from the machine name.

Primary references: [Core ML overview](https://developer.apple.com/documentation/coreml),
[MLModelConfiguration](https://developer.apple.com/documentation/coreml/mlmodelconfiguration),
[MLModel compute devices](https://developer.apple.com/documentation/coreml/mlmodel/configuration),
and [MLX architecture](https://ml-explore.github.io/mlx/).

## Task-by-runtime matrix

| Task | Candidate | Present disposition | What it can prove | Phase 0 disqualifiers / unknowns | Next action |
| --- | --- | --- | --- | --- | --- |
| STT | CoreML through an internal Rust adapter | `objc2-core-ml`: accepted, bounded; adapter: repository-owned | A real compiled model completed 100 create/predict/drop cycles; named inputs/outputs and bounded autorelease ownership are proven. | A complete ASR graph, scoped buffer API, typed errors, async cancellation, observed compute plan, quality, and distribution remain. Native objects are neither `Send` nor `Sync`. | Continue only as an owned-worker CoreML ASR candidate under the common benchmark. |
| STT | whisper.cpp plus CoreML/Metal | Repository-owned boundary candidate | Mature compatibility path for the existing Whisper baseline; upstream owns specialized decode/runtime work. | Exact build/options, model pairing, artifact cost, concurrency, and maintenance surface remain unmeasured here. | Keep for the bakeoff/compatibility backend; do not add a generic Rust wrapper. |
| STT | Apple SpeechAnalyzer / SpeechTranscriber | Advance as first-class bakeoff candidate through repository-owned Swift C ABI | Real PCM streaming, 27 volatile replacements plus one final, word timestamps/confidence, dynamic locales, asset install/reservation/release, and cancellation are proven. | Stable shipped bundle identity, opaque system model revision/size, clean cold start, energy, repeated variance, and broader quality remain. | Include in the common bakeoff with dynamic capabilities and system-managed provisioning. |
| STT or text generation | Official MLX via owned C++ adapter | Advance as third first-class foundation; repository-owned boundary | Whisper Tiny proves CPU/Metal portability across two MLX releases; Qwen3-ASR proves a quantized 28-layer decoder, owned KV cache, multilingual audiobook quality, repeated Rust lifecycle, and cooperative cancellation. | Held-out podcasts, long audio, incremental input, aligned timestamps, clean cold start, energy, and pruned release packaging remain unproven. | Integrate the bounded task ABI behind the common engine, then continue the target-domain benchmark. |
| STT or text generation | official `mlx-c` | Optional reference/control only | A secondary C-level check for a specific allocation, stream, or interface uncertainty. | No GitHub releases; every use must bind an audited commit to its MLX revision. C API use does not make a model integration smaller or safer by itself. | Use only when it answers a stated #6 question; never make it a product dependency by default. |
| STT or text generation | `mlx-rs`, OminiX-MLX, `mlx-node` | Reference only | Prior art, model-format clues, test vectors, and benchmark hypotheses. | Their wrappers/runtime ownership cannot silently become Cargo, build, or distribution dependencies. | Do not import. |
| Text generation | Remote HTTP provider | Repository-owned product adapter | Straightforward text-generation/enhancement path with no embedded native model runtime; preserves separate task ownership. | Credentials, network/error semantics, cost, latency, privacy, and API/model lifecycle must be evaluated independently from STT. | #7, after the common contracts are shaped. |
| Text generation | mistral.rs or Candle / Metal-native candidate | Reference only | Evidence about Metal-native local generation and model support. | Broad runtime feature/transitive/build surface; no initial embedded LLM decision follows from ASR results. | Phase 5 evaluation only. |
| TTS | Apple `AVSpeechSynthesizer` through an owned Swift C ABI | System baseline; foundation proven | A real German Wikipedia passage produced Rust-owned mono f32 PCM; voice inventory, native format, timing, stable busy rejection, cross-thread cancellation, and explicit buffer/string ownership are proven. | System model revision and delivery are opaque; a restricted process context cancelled before first audio; clean packaged identity, repeated variance, normalized shared PCM, listening quality, and Node mapping remain. | Retain as the system/lifecycle baseline and run all nine passages through the shared roundtrip matrix; do not accept the public contract yet. |
| TTS | Qwen3-TTS and Chatterbox through pinned MLX-Audio | Serious Phase 5 candidates; reference runner first | A pinned Apple Silicon implementation can establish model quality, audio, timing, and delivery evidence without predetermining the Rust API. | Converted model revisions/digests, Python/Swift ownership, dependency breadth, packaging, cancellation, and direct-official-MLX cost are not measured yet. | Pin Qwen3-TTS 0.6B CustomVoice, run the same passages, then compare retaining MLX-Audio with a narrow owned adapter over official MLX. |
| TTS | Qwen-Audio-3.0-TTS-Plus provider API | English remote quality ceiling | External provider-voice evidence prevents local candidates from defining quality only relative to a weak system baseline. | Hosted model is not the local open Qwen3-TTS model; current built-in voice evidence is English/Mandarin, provider snapshot is opaque, and no German inference follows from its leaderboard score. | Add only to the separate English control cell when credentials and provider snapshot metadata are available. |

The MLX C API documents opaque arrays, devices, and streams with explicit
construction/free operations. It can therefore serve as a controlled reference
when that clarifies a concrete #6 question, but the intended product boundary
is a small owned adapter directly over official MLX. In both cases arrays stay
on the far side of that adapter rather than becoming product API. See the
[MLX C overview](https://ml-explore.github.io/mlx-c/build/html/overview.html).

“Directly over official MLX” does not mean that Rust consumes the C++ API
without glue. Rust's stable native FFI is C-shaped; MLX's public implementation
API is C++ and uses language features that cannot be declared as an
`extern "C"` Rust module. The repository-owned C ABI is the deliberate seam:
Rust owns the safe task lifecycle and C++ owns MLX-native graph state. Binding
generators could automate that seam, but they would not remove it and would
create a broader upstream-coupled surface for no current product benefit.

## Capability vocabulary

Capability discovery describes product behavior and an actionable reason for
unavailability. It does not reveal the internal runtime that produced a result.

| Capability | Meaning | Examples |
| --- | --- | --- |
| `availability` | The backend can be constructed on this platform, or a stable reason why it cannot. | unsupported platform, model missing, asset unavailable, native load failure, credentials missing |
| `input` | Accepted caller-owned media forms. | file, bounded PCM, streaming PCM; no microphone/system capture in the library |
| `recognitionResults` | Output behavior for a speech engine. | language, segments, word/segment timestamps, confidence, finals-only, volatile range replacement |
| `generationResults` | Output behavior for a text engine. | incremental text deltas, completion-only, token metadata only where stable |
| `modelProvisioning` | How required assets become ready. | Cuttledoc manifest download, system asset installation, remote-only |
| `execution` | Product-level scheduling behavior. | one request at a time, bounded concurrent requests, local/remote, preheat supported |
| `cancellation` | The latest point at which cancellation takes effect. | before load, between chunks, cooperative in-flight, unsupported while native call runs |
| `diagnostics` | Optional non-contractual measurements. | requested compute policy, observed plan when available, model revision, load and first-result timing |

The internal adapter converts native results into the domain contracts in
[`docs/public-api.md`](public-api.md): STT emits ordered range-addressed
replace/revoke updates with volatile/final stability, while text generation
emits its own delta stream. A finals-only backend is not treated as a degraded
tensor runtime; it truthfully advertises `emitsVolatileResults = false`.

## Bounded experiment cards

### #5 — CoreML from a Rust-owned lifecycle

**Hypothesis:** the accepted `objc2` framework crates can support a small
internal CoreML adapter with correct ownership and no public Objective-C types.

**Minimum evidence:** load one checked-in real `.mlmodelc` fixture, set an
explicit `MLModelConfiguration`, create named input features, invoke prediction,
copy named outputs into Rust values, and run create → predict → close repeatedly.
Capture requested compute units plus any available device/plan diagnostics.

**Observed disposition (2026-07-16):** technically feasible through the
accepted bounded bindings. A real compiled Silero VAD model completed 100
create/predict/drop cycles with a stable scalar result and bounded
autorelease-pool ownership. The bindings mark native model/array/provider
objects as neither `Send` nor `Sync`, which directly supports ADR-0010's owned
worker proxy. A complete ASR graph, supported scoped buffer access,
compute-plan evidence, async cancellation, and the common quality benchmark
remain. Exact evidence is in
[`docs/spikes/coreml-rust.md`](spikes/coreml-rust.md).

**Concurrency rule to test:** one stateful `MLState` must be serialized; Apple
documents undefined behavior if the same state is used in concurrent
predictions. The spike must separately document what is safe for its selected
model instance and must use bounded autorelease pools. See
[MLState](https://developer.apple.com/documentation/coreml/mlstate) and
[async stateful prediction](https://developer.apple.com/documentation/coreml/mlmodel/prediction%28from%3Ausing%3Aoptions%3A%29-8b4qa).

**Stop conditions:** absent binding API, ownership behavior that requires a
large unsafe facade, unrecoverable model load/prediction error mapping,
material growth over repeated runs, or a deployment/packaging cost that breaks
the supported artifact path. A technical success changes neither the CoreML
adapter disposition nor the first-backend choice without the bakeoff evidence.

### #11 — Apple Speech through an owned Swift shim

**Hypothesis:** a minimal C ABI over a Swift actor can give Rust file/PCM STT,
asset management, streamed range updates, timestamps, and confidence without
making Swift types part of Cuttledoc's public API.

**Minimum evidence:** dynamically enumerate a usable locale; install/status
the required asset; transcribe a real fixture; record `audioTimeRange` and
confidence; show volatile-to-final behavior; cancel and clean up; and reproduce
the executable identity/asset behavior from an unbundled CLI. The shim owns the
Swift task/actor lifetime and maps callbacks to Rust-owned buffers.

**Observed disposition (2026-07-17):** advance through the repository-owned
Swift C ABI. The real FLEURS run produced 27 volatile replacements, one final
result with 19 word ranges/confidences, and no replacement-free revocation.
An unbundled ad-hoc CLI installed/reserved a missing `es-ES` asset and released
it on cancel/destroy, but Speech logged an unstable client identity because the
binary had no bundle identifier. Public f32 PCM must be copied and converted to
the framework-required Int16 buffer format. Exact evidence and remaining
system-asset opacity are in
[`docs/spikes/apple-speech-swift-shim.md`](spikes/apple-speech-swift-shim.md).

Apple documents `SpeechAnalyzer` as an actor whose modules return an
`AsyncSequence`, process one input sequence at a time, and require assets to be
available before analysis. The result-stream/revision mapping is therefore
based on the observed 27-to-1 update sequence rather than an assumed callback
model. See
[SpeechAnalyzer](https://developer.apple.com/documentation/Speech/SpeechAnalyzer)
and [Speech framework](https://developer.apple.com/documentation/speech/).

**Stop conditions:** required assets cannot be reliably installed/reserved for
the shipped CLI/library identity, the ABI forces a broad Swift framework,
volatile/replacement semantics cannot be mapped deterministically, or release
tooling makes the artifact path disproportionate to the measured benefit.

### #6 — Direct official MLX spike

**Hypothesis:** a small owned C++ adapter directly over a pinned official MLX
release can run one meaningful model path from Rust with acceptable upgrade and
distribution cost.

Run one meaningful task and fixture through the narrow owned adapter over a
pinned official MLX release. Record the source revision, interface width, model
conversion, load/warm/inference measurements, memory, output quality, binary
cost, and the work to advance across two official MLX releases. Use pinned
`mlx-c` only as an optional secondary control when it answers a named interface
or lifecycle question; do not make it a competing product path. Do not
benchmark unrelated demos or use a community wrapper as a primary path.

**Observed disposition (2026-07-17):** advance MLX as the third first-class
foundation. The repository-owned ABI first ran the complete Whisper Tiny
frontend and four-layer audio encoder on real FLEURS PCM on CPU and GPU. Three
repeated load/run/destroy cycles were stable, and the unchanged encoder source
produced the same per-device fingerprints on MLX 0.31.2 and 0.32.0. #15 then
added all decoder tensors, the official tokenizer vocabulary, greedy
timestamp-token decoding, and Rust-owned transcript/segment results. The
end-to-end text matches the pinned official Python reference and scores 5.26%
WER on the first fixture; warm GPU task time was about 115.8 ms. The v0.32.0
runtime package was an 18.7 MB shim plus 130.2 MB metallib. NPY weights must
load on CPU before copying to GPU, and current default-device use is
process-serialized.

The 2026-07-20 Qwen3-ASR follow-up then exercised all 18 audio and 28 quantized
text layers, a repository-owned KV cache, exact greedy transcript decoding, and
15 multilingual audiobook fixtures. Its session ABI uses explicit
thread-crossing official MLX streams and returned stable invalid-argument,
busy, and cancelled statuses while preserving six of six exact repeated
transcripts. Cancellation is cooperative after a synchronous graph or decoder
step, not mid-kernel preemption. Held-out podcasts, long audio, incremental
input, aligned timestamps, clean cold start, energy, and release pruning
remain. Exact evidence is in
[`docs/spikes/mlx-direct.md`](spikes/mlx-direct.md)
and
[`docs/spikes/qwen3-mlx-direct.md`](spikes/qwen3-mlx-direct.md).

## Next selection rule

The first local STT backend wins only if it passes the dependency gate **and**
has complete, comparable evidence for fixture quality, timings, memory, energy
method, model/artifact size, distribution, lifecycle, and stable domain-result
mapping. A candidate with better one-off speed but opaque asset installation,
unbounded native ownership, or unmaintainable packaging does not win.

The likely outcome may be more than one backend: a CoreML path can be the
model-controlled local option while system Speech is a valid system-managed
alternative, and whisper.cpp can remain a compatibility backend. Such a choice
still uses one stable STT contract and explicit capabilities; it does not make
the public API a union of native runtime APIs.
