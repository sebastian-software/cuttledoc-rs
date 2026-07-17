# ADR-0010: Capability-oriented engine ownership and backend identity

## Status

Accepted

## Context

ADRs 0004 and 0008 define task-oriented engines and range-addressed
transcription updates. The Phase 0 spikes now provide enough native evidence to
decide the remaining ownership and discovery boundaries:

- the contract reducer accepts finals-only and volatile-to-final streams;
- Apple Speech maps streamed PCM, volatile replacements, word timestamps,
  confidence, cancellation, and system-managed assets through an owned Swift C
  ABI;
- CoreML objects are neither `Send` nor `Sync` in the selected Rust bindings;
- the direct MLX adapter has a process-global default device, synchronous graph
  evaluation, and a safe cancellation boundary only between model steps; and
- the Node spike proves a thin prebuilt binding can expose a Rust-owned async
  iterator to both ESM and CommonJS consumers.

A public runtime trait or a promise that every native handle is thread-safe
would contradict this evidence. A closed backend enum would also mix product
selection with model families, providers, and internal runtimes.

## Decision

### Identity and selection

Use these terms consistently:

- A **task** is a stable product operation and result contract, initially
  speech recognition or text generation.
- A **backend** is one product-selectable implementation of one task. It owns
  orchestration and maps native behavior into the task contract.
- A **runtime** is an internal execution substrate such as CoreML, official
  MLX, whisper.cpp/Metal, Apple Speech, or an HTTP client.
- A **model** is a logical, revisioned unit used by a backend. It may be a
  Cuttledoc-managed artifact set, an opaque system asset, or a remote model ID.
- A **provider** is the authority that provisions a model or service, supplies
  credentials or system assets, and defines its lifecycle. A provider is not a
  result type.
- A **capability** is a discovered property of an exact backend/model/provider
  configuration on the current host. It is not inferred from the runtime name.

`auto` is a selection policy, not a backend. Rust uses an open `BackendId`
newtype plus known constants; Node uses known string completions plus
`string & {}`. Initial known recognition IDs are:

- `parakeet-local`;
- `whisper-local`;
- `apple-speech`; and
- `openai-transcription`.

The MLX encoder spike does not receive a public backend ID until an end-to-end
task adapter exists. Its runtime may later back a new backend or replace an
implementation behind an existing product backend without changing
transcription result types.

### Engine and worker ownership

A public engine is a `Send + Sync` proxy to an engine-owned scheduler. A native
runtime handle does not need to be `Send`, `Sync`, or movable:

1. the adapter creates and destroys native state on its owned worker or actor;
2. requests enter a bounded FIFO queue;
3. the default execution limit is one active request per engine;
4. a backend may advertise and use a higher bounded limit only after runtime
   evidence proves it safe; and
5. no native pointer, actor, executor, device, or stream crosses the adapter.

Queue policy is explicit: callers choose to await capacity or reject
immediately with `ENGINE_BUSY`. Streaming PCM uses a bounded asynchronous
writer. A successful write means the adapter accepted owned/copy-safe input;
when full, the write waits or returns backpressure according to policy. Audio
is never silently dropped.

The top-level one-shot API owns an ephemeral context and closes it before
return. Engine reuse and caching require an explicit `Cuttledoc` context; there
is no hidden process-global engine cache.

### Cancellation and shutdown

Cancellation is truthful rather than uniform:

- queued work is removed immediately;
- running work receives a cooperative signal;
- Apple Speech can cancel its actor task and finish its input stream;
- synchronous MLX or native decoder work observes cancellation at the next
  chunk or decoder-step boundary; and
- cancellation never destroys native state while a foreign call is active.

Capabilities report the latest cancellation boundary. `close()` is idempotent,
stops admission, cancels queued and active work, waits for the adapter's safe
boundary, destroys native state on its owner, and closes output streams. Only
awaited `close()` is deterministic; Rust `Drop` and Node finalizers are
best-effort safety nets.

### Capability and availability contract

Capability discovery returns task behavior, not native handles. Recognition
capabilities cover:

- file, bounded PCM, and streaming PCM input;
- supported languages and dynamic-locale status;
- segment/word timestamps and confidence;
- incremental final and volatile result behavior;
- model provisioning and supported model operations;
- serialized or bounded-concurrent execution;
- queue/backpressure policy;
- cancellation boundary; and
- optional runtime/model revision diagnostics.

Availability is a state with a stable machine-readable reason, not a Boolean
plus free text. Initial reason codes are:

- `unsupported_platform`;
- `unsupported_hardware`;
- `model_missing`;
- `model_invalid`;
- `system_asset_unavailable`;
- `credentials_missing`;
- `native_component_unavailable`;
- `disabled_by_policy`; and
- `native_load_failed`.

Human detail and remediation may expand without changing these codes.
Diagnostics may name CoreML, MLX, Metal, Apple Speech, or a remote provider,
but task results and update types do not.

### Model provisioning

The model catalog describes one of three provisioning modes:

1. `cuttledoc-managed`: immutable manifest, download, digest verification,
   atomic installation, and removal;
2. `system-managed`: provider inventory/install/reservation semantics, which
   may omit bytes, digest, quantization, and revision; or
3. `remote`: provider/model identity and credentials, with no local artifact
   operation.

Callers inspect supported operations instead of assuming every model can be
downloaded, verified, or removed as files.

### Result and binding parity

Speech recognition continues to emit ADR-0008 `TranscriptionUpdate` values.
Apple Speech emits volatile replacements and a final replacement; finals-only
backends emit only final replacements. Both use the same reducer.

Text generation has its own ordered text/token delta stream and completion
reason. It never reuses time ranges or transcription replacements. Speech
synthesis remains deferred under ADR-0009.

Rust domain types are authoritative. The Node package generates or mechanically
checks TypeScript declarations from the Rust binding surface in CI, and both
boundaries execute the same checked-in semantic contract vectors. JavaScript
does not reimplement the reducer.

## Consequences

### Positive

- Thread-affine CoreML and Swift state and process-serialized MLX state fit the
  same safe public ownership model.
- Runtime, model, and provider changes do not change task results.
- Availability failures are actionable and stable across Rust, CLI, and Node.
- System assets and remote models do not need fake filesystem manifests.
- Backpressure, cancellation, and shutdown have testable semantics.

### Negative

- Every engine needs a scheduler and explicit shutdown path.
- A proxy may add one queue hop even when a runtime could run directly.
- Capability records are richer than a backend-name list.
- Higher concurrency must be enabled by evidence rather than assumption.

## Component dispositions

| Component | Disposition at this decision |
| --- | --- |
| CoreML through `objc2-core-ml` | Accepted bounded dependency behind a repository-owned adapter |
| Apple Speech framework | System component behind a repository-owned Swift C ABI |
| Official MLX C++ core | Approved upstream foundation behind a repository-owned task C ABI |
| `mlx-c` and community MLX wrappers | Reference/control only |
| whisper.cpp | Repository-owned boundary candidate pending the production bakeoff |
| Legacy Parakeet and Whisper Node addons | Compatibility and benchmark references only |
| Remote HTTP providers | Repository-owned task adapters; provider SDKs require separate acceptance |
| `napi-rs` | Accepted bounded dependency for the thin generated/checked Node boundary |

## Validation

The Phase 0 reducer, Node packaging, CoreML, Apple Speech, and MLX spikes are
the evidence for this decision. Phase 1 must turn these shapes into workspace
types and CI checks. A backend may widen execution concurrency or claim a
stronger cancellation boundary only with a repeatable runtime test and updated
capability evidence.
