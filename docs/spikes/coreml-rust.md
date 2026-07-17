# CoreML from Rust result (#5)

**Status:** technically feasible on the macOS 26 Apple Silicon baseline; this
does not select a production STT backend.

**Evidence date:** 2026-07-16.

**Evidence host:** macOS 26.5.2 (25F84), arm64; Rust 1.95.0-nightly.

**Runnable artifact:** [`spikes/coreml-rust`](../../spikes/coreml-rust/).

## Question and boundary

Can a small, repository-owned Rust adapter load and invoke a genuine compiled
CoreML model while keeping Objective-C/CoreML objects out of the product API?

The checked-in executable answers that question with the accepted,
bounded `objc2`/`objc2-core-ml` dependency family. It loads a local compiled
`silero-vad-unified-v6.0.0.mlmodelc` fixture, explicitly requests
`MLComputeUnits::All`, supplies the named `Float32` inputs, runs synchronous
prediction, copies the scalar `vad_output` into Rust, then drops the complete
native graph. The fixture is not committed: model provenance, redistribution,
and model-quality evaluation belong to the later bakeoff.

## Reproduction and observed result

```sh
cargo check --manifest-path spikes/coreml-rust/Cargo.toml
CUTTLEDOC_COREML_MODEL=/absolute/path/to/silero-vad-unified-v6.0.0.mlmodelc \
  CUTTLEDOC_COREML_REPETITIONS=100 \
  cargo run --manifest-path spikes/coreml-rust/Cargo.toml --release
```

On the evidence host, all 100 create -> predict -> drop iterations completed
with `vad_output=0.0419312`. A separate `/usr/bin/time -l` invocation over the
same loop reported 1.25 s elapsed time, 18,612,224 bytes maximum resident size,
8,667,760 bytes peak memory footprint, zero swaps, and a 500 KiB executable.
These values are a lifecycle smoke-test measurement, not a recognition-quality
or device-throughput benchmark.

The direct dependency graph had no duplicate versions (`cargo tree
--duplicates` printed no duplicates). The checked-in `Cargo.lock` records the
tested Rust binding resolution (SHA-256
`fd705be7854b70c626b9bd42f8b963e3fbdefcf5df895849b3d628fa7e15311e`); the
model itself deliberately remains external to the source tree.

## Lifecycle findings

- `MLModel`, `MLMultiArray`, and feature-provider objects stay inside one
  `autoreleasepool` per iteration. The Rust executable does not retain native
  objects after the pool, and it copies the only scalar result before return.
- The current bindings mark these CoreML objects as neither `Send` nor `Sync`.
  A product adapter must use a dedicated worker/actor boundary rather than
  moving native objects through a general executor.
- The fixture uses `MLMultiArray::dataPointer` only to zero the input and copy
  the scalar output immediately. That API is deprecated in the bindings, so a
  product implementation should replace it with the supported scoped buffer
  handler before it becomes shared adapter code.
- `MLComputeUnits::All` is only a request. This run does not prove Neural
  Engine use because the minimal binding path did not expose an observed
  compute-plan/device record.
- Repeating the lifecycle found no failure or obvious growth signal, but 100
  in-process iterations are not proof of leak freedom. Longer Instruments and
  process-restart measurements remain required for a production decision.

## Decision and remaining work

The `objc2-core-ml` boundary is sufficient for a narrow internal experiment;
the result supports its **accepted, bounded** entry in the dependency policy.
It does not approve a public Objective-C API, a default STT engine, or a model
distribution route.

Before advancing the candidate, complete the remaining #5 evidence:

1. Test a checked, reproducible model fixture with its provenance/license and
   output schema recorded.
2. Add supported scoped `MLMultiArray` buffer access and typed error mapping.
3. Measure cold/warm load, first result, repeat-run memory, and actual
   compute-plan/device diagnostics where the OS exposes them.
4. Establish the owned asynchronous/cancellation boundary and serialize any
   shared `MLState`.
5. Compare accuracy, latency, memory, energy method, model size, and
   distribution cost against the other STT candidates using the common
   fixture set.

The selection criteria and the reasons this is not a device-use claim are in
[the Apple runtime evaluation](../apple-runtime-evaluation.md).
