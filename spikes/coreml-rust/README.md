# CoreML from Rust spike (#5)

This executable uses a real, compiled CoreML VAD model with three known input
features:

- `audio_input`: `Float32[1, 576]`
- `hidden_state`: `Float32[1, 128]`
- `cell_state`: `Float32[1, 128]`

It creates a fresh `MLModel` within an Objective-C autorelease pool, requests
`MLComputeUnits::All`, creates Rust-owned zero-filled `MLMultiArray` inputs,
executes synchronous prediction, copies the scalar `vad_output` into Rust, and
drops the complete native graph once per repetition. The model itself is never
checked into this repository.

## Run

Use a local compiled `silero-vad-unified-v6.0.0.mlmodelc` asset or another
compatible model, then run:

```sh
CUTTLEDOC_COREML_MODEL=/absolute/path/to/silero-vad-unified-v6.0.0.mlmodelc \
  cargo run --manifest-path spikes/coreml-rust/Cargo.toml --release
```

Set `CUTTLEDOC_COREML_REPETITIONS` to control the create → predict → drop loop.
The output is one `vad_output` scalar per run; it is intended to catch failures
and obvious lifecycle regressions, not to benchmark recognition quality.

## Boundary findings to capture

- `MLModel`, `MLMultiArray`, and feature-provider objects are not `Send` or
  `Sync`; a product adapter must retain them on one owned worker/actor rather
  than moving them through a general executor.
- This fixture uses explicit retained values plus a bounded autorelease pool.
  `MLMultiArray::dataPointer` is copied into Rust immediately after prediction;
  no foreign pointer crosses the adapter boundary.
- The spike uses synchronous prediction. The production adapter must separately
  decide whether to serialize an `MLModel`, use CoreML's async API, or run
  multiple owned models. Any shared `MLState` remains serialized.
- `MLComputeUnits::All` is a request, not proof that a particular chip or model
  used the Neural Engine. Record actual compute-plan/device diagnostics only
  where the tested OS exposes them.
