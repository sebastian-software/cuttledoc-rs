# Direct Voxtral Realtime adapter over official MLX (#19)

This spike is the first repository-owned Voxtral Realtime boundary over the
official MLX C++ core. It deliberately replaces the `mlx-audio` session as the
product-boundary experiment without rejecting its model implementation as a
reference oracle.

The current milestone proves model identity and streaming lifecycle mechanics:

1. The C++ adapter loads the pinned 3.13 GB safetensors through official MLX
   and validates 1,523 tensors, their dtypes, 406 affine 4-bit modules, and
   critical encoder and decoder shapes.
2. Rust owns a narrow C ABI session with an explicit queue capacity, an
   independent per-step ingestion budget, all-or-nothing feed backpressure,
   end-of-audio, cancellation, and deterministic status codes.
3. Each `step()` removes a fixed snapshot before executing official MLX work.
   New producer input therefore cannot extend the active step, which is the
   failure mode found in the reference runtime at 80 ms input cadence.
4. No MLX array, stream, model object, or C++ allocation crosses the ABI.

This milestone does **not** implement or claim transcription. The small
sum-of-squares operation is an execution fingerprint over exactly the bounded
audio slice, not a substitute for the Voxtral frontend, encoder, or decoder.

## Pinned inputs

- official MLX `v0.32.0`,
  `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`;
- official `mistralai/Voxtral-Mini-4B-Realtime-2602`,
  `2769294da9567371363522aac9bbcfdd19447add`;
- `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`,
  `fdebf7b2af834a1db4b8a3c99ab7480b333adf9e`.

The model and conversion are Apache-2.0. Official MLX is MIT. `mlx-audio`
remains a pinned reference-only oracle and is not linked by this adapter.

## Reproduce the boundary test

The default paths match the local Phase 0 workspace:

```sh
scripts/run-voxtral-mlx-direct-spike.sh
```

Override them when necessary:

```sh
CUTTLEDOC_MLX_SOURCE_DIR=/absolute/path/to/mlx-v0.32.0 \
CUTTLEDOC_VOXTRAL_MODEL_DIR=/absolute/path/to/model \
CUTTLEDOC_VOXTRAL_PCM_FIXTURE=/absolute/path/to/mono-16khz.f32le \
scripts/run-voxtral-mlx-direct-spike.sh
```

The checked German audiobook fixture contains 213,440 samples. Feeding it in
1,280-sample (80 ms) chunks into a 10,240-sample queue produced 40 explicit
backpressure responses and 42 successful steps. Every step consumed at most
5,120 samples (320 ms). The official-MLX fingerprint differed from the CPU
control by `1.24e-8` relative; empty-input, closed-input, done, and cancelled
states returned their pinned status codes.

## Remaining parity gates

1. Port the streaming audio frontend and compare every intermediate shape and
   numerical fingerprint with the pinned Python oracle.
2. Port the 32-layer causal encoder with its cache and sliding-window rules.
3. Port Tekken tokenization, delay conditioning, the 26-layer decoder, and its
   cache behind the same session handle.
4. Require exact fixed-fixture token/text parity before claiming ASR.
5. Rerun the 80/320 ms live-input, multilingual audiobook, and held-out
   German-first target-domain gates through this repository-owned boundary.
6. Only then compare packaging and lifecycle evidence with the pure-C/MPS
   control and decide whether Voxtral becomes a product engine.
