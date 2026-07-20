# Direct Voxtral Realtime adapter over official MLX (#19)

This spike is the first repository-owned Voxtral Realtime boundary over the
official MLX C++ core. It deliberately replaces the `mlx-audio` session as the
product-boundary experiment without rejecting its model implementation as a
reference oracle.

The current milestone proves model identity, streaming lifecycle mechanics, and
exact fixed-fixture batch-transcription parity through the complete model:

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
5. The same adapter reproduces the reference runtime's offline-streaming
   padding, 128-bin log-mel frontend, and two causal Conv1d/GELU stages using
   the pinned model weights and official MLX operations.
6. A reusable repository-owned rotating KV cache and sliding-window mask drive
   all 32 encoder transformer layers, RoPE, 4x downsampling, and the
   audio-language adapter without linking Python or `mlx-lm`.
7. Repository-owned delay conditioning, 26 decoder layers with grouped-query
   attention, decoder caches, the tied language-model head, greedy generation,
   and Tekken decoding produce a final transcript through the Rust CLI.

Batch transcription is usable. The bounded session's small sum-of-squares
operation remains only a lifecycle fingerprint, so this milestone does **not**
claim incremental or live streaming. The manifest keeps those two capability
surfaces separate.

The cache, causal-mask, sliding-window, quantized-linear, and activation
primitives are implemented in the shared `mlx-direct` support layer and are
model-independent. Voxtral still owns its layer graph, weight names,
dimensions, delay conditioning, generation policy, and tokenizer. Official MLX
provides tensor execution and Metal kernels; there is no call to a prebuilt
Voxtral implementation in MLX.

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

The checked German audiobook fixture contains 213,440 samples. The 480 ms
frontend probe applies the exact reference padding to 276,480 samples, produces
1,728 log-mel frames, and produces 864 vectors of width 1,280 after the causal
Conv1d stem. Against the pinned Python oracle, the maximum sampled absolute
error is `5.36e-7` and the maximum aggregate relative error is `3.64e-7`, below
the fixed `2e-6` and `2e-5` tolerances. A CPU smoke run preserves the output
shapes; the reproducible parity claim is based on the GPU run.

The same fixture drives the bounded lifecycle test. Feeding it in
1,280-sample (80 ms) chunks into a 10,240-sample queue produced 40 explicit
backpressure responses and 42 successful steps. Every step consumed at most
5,120 samples (320 ms). The official-MLX fingerprint differed from the CPU
control by `1.24e-8` relative; empty-input, closed-input, done, and cancelled
states returned their pinned status codes.

The reference oracle and direct result are checked in at
`benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.frontend-480ms.json`
and
`benchmarks/raw/phase0.voxtral-realtime-mlx-direct.frontend-480ms-1/result.json`.
The runner validates them with `scripts/validate-voxtral-mlx-frontend.mjs`.

The encoder then processes the 864 stem frames as 750- and 114-frame chunks.
The second chunk uses an explicit `114 x 863` causal sliding-window mask. Its
cache has an absolute offset of 864 and a logical size of 750 while retaining
863 materialized frames during the multi-token update, matching the pinned
reference semantics. Across 14 encoder, adapter, and cache fingerprints, the
maximum sampled absolute error is `2.29e-5` (tolerance `5e-5`) and the maximum
aggregate relative error is `4.20e-6` (tolerance `1e-5`). The GPU probe takes
391 ms and peaks at 2.46 GB on the recorded M1 Ultra run. The checked-in oracle
and result use the corresponding `encoder-480ms` paths under
`benchmarks/oracles` and `benchmarks/raw`.

The batch decoder adds a 39-token delay-conditioned prompt and processes 216
adapter frames through 26 decoder layers. On the pinned fixture it emits the
same 178 token IDs and the exact same German transcript as the pinned
`mlx-audio` reference. Across decoder fingerprints, the maximum sampled
absolute error is `2.44e-4` (tolerance `5e-4`) and the maximum aggregate
relative error is `2.09e-4` (tolerance `5e-4`). The recorded M1 Ultra run takes
about 3.94 seconds for 13.34 seconds of audio and peaks at 5.07 GB before memory
optimization. The checked-in decoder oracle/result and
`scripts/validate-voxtral-mlx-decoder.mjs` make exact token and text parity a
hard gate.

## Remaining parity gates

1. Move the proven frontend, encoder, decoder, and tokenizer state behind the
   bounded session handle and emit incremental transcript updates.
2. Preserve fixed step budgets, backpressure, cancellation, and exact final
   text while replacing the lifecycle fingerprint with real model work.
3. Rerun the 80/320 ms live-input, multilingual audiobook, and held-out
   German-first target-domain gates through this repository-owned boundary.
4. Only then compare packaging and lifecycle evidence with the pure-C/MPS
   control and decide whether Voxtral becomes a product engine.
