# Voxtral Realtime MLX reference

This spike evaluates Voxtral Mini 4B Realtime 2602 as a serious third
Apple-local ASR candidate. It is distinct from the Voxtral TTS spike: this
model consumes speech, uses a causal audio encoder, and exposes configurable
transcription delay.

The official model is Apache 2.0, supports 13 languages, and now has official
vLLM and Transformers implementations. Mistral also links community MLX, pure
C/MPS, and Rust implementations. The old repository disposition that treated
the model as vLLM/CUDA-only is therefore obsolete.

The first local oracle pins the 4-bit MLX conversion and `mlx-audio` revision.
It measures greedy decoding at 480 ms, the publisher's recommended
latency/quality setting, and at 2,400 ms as a quality control. This Python
runtime is reference evidence, not an accepted product dependency or Rust
boundary. A leading result may justify a repository-owned adapter over
official MLX or a narrower C/MPS integration.

Download and verify the 3.15 GB snapshot:

```sh
bash scripts/fetch-voxtral-realtime-mlx-model.sh \
  /absolute/path/to/voxtral-realtime-4b-mlx-4bit
```

Create the locked Python environment:

```sh
uv sync \
  --project spikes/voxtral-realtime-mlx-reference \
  --python /opt/homebrew/bin/python3.12
```

Run the 15-clip multilingual audiobook pilot at the recommended delay:

```sh
CUTTLEDOC_VOXTRAL_REALTIME_FIXTURE_DIR=/absolute/path/to/audiobook-pcm \
  CUTTLEDOC_VOXTRAL_REALTIME_DELAY_MS=480 \
  bash scripts/run-voxtral-realtime-mlx-reference.sh
```

The quality runner loads complete digest-pinned PCM clips before inference.
Although the implementation contains a stateful streaming session, this
offline matrix does not claim measured live-input first-result latency.
Streaming chunking, cancellation, and lifecycle behavior are separate gates.

## Recorded controls

The audiobook pilot and the independent FLEURS short-read control both favor
2,400 ms in aggregate:

| Source | 480 ms WER / CER | 2,400 ms WER / CER |
| --- | ---: | ---: |
| 15 audiobook clips | 5.84% / 1.67% | 4.25% / 1.45% |
| 10 FLEURS clips | 5.13% / 1.09% | 3.82% / 1.01% |

The exact FLEURS aggregates are
[`480 ms`](../../benchmarks/raw/phase0.voxtral-realtime-mlx-reference-480ms.multilingual-fleurs-10-1/result.json)
and
[`2,400 ms`](../../benchmarks/raw/phase0.voxtral-realtime-mlx-reference-2400ms.multilingual-fleurs-10-1/result.json).
Their boundary-review alignments are part of the shared
[`error report`](../../benchmarks/analysis/phase0.multilingual-fleurs-10-1.errors.json).

The aggregate agreement is not a universal delay rule. Per-language results
change by source, and the German FLEURS errors are dominated by token
boundaries rather than character changes. The next experiment must feed audio
incrementally and record first, stable, and final output plus cancellation and
repeated lifecycle behavior.

Run that true-input streaming experiment with 80 ms, wall-clock-paced chunks:

```sh
bash scripts/run-voxtral-realtime-streaming.sh
```

The probe deliberately distinguishes the available API from measured
semantics. It records every append delta against both wall time and the amount
of audio fed, endpoint finalization, step duration and scheduling lateness. It
also inspects the session for cancellation and finalization methods, abandons
one prefix session, and then runs repeated complete lifecycles. Dropping a
Python session reference is only an abandonment smoke test; it must not be
reported as cooperative cancellation.
