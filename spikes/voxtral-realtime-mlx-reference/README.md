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
