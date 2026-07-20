#!/usr/bin/env python3

"""Capture the pinned Voxtral causal encoder and adapter parity oracle."""

import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.stt import load
from mlx_lm.models.cache import RotatingKVCache

from frontend_oracle import fingerprint


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: encoder_oracle.py MODEL_DIR PCM_F32LE DELAY_MS FIXTURE_ID"
        )

    model_directory = Path(sys.argv[1])
    pcm_path = Path(sys.argv[2])
    delay_ms = int(sys.argv[3])
    fixture_id = sys.argv[4]
    if delay_ms <= 0:
        raise SystemExit("delay must be positive")

    pcm_bytes = pcm_path.read_bytes()
    if not pcm_bytes or len(pcm_bytes) % 4:
        raise SystemExit("fixture must contain non-empty little-endian float32 PCM")
    audio = np.frombuffer(pcm_bytes, dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise SystemExit("fixture must contain finite samples")

    model = load(model_directory, lazy=True, strict=False)
    mel, delay_tokens = model._prepare_mel(audio, delay_ms)
    encoder = model.encoder
    conv_stem = encoder.conv_stem(mel)
    mx.eval(conv_stem)

    sliding_window = encoder.config.sliding_window
    caches = [
        RotatingKVCache(max_size=sliding_window, keep=0)
        for _ in encoder.transformer_layers
    ]
    encoded_chunks = []
    fingerprints = {"conv_stem": fingerprint(conv_stem)}
    chunk_shapes = []

    for chunk_index, chunk_start in enumerate(
        range(0, conv_stem.shape[0], sliding_window)
    ):
        chunk_end = min(chunk_start + sliding_window, conv_stem.shape[0])
        x = conv_stem[chunk_start:chunk_end]
        chunk_len = x.shape[0]
        mask = caches[0].make_mask(chunk_len, window_size=sliding_window)
        chunk_shapes.append(
            {
                "index": chunk_index,
                "start": chunk_start,
                "length": chunk_len,
                "mask_kind": "causal" if isinstance(mask, str) else "array",
                "mask_shape": [] if isinstance(mask, str) else list(mask.shape),
            }
        )
        for layer_index, layer in enumerate(encoder.transformer_layers):
            x = layer(x, chunk_start, mask, cache=caches[layer_index])
            if layer_index in {0, 15, 31}:
                mx.eval(x)
                fingerprints[
                    f"chunk_{chunk_index}_layer_{layer_index}"
                ] = fingerprint(x)
        x = encoder.transformer_norm(x)
        mx.eval(x)
        fingerprints[f"chunk_{chunk_index}_norm"] = fingerprint(x)
        encoded_chunks.append(x)

    encoded = mx.concatenate(encoded_chunks, axis=0)
    downsample_factor = encoder.config.downsample_factor
    downsampled = encoded.reshape(
        encoded.shape[0] // downsample_factor,
        encoder.config.dim * downsample_factor,
    )
    projection0 = nn.gelu(encoder.audio_language_projection_0(downsampled))
    adapter = encoder.audio_language_projection_2(projection0)
    mx.eval(encoded, projection0, adapter)
    fingerprints["encoded"] = fingerprint(encoded)
    fingerprints["adapter_projection0_gelu"] = fingerprint(projection0)
    fingerprints["adapter"] = fingerprint(adapter)

    cache0 = caches[0]
    mx.eval(cache0.keys, cache0.values)
    fingerprints["layer_0_cache_keys"] = fingerprint(cache0.keys)
    fingerprints["layer_0_cache_values"] = fingerprint(cache0.values)

    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "oracle": "mlx-audio Voxtral Realtime causal encoder reference",
                "purpose": (
                    "development parity oracle for issue #19; "
                    "not a transcript or quality result"
                ),
                "runtime": {
                    "mlx": version("mlx"),
                    "mlx_audio": version("mlx-audio"),
                    "mlx_lm": version("mlx-lm"),
                    "numpy": version("numpy"),
                },
                "model": {
                    "source": (
                        "mistralai/Voxtral-Mini-4B-Realtime-2602@"
                        "2769294da9567371363522aac9bbcfdd19447add"
                    ),
                    "conversion": (
                        "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit@"
                        "fdebf7b2af834a1db4b8a3c99ab7480b333adf9e"
                    ),
                    "artifact_sha256": (
                        "6f59b425d8a1ceb2de795454558be63937cf75b59f9c9bc77accd85aaf32af05"
                    ),
                },
                "fixture": {
                    "id": fixture_id,
                    "pcm_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
                    "pcm_samples": int(audio.size),
                    "sample_rate_hz": 16_000,
                    "transcription_delay_ms": delay_ms,
                    "delay_tokens": delay_tokens,
                },
                "architecture": {
                    "layers": len(encoder.transformer_layers),
                    "dimension": encoder.config.dim,
                    "attention_heads": encoder.config.n_heads,
                    "head_dimension": encoder.config.head_dim,
                    "sliding_window": sliding_window,
                    "downsample_factor": downsample_factor,
                    "adapter_dimension": adapter.shape[1],
                },
                "chunks": chunk_shapes,
                "cache": {
                    "layer_0_offset": cache0.offset,
                    "layer_0_size": cache0.size(),
                    "layer_0_materialized_key_frames": cache0.keys.shape[2],
                    "layer_0_materialized_value_frames": cache0.values.shape[2],
                },
                "output": {
                    "encoded_frames": encoded.shape[0],
                    "adapter_frames": adapter.shape[0],
                },
                "fingerprints": fingerprints,
                "capabilities": {
                    "causal_encoder": True,
                    "rotating_kv_cache": True,
                    "sliding_window_attention": True,
                    "adapter_projection": True,
                    "decoder": False,
                    "transcription": False,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
