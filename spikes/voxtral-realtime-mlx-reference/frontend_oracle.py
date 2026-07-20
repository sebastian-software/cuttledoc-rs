#!/usr/bin/env python3

"""Capture the pinned Voxtral mel and causal-convolution frontend oracle."""

import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.stt import load

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
RAW_AUDIO_LENGTH_PER_TOKEN = 1_280
LEFT_PAD_TOKENS = 32


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: frontend_oracle.py MODEL_DIR PCM_F32LE DELAY_MS FIXTURE_ID"
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

    x = mel.T[None, :, :]
    conv0 = nn.gelu(encoder.conv_layers_0_conv(x))
    conv1_pretrunc = nn.gelu(encoder.conv_layers_1_conv(conv0)).squeeze(0)
    front_truncation = conv1_pretrunc.shape[0] % encoder.config.downsample_factor
    conv_stem = (
        conv1_pretrunc[front_truncation:]
        if front_truncation
        else conv1_pretrunc
    )
    mx.eval(mel, conv0, conv1_pretrunc, conv_stem)

    align_pad = (-len(audio)) % RAW_AUDIO_LENGTH_PER_TOKEN
    left_pad_samples = LEFT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOKEN
    right_pad_tokens = delay_tokens + 1 + 10
    right_pad_samples = align_pad + right_pad_tokens * RAW_AUDIO_LENGTH_PER_TOKEN
    padded_samples = left_pad_samples + len(audio) + right_pad_samples

    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "oracle": "mlx-audio Voxtral Realtime frontend reference",
                "purpose": (
                    "development parity oracle for issue #19; "
                    "not a transcript or quality result"
                ),
                "runtime": {
                    "mlx": version("mlx"),
                    "mlx_audio": version("mlx-audio"),
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
                    "sample_rate_hz": SAMPLE_RATE,
                },
                "padding": {
                    "transcription_delay_ms": delay_ms,
                    "delay_tokens": delay_tokens,
                    "left_pad_tokens": LEFT_PAD_TOKENS,
                    "left_pad_samples": left_pad_samples,
                    "alignment_pad_samples": align_pad,
                    "right_pad_tokens": right_pad_tokens,
                    "right_pad_samples": right_pad_samples,
                    "padded_samples": padded_samples,
                },
                "mel_frames": int(mel.shape[1]),
                "front_truncation_frames": int(front_truncation),
                "fingerprints": {
                    "mel_filters": fingerprint(model._ensure_mel_filters()),
                    "log_mel": fingerprint(mel),
                    "conv0_gelu": fingerprint(conv0),
                    "conv1_pretrunc_gelu": fingerprint(conv1_pretrunc),
                    "conv_stem": fingerprint(conv_stem),
                },
                "capabilities": {
                    "mel_frontend": True,
                    "causal_conv_stem": True,
                    "causal_encoder": False,
                    "transcription": False,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def fingerprint(value: mx.array) -> dict:
    data = np.ascontiguousarray(np.array(value).astype(np.float32, copy=False))
    flat = data.reshape(-1)
    if not flat.size:
        raise RuntimeError("cannot fingerprint an empty array")
    indices = sorted({0, min(1, flat.size - 1), min(2, flat.size - 1),
                      min(7, flat.size - 1), min(31, flat.size - 1),
                      flat.size // 3, (2 * flat.size) // 3, flat.size - 1})
    values = flat.astype(np.float64)
    mean = float(values.mean())
    return {
        "shape": list(data.shape),
        "source_dtype": str(value.dtype),
        "float32_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "mean": mean,
        "stddev": float(math.sqrt(np.mean((values - mean) ** 2))),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "l1": float(np.abs(values).sum()),
        "sample_indices": indices,
        "sample_values": [float(flat[index]) for index in indices],
    }


if __name__ == "__main__":
    main()
