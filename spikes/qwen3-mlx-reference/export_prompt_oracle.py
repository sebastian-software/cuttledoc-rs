#!/usr/bin/env python3

import hashlib
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: export_prompt_oracle.py <model-directory> <pcm-f32le> <language>"
        )

    model_directory, pcm_path, language = sys.argv[1:]
    pcm_bytes = open(pcm_path, "rb").read()
    if not pcm_bytes or len(pcm_bytes) % 4 != 0:
        raise SystemExit("PCM fixture must contain non-empty little-endian float32")
    audio = np.frombuffer(pcm_bytes, dtype="<f4")

    model = load(model_directory, lazy=False, strict=False)
    input_features, feature_attention_mask, num_audio_tokens = (
        model._preprocess_audio(audio)
    )
    input_ids = model._build_prompt(num_audio_tokens, language)
    audio_features = model.get_audio_features(
        input_features, feature_attention_mask
    )
    token_embeddings = model.model.embed_tokens(input_ids)
    audio_features_cast = audio_features.astype(token_embeddings.dtype)
    inputs_embeds = model._build_inputs_embeds(input_ids, audio_features)
    mx.eval(token_embeddings, audio_features_cast, inputs_embeds)

    token_ids = np.asarray(input_ids, dtype=np.int32)
    audio_indices = np.where(token_ids.reshape(-1) == model.config.audio_token_id)[
        0
    ].tolist()
    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "oracle": "mlx-audio Qwen3-ASR prompt reference",
                "model_directory": model_directory,
                "pcm_path": pcm_path,
                "pcm_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
                "language": language,
                "num_audio_tokens": num_audio_tokens,
                "prompt_length": int(token_ids.shape[-1]),
                "token_ids_sha256": hashlib.sha256(
                    token_ids.astype("<i4", copy=False).tobytes(order="C")
                ).hexdigest(),
                "token_ids": token_ids.reshape(-1).tolist(),
                "audio_token_indices": audio_indices,
                "fingerprints": {
                    "token_embeddings": fingerprint(token_embeddings),
                    "audio_features_bfloat16": fingerprint(audio_features_cast),
                    "inputs_embeds": fingerprint(inputs_embeds),
                },
            },
            separators=(",", ":"),
        )
    )


def fingerprint(value: mx.array) -> dict:
    source_dtype = str(value.dtype)
    materialized = np.asarray(mx.astype(mx.contiguous(value), mx.float32))
    flattened = materialized.reshape(-1)
    indices = sorted(
        {
            0,
            min(1, len(flattened) - 1),
            min(2, len(flattened) - 1),
            min(7, len(flattened) - 1),
            min(31, len(flattened) - 1),
            len(flattened) // 3,
            (2 * len(flattened)) // 3,
            len(flattened) - 1,
        }
    )
    return {
        "shape": list(materialized.shape),
        "source_dtype": source_dtype,
        "float32_sha256": hashlib.sha256(
            materialized.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest(),
        "mean": float(materialized.mean(dtype=np.float64)),
        "stddev": float(materialized.std(dtype=np.float64)),
        "minimum": float(materialized.min()),
        "maximum": float(materialized.max()),
        "l1": float(np.abs(materialized).sum(dtype=np.float64)),
        "sample_indices": indices,
        "sample_values": [float(flattened[index]) for index in indices],
    }


if __name__ == "__main__":
    main()
