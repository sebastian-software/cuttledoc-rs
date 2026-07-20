#!/usr/bin/env python3

import hashlib
import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.stt import load
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
    _get_feat_extract_output_lengths,
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: export_encoder_oracle.py <model-directory> <pcm-f32le>"
        )

    model_directory, pcm_path = sys.argv[1:]
    pcm_bytes = open(pcm_path, "rb").read()
    if not pcm_bytes or len(pcm_bytes) % 4 != 0:
        raise SystemExit("PCM fixture must contain non-empty little-endian float32")
    audio = np.frombuffer(pcm_bytes, dtype="<f4")

    started = time.perf_counter_ns()
    model = load(model_directory, lazy=False, strict=False)
    tower = model.audio_tower
    input_features, feature_attention_mask, num_audio_tokens = (
        model._preprocess_audio(audio)
    )

    fingerprints = {"input_features": fingerprint(input_features)}
    feature_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
    feature_lens_np = np.array(feature_lens)
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    chunk_size = tower.n_window * 2
    chunk_num = np.ceil(feature_lens_np / chunk_size).astype(np.int32)

    chunk_lengths = []
    chunks = []
    for batch_index, feature_length in enumerate(feature_lens_np):
        feature_length = int(feature_length)
        position = 0
        for chunk_index in range(int(chunk_num[batch_index])):
            remainder = feature_length % chunk_size
            chunk_length = (
                chunk_size
                if chunk_index < int(chunk_num[batch_index]) - 1 or remainder == 0
                else remainder
            )
            chunks.append(
                input_features[
                    batch_index, :, position : position + chunk_length
                ]
            )
            chunk_lengths.append(chunk_length)
            position += chunk_length

    max_chunk_length = max(chunk_lengths)
    padded_chunks = [
        mx.pad(chunk, [(0, 0), (0, max_chunk_length - chunk_length)])
        if chunk_length < max_chunk_length
        else chunk
        for chunk, chunk_length in zip(chunks, chunk_lengths)
    ]
    padded_feature = mx.stack(padded_chunks, axis=0)

    feature_lens_after_cnn = _get_feat_extract_output_lengths(
        mx.array(chunk_lengths)
    )
    feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
    max_len_after_cnn = int(feature_lens_after_cnn_np.max())

    hidden = padded_feature[:, :, :, None]
    hidden = nn.gelu(tower.conv2d1(hidden))
    fingerprints["conv2d1"] = fingerprint(hidden)
    hidden = nn.gelu(tower.conv2d2(hidden))
    fingerprints["conv2d2"] = fingerprint(hidden)
    hidden = nn.gelu(tower.conv2d3(hidden))
    fingerprints["conv2d3"] = fingerprint(hidden)

    batch_size, frequency, frames, channels = hidden.shape
    hidden = hidden.transpose(0, 2, 3, 1).reshape(
        batch_size, frames, channels * frequency
    )
    hidden = tower.conv_out(hidden)
    fingerprints["conv_out"] = fingerprint(hidden)
    hidden = hidden + tower.positional_embedding(hidden.shape[1])[None, :, :]

    hidden_list = [
        hidden[index, : int(feature_lens_after_cnn_np[index])]
        for index in range(hidden.shape[0])
    ]
    hidden = mx.concatenate(hidden_list, axis=0)
    fingerprints["encoder_input"] = fingerprint(hidden)

    aftercnn_lens_np = np.array(aftercnn_lens)
    window_aftercnn = max_len_after_cnn * (
        tower.n_window_infer // (tower.n_window * 2)
    )
    cumulative_chunk_lengths = [0]
    for cnn_length in aftercnn_lens_np:
        cnn_length = int(cnn_length)
        for _ in range(cnn_length // window_aftercnn):
            cumulative_chunk_lengths.append(window_aftercnn)
        remainder = cnn_length % window_aftercnn
        if remainder:
            cumulative_chunk_lengths.append(remainder)
    cumulative_sequence_lengths = np.cumsum(cumulative_chunk_lengths).tolist()

    attention_mask = tower._create_block_attention_mask(
        hidden.shape[0], cumulative_sequence_lengths, hidden.dtype
    )[None, None, :, :]
    hidden = hidden[None, :, :]
    for layer_index, layer in enumerate(tower.layers):
        hidden = layer(hidden, mask=attention_mask)
        if layer_index in (0, len(tower.layers) - 1):
            fingerprints[f"encoder_layer_{layer_index}"] = fingerprint(hidden)

    hidden = tower.ln_post(hidden[0])
    hidden = nn.gelu(tower.proj1(hidden))
    hidden = tower.proj2(hidden)
    fingerprints["audio_features"] = fingerprint(hidden)

    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "oracle": "mlx-audio Qwen3-ASR reference",
                "model_directory": model_directory,
                "pcm_path": pcm_path,
                "pcm_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
                "pcm_samples": len(audio),
                "feature_length": int(feature_lens_np[0]),
                "chunk_lengths": chunk_lengths,
                "aftercnn_length": int(aftercnn_lens_np[0]),
                "num_audio_tokens": num_audio_tokens,
                "attention_windows": cumulative_sequence_lengths,
                "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "fingerprints": fingerprints,
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
