#!/usr/bin/env python3

import hashlib
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load
from mlx_lm.models.cache import make_prompt_cache


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: export_decoder_oracle.py <model-directory> <pcm-f32le> <language>"
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
    inputs_embeds = model._build_inputs_embeds(input_ids, audio_features)
    mx.eval(inputs_embeds)

    cache = make_prompt_cache(model)
    model.model(inputs_embeds=inputs_embeds[:, :-1], cache=cache)
    hidden = model.model(inputs_embeds=inputs_embeds[:, -1:], cache=cache)
    logits = model.model.embed_tokens.as_linear(hidden)[:, -1, :]
    first_token = int(mx.argmax(logits, axis=-1).item())
    prefill_cache = {
        "layer_0_keys": fingerprint(cache[0].state[0]),
        "layer_0_values": fingerprint(cache[0].state[1]),
        "layer_27_keys": fingerprint(cache[-1].state[0]),
        "layer_27_values": fingerprint(cache[-1].state[1]),
    }

    first_embedding = model.model.embed_tokens(mx.array([[first_token]]))
    second_hidden = model.model(inputs_embeds=first_embedding, cache=cache)
    second_logits = model.model.embed_tokens.as_linear(second_hidden)[:, -1, :]
    second_token = int(mx.argmax(second_logits, axis=-1).item())

    generated_tokens = []
    first_logprobs = None
    for token, logprobs in model.stream_generate(
        audio, language=language, max_tokens=256, verbose=False
    ):
        if first_logprobs is None:
            first_logprobs = logprobs
        generated_tokens.append(int(token))
    text = model._tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "oracle": "mlx-audio Qwen3-ASR decoder reference",
                "model_directory": model_directory,
                "pcm_path": pcm_path,
                "pcm_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
                "language": language,
                "prompt_length": int(input_ids.shape[-1]),
                "first_token": first_token,
                "second_token": second_token,
                "generated_tokens": generated_tokens,
                "text": text,
                "prefill": {
                    "hidden": fingerprint(hidden),
                    "logits": fingerprint(logits),
                    **prefill_cache,
                },
                "second_step": {
                    "hidden": fingerprint(second_hidden),
                    "logits": fingerprint(second_logits),
                },
                "first_logprobs": fingerprint(first_logprobs),
                "first_top_tokens": top_tokens(logits, 10),
                "second_top_tokens": top_tokens(second_logits, 10),
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


def top_tokens(logits: mx.array, count: int) -> list[dict]:
    values = np.asarray(mx.astype(logits, mx.float32)).reshape(-1)
    indices = np.argsort(values)[-count:][::-1]
    return [{"token": int(index), "logit": float(values[index])} for index in indices]


if __name__ == "__main__":
    main()
