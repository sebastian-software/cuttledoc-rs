#!/usr/bin/env python3

"""Capture pinned Voxtral delay-conditioned decoder and text parity."""

import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load
from mlx_audio.stt.models.voxtral_realtime.decoder import compute_time_embedding

from frontend_oracle import fingerprint


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: decoder_oracle.py MODEL_DIR PCM_F32LE DELAY_MS FIXTURE_ID"
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
    model._ensure_ada_scales(delay_ms)
    mel, delay_tokens = model._prepare_mel(audio, delay_ms)
    adapter = model.encoder(mel)
    mx.eval(adapter)

    decoder = model.decoder
    prompt_ids = [model.config.bos_token_id] + [
        model.config.streaming_pad_token_id
    ] * (model.config.n_left_pad_tokens + delay_tokens)
    prompt_text_embeddings = decoder.embed_tokens(mx.array(prompt_ids))
    prefix_embeddings = adapter[: len(prompt_ids)] + prompt_text_embeddings

    time_embedding = compute_time_embedding(float(delay_tokens), decoder.config.dim)
    time_embedding = time_embedding.astype(
        decoder.layers[0].ada_rms_norm_t_cond.ada_down.weight.dtype
    )
    fingerprints = {
        "adapter": fingerprint(adapter),
        "time_embedding": fingerprint(time_embedding),
        "ada_scale_layer_0": fingerprint(decoder._ada_scales[0]),
        "ada_scale_layer_12": fingerprint(decoder._ada_scales[12]),
        "ada_scale_layer_25": fingerprint(decoder._ada_scales[25]),
        "prompt_text_embeddings": fingerprint(prompt_text_embeddings),
        "prefix_embeddings": fingerprint(prefix_embeddings),
    }

    caches = decoder.make_cache()
    hidden = decoder_forward(
        decoder,
        prefix_embeddings,
        start_pos=0,
        caches=caches,
        fingerprints=fingerprints,
        fingerprint_prefix="prefill",
    )
    logits = decoder.logits(hidden[-1])
    mx.eval(logits)
    fingerprints["prefill_logits"] = fingerprint(logits)

    generated = []
    next_token = int(mx.argmax(logits).item())
    last_logits = logits
    forward_steps = 0
    eos = model.config.eos_token_id
    for position in range(len(prompt_ids), adapter.shape[0]):
        token = next_token
        generated.append(token)
        if token == eos or len(generated) > 4096:
            break

        decoder_input = adapter[position] + decoder.embed_token(token)
        capture = forward_steps == 0
        if capture:
            fingerprints["decode_0_input"] = fingerprint(decoder_input)
        hidden = decoder_forward(
            decoder,
            decoder_input[None, :],
            start_pos=position,
            caches=caches,
            fingerprints=fingerprints if capture else None,
            fingerprint_prefix="decode_0" if capture else None,
        )
        logits = decoder.logits(hidden.squeeze(0))
        mx.eval(logits)
        if capture:
            fingerprints["decode_0_logits"] = fingerprint(logits)
        last_logits = logits
        next_token = int(mx.argmax(logits).item())
        forward_steps += 1
    else:
        generated.append(next_token)

    fingerprints["final_logits"] = fingerprint(last_logits)
    cache_keys, cache_values = caches[0].state
    mx.eval(cache_keys, cache_values)
    fingerprints["decoder_layer_0_cache_keys"] = fingerprint(cache_keys)
    fingerprints["decoder_layer_0_cache_values"] = fingerprint(cache_values)

    text_tokens = generated[:-1] if generated and generated[-1] == eos else generated
    text = model._tokenizer.decode(text_tokens).strip()
    if not text:
        raise RuntimeError("reference decoder emitted no text")

    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "oracle": "mlx-audio Voxtral Realtime decoder reference",
                "purpose": (
                    "fixed-fixture token and text parity oracle for issue #19; "
                    "not held-out quality evidence"
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
                    "layers": len(decoder.layers),
                    "dimension": decoder.config.dim,
                    "attention_heads": decoder.config.n_heads,
                    "kv_heads": decoder.config.n_kv_heads,
                    "head_dimension": decoder.config.head_dim,
                    "hidden_dimension": decoder.config.hidden_dim,
                    "sliding_window": decoder.config.sliding_window,
                    "vocabulary_size": decoder.config.vocab_size,
                    "ada_bottleneck_dimension": (
                        decoder.config.ada_rms_norm_t_cond_dim
                    ),
                },
                "prompt": {
                    "bos_token_id": model.config.bos_token_id,
                    "streaming_pad_token_id": model.config.streaming_pad_token_id,
                    "eos_token_id": eos,
                    "token_ids": prompt_ids,
                    "length": len(prompt_ids),
                    "adapter_frames": int(adapter.shape[0]),
                },
                "generation": {
                    "tokens": generated,
                    "token_count": len(generated),
                    "forward_steps": forward_steps,
                    "finish_reason": "eos" if generated[-1] == eos else "audio_end",
                    "text": text,
                },
                "cache": {
                    "layer_0_offset": caches[0].offset,
                    "layer_0_size": caches[0].size(),
                    "layer_0_state_frames": int(cache_keys.shape[2]),
                },
                "fingerprints": fingerprints,
                "capabilities": {
                    "delay_conditioning": True,
                    "decoder": True,
                    "decoder_kv_cache": True,
                    "tekken_decode": True,
                    "greedy_transcription": True,
                    "streaming_session": False,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def decoder_forward(
    decoder,
    embeddings,
    *,
    start_pos,
    caches,
    fingerprints=None,
    fingerprint_prefix=None,
):
    hidden = embeddings
    for layer_index, layer in enumerate(decoder.layers):
        ada_scale = decoder._ada_scales[layer_index]
        hidden = layer(
            hidden,
            start_pos,
            ada_scale=ada_scale,
            cache=caches[layer_index],
        )
        if fingerprints is not None and layer_index in {0, 12, 25}:
            mx.eval(hidden)
            fingerprints[
                f"{fingerprint_prefix}_layer_{layer_index}"
            ] = fingerprint(hidden)
    hidden = decoder.norm(hidden)
    if fingerprints is not None:
        mx.eval(hidden)
        fingerprints[f"{fingerprint_prefix}_norm"] = fingerprint(hidden)
    return hidden


if __name__ == "__main__":
    main()
