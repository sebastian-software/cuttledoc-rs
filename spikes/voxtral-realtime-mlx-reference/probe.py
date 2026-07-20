#!/usr/bin/env python3

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: probe.py <model-directory> <f32le-path> "
            "<delay-ms> <run-count> <max-tokens>"
        )

    model_directory, audio_path, delay_text, run_count_text, max_tokens_text = (
        sys.argv[1:]
    )
    delay_ms = int(delay_text)
    run_count = int(run_count_text)
    max_tokens = int(max_tokens_text)
    if delay_ms not in {*range(80, 1201, 80), 2400}:
        raise SystemExit("delay-ms must be 80..1200 in 80 ms steps, or 2400")
    if run_count < 1 or max_tokens < 1:
        raise SystemExit("run-count and max-tokens must be positive")

    audio = np.fromfile(Path(audio_path), dtype="<f4")
    if audio.size == 0 or not np.isfinite(audio).all():
        raise SystemExit("audio must contain finite mono f32le samples")

    load_started = time.perf_counter_ns()
    model = load(model_directory, lazy=False, strict=False)
    load_ms = elapsed_ms(load_started)

    runs = []
    for _ in range(run_count):
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        started = time.perf_counter_ns()
        output = model.generate(
            audio,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
            transcription_delay_ms=delay_ms,
            verbose=False,
        )
        inference_ms = elapsed_ms(started)
        if not output.text.strip():
            raise RuntimeError("Voxtral Realtime emitted no text")
        runs.append(
            {
                "inference_ms": inference_ms,
                "model_total_time_ms": output.total_time * 1000,
                "text": output.text.strip(),
                "prompt_tokens": output.prompt_tokens,
                "generation_tokens": output.generation_tokens,
                "total_tokens": output.total_tokens,
                "generation_tokens_per_second": output.generation_tps,
                "peak_memory_bytes": mx.get_peak_memory(),
            }
        )

    print(
        json.dumps(
            {
                "runtime": {
                    "mistral_common": version("mistral-common"),
                    "mlx": version("mlx"),
                    "mlx_audio": version("mlx-audio"),
                    "mlx_lm": version("mlx-lm"),
                },
                "input": {
                    "sample_count": int(audio.size),
                    "sample_rate_hz": 16000,
                    "transcription_delay_ms": delay_ms,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                "load_ms": load_ms,
                "runs": runs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1_000_000


if __name__ == "__main__":
    main()
