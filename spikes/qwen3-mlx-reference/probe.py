#!/usr/bin/env python3

import json
import sys
import time
from importlib.metadata import version

import mlx.core as mx
from mlx_audio.stt import load


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: probe.py <model-directory> <audio-path> <language> <run-count>"
        )

    model_directory, audio_path, language, run_count_text = sys.argv[1:]
    run_count = int(run_count_text)
    if run_count < 1:
        raise SystemExit("run-count must be positive")

    load_started = time.perf_counter_ns()
    model = load(model_directory, lazy=False, strict=False)
    load_ms = elapsed_ms(load_started)

    runs = []
    for _ in range(run_count):
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        started = time.perf_counter_ns()
        first_result_ms = None
        text_parts = []
        updates = []
        reported_language = language
        prompt_tokens = 0
        generation_tokens = 0

        for update in model.generate(
            audio_path,
            language=language,
            max_tokens=256,
            stream=True,
            verbose=False,
        ):
            emitted_ms = elapsed_ms(started)
            if update.text and first_result_ms is None:
                first_result_ms = emitted_ms
            if update.text:
                text_parts.append(update.text)
            reported_language = update.language or reported_language
            prompt_tokens = update.prompt_tokens or prompt_tokens
            generation_tokens = update.generation_tokens or generation_tokens
            updates.append(
                {
                    "text": update.text,
                    "is_final": update.is_final,
                    "emitted_ms": emitted_ms,
                    "start_seconds": update.start_time,
                    "end_seconds": update.end_time,
                }
            )

        if first_result_ms is None:
            raise RuntimeError("Qwen3-ASR emitted no text update")
        if not updates or not updates[-1]["is_final"]:
            raise RuntimeError("Qwen3-ASR stream omitted a final update")

        runs.append(
            {
                "inference_ms": elapsed_ms(started),
                "first_result_ms": first_result_ms,
                "text": "".join(text_parts).strip(),
                "reported_language": reported_language,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "peak_memory_bytes": mx.get_peak_memory(),
                "updates": updates,
            }
        )

    print(
        json.dumps(
            {
                "runtime": {
                    "mlx": version("mlx"),
                    "mlx_audio": version("mlx-audio"),
                    "mlx_lm": version("mlx-lm"),
                },
                "load_ms": load_ms,
                "runs": runs,
            },
            separators=(",", ":"),
        )
    )


def elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1_000_000


if __name__ == "__main__":
    main()
