#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
import math
import subprocess
import threading
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load


SAMPLE_RATE_HZ = 16_000


def main() -> None:
    args = parse_args()
    fixture, audio_path = load_fixture(
        Path(args.manifest),
        args.fixture_id,
        Path(args.fixture_dir),
    )
    audio = np.fromfile(audio_path, dtype="<f4")
    verify_audio(audio, audio_path, fixture)

    load_started = time.perf_counter_ns()
    model = load(args.model_directory, lazy=False, strict=False)
    load_ms = elapsed_ms(load_started)

    api_session = model.create_streaming_session(
        transcription_delay_ms=args.delay_ms[0],
    )
    session_class = type(api_session)
    api_contract = {
        "session_class": f"{session_class.__module__}.{session_class.__name__}",
        "feed": callable(getattr(session_class, "feed", None)),
        "step": callable(getattr(session_class, "step", None)),
        "close": callable(getattr(session_class, "close", None)),
        "done": isinstance(getattr(session_class, "done", None), property),
        "generated": hasattr(api_session, "generated"),
        "cancel": callable(getattr(session_class, "cancel", None)),
        "finalize_step": callable(
            getattr(session_class, "finalize_step", None)
        ),
        "close_semantics": "end-of-audio and right-padding flush; not cancellation",
    }
    del api_session

    abandonment = probe_abandonment(
        model,
        audio,
        delay_ms=args.delay_ms[0],
        prefix_ms=args.abandon_after_ms,
        max_decode_tokens=args.max_decode_tokens,
        max_tokens=args.max_tokens,
    )

    runs = []
    for delay_ms in args.delay_ms:
        for repetition in range(1, args.repetitions + 1):
            runs.append(
                run_stream(
                    model,
                    audio,
                    fixture["reference_text"],
                    delay_ms=delay_ms,
                    repetition=repetition,
                    chunk_ms=args.chunk_ms,
                    max_decode_tokens=args.max_decode_tokens,
                    max_tokens=args.max_tokens,
                    realtime=args.realtime,
                )
            )

    batch_controls = []
    for delay_ms in args.delay_ms:
        started = time.perf_counter_ns()
        output = model.generate(
            audio,
            max_tokens=args.max_tokens,
            temperature=0.0,
            stream=False,
            transcription_delay_ms=delay_ms,
            verbose=False,
        )
        batch_controls.append(
            {
                "transcription_delay_ms": delay_ms,
                "inference_ms": elapsed_ms(started),
                "text": output.text.strip(),
                "quality": quality_metrics(
                    fixture["reference_text"],
                    output.text,
                ),
                "matches_stream_runs": all(
                    run["text"] == output.text.strip()
                    for run in runs
                    if run["transcription_delay_ms"] == delay_ms
                ),
            }
        )

    result = {
        "schema_version": "1.0.0",
        "captured_at": utc_now(),
        "source_revision": git_revision(),
        "candidate": {
            "id": "voxtral-mini-4b-realtime-mlx-reference",
            "model": (
                "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
                "@fdebf7b2af834a1db4b8a3c99ab7480b333adf9e; converted from "
                "mistralai/Voxtral-Mini-4B-Realtime-2602"
                "@2769294da9567371363522aac9bbcfdd19447add"
            ),
            "runtime": (
                "reference-only mlx-audio v0.4.5"
                "@64e8416c303fb3b3463dab8eb4ebd78c55a87c1a "
                "over official MLX 0.32.0"
            ),
            "boundary": (
                "external Python streaming oracle; not a product dependency "
                "or accepted interop boundary"
            ),
        },
        "runtime": {
            "mistral_common": version("mistral-common"),
            "mlx": version("mlx"),
            "mlx_audio": version("mlx-audio"),
            "mlx_lm": version("mlx-lm"),
        },
        "fixture": {
            "id": fixture["id"],
            "language": fixture["language"],
            "gold_status": fixture.get("gold_status"),
            "reference_text": fixture["reference_text"],
            "audio_path": str(audio_path),
            "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": int(audio.size),
            "duration_ms": audio.size / SAMPLE_RATE_HZ * 1000,
        },
        "procedure": {
            "repetitions": args.repetitions,
            "transcription_delays_ms": args.delay_ms,
            "chunk_ms": args.chunk_ms,
            "max_decode_tokens_per_step": args.max_decode_tokens,
            "max_tokens": args.max_tokens,
            "realtime_pacing": args.realtime,
            "first_chunk_available_after_capture": args.realtime,
            "producer_thread_independent_from_mlx_executor": True,
            "consumer_step_after_feed_signal": True,
            "feed_signals_may_coalesce_while_mlx_step_runs": True,
            "additional_step_calls_after_close_until_done": True,
        },
        "load_ms": load_ms,
        "api_contract": api_contract,
        "abandonment_probe": abandonment,
        "runs": runs,
        "batch_controls": batch_controls,
    }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure true incremental audio input through the pinned "
            "Voxtral Realtime MLX reference session."
        )
    )
    parser.add_argument("model_directory")
    parser.add_argument("manifest")
    parser.add_argument("fixture_dir")
    parser.add_argument("fixture_id")
    parser.add_argument(
        "--delay-ms",
        action="append",
        type=int,
        required=True,
        help="Repeat for each transcription delay to measure.",
    )
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--max-decode-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--abandon-after-ms", type=int, default=960)
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="Feed chunks as fast as possible instead of wall-clock pacing.",
    )
    parser.set_defaults(realtime=True)
    args = parser.parse_args()
    for delay_ms in args.delay_ms:
        if delay_ms not in {*range(80, 1201, 80), 2400}:
            parser.error(
                "delay-ms must be 80..1200 in 80 ms steps, or 2400"
            )
    for name in (
        "chunk_ms",
        "max_decode_tokens",
        "max_tokens",
        "repetitions",
        "abandon_after_ms",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    chunk_samples = args.chunk_ms * SAMPLE_RATE_HZ / 1000
    if not chunk_samples.is_integer():
        parser.error("chunk-ms must produce an integral 16 kHz sample count")
    return args


def load_fixture(
    manifest_path: Path,
    fixture_id: str,
    fixture_dir: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text())
    fixtures = [
        fixture
        for fixture in manifest.get("fixtures", [])
        if fixture.get("id") == fixture_id
    ]
    if len(fixtures) != 1:
        raise SystemExit(
            f"expected one fixture named {fixture_id}, found {len(fixtures)}"
        )
    fixture = fixtures[0]
    normalized = fixture.get("normalized")
    if not normalized:
        raise SystemExit(
            "stream_probe currently requires a materialized normalized "
            "fixture such as audiobook-pilot.json"
        )
    audio_path = fixture_dir / (
        f"{fixture['language']}-{fixture['row_id']}."
        f"{normalized['extension']}"
    )
    return fixture, audio_path


def verify_audio(
    audio: np.ndarray,
    audio_path: Path,
    fixture: dict[str, Any],
) -> None:
    if audio.size == 0 or not np.isfinite(audio).all():
        raise SystemExit("audio must contain finite mono f32le samples")
    normalized = fixture["normalized"]
    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    if digest != normalized["sha256"]:
        raise SystemExit(
            f"{audio_path}: SHA-256 {digest} does not match the manifest"
        )
    if audio.size != normalized["sample_count"]:
        raise SystemExit(
            f"{audio_path}: sample count {audio.size} does not match manifest"
        )
    if not math.isclose(
        audio.size / SAMPLE_RATE_HZ * 1000,
        normalized["duration_ms"],
        abs_tol=0.001,
    ):
        raise SystemExit(f"{audio_path}: duration does not match manifest")


def probe_abandonment(
    model: Any,
    audio: np.ndarray,
    *,
    delay_ms: int,
    prefix_ms: int,
    max_decode_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    prefix_samples = min(
        audio.size,
        round(prefix_ms * SAMPLE_RATE_HZ / 1000),
    )
    session = model.create_streaming_session(
        max_tokens=max_tokens,
        temperature=0.0,
        transcription_delay_ms=delay_ms,
    )
    started = time.perf_counter_ns()
    session.feed(audio[:prefix_samples])
    deltas = session.step(max_decode_tokens=max_decode_tokens)
    generated_before_abandonment = len(session.generated)
    done_before_abandonment = session.done
    del session
    gc.collect()
    mx.clear_cache()
    return {
        "prefix_ms": prefix_samples / SAMPLE_RATE_HZ * 1000,
        "elapsed_ms": elapsed_ms(started),
        "delta_count": len(deltas),
        "generated_tokens": generated_before_abandonment,
        "done": done_before_abandonment,
        "native_cancel_available": False,
        "action": "session reference dropped without close; cache cleared",
        "claim": (
            "abandonment smoke only; this is not cooperative cancellation "
            "or proof that in-flight MLX work can be interrupted"
        ),
    }


def run_stream(
    model: Any,
    audio: np.ndarray,
    reference_text: str,
    *,
    delay_ms: int,
    repetition: int,
    chunk_ms: int,
    max_decode_tokens: int,
    max_tokens: int,
    realtime: bool,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    session = model.create_streaming_session(
        max_tokens=max_tokens,
        temperature=0.0,
        transcription_delay_ms=delay_ms,
    )
    chunk_samples = round(chunk_ms * SAMPLE_RATE_HZ / 1000)
    started = time.perf_counter_ns()
    events: list[dict[str, Any]] = []
    step_durations_ms: list[float] = []
    step_trace: list[dict[str, Any]] = []
    step_calls = 0
    feed_ready = threading.Event()
    feed_lock = threading.Lock()
    producer_state: dict[str, Any] = {
        "fed_samples": 0,
        "chunk_count": 0,
        "close_elapsed_ms": None,
        "done": False,
        "schedule_lateness_ms": [],
        "error": None,
    }

    def produce_audio() -> None:
        try:
            for offset in range(0, audio.size, chunk_samples):
                chunk = audio[offset : offset + chunk_samples]
                end_sample = offset + chunk.size
                if realtime:
                    deadline_ns = started + round(
                        end_sample / SAMPLE_RATE_HZ * 1_000_000_000
                    )
                    remaining_ns = deadline_ns - time.perf_counter_ns()
                    if remaining_ns > 0:
                        time.sleep(remaining_ns / 1_000_000_000)
                    lateness_ms = max(
                        0.0,
                        (time.perf_counter_ns() - deadline_ns) / 1_000_000,
                    )
                else:
                    lateness_ms = None
                session.feed(chunk)
                with feed_lock:
                    producer_state["fed_samples"] = end_sample
                    producer_state["chunk_count"] += 1
                    if lateness_ms is not None:
                        producer_state["schedule_lateness_ms"].append(
                            lateness_ms
                        )
                feed_ready.set()
            session.close()
            with feed_lock:
                producer_state["close_elapsed_ms"] = elapsed_ms(started)
                producer_state["done"] = True
            feed_ready.set()
        except BaseException as error:  # propagate producer failures
            with feed_lock:
                producer_state["error"] = repr(error)
                producer_state["done"] = True
            feed_ready.set()

    producer = threading.Thread(
        target=produce_audio,
        name=f"voxtral-audio-{delay_ms}-{repetition}",
        daemon=True,
    )
    producer.start()

    finalization_step_calls = 0
    empty_steps = 0
    while not session.done:
        with feed_lock:
            producer_done = producer_state["done"]
            fed_samples_before_step = producer_state["fed_samples"]
        if not producer_done:
            feed_ready.wait(timeout=0.1)
            feed_ready.clear()
            with feed_lock:
                producer_done = producer_state["done"]
                fed_samples_before_step = producer_state["fed_samples"]
        step_started_ms = elapsed_ms(started)
        deltas, step_ms = step_once(session, max_decode_tokens)
        step_calls += 1
        step_durations_ms.append(step_ms)
        with feed_lock:
            fed_samples_after_step = producer_state["fed_samples"]
            producer_done_after_step = producer_state["done"]
            producer_error = producer_state["error"]
        if producer_error is not None:
            raise RuntimeError(f"audio producer failed: {producer_error}")
        if producer_done_after_step:
            finalization_step_calls += 1
        step_trace.append(
            {
                "index": len(step_trace),
                "started_ms": step_started_ms,
                "duration_ms": step_ms,
                "audio_fed_before_step_ms": (
                    fed_samples_before_step / SAMPLE_RATE_HZ * 1000
                ),
                "audio_fed_after_step_ms": (
                    fed_samples_after_step / SAMPLE_RATE_HZ * 1000
                ),
                "producer_done_before_step": producer_done,
                "producer_done_after_step": producer_done_after_step,
                "delta_count": len(deltas),
            }
        )
        append_events(
            events,
            deltas,
            started,
            fed_samples_before_step,
            fed_samples_after_step,
        )
        if deltas:
            empty_steps = 0
        else:
            empty_steps += 1
        if producer_done_after_step and empty_steps > 10_000:
            raise RuntimeError(
                "streaming session made no finalization progress"
            )

    producer.join(timeout=5)
    if producer.is_alive():
        raise RuntimeError("audio producer did not terminate")
    final_elapsed_ms = elapsed_ms(started)
    text = "".join(event["delta"] for event in events).strip()
    if not text:
        raise RuntimeError("Voxtral Realtime streaming emitted no text")
    with feed_lock:
        close_elapsed_ms = producer_state["close_elapsed_ms"]
        audio_chunk_count = producer_state["chunk_count"]
        schedule_lateness_ms = list(
            producer_state["schedule_lateness_ms"]
        )
    first_event = events[0]
    audio_duration_ms = audio.size / SAMPLE_RATE_HZ * 1000
    steps_started_before_close = [
        step["duration_ms"]
        for step in step_trace
        if not step["producer_done_before_step"]
    ]
    steps_started_after_close = [
        step["duration_ms"]
        for step in step_trace
        if step["producer_done_before_step"]
    ]
    return {
        "transcription_delay_ms": delay_ms,
        "repetition": repetition,
        "text": text,
        "quality": quality_metrics(reference_text, text),
        "timing": {
            "first_append_ms": first_event["elapsed_ms"],
            "first_stable_ms": first_event["elapsed_ms"],
            "first_append_audio_fed_before_step_ms": (
                first_event["audio_fed_before_step_ms"]
            ),
            "first_append_audio_fed_at_emit_ms": (
                first_event["audio_fed_at_emit_ms"]
            ),
            "audio_close_ms": close_elapsed_ms,
            "final_ms": final_elapsed_ms,
            "endpoint_finalization_ms": final_elapsed_ms
            - audio_duration_ms,
            "total_step_compute_ms": sum(step_durations_ms),
            "mean_step_ms": mean(step_durations_ms),
            "p95_step_ms": percentile(step_durations_ms, 0.95),
            "maximum_step_ms": max(step_durations_ms),
            "maximum_step_started_before_close_ms": (
                max(steps_started_before_close)
                if steps_started_before_close
                else None
            ),
            "maximum_step_started_after_close_ms": (
                max(steps_started_after_close)
                if steps_started_after_close
                else None
            ),
            "mean_feed_schedule_lateness_ms": (
                mean(schedule_lateness_ms) if schedule_lateness_ms else None
            ),
            "p95_feed_schedule_lateness_ms": (
                percentile(schedule_lateness_ms, 0.95)
                if schedule_lateness_ms
                else None
            ),
            "maximum_feed_schedule_lateness_ms": (
                max(schedule_lateness_ms) if schedule_lateness_ms else None
            ),
        },
        "streaming": {
            "input_streaming_measured": True,
            "append_only": True,
            "update_count": len(events),
            "revoke_count": 0,
            "audio_chunk_count": audio_chunk_count,
            "step_call_count": step_calls,
            "finalization_step_call_count": finalization_step_calls,
            "generated_tokens": len(session.generated),
            "done": session.done,
        },
        "resources": {
            "runtime_peak_memory_bytes": mx.get_peak_memory(),
        },
        "events": events,
        "step_trace": step_trace,
    }


def step_once(
    session: Any,
    max_decode_tokens: int,
) -> tuple[list[str], float]:
    started = time.perf_counter_ns()
    deltas = session.step(max_decode_tokens=max_decode_tokens)
    return deltas, elapsed_ms(started)


def append_events(
    events: list[dict[str, Any]],
    deltas: list[str],
    started: int,
    fed_samples_before_step: int,
    fed_samples_after_step: int,
) -> None:
    for delta in deltas:
        if not delta:
            continue
        events.append(
            {
                "index": len(events),
                "elapsed_ms": elapsed_ms(started),
                "audio_fed_before_step_ms": (
                    fed_samples_before_step / SAMPLE_RATE_HZ * 1000
                ),
                "audio_fed_at_emit_ms": (
                    fed_samples_after_step / SAMPLE_RATE_HZ * 1000
                ),
                "delta": delta,
            }
        )


def quality_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    reference_words = words(reference)
    hypothesis_words = words(hypothesis)
    return {
        "wer": error_rate(reference_words, hypothesis_words),
        "cer": error_rate(
            list("".join(reference_words)),
            list("".join(hypothesis_words)),
        ),
    }


def words(text: str) -> list[str]:
    normalized = "".join(
        character
        if character.isspace()
        or unicodedata.category(character)[0] in {"L", "N"}
        else ""
        for character in text.lower()
    )
    return normalized.split()


def error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(
            hypothesis,
            start=1,
        ):
            substitution = previous[hypothesis_index - 1] + (
                reference_item != hypothesis_item
            )
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1] / len(reference)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * fraction) - 1),
    )
    return ordered[index]


def elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1_000_000


def utc_now() -> str:
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        + f".{time.time_ns() % 1_000_000_000 // 1_000_000:03d}Z"
    )


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


if __name__ == "__main__":
    main()
