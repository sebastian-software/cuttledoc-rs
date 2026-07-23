#!/usr/bin/env python3

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CHUNKING_REVISION = "sentence-aware-adaptive-v1"
CHUNK_TARGET_WORDS = 45
CHUNK_MAX_WORDS = 55
INTER_CHUNK_SILENCE_MS = 250
ADAPTIVE_MIN_SPLIT_WORDS = 4
ADAPTIVE_MAX_DEPTH = 4
TERMINAL_WORD = re.compile(r"[.!?…]+[\"'”’»›)\]]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Voxtral TTS jobs with one pinned BF16 model load."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(manifest: dict, model_dir: Path) -> None:
    snapshot_bytes = 0
    for artifact in manifest["artifact"]["artifacts"]:
        path = model_dir / artifact["path"]
        if not path.is_file():
            raise ValueError(f"Missing model artifact: {path}")
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Model artifact byte count drift: {path}")
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise ValueError(
                f"Model artifact SHA-256 drift: {path}; "
                f"expected {artifact['sha256']}, got {actual}"
            )
        snapshot_bytes += path.stat().st_size
    if snapshot_bytes != manifest["artifact"]["snapshot_bytes"]:
        raise ValueError("Model snapshot byte count differs from the manifest")


def validate_job(job: dict, model_dir: Path) -> str:
    required = {
        "id",
        "locale",
        "voice",
        "seed",
        "text_path",
        "text_sha256",
    }
    if set(job) != required:
        raise ValueError(f"{job.get('id', '<unknown>')}: invalid job fields")
    if not isinstance(job["seed"], int) or job["seed"] < 0:
        raise ValueError(f"{job['id']}: seed must be a non-negative integer")
    voice_path = model_dir / "voice_embedding" / f"{job['voice']}.safetensors"
    if not voice_path.is_file():
        raise ValueError(f"{job['id']}: missing voice embedding {job['voice']}")
    text = Path(job["text_path"]).read_text(encoding="utf-8").strip()
    if sha256_bytes(text.encode("utf-8")) != job["text_sha256"]:
        raise ValueError(f"{job['id']}: text digest drift")
    return text


def valid_checkpoint(directory: Path, job: dict, source_revision: str) -> bool:
    try:
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        audio_path = directory / result["audio"]["path"]
        return (
            result["job_id"] == job["id"]
            and result["source_revision"] == source_revision
            and result["input"]["text_sha256"] == job["text_sha256"]
            and result["generation"]["voice"] == job["voice"]
            and result["generation"]["seed"] == job["seed"]
            and result["generation"]["chunking"]["revision"] == CHUNKING_REVISION
            and result["termination"]["completed_all_chunks"] is True
            and result["termination"]["reached_max_tokens"] is False
            and audio_path.stat().st_size == result["audio"]["byte_count"]
            and sha256_file(audio_path) == result["audio"]["sha256"]
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def split_text_chunks(text: str) -> list[str]:
    words = text.split()
    if not words:
        raise ValueError("text must not be empty")
    sentences: list[list[str]] = []
    sentence: list[str] = []
    for word in words:
        sentence.append(word)
        if TERMINAL_WORD.search(word) is not None:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        while len(sentence) > CHUNK_MAX_WORDS:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(" ".join(sentence[:CHUNK_MAX_WORDS]))
            sentence = sentence[CHUNK_MAX_WORDS:]
        if current and len(current) + len(sentence) > CHUNK_MAX_WORDS:
            chunks.append(" ".join(current))
            current = []
        current.extend(sentence)
        if len(current) >= CHUNK_TARGET_WORDS:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    if any(len(chunk.split()) > CHUNK_MAX_WORDS for chunk in chunks):
        raise RuntimeError("chunk word limit was not enforced")
    if " ".join(chunks) != " ".join(words):
        raise RuntimeError("chunking changed normalized input text")
    return chunks


def split_chunk_for_retry(text: str) -> tuple[str, str]:
    words = text.split()
    if len(words) < ADAPTIVE_MIN_SPLIT_WORDS * 2:
        raise RuntimeError("chunk is too short for another deterministic split")
    midpoint = len(words) // 2
    boundaries = [
        index + 1
        for index, word in enumerate(words[:-1])
        if ADAPTIVE_MIN_SPLIT_WORDS <= index + 1
        and len(words) - (index + 1) >= ADAPTIVE_MIN_SPLIT_WORDS
        and TERMINAL_WORD.search(word) is not None
    ]
    split_at = (
        min(boundaries, key=lambda value: abs(value - midpoint))
        if boundaries
        else midpoint
    )
    split_at = max(
        ADAPTIVE_MIN_SPLIT_WORDS,
        min(split_at, len(words) - ADAPTIVE_MIN_SPLIT_WORDS),
    )
    left = " ".join(words[:split_at])
    right = " ".join(words[split_at:])
    if f"{left} {right}" != " ".join(words):
        raise RuntimeError("adaptive split changed normalized input text")
    return left, right


def generate_chunk(model, job: dict, text: str, generation: dict) -> tuple:
    np.random.seed(job["seed"])
    mx.random.seed(job["seed"])
    started = time.perf_counter_ns()
    first_audio_ns: int | None = None
    arrays: list[np.ndarray] = []
    outputs: list[dict] = []
    for output in model.generate(
        text=text,
        voice=job["voice"],
        temperature=generation["temperature"],
        max_tokens=generation["max_tokens"],
        top_k=generation["top_k"],
        top_p=generation["top_p"],
        verbose=False,
        stream=generation["stream"],
    ):
        if first_audio_ns is None:
            first_audio_ns = time.perf_counter_ns()
        arrays.append(np.asarray(output.audio, dtype="<f4").reshape(-1).copy())
        outputs.append(
            {
                "sample_count": int(output.samples),
                "sample_rate_hz": int(output.sample_rate),
                "token_count": int(output.token_count),
                "processing_time_seconds": float(output.processing_time_seconds),
                "peak_memory_gb_decimal": float(output.peak_memory_usage),
                "is_streaming_chunk": bool(output.is_streaming_chunk),
                "is_final_chunk": bool(output.is_final_chunk),
            }
        )
    completed = time.perf_counter_ns()
    if not arrays or first_audio_ns is None:
        raise RuntimeError(f"{job['id']}: Voxtral TTS returned no audio")
    audio = np.concatenate(arrays).astype("<f4", copy=False)
    token_count = sum(item["token_count"] for item in outputs)
    return (
        audio,
        outputs,
        first_audio_ns,
        started,
        completed,
        token_count,
        token_count >= generation["max_tokens"],
    )


def generate_chunk_adaptively(
    model,
    job: dict,
    text: str,
    generation: dict,
    path: str,
    depth: int,
    attempts: list[dict],
) -> list[dict]:
    (
        audio,
        outputs,
        first_audio_ns,
        started,
        completed,
        token_count,
        reached_max_tokens,
    ) = generate_chunk(model, job, text, generation)
    attempts.append(
        {
            "path": path,
            "depth": depth,
            "text_sha256": sha256_bytes(text.encode("utf-8")),
            "word_count": len(text.split()),
            "token_count": token_count,
            "synthesis_ms": (completed - started) / 1_000_000,
            "reached_max_tokens": reached_max_tokens,
        }
    )
    if not reached_max_tokens:
        return [
            {
                "path": path,
                "depth": depth,
                "text": text,
                "audio": audio,
                "outputs": outputs,
                "first_audio_ns": first_audio_ns,
                "started": started,
                "completed": completed,
                "token_count": token_count,
            }
        ]
    if depth >= ADAPTIVE_MAX_DEPTH:
        raise RuntimeError(
            f"{job['id']}: chunk {path} still reached the token limit "
            f"at adaptive depth {depth}"
        )
    try:
        left, right = split_chunk_for_retry(text)
    except RuntimeError as error:
        raise RuntimeError(
            f"{job['id']}: chunk {path} reached the token limit and "
            "cannot be split further"
        ) from error
    return [
        *generate_chunk_adaptively(
            model, job, left, generation, f"{path}.0", depth + 1, attempts
        ),
        *generate_chunk_adaptively(
            model, job, right, generation, f"{path}.1", depth + 1, attempts
        ),
    ]


def generate_job(
    model,
    job: dict,
    text: str,
    generation: dict,
    sample_rate: int,
    model_load_ms: float,
    source_revision: str,
    output_root: Path,
) -> dict:
    final_directory = output_root / job["id"]
    if valid_checkpoint(final_directory, job, source_revision):
        return {"id": job["id"], "status": "resumed"}
    shutil.rmtree(final_directory, ignore_errors=True)

    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    attempts: list[dict] = []
    completed_chunks: list[dict] = []
    for initial_index, chunk_text in enumerate(split_text_chunks(text)):
        completed_chunks.extend(
            generate_chunk_adaptively(
                model,
                job,
                chunk_text,
                generation,
                str(initial_index),
                0,
                attempts,
            )
        )
    if " ".join(chunk["text"] for chunk in completed_chunks) != " ".join(
        text.split()
    ):
        raise RuntimeError(f"{job['id']}: adaptive generation changed input text")

    audio_parts: list[np.ndarray] = []
    chunk_results: list[dict] = []
    first_audio_ns: int | None = None
    silence_samples = round(sample_rate * INTER_CHUNK_SILENCE_MS / 1000)
    for index, chunk in enumerate(completed_chunks):
        if first_audio_ns is None or chunk["first_audio_ns"] < first_audio_ns:
            first_audio_ns = chunk["first_audio_ns"]
        if index > 0:
            audio_parts.append(np.zeros(silence_samples, dtype="<f4"))
        audio_parts.append(chunk["audio"])
        chunk_results.append(
            {
                "index": index,
                "path": chunk["path"],
                "split_depth": chunk["depth"],
                "text_sha256": sha256_bytes(chunk["text"].encode("utf-8")),
                "character_count": len(chunk["text"]),
                "word_count": len(chunk["text"].split()),
                "seed": job["seed"],
                "audio_sample_count": int(chunk["audio"].size),
                "duration_ms": int(chunk["audio"].size) * 1000.0 / sample_rate,
                "token_count": chunk["token_count"],
                "synthesis_ms": (
                    chunk["completed"] - chunk["started"]
                )
                / 1_000_000,
                "outputs": chunk["outputs"],
                "completed": True,
            }
        )
    completed = time.perf_counter_ns()
    if not audio_parts or first_audio_ns is None:
        raise RuntimeError(f"{job['id']}: Voxtral TTS returned no audio")
    audio = np.concatenate(audio_parts).astype("<f4", copy=False)
    if not np.isfinite(audio).all():
        raise RuntimeError(f"{job['id']}: Voxtral TTS returned non-finite audio")
    raw = audio.tobytes()
    duration_ms = int(audio.size) * 1000.0 / sample_rate
    emitted_token_count = sum(item["token_count"] for item in chunk_results)
    attempted_token_count = sum(item["token_count"] for item in attempts)

    temporary = Path(tempfile.mkdtemp(prefix=f".{job['id']}-", dir=output_root))
    try:
        (temporary / "audio.f32le").write_bytes(raw)
        result = {
            "schema_version": "1.0.0",
            "job_id": job["id"],
            "captured_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "source_revision": source_revision,
            "input": {
                "locale": job["locale"],
                "text_sha256": job["text_sha256"],
                "character_count": len(text),
            },
            "generation": {
                "voice": job["voice"],
                "seed": job["seed"],
                **generation,
                "chunking": {
                    "revision": CHUNKING_REVISION,
                    "target_words": CHUNK_TARGET_WORDS,
                    "maximum_words": CHUNK_MAX_WORDS,
                    "inter_chunk_silence_ms": INTER_CHUNK_SILENCE_MS,
                    "seed_policy": "reset-generation-seed-per-chunk",
                    "adaptive_minimum_split_words": ADAPTIVE_MIN_SPLIT_WORDS,
                    "adaptive_maximum_depth": ADAPTIVE_MAX_DEPTH,
                },
            },
            "audio": {
                "path": "audio.f32le",
                "sample_format": "f32le",
                "sample_rate_hz": sample_rate,
                "channel_count": 1,
                "sample_count": int(audio.size),
                "byte_count": len(raw),
                "duration_ms": duration_ms,
                "sha256": sha256_bytes(raw),
                "minimum_sample": float(audio.min()),
                "maximum_sample": float(audio.max()),
                "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
                "non_finite_sample_count": 0,
            },
            "runtime": {
                "model_load_ms": model_load_ms,
                "first_audio_ms": (first_audio_ns - started) / 1_000_000,
                "complete_synthesis_ms": (completed - started) / 1_000_000,
                "real_time_factor": (completed - started) / 1_000_000 / duration_ms,
                "token_count": emitted_token_count,
                "attempted_token_count": attempted_token_count,
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                "chunks": chunk_results,
                "attempts": attempts,
                "packages": {
                    "mistral-common": importlib.metadata.version("mistral-common"),
                    "mlx": importlib.metadata.version("mlx"),
                    "mlx-audio": importlib.metadata.version("mlx-audio"),
                },
            },
            "termination": {
                "completed_all_chunks": True,
                "reached_max_tokens": False,
                "chunk_count": len(chunk_results),
                "adaptive_retry_count": sum(
                    attempt["reached_max_tokens"] for attempt in attempts
                ),
                "maximum_split_depth": max(
                    chunk["split_depth"] for chunk in chunk_results
                ),
                "configured_max_tokens_per_chunk": generation["max_tokens"],
            },
        }
        (temporary / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, final_directory)
        return {"id": job["id"], "status": "generated"}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def self_test() -> None:
    short = "One short sentence stays intact."
    if split_text_chunks(short) != [short]:
        raise RuntimeError("self-test short chunk failure")
    source = " ".join(
        f"Sentence {index} has enough words for a boundary." for index in range(20)
    )
    chunks = split_text_chunks(source)
    if len(chunks) < 2 or " ".join(chunks) != source:
        raise RuntimeError("self-test chunk preservation failure")
    if any(len(chunk.split()) > CHUNK_MAX_WORDS for chunk in chunks):
        raise RuntimeError("self-test chunk limit failure")
    retry_source = " ".join(f"word-{index}" for index in range(20))
    left, right = split_chunk_for_retry(retry_source)
    if f"{left} {right}" != retry_source:
        raise RuntimeError("self-test adaptive split preservation failure")
    print("Voxtral TTS batch runner: self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    global mx, load_model
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    if not all(
        [args.manifest, args.model_dir, args.jobs, args.output_dir, args.source_revision]
    ):
        raise ValueError(
            "manifest, model-dir, jobs, output-dir, and source-revision are required"
        )
    if not (
        len(args.source_revision) == 40
        and all(character in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("--source-revision must be a lowercase 40-character Git SHA")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs_document = json.loads(args.jobs.read_text(encoding="utf-8"))
    if manifest["id"] != "voxtral-4b-tts-2603-mlx-bf16":
        raise ValueError("runner accepts only the pinned Voxtral BF16 manifest")
    if jobs_document.get("schema_version") != "1.0.0":
        raise ValueError("unsupported jobs schema")
    jobs = jobs_document.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list")
    if len({job.get("id") for job in jobs}) != len(jobs):
        raise ValueError("job ids must be unique")

    texts = {
        job["id"]: validate_job(job, args.model_dir)
        for job in jobs
    }
    verify_model(manifest, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter_ns()
    model = load_model(args.model_dir)
    model_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    sample_rate = manifest["calibration"]["sample_rate_hz"]
    if model.model_type != "voxtral_tts":
        raise ValueError(f"unexpected model type: {model.model_type}")
    if model.sample_rate != sample_rate:
        raise ValueError(
            f"sample-rate drift: expected {sample_rate}, got {model.sample_rate}"
        )

    generation = manifest["calibration"]["generation"]
    results = []
    for index, job in enumerate(jobs, start=1):
        outcome = generate_job(
            model,
            job,
            texts[job["id"]],
            generation,
            sample_rate,
            model_load_ms,
            args.source_revision,
            args.output_dir,
        )
        results.append(outcome)
        print(
            f"voxtral tts: {job['id']} ({index}/{len(jobs)}, {outcome['status']})",
            flush=True,
        )
    print(json.dumps({"jobs": results}, separators=(",", ":")))


if __name__ == "__main__":
    main()
