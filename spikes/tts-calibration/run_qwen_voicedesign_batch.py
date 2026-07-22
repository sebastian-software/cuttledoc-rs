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


CHUNKING_REVISION = "sentence-aware-word-bounded-v1"
CHUNK_TARGET_WORDS = 45
CHUNK_MAX_WORDS = 55
INTER_CHUNK_SILENCE_MS = 250
TERMINAL_WORD = re.compile(r"[.!?…]+[\"'”’»›)\]]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Qwen3-TTS VoiceDesign jobs with one model load."
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


def valid_checkpoint(directory: Path, job: dict, source_revision: str) -> bool:
    try:
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        audio_path = directory / result["audio"]["path"]
        return (
            result["job_id"] == job["id"]
            and result["source_revision"] == source_revision
            and result["input"]["text_sha256"] == job["text_sha256"]
            and result["generation"]["instruction"] == job["instruction"]
            and result["generation"]["seed"] == job["seed"]
            and result["generation"]["chunking"]["revision"]
            == CHUNKING_REVISION
            and result["termination"]["completed_all_chunks"] is True
            and result["termination"]["reached_max_tokens"] is False
            and audio_path.stat().st_size == result["audio"]["byte_count"]
            and sha256_file(audio_path) == result["audio"]["sha256"]
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def validate_job(job: dict) -> str:
    required = {
        "id",
        "locale",
        "language",
        "instruction",
        "seed",
        "text_path",
        "text_sha256",
    }
    if set(job) != required:
        raise ValueError(f"{job.get('id', '<unknown>')}: invalid job fields")
    if not isinstance(job["seed"], int) or job["seed"] < 0:
        raise ValueError(f"{job['id']}: seed must be a non-negative integer")
    if not job["instruction"].strip():
        raise ValueError(f"{job['id']}: instruction must not be empty")
    text = Path(job["text_path"]).read_text(encoding="utf-8").strip()
    if sha256_bytes(text.encode("utf-8")) != job["text_sha256"]:
        raise ValueError(f"{job['id']}: text digest drift")
    return text


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


def generate_chunk(model, job: dict, text: str, generation: dict) -> tuple:
    # Reusing the identity seed for every chunk stabilizes the designed voice
    # while keeping the complete document exactly reproducible.
    np.random.seed(job["seed"])
    mx.random.seed(job["seed"])
    generator = model.generate_voice_design(
        text=text,
        instruct=job["instruction"],
        language=job["language"],
        temperature=generation["temperature"],
        max_tokens=generation["max_tokens"],
        top_k=generation["top_k"],
        top_p=generation["top_p"],
        repetition_penalty=generation["repetition_penalty"],
        verbose=False,
        stream=generation["stream"],
    )
    arrays: list[np.ndarray] = []
    outputs: list[dict] = []
    first_audio_ns: int | None = None
    started = time.perf_counter_ns()
    for output in generator:
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
                "is_final_chunk": bool(output.is_final_chunk),
            }
        )
    completed = time.perf_counter_ns()
    if not arrays or first_audio_ns is None:
        raise RuntimeError(f"{job['id']}: Qwen3-TTS returned no audio")
    token_count = sum(item["token_count"] for item in outputs)
    # mlx-audio 0.4.5 reports is_final_chunk=false for both capped and normally
    # completed non-streaming calls, so the pinned token ceiling is authoritative.
    reached_max_tokens = token_count >= generation["max_tokens"]
    if reached_max_tokens:
        raise RuntimeError(
            f"{job['id']}: chunk reached the {generation['max_tokens']} token limit"
        )
    audio = np.concatenate(arrays).astype("<f4", copy=False)
    return audio, outputs, first_audio_ns, started, completed, token_count


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
    text_chunks = split_text_chunks(text)
    audio_parts: list[np.ndarray] = []
    chunk_results: list[dict] = []
    first_audio_ns: int | None = None
    silence_samples = round(sample_rate * INTER_CHUNK_SILENCE_MS / 1000)
    for index, chunk_text in enumerate(text_chunks):
        (
            chunk_audio,
            outputs,
            chunk_first_audio_ns,
            chunk_started,
            chunk_completed,
            token_count,
        ) = generate_chunk(model, job, chunk_text, generation)
        if first_audio_ns is None:
            first_audio_ns = chunk_first_audio_ns
        if index > 0:
            audio_parts.append(np.zeros(silence_samples, dtype="<f4"))
        audio_parts.append(chunk_audio)
        chunk_results.append(
            {
                "index": index,
                "text_sha256": sha256_bytes(chunk_text.encode("utf-8")),
                "character_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "seed": job["seed"],
                "audio_sample_count": int(chunk_audio.size),
                "duration_ms": int(chunk_audio.size) * 1000.0 / sample_rate,
                "token_count": token_count,
                "synthesis_ms": (chunk_completed - chunk_started) / 1_000_000,
                "outputs": outputs,
                "completed": True,
            }
        )
    completed = time.perf_counter_ns()
    if not audio_parts or first_audio_ns is None:
        raise RuntimeError(f"{job['id']}: Qwen3-TTS returned no audio")
    audio = np.concatenate(audio_parts).astype("<f4", copy=False)
    if not np.isfinite(audio).all():
        raise RuntimeError(f"{job['id']}: Qwen3-TTS returned non-finite audio")
    raw = audio.tobytes()
    token_count = sum(item["token_count"] for item in chunk_results)
    duration_ms = int(audio.size) * 1000.0 / sample_rate
    temporary = Path(tempfile.mkdtemp(prefix=f".{job['id']}-", dir=output_root))
    try:
        (temporary / "audio.f32le").write_bytes(raw)
        result = {
            "schema_version": "1.0.0",
            "job_id": job["id"],
            "captured_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "source_revision": source_revision,
            "input": {
                "locale": job["locale"],
                "text_sha256": job["text_sha256"],
                "character_count": len(text),
            },
            "generation": {
                "language": job["language"],
                "instruction": job["instruction"],
                "seed": job["seed"],
                **generation,
                "chunking": {
                    "revision": CHUNKING_REVISION,
                    "target_words": CHUNK_TARGET_WORDS,
                    "maximum_words": CHUNK_MAX_WORDS,
                    "inter_chunk_silence_ms": INTER_CHUNK_SILENCE_MS,
                    "seed_policy": "reset-identity-seed-per-chunk",
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
                "token_count": token_count,
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                "chunks": chunk_results,
                "packages": {
                    "mlx": importlib.metadata.version("mlx"),
                    "mlx-audio": importlib.metadata.version("mlx-audio"),
                },
            },
            "termination": {
                "completed_all_chunks": True,
                "reached_max_tokens": False,
                "chunk_count": len(chunk_results),
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
    raw = np.array([-0.5, 0.5], dtype="<f4").tobytes()
    if sha256_bytes(raw) != hashlib.sha256(raw).hexdigest():
        raise RuntimeError("self-test digest failure")
    short = "One short sentence stays intact."
    if split_text_chunks(short) != [short]:
        raise RuntimeError("self-test short chunk failure")
    source = " ".join(
        f"Sentence {index} has enough words for a boundary."
        for index in range(20)
    )
    chunks = split_text_chunks(source)
    if len(chunks) < 2 or " ".join(chunks) != source:
        raise RuntimeError("self-test chunk preservation failure")
    if any(len(chunk.split()) > CHUNK_MAX_WORDS for chunk in chunks):
        raise RuntimeError("self-test chunk limit failure")
    print("Qwen VoiceDesign batch runner: self-test passed")


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
        raise ValueError("manifest, model-dir, jobs, output-dir, and source-revision are required")
    if not (
        len(args.source_revision) == 40
        and all(character in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("--source-revision must be a lowercase 40-character Git SHA")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs_document = json.loads(args.jobs.read_text(encoding="utf-8"))
    if manifest["id"] != "qwen3-tts-12hz-1.7b-voicedesign-mlx-bf16":
        raise ValueError("runner accepts only the pinned Qwen VoiceDesign manifest")
    if jobs_document.get("schema_version") != "1.0.0":
        raise ValueError("unsupported jobs schema")
    jobs = jobs_document.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list")
    if len({job.get("id") for job in jobs}) != len(jobs):
        raise ValueError("job ids must be unique")
    texts = {job["id"]: validate_job(job) for job in jobs}
    verify_model(manifest, args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter_ns()
    model = load_model(args.model_dir)
    model_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    sample_rate = manifest["calibration"]["sample_rate_hz"]
    if model.config.tts_model_type != "voice_design":
        raise ValueError(f"unexpected model type: {model.config.tts_model_type}")
    if model.sample_rate != sample_rate:
        raise ValueError(f"sample-rate drift: expected {sample_rate}, got {model.sample_rate}")
    supported = {language.lower() for language in model.get_supported_languages()}
    generation = manifest["calibration"]["generation"]
    results = []
    for index, job in enumerate(jobs, start=1):
        if job["language"].lower() not in supported:
            raise ValueError(f"{job['id']}: unsupported language {job['language']}")
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
            f"qwen tts: {job['id']} ({index}/{len(jobs)}, {outcome['status']})",
            flush=True,
        )
    print(json.dumps({"jobs": results}, separators=(",", ":")))


if __name__ == "__main__":
    main()
