#!/usr/bin/env python3

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one pinned Qwen3-TTS VoiceDesign calibration profile."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def selected_profile(manifest: dict, profile_id: str) -> dict:
    matches = [
        profile
        for profile in manifest["calibration"]["profiles"]
        if profile["id"] == profile_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate calibration profile: {profile_id}")
    profile = matches[0]
    if manifest["calibration"]["voice_mode"] != "description":
        raise ValueError("Qwen VoiceDesign requires a description profile")
    if profile["voice"] is not None or not profile["instruction"]:
        raise ValueError("VoiceDesign profile must contain only an instruction")
    return profile


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


def verify_text(selection: dict, profile: dict, text_file: Path) -> tuple[str, dict, dict]:
    matches = [
        (source, passage)
        for source in selection["sources"]
        for passage in source["passages"]
        if passage["id"] == profile["passage_id"]
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate passage: {profile['passage_id']}")
    source, passage = matches[0]
    if source["locale"] != profile["locale"]:
        raise ValueError("Profile locale differs from selected passage")
    text = text_file.read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != passage["spoken_sha256"] or len(text) != passage["character_count"]:
        raise ValueError("Materialized passage differs from the pinned selection")
    return text, source, passage


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm16 = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def main() -> None:
    args = parse_args()
    if not (
        len(args.source_revision) == 40
        and all(character in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("--source-revision must be a lowercase 40-character Git SHA")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if manifest["id"] != "qwen3-tts-12hz-1.7b-voicedesign-mlx-bf16":
        raise ValueError("Runner accepts only the pinned Qwen VoiceDesign manifest")
    profile = selected_profile(manifest, args.profile)
    verify_model(manifest, args.model_dir)
    text, source, passage = verify_text(selection, profile, args.text_file)
    generation = manifest["calibration"]["generation"]
    sample_rate = manifest["calibration"]["sample_rate_hz"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(profile["seed"])
    mx.random.seed(profile["seed"])
    mx.reset_peak_memory()

    load_started = time.perf_counter_ns()
    model = load_model(args.model_dir)
    load_completed = time.perf_counter_ns()
    if model.config.tts_model_type != "voice_design":
        raise ValueError(f"Unexpected model type: {model.config.tts_model_type}")
    if model.sample_rate != sample_rate:
        raise ValueError(f"Sample-rate drift: expected {sample_rate}, got {model.sample_rate}")
    if profile["language"].lower() not in {
        language.lower() for language in model.get_supported_languages()
    }:
        raise ValueError(f"Unsupported language: {profile['language']}")

    generation_started = time.perf_counter_ns()
    generator = model.generate_voice_design(
        text=text,
        instruct=profile["instruction"],
        language=profile["language"],
        temperature=generation["temperature"],
        max_tokens=generation["max_tokens"],
        top_k=generation["top_k"],
        top_p=generation["top_p"],
        repetition_penalty=generation["repetition_penalty"],
        verbose=False,
        stream=generation["stream"],
    )

    arrays: list[np.ndarray] = []
    output_metrics: list[dict] = []
    first_audio_ns: int | None = None
    for output in generator:
        if first_audio_ns is None:
            first_audio_ns = time.perf_counter_ns()
        array = np.asarray(output.audio, dtype="<f4").reshape(-1).copy()
        arrays.append(array)
        output_metrics.append(
            {
                "sample_count": int(output.samples),
                "sample_rate_hz": int(output.sample_rate),
                "token_count": int(output.token_count),
                "processing_time_seconds": float(output.processing_time_seconds),
                "mlx_audio_reported_duration_over_processing": float(
                    output.real_time_factor
                ),
                "peak_memory_gb_decimal": float(output.peak_memory_usage),
                "is_streaming_chunk": bool(output.is_streaming_chunk),
                "is_final_chunk": bool(output.is_final_chunk),
            }
        )
    generation_completed = time.perf_counter_ns()

    if not arrays or first_audio_ns is None:
        raise RuntimeError("Qwen3-TTS returned no audio")
    audio = np.concatenate(arrays).astype("<f4", copy=False)
    if not np.isfinite(audio).all():
        raise RuntimeError("Qwen3-TTS returned non-finite audio")

    raw_audio = audio.tobytes()
    raw_path = args.output_dir / "audio.f32le"
    wav_path = args.output_dir / "audio.pcm16.wav"
    raw_path.write_bytes(raw_audio)
    write_pcm16_wav(wav_path, audio, sample_rate)

    sample_count = int(audio.size)
    duration_ms = sample_count * 1000.0 / sample_rate
    complete_ms = (generation_completed - generation_started) / 1_000_000
    first_audio_ms = (first_audio_ns - generation_started) / 1_000_000
    token_count = sum(item["token_count"] for item in output_metrics)
    reached_limit = token_count >= generation["max_tokens"]
    generation_contract = {
        "profile_id": profile["id"],
        "passage_id": profile["passage_id"],
        "locale": profile["locale"],
        "language": profile["language"],
        "voice": profile["voice"],
        "instruction": profile["instruction"],
        "seed": profile["seed"],
        **generation,
        "sample_rate_hz": sample_rate,
    }

    result = {
        "schema_version": "1.0.0",
        "run_id": f"phase5.qwen3-tts-1.7b-voicedesign.{profile['id']}.1",
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_revision": args.source_revision,
        "selection_revision": selection["revision"],
        "purpose": "calibration",
        "candidate": {
            "id": "qwen3-tts-1.7b-voicedesign-mlx-audio",
            "task": "tts",
            "model": {
                "repository": manifest["artifact"]["repository"],
                "revision": manifest["artifact"]["revision"],
                "license": manifest["artifact"]["license"],
                "snapshot_bytes": manifest["artifact"]["snapshot_bytes"],
            },
            "runtime": {
                "repository": manifest["reference_runtime"]["repository"],
                "revision": manifest["reference_runtime"]["revision"],
                "version": package_version("mlx-audio"),
                "boundary": manifest["reference_runtime"]["boundary"],
                "license": manifest["reference_runtime"]["license"],
            },
            "voice": {
                "mode": manifest["calibration"]["voice_mode"],
                "profile_id": profile["id"],
                "instruction": profile["instruction"],
                "language": profile["language"],
                "locale": profile["locale"],
                "seed": profile["seed"],
            },
        },
        "host": {
            "id": "mac-studio-m1-ultra-local",
            "chip": command_output("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": int(command_output("sysctl", "-n", "hw.memsize")),
            "os": f"macOS {command_output('sw_vers', '-productVersion')} "
            f"({command_output('sw_vers', '-buildVersion')})",
            "architecture": platform.machine(),
            "execution_context": "normal host process outside the restricted diagnostic sandbox",
        },
        "input": {
            "passage_id": passage["id"],
            "source_id": source["id"],
            "locale": source["locale"],
            "character_count": len(text),
            "text_sha256": passage["spoken_sha256"],
            "license": source["license"],
        },
        "procedure": {
            "repetitions": 1,
            "generation": generation_contract,
            "stream": False,
            "cold_load": "Fresh Python process; OS file and Metal caches were not cleared.",
            "command": "bash scripts/run-qwen3-tts-voicedesign-calibration.sh",
            "model_verification": "Every artifact verified by byte count and SHA-256 before load.",
            "raw_audio": "local-required; not checked into Git",
            "raw_artifacts": [raw_path.name, wav_path.name, "result.json"],
        },
        "result": {
            "status": "partial" if reached_limit else "measured",
            "audio": {
                "sample_format": "f32le",
                "sample_rate_hz": sample_rate,
                "channel_count": 1,
                "sample_count": sample_count,
                "byte_count": len(raw_audio),
                "duration_ms": duration_ms,
                "sha256": hashlib.sha256(raw_audio).hexdigest(),
                "minimum_sample": float(audio.min()),
                "maximum_sample": float(audio.max()),
                "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
                "non_finite_sample_count": 0,
            },
            "timing": {
                "model_load_ms": (load_completed - load_started) / 1_000_000,
                "first_audio_ms": first_audio_ms,
                "complete_synthesis_ms": complete_ms,
                "real_time_factor": complete_ms / duration_ms,
                "output_count": len(output_metrics),
                "token_count": token_count,
            },
            "resources": {
                "maximum_resident_set_size_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                "model_snapshot_bytes": manifest["artifact"]["snapshot_bytes"],
            },
            "runtime_output": output_metrics,
            "termination": {
                "reached_max_tokens": reached_limit,
                "configured_max_tokens": generation["max_tokens"],
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "packages": {
                "mlx": package_version("mlx"),
                "mlx-audio": package_version("mlx-audio"),
                "mlx-lm": package_version("mlx-lm"),
                "numpy": package_version("numpy"),
                "transformers": package_version("transformers"),
            },
            "supported_speakers": model.get_supported_speakers(),
            "supported_languages": model.get_supported_languages(),
        },
        "conclusion": {
            "reference_path_proven": not reached_limit,
            "public_contract_accepted": False,
            "quality_decision_ready": False,
            "next": "Normalize the audio to 16 kHz and run the five required ASR backends.",
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
