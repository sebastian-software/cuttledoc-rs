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


PASSAGE_SHA256 = (
    "e21020c7d9aa51a52a023b9e9c3f153c21404ace1d8e9b67d69c0ff1fc0d553e"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned Voxtral TTS MLX reference diagnostic."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(
    manifest: dict, model_dir: Path, text_file: Path
) -> tuple[str, str]:
    snapshot_bytes = 0
    for artifact in manifest["artifacts"]:
        path = model_dir / artifact["path"]
        if not path.is_file():
            raise ValueError(f"Missing model artifact: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != artifact["bytes"]:
            raise ValueError(f"Model artifact byte count drift: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Model artifact SHA-256 drift: {path}")
        snapshot_bytes += actual_bytes
    if snapshot_bytes != manifest["conversion"]["snapshot_bytes"]:
        raise ValueError(
            "Model snapshot byte count differs from the pinned manifest"
        )

    text = text_file.read_text(encoding="utf-8").strip()
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_sha256 != PASSAGE_SHA256:
        raise ValueError(
            f"Text SHA-256 drift: expected {PASSAGE_SHA256}, got {text_sha256}"
        )
    return text, text_sha256


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype("<i2")
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
    text, text_sha256 = verify_inputs(manifest, args.model_dir, args.text_file)
    contract = manifest["generation_contract"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    voice_path = (
        args.model_dir
        / "voice_embedding"
        / f"{contract['speaker']}.safetensors"
    )
    if not voice_path.is_file():
        raise ValueError(f"Pinned voice embedding is missing: {voice_path}")

    np.random.seed(contract["seed"])
    mx.random.seed(contract["seed"])
    mx.reset_peak_memory()

    load_started = time.perf_counter_ns()
    model = load_model(args.model_dir)
    load_completed = time.perf_counter_ns()
    if model.sample_rate != contract["sample_rate_hz"]:
        raise ValueError(
            f"Sample-rate drift: expected {contract['sample_rate_hz']}, "
            f"got {model.sample_rate}"
        )

    generation_started = time.perf_counter_ns()
    generator = model.generate(
        text=text,
        voice=contract["speaker"],
        temperature=contract["temperature"],
        max_tokens=contract["max_tokens"],
        top_k=contract["top_k"],
        top_p=contract["top_p"],
        verbose=False,
        stream=False,
    )

    arrays: list[np.ndarray] = []
    output_metrics: list[dict] = []
    first_audio_ns: int | None = None
    for generated in generator:
        if first_audio_ns is None:
            first_audio_ns = time.perf_counter_ns()
        array = np.asarray(generated.audio, dtype="<f4").reshape(-1).copy()
        arrays.append(array)
        output_metrics.append(
            {
                "sample_count": int(generated.samples),
                "sample_rate_hz": int(generated.sample_rate),
                "token_count": int(generated.token_count),
                "processing_time_seconds": float(
                    generated.processing_time_seconds
                ),
                "mlx_audio_reported_duration_over_processing": float(
                    generated.real_time_factor
                ),
                "peak_memory_gb_decimal": float(generated.peak_memory_usage),
                "is_streaming_chunk": bool(generated.is_streaming_chunk),
                "is_final_chunk": bool(generated.is_final_chunk),
            }
        )
    generation_completed = time.perf_counter_ns()

    if not arrays or first_audio_ns is None:
        raise RuntimeError("Voxtral TTS returned no audio")
    audio = np.concatenate(arrays).astype("<f4", copy=False)
    if not np.isfinite(audio).all():
        raise RuntimeError("Voxtral TTS returned non-finite audio")

    raw_audio = audio.tobytes()
    raw_path = args.output_dir / "audio.f32le"
    wav_path = args.output_dir / "audio.pcm16.wav"
    raw_path.write_bytes(raw_audio)
    write_pcm16_wav(wav_path, audio, contract["sample_rate_hz"])

    sample_count = int(audio.size)
    duration_ms = sample_count * 1000.0 / contract["sample_rate_hz"]
    complete_synthesis_ms = (
        generation_completed - generation_started
    ) / 1_000_000
    first_audio_ms = (first_audio_ns - generation_started) / 1_000_000
    model_load_ms = (load_completed - load_started) / 1_000_000
    token_count = sum(item["token_count"] for item in output_metrics)
    terminated_at_limit = token_count >= contract["max_tokens"]

    result = {
        "schema_version": "1.0.0",
        "run_id": "phase5.voxtral-tts-4b-mlx-reference.synthetic-de-origin-1",
        "captured_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "source_revision": args.source_revision,
        "selection_revision": "synthetic-roundtrip-passages-1",
        "purpose": "development-diagnostic",
        "candidate": {
            "id": "voxtral-tts-4b-mlx-audio",
            "task": "tts",
            "model": {
                "repository": manifest["conversion"]["repository"],
                "revision": manifest["conversion"]["revision"],
                "license": manifest["conversion"]["license"],
                "snapshot_bytes": manifest["conversion"]["snapshot_bytes"],
            },
            "runtime": {
                "repository": manifest["reference_runtime"]["repository"],
                "revision": manifest["reference_runtime"]["revision"],
                "version": package_version("mlx-audio"),
                "boundary": manifest["reference_runtime"]["boundary"],
                "license": manifest["reference_runtime"]["license"],
            },
            "voice": {
                "name": contract["speaker"],
                "language": contract["speaker_native_language"],
                "native_language": contract["speaker_native_language"],
                "cross_lingual": contract["cross_lingual"],
            },
        },
        "host": {
            "id": "mac-studio-m1-ultra-local",
            "chip": command_output("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": int(command_output("sysctl", "-n", "hw.memsize")),
            "os": f"macOS {command_output('sw_vers', '-productVersion')} "
            f"({command_output('sw_vers', '-buildVersion')})",
            "architecture": platform.machine(),
            "execution_context": (
                "normal host process outside the restricted diagnostic sandbox"
            ),
        },
        "input": {
            "passage_id": contract["passage_id"],
            "source_id": "de-wikipedia-kuenstliche-intelligenz-268935951",
            "locale": contract["locale"],
            "character_count": len(text),
            "text_sha256": text_sha256,
            "license": "CC-BY-SA-4.0",
        },
        "procedure": {
            "repetitions": 1,
            "generation": contract,
            "stream": False,
            "cold_load": (
                "Fresh Python process; OS file and Metal caches were not cleared."
            ),
            "command": (
                "CUTTLEDOC_VOXTRAL_TTS_MODEL_DIR=/local/model "
                "CUTTLEDOC_TTS_TEXT_DIR=/local/passages "
                "CUTTLEDOC_VOXTRAL_TTS_OUTPUT_DIR=/local/output "
                "bash scripts/run-voxtral-tts-mlx-reference.sh"
            ),
            "raw_audio": (
                "local-required; not checked into Git; CC-BY-SA input and "
                "CC-BY-NC model handling required"
            ),
            "raw_artifacts": [
                raw_path.name,
                wav_path.name,
                "result.json",
            ],
        },
        "result": {
            "status": "partial" if terminated_at_limit else "measured",
            "audio": {
                "sample_format": "f32le",
                "sample_rate_hz": contract["sample_rate_hz"],
                "channel_count": 1,
                "sample_count": sample_count,
                "byte_count": len(raw_audio),
                "duration_ms": duration_ms,
                "sha256": hashlib.sha256(raw_audio).hexdigest(),
                "minimum_sample": float(audio.min()),
                "maximum_sample": float(audio.max()),
                "rms": float(
                    np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
                ),
                "non_finite_sample_count": int(
                    sample_count - np.count_nonzero(np.isfinite(audio))
                ),
            },
            "timing": {
                "model_load_ms": model_load_ms,
                "first_audio_ms": first_audio_ms,
                "complete_synthesis_ms": complete_synthesis_ms,
                "real_time_factor": complete_synthesis_ms / duration_ms,
                "output_count": len(output_metrics),
                "token_count": token_count,
            },
            "resources": {
                "maximum_resident_set_size_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                "model_snapshot_bytes": manifest["conversion"][
                    "snapshot_bytes"
                ],
            },
            "runtime_output": output_metrics,
            "termination": {
                "reached_max_tokens": terminated_at_limit,
                "configured_max_tokens": contract["max_tokens"],
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
            "supported_speakers": manifest["architecture"]["voices"],
            "supported_languages": manifest["source"]["supported_languages"],
        },
        "conclusion": {
            "reference_path_proven": not terminated_at_limit,
            "public_contract_accepted": False,
            "production_dependency_eligible": False,
            "license_blocker": (
                "The local model and supplied voice references are "
                "CC-BY-NC-4.0; a separate commercial license or API path is "
                "required for production use."
            ),
            "next": (
                "Inspect German intelligibility and prosody, normalize to "
                "shared 16 kHz mono PCM, and run the four ASR backends."
            ),
        },
    }

    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
