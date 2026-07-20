#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any


SAMPLE_RATE_HZ = 16_000
FIXTURE_IDS = [
    "audiobook-de-135_82_000105",
    "audiobook-en-2277-149874-0000",
    "audiobook-es-10367_10282_000000",
    "audiobook-fr-1591_1028_000000",
    "audiobook-pt-12710_10229_000000",
]
DELAYS_MS = [480, 2_400]


def main() -> None:
    arguments = parse_arguments()
    repository = Path(__file__).resolve().parent.parent
    manifest_path = Path(arguments.manifest).resolve()
    fixture_directory = Path(arguments.fixture_directory).resolve()
    binary = Path(arguments.binary).resolve()
    model_directory = Path(arguments.model_directory).resolve()
    manifest = json.loads(manifest_path.read_text())
    fixtures = {fixture["id"]: fixture for fixture in manifest["fixtures"]}
    missing = [fixture_id for fixture_id in FIXTURE_IDS if fixture_id not in fixtures]
    if missing:
        raise SystemExit(f"manifest is missing fixtures: {', '.join(missing)}")

    results = []
    for fixture_id in FIXTURE_IDS:
        fixture = fixtures[fixture_id]
        audio_path = fixture_directory / (
            f"{fixture['language']}-{fixture['row_id']}."
            f"{fixture['normalized']['extension']}"
        )
        verify_audio(audio_path, fixture)
        for delay_ms in DELAYS_MS:
            completed = subprocess.run(
                [
                    str(binary),
                    "stream",
                    str(model_directory),
                    str(audio_path),
                    str(delay_ms),
                    "320",
                    "16",
                    "gpu",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            run = json.loads(completed.stdout)
            quality = quality_metrics(fixture["reference_text"], run["text"])
            results.append(
                {
                    "fixture": {
                        "id": fixture["id"],
                        "language": fixture["language"],
                        "gold_status": fixture["gold_status"],
                        "reference_text": fixture["reference_text"],
                        "audio_sha256": fixture["normalized"]["sha256"],
                        "sample_count": fixture["normalized"]["sample_count"],
                        "duration_ms": fixture["normalized"]["duration_ms"],
                    },
                    "transcription_delay_ms": delay_ms,
                    "quality": quality,
                    "run": run,
                }
            )

    artifact_paths = {
        "adapter_dylib": binary.parent / "libcuttledoc_voxtral_mlx_shim.dylib",
        "rust_driver": binary,
        "mlx_metallib": binary.parent / "mlx.metallib",
        "model_weights": model_directory / "model.safetensors",
    }
    result = {
        "schema_version": "1.0.0",
        "captured_at": utc_now(),
        "source_revision": command_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"]
        ).strip(),
        "candidate": {
            "id": "voxtral-mini-4b-realtime-mlx-direct",
            "boundary": "repository-owned Rust/C ABI/C++ adapter over official MLX",
        },
        "fixture_manifest_revision": manifest["revision"],
        "procedure": {
            "fixture_selection": "one pinned development audiobook per language",
            "languages": ["de", "en", "es", "fr", "pt"],
            "transcription_delays_ms": DELAYS_MS,
            "chunk_ms": 320,
            "max_decode_tokens_per_step": 16,
            "fresh_process_per_run": True,
            "run_count": len(results),
            "scope": "multilingual live-operation control; not held-out model selection",
        },
        "artifacts": {
            name: {"path": str(path), "bytes": path.stat().st_size}
            for name, path in artifact_paths.items()
        },
        "summary": summarize(results),
        "results": results,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the direct official-MLX Voxtral stream on one pinned "
            "development audiobook per language and delay."
        )
    )
    parser.add_argument("binary")
    parser.add_argument("model_directory")
    parser.add_argument("manifest")
    parser.add_argument("fixture_directory")
    return parser.parse_args()


def verify_audio(path: Path, fixture: dict[str, Any]) -> None:
    if not path.is_file():
        raise SystemExit(f"missing normalized fixture: {path}")
    data = path.read_bytes()
    normalized = fixture["normalized"]
    digest = hashlib.sha256(data).hexdigest()
    if digest != normalized["sha256"]:
        raise SystemExit(f"{path}: SHA-256 {digest} differs from manifest")
    if len(data) != normalized["bytes"] or len(data) // 4 != normalized["sample_count"]:
        raise SystemExit(f"{path}: byte or sample count differs from manifest")


def quality_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_words = words(reference)
    hypothesis_words = words(hypothesis)
    reference_characters = list("".join(reference_words))
    hypothesis_characters = list("".join(hypothesis_words))
    word_edits = edit_distance(reference_words, hypothesis_words)
    character_edits = edit_distance(reference_characters, hypothesis_characters)
    return {
        "word_edits": word_edits,
        "reference_word_count": len(reference_words),
        "wer": word_edits / len(reference_words),
        "character_edits": character_edits,
        "reference_character_count": len(reference_characters),
        "cer": character_edits / len(reference_characters),
    }


def words(text: str) -> list[str]:
    normalized = "".join(
        character
        if character.isspace() or unicodedata.category(character)[0] in {"L", "N"}
        else ""
        for character in text.lower()
    )
    return normalized.split()


def edit_distance(reference: list[Any], hypothesis: list[Any]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_delay = {}
    for delay_ms in DELAYS_MS:
        matching = [
            result for result in results if result["transcription_delay_ms"] == delay_ms
        ]
        by_delay[str(delay_ms)] = {
            "macro_wer": sum(result["quality"]["wer"] for result in matching)
            / len(matching),
            "macro_cer": sum(result["quality"]["cer"] for result in matching)
            / len(matching),
            "mean_first_append_ms": sum(
                result["run"]["timing"]["first_append_ms"] for result in matching
            )
            / len(matching),
            "maximum_step_wall_ms": max(
                result["run"]["timing"]["maximum_step_wall_ms"]
                for result in matching
            ),
            "maximum_endpoint_finalization_ms": max(
                result["run"]["timing"]["endpoint_finalization_ms"]
                for result in matching
            ),
        }
    return {"by_delay_ms": by_delay}


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    main()
