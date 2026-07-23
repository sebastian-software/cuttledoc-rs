#!/usr/bin/env python3

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one pinned local MLX candidate over factorial documents."
    )
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(screen: dict, manifest: dict, model_dir: Path, prompt: Path) -> None:
    snapshot_bytes = 0
    for artifact in manifest["artifacts"]:
        path = model_dir / artifact["path"]
        if not path.is_file():
            raise ValueError(f"Missing model artifact: {path}")
        size = path.stat().st_size
        snapshot_bytes += size
        if size != artifact["bytes"]:
            raise ValueError(f"Model artifact byte-count drift: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Model artifact SHA-256 drift: {path}")
    if snapshot_bytes != manifest["conversion"]["snapshot_bytes"]:
        raise ValueError("Model snapshot byte-count drift")
    if sha256_file(prompt) != screen["generation"]["prompt_sha256"]:
        raise ValueError("Prompt SHA-256 drift")


def render_prompt(
    template: str, document: dict, target_index: int | None
) -> str:
    model_input = document["model_input"]
    target_section = None
    context_sections = []
    if target_index is not None:
        if not 0 <= target_index < len(model_input["sections"]):
            raise ValueError(f"Target index out of range: {target_index}")
        target_section = {
            "position": target_index,
            "text": model_input["sections"][target_index]["text"],
        }
        context_sections = [
            {"position": index, "text": section["text"]}
            for index, section in enumerate(model_input["sections"])
            if index != target_index
        ]
    replacements = {
        "{{language}}": model_input["language"],
        "{{locale}}": model_input["locale"],
        "{{domain}}": model_input["domain"],
        "{{backend}}": model_input["asr_backend"],
        "{{error_profile}}": model_input["error_profile"],
        "{{glossary}}": json.dumps(model_input["glossary"], ensure_ascii=False),
        "{{sections}}": json.dumps(
            model_input["sections"], ensure_ascii=False, indent=2
        ),
        "{{target_section}}": json.dumps(
            target_section, ensure_ascii=False, indent=2
        ),
        "{{context_sections}}": json.dumps(
            context_sections, ensure_ascii=False, indent=2
        ),
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("Prompt contains an unresolved template marker")
    return rendered


def generate(model, tokenizer, prompt: str, generation: dict) -> dict:
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **generation.get("chat_template_options", {}),
    )
    mx.random.seed(generation["seed"])
    sampler = make_sampler(temp=generation["temperature"])
    started = time.perf_counter_ns()
    first_token_ns = None
    text_segments: list[str] = []
    token_ids: list[int] = []
    final_response = None
    for response in stream_generate(
        model,
        tokenizer,
        templated,
        max_tokens=generation["max_tokens"],
        sampler=sampler,
    ):
        if first_token_ns is None:
            first_token_ns = time.perf_counter_ns()
        text_segments.append(response.text)
        token_ids.append(int(response.token))
        final_response = response
    completed = time.perf_counter_ns()
    if first_token_ns is None or final_response is None:
        raise RuntimeError("MLX-LM returned no generation updates")
    return {
        "text": "".join(text_segments).strip(),
        "token_ids": token_ids,
        "prompt_tokens": int(final_response.prompt_tokens),
        "generation_tokens": int(final_response.generation_tokens),
        "finish_reason": final_response.finish_reason,
        "first_token_ms": (first_token_ns - started) / 1_000_000,
        "complete_generation_ms": (completed - started) / 1_000_000,
        "prompt_tokens_per_second": float(final_response.prompt_tps),
        "generation_tokens_per_second": float(final_response.generation_tps),
        "mlx_peak_memory_bytes": int(final_response.peak_memory * 1_000_000_000),
    }


def parse_sections(
    raw_text: str,
    input_sections: list[dict],
    generation: dict,
    target_index: int | None,
) -> dict:
    payload = raw_text.strip()
    transport_envelope = "bare-json"
    if payload.startswith("```json\n") and payload.endswith("\n```"):
        payload = payload[len("```json\n") : -len("\n```")]
        transport_envelope = "markdown-json-fence"
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        return {
            "valid": False,
            "error": str(error),
            "sections": [],
            "transport_envelope": transport_envelope,
        }
    expected_ids = [section["id"] for section in input_sections]
    output_contract = generation["output_contract"]
    if output_contract in {
        "bounded-transcript-target-complete-v4",
        "bounded-transcript-target-complete-v5",
    }:
        if (
            target_index is None
            or not isinstance(value, dict)
            or set(value) != {"text"}
            or not isinstance(value["text"], str)
            or not value["text"].strip()
        ):
            return {
                "valid": False,
                "error": "Output must contain exactly one non-empty text field",
                "sections": [],
            }
        return {
            "valid": True,
            "error": None,
            "sections": [
                {"id": expected_ids[target_index], "text": value["text"]}
            ],
            "patches": None,
            "transport_envelope": transport_envelope,
        }
    if output_contract in {
        "bounded-transcript-target-patches-v4",
        "bounded-transcript-target-patches-v5",
    }:
        if (
            target_index is None
            or not isinstance(value, dict)
            or set(value) != {"patches"}
            or not isinstance(value["patches"], list)
        ):
            return {
                "valid": False,
                "error": "Output must contain exactly one patches array",
                "sections": [],
            }
        limits = generation["patch_limits"]
        patches = value["patches"]
        if len(patches) > limits["maximum_patches"]:
            return {
                "valid": False,
                "error": "Patch count exceeds the configured maximum",
                "sections": [],
            }
        target = input_sections[target_index]["text"]
        ranges = []
        total_source_characters = 0
        for patch in patches:
            if (
                not isinstance(patch, dict)
                or set(patch) != {"old", "new"}
                or not isinstance(patch["old"], str)
                or not patch["old"]
                or not isinstance(patch["new"], str)
                or patch["old"] == patch["new"]
            ):
                return {
                    "valid": False,
                    "error": "Every patch must contain distinct old and new strings",
                    "sections": [],
                }
            if (
                len(patch["old"]) > limits["maximum_source_characters_per_patch"]
                or len(patch["new"])
                > limits["maximum_replacement_characters_per_patch"]
            ):
                return {
                    "valid": False,
                    "error": "Patch span exceeds the configured character limit",
                    "sections": [],
                }
            if target.count(patch["old"]) != 1:
                return {
                    "valid": False,
                    "error": "Every old patch span must occur exactly once in the target",
                    "sections": [],
                }
            start = target.index(patch["old"])
            ranges.append((start, start + len(patch["old"]), patch))
            total_source_characters += len(patch["old"])
        if total_source_characters > limits["maximum_total_source_characters"]:
            return {
                "valid": False,
                "error": "Total patched source text exceeds the configured maximum",
                "sections": [],
            }
        ranges.sort(key=lambda item: item[0])
        if any(
            current[0] < previous[1]
            for previous, current in zip(ranges, ranges[1:])
        ):
            return {
                "valid": False,
                "error": "Patch spans must not overlap",
                "sections": [],
            }
        pieces = []
        cursor = 0
        for start, end, patch in ranges:
            pieces.extend([target[cursor:start], patch["new"]])
            cursor = end
        pieces.append(target[cursor:])
        return {
            "valid": True,
            "error": None,
            "sections": [
                {"id": expected_ids[target_index], "text": "".join(pieces)}
            ],
            "patches": patches,
            "transport_envelope": transport_envelope,
        }
    sections = value.get("sections") if isinstance(value, dict) else None
    if not isinstance(sections, list):
        return {
            "valid": False,
            "error": "Output must be an object with a sections array",
            "sections": [],
        }
    if output_contract == "bounded-transcript-sections-positional-v3":
        positions = []
        normalized = []
        for section in sections:
            if (
                not isinstance(section, dict)
                or not isinstance(section.get("index"), int)
                or isinstance(section.get("index"), bool)
            ):
                return {
                    "valid": False,
                    "error": "Every section must have an integer index",
                    "sections": [],
                }
            if not isinstance(section.get("text"), str):
                return {
                    "valid": False,
                    "error": "Every section must have a string text",
                    "sections": [],
                }
            positions.append(section["index"])
            if 0 <= section["index"] < len(expected_ids):
                normalized.append(
                    {
                        "id": expected_ids[section["index"]],
                        "text": section["text"],
                    }
                )
        if positions != list(range(len(expected_ids))):
            return {
                "valid": False,
                "error": "Section indices or order differ from the input contract",
                "sections": normalized,
            }
        return {"valid": True, "error": None, "sections": normalized}
    if output_contract != "bounded-transcript-sections-local-diff-v2":
        return {
            "valid": False,
            "error": f"Unsupported output contract: {output_contract}",
            "sections": [],
        }
    ids = []
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("id"), str):
            return {
                "valid": False,
                "error": "Every section must have a string id",
                "sections": [],
            }
        if not isinstance(section.get("text"), str):
            return {
                "valid": False,
                "error": "Every section must have a string text",
                "sections": [],
            }
        ids.append(section["id"])
    if ids != expected_ids:
        return {
            "valid": False,
            "error": "Section ids or order differ from the input contract",
            "sections": sections,
        }
    return {"valid": True, "error": None, "sections": sections}


def valid_existing(
    path: Path,
    candidate_id: str,
    document_sha256: str,
    prompt_sha256: str,
    repetition: int,
    screen_id: str,
    target_index: int | None,
) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return (
            result["candidate"]["manifest_id"] == candidate_id
            and result["document"]["sha256"] == document_sha256
            and result["document"].get("target_index") == target_index
            and result["prompt"]["sha256"] == prompt_sha256
            and result["repetition"] == repetition
            and result["screen_id"] == screen_id
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = parse_args()
    if args.repetitions is not None and args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if not (
        len(args.source_revision) == 40
        and all(character in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("--source-revision must be a lowercase 40-character Git SHA")

    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    generation = screen["generation"]
    repetitions = args.repetitions or generation["repetitions"]
    prompt_sha256 = generation["prompt_sha256"]
    verify_inputs(screen, manifest, args.model_dir, args.prompt)

    pending = []
    resumed = 0
    for job in jobs:
        document_path = Path(job["document_path"])
        document_bytes = document_path.read_bytes()
        document_sha256 = sha256_bytes(document_bytes)
        request_id = job.get("request_id", job["document_id"])
        for repetition in range(1, repetitions + 1):
            output = args.output_dir / request_id / f"repeat-{repetition}.json"
            if not args.force and valid_existing(
                output,
                manifest["id"],
                document_sha256,
                prompt_sha256,
                repetition,
                screen["id"],
                job.get("target_index"),
            ):
                resumed += 1
            else:
                pending.append(
                    (job, document_path, document_sha256, repetition, output)
                )

    if not pending:
        print(
            json.dumps(
                {
                    "candidate": manifest["id"],
                    "completed": 0,
                    "resumed": resumed,
                    "selected_requests": len(jobs) * repetitions,
                }
            )
        )
        return

    load_started = time.perf_counter_ns()
    model, tokenizer = load(str(args.model_dir))
    load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    prompt_template = args.prompt.read_text(encoding="utf-8")
    completed = 0
    for job, document_path, document_sha256, repetition, output in pending:
        document = json.loads(document_path.read_text(encoding="utf-8"))
        target_index = job.get("target_index")
        rendered_prompt = render_prompt(prompt_template, document, target_index)
        generation_result = generate(model, tokenizer, rendered_prompt, generation)
        parsed = parse_sections(
            generation_result["text"],
            document["model_input"]["sections"],
            generation,
            target_index,
        )
        reached_limit = generation_result["generation_tokens"] >= generation["max_tokens"]
        result = {
            "schema_version": "1.0.0",
            "id": (
                f"{manifest['id']}--"
                f"{job.get('request_id', job['document_id'])}--"
                f"repeat-{repetition}"
            ),
            "screen_id": screen["id"],
            "plan_id": screen["plan_id"],
            "plan_revision": screen["plan_revision"],
            "source_revision": args.source_revision,
            "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "candidate": {
                "manifest_id": manifest["id"],
                "role": manifest["candidate_role"],
                "conversion": manifest["conversion"],
                "runtime": manifest["reference_runtime"],
            },
            "document": {
                "id": job["document_id"],
                "path": str(document_path),
                "sha256": document_sha256,
                "locale": document["model_input"]["locale"],
                "tts_engine": document["dimensions"]["tts_engine"],
                "voice_slot_id": document["dimensions"]["voice_slot_id"],
                "realization_id": document["dimensions"]["realization_id"],
                "stt_model": document["dimensions"]["stt_model"],
                "target_index": target_index,
                "target_section_id": (
                    None
                    if target_index is None
                    else document["model_input"]["sections"][target_index]["id"]
                ),
            },
            "prompt": {
                "id": generation["prompt_id"],
                "sha256": prompt_sha256,
                "rendered_sha256": sha256_bytes(rendered_prompt.encode("utf-8")),
                "reference_visible_to_model": False,
            },
            "repetition": repetition,
            "generation": {
                "seed": generation["seed"],
                "temperature": generation["temperature"],
                "max_tokens": generation["max_tokens"],
                "finish_reason": generation_result["finish_reason"],
                "reached_token_limit": reached_limit,
            },
            "measurements": {
                "model_load_ms": load_ms,
                "first_token_ms": generation_result["first_token_ms"],
                "complete_generation_ms": generation_result["complete_generation_ms"],
                "prompt_tokens": generation_result["prompt_tokens"],
                "generation_tokens": generation_result["generation_tokens"],
                "prompt_tokens_per_second": generation_result[
                    "prompt_tokens_per_second"
                ],
                "generation_tokens_per_second": generation_result[
                    "generation_tokens_per_second"
                ],
                "maximum_resident_set_size_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "mlx_peak_memory_bytes": generation_result["mlx_peak_memory_bytes"],
            },
            "output": {
                "raw_text": generation_result["text"],
                "raw_text_sha256": sha256_bytes(
                    generation_result["text"].encode("utf-8")
                ),
                "token_ids": generation_result["token_ids"],
                "parser": {
                    "valid": parsed["valid"],
                    "error": parsed["error"],
                    "transport_envelope": parsed.get(
                        "transport_envelope",
                        (
                            "markdown-json-fence"
                            if generation_result["text"].strip().startswith(
                                "```json\n"
                            )
                            and generation_result["text"].strip().endswith(
                                "\n```"
                            )
                            else "bare-json"
                        ),
                    ),
                },
                "sections": parsed["sections"],
                "patches": parsed.get("patches"),
                "mechanically_accepted": parsed["valid"] and not reached_limit,
            },
            "environment": {
                "python": os.sys.version.split()[0],
                "packages": {
                    "mlx": importlib.metadata.version("mlx"),
                    "mlx-lm": importlib.metadata.version("mlx-lm"),
                },
            },
        }
        atomic_json(output, result)
        completed += 1
        print(
            f"local llm: {manifest['id']} "
            f"{job.get('request_id', job['document_id'])} "
            f"repeat {repetition} ({completed + resumed}/{len(jobs) * repetitions}) "
            f"parser={'ok' if parsed['valid'] else 'invalid'} "
            f"tokens={generation_result['generation_tokens']}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "candidate": manifest["id"],
                "completed": completed,
                "resumed": resumed,
                "selected_requests": len(jobs) * repetitions,
            }
        )
    )


if __name__ == "__main__":
    main()
