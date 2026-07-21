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
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned MLX text-generation reference probe."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def normalized_words(text: str) -> list[str]:
    normalized = "".join(
        character.casefold()
        if unicodedata.category(character)[0] in {"L", "N"}
        else " "
        for character in text
    )
    return normalized.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalized_words(reference)
    hypothesis_words = normalized_words(hypothesis)
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words) if reference_words else 0.0


def render_prompt(template: str, fixture: dict, contract: dict) -> str:
    def render_value(value) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    replacements = {
        "{{language}}": fixture["language"],
        "{{locale}}": fixture.get("locale", "unspecified"),
        "{{domain}}": fixture["domain"],
        "{{backend}}": fixture.get("asr_backend", "unspecified"),
        "{{error_profile}}": fixture.get(
            "error_profile", "No independent error profile supplied"
        ),
        "{{glossary}}": fixture.get("glossary", []),
        "{{protected_spans}}": fixture.get("protected_spans", []),
        "{{suspect_spans}}": fixture.get("suspect_spans", []),
        "{{transcript}}": fixture["transcript"],
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, render_value(value))
    if contract["render_mode"] == "append-transcript":
        if "{{transcript}}" in template:
            raise ValueError("Append-transcript prompt must not contain a transcript marker")
        rendered = f"{rendered.rstrip()}\n{fixture['transcript']}"
    elif contract["render_mode"] != "template":
        raise ValueError(f"Unsupported prompt render mode: {contract['render_mode']}")
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("Prompt contains an unresolved template marker")
    return rendered


def verify_inputs(manifest: dict, contract: dict, model_dir: Path, prompt: Path) -> None:
    snapshot_bytes = 0
    for artifact in manifest["artifacts"]:
        path = model_dir / artifact["path"]
        if not path.is_file():
            raise ValueError(f"Missing model artifact: {path}")
        contents = path.read_bytes()
        snapshot_bytes += len(contents)
        if len(contents) != artifact["bytes"]:
            raise ValueError(f"Model artifact byte-count drift: {path}")
        if sha256_bytes(contents) != artifact["sha256"]:
            raise ValueError(f"Model artifact SHA-256 drift: {path}")
    if snapshot_bytes != manifest["conversion"]["snapshot_bytes"]:
        raise ValueError("Model snapshot byte-count drift")

    prompt_digest = sha256_bytes(prompt.read_bytes())
    if prompt_digest != contract["prompt_sha256"]:
        raise ValueError(f"Prompt SHA-256 drift: {prompt_digest}")


def word_diff(source: str, target: str) -> dict:
    source_words = normalized_words(source)
    target_words = normalized_words(target)
    rows = len(source_words) + 1
    columns = len(target_words) + 1
    distances = [[0] * columns for _ in range(rows)]
    for source_index in range(rows):
        distances[source_index][0] = source_index
    for target_index in range(columns):
        distances[0][target_index] = target_index
    for source_index in range(1, rows):
        for target_index in range(1, columns):
            distances[source_index][target_index] = min(
                distances[source_index - 1][target_index] + 1,
                distances[source_index][target_index - 1] + 1,
                distances[source_index - 1][target_index - 1]
                + (source_words[source_index - 1] != target_words[target_index - 1]),
            )

    operations = []
    source_index = len(source_words)
    target_index = len(target_words)
    while source_index > 0 or target_index > 0:
        if (
            source_index > 0
            and target_index > 0
            and source_words[source_index - 1] == target_words[target_index - 1]
            and distances[source_index][target_index]
            == distances[source_index - 1][target_index - 1]
        ):
            source_index -= 1
            target_index -= 1
        elif (
            source_index > 0
            and target_index > 0
            and distances[source_index][target_index]
            == distances[source_index - 1][target_index - 1] + 1
        ):
            operations.append(
                {
                    "type": "replace",
                    "input_index": source_index - 1,
                    "output_index": target_index - 1,
                    "input": source_words[source_index - 1],
                    "output": target_words[target_index - 1],
                }
            )
            source_index -= 1
            target_index -= 1
        elif (
            source_index > 0
            and distances[source_index][target_index]
            == distances[source_index - 1][target_index] + 1
        ):
            operations.append(
                {
                    "type": "delete",
                    "input_index": source_index - 1,
                    "output_index": target_index,
                    "input": source_words[source_index - 1],
                    "output": None,
                }
            )
            source_index -= 1
        else:
            operations.append(
                {
                    "type": "insert",
                    "input_index": source_index,
                    "output_index": target_index - 1,
                    "input": None,
                    "output": target_words[target_index - 1],
                }
            )
            target_index -= 1
    operations.reverse()
    return {
        "input_word_count": len(source_words),
        "output_word_count": len(target_words),
        "edit_distance": distances[-1][-1],
        "operations": operations,
    }


def count_word_sequence(words: list[str], sequence: list[str]) -> int:
    if not sequence:
        return 0
    width = len(sequence)
    return sum(
        words[index : index + width] == sequence
        for index in range(len(words) - width + 1)
    )


def parse_output(raw_text: str, contract: dict, fixture: dict) -> dict:
    output_mode = contract["output_mode"]
    if output_mode == "plain-text":
        return {
            "text": raw_text,
            "reported_edits": None,
            "parser": {"valid": True, "error": None},
            "audit": {
                "lexical_edits_fully_reported": None,
                "protected_spans_unchanged": None,
                "reported_lexical_edit_count": None,
            },
        }
    if output_mode != "json-edits":
        raise ValueError(f"Unsupported output mode: {output_mode}")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        return {
            "text": "",
            "reported_edits": None,
            "parser": {"valid": False, "error": str(error)},
            "audit": {
                "lexical_edits_fully_reported": False,
                "protected_spans_unchanged": False,
                "reported_lexical_edit_count": None,
            },
        }

    if not isinstance(parsed, dict) or not isinstance(parsed.get("text"), str):
        return {
            "text": "",
            "reported_edits": None,
            "parser": {
                "valid": False,
                "error": "Output must be an object with a string text field",
            },
            "audit": {
                "lexical_edits_fully_reported": False,
                "protected_spans_unchanged": False,
                "reported_lexical_edit_count": None,
            },
        }
    edits = parsed.get("edits")
    if not isinstance(edits, list) or any(
        not isinstance(edit, dict)
        or not isinstance(edit.get("original"), str)
        or not isinstance(edit.get("replacement"), str)
        or edit.get("class")
        not in {
            "surface",
            "word-boundary",
            "contextual-misrecognition",
            "asr-inflection",
        }
        or not isinstance(edit.get("reason"), str)
        or not isinstance(edit.get("confidence"), (int, float))
        or not 0 <= edit["confidence"] <= 1
        for edit in edits
    ):
        return {
            "text": parsed["text"],
            "reported_edits": edits,
            "parser": {
                "valid": False,
                "error": "edits must match the bounded lexical audit schema",
            },
            "audit": {
                "lexical_edits_fully_reported": False,
                "protected_spans_unchanged": False,
                "reported_lexical_edit_count": None,
            },
        }

    diff = word_diff(fixture["transcript"], parsed["text"])
    operation_pairs = Counter(
        (
            tuple(normalized_words(operation["input"] or "")),
            tuple(normalized_words(operation["output"] or "")),
        )
        for operation in diff["operations"]
    )
    reported_pairs = Counter()
    for edit in edits:
        pair = (
            tuple(normalized_words(edit["original"])),
            tuple(normalized_words(edit["replacement"])),
        )
        if pair[0] != pair[1]:
            reported_pairs[pair] += 1

    input_words = normalized_words(fixture["transcript"])
    output_words = normalized_words(parsed["text"])
    protected_unchanged = all(
        count_word_sequence(input_words, normalized_words(span))
        == count_word_sequence(output_words, normalized_words(span))
        for span in fixture.get("protected_spans", [])
    )
    return {
        "text": parsed["text"],
        "reported_edits": edits,
        "parser": {"valid": True, "error": None},
        "audit": {
            "lexical_edits_fully_reported": reported_pairs == operation_pairs,
            "protected_spans_unchanged": protected_unchanged,
            "reported_lexical_edit_count": sum(reported_pairs.values()),
        },
    }


def run_generation(model, tokenizer, prompt: str, contract: dict) -> dict:
    messages = [{"role": "user", "content": prompt}]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **contract.get("chat_template_options", {}),
    )
    mx.random.seed(contract["seed"])
    sampler = make_sampler(temp=contract["temperature"])

    started = time.perf_counter_ns()
    first_token_ns = None
    text_segments: list[str] = []
    token_ids: list[int] = []
    final_response = None
    for response in stream_generate(
        model,
        tokenizer,
        templated_prompt,
        max_tokens=contract["max_tokens"],
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
        "update_count": len(token_ids),
    }


def run_cancellation_probe(model, tokenizer, prompt: str, contract: dict) -> dict:
    templated_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **contract.get("chat_template_options", {}),
    )
    cancel_after = contract["cancellation_probe_after_tokens"]
    generator = stream_generate(
        model,
        tokenizer,
        templated_prompt,
        max_tokens=contract["max_tokens"],
        sampler=make_sampler(temp=contract["temperature"]),
    )
    started = time.perf_counter_ns()
    observed = 0
    for response in generator:
        observed = int(response.generation_tokens)
        if observed >= cancel_after:
            break
    generator.close()
    mx.synchronize()
    return {
        "requested_after_tokens": cancel_after,
        "observed_tokens": observed,
        "cooperative_stop_ms": (time.perf_counter_ns() - started) / 1_000_000,
        "process_remained_usable": observed >= cancel_after,
        "semantics": "The consumer closes generation between token updates; no thread or process termination is required.",
    }


def main() -> None:
    args = parse_args()
    if not (
        len(args.source_revision) == 40
        and all(character in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("--source-revision must be a lowercase 40-character Git SHA")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    experiment = (
        json.loads(args.experiment.read_text(encoding="utf-8"))
        if args.experiment is not None
        else None
    )
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    contract = (
        experiment["generation_contract"]
        if experiment is not None
        else manifest["generation_contract"]
    )
    result_contract = (
        experiment["result_contract"]
        if experiment is not None
        else manifest["result_contract"]
    )
    if (
        experiment is not None
        and not args.manifest.as_posix().endswith(experiment["model_manifest_path"])
    ):
        raise ValueError("Experiment model path does not match --manifest")
    verify_inputs(manifest, contract, args.model_dir, args.prompt)
    prompt_template = args.prompt.read_text(encoding="utf-8")
    rendered_prompt = render_prompt(prompt_template, fixture, contract)

    if args.fixture.as_posix().endswith(result_contract["fixture_path"]) is False:
        raise ValueError("Fixture path does not match the manifest result contract")
    if args.prompt.as_posix().endswith(contract["prompt_path"]) is False:
        raise ValueError("Prompt path does not match the manifest generation contract")

    mx.reset_peak_memory()
    load_started = time.perf_counter_ns()
    model, tokenizer = load(str(args.model_dir))
    load_completed = time.perf_counter_ns()

    generation = run_generation(model, tokenizer, rendered_prompt, contract)
    deterministic_repeat = run_generation(model, tokenizer, rendered_prompt, contract)
    cancellation = run_cancellation_probe(model, tokenizer, rendered_prompt, contract)
    parsed_output = parse_output(generation["text"], contract, fixture)
    corrected_text = parsed_output["text"].strip()
    lexical_invariant = normalized_words(corrected_text) == normalized_words(
        fixture["transcript"]
    )
    output_nonempty = len(corrected_text) > 0
    policy_mode = contract["policy_mode"]
    if policy_mode == "surface-only":
        mechanical_output_accepted = output_nonempty and lexical_invariant
    elif policy_mode == "historical-control":
        mechanical_output_accepted = output_nonempty
    elif policy_mode == "bounded-lexical":
        mechanical_output_accepted = (
            output_nonempty
            and parsed_output["parser"]["valid"]
            and parsed_output["audit"]["lexical_edits_fully_reported"]
            and parsed_output["audit"]["protected_spans_unchanged"]
        )
    else:
        raise ValueError(f"Unsupported policy mode: {policy_mode}")

    result = {
        "schema_version": "1.0.0",
        "run_id": result_contract["run_id"],
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_revision": args.source_revision,
        "purpose": result_contract["purpose"],
        "candidate": {
            "task": "text-generation",
            "manifest_id": manifest["id"],
            "role": manifest["candidate_role"],
            "model": manifest["conversion"],
            "source_model": manifest["source"],
            "runtime": manifest["reference_runtime"],
        },
        "fixture": {
            "id": fixture["id"],
            "language": fixture["language"],
            "locale": fixture["locale"],
            "domain": fixture["domain"],
            "asr_backend": fixture["asr_backend"],
            "development_only": fixture["development_only"],
            "gold_status": fixture["gold_status"],
            "transcript_sha256": sha256_bytes(
                fixture["transcript"].encode("utf-8")
            ),
            "reference_visible_to_model": fixture["inference_policy"][
                "reference_visible_to_model"
            ],
        },
        "host": {
            "id": "mac-studio-m1-ultra-local",
            "chip": command_output("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": int(command_output("sysctl", "-n", "hw.memsize")),
            "os": f"macOS {command_output('sw_vers', '-productVersion')} ({command_output('sw_vers', '-buildVersion')})",
            "architecture": platform.machine(),
            "execution_context": "normal host process outside the restricted diagnostic sandbox",
        },
        "procedure": {
            "experiment_id": experiment["id"] if experiment is not None else None,
            "model_load": "Fresh Python process; OS file and Metal caches were not cleared.",
            "generation": contract,
            "complete_generation_repetitions": 2,
            "prompt_visible_fields": contract["context_fields"],
            "evaluation_reference_visible_to_model": False,
            "command": "CUTTLEDOC_TEXT_GENERATION_MANIFEST=/repo/candidate.json CUTTLEDOC_TEXT_GENERATION_MODEL_DIR=/local/model CUTTLEDOC_TEXT_GENERATION_OUTPUT=/tmp/result.json bash scripts/run-text-generation-mlx-reference.sh",
        },
        "measurements": {
            "model_load_ms": (load_completed - load_started) / 1_000_000,
            "first_token_ms": generation["first_token_ms"],
            "complete_generation_ms": generation["complete_generation_ms"],
            "prompt_tokens": generation["prompt_tokens"],
            "generation_tokens": generation["generation_tokens"],
            "prompt_tokens_per_second": generation["prompt_tokens_per_second"],
            "generation_tokens_per_second": generation[
                "generation_tokens_per_second"
            ],
            "maximum_resident_set_size_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
            "model_snapshot_bytes": manifest["conversion"]["snapshot_bytes"],
            "deterministic_repeat": {
                "text_identical": deterministic_repeat["text"] == generation["text"],
                "token_ids_identical": deterministic_repeat["token_ids"]
                == generation["token_ids"],
                "complete_generation_ms": deterministic_repeat[
                    "complete_generation_ms"
                ],
                "first_token_ms": deterministic_repeat["first_token_ms"],
            },
        },
        "streaming": {
            "supported": True,
            "update_count": generation["update_count"],
            "update_semantics": "append-only text deltas with token identifiers and one terminal finish reason",
            "finish_reason": generation["finish_reason"],
            "cancellation": cancellation,
        },
        "output": {
            "raw_text": generation["text"],
            "raw_text_sha256": sha256_bytes(generation["text"].encode("utf-8")),
            "text": corrected_text,
            "text_sha256": sha256_bytes(corrected_text.encode("utf-8")),
            "token_ids": generation["token_ids"],
            "gates": {
                "nonempty": output_nonempty,
                "case_and_punctuation_insensitive_lexical_invariant": lexical_invariant,
                "accepted": mechanical_output_accepted,
                "policy_mode": policy_mode,
                "quality_accepted": False,
            },
            "parser": parsed_output["parser"],
            "reported_edits": parsed_output["reported_edits"],
            "audit": parsed_output["audit"],
            "lexical_diff": word_diff(fixture["transcript"], corrected_text),
            "development_quality": {
                "input_wer": word_error_rate(
                    fixture["evaluation_reference"], fixture["transcript"]
                ),
                "output_wer": (
                    word_error_rate(fixture["evaluation_reference"], corrected_text)
                    if parsed_output["parser"]["valid"] and output_nonempty
                    else None
                ),
                "scored": parsed_output["parser"]["valid"] and output_nonempty,
                "claim_limit": "The dataset transcript is unverified and the fixture is development-exposed; these values are diagnostic only.",
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "packages": {
                "mlx": package_version("mlx"),
                "mlx-lm": package_version("mlx-lm"),
                "tokenizers": package_version("tokenizers"),
                "transformers": package_version("transformers"),
            },
        },
        "conclusion": {
            "reference_runtime_executed": True,
            "surface_candidate_accepted": (
                mechanical_output_accepted if policy_mode == "surface-only" else None
            ),
            "development_output_contract_accepted": mechanical_output_accepted,
            "model_quality_selected": False,
            "product_runtime_accepted": False,
            "reason": "The runtime path is independent of the output gate. A development fixture can establish execution, mechanical policy behavior, streaming, cancellation, determinism, and cost; it cannot select a correction model or promote the Python reference into the product.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
