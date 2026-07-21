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


def render_prompt(template: str, fixture: dict) -> str:
    replacements = {
        "{{language}}": fixture["language"],
        "{{domain}}": fixture["domain"],
        "{{transcript}}": fixture["transcript"],
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("Prompt contains an unresolved template marker")
    return rendered


def verify_inputs(manifest: dict, model_dir: Path, prompt: Path) -> None:
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
    if prompt_digest != manifest["generation_contract"]["prompt_sha256"]:
        raise ValueError(f"Prompt SHA-256 drift: {prompt_digest}")


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
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    verify_inputs(manifest, args.model_dir, args.prompt)
    prompt_template = args.prompt.read_text(encoding="utf-8")
    rendered_prompt = render_prompt(prompt_template, fixture)
    contract = manifest["generation_contract"]
    result_contract = manifest["result_contract"]

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
    lexical_invariant = normalized_words(generation["text"]) == normalized_words(
        fixture["transcript"]
    )
    output_nonempty = len(generation["text"]) > 0

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
            "model_load": "Fresh Python process; OS file and Metal caches were not cleared.",
            "generation": contract,
            "complete_generation_repetitions": 2,
            "prompt_visible_fields": ["language", "domain", "transcript"],
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
            "text": generation["text"],
            "text_sha256": sha256_bytes(generation["text"].encode("utf-8")),
            "token_ids": generation["token_ids"],
            "gates": {
                "nonempty": output_nonempty,
                "case_and_punctuation_insensitive_lexical_invariant": lexical_invariant,
                "accepted": output_nonempty and lexical_invariant,
            },
            "development_quality": {
                "input_wer": word_error_rate(
                    fixture["evaluation_reference"], fixture["transcript"]
                ),
                "output_wer": word_error_rate(
                    fixture["evaluation_reference"], generation["text"]
                ),
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
            "surface_candidate_accepted": output_nonempty and lexical_invariant,
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
