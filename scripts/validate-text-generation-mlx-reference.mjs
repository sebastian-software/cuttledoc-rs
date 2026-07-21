#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const fixturePath = join(
  repoRoot,
  'benchmarks/postprocessing/fixtures/issue7-de-audiobook-whisper.json',
);
const resultPath = join(
  repoRoot,
  'benchmarks/postprocessing/runs/phase5.qwen3-0.6b-4bit-mlx-reference.issue7-de-audiobook-whisper-1.json',
);
const manifestPath = join(
  repoRoot,
  'spikes/text-generation-mlx-reference/model-manifest.json',
);
const promptPath = join(
  repoRoot,
  'benchmarks/postprocessing/prompts/surface-only-v1.txt',
);

const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const readJson = async (path) => JSON.parse(await readFile(path, 'utf8'));
const isPositive = (value) => Number.isFinite(value) && value > 0;

function validate(manifest, fixture, result, promptBytes) {
  const errors = [];
  const contract = manifest.generation_contract;
  const model = result.candidate?.model;
  const runtime = result.candidate?.runtime;

  if (manifest.schema_version !== '1.0.0' || manifest.task !== 'text-generation') {
    errors.push('manifest must identify the text-generation task');
  }
  for (const revision of [
    manifest.source?.observed_revision,
    manifest.conversion?.revision,
    manifest.reference_runtime?.revision,
    manifest.reference_runtime?.mlx_revision,
  ]) {
    if (!/^[0-9a-f]{40}$/.test(revision ?? '')) {
      errors.push(`invalid pinned revision: ${revision}`);
    }
  }
  if (manifest.source?.license !== 'Apache-2.0' ||
      manifest.conversion?.license !== 'Apache-2.0' ||
      manifest.reference_runtime?.license !== 'MIT' ||
      !(manifest.source?.provenance_limit?.length > 0)) {
    errors.push('model, conversion, and runtime licenses must remain explicit');
  }
  if (manifest.reference_runtime?.boundary !==
      'Pinned Python reference over official Apple MLX; not an accepted product dependency or stable API') {
    errors.push('the Python reference must not become a product boundary');
  }
  const artifacts = manifest.artifacts ?? [];
  if (artifacts.length === 0 || new Set(artifacts.map(({ path }) => path)).size !== artifacts.length ||
      artifacts.some(({ bytes, sha256: digest }) =>
        !Number.isInteger(bytes) || bytes <= 0 || !/^[0-9a-f]{64}$/.test(digest ?? ''))) {
    errors.push('model artifacts must have unique paths, byte counts, and SHA-256 digests');
  }
  if (artifacts.reduce((sum, artifact) => sum + artifact.bytes, 0) !==
      manifest.conversion?.snapshot_bytes) {
    errors.push('artifact byte counts must equal the model snapshot byte count');
  }
  if (sha256(promptBytes) !== contract?.prompt_sha256 ||
      contract?.prompt_id !== 'surface-only-v1' ||
      contract?.temperature !== 0 ||
      contract?.seed !== 0 ||
      contract?.stream !== true) {
    errors.push('prompt and deterministic generation contract must remain pinned');
  }

  if (fixture.id !== 'issue7-de-audiobook-whisper' ||
      fixture.purpose !== 'development-runtime-probe' ||
      fixture.development_only !== true ||
      fixture.gold_status !== 'dataset-transcript-unverified' ||
      fixture.inference_policy?.reference_visible_to_model !== false ||
      fixture.inference_policy?.lexical_changes_allowed !== false ||
      fixture.inference_policy?.prompt_id !== contract.prompt_id) {
    errors.push('fixture must remain development-only with evaluation gold hidden');
  }

  if (result.schema_version !== '1.0.0' ||
      result.purpose !== 'development-runtime-probe' ||
      !/^[0-9a-f]{40}$/.test(result.source_revision ?? '') ||
      !/^\d{4}-\d{2}-\d{2}T/.test(result.captured_at ?? '') ||
      result.candidate?.task !== 'text-generation') {
    errors.push('result identity and provenance are invalid');
  }
  if (model?.repository !== manifest.conversion.repository ||
      model?.revision !== manifest.conversion.revision ||
      model?.snapshot_bytes !== manifest.conversion.snapshot_bytes ||
      runtime?.revision !== manifest.reference_runtime.revision ||
      runtime?.mlx_revision !== manifest.reference_runtime.mlx_revision) {
    errors.push('result model/runtime identity must match the manifest');
  }
  if (result.fixture?.id !== fixture.id ||
      result.fixture?.transcript_sha256 !== sha256(fixture.transcript) ||
      result.fixture?.reference_visible_to_model !== false ||
      result.procedure?.evaluation_reference_visible_to_model !== false ||
      result.procedure?.generation?.prompt_sha256 !== contract.prompt_sha256) {
    errors.push('result must preserve the hidden-reference prompt boundary');
  }
  for (const field of [
    'model_load_ms',
    'first_token_ms',
    'complete_generation_ms',
    'prompt_tokens',
    'generation_tokens',
    'prompt_tokens_per_second',
    'generation_tokens_per_second',
    'maximum_resident_set_size_bytes',
    'mlx_peak_memory_bytes',
    'model_snapshot_bytes',
  ]) {
    if (!isPositive(result.measurements?.[field])) {
      errors.push(`measurements.${field} must be positive`);
    }
  }
  const tokenIds = result.output?.token_ids;
  if (!Array.isArray(tokenIds) || tokenIds.length !== result.measurements?.generation_tokens ||
      tokenIds.length !== result.streaming?.update_count ||
      tokenIds.some((token) => !Number.isInteger(token) || token < 0)) {
    errors.push('token and streaming update counts must agree');
  }
  if (sha256(result.output?.text ?? '') !== result.output?.text_sha256 ||
      !(result.output?.text?.length > 0)) {
    errors.push('output text must be non-empty and digest-pinned');
  }
  if (result.measurements?.deterministic_repeat?.text_identical !== true ||
      result.measurements?.deterministic_repeat?.token_ids_identical !== true) {
    errors.push('the deterministic repetition must preserve text and token ids');
  }
  if (result.streaming?.supported !== true ||
      result.streaming?.finish_reason !== 'stop' ||
      result.streaming?.cancellation?.observed_tokens !==
        result.streaming?.cancellation?.requested_after_tokens ||
      result.streaming?.cancellation?.process_remained_usable !== true) {
    errors.push('streaming and cooperative cancellation evidence is incomplete');
  }
  if (result.output?.gates?.nonempty !== true ||
      result.output?.gates?.case_and_punctuation_insensitive_lexical_invariant !== false ||
      result.output?.gates?.accepted !== false ||
      result.conclusion?.reference_runtime_executed !== true ||
      result.conclusion?.surface_candidate_accepted !== false ||
      result.conclusion?.product_runtime_accepted !== false) {
    errors.push('the rejected output must not promote the model or runtime');
  }
  const inputWer = result.output?.development_quality?.input_wer;
  const outputWer = result.output?.development_quality?.output_wer;
  if (Math.abs(inputWer - 1 / 39) > 1e-12 ||
      Math.abs(outputWer - 2 / 39) > 1e-12 ||
      !(outputWer > inputWer)) {
    errors.push('the diagnostic WER regression must remain explicit');
  }
  return errors;
}

const [manifest, fixture, result, promptBytes] = await Promise.all([
  readJson(manifestPath),
  readJson(fixturePath),
  readJson(resultPath),
  readFile(promptPath),
]);

const failures = validate(manifest, fixture, result, promptBytes);
if (process.argv.includes('--self-test')) {
  const invalid = structuredClone(result);
  invalid.output.gates.accepted = true;
  if (validate(manifest, fixture, invalid, promptBytes).length === 0) {
    failures.push('self-test failed to reject an incorrectly accepted output');
  }
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log(
    `text-generation MLX reference: ${result.measurements.generation_tokens} token(s), ` +
    `${result.measurements.generation_tokens_per_second.toFixed(1)} token/s, ` +
    'surface output rejected as expected',
  );
}
