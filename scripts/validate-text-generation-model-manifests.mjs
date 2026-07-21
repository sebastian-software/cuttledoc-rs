#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const spikeRoot = join(repoRoot, 'spikes/text-generation-mlx-reference');
const manifestRoot = join(spikeRoot, 'model-manifest.json');
const candidateRoot = join(spikeRoot, 'candidates');
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const gitRevision = /^[0-9a-f]{40}$/;
const artifactDigest = /^[0-9a-f]{64}$/;
const isPositive = (value) => Number.isFinite(value) && value > 0;

const readJson = async (path) => JSON.parse(await readFile(path, 'utf8'));

function repositoryPath(path) {
  if (typeof path !== 'string' || path.length === 0 || isAbsolute(path)) {
    throw new Error(`invalid repository-relative path: ${path}`);
  }
  const absolute = resolve(repoRoot, path);
  const fromRoot = relative(repoRoot, absolute);
  if (fromRoot.startsWith('..') || isAbsolute(fromRoot)) {
    throw new Error(`path leaves the repository: ${path}`);
  }
  return absolute;
}

function validateManifestShape(manifest) {
  const errors = [];
  const contract = manifest.generation_contract;
  const result = manifest.result_contract;

  if (manifest.schema_version !== '1.0.0' ||
      manifest.task !== 'text-generation' ||
      !/^[a-z0-9][a-z0-9.-]+$/.test(manifest.id ?? '') ||
      !(manifest.candidate_role?.length > 0)) {
    errors.push('manifest identity is invalid');
  }
  for (const revision of [
    manifest.source?.observed_revision,
    manifest.conversion?.revision,
    manifest.reference_runtime?.revision,
    manifest.reference_runtime?.mlx_revision,
  ]) {
    if (!gitRevision.test(revision ?? '')) {
      errors.push(`invalid pinned revision: ${revision}`);
    }
  }
  if (!(manifest.source?.repository?.length > 0) ||
      !(manifest.source?.license?.length > 0) ||
      !(manifest.conversion?.repository?.length > 0) ||
      !(manifest.conversion?.license?.length > 0) ||
      manifest.reference_runtime?.license !== 'MIT' ||
      manifest.reference_runtime?.boundary !==
        'Pinned Python reference over official Apple MLX; not an accepted product dependency or stable API') {
    errors.push('source, conversion, and reference-runtime ownership is incomplete');
  }

  const artifacts = manifest.artifacts ?? [];
  const paths = artifacts.map(({ path }) => path);
  if (artifacts.length === 0 || new Set(paths).size !== paths.length ||
      artifacts.some(({ path, bytes, sha256: digest }) =>
        typeof path !== 'string' || path.length === 0 || path.startsWith('/') ||
        path.split('/').includes('..') || !Number.isInteger(bytes) || bytes <= 0 ||
        !artifactDigest.test(digest ?? ''))) {
    errors.push('artifact paths, byte counts, or digests are invalid');
  }
  if (artifacts.reduce((sum, artifact) => sum + artifact.bytes, 0) !==
      manifest.conversion?.snapshot_bytes) {
    errors.push('artifact byte counts do not equal conversion.snapshot_bytes');
  }

  if (!(contract?.prompt_id?.length > 0) ||
      !artifactDigest.test(contract?.prompt_sha256 ?? '') ||
      contract?.policy_mode !== 'surface-only' ||
      contract?.render_mode !== 'template' ||
      contract?.temperature !== 0 || contract?.seed !== 0 ||
      contract?.stream !== true || !Number.isInteger(contract?.max_tokens) ||
      contract.max_tokens <= 0 ||
      !Number.isInteger(contract?.cancellation_probe_after_tokens) ||
      contract.cancellation_probe_after_tokens <= 0 ||
      contract.cancellation_probe_after_tokens >= contract.max_tokens ||
      typeof contract?.chat_template_options !== 'object' ||
      contract.chat_template_options === null) {
    errors.push('generation contract must be deterministic and complete');
  }
  try {
    repositoryPath(contract?.prompt_path);
    repositoryPath(result?.fixture_path);
    repositoryPath(result?.result_path);
  } catch (error) {
    errors.push(error.message);
  }
  if (!(result?.run_id?.length > 0) ||
      result?.purpose !== 'development-runtime-probe' ||
      typeof result?.result_required !== 'boolean') {
    errors.push('result contract is invalid');
  }
  return errors;
}

async function validateManifest(manifest, manifestPath) {
  const errors = validateManifestShape(manifest);
  const contract = manifest.generation_contract;
  const resultContract = manifest.result_contract;
  let promptBytes;
  let fixture;

  try {
    [promptBytes, fixture] = await Promise.all([
      readFile(repositoryPath(contract.prompt_path)),
      readJson(repositoryPath(resultContract.fixture_path)),
    ]);
  } catch (error) {
    errors.push(`cannot read prompt or fixture: ${error.message}`);
    return errors.map((error) => `${manifestPath}: ${error}`);
  }

  if (sha256(promptBytes) !== contract.prompt_sha256) {
    errors.push('prompt digest does not match generation contract');
  }
  if (fixture.development_only !== true ||
      fixture.inference_policy?.reference_visible_to_model !== false ||
      fixture.inference_policy?.prompt_id !== contract.prompt_id) {
    errors.push('fixture must be development-only with evaluation reference hidden');
  }

  const resultPath = repositoryPath(resultContract.result_path);
  let run;
  try {
    run = await readJson(resultPath);
  } catch (error) {
    if (resultContract.result_required) {
      errors.push(`required result cannot be read: ${error.message}`);
    }
  }

  if (run !== undefined) {
    if (run.schema_version !== '1.0.0' ||
        run.run_id !== resultContract.run_id ||
        run.purpose !== resultContract.purpose ||
        !gitRevision.test(run.source_revision ?? '') ||
        run.candidate?.task !== 'text-generation' ||
        run.candidate?.model?.repository !== manifest.conversion.repository ||
        run.candidate?.model?.revision !== manifest.conversion.revision ||
        run.candidate?.runtime?.revision !== manifest.reference_runtime.revision ||
        run.candidate?.runtime?.mlx_revision !== manifest.reference_runtime.mlx_revision) {
      errors.push('result identity does not match the manifest');
    }
    if (run.candidate?.manifest_id !== undefined &&
        (run.candidate.manifest_id !== manifest.id ||
         run.candidate.role !== manifest.candidate_role)) {
      errors.push('result candidate role does not match the manifest');
    }
    if (run.fixture?.id !== fixture.id ||
        run.fixture?.transcript_sha256 !== sha256(fixture.transcript) ||
        run.fixture?.reference_visible_to_model !== false ||
        run.procedure?.evaluation_reference_visible_to_model !== false ||
        run.procedure?.generation?.prompt_sha256 !== contract.prompt_sha256) {
      errors.push('result violates the hidden-reference prompt boundary');
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
      if (!isPositive(run.measurements?.[field])) {
        errors.push(`result measurements.${field} must be positive`);
      }
    }
    if (run.measurements?.model_snapshot_bytes !== manifest.conversion.snapshot_bytes ||
        sha256(run.output?.text ?? '') !== run.output?.text_sha256 ||
        !Array.isArray(run.output?.token_ids) ||
        run.output.token_ids.length !== run.measurements?.generation_tokens ||
        run.output.token_ids.length !== run.streaming?.update_count ||
        run.measurements?.deterministic_repeat?.text_identical !== true ||
        run.measurements?.deterministic_repeat?.token_ids_identical !== true ||
        run.streaming?.supported !== true ||
        run.streaming?.cancellation?.observed_tokens !==
          run.streaming?.cancellation?.requested_after_tokens ||
        run.streaming?.cancellation?.process_remained_usable !== true ||
        run.conclusion?.reference_runtime_executed !== true ||
        run.conclusion?.product_runtime_accepted !== false) {
      errors.push('result runtime, streaming, or non-promotion evidence is incomplete');
    }
    if (contract.prompt_id === 'surface-only-v1') {
      const accepted = run.output?.gates?.nonempty === true &&
        run.output?.gates?.case_and_punctuation_insensitive_lexical_invariant === true;
      if (run.output?.gates?.accepted !== accepted ||
          run.conclusion?.surface_candidate_accepted !== accepted) {
        errors.push('surface-only acceptance must equal the external lexical gate');
      }
    }
  }

  return errors.map((error) => `${manifestPath}: ${error}`);
}

async function candidateManifestPaths() {
  const paths = [manifestRoot];
  try {
    const entries = await readdir(candidateRoot, { withFileTypes: true });
    paths.push(...entries
      .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
      .map((entry) => join(candidateRoot, entry.name))
      .sort());
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  return paths;
}

const manifestPaths = await candidateManifestPaths();
const manifests = await Promise.all(manifestPaths.map(readJson));
const failures = (await Promise.all(manifests.map((manifest, index) =>
  validateManifest(manifest, relative(repoRoot, manifestPaths[index]))))).flat();

const ids = manifests.map(({ id }) => id);
if (new Set(ids).size !== ids.length) {
  failures.push('text-generation model manifest ids must be unique');
}

if (process.argv.includes('--self-test')) {
  const invalid = structuredClone(manifests[0]);
  invalid.artifacts.push(structuredClone(invalid.artifacts[0]));
  if (validateManifestShape(invalid).length === 0) {
    failures.push('self-test failed to reject a duplicate artifact path');
  }
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log(
    `text-generation model manifests: ${manifests.length} pinned candidate(s), ` +
    `${manifests.filter((manifest) => manifest.result_contract.result_required).length} required result(s)`,
  );
}
