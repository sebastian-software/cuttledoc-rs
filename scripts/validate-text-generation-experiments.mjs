#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const experimentRoot = join(
  repoRoot,
  'spikes/text-generation-mlx-reference/experiments',
);
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const digest = /^[0-9a-f]{64}$/;
const revision = /^[0-9a-f]{40}$/;
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

function normalizedWords(text) {
  return [...text.normalize('NFC').toLocaleLowerCase('und')]
    .map((character) => /[\p{L}\p{N}]/u.test(character) ? character : ' ')
    .join('')
    .split(/\s+/u)
    .filter(Boolean);
}

function wordDistance(source, target) {
  const sourceWords = normalizedWords(source);
  const targetWords = normalizedWords(target);
  let previous = Array.from({ length: targetWords.length + 1 }, (_, index) => index);
  for (let sourceIndex = 1; sourceIndex <= sourceWords.length; sourceIndex += 1) {
    const current = [sourceIndex];
    for (let targetIndex = 1; targetIndex <= targetWords.length; targetIndex += 1) {
      current.push(Math.min(
        current.at(-1) + 1,
        previous[targetIndex] + 1,
        previous[targetIndex - 1] +
          Number(sourceWords[sourceIndex - 1] !== targetWords[targetIndex - 1]),
      ));
    }
    previous = current;
  }
  return previous.at(-1);
}

function validateShape(experiment) {
  const errors = [];
  const contract = experiment.generation_contract;
  const result = experiment.result_contract;
  if (experiment.schema_version !== '1.0.0' ||
      !(experiment.id?.length > 0)) {
    errors.push('experiment identity is invalid');
  }
  for (const path of [
    experiment.model_manifest_path,
    contract?.prompt_path,
    result?.fixture_path,
    result?.result_path,
  ]) {
    try {
      repositoryPath(path);
    } catch (error) {
      errors.push(error.message);
    }
  }
  const promptShapeValid =
    (contract?.prompt_id === 'historical-cuttledoc-v2' &&
     contract?.policy_mode === 'historical-control' &&
     contract?.render_mode === 'append-transcript' &&
     contract?.output_mode === 'plain-text') ||
    (contract?.prompt_id === 'conservative-error-profile-v1' &&
     contract?.policy_mode === 'bounded-lexical' &&
     contract?.render_mode === 'template' &&
     contract?.output_mode === 'json-edits');
  if (!promptShapeValid ||
      !Array.isArray(contract?.context_fields) ||
      contract.context_fields.length === 0 ||
      !digest.test(contract?.prompt_sha256 ?? '') ||
      contract?.chat_template_options?.enable_thinking !== false ||
      contract?.seed !== 0 || contract?.temperature !== 0 ||
      contract?.stream !== true || !Number.isInteger(contract?.max_tokens) ||
      contract.max_tokens <= 0 ||
      !Number.isInteger(contract?.cancellation_probe_after_tokens) ||
      contract.cancellation_probe_after_tokens <= 0 ||
      contract.cancellation_probe_after_tokens >= contract.max_tokens) {
    errors.push('deterministic generation contract is invalid');
  }
  if (!(result?.run_id?.length > 0) ||
      result?.purpose !== 'development-runtime-probe' ||
      typeof result?.result_required !== 'boolean') {
    errors.push('result contract is invalid');
  }
  return errors;
}

async function validateExperiment(experiment, path) {
  const errors = validateShape(experiment);
  let model;
  let fixture;
  let promptBytes;
  try {
    [model, fixture, promptBytes] = await Promise.all([
      readJson(repositoryPath(experiment.model_manifest_path)),
      readJson(repositoryPath(experiment.result_contract.fixture_path)),
      readFile(repositoryPath(experiment.generation_contract.prompt_path)),
    ]);
  } catch (error) {
    errors.push(`cannot read experiment input: ${error.message}`);
    return errors.map((error) => `${path}: ${error}`);
  }
  if (model.task !== 'text-generation' ||
      sha256(promptBytes) !== experiment.generation_contract.prompt_sha256 ||
      fixture.development_only !== true ||
      fixture.inference_policy?.reference_visible_to_model !== false) {
    errors.push('model, prompt, or hidden-reference fixture input is invalid');
  }

  let run;
  try {
    run = await readJson(repositoryPath(experiment.result_contract.result_path));
  } catch (error) {
    if (experiment.result_contract.result_required) {
      errors.push(`required result cannot be read: ${error.message}`);
    }
  }
  if (run !== undefined) {
    const outputWords = normalizedWords(run.output?.text ?? '');
    const inputWords = normalizedWords(fixture.transcript);
    if (run.run_id !== experiment.result_contract.run_id ||
        run.purpose !== experiment.result_contract.purpose ||
        !revision.test(run.source_revision ?? '') ||
        run.candidate?.manifest_id !== model.id ||
        run.candidate?.role !== model.candidate_role ||
        run.candidate?.model?.revision !== model.conversion.revision ||
        run.procedure?.experiment_id !== experiment.id ||
        run.procedure?.generation?.prompt_sha256 !==
          experiment.generation_contract.prompt_sha256 ||
        run.fixture?.id !== fixture.id ||
        run.fixture?.reference_visible_to_model !== false ||
        run.procedure?.evaluation_reference_visible_to_model !== false) {
      errors.push('result identity or hidden-reference boundary does not match');
    }
    const outputNonempty = run.output?.text?.length > 0;
    const invariant = inputWords.join('\n') === outputWords.join('\n');
    const expectedAccepted = experiment.generation_contract.policy_mode ===
      'historical-control'
      ? outputNonempty
      : outputNonempty && run.output?.parser?.valid === true &&
        run.output?.audit?.lexical_edits_fully_reported === true &&
        run.output?.audit?.protected_spans_unchanged === true;
    if ((experiment.generation_contract.policy_mode === 'historical-control' &&
         !outputNonempty) ||
        sha256(run.output.text) !== run.output.text_sha256 ||
        run.output?.gates?.policy_mode !==
          experiment.generation_contract.policy_mode ||
        run.output?.gates?.accepted !== expectedAccepted ||
        run.output?.gates?.quality_accepted !== false ||
        run.output?.gates?.case_and_punctuation_insensitive_lexical_invariant !==
          invariant ||
        run.output?.lexical_diff?.input_word_count !== inputWords.length ||
        run.output?.lexical_diff?.output_word_count !== outputWords.length ||
        run.output?.lexical_diff?.edit_distance !==
          wordDistance(fixture.transcript, run.output.text) ||
        run.output?.lexical_diff?.operations?.length !==
          run.output?.lexical_diff?.edit_distance) {
      errors.push('output contract or lexical diff is invalid');
    }
    if (experiment.generation_contract.output_mode === 'json-edits' &&
        (!(run.output?.raw_text?.length > 0) ||
         sha256(run.output.raw_text) !== run.output.raw_text_sha256 ||
         typeof run.output?.parser?.valid !== 'boolean' ||
         typeof run.output?.audit?.lexical_edits_fully_reported !== 'boolean' ||
         typeof run.output?.audit?.protected_spans_unchanged !== 'boolean')) {
      errors.push('structured output parser and audit evidence is incomplete');
    }
    if (run.measurements?.deterministic_repeat?.text_identical !== true ||
        run.measurements?.deterministic_repeat?.token_ids_identical !== true ||
        run.streaming?.cancellation?.observed_tokens !==
          run.streaming?.cancellation?.requested_after_tokens ||
        run.streaming?.cancellation?.process_remained_usable !== true ||
        run.conclusion?.reference_runtime_executed !== true ||
        run.conclusion?.surface_candidate_accepted !== null ||
        run.conclusion?.development_output_contract_accepted !== expectedAccepted ||
        run.conclusion?.model_quality_selected !== false ||
        run.conclusion?.product_runtime_accepted !== false) {
      errors.push('runtime evidence must remain deterministic and non-promoting');
    }
  }
  return errors.map((error) => `${path}: ${error}`);
}

const entries = await readdir(experimentRoot, { withFileTypes: true });
const paths = entries
  .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
  .map((entry) => join(experimentRoot, entry.name))
  .sort();
const experiments = await Promise.all(paths.map(readJson));
const failures = (await Promise.all(experiments.map((experiment, index) =>
  validateExperiment(experiment, relative(repoRoot, paths[index]))))).flat();

if (new Set(experiments.map(({ id }) => id)).size !== experiments.length ||
    new Set(experiments.map(({ result_contract: result }) => result.run_id)).size !==
      experiments.length) {
  failures.push('experiment and run ids must be unique');
}

if (process.argv.includes('--self-test')) {
  const invalid = structuredClone(experiments[0]);
  invalid.generation_contract.prompt_sha256 = '0'.repeat(64);
  const invalidFailures = await validateExperiment(invalid, 'self-test');
  if (invalidFailures.length === 0) {
    failures.push('self-test failed to reject prompt digest drift');
  }
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log(
    `text-generation experiments: ${experiments.length} pinned experiment(s), ` +
    `${experiments.filter((experiment) => experiment.result_contract.result_required).length} required result(s)`,
  );
}
