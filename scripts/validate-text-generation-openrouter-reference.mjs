#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const referenceRoot = join(repoRoot, 'spikes/text-generation-openrouter-reference');
const candidateRoot = join(referenceRoot, 'candidates');
const experimentRoot = join(referenceRoot, 'experiments');
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

function containsSecret(value) {
  const serialized = JSON.stringify(value);
  return /sk-or-v1-|openrouter_api_key|"authorization"|"api[_-]?key"/iu
    .test(serialized);
}

function normalizedWords(text) {
  return [...text.normalize('NFC').toLocaleLowerCase('und')]
    .map((character) => /[\p{L}\p{N}]/u.test(character) ? character : ' ')
    .join('')
    .split(/\s+/u)
    .filter(Boolean);
}

function wordDiff(source, target) {
  const sourceWords = normalizedWords(source);
  const targetWords = normalizedWords(target);
  const distances = Array.from(
    { length: sourceWords.length + 1 },
    () => Array(targetWords.length + 1).fill(0),
  );
  for (let index = 0; index <= sourceWords.length; index += 1) {
    distances[index][0] = index;
  }
  for (let index = 0; index <= targetWords.length; index += 1) {
    distances[0][index] = index;
  }
  for (let sourceIndex = 1; sourceIndex <= sourceWords.length; sourceIndex += 1) {
    for (let targetIndex = 1; targetIndex <= targetWords.length; targetIndex += 1) {
      distances[sourceIndex][targetIndex] = Math.min(
        distances[sourceIndex - 1][targetIndex] + 1,
        distances[sourceIndex][targetIndex - 1] + 1,
        distances[sourceIndex - 1][targetIndex - 1] + Number(
          sourceWords[sourceIndex - 1] !== targetWords[targetIndex - 1],
        ),
      );
    }
  }
  const operations = [];
  let sourceIndex = sourceWords.length;
  let targetIndex = targetWords.length;
  while (sourceIndex > 0 || targetIndex > 0) {
    if (sourceIndex > 0 && targetIndex > 0 &&
        sourceWords[sourceIndex - 1] === targetWords[targetIndex - 1] &&
        distances[sourceIndex][targetIndex] ===
          distances[sourceIndex - 1][targetIndex - 1]) {
      sourceIndex -= 1;
      targetIndex -= 1;
    } else if (sourceIndex > 0 && targetIndex > 0 &&
        distances[sourceIndex][targetIndex] ===
          distances[sourceIndex - 1][targetIndex - 1] + 1) {
      operations.push({
        type: 'replace',
        input_index: sourceIndex - 1,
        output_index: targetIndex - 1,
        input: sourceWords[sourceIndex - 1],
        output: targetWords[targetIndex - 1],
      });
      sourceIndex -= 1;
      targetIndex -= 1;
    } else if (sourceIndex > 0 &&
        distances[sourceIndex][targetIndex] ===
          distances[sourceIndex - 1][targetIndex] + 1) {
      operations.push({
        type: 'delete',
        input_index: sourceIndex - 1,
        output_index: targetIndex,
        input: sourceWords[sourceIndex - 1],
        output: null,
      });
      sourceIndex -= 1;
    } else {
      operations.push({
        type: 'insert',
        input_index: sourceIndex,
        output_index: targetIndex - 1,
        input: null,
        output: targetWords[targetIndex - 1],
      });
      targetIndex -= 1;
    }
  }
  operations.reverse();
  return {
    input_word_count: sourceWords.length,
    output_word_count: targetWords.length,
    edit_distance: distances.at(-1).at(-1),
    operations,
  };
}

function countSequence(words, sequence) {
  if (sequence.length === 0) return 0;
  let count = 0;
  for (let index = 0; index <= words.length - sequence.length; index += 1) {
    if (sequence.every((word, offset) => words[index + offset] === word)) count += 1;
  }
  return count;
}

function pairCounts(pairs) {
  const counts = new Map();
  for (const pair of pairs) {
    const key = JSON.stringify(pair);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

function equalCounts(left, right) {
  return left.size === right.size &&
    [...left].every(([key, value]) => right.get(key) === value);
}

function externalAudit(fixture, text, edits) {
  const diff = wordDiff(fixture.transcript, text);
  const operations = pairCounts(diff.operations.map((operation) => [
    normalizedWords(operation.input ?? ''),
    normalizedWords(operation.output ?? ''),
  ]));
  const reported = Array.isArray(edits)
    ? pairCounts(edits.map((edit) => [
      normalizedWords(edit.original ?? ''),
      normalizedWords(edit.replacement ?? ''),
    ]).filter(([source, target]) =>
      JSON.stringify(source) !== JSON.stringify(target)))
    : new Map();
  const inputWords = normalizedWords(fixture.transcript);
  const outputWords = normalizedWords(text);
  return {
    diff,
    lexicalEditsFullyReported: equalCounts(operations, reported),
    protectedSpansUnchanged: (fixture.protected_spans ?? []).every((span) => {
      const words = normalizedWords(span);
      return countSequence(inputWords, words) === countSequence(outputWords, words);
    }),
    reportedLexicalEditCount: [...reported.values()]
      .reduce((sum, value) => sum + value, 0),
  };
}

function validateCandidateShape(candidate) {
  const errors = [];
  const model = candidate.model;
  const gateway = candidate.gateway;
  const policy = gateway?.request_policy;
  const defaults = candidate.request_defaults;
  if (candidate.schema_version !== '1.0.0' ||
      candidate.task !== 'text-generation' ||
      !/^[a-z0-9][a-z0-9.-]+$/.test(candidate.id ?? '') ||
      !(candidate.candidate_role?.length > 0)) {
    errors.push('candidate identity is invalid');
  }
  if (!(model?.requested_id?.includes('/')) ||
      !Array.isArray(model?.allowed_response_ids) ||
      !model.allowed_response_ids.includes(model.requested_id) ||
      !(model?.upstream_identity?.length > 0) ||
      !(model?.provider_model_identity?.length > 0) ||
      !['open', 'proprietary'].includes(model?.weights) ||
      !(model?.license?.length > 0) ||
      !Number.isInteger(model?.context_tokens) || model.context_tokens <= 0 ||
      !(model?.identity_limit?.length > 0)) {
    errors.push('model identity and ownership are incomplete');
  }
  if (gateway?.name !== 'OpenRouter' ||
      gateway?.endpoint !== 'https://openrouter.ai/api/v1/chat/completions' ||
      !/^\d{4}-\d{2}-\d{2}$/.test(gateway?.catalog_captured_at ?? '') ||
      !/^[a-z0-9-]+$/.test(gateway?.provider_slug ?? '') ||
      !(gateway?.provider_name?.length > 0) ||
      !(gateway?.endpoint_tag?.length > 0) ||
      !(gateway?.endpoint_identity?.length > 0) ||
      !Array.isArray(gateway?.supported_parameters) ||
      !gateway.supported_parameters.includes('response_format') ||
      !gateway.supported_parameters.includes('structured_outputs')) {
    errors.push('gateway and endpoint identity are incomplete');
  }
  if (!Array.isArray(policy?.only) || policy.only.length !== 1 ||
      policy.only[0] !== gateway?.provider_slug ||
      policy?.allow_fallbacks !== false ||
      policy?.require_parameters !== true ||
      policy?.data_collection !== 'deny' || policy?.zdr !== true) {
    errors.push('provider routing must be pinned, no-fallback, and zero-retention');
  }
  const pricing = Object.values(gateway?.pricing_snapshot_usd_per_token ?? {});
  if (pricing.length < 2 || pricing.some((value) =>
    typeof value !== 'string' || !Number.isFinite(Number(value)) || Number(value) < 0)) {
    errors.push('pricing snapshot is invalid');
  }
  const reasoningValid = defaults?.reasoning === null ||
    (defaults?.reasoning?.effort === 'none' && defaults.reasoning.exclude === true);
  if (![0, null].includes(defaults?.temperature) ||
      ![0, null].includes(defaults?.seed) || !reasoningValid ||
      defaults?.response_format?.type !== 'json_schema' ||
      defaults?.response_format?.schema_id !==
        'bounded-transcript-correction-v1' ||
      defaults?.response_format?.strict !== true) {
    errors.push('request defaults are not conservative or strictly structured');
  }
  if (!(candidate.claim_limit?.length > 0) || containsSecret(candidate)) {
    errors.push('candidate claim limit is missing or secret material is present');
  }
  return errors;
}

function validateExperimentShape(experiment) {
  const errors = [];
  const contract = experiment.generation_contract;
  const result = experiment.result_contract;
  if (experiment.schema_version !== '1.0.0' || !(experiment.id?.length > 0)) {
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
  if (contract?.prompt_id !== 'conservative-error-profile-v1' ||
      contract?.policy_mode !== 'bounded-lexical' ||
      contract?.render_mode !== 'template' ||
      contract?.output_mode !== 'json-edits' ||
      contract?.structured_output_contract !==
        'bounded-transcript-correction-v1' ||
      !Array.isArray(contract?.context_fields) ||
      !contract.context_fields.includes('transcript') ||
      !digest.test(contract?.prompt_sha256 ?? '') ||
      !Number.isInteger(contract?.max_tokens) || contract.max_tokens <= 0 ||
      !Number.isInteger(contract?.timeout_ms) || contract.timeout_ms < 1000) {
    errors.push('generation contract is invalid');
  }
  if (!(result?.run_id?.length > 0) ||
      result?.purpose !== 'development-quality-screen' ||
      typeof result?.result_required !== 'boolean') {
    errors.push('result contract is invalid');
  }
  if (containsSecret(experiment)) errors.push('secret material is present');
  return errors;
}

async function validateCandidate(candidate, path) {
  return validateCandidateShape(candidate).map((error) => `${path}: ${error}`);
}

async function validateExperiment(experiment, path) {
  const errors = validateExperimentShape(experiment);
  let candidate;
  let fixture;
  let promptBytes;
  try {
    [candidate, fixture, promptBytes] = await Promise.all([
      readJson(repositoryPath(experiment.model_manifest_path)),
      readJson(repositoryPath(experiment.result_contract.fixture_path)),
      readFile(repositoryPath(experiment.generation_contract.prompt_path)),
    ]);
  } catch (error) {
    errors.push(`cannot read experiment input: ${error.message}`);
    return errors.map((error) => `${path}: ${error}`);
  }
  if (candidate.task !== 'text-generation' ||
      sha256(promptBytes) !== experiment.generation_contract.prompt_sha256 ||
      fixture.development_only !== true ||
      fixture.inference_policy?.reference_visible_to_model !== false ||
      fixture.inference_policy?.prompt_id !==
        experiment.generation_contract.prompt_id) {
    errors.push('candidate, prompt, or hidden-reference fixture is invalid');
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
    if (containsSecret(run)) errors.push('result contains secret material');
    if (run.schema_version !== '1.0.0' ||
        run.run_id !== experiment.result_contract.run_id ||
        run.purpose !== experiment.result_contract.purpose ||
        !revision.test(run.source_revision ?? '') ||
        run.candidate?.task !== 'text-generation' ||
        run.candidate?.manifest_id !== candidate.id ||
        run.candidate?.role !== candidate.candidate_role ||
        run.candidate?.model?.requested_id !== candidate.model.requested_id ||
        run.candidate?.gateway?.name !== 'OpenRouter' ||
        run.candidate?.gateway?.provider_slug !== candidate.gateway.provider_slug ||
        run.candidate?.gateway?.provider_name !== candidate.gateway.provider_name ||
        run.candidate?.gateway?.served_provider !== candidate.gateway.provider_name ||
        !candidate.model.allowed_response_ids.includes(
          run.candidate?.gateway?.served_model,
        )) {
      errors.push('result candidate or pinned provider identity does not match');
    }
    if (run.fixture?.id !== fixture.id ||
        run.fixture?.transcript_sha256 !== sha256(fixture.transcript) ||
        run.fixture?.reference_visible_to_model !== false ||
        run.procedure?.experiment_id !== experiment.id ||
        run.procedure?.generation?.prompt_sha256 !==
          experiment.generation_contract.prompt_sha256 ||
        run.procedure?.evaluation_reference_visible_to_model !== false ||
        run.procedure?.gateway_request?.provider?.allow_fallbacks !== false ||
        run.procedure?.gateway_request?.provider?.data_collection !== 'deny' ||
        run.procedure?.gateway_request?.provider?.zdr !== true) {
      errors.push('result violates the hidden-reference or routing boundary');
    }

    const requests = run.measurements?.requests;
    if (!Array.isArray(requests) || requests.length !== 2 ||
        requests.some((request) =>
          !(request?.response_id?.length > 0) ||
          request.served_provider !== candidate.gateway.provider_name ||
          !candidate.model.allowed_response_ids.includes(request.served_model) ||
          !(request.client_observed_complete_ms > 0) ||
          !Number.isInteger(request.usage?.prompt_tokens) ||
          request.usage.prompt_tokens <= 0 ||
          !Number.isInteger(request.usage?.completion_tokens) ||
          request.usage.completion_tokens <= 0 ||
          !Number.isFinite(request.usage?.cost_usd) ||
          request.usage.cost_usd < 0)) {
      errors.push('two complete provider-pinned request records are required');
    } else {
      const totalCost = requests.reduce(
        (sum, request) => sum + request.usage.cost_usd,
        0,
      );
      if (Math.abs(totalCost - run.measurements.aggregate.total_cost_usd) > 1e-12) {
        errors.push('aggregate cost does not equal the two request costs');
      }
    }
    if (typeof run.measurements?.deterministic_repeat?.text_identical !== 'boolean' ||
        !run.measurements?.deterministic_repeat?.requirement?.includes(
          'observation-only',
        )) {
      errors.push('remote repeat equality must remain an observation, not a gate');
    }

    const outputText = run.output?.text ?? '';
    const rawText = run.output?.raw_text ?? '';
    const audit = externalAudit(fixture, outputText, run.output?.reported_edits);
    const expectedAccepted = outputText.length > 0 &&
      run.output?.parser?.valid === true &&
      audit.lexicalEditsFullyReported && audit.protectedSpansUnchanged;
    if (sha256(outputText) !== run.output?.text_sha256 ||
        sha256(rawText) !== run.output?.raw_text_sha256 ||
        run.output?.audit?.lexical_edits_fully_reported !==
          audit.lexicalEditsFullyReported ||
        run.output?.audit?.protected_spans_unchanged !==
          audit.protectedSpansUnchanged ||
        run.output?.audit?.reported_lexical_edit_count !==
          audit.reportedLexicalEditCount ||
        JSON.stringify(run.output?.lexical_diff) !== JSON.stringify(audit.diff) ||
        run.output?.gates?.policy_mode !== 'bounded-lexical' ||
        run.output?.gates?.accepted !== expectedAccepted ||
        run.output?.gates?.quality_accepted !== false) {
      errors.push('output digest, lexical audit, or mechanical gate is invalid');
    }
    if (!Number.isFinite(run.output?.diagnostic_wer?.input) ||
        !Number.isFinite(run.output?.diagnostic_wer?.output) ||
        run.output?.diagnostic_wer?.reference_status !== fixture.gold_status) {
      errors.push('development-only diagnostic WER is incomplete');
    }
    if (run.conclusion?.remote_reference_executed !== true ||
        run.conclusion?.development_output_contract_accepted !==
          expectedAccepted ||
        run.conclusion?.model_quality_selected !== false ||
        run.conclusion?.product_runtime_accepted !== false) {
      errors.push('result must remain a non-promoting remote quality screen');
    }
  }
  return errors.map((error) => `${path}: ${error}`);
}

async function jsonPaths(root) {
  const entries = await readdir(root, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => join(root, entry.name))
    .sort();
}

const [candidatePaths, experimentPaths] = await Promise.all([
  jsonPaths(candidateRoot),
  jsonPaths(experimentRoot),
]);
const [candidates, experiments] = await Promise.all([
  Promise.all(candidatePaths.map(readJson)),
  Promise.all(experimentPaths.map(readJson)),
]);
const failures = (await Promise.all([
  ...candidates.map((candidate, index) =>
    validateCandidate(candidate, relative(repoRoot, candidatePaths[index]))),
  ...experiments.map((experiment, index) =>
    validateExperiment(experiment, relative(repoRoot, experimentPaths[index]))),
])).flat();

if (new Set(candidates.map(({ id }) => id)).size !== candidates.length) {
  failures.push('OpenRouter candidate ids must be unique');
}
if (new Set(experiments.map(({ id }) => id)).size !== experiments.length ||
    new Set(experiments.map(({ result_contract: result }) => result.run_id)).size !==
      experiments.length) {
  failures.push('OpenRouter experiment and run ids must be unique');
}

if (process.argv.includes('--self-test')) {
  const fallback = structuredClone(candidates[0]);
  fallback.gateway.request_policy.allow_fallbacks = true;
  if (validateCandidateShape(fallback).length === 0) {
    failures.push('self-test failed to reject provider fallback');
  }
  const secret = structuredClone(candidates[0]);
  secret.gateway.api_key = 'sk-or-v1-self-test';
  if (validateCandidateShape(secret).length === 0) {
    failures.push('self-test failed to reject secret material');
  }
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log(
    `OpenRouter text-generation reference: ${candidates.length} candidate(s), ` +
    `${experiments.length} experiment(s), ` +
    `${experiments.filter((experiment) =>
      experiment.result_contract.result_required).length} required result(s)`,
  );
}
