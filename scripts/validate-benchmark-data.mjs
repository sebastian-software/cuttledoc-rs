#!/usr/bin/env node

import { readFile, readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const schemaVersion = '1.0.0';

function at(object, path) {
  return path.split('.').reduce((value, key) => value?.[key], object);
}

function isMetric(value, integer = false) {
  return (
    value === null ||
    (typeof value === 'number' && Number.isFinite(value) && value >= 0 && (!integer || Number.isInteger(value)))
  );
}

function validateRun(run, fixtures) {
  const errors = [];
  const requiredStrings = [
    'run_id',
    'captured_at',
    'candidate.model.id',
    'candidate.model.revision',
    'candidate.model.quantization',
    'candidate.model.license',
    'candidate.runtime.id',
    'candidate.runtime.version',
    'candidate.runtime.boundary',
    'host.id',
    'host.chip',
    'host.os',
    'engineering_cost.conversion_effort',
    'engineering_cost.update_cost',
    'engineering_cost.packaging_cost',
    'reproduction.command',
  ];
  for (const path of requiredStrings) {
    if (typeof at(run, path) !== 'string' || at(run, path).length === 0) {
      errors.push(`${path} must be a non-empty string`);
    }
  }

  if (run.schema_version !== schemaVersion) errors.push(`schema_version must be ${schemaVersion}`);
  if (!/^[0-9a-f]{40}$/.test(run.source_revision ?? '')) errors.push('source_revision must be a 40-character Git SHA');
  if (run.candidate?.task !== 'asr') errors.push('candidate.task must be asr');
  if (run.host?.architecture !== 'arm64') errors.push('host.architecture must be arm64');
  if (!['ac', 'battery', 'unknown'].includes(run.host?.power_state)) errors.push('host.power_state is invalid');
  if (run.host?.memory_bytes !== null && !isMetric(run.host?.memory_bytes, true)) errors.push('host.memory_bytes must be a positive integer or null');
  if (run.fixture_id !== null && !fixtures.has(run.fixture_id)) errors.push(`fixture_id ${run.fixture_id} is not in the manifest`);

  const metricPaths = [
    ['measurements.quality.wer', false],
    ['measurements.quality.cer', false],
    ['measurements.timing.audio_duration_ms', false],
    ['measurements.timing.cold_load_ms', false],
    ['measurements.timing.warm_inference_ms', false],
    ['measurements.timing.real_time_factor', false],
    ['measurements.resources.peak_memory_bytes', true],
    ['measurements.resources.model_size_bytes', true],
    ['measurements.resources.binary_size_bytes', true],
    ['measurements.streaming.first_result_ms', false],
    ['measurements.streaming.update_count', true],
    ['measurements.streaming.volatile_update_count', true],
    ['measurements.streaming.final_update_count', true],
    ['measurements.streaming.revoke_count', true],
    ['measurements.energy.sample_rate_ms', true],
    ['measurements.energy.sample_count', true],
    ['measurements.energy.energy_joules', false],
  ];
  for (const [path, integer] of metricPaths) {
    if (!isMetric(at(run, path), integer)) errors.push(`${path} must be a non-negative metric or null`);
  }

  if (!['none', 'segment', 'word', 'unknown'].includes(run.measurements?.streaming?.timestamps)) errors.push('measurements.streaming.timestamps is invalid');
  if (!['powermetrics', 'external-meter', 'not-measured'].includes(run.measurements?.energy?.method)) errors.push('measurements.energy.method is invalid');
  if (!Array.isArray(run.reproduction?.raw_artifacts)) errors.push('reproduction.raw_artifacts must be an array');
  if (!Array.isArray(run.result?.observations)) errors.push('result.observations must be an array');

  const status = run.result?.status;
  if (!['measured', 'partial', 'blocked'].includes(status)) errors.push('result.status is invalid');
  if ((status === 'blocked' || status === 'partial') && !(run.result?.blocker?.length > 0)) errors.push(`${status} result requires a precise blocker`);
  if (status === 'measured') {
    const fixture = fixtures.get(run.fixture_id);
    if (!fixture || fixture.purpose !== 'quality') errors.push('measured result requires a quality fixture');
    if (!Number.isInteger(run.host?.memory_bytes) || run.host.memory_bytes <= 0) errors.push('measured result requires host.memory_bytes');
    for (const path of [
      'measurements.quality.wer',
      'measurements.quality.cer',
      'measurements.timing.audio_duration_ms',
      'measurements.timing.cold_load_ms',
      'measurements.timing.warm_inference_ms',
      'measurements.timing.real_time_factor',
      'measurements.resources.peak_memory_bytes',
      'measurements.resources.model_size_bytes',
    ]) {
      if (typeof at(run, path) !== 'number') errors.push(`measured result requires ${path}`);
    }
    if (run.reproduction.raw_artifacts.length === 0) errors.push('measured result requires at least one raw artifact');
  }

  if (run.measurements?.streaming?.supported) {
    if (typeof run.measurements.streaming.first_result_ms !== 'number') errors.push('streaming result requires first_result_ms');
    if (!(run.measurements.streaming.update_count > 0)) errors.push('streaming result requires update_count');
    if (!(run.measurements.streaming.final_update_count > 0)) errors.push('streaming result requires final_update_count');
  }

  return errors;
}

function validateManifest(manifest) {
  const errors = [];
  if (manifest.schema_version !== schemaVersion) errors.push(`manifest schema_version must be ${schemaVersion}`);
  if (!Array.isArray(manifest.fixtures) || manifest.fixtures.length === 0) errors.push('manifest must contain fixtures');
  const ids = new Set();
  for (const fixture of manifest.fixtures ?? []) {
    if (ids.has(fixture.id)) errors.push(`duplicate fixture id: ${fixture.id}`);
    ids.add(fixture.id);
    if (!['smoke', 'quality'].includes(fixture.purpose)) errors.push(`${fixture.id}: invalid purpose`);
    if (!(fixture.reference_text?.length > 0)) errors.push(`${fixture.id}: reference_text is required`);
    if (fixture.purpose === 'quality') {
      if (!/^[0-9a-f]{64}$/.test(fixture.sha256 ?? '')) errors.push(`${fixture.id}: quality fixture requires sha256`);
      if (!fixture.provenance?.redistributable && fixture.availability !== 'local-required') {
        errors.push(`${fixture.id}: non-redistributable quality fixture must be local-required`);
      }
    }
  }
  return { errors, ids };
}

function arrayEquals(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function nonNullMaximum(values) {
  const present = values.filter((value) => value !== null);
  return present.length === 0 ? null : Math.max(...present);
}

async function validateMatrix(matrix, manifest, path) {
  const errors = [];
  if (matrix.schema_version !== schemaVersion) errors.push(`schema_version must be ${schemaVersion}`);
  if (!(matrix.matrix_id?.length > 0)) errors.push('matrix_id must be a non-empty string');
  if (!/^[0-9a-f]{40}$/.test(matrix.source_revision ?? '')) errors.push('source_revision must be a 40-character Git SHA');
  if (matrix.fixture_manifest_revision !== manifest.revision) {
    errors.push(`fixture_manifest_revision must be ${manifest.revision}`);
  }
  if (!Array.isArray(matrix.fixture_ids) || matrix.fixture_ids.length === 0) {
    errors.push('fixture_ids must be a non-empty array');
  }
  if (new Set(matrix.fixture_ids ?? []).size !== (matrix.fixture_ids ?? []).length) {
    errors.push('fixture_ids must be unique');
  }
  for (const fixtureId of matrix.fixture_ids ?? []) {
    const fixture = manifest.fixtures.find((item) => item.id === fixtureId);
    if (!fixture || fixture.purpose !== 'quality') errors.push(`${fixtureId}: matrix fixture must be a quality fixture`);
  }
  if (!arrayEquals(matrix.procedure?.fixture_order ?? [], matrix.fixture_ids ?? [])) {
    errors.push('procedure.fixture_order must equal fixture_ids');
  }

  const candidateIds = new Set();
  for (const candidate of matrix.candidates ?? []) {
    if (candidateIds.has(candidate.id)) errors.push(`duplicate candidate id: ${candidate.id}`);
    candidateIds.add(candidate.id);
    if (!(candidate.raw_artifact?.startsWith('benchmarks/raw/') && candidate.raw_artifact.endsWith('/result.json'))) {
      errors.push(`${candidate.id}: raw_artifact must name a checked-in benchmark result`);
      continue;
    }

    let raw;
    try {
      raw = await readJson(resolve(repoRoot, candidate.raw_artifact));
    } catch (error) {
      errors.push(`${candidate.id}: cannot read ${candidate.raw_artifact}: ${error.message}`);
      continue;
    }
    if (raw.schema_version !== schemaVersion) errors.push(`${candidate.id}: raw schema_version must be ${schemaVersion}`);
    if (raw.candidate?.id !== candidate.id) errors.push(`${candidate.id}: raw candidate id does not match`);
    if (raw.source_revision !== matrix.source_revision) errors.push(`${candidate.id}: raw source_revision does not match`);
    if (raw.fixture_manifest_revision !== matrix.fixture_manifest_revision) {
      errors.push(`${candidate.id}: raw fixture_manifest_revision does not match`);
    }
    const rawFixtureIds = (raw.results ?? []).map((result) => result.fixture_id);
    if (!arrayEquals(rawFixtureIds, matrix.fixture_ids)) errors.push(`${candidate.id}: raw fixture order does not match`);

    const rawMetrics = {
      macroWer: raw.summary?.macro_wer,
      macroCer: raw.summary?.macro_cer,
      meanCold: mean(raw.results.map((result) => result.timing.cold_load_ms)),
      minimumCold: Math.min(...raw.results.map((result) => result.timing.cold_load_ms)),
      maximumCold: Math.max(...raw.results.map((result) => result.timing.cold_load_ms)),
      meanWarm: raw.summary?.mean_warm_inference_ms,
      meanRtf: raw.summary?.mean_real_time_factor,
      maximumPeakMemory: raw.summary?.maximum_peak_memory_bytes,
      runtimePeakMemory: nonNullMaximum(raw.results.map((result) => result.resources.runtime_peak_memory_bytes)),
      modelSize: nonNullMaximum(raw.results.map((result) => result.resources.model_size_bytes)),
      binarySize: nonNullMaximum(raw.results.map((result) => result.resources.binary_size_bytes)),
      meanFirstResult: (() => {
        const values = raw.results
          .map((result) => result.streaming.first_result_ms)
          .filter((value) => value !== null);
        return values.length === 0 ? null : mean(values);
      })(),
      streamingSupported: raw.results.every((result) => result.streaming.supported),
      timestamps: [...new Set(raw.results.map((result) => result.streaming.timestamps))],
    };
    const comparisons = [
      ['quality.macro_wer', candidate.quality?.macro_wer, rawMetrics.macroWer],
      ['quality.macro_cer', candidate.quality?.macro_cer, rawMetrics.macroCer],
      ['timing.mean_cold_load_ms', candidate.timing?.mean_cold_load_ms, rawMetrics.meanCold],
      ['timing.minimum_cold_load_ms', candidate.timing?.minimum_cold_load_ms, rawMetrics.minimumCold],
      ['timing.maximum_cold_load_ms', candidate.timing?.maximum_cold_load_ms, rawMetrics.maximumCold],
      ['timing.mean_warm_inference_ms', candidate.timing?.mean_warm_inference_ms, rawMetrics.meanWarm],
      ['timing.mean_real_time_factor', candidate.timing?.mean_real_time_factor, rawMetrics.meanRtf],
      ['resources.maximum_peak_memory_bytes', candidate.resources?.maximum_peak_memory_bytes, rawMetrics.maximumPeakMemory],
      ['resources.runtime_peak_memory_bytes', candidate.resources?.runtime_peak_memory_bytes, rawMetrics.runtimePeakMemory],
      ['resources.model_size_bytes', candidate.resources?.model_size_bytes, rawMetrics.modelSize],
      ['resources.binary_size_bytes', candidate.resources?.binary_size_bytes, rawMetrics.binarySize],
      ['streaming.mean_first_result_ms', candidate.streaming?.mean_first_result_ms, rawMetrics.meanFirstResult],
      ['streaming.supported', candidate.streaming?.supported, rawMetrics.streamingSupported],
    ];
    for (const [field, actual, expected] of comparisons) {
      if (actual !== expected) errors.push(`${candidate.id}: ${field} does not match raw artifact`);
    }
    if (rawMetrics.timestamps.length !== 1 || candidate.streaming?.timestamps !== rawMetrics.timestamps[0]) {
      errors.push(`${candidate.id}: streaming.timestamps does not match raw artifact`);
    }
    if (JSON.stringify(candidate.quality?.by_language) !== JSON.stringify(raw.summary?.by_language)) {
      errors.push(`${candidate.id}: quality.by_language does not match raw artifact`);
    }
  }

  if ((matrix.candidates ?? []).length < 2) errors.push('matrix requires at least two candidates');
  if (!candidateIds.has(matrix.recommendation?.primary)) errors.push('recommendation.primary must name a candidate');
  if (!candidateIds.has(matrix.recommendation?.fallback)) errors.push('recommendation.fallback must name a candidate');
  if (matrix.recommendation?.primary === matrix.recommendation?.fallback) {
    errors.push('recommendation primary and fallback must differ');
  }
  for (const id of matrix.recommendation?.research ?? []) {
    if (!candidateIds.has(id)) errors.push(`recommendation research candidate does not exist: ${id}`);
  }

  return errors.map((error) => `${path}: ${error}`);
}

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

const manifestPath = join(repoRoot, 'benchmarks/fixtures/manifest.json');
const schemaPaths = [
  join(repoRoot, 'benchmarks/schema/run.schema.json'),
  join(repoRoot, 'benchmarks/schema/fixture-manifest.schema.json'),
  join(repoRoot, 'benchmarks/schema/matrix.schema.json'),
];
for (const path of schemaPaths) {
  const schema = await readJson(path);
  if (schema.$schema !== 'https://json-schema.org/draft/2020-12/schema') {
    throw new Error(`${path}: expected JSON Schema draft 2020-12`);
  }
}

const manifest = await readJson(manifestPath);
const manifestValidation = validateManifest(manifest);
const requestedRun = process.argv.indexOf('--run');
const runPaths = requestedRun >= 0
  ? [resolve(process.cwd(), process.argv[requestedRun + 1])]
  : (await readdir(join(repoRoot, 'benchmarks/runs')))
      .filter((name) => name.endsWith('.json'))
      .sort()
      .map((name) => join(repoRoot, 'benchmarks/runs', name));

const failures = manifestValidation.errors.map((error) => `${manifestPath}: ${error}`);
const runs = [];
for (const path of runPaths) {
  const run = await readJson(path);
  runs.push(run);
  failures.push(...validateRun(run, new Map(manifest.fixtures.map((fixture) => [fixture.id, fixture]))).map((error) => `${path}: ${error}`));
}

const matrixDirectory = join(repoRoot, 'benchmarks/matrices');
const matrixPaths = (await readdir(matrixDirectory))
  .filter((name) => name.endsWith('.json'))
  .sort()
  .map((name) => join(matrixDirectory, name));
const matrices = [];
for (const path of matrixPaths) {
  const matrix = await readJson(path);
  matrices.push(matrix);
  failures.push(...await validateMatrix(matrix, manifest, path));
}

if (process.argv.includes('--self-test')) {
  const invalid = structuredClone(runs[0]);
  invalid.result.status = 'measured';
  invalid.result.blocker = null;
  invalid.fixture_id = null;
  if (validateRun(invalid, new Map(manifest.fixtures.map((fixture) => [fixture.id, fixture]))).length === 0) {
    failures.push('validator self-test failed to reject an incomplete measured run');
  }
  const invalidMatrix = structuredClone(matrices[0]);
  invalidMatrix.candidates[0].quality.macro_wer += 1;
  if ((await validateMatrix(invalidMatrix, manifest, '<matrix-self-test>')).length === 0) {
    failures.push('validator self-test failed to reject a matrix that diverges from its raw artifact');
  }
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(
  `benchmark data: ${manifest.fixtures.length} fixture(s), ${runs.length} run record(s), ` +
  `${matrices.length} matrix record(s), schema ${schemaVersion}`,
);
