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

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

const manifestPath = join(repoRoot, 'benchmarks/fixtures/manifest.json');
const schemaPaths = [
  join(repoRoot, 'benchmarks/schema/run.schema.json'),
  join(repoRoot, 'benchmarks/schema/fixture-manifest.schema.json'),
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

if (process.argv.includes('--self-test')) {
  const invalid = structuredClone(runs[0]);
  invalid.result.status = 'measured';
  invalid.result.blocker = null;
  invalid.fixture_id = null;
  if (validateRun(invalid, new Map(manifest.fixtures.map((fixture) => [fixture.id, fixture]))).length === 0) {
    failures.push('validator self-test failed to reject an incomplete measured run');
  }
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`benchmark data: ${manifest.fixtures.length} fixture(s), ${runs.length} run record(s), schema ${schemaVersion}`);
