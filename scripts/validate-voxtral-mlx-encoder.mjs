#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, actualPath } = parseArguments(process.argv.slice(2));
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const actualDocument = JSON.parse(await readFile(actualPath, 'utf8'));
const actual = actualDocument.result?.probe ?? actualDocument;
const stages = [
  'conv_stem',
  'chunk_0_layer_0',
  'chunk_0_layer_15',
  'chunk_0_layer_31',
  'chunk_0_norm',
  'chunk_1_layer_0',
  'chunk_1_layer_15',
  'chunk_1_layer_31',
  'chunk_1_norm',
  'encoded',
  'adapter_projection0_gelu',
  'adapter',
  'layer_0_cache_keys',
  'layer_0_cache_values',
];
const sampleAbsoluteTolerance = 5e-5;
const aggregateRelativeTolerance = 1e-5;
const errors = [];
const comparisons = {};

if (actual.stage !== 'voxtral-causal-encoder' ||
    actual.status !== 'ok' ||
    actual.boundary !== 'official-mlx-cpp' ||
    actual.device !== 'gpu') {
  errors.push('actual result must be the direct official-MLX GPU encoder');
}
if (actual.pcm_samples !== oracle.fixture?.pcm_samples ||
    actual.transcription_delay_ms !==
      oracle.fixture?.transcription_delay_ms ||
    actual.delay_tokens !== oracle.fixture?.delay_tokens) {
  errors.push('fixture length and delay must match the pinned oracle');
}
for (const field of [
  'layers',
  'dimension',
  'attention_heads',
  'head_dimension',
  'sliding_window',
  'downsample_factor',
  'adapter_dimension',
]) {
  if (actual.architecture?.[field] !== oracle.architecture?.[field]) {
    errors.push(
      `architecture.${field}: ${actual.architecture?.[field]}, expected ${oracle.architecture?.[field]}`,
    );
  }
}
if (JSON.stringify(actual.chunks) !== JSON.stringify(oracle.chunks)) {
  errors.push('encoder chunk boundaries or attention-mask shapes differ');
}
for (const field of [
  'layer_0_offset',
  'layer_0_size',
  'layer_0_materialized_key_frames',
  'layer_0_materialized_value_frames',
]) {
  if (actual.cache?.[field] !== oracle.cache?.[field]) {
    errors.push(
      `cache.${field}: ${actual.cache?.[field]}, expected ${oracle.cache?.[field]}`,
    );
  }
}
for (const field of ['encoded_frames', 'adapter_frames']) {
  if (actual.output?.[field] !== oracle.output?.[field]) {
    errors.push(
      `output.${field}: ${actual.output?.[field]}, expected ${oracle.output?.[field]}`,
    );
  }
}
if (actual.capabilities?.causal_encoder !== true ||
    actual.capabilities?.rotating_kv_cache !== true ||
    actual.capabilities?.sliding_window_attention !== true ||
    actual.capabilities?.adapter_projection !== true ||
    actual.capabilities?.decoder !== false ||
    actual.capabilities?.transcription !== false) {
  errors.push('encoder capabilities must stop before decoder/transcription');
}

let maximumSampleAbsoluteError = 0;
let maximumAggregateRelativeError = 0;
for (const stage of stages) {
  const expected = oracle.fingerprints?.[stage];
  const observed = actual.fingerprints?.[stage];
  if (!expected || !observed) {
    errors.push(`${stage}: missing fingerprint`);
    continue;
  }
  if (!equalArrays(observed.shape, expected.shape)) {
    errors.push(
      `${stage}: shape ${JSON.stringify(observed.shape)}, expected ${JSON.stringify(expected.shape)}`,
    );
    continue;
  }
  if (!equalArrays(observed.sample_indices, expected.sample_indices)) {
    errors.push(`${stage}: sample indices differ from the pinned oracle`);
    continue;
  }

  const sampleAbsoluteErrors = observed.sample_values.map((value, index) =>
    Math.abs(value - expected.sample_values[index]),
  );
  const aggregateRelativeErrors = Object.fromEntries(
    ['mean', 'stddev', 'minimum', 'maximum', 'l1'].map((field) => [
      field,
      relativeError(observed[field], expected[field]),
    ]),
  );
  const stageMaximumSampleAbsoluteError = Math.max(...sampleAbsoluteErrors);
  const stageMaximumAggregateRelativeError = Math.max(
    ...Object.values(aggregateRelativeErrors),
  );
  maximumSampleAbsoluteError = Math.max(
    maximumSampleAbsoluteError,
    stageMaximumSampleAbsoluteError,
  );
  maximumAggregateRelativeError = Math.max(
    maximumAggregateRelativeError,
    stageMaximumAggregateRelativeError,
  );
  comparisons[stage] = {
    maximum_sample_absolute_error: stageMaximumSampleAbsoluteError,
    maximum_aggregate_relative_error: stageMaximumAggregateRelativeError,
    aggregate_relative_errors: aggregateRelativeErrors,
  };

  if (stageMaximumSampleAbsoluteError > sampleAbsoluteTolerance) {
    errors.push(
      `${stage}: maximum sample absolute error ${stageMaximumSampleAbsoluteError} exceeds ${sampleAbsoluteTolerance}`,
    );
  }
  if (stageMaximumAggregateRelativeError > aggregateRelativeTolerance) {
    errors.push(
      `${stage}: maximum aggregate relative error ${stageMaximumAggregateRelativeError} exceeds ${aggregateRelativeTolerance}`,
    );
  }
}

const result = {
  status: errors.length === 0 ? 'ok' : 'error',
  oracle: oraclePath,
  actual: actualPath,
  tolerances: {
    sample_absolute: sampleAbsoluteTolerance,
    aggregate_relative: aggregateRelativeTolerance,
  },
  maximum_sample_absolute_error: maximumSampleAbsoluteError,
  maximum_aggregate_relative_error: maximumAggregateRelativeError,
  comparisons,
  errors,
};
console.log(JSON.stringify(result, null, 2));
if (errors.length > 0) process.exit(1);

function parseArguments(arguments_) {
  if (arguments_.length !== 4 ||
      arguments_[0] !== '--oracle' ||
      !arguments_[1] ||
      arguments_[2] !== '--actual' ||
      !arguments_[3]) {
    console.error(
      'usage: node scripts/validate-voxtral-mlx-encoder.mjs --oracle ORACLE_JSON --actual ACTUAL_JSON',
    );
    process.exit(2);
  }
  return {
    oraclePath: resolve(arguments_[1]),
    actualPath: resolve(arguments_[3]),
  };
}

function equalArrays(left, right) {
  return (
    Array.isArray(left) &&
    Array.isArray(right) &&
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function relativeError(observed, expected) {
  return Math.abs(observed - expected) / Math.max(Math.abs(expected), 1e-12);
}
