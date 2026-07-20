#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, actualPath } = parseArguments(process.argv.slice(2));
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const actualDocument = JSON.parse(await readFile(actualPath, 'utf8'));
const actual = actualDocument.result?.probe ?? actualDocument;
const stages = [
  'adapter',
  'time_embedding',
  'ada_scale_layer_0',
  'ada_scale_layer_12',
  'ada_scale_layer_25',
  'prompt_text_embeddings',
  'prefix_embeddings',
  'prefill_layer_0',
  'prefill_layer_12',
  'prefill_layer_25',
  'prefill_norm',
  'prefill_logits',
  'decode_0_input',
  'decode_0_layer_0',
  'decode_0_layer_12',
  'decode_0_layer_25',
  'decode_0_norm',
  'decode_0_logits',
  'final_logits',
  'decoder_layer_0_cache_keys',
  'decoder_layer_0_cache_values',
];
const sampleAbsoluteTolerance = 5e-4;
const aggregateRelativeTolerance = 5e-4;
const errors = [];
const comparisons = {};

if (actual.stage !== 'voxtral-greedy-transcription' ||
    actual.status !== 'ok' ||
    actual.boundary !== 'official-mlx-cpp' ||
    actual.device !== 'gpu') {
  errors.push('actual result must be the direct official-MLX GPU decoder');
}
if (actual.pcm_samples !== oracle.fixture?.pcm_samples ||
    actual.transcription_delay_ms !== oracle.fixture?.transcription_delay_ms ||
    actual.delay_tokens !== oracle.fixture?.delay_tokens) {
  errors.push('fixture length and delay must match the pinned oracle');
}
for (const field of [
  'layers',
  'dimension',
  'attention_heads',
  'kv_heads',
  'head_dimension',
  'hidden_dimension',
  'sliding_window',
  'vocabulary_size',
  'ada_bottleneck_dimension',
]) {
  if (actual.architecture?.[field] !== oracle.architecture?.[field]) {
    errors.push(
      `architecture.${field}: ${actual.architecture?.[field]}, expected ${oracle.architecture?.[field]}`,
    );
  }
}
if (JSON.stringify(actual.prompt) !== JSON.stringify(oracle.prompt)) {
  errors.push('delay-conditioned prompt differs from the pinned oracle');
}
for (const field of ['token_count', 'forward_steps', 'finish_reason', 'text']) {
  if (actual.generation?.[field] !== oracle.generation?.[field]) {
    errors.push(
      `generation.${field}: ${JSON.stringify(actual.generation?.[field])}, expected ${JSON.stringify(oracle.generation?.[field])}`,
    );
  }
}
if (!equalArrays(actual.generation?.tokens, oracle.generation?.tokens)) {
  errors.push('generated token sequence differs from the pinned oracle');
}
if (JSON.stringify(actual.cache) !== JSON.stringify(oracle.cache)) {
  errors.push('decoder cache offsets or materialized shape differ');
}
for (const capability of [
  'delay_conditioning',
  'decoder',
  'decoder_kv_cache',
  'tekken_decode',
  'greedy_transcription',
]) {
  if (actual.capabilities?.[capability] !== true) {
    errors.push(`capabilities.${capability} must be true`);
  }
}
if (actual.capabilities?.streaming_session !== false) {
  errors.push('batch decoder must not claim a complete streaming session');
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
  exact_token_parity: equalArrays(
    actual.generation?.tokens,
    oracle.generation?.tokens,
  ),
  exact_text_parity: actual.generation?.text === oracle.generation?.text,
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
      'usage: node scripts/validate-voxtral-mlx-decoder.mjs --oracle ORACLE_JSON --actual ACTUAL_JSON',
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
