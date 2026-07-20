#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, actualPath } = parseArguments(process.argv.slice(2));
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const actualDocument = JSON.parse(await readFile(actualPath, 'utf8'));
const actual = actualDocument.result?.probe ?? actualDocument;
const aggregateRelativeTolerance = 2e-5;
const errors = [];
const comparisons = {};

const expectedTokenIds = [
  ...oracle.token_layout.prefix,
  ...Array(oracle.token_layout.audio_token_count).fill(
    oracle.token_layout.audio_token_id,
  ),
  ...oracle.token_layout.suffix,
];
const expectedAudioIndices = Array.from(
  {
    length:
      oracle.audio_token_range.end_exclusive - oracle.audio_token_range.start,
  },
  (_, index) => oracle.audio_token_range.start + index,
);
const observedTokenIds = actual.token_ids ?? expandTokenLayout(actual.token_layout);
const observedAudioIndices =
  actual.audio_token_indices ?? expandIndexRange(actual.audio_token_range);

for (const [field, expected] of [
  ['language', oracle.language],
  ['num_audio_tokens', oracle.num_audio_tokens],
  ['prompt_length', oracle.prompt_length],
]) {
  if (actual[field] !== expected) {
    errors.push(`${field}: ${actual[field]}, expected ${expected}`);
  }
}
if (!equalArrays(observedTokenIds, expectedTokenIds)) {
  errors.push('token_ids differ from the pinned Transformers reference');
}
if (!equalArrays(observedAudioIndices, expectedAudioIndices)) {
  errors.push('audio_token_indices differ from the pinned reference range');
}

for (const stage of [
  'token_embeddings',
  'audio_features_bfloat16',
  'inputs_embeds',
]) {
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

  const maximumSampleAbsoluteError = Math.max(
    ...observed.sample_values.map((value, index) =>
      Math.abs(value - expected.sample_values[index]),
    ),
  );
  const aggregateRelativeErrors = Object.fromEntries(
    ['mean', 'stddev', 'minimum', 'maximum', 'l1'].map((field) => [
      field,
      relativeError(observed[field], expected[field]),
    ]),
  );
  const maximumAggregateRelativeError = Math.max(
    ...Object.values(aggregateRelativeErrors),
  );
  comparisons[stage] = {
    maximum_sample_absolute_error: maximumSampleAbsoluteError,
    maximum_aggregate_relative_error: maximumAggregateRelativeError,
    aggregate_relative_errors: aggregateRelativeErrors,
  };
  if (maximumSampleAbsoluteError !== 0) {
    errors.push(
      `${stage}: sampled BF16 values differ by ${maximumSampleAbsoluteError}`,
    );
  }
  if (maximumAggregateRelativeError > aggregateRelativeTolerance) {
    errors.push(
      `${stage}: maximum aggregate relative error ${maximumAggregateRelativeError} exceeds ${aggregateRelativeTolerance}`,
    );
  }
}

console.log(
  JSON.stringify(
    {
      status: errors.length === 0 ? 'ok' : 'error',
      oracle: oraclePath,
      actual: actualPath,
      exact: ['token_ids', 'audio_token_indices', 'sample_values'],
      aggregate_relative_tolerance: aggregateRelativeTolerance,
      comparisons,
      errors,
    },
    null,
    2,
  ),
);
if (errors.length > 0) process.exit(1);

function parseArguments(arguments_) {
  if (
    arguments_.length !== 4 ||
    arguments_[0] !== '--oracle' ||
    !arguments_[1] ||
    arguments_[2] !== '--actual' ||
    !arguments_[3]
  ) {
    console.error(
      'usage: node scripts/validate-qwen3-mlx-prompt.mjs --oracle ORACLE_JSON --actual ACTUAL_JSON',
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

function expandTokenLayout(layout) {
  if (!layout) return undefined;
  return [
    ...layout.prefix,
    ...Array(layout.audio_token_count).fill(layout.audio_token_id),
    ...layout.suffix,
  ];
}

function expandIndexRange(range) {
  if (!range) return undefined;
  return Array.from(
    { length: range.end_exclusive - range.start },
    (_, index) => range.start + index,
  );
}
