#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, actualPath } = parseArguments(process.argv.slice(2));
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const actualDocument = JSON.parse(await readFile(actualPath, 'utf8'));
const actual = actualDocument.result?.probe ?? actualDocument;
const stages = [
  'input_features',
  'conv2d1',
  'conv2d2',
  'conv2d3',
  'conv_out',
];
const errors = [];
const comparisons = {};

if (actual.feature_length !== oracle.feature_length) {
  errors.push(
    `feature_length: ${actual.feature_length}, expected ${oracle.feature_length}`,
  );
}
if (!equalArrays(actual.chunk_lengths, oracle.chunk_lengths)) {
  errors.push('chunk_lengths differ from the pinned oracle');
}

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
  const maximumSampleAbsoluteError = Math.max(...sampleAbsoluteErrors);
  const maximumAggregateRelativeError = Math.max(
    ...Object.values(aggregateRelativeErrors),
  );
  comparisons[stage] = {
    maximum_sample_absolute_error: maximumSampleAbsoluteError,
    maximum_aggregate_relative_error: maximumAggregateRelativeError,
    aggregate_relative_errors: aggregateRelativeErrors,
  };

  if (maximumSampleAbsoluteError > 2e-6) {
    errors.push(
      `${stage}: maximum sample absolute error ${maximumSampleAbsoluteError} exceeds 2e-6`,
    );
  }
  if (maximumAggregateRelativeError > 2e-5) {
    errors.push(
      `${stage}: maximum aggregate relative error ${maximumAggregateRelativeError} exceeds 2e-5`,
    );
  }
}

const result = {
  status: errors.length === 0 ? 'ok' : 'error',
  oracle: oraclePath,
  actual: actualPath,
  tolerances: {
    sample_absolute: 2e-6,
    aggregate_relative: 2e-5,
  },
  comparisons,
  errors,
};
console.log(JSON.stringify(result, null, 2));
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
      'usage: node scripts/validate-qwen3-mlx-frontend.mjs --oracle ORACLE_JSON --actual ACTUAL_JSON',
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
