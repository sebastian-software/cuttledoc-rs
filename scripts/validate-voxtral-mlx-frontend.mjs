#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, actualPath } = parseArguments(process.argv.slice(2));
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const actualDocument = JSON.parse(await readFile(actualPath, 'utf8'));
const actual = actualDocument.result?.probe ?? actualDocument;
const stages = [
  'mel_filters',
  'log_mel',
  'conv0_gelu',
  'conv1_pretrunc_gelu',
  'conv_stem',
];
const sampleAbsoluteTolerance = 2e-6;
const aggregateRelativeTolerance = 2e-5;
const errors = [];
const comparisons = {};

if (actual.stage !== 'voxtral-audio-frontend' ||
    actual.status !== 'ok' ||
    actual.boundary !== 'official-mlx-cpp' ||
    actual.device !== 'gpu') {
  errors.push('actual result must be the direct official-MLX GPU frontend');
}
if (actual.pcm_samples !== oracle.fixture?.pcm_samples) {
  errors.push(
    `pcm_samples: ${actual.pcm_samples}, expected ${oracle.fixture?.pcm_samples}`,
  );
}
for (const field of [
  'transcription_delay_ms',
  'delay_tokens',
  'left_pad_tokens',
  'left_pad_samples',
  'alignment_pad_samples',
  'right_pad_tokens',
  'right_pad_samples',
  'padded_samples',
]) {
  if (actual.padding?.[field] !== oracle.padding?.[field]) {
    errors.push(
      `padding.${field}: ${actual.padding?.[field]}, expected ${oracle.padding?.[field]}`,
    );
  }
}
if (actual.mel_frames !== oracle.mel_frames) {
  errors.push(`mel_frames: ${actual.mel_frames}, expected ${oracle.mel_frames}`);
}
if (actual.front_truncation_frames !== oracle.front_truncation_frames) {
  errors.push(
    `front_truncation_frames: ${actual.front_truncation_frames}, expected ${oracle.front_truncation_frames}`,
  );
}
if (actual.capabilities?.mel_frontend !== true ||
    actual.capabilities?.causal_conv_stem !== true ||
    actual.capabilities?.causal_encoder !== false ||
    actual.capabilities?.transcription !== false) {
  errors.push('frontend capabilities must stop before encoder/transcription');
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
      'usage: node scripts/validate-voxtral-mlx-frontend.mjs --oracle ORACLE_JSON --actual ACTUAL_JSON',
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
