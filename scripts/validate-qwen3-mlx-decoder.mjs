#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, actualPath } = parseArguments(process.argv.slice(2));
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const actualDocument = JSON.parse(await readFile(actualPath, 'utf8'));
const actual = actualDocument.result?.probe ?? actualDocument;
const errors = [];
const comparisons = {};
const isTranscription = actual.generated_tokens !== undefined;

for (const [field, expected] of [
  ['language', oracle.language],
  ['prompt_length', oracle.prompt_length],
  ...(isTranscription
    ? []
    : [
        ['first_token', oracle.first_token],
        ['second_token', oracle.second_token],
      ]),
]) {
  if (actual[field] !== expected) {
    errors.push(`${field}: ${actual[field]}, expected ${expected}`);
  }
}
if (
  !isTranscription &&
  actual.cache_offset_after_prefill !== oracle.prompt_length
) {
  errors.push(
    `cache_offset_after_prefill: ${actual.cache_offset_after_prefill}, expected ${oracle.prompt_length}`,
  );
}

if (!isTranscription) {
  for (const section of ['prefill', 'second_step']) {
    for (const [stage, expected] of Object.entries(oracle[section])) {
      const observed = actual[section]?.[stage];
      if (!observed) {
        errors.push(`${section}.${stage}: missing fingerprint`);
        continue;
      }
      if (!equalArrays(observed.shape, expected.shape)) {
        errors.push(
          `${section}.${stage}: shape ${JSON.stringify(observed.shape)}, expected ${JSON.stringify(expected.shape)}`,
        );
        continue;
      }
      if (!equalArrays(observed.sample_indices, expected.sample_indices)) {
        errors.push(`${section}.${stage}: sample indices differ`);
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
      comparisons[`${section}.${stage}`] = {
        maximum_sample_absolute_error: maximumSampleAbsoluteError,
        maximum_aggregate_relative_error: maximumAggregateRelativeError,
      };

      const isFirstLayerCache =
        section === 'prefill' &&
        (stage === 'layer_0_keys' || stage === 'layer_0_values');
      const sampleTolerance = isFirstLayerCache ? 0 : 0.75;
      const aggregateTolerance = isFirstLayerCache ? 5e-4 : 5e-2;
      if (maximumSampleAbsoluteError > sampleTolerance) {
        errors.push(
          `${section}.${stage}: sample error ${maximumSampleAbsoluteError} exceeds ${sampleTolerance}`,
        );
      }
      if (maximumAggregateRelativeError > aggregateTolerance) {
        errors.push(
          `${section}.${stage}: aggregate error ${maximumAggregateRelativeError} exceeds ${aggregateTolerance}`,
        );
      }
    }
  }

  for (const field of ['first_top_tokens', 'second_top_tokens']) {
    const observed = actual[field] ?? [];
    const expected = oracle[field];
    if (observed[0]?.token !== expected[0].token) {
      errors.push(`${field}: top token differs`);
    }
    const expectedSet = new Set(expected.map(({ token }) => token));
    const overlap = observed.filter(({ token }) =>
      expectedSet.has(token),
    ).length;
    comparisons[field] = { top_10_token_overlap: overlap };
    if (overlap < 9) {
      errors.push(`${field}: top-10 token overlap ${overlap} is below 9`);
    }
  }
}

if (isTranscription) {
  if (!equalArrays(actual.generated_tokens, oracle.generated_tokens)) {
    errors.push('generated_tokens differ from the reference transcript');
  }
  if (actual.text !== oracle.text) {
    errors.push('decoded text differs from the reference transcript');
  }
  if (actual.generation_tokens !== oracle.generated_tokens.length) {
    errors.push(
      `generation_tokens: ${actual.generation_tokens}, expected ${oracle.generated_tokens.length}`,
    );
  }
  if (actual.finish_reason !== 'eos') {
    errors.push(`finish_reason: ${actual.finish_reason}, expected eos`);
  }
  if (![151643, 151645].includes(actual.stop_token)) {
    errors.push(`stop_token: ${actual.stop_token}, expected a Qwen EOS token`);
  }
  const expectedCacheOffset =
    oracle.prompt_length + oracle.generated_tokens.length;
  if (actual.cache_offset_after_generation !== expectedCacheOffset) {
    errors.push(
      `cache_offset_after_generation: ${actual.cache_offset_after_generation}, expected ${expectedCacheOffset}`,
    );
  }
}

console.log(
  JSON.stringify(
    {
      status: errors.length === 0 ? 'ok' : 'error',
      oracle: oraclePath,
      actual: actualPath,
      exact: isTranscription
        ? [
            'generated_tokens',
            'text',
            'generation_tokens',
            'finish_reason',
            'cache_offset_after_generation',
          ]
        : [
            'first_token',
            'second_token',
            'cache_offset_after_prefill',
            'layer_0_cache_samples',
          ],
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
      'usage: node scripts/validate-qwen3-mlx-decoder.mjs --oracle ORACLE_JSON --actual ACTUAL_JSON',
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
