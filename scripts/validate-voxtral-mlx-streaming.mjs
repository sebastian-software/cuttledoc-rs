#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, actualPath, delayMs, chunkMs } = parseArguments(
  process.argv.slice(2),
);
const oracle = JSON.parse(await readFile(oraclePath, 'utf8'));
const actual = JSON.parse(await readFile(actualPath, 'utf8'));
const expectedRuns = oracle.runs?.filter(
  (run) => run.transcription_delay_ms === delayMs,
) ?? [];
const expectedTexts = new Set(expectedRuns.map((run) => run.text));
const expectedTokenCounts = new Set(
  expectedRuns.map((run) => run.streaming?.generated_tokens),
);
const errors = [];

if (
  actual.status !== 'ok' ||
  actual.stage !== 'voxtral-incremental-streaming' ||
  actual.boundary !== 'repository-owned-rust-c-abi-over-official-mlx' ||
  actual.device !== 'gpu'
) {
  errors.push('actual result must be the repository-owned official-MLX GPU stream');
}
if (expectedRuns.length === 0) {
  errors.push(`oracle contains no ${delayMs} ms run`);
}
if (actual.transcription_delay_ms !== delayMs) {
  errors.push(`actual delay is ${actual.transcription_delay_ms}, expected ${delayMs}`);
}
if (actual.procedure?.chunk_ms !== chunkMs) {
  errors.push(`actual chunk size is ${actual.procedure?.chunk_ms} ms, expected ${chunkMs} ms`);
}
if (actual.fixture?.pcm_samples !== oracle.fixture?.sample_count) {
  errors.push('actual and oracle fixture sample counts differ');
}
if (!expectedTexts.has(actual.text)) {
  errors.push('final text differs from every matching pinned streaming-oracle run');
}
if (!expectedTokenCounts.has(actual.streaming?.generated_tokens)) {
  errors.push('generated-token count differs from every matching oracle run');
}
if (
  actual.streaming?.append_only !== true ||
  actual.streaming?.revoke_count !== 0 ||
  actual.streaming?.done !== true ||
  !(actual.streaming?.update_count > 0)
) {
  errors.push('stream must finish with non-empty append-only updates and no revocations');
}
if (actual.streaming?.total_ingested_samples !== actual.fixture?.pcm_samples) {
  errors.push('not every fixture sample was ingested exactly once');
}
if (
  !(actual.streaming?.maximum_ingested_samples > 0) ||
  actual.streaming?.maximum_ingested_samples >
    actual.procedure?.max_ingest_samples_per_step
) {
  errors.push('one step exceeded the configured audio-ingestion budget');
}
if (
  !actual.procedure?.realtime_pacing ||
  !actual.procedure?.producer_thread_independent_from_mlx_executor
) {
  errors.push('evidence must use real-time pacing and an independent producer');
}
if (
  !(actual.timing?.process_load_ms > 0) ||
  !(actual.timing?.first_append_ms > 0) ||
  actual.timing?.first_stable_ms !== actual.timing?.first_append_ms ||
  !(actual.timing?.final_ms > 0)
) {
  errors.push('process-load, first-append, and final timing must be measured');
}
if (actual.timing?.maximum_step_wall_ms >= 5_000) {
  errors.push('maximum bounded direct step must stay below the 5-second regression ceiling');
}
if (actual.timing?.endpoint_finalization_ms >= 5_000) {
  errors.push('endpoint finalization must stay below the 5-second regression ceiling');
}
if (!Array.isArray(actual.events) || actual.events.length !== actual.streaming?.update_count) {
  errors.push('event count must match the reported append-update count');
} else if (actual.events.map((event) => event.delta).join('') !== actual.text) {
  errors.push('concatenated append deltas do not reconstruct final text');
}

const result = {
  status: errors.length === 0 ? 'ok' : 'error',
  oracle: oraclePath,
  actual: actualPath,
  transcription_delay_ms: delayMs,
  chunk_ms: chunkMs,
  exact_streaming_text_parity: expectedTexts.has(actual.text),
  exact_generated_token_count_parity: expectedTokenCounts.has(
    actual.streaming?.generated_tokens,
  ),
  bounded_step_regression_ceiling_ms: 5_000,
  observed_maximum_step_wall_ms: actual.timing?.maximum_step_wall_ms,
  errors,
};
console.log(JSON.stringify(result, null, 2));
if (errors.length > 0) process.exit(1);

function parseArguments(arguments_) {
  if (
    arguments_.length !== 8 ||
    arguments_[0] !== '--oracle' ||
    !arguments_[1] ||
    arguments_[2] !== '--actual' ||
    !arguments_[3] ||
    arguments_[4] !== '--delay-ms' ||
    !arguments_[5] ||
    arguments_[6] !== '--chunk-ms' ||
    !arguments_[7]
  ) {
    console.error(
      'usage: node scripts/validate-voxtral-mlx-streaming.mjs --oracle ORACLE_JSON --actual ACTUAL_JSON --delay-ms DELAY --chunk-ms CHUNK',
    );
    process.exit(2);
  }
  const delayMs = Number(arguments_[5]);
  const chunkMs = Number(arguments_[7]);
  if (!Number.isInteger(delayMs) || delayMs <= 0 || !Number.isInteger(chunkMs) || chunkMs <= 0) {
    console.error('delay and chunk size must be positive integers');
    process.exit(2);
  }
  return {
    oraclePath: resolve(arguments_[1]),
    actualPath: resolve(arguments_[3]),
    delayMs,
    chunkMs,
  };
}
