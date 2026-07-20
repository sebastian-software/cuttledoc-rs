#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { oraclePath, lifecyclePath, cancellationPath } = parseArguments(
  process.argv.slice(2),
);
const [oracle, lifecycle, cancellation] = await Promise.all(
  [oraclePath, lifecyclePath, cancellationPath].map(async (path) =>
    JSON.parse(await readFile(path, 'utf8')),
  ),
);
const errors = [];
const sessions = lifecycle.sessions ?? [];
const runs = sessions.flatMap((session) => session.runs ?? []);
let exactTranscriptionCount = 0;

if (lifecycle.status !== 'ok') {
  errors.push(`lifecycle status: ${lifecycle.status}, expected ok`);
}
if (lifecycle.stage !== 'qwen3-session-lifecycle') {
  errors.push(`lifecycle stage: ${lifecycle.stage}`);
}
if (lifecycle.lifecycle_count !== sessions.length) {
  errors.push(
    `lifecycle_count: ${lifecycle.lifecycle_count}, observed ${sessions.length}`,
  );
}
if (sessions.length < 3) {
  errors.push(`lifecycle count ${sessions.length} is below 3`);
}
if (runs.length < 6) {
  errors.push(`transcription count ${runs.length} is below 6`);
}
if (lifecycle.stable_errors?.invalid_argument?.status !== 1) {
  errors.push(
    `invalid-argument status: ${lifecycle.stable_errors?.invalid_argument?.status}, expected 1`,
  );
}

for (const [sessionIndex, session] of sessions.entries()) {
  if (!Number.isFinite(session.create_ms) || session.create_ms < 0) {
    errors.push(`session ${sessionIndex}: invalid create_ms`);
  }
  if (!Number.isFinite(session.destroy_ms) || session.destroy_ms < 0) {
    errors.push(`session ${sessionIndex}: invalid destroy_ms`);
  }
  if (session.runs?.length !== lifecycle.runs_per_lifecycle) {
    errors.push(
      `session ${sessionIndex}: ${session.runs?.length} runs, expected ${lifecycle.runs_per_lifecycle}`,
    );
  }
}

for (const [runIndex, run] of runs.entries()) {
  const actual = run.transcription;
  if (!actual) {
    errors.push(`run ${runIndex}: missing transcription`);
    continue;
  }
  if (!equalArrays(actual.generated_tokens, oracle.generated_tokens)) {
    errors.push(`run ${runIndex}: generated_tokens differ`);
  }
  if (actual.text !== oracle.text) {
    errors.push(`run ${runIndex}: decoded text differs`);
  }
  if (actual.language !== oracle.language) {
    errors.push(
      `run ${runIndex}: language ${actual.language}, expected ${oracle.language}`,
    );
  }
  if (actual.prompt_length !== oracle.prompt_length) {
    errors.push(
      `run ${runIndex}: prompt_length ${actual.prompt_length}, expected ${oracle.prompt_length}`,
    );
  }
  if (actual.generation_tokens !== oracle.generated_tokens.length) {
    errors.push(`run ${runIndex}: generation token count differs`);
  }
  if (actual.finish_reason !== 'eos' || ![151643, 151645].includes(actual.stop_token)) {
    errors.push(`run ${runIndex}: generation did not finish with Qwen EOS`);
  }
  if (
    actual.cache_offset_after_generation !==
    oracle.prompt_length + oracle.generated_tokens.length
  ) {
    errors.push(`run ${runIndex}: final cache offset differs`);
  }
  if (
    equalArrays(actual.generated_tokens, oracle.generated_tokens) &&
    actual.text === oracle.text &&
    actual.generation_tokens === oracle.generated_tokens.length &&
    actual.finish_reason === 'eos' &&
    [151643, 151645].includes(actual.stop_token) &&
    actual.cache_offset_after_generation ===
      oracle.prompt_length + oracle.generated_tokens.length
  ) {
    exactTranscriptionCount += 1;
  }
}

if (cancellation.status !== 'ok') {
  errors.push(`cancellation status: ${cancellation.status}, expected ok`);
}
if (cancellation.stage !== 'qwen3-session-cancellation') {
  errors.push(`cancellation stage: ${cancellation.stage}`);
}
if (cancellation.busy_probe?.status !== 4) {
  errors.push(
    `busy status: ${cancellation.busy_probe?.status}, expected 4`,
  );
}
if (cancellation.cancelled_call?.status !== 3) {
  errors.push(
    `cancelled status: ${cancellation.cancelled_call?.status}, expected 3`,
  );
}
if (
  !cancellation.cancelled_call?.message?.startsWith(
    'transcription cancelled at ',
  )
) {
  errors.push('cancelled call does not identify its observation boundary');
}

console.log(
  JSON.stringify(
    {
      status: errors.length === 0 ? 'ok' : 'error',
      oracle: oraclePath,
      lifecycle: lifecyclePath,
      cancellation: cancellationPath,
      observed: {
        lifecycle_count: sessions.length,
        transcription_count: runs.length,
        exact_transcription_count: exactTranscriptionCount,
        invalid_argument_status:
          lifecycle.stable_errors?.invalid_argument?.status,
        busy_status: cancellation.busy_probe?.status,
        cancelled_status: cancellation.cancelled_call?.status,
        cancellation_boundary: cancellation.cancelled_call?.message,
      },
      errors,
    },
    null,
    2,
  ),
);
if (errors.length > 0) process.exit(1);

function parseArguments(arguments_) {
  if (
    arguments_.length !== 6 ||
    arguments_[0] !== '--oracle' ||
    !arguments_[1] ||
    arguments_[2] !== '--lifecycle' ||
    !arguments_[3] ||
    arguments_[4] !== '--cancellation' ||
    !arguments_[5]
  ) {
    console.error(
      'usage: node scripts/validate-qwen3-mlx-lifecycle.mjs --oracle ORACLE_JSON --lifecycle LIFECYCLE_JSON --cancellation CANCELLATION_JSON',
    );
    process.exit(2);
  }
  return {
    oraclePath: resolve(arguments_[1]),
    lifecyclePath: resolve(arguments_[3]),
    cancellationPath: resolve(arguments_[5]),
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
