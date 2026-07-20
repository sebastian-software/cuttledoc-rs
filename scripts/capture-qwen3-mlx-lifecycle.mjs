#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const options = parseArguments(process.argv.slice(2));
const [lifecycle, cancellation, oracle] = await Promise.all(
  [options.lifecyclePath, options.cancellationPath, options.oraclePath].map(
    async (path) => JSON.parse(await readFile(path, 'utf8')),
  ),
);
const runs = lifecycle.sessions.flatMap((session) => session.runs);
const exactRuns = runs.filter(({ transcription }) =>
  exactTranscription(transcription, oracle),
);

if (lifecycle.sessions.length < 3 || runs.length < 6) {
  throw new Error('lifecycle evidence requires at least 3 sessions and 6 runs');
}
if (exactRuns.length !== runs.length) {
  throw new Error(
    `${exactRuns.length}/${runs.length} lifecycle transcripts match the oracle`,
  );
}
if (lifecycle.stable_errors?.invalid_argument?.status !== 1) {
  throw new Error('invalid-argument probe did not return status 1');
}
if (cancellation.busy_probe?.status !== 4) {
  throw new Error('busy probe did not return status 4');
}
if (cancellation.cancelled_call?.status !== 3) {
  throw new Error('cancelled call did not return status 3');
}

const firstRuns = lifecycle.sessions.map((session) => session.runs[0]);
const sameHandleRuns = lifecycle.sessions.flatMap((session) =>
  session.runs.slice(1),
);
const record = {
  schema_version: '1.0.0',
  experiment_id: 'phase0.qwen3-mlx-direct.lifecycle-1',
  captured_at: new Date().toISOString(),
  source_revision: options.sourceRevision,
  issue: 'https://github.com/sebastian-software/cuttledoc-rs/issues/17',
  measurement_scope:
    'Repository-owned Rust lifecycle and stable C ABI semantics over the direct official-MLX Qwen3-ASR adapter. This development record checks repeated create/transcribe/destroy behavior, exact fixed-fixture parity, invalid arguments, concurrent busy handling, and cancellation at a natural synchronous MLX evaluation boundary. It is not a production throughput benchmark.',
  host: {
    architecture: 'arm64',
    processor: 'Apple M1 Ultra',
    memory_bytes: 68719476736,
    os: 'macOS 26.5.2 (25F84)',
  },
  toolchain: {
    mlx: {
      version: '0.32.0',
      revision: '7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247',
    },
    cmake: '4.4.0',
    rustc: '1.95.0-nightly (842bd5be2 2026-01-29)',
  },
  model: {
    source: 'Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0',
    conversion:
      'mlx-community/Qwen3-ASR-0.6B-8bit@89e96d92ba34aca20b3e29fb10cc284097d1219f',
    artifact_sha256:
      'b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8',
  },
  fixture: {
    id: 'audiobook-en-2277-149874-0000',
    manifest: 'benchmarks/fixtures/audiobook-pilot.json',
    oracle:
      'benchmarks/oracles/qwen3-asr-0.6b.audiobook-en-2277-149874-0000.decoder-en.json',
    pcm_sha256:
      'e393548961cfd11b2ca59aa25e68984f551f952165ad1d18a485505b3bf16d41',
    pcm_samples: lifecycle.pcm_samples,
    sample_rate_hz: 16000,
    language_prompt: lifecycle.language,
  },
  procedure: {
    device: lifecycle.device,
    lifecycle_count: lifecycle.lifecycle_count,
    runs_per_lifecycle: lifecycle.runs_per_lifecycle,
    session_streams:
      'One reusable official MLX thread-unsafe CPU stream and one reusable GPU stream, protected by the adapter runtime mutex',
    cancellation_signal_delay_ms: 50,
    cancellation_observation:
      'The atomic signal cannot interrupt a synchronous MLX graph or Metal kernel; it is checked after decoder prefill and after each decoder step.',
  },
  summary: {
    exact_transcriptions: exactRuns.length,
    total_transcriptions: runs.length,
    mean_create_ms: mean(lifecycle.sessions.map(({ create_ms }) => create_ms)),
    mean_destroy_ms: mean(
      lifecycle.sessions.map(({ destroy_ms }) => destroy_ms),
    ),
    mean_first_run_wall_ms: mean(firstRuns.map(({ wall_ms }) => wall_ms)),
    mean_same_handle_run_wall_ms: mean(
      sameHandleRuns.map(({ wall_ms }) => wall_ms),
    ),
    maximum_runtime_peak_memory_bytes: Math.max(
      ...runs.map(({ transcription }) => transcription.peak_memory_bytes),
    ),
    invalid_argument_status:
      lifecycle.stable_errors.invalid_argument.status,
    busy_status: cancellation.busy_probe.status,
    cancelled_status: cancellation.cancelled_call.status,
    cancellation_boundary: cancellation.cancelled_call.message,
    cancelled_worker_wall_ms: cancellation.worker_wall_ms,
    cancel_to_return_ms: cancellation.cancel_to_return_ms,
  },
  result: {
    status: 'ok',
    parity: {
      exact_generated_tokens: true,
      exact_decoded_text: true,
      exact_generation_token_count: true,
      exact_finish_reason: true,
      exact_cache_offset: true,
    },
    lifecycle,
    cancellation,
  },
  artifacts: {
    shim_dylib_bytes: 18938944,
    rust_probe_bytes: 714456,
  },
  next_gate:
    'Integrate this task-level boundary into the Rust engine abstraction, then add held-out professional-podcast coverage before product selection.',
};

await mkdir(dirname(options.outputPath), { recursive: true });
await writeFile(options.outputPath, `${JSON.stringify(record, null, 2)}\n`);
console.log(
  JSON.stringify(
    {
      status: 'ok',
      output: options.outputPath,
      exact_transcriptions: exactRuns.length,
      invalid_argument_status:
        lifecycle.stable_errors.invalid_argument.status,
      busy_status: cancellation.busy_probe.status,
      cancelled_status: cancellation.cancelled_call.status,
    },
    null,
    2,
  ),
);

function exactTranscription(actual, expected) {
  return (
    equalArrays(actual.generated_tokens, expected.generated_tokens) &&
    actual.text === expected.text &&
    actual.generation_tokens === expected.generated_tokens.length &&
    actual.finish_reason === 'eos' &&
    [151643, 151645].includes(actual.stop_token) &&
    actual.cache_offset_after_generation ===
      expected.prompt_length + expected.generated_tokens.length
  );
}

function equalArrays(left, right) {
  return (
    Array.isArray(left) &&
    Array.isArray(right) &&
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function parseArguments(arguments_) {
  const values = new Map();
  for (let index = 0; index < arguments_.length; index += 2) {
    values.set(arguments_[index], arguments_[index + 1]);
  }
  for (const name of [
    '--lifecycle',
    '--cancellation',
    '--oracle',
    '--output',
    '--source-revision',
  ]) {
    if (!values.get(name)) {
      console.error(
        'usage: node scripts/capture-qwen3-mlx-lifecycle.mjs --lifecycle LIFECYCLE_JSON --cancellation CANCELLATION_JSON --oracle ORACLE_JSON --output RESULT_JSON --source-revision GIT_SHA',
      );
      process.exit(2);
    }
  }
  return {
    lifecyclePath: resolve(values.get('--lifecycle')),
    cancellationPath: resolve(values.get('--cancellation')),
    oraclePath: resolve(values.get('--oracle')),
    outputPath: resolve(values.get('--output')),
    sourceRevision: values.get('--source-revision'),
  };
}
