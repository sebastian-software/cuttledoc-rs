#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import {
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { cpus, tmpdir, totalmem } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const scriptPath = fileURLToPath(import.meta.url);
const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    manifest: {
      type: 'string',
      default: join(
        repoRoot,
        'benchmarks/fixtures/target-domain-corpus.json',
      ),
    },
    plan: {
      type: 'string',
      default: join(
        repoRoot,
        'benchmarks/fixtures/target-domain-plan.json',
      ),
    },
    'audio-dir': {
      type: 'string',
      default: join(repoRoot, 'artifacts/target-domain/normalized'),
    },
    output: { type: 'string' },
    'source-group': { type: 'string' },
    backend: { type: 'string', default: 'all' },
    'allow-test': { type: 'boolean', default: false },
    'whisper-module-dir': {
      type: 'string',
      default: '/Users/sebastian/Workspace/whisper-coreml',
    },
    'whisper-model-dir': {
      type: 'string',
      default: '/Users/sebastian/.cache/whisper-coreml/models',
    },
    'parakeet-module-dir': {
      type: 'string',
      default: '/Users/sebastian/Workspace/parakeet-coreml',
    },
    'parakeet-model-dir': {
      type: 'string',
      default: '/Users/sebastian/.cache/parakeet-coreml/models',
    },
    'parakeet-vad-dir': {
      type: 'string',
      default: '/Users/sebastian/.cache/parakeet-coreml/vad',
    },
    'apple-build-dir': {
      type: 'string',
      default: '/private/tmp/cuttledoc-qwen3-tts-apple-asr-build',
    },
    'qwen-binary': {
      type: 'string',
      default:
        '/private/tmp/cuttledoc-qwen3-mlx-direct-build/' +
        'cuttledoc-qwen3-mlx-inspect',
    },
    'qwen-model-dir': {
      type: 'string',
      default: '/private/tmp/cuttledoc-qwen3-asr/model',
    },
    'qwen-chunk-ms': { type: 'string', default: '30000' },
    'voxtral-binary': {
      type: 'string',
      default:
        '/private/tmp/cuttledoc-voxtral-mlx-direct-build/' +
        'cuttledoc-voxtral-mlx',
    },
    'voxtral-model-dir': {
      type: 'string',
      default: '/private/tmp/cuttledoc-voxtral-realtime-4b-mlx-4bit',
    },
    'voxtral-max-tokens': { type: 'string', default: '8192' },
  },
});

const command = positionals[0];
if (!['run', 'self-test'].includes(command)) {
  throw new Error(
    'usage: node scripts/run-target-domain-asr.mjs <run|self-test> ' +
      '--output PATH [--source-group ID] ' +
      '[--backend all|apple|whisper|qwen3|parakeet|voxtral] [--allow-test]',
  );
}

try {
  if (command === 'self-test') {
    await runSelfTest();
  } else {
    await run();
  }
} catch (error) {
  process.stderr.write(`target-domain ASR: ${error.message}\n`);
  process.exitCode = 1;
}

async function run() {
  if (!values.output) {
    throw new Error('--output is required');
  }
  const manifestPath = resolve(values.manifest);
  const planPath = resolve(values.plan);
  const audioDirectory = resolve(values['audio-dir']);
  const outputPath = resolve(values.output);
  const manifestBytes = await readFile(manifestPath);
  const planBytes = await readFile(planPath);
  const manifest = JSON.parse(manifestBytes);
  const plan = JSON.parse(planBytes);
  validatePlanAndManifest(plan, manifest);

  const allSourceGroups = manifest.cells.flatMap((cell) =>
    cell.source_groups.map((sourceGroup) => ({ cell, sourceGroup })),
  );
  const selected = values['source-group']
    ? allSourceGroups.find(({ sourceGroup }) =>
        sourceGroup.id === values['source-group'])
    : allSourceGroups.find(({ sourceGroup }) =>
        sourceGroup.split === 'validation');
  if (!selected) {
    throw new Error(
      values['source-group']
        ? `unknown source group: ${values['source-group']}`
        : 'no validation source group is available',
    );
  }
  const { cell, sourceGroup } = selected;
  if (sourceGroup.split === 'test' && !values['allow-test']) {
    throw new Error(
      'test sources remain closed; pass --allow-test only after the ' +
        'validation configuration is frozen',
    );
  }
  if (sourceGroup.split === 'test' &&
      outputPath.startsWith(`${join(repoRoot, 'benchmarks')}/`)) {
    throw new Error('test ASR drafts must not be written into benchmark evidence');
  }

  const backendIds = backendIdsForOption(values.backend, plan);
  const audioPath = join(
    audioDirectory,
    sourceGroup.normalized_audio.artifact_name,
  );
  const audioBytes = await readFile(audioPath);
  verifyAudio(sourceGroup, audioBytes);
  const provenance = {
    source_revision: gitOutput(['rev-parse', 'HEAD']),
    runner_sha256: digest(await readFile(scriptPath)),
    plan_sha256: digest(planBytes),
    corpus_sha256: digest(manifestBytes),
  };
  const existing = await readJsonIfPresent(outputPath);
  const record = existing ?? newRecord({
    plan,
    manifest,
    cell,
    sourceGroup,
    provenance,
  });
  validateResume(record, {
    plan,
    manifest,
    sourceGroup,
    provenance,
  });

  const completed = new Set(record.results.map((result) => result.backend.id));
  for (const backendId of backendIds) {
    if (completed.has(backendId)) {
      process.stderr.write(`target-domain ASR: verified existing ${backendId}\n`);
      continue;
    }
    process.stderr.write(`target-domain ASR: running ${backendId}\n`);
    const result = await runBackend(backendId, {
      sourceGroup,
      audioPath,
      durationMs: sourceGroup.selection.duration_ms,
    });
    result.transcript.sha256 = digest(Buffer.from(result.transcript.text));
    result.transcript.word_count = words(result.transcript.text).length;
    record.results.push(result);
    orderResults(record, plan.candidate_backends);
    updateStatus(record, plan.candidate_backends);
    await atomicJson(outputPath, record);
    process.stderr.write(
      `target-domain ASR: saved ${backendId} ` +
        `(${result.transcript.word_count} words)\n`,
    );
  }
  updateStatus(record, plan.candidate_backends);
  await atomicJson(outputPath, record);
  process.stdout.write(
    `${JSON.stringify({
      output: outputPath,
      status: record.status,
      completed_backends: record.results.map((result) => result.backend.id),
      missing_backends: record.missing_backends,
      incomplete_coverage_backends: record.results
        .filter((result) => result.coverage?.status !== 'complete')
        .map((result) => result.backend.id),
    }, null, 2)}\n`,
  );
}

function newRecord({
  plan,
  manifest,
  cell,
  sourceGroup,
  provenance,
}) {
  return {
    schema_version: '1.0.0',
    id: `target-domain-asr-drafts--${sourceGroup.id}--1`,
    status: 'partial',
    captured_at: new Date().toISOString(),
    evidence_captured_through: null,
    source_revision: provenance.source_revision,
    runner_sha256: provenance.runner_sha256,
    plan: {
      revision: plan.revision,
      sha256: provenance.plan_sha256,
    },
    corpus: {
      revision: manifest.revision,
      sha256: provenance.corpus_sha256,
    },
    source_group: {
      id: sourceGroup.id,
      cell_id: cell.id,
      title: sourceGroup.title,
      split: sourceGroup.split,
      locale: sourceGroup.locale,
      domain: sourceGroup.domain,
      speakers: sourceGroup.speakers,
      source_range: {
        start_ms: sourceGroup.selection.start_ms,
        end_ms: sourceGroup.selection.end_ms,
        duration_ms: sourceGroup.selection.duration_ms,
      },
      attribution: sourceGroup.attribution,
    },
    audio: {
      artifact_name: sourceGroup.normalized_audio.artifact_name,
      availability: 'local-required',
      local_path: null,
      sha256: sourceGroup.normalized_audio.sha256,
      bytes: sourceGroup.normalized_audio.bytes,
      sample_count: sourceGroup.normalized_audio.sample_count,
      sample_rate_hz: 16_000,
      channels: 1,
      sample_format: 'float32le',
    },
    host: hostMetadata(),
    procedure: {
      purpose:
        'Generate model-specific draft transcripts for independent gold review ' +
        'and later held-out transcript-enhancement evaluation.',
      backend_order: plan.candidate_backends,
      repetitions: 1,
      performance_scope:
        'Single observational draft-generation pass with warm local caches; ' +
        'not a latency or energy comparison.',
      test_policy:
        'Validation may prepare the configuration. Test sources stay closed ' +
        'until that configuration is frozen.',
      gold_policy:
        'No ASR output is gold. A separate human must listen to the complete ' +
        'digest-pinned audio and verify every turn.',
    },
    results: [],
    missing_backends: [...plan.candidate_backends],
    claim_limit:
      'Unscored validation drafts only. Human gold is pending, so this record ' +
      'cannot report WER, rank ASR backends, evaluate postprocessing, or select ' +
      'a product model.',
  };
}

function validateResume(record, {
  plan,
  manifest,
  sourceGroup,
  provenance,
}) {
  if (record.schema_version !== '1.0.0' ||
      record.source_group?.id !== sourceGroup.id ||
      record.source_group?.split !== sourceGroup.split ||
      record.audio?.sha256 !== sourceGroup.normalized_audio.sha256 ||
      record.audio?.bytes !== sourceGroup.normalized_audio.bytes ||
      record.plan?.revision !== plan.revision ||
      record.plan?.sha256 !== provenance.plan_sha256 ||
      record.corpus?.revision !== manifest.revision ||
      record.corpus?.sha256 !== provenance.corpus_sha256 ||
      record.runner_sha256 !== provenance.runner_sha256) {
    throw new Error('existing output does not match this runner, corpus, or plan');
  }
  const ids = record.results?.map((result) => result.backend?.id) ?? [];
  if (new Set(ids).size !== ids.length ||
      ids.some((id) => !plan.candidate_backends.includes(id))) {
    throw new Error('existing output contains duplicate or unknown backends');
  }
  for (const result of record.results) {
    if (!/^[0-9a-f]{64}$/.test(result.transcript?.sha256 ?? '') ||
        digest(Buffer.from(result.transcript?.text ?? '')) !==
          result.transcript.sha256 ||
        !['complete', 'incomplete'].includes(result.coverage?.status)) {
      throw new Error(
        `${result.backend?.id}: transcript digest or coverage mismatch`,
      );
    }
  }
}

function validatePlanAndManifest(plan, manifest) {
  if (plan.revision !== 'target-domain-held-out-3' ||
      manifest.plan_revision !== plan.revision ||
      manifest.revision !== 'target-domain-corpus-de-podcast-2') {
    throw new Error('expected the target-domain held-out plan 3 corpus');
  }
  const expected = [
    'apple-speechtranscriber',
    'whisper-large-v3-turbo-coreml-whispercpp',
    'qwen3-asr-0.6b-mlx-direct',
    'parakeet-tdt-0.6b-v3-coreml',
    'voxtral-realtime-4b-mlx-direct-2400ms',
  ];
  if (JSON.stringify(plan.candidate_backends) !== JSON.stringify(expected)) {
    throw new Error('target-domain backend order is not frozen');
  }
}

function backendIdsForOption(option, plan) {
  const mapping = {
    apple: 'apple-speechtranscriber',
    whisper: 'whisper-large-v3-turbo-coreml-whispercpp',
    qwen3: 'qwen3-asr-0.6b-mlx-direct',
    parakeet: 'parakeet-tdt-0.6b-v3-coreml',
    voxtral: 'voxtral-realtime-4b-mlx-direct-2400ms',
  };
  if (option === 'all') return [...plan.candidate_backends];
  const id = mapping[option];
  if (!id) {
    throw new Error(
      '--backend must be all, apple, whisper, qwen3, parakeet, or voxtral',
    );
  }
  return [id];
}

async function runBackend(id, context) {
  if (id === 'apple-speechtranscriber') {
    return runApple(context);
  }
  if (id === 'whisper-large-v3-turbo-coreml-whispercpp') {
    return runLegacy('whisper', context);
  }
  if (id === 'qwen3-asr-0.6b-mlx-direct') {
    return runQwen(context);
  }
  if (id === 'parakeet-tdt-0.6b-v3-coreml') {
    return runLegacy('parakeet', context);
  }
  if (id === 'voxtral-realtime-4b-mlx-direct-2400ms') {
    return runVoxtral(context);
  }
  throw new Error(`unsupported backend: ${id}`);
}

function runApple({ sourceGroup, audioPath, durationMs }) {
  const binary = resolve(
    values['apple-build-dir'],
    'cuttledoc-speech-spike',
  );
  const timed = timedCommand(binary, [
    audioPath,
    '--locale',
    sourceGroup.locale,
  ], {
    ...process.env,
    DYLD_LIBRARY_PATH: resolve(values['apple-build-dir']),
  });
  const summary = prefixedJson(timed.stdout, 'SESSION_SUMMARY ');
  const updates = timed.stdout
    .split('\n')
    .filter((line) => line.startsWith('UPDATE '))
    .map((line) => JSON.parse(line.slice('UPDATE '.length)));
  const finalUpdates = updates.filter((update) => update.stability === 'final');
  const text = finalUpdates.map((update) => update.text).join('');
  const segments = finalUpdates.flatMap((update) =>
    (update.segments ?? []).map(normalizeAppleSegment));
  if (text.trim().length === 0) {
    throw new Error('Apple SpeechTranscriber returned no final text');
  }
  return resultRecord({
    id: 'apple-speechtranscriber',
    capturedAt: new Date().toISOString(),
    model:
      'macOS system-managed SpeechTranscriber assets for ' +
      sourceGroup.locale,
    runtime:
      'macOS SpeechAnalyzer through the repository-owned Swift C ABI',
    boundary: 'repository-owned Swift task ABI called from Rust',
    text,
    segments,
    coverage: timestampCoverage(segments, durationMs),
    timing: {
      wall_ms: timed.wallMs,
      backend_ms: summary.elapsed_ms,
      first_result_ms: summary.first_result_ms,
      real_time_factor: summary.elapsed_ms / durationMs,
    },
    resources: {},
    streaming: {
      update_count: summary.update_count,
      final_update_count: summary.final_update_count,
      volatile_update_count: summary.volatile_update_count,
      revoke_count: summary.revoke_count,
    },
  });
}

async function runLegacy(name, {
  sourceGroup,
  audioPath,
  durationMs,
}) {
  const pcmBytes = await readFile(audioPath);
  const owned = pcmBytes.buffer.slice(
    pcmBytes.byteOffset,
    pcmBytes.byteOffset + pcmBytes.byteLength,
  );
  const samples = new Float32Array(owned);
  const require = createRequire(import.meta.url);
  const moduleDirectory = resolve(
    values[name === 'whisper'
      ? 'whisper-module-dir'
      : 'parakeet-module-dir'],
  );
  const modelDirectory = resolve(
    values[name === 'whisper'
      ? 'whisper-model-dir'
      : 'parakeet-model-dir'],
  );
  const api = require(join(moduleDirectory, 'dist/index.cjs'));
  const engine = name === 'whisper'
    ? new api.WhisperAsrEngine({
        modelPath: join(modelDirectory, 'ggml-large-v3-turbo.bin'),
        language: sourceGroup.locale.slice(0, 2),
        useGpu: true,
      })
    : new api.ParakeetAsrEngine({
        modelDir: modelDirectory,
        vadDir: resolve(values['parakeet-vad-dir']),
        autoDownload: false,
      });
  let version;
  let loadMs;
  let inferenceMs;
  let native;
  try {
    const loadStarted = performance.now();
    await engine.initialize();
    loadMs = performance.now() - loadStarted;
    version = engine.getVersion();
    const inferenceStarted = performance.now();
    native = name === 'whisper'
      ? await engine.transcribe(samples, 16_000)
      : await engine.transcribe(samples, { sampleRate: 16_000 });
    inferenceMs = performance.now() - inferenceStarted;
  } finally {
    engine.cleanup();
  }
  if (!(native?.text?.trim().length > 0)) {
    throw new Error(`${name} returned no text`);
  }
  const segments = (native.segments ?? []).map((segment) =>
    normalizeLegacySegment(segment, durationMs));
  return resultRecord({
    id: name === 'whisper'
      ? 'whisper-large-v3-turbo-coreml-whispercpp'
      : 'parakeet-tdt-0.6b-v3-coreml',
    capturedAt: new Date().toISOString(),
    model: name === 'whisper'
      ? 'whisper large-v3-turbo Core ML encoder plus whisper.cpp decoder'
      : 'FluidInference parakeet-tdt-0.6b-v3 Core ML plus Silero VAD',
    runtime:
      `${name === 'whisper' ? 'whisper-coreml' : 'parakeet-coreml'} ` +
      `Node adapter @${version}`,
    boundary: 'existing legacy Core ML Node adapter in a normal host process',
    text: native.text,
    segments,
    coverage: timestampCoverage(segments, durationMs),
    timing: {
      wall_ms: inferenceMs,
      load_ms: loadMs,
      backend_ms: native.durationMs,
      real_time_factor: inferenceMs / durationMs,
    },
    resources: {
      process_maximum_resident_set_size_bytes:
        process.resourceUsage().maxRSS * 1024,
    },
    streaming: {
      update_count: 1,
      final_update_count: 1,
      volatile_update_count: 0,
      revoke_count: 0,
    },
  });
}

async function runQwen({ sourceGroup, audioPath, durationMs }) {
  const chunkMs = Number.parseInt(values['qwen-chunk-ms'], 10);
  if (!Number.isInteger(chunkMs) || chunkMs < 5_000 || chunkMs > 60_000) {
    throw new Error('--qwen-chunk-ms must be 5000..60000');
  }
  const manifest = JSON.parse(await readFile(
    join(repoRoot, 'spikes/qwen3-mlx-direct/model-manifest.json'),
  ));
  const audioBytes = await readFile(audioPath);
  const bytesPerMillisecond = 16_000 * 4 / 1_000;
  const chunkCount = Math.ceil(durationMs / chunkMs);
  const temporaryDirectory = await mkdtemp(
    join(tmpdir(), 'cuttledoc-target-qwen-'),
  );
  const chunks = [];
  try {
    for (let index = 0; index < chunkCount; index += 1) {
      const startMs = index * chunkMs;
      const endMs = Math.min(startMs + chunkMs, durationMs);
      const startByte = startMs * bytesPerMillisecond;
      const endByte = endMs * bytesPerMillisecond;
      const chunkBytes = audioBytes.subarray(startByte, endByte);
      const chunkPath = join(
        temporaryDirectory,
        `chunk-${String(index).padStart(3, '0')}.f32le`,
      );
      await writeFile(chunkPath, chunkBytes);
      process.stderr.write(
        `target-domain ASR: Qwen chunk ${index + 1}/${chunkCount}\n`,
      );
      const timed = timedCommand(resolve(values['qwen-binary']), [
        'transcribe',
        resolve(values['qwen-model-dir']),
        chunkPath,
        sourceGroup.locale.slice(0, 2),
        'gpu',
      ]);
      const native = JSON.parse(timed.stdout);
      if (!(native.text?.trim().length > 0)) {
        throw new Error(`Qwen3-ASR returned no text for chunk ${index}`);
      }
      chunks.push({
        index,
        start_ms: startMs,
        end_ms: endMs,
        audio_sha256: digest(chunkBytes),
        transcript_sha256: digest(Buffer.from(native.text)),
        text: native.text,
        token_count: native.generation_tokens,
        finish_reason: native.finish_reason,
        stop_token: native.stop_token,
        backend_ms: native.elapsed_ms,
        wall_ms: timed.wallMs,
        mlx_peak_memory_bytes: native.peak_memory_bytes,
      });
    }
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
  const truncatedChunks = chunks
    .filter((chunk) => chunk.finish_reason === 'length')
    .map((chunk) => chunk.index);
  const text = chunks.map((chunk) => chunk.text.trim()).join('\n');
  const backendMs = sum(chunks.map((chunk) => chunk.backend_ms));
  return resultRecord({
    id: 'qwen3-asr-0.6b-mlx-direct',
    capturedAt: new Date().toISOString(),
    model:
      `${manifest.conversion.repository}@${manifest.conversion.revision}`,
    runtime:
      'repository-owned Qwen3-ASR adapter over official MLX 0.32.0',
    boundary: 'repository-owned C++ task ABI over official MLX',
    text,
    segments: chunks.map((chunk) => ({
      start_ms: chunk.start_ms,
      end_ms: chunk.end_ms,
      text: chunk.text,
      confidence: null,
    })),
    coverage: {
      status: truncatedChunks.length === 0 ? 'complete' : 'incomplete',
      method: 'deterministic-fixed-nonoverlap-chunks',
      expected_duration_ms: durationMs,
      processed_duration_ms:
        sum(chunks.map((chunk) => chunk.end_ms - chunk.start_ms)),
      tolerance_ms: 0,
      incomplete_reason: truncatedChunks.length === 0
        ? null
        : `generation limit reached in chunks: ${truncatedChunks.join(', ')}`,
    },
    timing: {
      wall_ms: sum(chunks.map((chunk) => chunk.wall_ms)),
      backend_ms: backendMs,
      real_time_factor: backendMs / durationMs,
    },
    resources: {
      mlx_peak_memory_bytes: Math.max(
        ...chunks.map((chunk) => chunk.mlx_peak_memory_bytes),
      ),
    },
    streaming: {
      update_count: chunks.length,
      final_update_count: chunks.length,
      volatile_update_count: 0,
      revoke_count: 0,
    },
    generation: {
      method: 'fixed-nonoverlap-v1',
      chunk_ms: chunkMs,
      chunk_count: chunks.length,
      overlap_ms: 0,
      merge: 'concatenate-in-source-order',
      token_count: sum(chunks.map((chunk) => chunk.token_count)),
      finish_reason:
        truncatedChunks.length === 0 ? 'all_chunks_eos' : 'chunk_length',
      truncated_chunks: truncatedChunks,
      limitation:
        'Words crossing a fixed chunk boundary can be split or omitted; the ' +
        'complete transcript still requires independent human review.',
      chunks: chunks.map(({
        text: _text,
        mlx_peak_memory_bytes: _peakMemory,
        ...chunk
      }) => chunk),
    },
  });
}

async function runVoxtral({ audioPath, durationMs }) {
  const maxTokens = Number.parseInt(values['voxtral-max-tokens'], 10);
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 8192) {
    throw new Error('--voxtral-max-tokens must be 1..8192');
  }
  const manifest = JSON.parse(await readFile(
    join(repoRoot, 'spikes/voxtral-realtime-mlx-direct/model-manifest.json'),
  ));
  const timed = timedCommand(resolve(values['voxtral-binary']), [
    'transcribe',
    resolve(values['voxtral-model-dir']),
    audioPath,
    '2400',
    String(maxTokens),
    'gpu',
  ]);
  const native = JSON.parse(timed.stdout);
  if (!(native.generation?.text?.trim().length > 0)) {
    throw new Error('Voxtral Realtime returned no text');
  }
  return resultRecord({
    id: 'voxtral-realtime-4b-mlx-direct-2400ms',
    capturedAt: new Date().toISOString(),
    model:
      `${manifest.conversion.repository}@${manifest.conversion.revision}`,
    runtime:
      `${manifest.official_runtime.repository} ` +
      `${manifest.official_runtime.version}@` +
      `${manifest.official_runtime.revision}`,
    boundary: manifest.official_runtime.boundary,
    text: native.generation.text,
    segments: [],
    coverage: {
      status:
        native.generation.finish_reason === 'max_tokens'
          ? 'incomplete'
          : 'complete',
      method: 'offline-streaming-audio-end',
      expected_duration_ms: durationMs,
      processed_duration_ms:
        native.generation.finish_reason === 'max_tokens' ? null : durationMs,
      tolerance_ms: 0,
      incomplete_reason:
        native.generation.finish_reason === 'max_tokens'
          ? `generation stopped at the ${maxTokens}-token safety limit`
          : null,
    },
    timing: {
      wall_ms: timed.wallMs,
      backend_ms: native.elapsed_ms,
      real_time_factor: native.elapsed_ms / durationMs,
    },
    resources: {
      mlx_peak_memory_bytes: native.peak_memory_bytes,
    },
    streaming: {
      update_count: 1,
      final_update_count: 1,
      volatile_update_count: 0,
      revoke_count: 0,
    },
    generation: {
      token_count: native.generation.token_count,
      finish_reason: native.generation.finish_reason,
      transcription_delay_ms: native.transcription_delay_ms,
      max_tokens: maxTokens,
    },
  });
}

function resultRecord({
  id,
  capturedAt,
  model,
  runtime,
  boundary,
  text,
  segments,
  timing,
  resources,
  streaming,
  generation,
  coverage,
}) {
  return {
    captured_at: capturedAt,
    backend: { id, model, runtime, boundary },
    transcript: {
      text,
      sha256: null,
      word_count: null,
      segment_count: segments.length,
      segments,
    },
    timing,
    resources,
    streaming,
    coverage,
    ...(generation ? { generation } : {}),
    quality: {
      status: 'unscored-human-gold-pending',
      wer: null,
      cer: null,
      semantic_errors: null,
    },
  };
}

function normalizeAppleSegment(segment) {
  return {
    start_ms: segment.start_ms ?? null,
    end_ms: segment.end_ms ?? null,
    text: segment.text,
    confidence: segment.confidence ?? null,
  };
}

function normalizeLegacySegment(segment, durationMs) {
  if ('startMs' in segment) {
    return {
      start_ms: segment.startMs,
      end_ms: segment.endMs,
      text: segment.text,
      confidence: segment.confidence ?? null,
    };
  }
  return {
    start_ms: segment.startTime * 1000,
    end_ms: Math.min(segment.endTime * 1000, durationMs),
    text: segment.text,
    confidence: null,
  };
}

function timestampCoverage(segments, durationMs) {
  const finalSegmentEndMs = maximumSegmentEnd(segments);
  const toleranceMs = 2_000;
  const complete =
    finalSegmentEndMs !== null &&
    finalSegmentEndMs >= durationMs - toleranceMs;
  return {
    status: complete ? 'complete' : 'incomplete',
    method: 'final-segment-timestamp',
    expected_duration_ms: durationMs,
    processed_duration_ms: finalSegmentEndMs,
    final_segment_end_ms: finalSegmentEndMs,
    tolerance_ms: toleranceMs,
    incomplete_reason: complete
      ? null
      : 'no final segment reaches the end-of-audio tolerance',
  };
}

function maximumSegmentEnd(segments) {
  const ends = segments
    .map((segment) => segment.end_ms)
    .filter((end) => Number.isFinite(end));
  return ends.length === 0 ? null : Math.max(...ends);
}

function verifyAudio(sourceGroup, bytes) {
  if (bytes.length !== sourceGroup.normalized_audio.bytes ||
      digest(bytes) !== sourceGroup.normalized_audio.sha256 ||
      bytes.length % 4 !== 0 ||
      bytes.length / 4 !== sourceGroup.normalized_audio.sample_count) {
    throw new Error(`${sourceGroup.id}: normalized audio digest or shape differs`);
  }
}

function updateStatus(record, expectedBackends) {
  const completed = new Set(record.results.map((result) => result.backend.id));
  record.missing_backends = expectedBackends.filter((id) => !completed.has(id));
  if (record.missing_backends.length > 0) {
    record.status = 'partial';
  } else if (
    record.results.every((result) => result.coverage?.status === 'complete')
  ) {
    record.status = 'complete';
  } else {
    record.status = 'complete-with-incomplete-drafts';
  }
  record.evidence_captured_through =
    record.results.at(-1)?.captured_at ?? null;
}

function orderResults(record, expectedBackends) {
  const order = new Map(expectedBackends.map((id, index) => [id, index]));
  record.results.sort(
    (left, right) =>
      order.get(left.backend.id) - order.get(right.backend.id),
  );
}

function hostMetadata() {
  return {
    id: 'mac-studio-m1-ultra-local',
    chip:
      optionalCommandOutput('sysctl', ['-n', 'machdep.cpu.brand_string']) ??
      cpus()[0]?.model ??
      'unknown',
    memory_bytes: totalmem(),
    os:
      `${commandOutput('sw_vers', ['-productName']).trim()} ` +
      `${commandOutput('sw_vers', ['-productVersion']).trim()} ` +
      `(${commandOutput('sw_vers', ['-buildVersion']).trim()})`,
    architecture: process.arch,
    power_state: 'unknown',
  };
}

function optionalCommandOutput(command, arguments_) {
  try {
    return commandOutput(command, arguments_).trim() || null;
  } catch {
    return null;
  }
}

function timedCommand(command, arguments_, environment = process.env) {
  const started = performance.now();
  const result = spawnSync(command, arguments_, {
    cwd: repoRoot,
    env: environment,
    encoding: 'utf8',
    maxBuffer: 128 * 1024 * 1024,
  });
  const wallMs = performance.now() - started;
  if (result.status !== 0) {
    throw new Error(
      `${command} failed with ${result.status}: ` +
        `${result.stderr?.trim() || result.error?.message || '<no stderr>'}`,
    );
  }
  return { stdout: result.stdout, stderr: result.stderr, wallMs };
}

function commandOutput(command, arguments_) {
  const result = spawnSync(command, arguments_, {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} failed with ${result.status}: ` +
        `${result.stderr?.trim() || result.error?.message || '<no stderr>'}`,
    );
  }
  return result.stdout;
}

function gitOutput(arguments_) {
  return commandOutput('git', arguments_).trim();
}

function prefixedJson(output, prefix) {
  const line = output
    .split('\n')
    .find((candidate) => candidate.startsWith(prefix));
  if (!line) throw new Error(`missing ${prefix.trim()} output`);
  return JSON.parse(line.slice(prefix.length));
}

async function atomicJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.partial-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporary, path);
}

async function readJsonIfPresent(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

function words(text) {
  return text
    .toLocaleLowerCase('de-DE')
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .split(/\s+/)
    .filter(Boolean);
}

function sum(numbers) {
  return numbers.reduce((total, number) => total + number, 0);
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function runSelfTest() {
  const plan = {
    candidate_backends: [
      'apple-speechtranscriber',
      'whisper-large-v3-turbo-coreml-whispercpp',
      'qwen3-asr-0.6b-mlx-direct',
      'parakeet-tdt-0.6b-v3-coreml',
      'voxtral-realtime-4b-mlx-direct-2400ms',
    ],
  };
  if (backendIdsForOption('all', plan).length !== 5 ||
      backendIdsForOption('voxtral', plan)[0] !==
        'voxtral-realtime-4b-mlx-direct-2400ms') {
    throw new Error('backend selection self-test failed');
  }
  const bytes = Buffer.alloc(16);
  const sourceGroup = {
    id: 'self-test',
    normalized_audio: {
      bytes: bytes.length,
      sha256: digest(bytes),
      sample_count: bytes.length / 4,
    },
  };
  verifyAudio(sourceGroup, bytes);
  const record = {
    results: [
      { backend: { id: plan.candidate_backends[0] }, captured_at: 'now' },
    ],
  };
  updateStatus(record, plan.candidate_backends);
  if (record.status !== 'partial' || record.missing_backends.length !== 4) {
    throw new Error('resume status self-test failed');
  }
  const incompleteRecord = {
    results: plan.candidate_backends.map((id) => ({
      backend: { id },
      captured_at: 'now',
      coverage: {
        status: id === plan.candidate_backends[2]
          ? 'incomplete'
          : 'complete',
      },
    })),
  };
  updateStatus(incompleteRecord, plan.candidate_backends);
  if (incompleteRecord.status !== 'complete-with-incomplete-drafts' ||
      incompleteRecord.missing_backends.length !== 0) {
    throw new Error('incomplete coverage status self-test failed');
  }
  const completeRecord = {
    results: plan.candidate_backends.map((id) => ({
      backend: { id },
      captured_at: 'now',
      coverage: { status: 'complete' },
    })),
  };
  updateStatus(completeRecord, plan.candidate_backends);
  if (completeRecord.status !== 'complete') {
    throw new Error('complete coverage status self-test failed');
  }
  let rejected = false;
  try {
    backendIdsForOption('unknown', plan);
  } catch {
    rejected = true;
  }
  if (!rejected) throw new Error('unknown backend was accepted');
  process.stdout.write('target-domain ASR runner: self-test passed\n');
}
