#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { basename, dirname, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    result: { type: 'string' },
    audio: { type: 'string' },
    reference: { type: 'string' },
    output: { type: 'string' },
    'whisper-module-dir': { type: 'string' },
    'whisper-model-dir': { type: 'string' },
    'parakeet-module-dir': { type: 'string' },
    'parakeet-model-dir': { type: 'string' },
    'parakeet-vad-dir': { type: 'string' },
    'apple-build-dir': { type: 'string' },
    'qwen-binary': { type: 'string' },
    'qwen-model-dir': { type: 'string' },
    'voxtral-binary': { type: 'string' },
    'voxtral-model-dir': { type: 'string' },
  },
});

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const resultPath = requiredPath(values.result, '--result');
const audioPath = requiredPath(values.audio, '--audio');
const referencePath = requiredPath(values.reference, '--reference');
const outputPath = requiredPath(values.output, '--output');
const ttsRun = JSON.parse(await readFile(resultPath, 'utf8'));
const locale = ttsRun.input?.locale;
if (!/^[a-z]{2}-(?:[A-Z]{2}|\d{3})$/.test(locale ?? '')) {
  throw new Error(`TTS run has no supported BCP 47 locale: ${locale}`);
}
const language = locale.slice(0, 2);
const reference = (await readFile(referencePath, 'utf8')).trim();
const referenceDigest = createHash('sha256').update(reference).digest('hex');
if (referenceDigest !== ttsRun.input?.text_sha256) {
  throw new Error(
    `reference SHA-256 differs from the TTS run: ${referenceDigest}`,
  );
}
if (ttsRun.result?.audio?.sample_format !== 'f32le' ||
    ttsRun.result?.audio?.channel_count !== 1) {
  throw new Error('TTS run must describe mono f32le audio');
}
const sourceAudio = await readFile(audioPath);
const sourceDigest = createHash('sha256').update(sourceAudio).digest('hex');
if (sourceDigest !== ttsRun.result.audio.sha256 ||
    sourceAudio.length !== ttsRun.result.audio.byte_count) {
  throw new Error('source audio differs from the TTS run record');
}

const paths = {
  whisperModule: localPath(
    values['whisper-module-dir'],
    '/Users/sebastian/Workspace/whisper-coreml',
  ),
  whisperModel: localPath(
    values['whisper-model-dir'],
    '/Users/sebastian/.cache/whisper-coreml/models',
  ),
  parakeetModule: localPath(
    values['parakeet-module-dir'],
    '/Users/sebastian/Workspace/parakeet-coreml',
  ),
  parakeetModel: localPath(
    values['parakeet-model-dir'],
    '/Users/sebastian/.cache/parakeet-coreml/models',
  ),
  parakeetVad: localPath(
    values['parakeet-vad-dir'],
    '/Users/sebastian/.cache/parakeet-coreml/vad',
  ),
  appleBuild: localPath(
    values['apple-build-dir'],
    '/private/tmp/cuttledoc-qwen3-tts-apple-asr-build',
  ),
  qwenBinary: localPath(
    values['qwen-binary'],
    '/private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect',
  ),
  qwenModel: localPath(
    values['qwen-model-dir'],
    '/private/tmp/cuttledoc-qwen3-asr/model',
  ),
  voxtralBinary: localPath(
    values['voxtral-binary'],
    '/private/tmp/cuttledoc-voxtral-mlx-direct-build/cuttledoc-voxtral-mlx',
  ),
  voxtralModel: localPath(
    values['voxtral-model-dir'],
    '/private/tmp/cuttledoc-voxtral-realtime-4b-mlx-4bit',
  ),
};

const temporaryDirectory = await mkdtemp('/private/tmp/cuttledoc-tts-asr-');
try {
  const normalizedPath = join(temporaryDirectory, 'normalized-16k.f32le');
  commandOutput('ffmpeg', [
    '-y',
    '-v',
    'error',
    '-f',
    'f32le',
    '-ar',
    String(ttsRun.result.audio.sample_rate_hz),
    '-ac',
    '1',
    '-i',
    audioPath,
    '-ar',
    '16000',
    '-ac',
    '1',
    '-f',
    'f32le',
    '-acodec',
    'pcm_f32le',
    normalizedPath,
  ]);
  const normalizedBytes = await readFile(normalizedPath);
  if (normalizedBytes.length === 0 || normalizedBytes.length % 4 !== 0) {
    throw new Error('FFmpeg returned invalid normalized PCM');
  }
  const normalized = {
    sample_format: 'f32le',
    sample_rate_hz: 16_000,
    channel_count: 1,
    sample_count: normalizedBytes.length / 4,
    byte_count: normalizedBytes.length,
    duration_ms: normalizedBytes.length / 4 / 16,
    sha256: createHash('sha256').update(normalizedBytes).digest('hex'),
  };

  const backends = [];
  backends.push(await runLegacy('whisper', normalizedPath, temporaryDirectory));
  backends.push(await runLegacy('parakeet', normalizedPath, temporaryDirectory));
  backends.push(runQwen(normalizedPath));
  backends.push(runVoxtral(normalizedPath));
  backends.push(runApple(normalizedPath));

  const lowestWer = Math.min(...backends.map((item) => item.quality.wer));
  ttsRun.result.asr_content_checks = {
    status: 'complete',
    normalized_audio: normalized,
    normalization:
      'Unicode lowercase; punctuation and whitespace removed consistently with the phase-0 matrix.',
    backends,
    comparison: {
      completed_backend_count: backends.length,
      expected_backend_count: 5,
      remaining_backends: [],
      lowest_observed_wer_backends: backends
        .filter((item) => item.quality.wer === lowestWer)
        .map((item) => item.backend.id),
      observations: [
        'All five required ASR backends received the identical normalized PCM digest.',
        'WER and CER are receiver-specific roundtrip diagnostics, not ' +
          'standalone TTS quality scores.',
        'Listening review remains required for pronunciation, voice identity, and prosody.',
      ],
    },
  };
  ttsRun.conclusion.next =
    'Review pronunciation and prosody, then apply the recorded calibration gate.';
  await writeFile(outputPath, `${JSON.stringify(ttsRun, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify({
      output: outputPath,
      normalized_audio: normalized,
      wer: Object.fromEntries(
        backends.map((item) => [item.backend.id, item.quality.wer]),
      ),
    }, null, 2)}\n`,
  );
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}

async function runLegacy(backend, normalizedPath, temporaryDirectory) {
  const outputPath = join(temporaryDirectory, `${backend}.json`);
  const arguments_ = [
    join(repoRoot, 'scripts/run-legacy-asr-baseline.mjs'),
    '--backend',
    backend,
    '--fixture',
    normalizedPath,
    '--fixture-format',
    'f32le',
    '--reference',
    referencePath,
    '--module-dir',
    backend === 'whisper' ? paths.whisperModule : paths.parakeetModule,
    '--model-dir',
    backend === 'whisper' ? paths.whisperModel : paths.parakeetModel,
    '--language',
    language,
    '--repetitions',
    '1',
    '--output',
    outputPath,
  ];
  if (backend === 'parakeet') {
    arguments_.push('--vad-dir', paths.parakeetVad);
  }
  commandOutput(process.execPath, arguments_);
  const native = JSON.parse(await readFile(outputPath, 'utf8'));
  const representative = native.repetitions.at(-1);
  return contentCheck({
    id:
      backend === 'whisper'
        ? 'whisper-large-v3-turbo-coreml-whispercpp'
        : 'parakeet-tdt-0.6b-v3-coreml',
    locale: backend === 'whisper' ? language : 'model-managed',
    execution: 'existing legacy CoreML Node adapter in a normal host process',
    text: representative.text,
    segmentCount: representative.segments.length,
    updateCount: 1,
    finalUpdateCount: 1,
    volatileUpdateCount: 0,
    revokeCount: 0,
    timing: {
      cold_load_ms: native.metrics.cold_load_ms,
      complete_inference_ms: native.metrics.warm_inference_ms,
      real_time_factor:
        native.metrics.warm_inference_ms / native.fixture.audio_duration_ms,
    },
    resources: {
      model_bytes: native.metrics.model_size_bytes,
      vad_bytes: native.metrics.vad_size_bytes || undefined,
      binary_bytes: native.metrics.binary_size_bytes,
      maximum_resident_set_size_bytes: native.metrics.peak_memory_bytes,
    },
  });
}

function runQwen(normalizedPath) {
  const native = JSON.parse(commandOutput(paths.qwenBinary, [
    'transcribe',
    paths.qwenModel,
    normalizedPath,
    language,
    'gpu',
  ]));
  return contentCheck({
    id: 'qwen3-asr-0.6b-mlx-direct',
    locale: native.language,
    execution: 'repository-owned C++ task ABI over official MLX',
    text: native.text,
    segmentCount: 1,
    updateCount: 1,
    finalUpdateCount: 1,
    volatileUpdateCount: 0,
    revokeCount: 0,
    timing: {
      complete_inference_ms: native.elapsed_ms,
      real_time_factor: native.elapsed_ms / ttsRun.result.audio.duration_ms,
    },
    resources: { mlx_peak_memory_bytes: native.peak_memory_bytes },
    generation: {
      token_count: native.generation_tokens,
      finish_reason: native.finish_reason,
      stop_token: native.stop_token,
    },
  });
}

function runVoxtral(normalizedPath) {
  const native = JSON.parse(commandOutput(paths.voxtralBinary, [
    'transcribe',
    paths.voxtralModel,
    normalizedPath,
    '2400',
    '1536',
    'gpu',
  ]));
  return contentCheck({
    id: 'voxtral-realtime-4b-mlx-direct-2400ms',
    locale: 'model-managed',
    execution: 'repository-owned C++ task ABI over official MLX',
    text: native.generation.text,
    segmentCount: 1,
    updateCount: 1,
    finalUpdateCount: 1,
    volatileUpdateCount: 0,
    revokeCount: 0,
    timing: {
      complete_inference_ms: native.elapsed_ms,
      real_time_factor: native.elapsed_ms / ttsRun.result.audio.duration_ms,
    },
    resources: { mlx_peak_memory_bytes: native.peak_memory_bytes },
    generation: {
      token_count: native.generation.token_count,
      finish_reason: native.generation.finish_reason,
      transcription_delay_ms: native.transcription_delay_ms,
    },
  });
}

function runApple(normalizedPath) {
  const binary = join(paths.appleBuild, 'cuttledoc-speech-spike');
  const output = commandOutput(binary, [normalizedPath, '--locale', locale], {
    ...process.env,
    DYLD_LIBRARY_PATH: paths.appleBuild,
  });
  const summary = prefixedJson(output, 'SESSION_SUMMARY ');
  const updates = output
    .split('\n')
    .filter((line) => line.startsWith('UPDATE '))
    .map((line) => JSON.parse(line.slice('UPDATE '.length)));
  const finalUpdates = updates.filter((update) => update.stability === 'final');
  const text = finalUpdates.map((update) => update.text).join('');
  return contentCheck({
    id: 'apple-speechtranscriber',
    locale,
    execution: 'repository-owned Rust/Swift task ABI in a normal host process',
    text,
    segmentCount: finalUpdates.reduce(
      (sum, update) => sum + update.segments.length,
      0,
    ),
    updateCount: summary.update_count,
    finalUpdateCount: summary.final_update_count,
    volatileUpdateCount: summary.volatile_update_count,
    revokeCount: summary.revoke_count,
    timing: {
      first_result_ms: summary.first_result_ms,
      complete_inference_ms: summary.elapsed_ms,
      real_time_factor: summary.elapsed_ms / ttsRun.result.audio.duration_ms,
    },
  });
}

function contentCheck({
  id,
  locale,
  execution,
  text,
  segmentCount,
  updateCount,
  finalUpdateCount,
  volatileUpdateCount,
  revokeCount,
  timing,
  resources,
  generation,
}) {
  const referenceWords = words(reference);
  const hypothesisWords = words(text);
  const referenceCharacters = referenceWords.join('').split('');
  const hypothesisCharacters = hypothesisWords.join('').split('');
  const wordEdits = distance(referenceWords, hypothesisWords);
  const characterEdits = distance(referenceCharacters, hypothesisCharacters);
  return {
    captured_at: new Date().toISOString(),
    backend: { id, locale, execution },
    transcript: {
      text,
      sha256: createHash('sha256').update(text).digest('hex'),
      segment_count: segmentCount,
      update_count: updateCount,
      final_update_count: finalUpdateCount,
      volatile_update_count: volatileUpdateCount,
      revoke_count: revokeCount,
    },
    quality: {
      reference_word_count: referenceWords.length,
      hypothesis_word_count: hypothesisWords.length,
      word_edits: wordEdits,
      wer: wordEdits / referenceWords.length,
      reference_character_count: referenceCharacters.length,
      hypothesis_character_count: hypothesisCharacters.length,
      character_edits: characterEdits,
      cer: characterEdits / referenceCharacters.length,
    },
    timing,
    ...(resources ? { resources: withoutUndefined(resources) } : {}),
    ...(generation ? { generation } : {}),
  };
}

function commandOutput(command, arguments_, environment = process.env) {
  const result = spawnSync(command, arguments_, {
    cwd: repoRoot,
    env: environment,
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${basename(command)} failed (${result.status}): ${result.stderr || result.stdout}`,
    );
  }
  return result.stdout.trim();
}

function prefixedJson(output, prefix) {
  const line = output.split('\n').find((item) => item.startsWith(prefix));
  if (!line) throw new Error(`missing ${prefix.trim()} in Apple output`);
  return JSON.parse(line.slice(prefix.length));
}

function words(text) {
  return text
    .toLocaleLowerCase('en-US')
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .split(/\s+/)
    .filter(Boolean);
}

function distance(referenceItems, hypothesisItems) {
  let previous = Array.from(
    { length: hypothesisItems.length + 1 },
    (_, index) => index,
  );
  for (
    let referenceIndex = 1;
    referenceIndex <= referenceItems.length;
    referenceIndex += 1
  ) {
    const current = [referenceIndex];
    for (
      let hypothesisIndex = 1;
      hypothesisIndex <= hypothesisItems.length;
      hypothesisIndex += 1
    ) {
      const substitution =
        previous[hypothesisIndex - 1] +
        (referenceItems[referenceIndex - 1] ===
        hypothesisItems[hypothesisIndex - 1]
          ? 0
          : 1);
      current[hypothesisIndex] = Math.min(
        previous[hypothesisIndex] + 1,
        current[hypothesisIndex - 1] + 1,
        substitution,
      );
    }
    previous = current;
  }
  return previous[hypothesisItems.length];
}

function requiredPath(value, option) {
  if (!value) throw new Error(`${option} is required`);
  return resolve(value);
}

function localPath(value, fallback) {
  return resolve(value ?? fallback);
}

function withoutUndefined(value) {
  return Object.fromEntries(
    Object.entries(value).filter(([, item]) => item !== undefined),
  );
}
