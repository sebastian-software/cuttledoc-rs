#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  readFile,
  stat,
  unlink,
  writeFile,
} from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { totalmem } from 'node:os';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({
  options: {
    candidate: { type: 'string' },
    output: { type: 'string' },
    repetitions: { type: 'string', default: '2' },
    'fixture-dir': {
      type: 'string',
      default: resolve(
        repoRoot,
        '../cuttledoc/packages/cuttledoc/fixtures',
      ),
    },
  },
});

const supportedCandidates = new Set([
  'mlx',
  'apple-speech',
  'parakeet',
  'whisper',
]);
if (!supportedCandidates.has(values.candidate)) {
  throw new Error(
    '--candidate must be mlx, apple-speech, parakeet, or whisper',
  );
}
const candidate = values.candidate;
const repetitions = Number.parseInt(values.repetitions, 10);
if (!Number.isInteger(repetitions) || repetitions < 1) {
  throw new Error('--repetitions must be a positive integer');
}

const manifest = JSON.parse(
  await readFile(
    join(repoRoot, 'benchmarks/fixtures/manifest.json'),
    'utf8',
  ),
);
const fixtures = manifest.fixtures.filter(
  (fixture) => fixture.purpose === 'quality',
);
if (fixtures.length !== 10) {
  throw new Error(
    `expected the bounded ten-fixture quality set, found ${fixtures.length}`,
  );
}

const fixtureDirectory = resolve(values['fixture-dir']);
const capturedAt = new Date().toISOString();
const results = [];
for (const fixture of fixtures) {
  const pcmPath = await materializeFixture(fixture);
  try {
    process.stderr.write(`${candidate}: ${fixture.id}\n`);
    results.push(await runCandidate(fixture, pcmPath));
  } finally {
    await unlink(pcmPath).catch(() => {});
  }
}

const record = {
  schema_version: '1.0.0',
  matrix_run_id: `phase0.${candidate}.multilingual-fleurs-10-1`,
  captured_at: capturedAt,
  source_revision: commandOutput('git', ['rev-parse', 'HEAD'], {
    cwd: repoRoot,
  }).trim(),
  fixture_manifest_revision: manifest.revision,
  candidate: candidateMetadata(candidate),
  host: {
    id: 'mac-studio-m1-ultra-local',
    chip: commandOutput('sysctl', ['-n', 'machdep.cpu.brand_string']).trim(),
    memory_bytes: totalmem(),
    os: `${commandOutput('sw_vers', ['-productName']).trim()} ${commandOutput(
      'sw_vers',
      ['-productVersion'],
    ).trim()} (${commandOutput('sw_vers', ['-buildVersion']).trim()})`,
    architecture: process.arch,
    power_state: 'unknown',
  },
  procedure: {
    fixture_count: fixtures.length,
    repetitions,
    fixture_order: fixtures.map((fixture) => fixture.id),
    quality_normalization:
      'Unicode lowercase; punctuation and whitespace removed consistently with the phase-0 single-fixture records',
    cold_load:
      'First candidate-specific create/initialize value in a fresh process; system and model caches are not cleared',
    warm_inference:
      candidate === 'apple-speech'
        ? 'Mean complete streamed inference over fresh processes after one discarded process warm-up'
        : 'Mean complete inference inside the initialized engine after one discarded warm-up',
    energy:
      'Not captured in this breadth pass; the alternating powermetrics procedure remains separate',
  },
  summary: summarize(results),
  results,
};

const serialized = `${JSON.stringify(record, null, 2)}\n`;
if (values.output) {
  const outputPath = resolve(values.output);
  await writeFile(outputPath, serialized);
  process.stderr.write(`Wrote ${outputPath}\n`);
} else {
  process.stdout.write(serialized);
}

async function runCandidate(fixture, pcmPath) {
  switch (candidate) {
    case 'mlx':
      return runMlx(fixture, pcmPath);
    case 'apple-speech':
      return runAppleSpeech(fixture, pcmPath);
    case 'parakeet':
    case 'whisper':
      return runLegacy(fixture);
    default:
      throw new Error(`unsupported candidate: ${candidate}`);
  }
}

async function runMlx(fixture, pcmPath) {
  const probe = requiredEnvironment('CUTTLEDOC_MLX_PROBE');
  const modelDirectory = requiredEnvironment('CUTTLEDOC_MLX_MODEL_DIR');
  const language = languageCode(fixture.language);
  const timed = timedCommand(probe, [
    modelDirectory,
    pcmPath,
    'gpu',
    language,
    '1',
    String(repetitions + 1),
  ]);
  const native = JSON.parse(timed.stdout.trim());
  const session = native.sessions[0];
  const measured = session.runs.slice(1);
  const representative = measured.at(-1).transcription;
  const modelPath = await firstExisting([
    join(modelDirectory, 'downloads/weights.npz'),
    join(modelDirectory, 'weights.npz'),
  ]);
  const probeDirectory = dirname(probe);
  const binarySizeBytes =
    (await stat(join(probeDirectory, 'libcuttledoc_mlx_shim.dylib'))).size +
    (await stat(join(probeDirectory, 'mlx.metallib'))).size;

  return resultRecord({
    fixture,
    text: representative.text,
    reportedLanguage: representative.language,
    coldLoadMs: session.load_wall_ms,
    warmInferenceMs: mean(
      measured.map((sample) => sample.transcription.inference_ms),
    ),
    peakMemoryBytes: timed.maximumResidentSetBytes,
    runtimePeakMemoryBytes: Math.max(
      ...measured.map(
        (sample) => sample.transcription.peak_memory_bytes,
      ),
    ),
    modelSizeBytes: (await stat(modelPath)).size,
    binarySizeBytes,
    streaming: {
      supported: false,
      first_result_ms: null,
      update_count: 1,
      volatile_update_count: 0,
      final_update_count: 1,
      revoke_count: 0,
      timestamps:
        representative.segments.length > 0 ? 'segment' : 'none',
    },
    segments: representative.segments,
    repetitions: measured.map((sample) => ({
      wall_ms: sample.wall_ms,
      inference_ms: sample.transcription.inference_ms,
      encoder_ms: sample.transcription.encoder_ms,
      decoder_ms: sample.transcription.decoder_ms,
      text: sample.transcription.text,
    })),
    warmup: {
      wall_ms: session.runs[0].wall_ms,
      inference_ms: session.runs[0].transcription.inference_ms,
      text: session.runs[0].transcription.text,
    },
  });
}

async function runAppleSpeech(fixture, pcmPath) {
  const probe = requiredEnvironment('CUTTLEDOC_SPEECH_PROBE');
  const libraryDirectory = requiredEnvironment(
    'CUTTLEDOC_SPEECH_LIBRARY_DIR',
  );
  const runs = [];
  for (let index = 0; index < repetitions + 1; index += 1) {
    const timed = timedCommand(
      probe,
      [pcmPath, '--locale', fixture.language],
      {
        env: {
          ...process.env,
          DYLD_LIBRARY_PATH: libraryDirectory,
        },
      },
    );
    runs.push({
      ...parseAppleSpeech(timed.stdout),
      maximum_resident_set_bytes: timed.maximumResidentSetBytes,
    });
  }
  const measured = runs.slice(1);
  const representative = measured.at(-1);
  const binarySizeBytes =
    (await stat(probe)).size +
    (await stat(join(libraryDirectory, 'libcuttledoc_speech_shim.dylib')))
      .size;

  return resultRecord({
    fixture,
    text: representative.final.text,
    reportedLanguage: representative.metadata.locale,
    coldLoadMs: runs[0].create_ms,
    warmInferenceMs: mean(
      measured.map((sample) => sample.summary.elapsed_ms),
    ),
    peakMemoryBytes: Math.max(
      ...measured.map((sample) => sample.maximum_resident_set_bytes),
    ),
    runtimePeakMemoryBytes: null,
    modelSizeBytes: null,
    binarySizeBytes,
    streaming: {
      supported: true,
      first_result_ms: mean(
        measured.map((sample) => sample.summary.first_result_ms),
      ),
      update_count: representative.summary.update_count,
      volatile_update_count:
        representative.summary.volatile_update_count,
      final_update_count: representative.summary.final_update_count,
      revoke_count: representative.summary.revoke_count,
      timestamps:
        representative.final.segments.length > 0 ? 'word' : 'none',
    },
    segments: representative.final.segments,
    repetitions: measured.map((sample) => ({
      create_ms: sample.create_ms,
      inference_ms: sample.summary.elapsed_ms,
      first_result_ms: sample.summary.first_result_ms,
      update_count: sample.summary.update_count,
      text: sample.final.text,
    })),
    warmup: {
      create_ms: runs[0].create_ms,
      inference_ms: runs[0].summary.elapsed_ms,
      first_result_ms: runs[0].summary.first_result_ms,
      text: runs[0].final.text,
    },
  });
}

async function runLegacy(fixture) {
  const moduleDirectory = requiredEnvironment(
    candidate === 'parakeet'
      ? 'CUTTLEDOC_PARAKEET_MODULE_DIR'
      : 'CUTTLEDOC_WHISPER_MODULE_DIR',
  );
  const modelDirectory = requiredEnvironment(
    candidate === 'parakeet'
      ? 'CUTTLEDOC_PARAKEET_MODEL_DIR'
      : 'CUTTLEDOC_WHISPER_MODEL_DIR',
  );
  const vadDirectory =
    candidate === 'parakeet'
      ? requiredEnvironment('CUTTLEDOC_PARAKEET_VAD_DIR')
      : null;
  const language = languageCode(fixture.language);
  const prefix = fixturePrefix(fixture);
  const fixturePath = join(fixtureDirectory, `${prefix}.ogg`);
  const referencePath = join(fixtureDirectory, `${prefix}.txt`);
  const outputPath = `/private/tmp/cuttledoc-${candidate}-${process.pid}-${prefix}.json`;
  const runnerArguments = [
    join(repoRoot, 'scripts/run-legacy-asr-baseline.mjs'),
    '--backend',
    candidate,
    '--fixture',
    fixturePath,
    '--reference',
    referencePath,
    '--module-dir',
    moduleDirectory,
    '--model-dir',
    modelDirectory,
    '--language',
    language,
    '--repetitions',
    String(repetitions),
    '--output',
    outputPath,
  ];
  if (vadDirectory) {
    runnerArguments.push('--vad-dir', vadDirectory);
  }
  commandOutput(process.execPath, runnerArguments, {
    cwd: repoRoot,
    maxBuffer: 128 * 1024 * 1024,
  });
  const native = JSON.parse(await readFile(outputPath, 'utf8'));
  await unlink(outputPath).catch(() => {});
  const representative = native.repetitions.at(-1);

  return resultRecord({
    fixture,
    text: representative.text,
    reportedLanguage: representative.language,
    coldLoadMs: native.metrics.cold_load_ms,
    warmInferenceMs: native.metrics.warm_inference_ms,
    peakMemoryBytes: native.metrics.peak_memory_bytes,
    runtimePeakMemoryBytes: null,
    modelSizeBytes: native.metrics.model_size_bytes,
    binarySizeBytes: native.metrics.binary_size_bytes,
    streaming: {
      supported: false,
      first_result_ms: null,
      update_count: 1,
      volatile_update_count: 0,
      final_update_count: 1,
      revoke_count: 0,
      timestamps: native.behavior.timestamps,
    },
    segments: representative.segments,
    repetitions: native.repetitions,
    warmup: native.warmup,
  });
}

function resultRecord({
  fixture,
  text,
  reportedLanguage,
  coldLoadMs,
  warmInferenceMs,
  peakMemoryBytes,
  runtimePeakMemoryBytes,
  modelSizeBytes,
  binarySizeBytes,
  streaming,
  segments,
  repetitions: measuredRepetitions,
  warmup,
}) {
  const quality = qualityMetrics(fixture.reference_text, text);
  return {
    fixture_id: fixture.id,
    requested_language: languageCode(fixture.language),
    reported_language: reportedLanguage,
    reference_text: fixture.reference_text,
    text,
    quality,
    timing: {
      audio_duration_ms: fixture.duration_ms,
      cold_load_ms: coldLoadMs,
      warm_inference_ms: warmInferenceMs,
      real_time_factor: warmInferenceMs / fixture.duration_ms,
    },
    resources: {
      peak_memory_bytes: peakMemoryBytes,
      runtime_peak_memory_bytes: runtimePeakMemoryBytes,
      model_size_bytes: modelSizeBytes,
      binary_size_bytes: binarySizeBytes,
    },
    streaming,
    segments,
    warmup,
    repetitions: measuredRepetitions,
  };
}

function parseAppleSpeech(output) {
  const lines = output.split('\n');
  const value = (prefix) => {
    const line = lines.find((candidateLine) =>
      candidateLine.startsWith(prefix),
    );
    if (!line) {
      throw new Error(`Apple Speech output omitted ${prefix.trim()}`);
    }
    return line.slice(prefix.length);
  };
  const updates = lines
    .filter((line) => line.startsWith('UPDATE '))
    .map((line) => JSON.parse(line.slice('UPDATE '.length)));
  const final = reduceAppleSpeech(updates);
  return {
    create_ms: Number.parseFloat(value('CREATE_MS ')),
    metadata: JSON.parse(value('SESSION_METADATA ')),
    summary: JSON.parse(value('SESSION_SUMMARY ')),
    final,
  };
}

function reduceAppleSpeech(updates) {
  let stored = [];
  for (const update of updates) {
    const affected = update.replace_range;
    if (
      stored.some(
        (segment) =>
          segment.stability === 'final' &&
          rangesOverlap(segment, affected),
      )
    ) {
      throw new Error(
        `Apple Speech update ${update.sequence} overlaps finalized content`,
      );
    }
    stored = stored.filter(
      (segment) =>
        segment.stability === 'final' ||
        !rangesOverlap(segment, affected),
    );
    if (update.kind === 'replace') {
      stored.push(
        ...update.segments.map((segment) => ({
          ...segment,
          stability: update.stability,
        })),
      );
      stored.sort(
        (left, right) =>
          left.start_ms - right.start_ms ||
          left.end_ms - right.end_ms,
      );
    }
  }
  const segments = stored
    .filter((segment) => segment.stability === 'final')
    .map(({ stability: _, ...segment }) => segment);
  if (segments.length === 0) {
    throw new Error('Apple Speech emitted no final segments');
  }
  return {
    text: segments
      .map((segment) => segment.text)
      .filter(Boolean)
      .join(' '),
    segments,
  };
}

function rangesOverlap(left, right) {
  return left.start_ms < right.end_ms && right.start_ms < left.end_ms;
}

async function materializeFixture(fixture) {
  const prefix = fixturePrefix(fixture);
  const source = join(fixtureDirectory, `${prefix}.ogg`);
  const output = `/private/tmp/cuttledoc-${process.pid}-${prefix}.f32le`;
  commandOutput('ffmpeg', [
    '-y',
    '-v',
    'error',
    '-i',
    source,
    '-ar',
    '16000',
    '-ac',
    '1',
    '-f',
    'f32le',
    '-acodec',
    'pcm_f32le',
    output,
  ]);
  const bytes = await readFile(output);
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (digest !== fixture.sha256) {
    throw new Error(
      `${fixture.id}: normalized PCM digest ${digest} does not match manifest`,
    );
  }
  if ((bytes.length / 4 / 16) !== fixture.duration_ms) {
    throw new Error(`${fixture.id}: normalized duration does not match manifest`);
  }
  return output;
}

function fixturePrefix(fixture) {
  const prefixes = {
    'en-US': 'en',
    'de-DE': 'de',
    'es-419': 'es',
    'fr-FR': 'fr',
    'pt-BR': 'pt',
  };
  const prefix = prefixes[fixture.language];
  if (!prefix) {
    throw new Error(`unsupported fixture locale: ${fixture.language}`);
  }
  return `fleurs-${prefix}-${fixture.id.slice(-3)}`;
}

function languageCode(locale) {
  return locale.split('-')[0].toLowerCase();
}

function qualityMetrics(reference, hypothesis) {
  return {
    wer: errorRate(words(reference), words(hypothesis)),
    cer: errorRate(characters(reference), characters(hypothesis)),
  };
}

function words(text) {
  return text
    .toLocaleLowerCase('en-US')
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .split(/\s+/)
    .filter(Boolean);
}

function characters(text) {
  return words(text).join('').split('');
}

function errorRate(referenceItems, hypothesisItems) {
  if (referenceItems.length === 0) {
    return hypothesisItems.length === 0 ? 0 : 1;
  }
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
  return previous[hypothesisItems.length] / referenceItems.length;
}

function summarize(candidateResults) {
  const languages = {};
  for (const result of candidateResults) {
    const language = result.requested_language;
    const group = languages[language] ?? [];
    group.push(result);
    languages[language] = group;
  }
  return {
    macro_wer: mean(
      candidateResults.map((result) => result.quality.wer),
    ),
    macro_cer: mean(
      candidateResults.map((result) => result.quality.cer),
    ),
    mean_warm_inference_ms: mean(
      candidateResults.map((result) => result.timing.warm_inference_ms),
    ),
    mean_real_time_factor: mean(
      candidateResults.map((result) => result.timing.real_time_factor),
    ),
    maximum_peak_memory_bytes: Math.max(
      ...candidateResults.map(
        (result) => result.resources.peak_memory_bytes,
      ),
    ),
    by_language: Object.fromEntries(
      Object.entries(languages).map(([language, languageResults]) => [
        language,
        {
          fixture_count: languageResults.length,
          macro_wer: mean(
            languageResults.map((result) => result.quality.wer),
          ),
          macro_cer: mean(
            languageResults.map((result) => result.quality.cer),
          ),
        },
      ]),
    ),
  };
}

function candidateMetadata(name) {
  switch (name) {
    case 'mlx':
      return {
        id: 'mlx-whisper-tiny',
        model:
          'mlx-community/whisper-tiny@78c52ab98ca87f570bc57ad852e15ef7060f9f76',
        runtime:
          'official MLX v0.32.0@7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247',
        boundary: 'repository-owned C++ task ABI called from Rust',
        license:
          'MIT source model and official converter; pinned converted model card has no structured license field',
      };
    case 'apple-speech':
      return {
        id: 'apple-speechtranscriber',
        model: 'macOS system-managed SpeechTranscriber assets',
        runtime: 'macOS SpeechAnalyzer through a repository-owned Swift C ABI',
        boundary: 'repository-owned Swift task ABI called from Rust',
        license: 'Apple platform asset; not redistributed',
      };
    case 'parakeet':
      return {
        id: 'parakeet-tdt-0.6b-v3-coreml',
        model:
          'FluidInference/parakeet-tdt-0.6b-v3-coreml@aed02740059203c4a87495924f685de3722ae9ce',
        runtime:
          'legacy parakeet-coreml@3a29d6f80bfa5f95e791d21cfa86a0154806ef47',
        boundary: 'legacy Node addon driving CoreML and Silero VAD',
        license: 'CC-BY-4.0 model; MIT VAD',
      };
    case 'whisper':
      return {
        id: 'whisper-large-v3-turbo-coreml-whispercpp',
        model:
          'whisper.cpp@5359861c739e955e79d9a303bcbc70fb988958b1 plus whisper-coreml-models@dd3515371e6b560b63ec275abf020153a45caa60',
        runtime:
          'legacy whisper-coreml@20f619e02b46e64dc819958d3ab83bd029607f0a',
        boundary:
          'legacy Node addon using a CoreML encoder and whisper.cpp decoder',
        license: 'MIT model artifacts and runtime',
      };
    default:
      throw new Error(`unsupported candidate: ${name}`);
  }
}

function timedCommand(command, arguments_, options = {}) {
  const result = spawnSync('/usr/bin/time', ['-l', command, ...arguments_], {
    cwd: options.cwd ?? repoRoot,
    env: options.env ?? process.env,
    encoding: 'utf8',
    maxBuffer: options.maxBuffer ?? 128 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} failed with ${result.status}\n${result.stdout}\n${result.stderr}`,
    );
  }
  const matches = [
    ...result.stderr.matchAll(
      /^\s*(\d+)\s+maximum resident set size$/gm,
    ),
  ];
  const maximumResidentSetBytes = Number.parseInt(
    matches.at(-1)?.[1] ?? '',
    10,
  );
  if (!Number.isInteger(maximumResidentSetBytes)) {
    throw new Error(`could not parse maximum RSS from ${command}`);
  }
  return {
    stdout: result.stdout,
    stderr: result.stderr,
    maximumResidentSetBytes,
  };
}

function commandOutput(command, arguments_, options = {}) {
  const result = spawnSync(command, arguments_, {
    cwd: options.cwd ?? repoRoot,
    env: options.env ?? process.env,
    encoding: 'utf8',
    maxBuffer: options.maxBuffer ?? 16 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} failed with ${result.status}\n${result.stdout}\n${result.stderr}`,
    );
  }
  return result.stdout;
}

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is required for candidate ${candidate}`);
  }
  return resolve(value);
}

async function firstExisting(paths) {
  for (const path of paths) {
    try {
      await stat(path);
      return path;
    } catch {
      // Continue to the next accepted materialization layout.
    }
  }
  throw new Error(`none of the expected paths exist: ${paths.join(', ')}`);
}

function mean(values_) {
  if (values_.length === 0 || values_.some((value) => !Number.isFinite(value))) {
    throw new Error('cannot compute a mean from missing measurements');
  }
  return values_.reduce((sum, value) => sum + value, 0) / values_.length;
}
