#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { basename, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    backend: { type: 'string' },
    fixture: { type: 'string' },
    reference: { type: 'string' },
    'module-dir': { type: 'string' },
    'model-dir': { type: 'string' },
    'vad-dir': { type: 'string' },
    output: { type: 'string' },
    repetitions: { type: 'string', default: '5' },
  },
});

const backend = values.backend;
if (backend !== 'parakeet' && backend !== 'whisper') {
  throw new Error('--backend must be parakeet or whisper');
}

const fixturePath = requiredPath(values.fixture, '--fixture');
const referencePath = requiredPath(values.reference, '--reference');
const moduleDir = requiredPath(values['module-dir'], '--module-dir');
const modelDir = requiredPath(values['model-dir'], '--model-dir');
const vadDir = values['vad-dir'] ? resolve(values['vad-dir']) : undefined;
const repetitions = Number.parseInt(values.repetitions, 10);
if (!Number.isInteger(repetitions) || repetitions < 1) {
  throw new Error('--repetitions must be a positive integer');
}

const reference = (await readFile(referencePath, 'utf8')).trim();
const samples = decodePcm(fixturePath);
const audioDurationMs = (samples.length / 16_000) * 1_000;
const require = createRequire(import.meta.url);
const api = require(join(moduleDir, 'dist/index.cjs'));
const addonPath = join(
  moduleDir,
  'build/Release',
  backend === 'parakeet' ? 'coreml_asr.node' : 'whisper_asr.node',
);

let engine;
if (backend === 'parakeet') {
  engine = new api.ParakeetAsrEngine({
    modelDir,
    vadDir,
    autoDownload: false,
  });
} else {
  engine = new api.WhisperAsrEngine({
    modelPath: join(modelDir, 'ggml-large-v3-turbo.bin'),
    language: 'en',
    useGpu: true,
  });
}

let coldLoadMs;
let warmup;
const measured = [];
try {
  const loadStarted = performance.now();
  await engine.initialize();
  coldLoadMs = performance.now() - loadStarted;

  warmup = await transcribe(engine, backend, samples);
  for (let index = 0; index < repetitions; index += 1) {
    const started = performance.now();
    const result = await transcribe(engine, backend, samples);
    measured.push({
      iteration: index + 1,
      wall_ms: performance.now() - started,
      backend_ms: result.durationMs,
      text: result.text,
      language: result.language ?? 'en',
      segments: normalizeSegments(result.segments, audioDurationMs),
    });
  }
} finally {
  engine.cleanup();
}

const representative = measured.at(-1);
const warmInferenceMs =
  measured.reduce((sum, sample) => sum + sample.wall_ms, 0) / measured.length;
const modelSizeBytes = await directorySize(modelDir);
const vadSizeBytes = vadDir ? await directorySize(vadDir) : 0;

const record = {
      schema_version: '1.0.0',
      backend,
      captured_at: new Date().toISOString(),
      fixture: {
        path: fixturePath,
        name: basename(fixturePath),
        reference,
        audio_duration_ms: audioDurationMs,
        samples: samples.length,
        sample_rate_hz: 16_000,
        source_sha256: await fileDigest(fixturePath),
        pcm_sha256: createHash('sha256')
          .update(
            Buffer.from(
              samples.buffer,
              samples.byteOffset,
              samples.byteLength,
            ),
          )
          .digest('hex'),
        reference_sha256: await fileDigest(referencePath),
      },
      runtime: {
        module_revision: moduleDir,
        version: engine.getVersion(),
      },
      metrics: {
        wer: errorRate(words(reference), words(representative.text)),
        cer: errorRate(characters(reference), characters(representative.text)),
        cold_load_ms: coldLoadMs,
        warm_inference_ms: warmInferenceMs,
        real_time_factor: warmInferenceMs / audioDurationMs,
        model_size_bytes: modelSizeBytes,
        vad_size_bytes: vadSizeBytes,
        binary_size_bytes: (await stat(addonPath)).size,
      },
      behavior: {
        timestamps:
          representative.segments.length > 0 ? 'segment' : 'none',
        final_update_count: 1,
        volatile_update_count: 0,
        revoke_count: 0,
      },
      warmup: {
        backend_ms: warmup.durationMs,
        text: warmup.text,
      },
      repetitions: measured,
};
const serialized = `${JSON.stringify(record, null, 2)}\n`;
if (values.output) {
  const outputPath = resolve(values.output);
  await writeFile(outputPath, serialized);
  process.stderr.write(`Wrote ${outputPath}\n`);
} else {
  process.stdout.write(serialized);
}

function requiredPath(value, option) {
  if (!value) {
    throw new Error(`${option} is required`);
  }
  return resolve(value);
}

function decodePcm(path) {
  const result = spawnSync(
    'ffmpeg',
    [
      '-v',
      'error',
      '-i',
      path,
      '-ar',
      '16000',
      '-ac',
      '1',
      '-f',
      'f32le',
      '-acodec',
      'pcm_f32le',
      '-',
    ],
    {
      encoding: null,
      maxBuffer: 128 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(
      `ffmpeg failed: ${result.stderr?.toString('utf8') ?? result.error}`,
    );
  }
  const bytes = result.stdout;
  const owned = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  return new Float32Array(owned);
}

async function transcribe(instance, name, pcm) {
  if (name === 'parakeet') {
    return instance.transcribe(pcm, { sampleRate: 16_000 });
  }
  return instance.transcribe(pcm, 16_000);
}

function normalizeSegments(segments, audioDurationMs) {
  return segments.map((segment) => {
    if ('startMs' in segment) {
      return {
        start_ms: segment.startMs,
        end_ms: segment.endMs,
        text: segment.text,
        confidence: segment.confidence ?? null,
      };
    }
    return {
      start_ms: segment.startTime * 1_000,
      end_ms: Math.min(segment.endTime * 1_000, audioDurationMs),
      text: segment.text,
      confidence: null,
    };
  });
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
  for (let referenceIndex = 1; referenceIndex <= referenceItems.length; referenceIndex += 1) {
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

async function directorySize(directory) {
  let total = 0;
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      total += await directorySize(entryPath);
    } else if (entry.isFile()) {
      total += (await stat(entryPath)).size;
    }
  }
  return total;
}

async function fileDigest(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}
