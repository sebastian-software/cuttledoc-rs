#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  access,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  rename,
  rm,
  unlink,
  writeFile,
} from 'node:fs/promises';
import { totalmem } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultOutputDirectory = join(
  repoRoot,
  'artifacts/postprocessing-factorial-local',
);
const defaultTextDirectory =
  '/private/tmp/cuttledoc-synthetic-passages-4-verify';
const defaultQwenModelDirectory =
  '/tmp/cuttledoc-qwen3-tts-1.7b-voicedesign-mlx-bf16';
const defaultVoxtralModelDirectory =
  '/tmp/cuttledoc-voxtral-tts-4b-mlx-bf16';

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    'output-dir': { type: 'string', default: defaultOutputDirectory },
    'text-dir': { type: 'string', default: defaultTextDirectory },
    plan: {
      type: 'string',
      default: join(repoRoot, 'benchmarks/postprocessing/factorial-plan.json'),
    },
    ledger: {
      type: 'string',
      default: join(repoRoot, 'benchmarks/postprocessing/factorial-cells.json'),
    },
    selection: {
      type: 'string',
      default: join(
        repoRoot,
        'benchmarks/fixtures/synthetic-roundtrip-selection.json',
      ),
    },
    'qwen-model-dir': {
      type: 'string',
      default: defaultQwenModelDirectory,
    },
    'voxtral-model-dir': {
      type: 'string',
      default: defaultVoxtralModelDirectory,
    },
    'apple-build-dir': {
      type: 'string',
      default: '/private/tmp/cuttledoc-factorial-apple-tts-build',
    },
    locale: { type: 'string' },
    limit: { type: 'string' },
    'qualification-only': { type: 'boolean', default: false },
    force: { type: 'boolean', default: false },
  },
});

const command = positionals[0];
if (![
  'init',
  'resolve-voices',
  'run-apple-tts',
  'status',
  'self-test',
].includes(command)) {
  throw new Error(
    'usage: node scripts/run-postprocessing-factorial-local.mjs ' +
      '<init|resolve-voices|run-apple-tts|status|self-test> [options]',
  );
}

const paths = {
  output: resolve(values['output-dir']),
  text: resolve(values['text-dir']),
  plan: resolve(values.plan),
  ledger: resolve(values.ledger),
  selection: resolve(values.selection),
  qwenModel: resolve(values['qwen-model-dir']),
  voxtralModel: resolve(values['voxtral-model-dir']),
  appleBuild: resolve(values['apple-build-dir']),
};
paths.state = join(paths.output, 'state.json');
paths.lock = join(paths.output, '.run.lock');
paths.audio = join(paths.output, 'audio');
paths.appleInventory = join(paths.output, 'apple-voice-inventory.json');

if (command === 'self-test') {
  selfTest();
} else if (command === 'status') {
  await printStatus(await loadState(paths.state));
} else {
  await mkdir(paths.output, { recursive: true });
  await withRunLock(async () => {
    if (command === 'init') {
      const state = await initializeState();
      await writeState(state);
      await printStatus(state);
    } else if (command === 'resolve-voices') {
      const state = await initializeState();
      await resolveVoices(state);
      await writeState(state);
      await printStatus(state);
    } else if (command === 'run-apple-tts') {
      const state = await initializeState();
      await requireAppleResolutions(state);
      await runAppleTts(state);
      await writeState(state);
      await printStatus(state);
    }
  });
}

async function initializeState() {
  const [plan, ledger, selection] = await Promise.all([
    readJson(paths.plan),
    readJson(paths.ledger),
    readJson(paths.selection),
  ]);
  validateContracts(plan, ledger, selection);
  const input = await validateMaterializedPassages(plan, selection);
  const previous = await readJsonIfPresent(paths.state);
  if (previous && (
    previous.plan_id !== plan.id ||
    previous.plan_revision !== plan.revision ||
    previous.source_selection_revision !== selection.revision
  )) {
    throw new Error(
      `${paths.state}: state belongs to a different pinned benchmark revision`,
    );
  }
  const now = new Date().toISOString();
  const state = previous ?? {
    schema_version: '1.0.0',
    id: 'postprocessing-factorial-local-run-1',
    plan_id: plan.id,
    plan_revision: plan.revision,
    source_selection_revision: selection.revision,
    source_revision: gitRevision(),
    created_at: now,
    host: hostRecord(),
    paths: {
      output_directory: paths.output,
      text_directory: paths.text,
    },
    inputs: {},
    capabilities: {},
    voice_resolutions: plan.voice_slots.map((slot) => ({
      voice_slot_id: slot.id,
      locale: slot.locale,
      tts_engine: slot.tts_engine,
      selector: slot.selector,
      voice_locale: slot.voice_locale,
      voice_identity_seed: slot.voice_identity_seed,
      status:
        slot.tts_engine === 'apple-avspeechsynthesizer'
          ? 'inventory-required'
          : 'pinned-candidate',
      evidence: 'factorial-plan',
    })),
    progress: {},
  };
  state.source_revision = gitRevision();
  state.updated_at = now;
  state.inputs = input;
  state.capabilities = await capabilities();
  state.progress = await progress(ledger);
  return state;
}

function validateContracts(plan, ledger, selection) {
  if (ledger.plan_id !== plan.id || ledger.plan_revision !== plan.revision) {
    throw new Error('execution ledger differs from the pinned factorial plan');
  }
  if (
    plan.source_selection.revision !== selection.revision ||
    ledger.source_selection_revision !== selection.revision
  ) {
    throw new Error('source selection revision differs across local inputs');
  }
  if (
    plan.passage_slots.length !== 30 ||
    plan.voice_slots.length !== 30 ||
    ledger.audio_units.length !== ledger.counts.total_audio_units ||
    ledger.stt_units.length !== ledger.counts.total_stt_units
  ) {
    throw new Error('factorial plan or ledger counts do not reconcile');
  }
}

async function validateMaterializedPassages(plan, selection) {
  const manifest = await readJson(join(paths.text, 'manifest.json'));
  if (manifest.selection_revision !== selection.revision) {
    throw new Error(`${paths.text}: materialized selection revision differs`);
  }
  const selected = new Map();
  for (const source of selection.sources) {
    for (const passage of source.passages) {
      if (selected.has(passage.id)) {
        throw new Error(`duplicate selected passage: ${passage.id}`);
      }
      selected.set(passage.id, { source, passage });
    }
  }
  const digests = {};
  for (const slot of plan.passage_slots) {
    const candidate = selected.get(slot.passage_id);
    if (!candidate) throw new Error(`${slot.id}: passage is not selected`);
    const textPath = join(paths.text, `${slot.passage_id}.txt`);
    const text = (await readFile(textPath, 'utf8')).trim();
    const digest = sha256(Buffer.from(text));
    if (
      digest !== candidate.passage.spoken_sha256 ||
      text.length !== candidate.passage.character_count
    ) {
      throw new Error(`${slot.id}: materialized text differs from selection`);
    }
    digests[slot.passage_id] = digest;
  }
  return {
    passage_slots: plan.passage_slots.length,
    unique_passages: Object.keys(digests).length,
    verified_text_sha256: digests,
  };
}

async function capabilities() {
  const [qwen, voxtral, ffmpeg, xcrun, rustc] = await Promise.all([
    pathCapability(paths.qwenModel),
    pathCapability(paths.voxtralModel),
    commandCapability('ffmpeg'),
    commandCapability('xcrun'),
    commandCapability('rustc'),
  ]);
  return {
    apple_tts: {
      platform_supported: process.platform === 'darwin',
      xcrun,
      rustc,
    },
    normalization: { ffmpeg },
    qwen3_tts_bf16: qwen,
    voxtral_tts_bf16: voxtral,
  };
}

async function resolveVoices(state) {
  if (process.platform !== 'darwin') {
    throw new Error('Apple voice inventory requires macOS');
  }
  const binary = await buildAppleTtsAdapter();
  const plan = await readJson(paths.plan);
  const inventories = [];
  for (const locale of plan.axes.locales) {
    const output = commandOutput(binary, ['inventory', locale], {
      env: appleEnvironment(),
    });
    const inventory = JSON.parse(output.trim());
    const selected = pickAppleVoices(locale, inventory.voices);
    inventories.push({ ...inventory, selected });
    const slots = state.voice_resolutions
      .filter((item) =>
        item.tts_engine === 'apple-avspeechsynthesizer' &&
        item.locale === locale)
      .sort((left, right) => left.voice_slot_id.localeCompare(right.voice_slot_id));
    if (slots.length !== 2) throw new Error(`${locale}: expected two Apple slots`);
    for (let index = 0; index < slots.length; index += 1) {
      slots[index].selector = selected[index].identifier;
      slots[index].voice_locale = selected[index].language;
      slots[index].status = 'resolved-candidate';
      slots[index].evidence = relative(paths.output, paths.appleInventory);
      slots[index].inventory = selected[index];
    }
  }
  await atomicJson(paths.appleInventory, {
    schema_version: '1.0.0',
    captured_at: new Date().toISOString(),
    host: state.host,
    inventories,
  });
  resolvePinnedModelVoices(state, plan);
}

function resolvePinnedModelVoices(state, plan) {
  const qwenAvailable = state.capabilities.qwen3_tts_bf16.available;
  const voxtralAvailable = state.capabilities.voxtral_tts_bf16.available;
  for (const resolution of state.voice_resolutions) {
    const slot = plan.voice_slots.find((item) => item.id === resolution.voice_slot_id);
    if (resolution.tts_engine === 'qwen3-tts-1.7b-voicedesign-mlx-audio') {
      resolution.status = qwenAvailable ? 'resolved-candidate' : 'model-missing';
      resolution.evidence =
        'spikes/tts-calibration/qwen3-tts-1.7b-voicedesign-bf16.json';
      resolution.selector = slot.selector;
    } else if (resolution.tts_engine === 'voxtral-tts-4b-bf16-mlx-audio') {
      resolution.status = voxtralAvailable ? 'resolved-candidate' : 'model-missing';
      resolution.evidence =
        'spikes/tts-calibration/voxtral-tts-4b-mlx-bf16.json';
      resolution.selector = slot.selector;
    }
  }
}

function pickAppleVoices(locale, voices) {
  if (!Array.isArray(voices) || voices.length < 2) {
    throw new Error(`${locale}: fewer than two Apple voices are installed`);
  }
  const scored = [...voices].sort((left, right) => {
    const leftScore = appleVoiceScore(locale, left);
    const rightScore = appleVoiceScore(locale, right);
    return rightScore - leftScore ||
      left.identifier.localeCompare(right.identifier);
  });
  const first = scored[0];
  const differentGender = scored.find((voice) =>
    voice.identifier !== first.identifier &&
    voice.gender_raw_value !== 0 &&
    first.gender_raw_value !== 0 &&
    voice.gender_raw_value !== first.gender_raw_value);
  const second = differentGender ?? scored.find(
    (voice) => voice.identifier !== first.identifier,
  );
  if (!second) throw new Error(`${locale}: could not select two Apple voices`);
  return [first, second];
}

function appleVoiceScore(locale, voice) {
  return (voice.language === locale ? 10_000 : 0) +
    (voice.quality_raw_value ?? 0) * 100 +
    (voice.gender_raw_value === 0 ? 0 : 10);
}

async function requireAppleResolutions(state) {
  const unresolved = state.voice_resolutions.filter((item) =>
    item.tts_engine === 'apple-avspeechsynthesizer' &&
    !['resolved-candidate', 'qualified'].includes(item.status));
  if (unresolved.length > 0) {
    throw new Error(
      `resolve Apple voices first; ${unresolved.length} slot(s) remain unresolved`,
    );
  }
}

async function runAppleTts(state) {
  const ledger = await readJson(paths.ledger);
  const selection = await readJson(paths.selection);
  const selectedPassages = selectedPassageMap(selection);
  const voiceMap = new Map(
    state.voice_resolutions.map((item) => [item.voice_slot_id, item]),
  );
  const binary = await buildAppleTtsAdapter();
  const limit = parseLimit(values.limit);
  let units = ledger.audio_units.filter((unit) =>
    unit.tts_engine === 'apple-avspeechsynthesizer' &&
    (!values.locale || unit.locale === values.locale));
  if (values['qualification-only']) {
    units = units.filter((unit) =>
      unit.passage_slot_id.endsWith('technical-a') &&
      unit.generation_repeat === 1);
  }
  if (limit !== null) units = units.slice(0, limit);
  let completed = 0;
  let skipped = 0;
  for (const unit of units) {
    const resolution = voiceMap.get(unit.voice_slot_id);
    if (!resolution?.selector) {
      throw new Error(`${unit.id}: resolved Apple voice is missing`);
    }
    const outputDirectory = join(paths.audio, unit.id);
    if (!values.force && await validAudioArtifact(outputDirectory, unit)) {
      skipped += 1;
      continue;
    }
    if (values.force) await rm(outputDirectory, { recursive: true, force: true });
    await generateAppleAudio({
      binary,
      unit,
      resolution,
      selectedPassages,
      outputDirectory,
      state,
    });
    completed += 1;
    process.stderr.write(
      `apple tts: ${unit.id} (${completed + skipped}/${units.length})\n`,
    );
    state.updated_at = new Date().toISOString();
    state.progress = await progress(ledger);
    await writeState(state);
  }
  process.stdout.write(
    `Apple TTS completed ${completed}, resumed ${skipped}, selected ${units.length}\n`,
  );
}

async function generateAppleAudio({
  binary,
  unit,
  resolution,
  selectedPassages,
  outputDirectory,
  state,
}) {
  const selected = selectedPassages.get(unit.passage_id);
  if (!selected) throw new Error(`${unit.id}: selected passage is missing`);
  const temporary = await mkdtemp(join(paths.output, '.audio-tmp-'));
  try {
    const rawPath = join(temporary, 'audio.f32le');
    const normalizedPath = join(temporary, 'normalized-16k.f32le');
    const textPath = join(paths.text, `${unit.passage_id}.txt`);
    const stdout = commandOutput(binary, [
      'synthesize',
      textPath,
      rawPath,
      '--locale',
      unit.locale,
      '--voice',
      resolution.selector,
    ], { env: appleEnvironment(), maxBuffer: 32 * 1024 * 1024 });
    const metadata = prefixedJson(stdout, 'SESSION_METADATA ');
    const summary = prefixedJson(stdout, 'SYNTHESIS_SUMMARY ');
    const sampleRate = prefixedNumber(stdout, 'OUTPUT_SAMPLE_RATE_HZ ');
    const sampleCount = prefixedNumber(stdout, 'OUTPUT_SAMPLE_COUNT ');
    commandOutput('ffmpeg', [
      '-y', '-v', 'error',
      '-f', 'f32le', '-ar', String(sampleRate), '-ac', '1', '-i', rawPath,
      '-ar', '16000', '-ac', '1', '-f', 'f32le', '-acodec', 'pcm_f32le',
      normalizedPath,
    ]);
    const [raw, normalized] = await Promise.all([
      readFile(rawPath),
      readFile(normalizedPath),
    ]);
    if (raw.length !== sampleCount * 4 || normalized.length % 4 !== 0) {
      throw new Error(`${unit.id}: invalid PCM byte count`);
    }
    const manifest = {
      schema_version: '1.0.0',
      id: unit.id,
      plan_id: state.plan_id,
      plan_revision: state.plan_revision,
      source_selection_revision: state.source_selection_revision,
      source_revision: gitRevision(),
      captured_at: new Date().toISOString(),
      unit,
      voice: {
        voice_slot_id: resolution.voice_slot_id,
        identifier: resolution.selector,
        requested_locale: unit.locale,
        resolved_locale: resolution.voice_locale,
        inventory: resolution.inventory,
        session: metadata,
      },
      input: {
        path: relative(repoRoot, textPath),
        passage_id: unit.passage_id,
        source_id: selected.source.id,
        character_count: selected.passage.character_count,
        text_sha256: selected.passage.spoken_sha256,
        license: selected.source.license,
      },
      audio: {
        path: 'audio.f32le',
        sample_format: 'f32le',
        sample_rate_hz: sampleRate,
        channel_count: 1,
        sample_count: sampleCount,
        byte_count: raw.length,
        duration_ms: sampleCount * 1000 / sampleRate,
        sha256: sha256(raw),
        ...audioStatistics(raw),
      },
      normalized_audio: {
        path: 'normalized-16k.f32le',
        sample_format: 'f32le',
        sample_rate_hz: 16000,
        channel_count: 1,
        sample_count: normalized.length / 4,
        byte_count: normalized.length,
        duration_ms: normalized.length / 4 / 16,
        sha256: sha256(normalized),
      },
      runtime_summary: summary,
      disposition: 'generated-pending-stt-qualification',
    };
    await writeFile(join(temporary, 'manifest.json'), json(manifest));
    await mkdir(dirname(outputDirectory), { recursive: true });
    await rename(temporary, outputDirectory);
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}

function audioStatistics(buffer) {
  let minimum = Infinity;
  let maximum = -Infinity;
  let sumSquares = 0;
  let nonFinite = 0;
  for (let offset = 0; offset < buffer.length; offset += 4) {
    const value = buffer.readFloatLE(offset);
    if (!Number.isFinite(value)) {
      nonFinite += 1;
      continue;
    }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sumSquares += value * value;
  }
  const finiteCount = buffer.length / 4 - nonFinite;
  if (finiteCount === 0) throw new Error('audio contains no finite samples');
  return {
    minimum_sample: minimum,
    maximum_sample: maximum,
    rms: Math.sqrt(sumSquares / finiteCount),
    non_finite_sample_count: nonFinite,
  };
}

async function validAudioArtifact(directory, unit) {
  try {
    const manifest = await readJson(join(directory, 'manifest.json'));
    if (
      manifest.id !== unit.id ||
      manifest.plan_revision !== (await readJson(paths.plan)).revision
    ) return false;
    for (const record of [manifest.audio, manifest.normalized_audio]) {
      const bytes = await readFile(join(directory, record.path));
      if (bytes.length !== record.byte_count || sha256(bytes) !== record.sha256) {
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

async function progress(ledger) {
  const completedByEngine = {};
  let completedAudio = 0;
  try {
    const entries = await readdir(paths.audio, { withFileTypes: true });
    const units = new Map(ledger.audio_units.map((unit) => [unit.id, unit]));
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const unit = units.get(entry.name);
      if (!unit || !await validAudioArtifact(join(paths.audio, entry.name), unit)) {
        continue;
      }
      completedAudio += 1;
      completedByEngine[unit.tts_engine] =
        (completedByEngine[unit.tts_engine] ?? 0) + 1;
    }
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  return {
    audio: {
      expected: ledger.counts.total_audio_units,
      completed: completedAudio,
      remaining: ledger.counts.total_audio_units - completedAudio,
      completed_by_engine: completedByEngine,
    },
    stt: {
      expected: ledger.counts.total_stt_units,
      completed: 0,
      remaining: ledger.counts.total_stt_units,
    },
    llm_documents: {
      expected: ledger.counts.llm_documents,
      materialized: 0,
    },
  };
}

async function buildAppleTtsAdapter() {
  const binary = join(paths.appleBuild, 'cuttledoc-apple-tts-spike');
  if (!values.force && await exists(binary)) return binary;
  await mkdir(join(paths.appleBuild, 'module-cache'), { recursive: true });
  const swiftSource = join(
    repoRoot,
    'spikes/apple-tts-shim/Sources/CuttledocTtsShim.swift',
  );
  const rustSource = join(repoRoot, 'spikes/apple-tts-shim/rust/main.rs');
  commandOutput('xcrun', [
    'swiftc',
    '-module-cache-path', join(paths.appleBuild, 'module-cache'),
    '-emit-library',
    '-emit-module',
    '-module-name', 'CuttledocTtsShim',
    swiftSource,
    '-o', join(paths.appleBuild, 'libcuttledoc_tts_shim.dylib'),
  ], {
    env: {
      ...process.env,
      CLANG_MODULE_CACHE_PATH: join(paths.appleBuild, 'module-cache'),
    },
  });
  commandOutput('rustc', [
    '--edition', '2021',
    rustSource,
    '-L', `native=${paths.appleBuild}`,
    '-l', 'dylib=cuttledoc_tts_shim',
    '-o', binary,
  ]);
  return binary;
}

function appleEnvironment() {
  return { ...process.env, DYLD_LIBRARY_PATH: paths.appleBuild };
}

async function printStatus(state) {
  const statusCounts = new Map();
  for (const resolution of state.voice_resolutions) {
    statusCounts.set(
      resolution.status,
      (statusCounts.get(resolution.status) ?? 0) + 1,
    );
  }
  const statuses = [...statusCounts]
    .map(([status, count]) => ({ status, count }))
    .sort((left, right) => left.status.localeCompare(right.status));
  process.stdout.write(json({
    id: state.id,
    plan_revision: state.plan_revision,
    source_revision: state.source_revision,
    paths: state.paths,
    capabilities: state.capabilities,
    voice_statuses: statuses,
    progress: state.progress,
  }));
}

async function writeState(state) {
  state.updated_at = new Date().toISOString();
  await atomicJson(paths.state, state);
}

async function atomicJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, json(value));
  await rename(temporary, path);
}

async function withRunLock(callback) {
  let handle;
  try {
    handle = await open(paths.lock, 'wx');
    await handle.writeFile(`${process.pid}\n`);
  } catch (error) {
    if (error.code === 'EEXIST') {
      throw new Error(`${paths.lock}: another local factorial command is active`);
    }
    throw error;
  }
  try {
    await callback();
  } finally {
    await handle.close();
    await unlink(paths.lock).catch(() => {});
  }
}

function selectedPassageMap(selection) {
  return new Map(selection.sources.flatMap((source) =>
    source.passages.map((passage) => [passage.id, { source, passage }])));
}

function hostRecord() {
  if (process.platform !== 'darwin') {
    return {
      platform: process.platform,
      architecture: process.arch,
      memory_bytes: totalmem(),
    };
  }
  return {
    platform: 'macOS',
    os_version: optionalCommandOutput('sw_vers', ['-productVersion']),
    os_build: optionalCommandOutput('sw_vers', ['-buildVersion']),
    chip: optionalCommandOutput('sysctl', ['-n', 'machdep.cpu.brand_string']),
    architecture: process.arch,
    memory_bytes: totalmem(),
  };
}

function gitRevision() {
  return commandOutput('git', ['rev-parse', 'HEAD'], { cwd: repoRoot }).trim();
}

async function pathCapability(path) {
  return { path, available: await exists(path) };
}

async function commandCapability(commandName) {
  const result = spawnSync('/usr/bin/env', ['which', commandName], {
    encoding: 'utf8',
  });
  return {
    command: commandName,
    available: result.status === 0,
    path: result.status === 0 ? result.stdout.trim() : null,
  };
}

function commandOutput(commandName, arguments_, options = {}) {
  const result = spawnSync(commandName, arguments_, {
    encoding: 'utf8',
    maxBuffer: options.maxBuffer ?? 8 * 1024 * 1024,
    ...options,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `${commandName} ${arguments_.join(' ')} failed (${result.status})\n` +
      `${result.stdout ?? ''}${result.stderr ?? ''}`,
    );
  }
  return result.stdout;
}

function optionalCommandOutput(commandName, arguments_) {
  const result = spawnSync(commandName, arguments_, {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024,
  });
  return result.status === 0 ? result.stdout.trim() : 'unavailable';
}

function prefixedJson(output, prefix) {
  const line = output.split('\n').find((candidate) => candidate.startsWith(prefix));
  if (!line) throw new Error(`missing command output prefix: ${prefix}`);
  return JSON.parse(line.slice(prefix.length));
}

function prefixedNumber(output, prefix) {
  const line = output.split('\n').find((candidate) => candidate.startsWith(prefix));
  const value = Number(line?.slice(prefix.length));
  if (!Number.isFinite(value)) {
    throw new Error(`missing numeric command output prefix: ${prefix}`);
  }
  return value;
}

function parseLimit(value) {
  if (value === undefined) return null;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error('--limit must be a positive integer');
  }
  return parsed;
}

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function readJsonIfPresent(path) {
  try {
    return await readJson(path);
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

async function loadState(path) {
  const state = await readJsonIfPresent(path);
  if (!state) throw new Error(`${path}: run init first`);
  const ledger = await readJson(paths.ledger);
  state.capabilities = await capabilities();
  state.progress = await progress(ledger);
  return state;
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function json(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function selfTest() {
  const selected = pickAppleVoices('de-DE', [
    {
      identifier: 'fallback-female',
      language: 'de-AT',
      quality_raw_value: 2,
      gender_raw_value: 2,
    },
    {
      identifier: 'exact-male',
      language: 'de-DE',
      quality_raw_value: 1,
      gender_raw_value: 1,
    },
    {
      identifier: 'exact-female',
      language: 'de-DE',
      quality_raw_value: 1,
      gender_raw_value: 2,
    },
  ]);
  if (
    selected[0].identifier !== 'exact-female' ||
    selected[1].identifier !== 'exact-male'
  ) {
    throw new Error('self-test: deterministic Apple voice selection failed');
  }
  const pcm = Buffer.alloc(8);
  pcm.writeFloatLE(-0.5, 0);
  pcm.writeFloatLE(0.5, 4);
  const stats = audioStatistics(pcm);
  if (
    stats.minimum_sample !== -0.5 ||
    stats.maximum_sample !== 0.5 ||
    Math.abs(stats.rms - 0.5) > 1e-12
  ) {
    throw new Error('self-test: PCM statistics failed');
  }
  if (parseLimit(undefined) !== null || parseLimit('2') !== 2) {
    throw new Error('self-test: limit parsing failed');
  }
  process.stdout.write('factorial local runner: self-test passed\n');
}
