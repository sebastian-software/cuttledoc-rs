#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
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
const defaultWhisperModuleDirectory = '/Users/sebastian/Workspace/whisper-coreml';
const defaultWhisperModelDirectory =
  '/Users/sebastian/.cache/whisper-coreml/models';
const defaultParakeetModuleDirectory = '/Users/sebastian/Workspace/parakeet-coreml';
const defaultParakeetModelDirectory =
  '/Users/sebastian/.cache/parakeet-coreml/models';
const defaultParakeetVadDirectory = '/Users/sebastian/.cache/parakeet-coreml/vad';
const defaultQwenAsrBinary =
  '/private/tmp/cuttledoc-qwen3-mlx-direct-build/cuttledoc-qwen3-mlx-inspect';
const defaultQwenAsrModelDirectory = '/private/tmp/cuttledoc-qwen3-asr/model';
const defaultQualificationOutput = join(
  repoRoot,
  'benchmarks/postprocessing/qualifications/' +
    'apple-avspeechsynthesizer.technical-a-1.json',
);

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
    backend: { type: 'string', default: 'all' },
    'whisper-module-dir': {
      type: 'string',
      default: defaultWhisperModuleDirectory,
    },
    'whisper-model-dir': {
      type: 'string',
      default: defaultWhisperModelDirectory,
    },
    'parakeet-module-dir': {
      type: 'string',
      default: defaultParakeetModuleDirectory,
    },
    'parakeet-model-dir': {
      type: 'string',
      default: defaultParakeetModelDirectory,
    },
    'parakeet-vad-dir': {
      type: 'string',
      default: defaultParakeetVadDirectory,
    },
    'qwen-asr-binary': { type: 'string', default: defaultQwenAsrBinary },
    'qwen-asr-model-dir': {
      type: 'string',
      default: defaultQwenAsrModelDirectory,
    },
    'qualification-output': {
      type: 'string',
      default: defaultQualificationOutput,
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
  'run-stt',
  'summarize-qualification',
  'status',
  'self-test',
].includes(command)) {
  throw new Error(
    'usage: node scripts/run-postprocessing-factorial-local.mjs ' +
      '<init|resolve-voices|run-apple-tts|run-stt|' +
      'summarize-qualification|status|self-test> [options]',
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
  whisperModule: resolve(values['whisper-module-dir']),
  whisperModel: resolve(values['whisper-model-dir']),
  parakeetModule: resolve(values['parakeet-module-dir']),
  parakeetModel: resolve(values['parakeet-model-dir']),
  parakeetVad: resolve(values['parakeet-vad-dir']),
  qwenAsrBinary: resolve(values['qwen-asr-binary']),
  qwenAsrModel: resolve(values['qwen-asr-model-dir']),
  qualificationOutput: resolve(values['qualification-output']),
};
paths.state = join(paths.output, 'state.json');
paths.lock = join(paths.output, '.run.lock');
paths.audio = join(paths.output, 'audio');
paths.stt = join(paths.output, 'stt');
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
    } else if (command === 'run-stt') {
      const state = await initializeState();
      await runStt(state);
      await writeState(state);
      await printStatus(state);
    } else if (command === 'summarize-qualification') {
      const state = await initializeState();
      await summarizeAppleQualification(state);
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
  const [
    qwen,
    voxtral,
    ffmpeg,
    xcrun,
    rustc,
    whisperModule,
    whisperModel,
    parakeetModule,
    parakeetModel,
    parakeetVad,
    qwenAsrBinary,
    qwenAsrModel,
  ] = await Promise.all([
    pathCapability(paths.qwenModel),
    pathCapability(paths.voxtralModel),
    commandCapability('ffmpeg'),
    commandCapability('xcrun'),
    commandCapability('rustc'),
    pathCapability(paths.whisperModule),
    pathCapability(paths.whisperModel),
    pathCapability(paths.parakeetModule),
    pathCapability(paths.parakeetModel),
    pathCapability(paths.parakeetVad),
    pathCapability(paths.qwenAsrBinary),
    pathCapability(paths.qwenAsrModel),
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
    stt: {
      whisper: {
        available: whisperModule.available && whisperModel.available,
        module: whisperModule,
        model: whisperModel,
      },
      parakeet: {
        available:
          parakeetModule.available &&
          parakeetModel.available &&
          parakeetVad.available,
        module: parakeetModule,
        model: parakeetModel,
        vad: parakeetVad,
      },
      qwen3_mlx_direct: {
        available: qwenAsrBinary.available && qwenAsrModel.available,
        binary: qwenAsrBinary,
        model: qwenAsrModel,
      },
    },
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
  const familyScore =
    voice.identifier.includes('.voice.compact.') ||
    voice.identifier.includes('.voice.super-compact.')
      ? 20_000
      : voice.identifier.includes('.speech.synthesis.voice.')
        ? 15_000
        : 0;
  return familyScore + (voice.language === locale ? 5_000 : 0) +
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

async function runStt(state) {
  const ledger = await readJson(paths.ledger);
  const selection = await readJson(paths.selection);
  const selectedPassages = selectedPassageMap(selection);
  const audioUnits = new Map(ledger.audio_units.map((unit) => [unit.id, unit]));
  const requestedModels = sttModelsForOption(values.backend);
  const limit = parseLimit(values.limit);
  let units = [];
  for (const unit of ledger.stt_units) {
    if (!requestedModels.has(unit.stt_model)) continue;
    const audioUnit = audioUnits.get(unit.audio_unit_id);
    if (!audioUnit || (values.locale && unit.locale !== values.locale)) continue;
    if (values['qualification-only'] && !(
      audioUnit.passage_slot_id.endsWith('technical-a') &&
      audioUnit.generation_repeat === 1
    )) continue;
    const audioDirectory = join(paths.audio, unit.audio_unit_id);
    if (!await validAudioArtifact(audioDirectory, audioUnit)) continue;
    units.push(unit);
  }
  if (limit !== null) units = units.slice(0, limit);
  if (units.length === 0) {
    throw new Error('no generated audio units match the requested STT slice');
  }
  const groups = new Map();
  for (const unit of units) {
    const key = unit.stt_model === 'whisper-large-v3-turbo-coreml-whispercpp'
      ? `${unit.stt_model}:${unit.locale.slice(0, 2)}`
      : unit.stt_model;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(unit);
  }
  let completed = 0;
  let skipped = 0;
  for (const group of groups.values()) {
    const remaining = [];
    for (const unit of group) {
      const outputDirectory = join(paths.stt, unit.id);
      if (!values.force && await validSttArtifact(outputDirectory, unit)) {
        skipped += 1;
      } else {
        if (values.force) {
          await rm(outputDirectory, { recursive: true, force: true });
        }
        remaining.push(unit);
      }
    }
    if (remaining.length === 0) continue;
    if (remaining[0].stt_model === 'qwen3-asr-0.6b-mlx-direct') {
      for (const unit of remaining) {
        const context = await loadSttContext(unit, audioUnits, selectedPassages);
        const result = runQwenAsr(context, unit);
        await writeSttArtifact(unit, context, result, state);
        completed += 1;
        process.stderr.write(
          `qwen3 asr: ${unit.id} (${completed + skipped}/${units.length})\n`,
        );
      }
    } else {
      completed += await runLegacySttGroup(
        remaining,
        audioUnits,
        selectedPassages,
        state,
      );
    }
    state.progress = await progress(ledger);
    await writeState(state);
  }
  process.stdout.write(
    `STT completed ${completed}, resumed ${skipped}, selected ${units.length}\n`,
  );
}

async function summarizeAppleQualification(state) {
  const ledger = await readJson(paths.ledger);
  const qualificationAudioUnits = ledger.audio_units.filter((unit) =>
    unit.tts_engine === 'apple-avspeechsynthesizer' &&
    unit.passage_slot_id.endsWith('technical-a') &&
    unit.generation_repeat === 1);
  if (qualificationAudioUnits.length !== 10) {
    throw new Error('expected ten Apple qualification audio units');
  }

  const audioById = new Map();
  const capturedAt = [];
  const sourceRevisions = new Set();
  for (const unit of qualificationAudioUnits) {
    const directory = join(paths.audio, unit.id);
    if (!await validAudioArtifact(directory, unit)) {
      throw new Error(`${unit.id}: missing or invalid qualification audio`);
    }
    const manifest = await readJson(join(directory, 'manifest.json'));
    audioById.set(unit.id, manifest);
    capturedAt.push(manifest.captured_at);
    sourceRevisions.add(manifest.source_revision);
  }

  const expectedSttUnits = ledger.stt_units.filter((unit) =>
    audioById.has(unit.audio_unit_id));
  if (expectedSttUnits.length !== 30) {
    throw new Error('expected thirty Apple qualification STT units');
  }
  const sttByAudio = new Map();
  for (const unit of expectedSttUnits) {
    const directory = join(paths.stt, unit.id);
    if (!await validSttArtifact(directory, unit)) {
      throw new Error(`${unit.id}: missing or invalid qualification transcript`);
    }
    const manifest = await readJson(join(directory, 'manifest.json'));
    if (!sttByAudio.has(unit.audio_unit_id)) {
      sttByAudio.set(unit.audio_unit_id, []);
    }
    sttByAudio.get(unit.audio_unit_id).push(manifest);
    capturedAt.push(manifest.captured_at);
    sourceRevisions.add(manifest.source_revision);
  }
  if (sourceRevisions.size !== 1) {
    throw new Error('qualification artifacts span multiple source revisions');
  }

  const maximumBestReceiverWer = 0.2;
  const minimumMasterRms = 0.01;
  const voices = qualificationAudioUnits.map((unit) => {
    const audio = audioById.get(unit.id);
    const observations = sttByAudio.get(unit.id)
      .sort((left, right) =>
        left.unit.stt_model.localeCompare(right.unit.stt_model))
      .map((manifest) => ({
        stt_model: manifest.unit.stt_model,
        transcript_sha256: manifest.transcript.sha256,
        transcript_word_count: manifest.transcript.word_count,
        wer: manifest.quality.wer,
        cer: manifest.quality.cer,
        inference_ms: manifest.runtime.inference_ms,
      }));
    const bestWer = Math.min(...observations.map((item) => item.wer));
    const passes =
      audio.audio.non_finite_sample_count === 0 &&
      audio.audio.rms >= minimumMasterRms &&
      observations.length === 3 &&
      bestWer <= maximumBestReceiverWer;
    return {
      voice_slot_id: unit.voice_slot_id,
      locale: unit.locale,
      passage_slot_id: unit.passage_slot_id,
      passage_id: unit.passage_id,
      identifier: audio.voice.identifier,
      resolved_locale: audio.voice.resolved_locale,
      name: audio.voice.inventory.name,
      audio_unit_id: unit.id,
      master_sha256: audio.audio.sha256,
      normalized_sha256: audio.normalized_audio.sha256,
      duration_ms: audio.audio.duration_ms,
      sample_rate_hz: audio.audio.sample_rate_hz,
      rms: audio.audio.rms,
      non_finite_sample_count: audio.audio.non_finite_sample_count,
      best_wer: bestWer,
      worst_wer: Math.max(...observations.map((item) => item.wer)),
      observations,
      technical_gate: passes ? 'passed' : 'failed',
    };
  }).sort((left, right) =>
    left.voice_slot_id.localeCompare(right.voice_slot_id));
  const observations = voices.flatMap((voice) => voice.observations.map(
    (observation) => ({ ...observation, locale: voice.locale }),
  ));
  const report = {
    schema_version: '1.0.0',
    id: 'apple-avspeechsynthesizer-technical-a-qualification-1',
    plan_id: state.plan_id,
    plan_revision: state.plan_revision,
    source_selection_revision: state.source_selection_revision,
    evidence_source_revision: [...sourceRevisions][0],
    evidence_captured_through: capturedAt.sort().at(-1),
    qualification_slice: {
      tts_engine: 'apple-avspeechsynthesizer',
      passage_selector: '*-technical-a',
      generation_repeat: 1,
      voice_count: voices.length,
      audio_count: voices.length,
      stt_count: observations.length,
    },
    policy: {
      purpose: 'catastrophic technical screen before factorial expansion',
      minimum_master_rms: minimumMasterRms,
      maximum_non_finite_samples: 0,
      required_stt_models_per_voice: 3,
      maximum_best_receiver_wer: maximumBestReceiverWer,
      receiver_rule:
        'At least one locked STT receiver must meet the WER threshold; ' +
        'receiver disagreement is retained as evidence, not used to reject ' +
        'an otherwise intelligible voice.',
    },
    aggregates: {
      by_stt_model: groupedQualificationMetrics(observations, 'stt_model'),
      by_locale: groupedQualificationMetrics(observations, 'locale'),
    },
    voices,
    disposition: voices.every((voice) => voice.technical_gate === 'passed')
      ? 'technical-gate-passed'
      : 'technical-gate-failed',
    limitations: [
      'This is one technical passage per voice, not the factorial benchmark.',
      'Synthetic clean speech cannot establish quality on human podcasts or audiobooks.',
      'WER measures the complete TTS-to-STT channel and is receiver-specific.',
      'The technical gate does not replace optional perceptual listening review.',
    ],
  };
  await atomicJson(paths.qualificationOutput, report);
  process.stdout.write(
    `Wrote ${relative(repoRoot, paths.qualificationOutput)}: ` +
    `${voices.length} voices, ${observations.length} STT observations, ` +
    `${report.disposition}\n`,
  );
}

function groupedQualificationMetrics(observations, field) {
  const groups = new Map();
  for (const observation of observations) {
    const key = observation[field];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(observation);
  }
  return [...groups].sort(([left], [right]) => left.localeCompare(right))
    .map(([key, group]) => ({
      [field]: key,
      observation_count: group.length,
      mean_wer: mean(group.map((item) => item.wer)),
      minimum_wer: Math.min(...group.map((item) => item.wer)),
      maximum_wer: Math.max(...group.map((item) => item.wer)),
      mean_cer: mean(group.map((item) => item.cer)),
    }));
}

function mean(numbers) {
  return numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
}

function sttModelsForOption(option) {
  const aliases = {
    whisper: 'whisper-large-v3-turbo-coreml-whispercpp',
    qwen3: 'qwen3-asr-0.6b-mlx-direct',
    parakeet: 'parakeet-tdt-0.6b-v3-coreml',
  };
  if (option === 'all') return new Set(Object.values(aliases));
  if (!aliases[option]) {
    throw new Error('--backend must be all, whisper, qwen3, or parakeet');
  }
  return new Set([aliases[option]]);
}

async function runLegacySttGroup(units, audioUnits, selectedPassages, state) {
  const model = units[0].stt_model;
  const backend = model === 'whisper-large-v3-turbo-coreml-whispercpp'
    ? 'whisper'
    : 'parakeet';
  const moduleDirectory = backend === 'whisper'
    ? paths.whisperModule
    : paths.parakeetModule;
  const require = createRequire(import.meta.url);
  const api = require(join(moduleDirectory, 'dist/index.cjs'));
  const language = units[0].locale.slice(0, 2);
  const engine = backend === 'whisper'
    ? new api.WhisperAsrEngine({
        modelPath: join(paths.whisperModel, 'ggml-large-v3-turbo.bin'),
        language,
        useGpu: true,
      })
    : new api.ParakeetAsrEngine({
        modelDir: paths.parakeetModel,
        vadDir: paths.parakeetVad,
        autoDownload: false,
      });
  const loadStarted = performance.now();
  await engine.initialize();
  const loadMs = performance.now() - loadStarted;
  try {
    const warmupContext = await loadSttContext(
      units[0],
      audioUnits,
      selectedPassages,
    );
    await transcribeLegacy(engine, backend, warmupContext.samples);
    let completed = 0;
    for (const unit of units) {
      const context = await loadSttContext(unit, audioUnits, selectedPassages);
      const started = performance.now();
      const native = await transcribeLegacy(engine, backend, context.samples);
      const wallMs = performance.now() - started;
      const result = {
        text: native.text.trim(),
        runtime: {
          backend,
          version: engine.getVersion(),
          requested_language: backend === 'whisper' ? language : 'model-managed',
          model_load_ms: loadMs,
          inference_ms: wallMs,
          backend_ms: native.durationMs,
          segments: normalizeSegments(
            native.segments,
            context.audioManifest.normalized_audio.duration_ms,
          ),
        },
      };
      await writeSttArtifact(unit, context, result, state);
      completed += 1;
      process.stderr.write(
        `${backend}: ${unit.id} (${completed}/${units.length})\n`,
      );
    }
    return completed;
  } finally {
    engine.cleanup();
  }
}

async function transcribeLegacy(engine, backend, samples) {
  return backend === 'parakeet'
    ? engine.transcribe(samples, { sampleRate: 16_000 })
    : engine.transcribe(samples, 16_000);
}

function runQwenAsr(context, unit) {
  const libraryDirectory = dirname(paths.qwenAsrBinary);
  const native = JSON.parse(commandOutput(paths.qwenAsrBinary, [
    'transcribe',
    paths.qwenAsrModel,
    context.normalizedPath,
    unit.locale.slice(0, 2),
    'gpu',
  ], {
    env: { ...process.env, DYLD_LIBRARY_PATH: libraryDirectory },
    maxBuffer: 32 * 1024 * 1024,
  }).trim());
  return {
    text: native.text.trim(),
    runtime: {
      backend: 'qwen3-asr-0.6b-mlx-direct',
      requested_language: unit.locale.slice(0, 2),
      inference_ms: native.elapsed_ms,
      peak_memory_bytes: native.peak_memory_bytes,
      prompt_tokens: native.prompt_length,
      generation_tokens: native.generation_tokens,
      finish_reason: native.finish_reason,
      stop_token: native.stop_token,
    },
  };
}

async function loadSttContext(unit, audioUnits, selectedPassages) {
  const audioUnit = audioUnits.get(unit.audio_unit_id);
  if (!audioUnit) throw new Error(`${unit.id}: audio unit is missing`);
  const audioDirectory = join(paths.audio, unit.audio_unit_id);
  const audioManifest = await readJson(join(audioDirectory, 'manifest.json'));
  const normalizedPath = join(
    audioDirectory,
    audioManifest.normalized_audio.path,
  );
  const bytes = await readFile(normalizedPath);
  if (
    bytes.length !== audioManifest.normalized_audio.byte_count ||
    sha256(bytes) !== audioManifest.normalized_audio.sha256
  ) {
    throw new Error(`${unit.id}: normalized audio digest differs`);
  }
  const owned = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  const selected = selectedPassages.get(audioUnit.passage_id);
  if (!selected) throw new Error(`${unit.id}: selected passage is missing`);
  const reference = (await readFile(
    join(paths.text, `${audioUnit.passage_id}.txt`),
    'utf8',
  )).trim();
  if (sha256(Buffer.from(reference)) !== selected.passage.spoken_sha256) {
    throw new Error(`${unit.id}: reference text digest differs`);
  }
  return {
    audioUnit,
    audioDirectory,
    audioManifest,
    normalizedPath,
    samples: new Float32Array(owned),
    selected,
    reference,
  };
}

async function writeSttArtifact(unit, context, result, state) {
  const outputDirectory = join(paths.stt, unit.id);
  const temporary = await mkdtemp(join(paths.output, '.stt-tmp-'));
  try {
    const text = result.text.trim();
    const transcriptBytes = Buffer.from(`${text}\n`);
    const manifest = {
      schema_version: '1.0.0',
      id: unit.id,
      plan_id: state.plan_id,
      plan_revision: state.plan_revision,
      source_selection_revision: state.source_selection_revision,
      source_revision: gitRevision(),
      captured_at: new Date().toISOString(),
      unit,
      source_audio: {
        audio_unit_id: unit.audio_unit_id,
        normalized_path: relative(paths.output, context.normalizedPath),
        normalized_sha256: context.audioManifest.normalized_audio.sha256,
        duration_ms: context.audioManifest.normalized_audio.duration_ms,
      },
      reference: {
        passage_id: context.audioUnit.passage_id,
        text_sha256: context.selected.passage.spoken_sha256,
        word_count: words(context.reference).length,
      },
      transcript: {
        path: 'transcript.txt',
        sha256: sha256(transcriptBytes),
        character_count: text.length,
        word_count: words(text).length,
        text,
      },
      quality: {
        wer: errorRate(words(context.reference), words(text)),
        cer: errorRate(characters(context.reference), characters(text)),
      },
      runtime: result.runtime,
      disposition: 'measured-local-transcript',
    };
    await writeFile(join(temporary, 'transcript.txt'), transcriptBytes);
    await writeFile(join(temporary, 'manifest.json'), json(manifest));
    await mkdir(dirname(outputDirectory), { recursive: true });
    await rename(temporary, outputDirectory);
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}

async function validSttArtifact(directory, unit) {
  try {
    const manifest = await readJson(join(directory, 'manifest.json'));
    if (manifest.id !== unit.id || manifest.plan_revision !==
      (await readJson(paths.plan)).revision) return false;
    const transcript = await readFile(join(directory, manifest.transcript.path));
    if (sha256(transcript) !== manifest.transcript.sha256) return false;
    const audioManifest = await readJson(join(
      paths.audio,
      unit.audio_unit_id,
      'manifest.json',
    ));
    return manifest.source_audio.normalized_sha256 ===
      audioManifest.normalized_audio.sha256;
  } catch {
    return false;
  }
}

function normalizeSegments(segments, audioDurationMs) {
  return (segments ?? []).map((segment) => {
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
  if (referenceItems.length === 0) return hypothesisItems.length === 0 ? 0 : 1;
  let previous = Array.from(
    { length: hypothesisItems.length + 1 },
    (_, index) => index,
  );
  for (let referenceIndex = 1;
    referenceIndex <= referenceItems.length;
    referenceIndex += 1) {
    const current = [referenceIndex];
    for (let hypothesisIndex = 1;
      hypothesisIndex <= hypothesisItems.length;
      hypothesisIndex += 1) {
      const substitution = previous[hypothesisIndex - 1] +
        (referenceItems[referenceIndex - 1] ===
        hypothesisItems[hypothesisIndex - 1] ? 0 : 1);
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
  const completedBySttModel = {};
  let completedAudio = 0;
  let completedStt = 0;
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
  try {
    const entries = await readdir(paths.stt, { withFileTypes: true });
    const units = new Map(ledger.stt_units.map((unit) => [unit.id, unit]));
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const unit = units.get(entry.name);
      if (!unit || !await validSttArtifact(join(paths.stt, entry.name), unit)) {
        continue;
      }
      completedStt += 1;
      completedBySttModel[unit.stt_model] =
        (completedBySttModel[unit.stt_model] ?? 0) + 1;
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
      completed: completedStt,
      remaining: ledger.counts.total_stt_units - completedStt,
      completed_by_model: completedBySttModel,
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
      identifier: 'com.apple.voice.compact.de-AT.fallback-female',
      language: 'de-AT',
      quality_raw_value: 2,
      gender_raw_value: 2,
    },
    {
      identifier: 'com.apple.voice.compact.de-DE.exact-male',
      language: 'de-DE',
      quality_raw_value: 1,
      gender_raw_value: 1,
    },
    {
      identifier: 'com.apple.voice.compact.de-DE.exact-female',
      language: 'de-DE',
      quality_raw_value: 1,
      gender_raw_value: 2,
    },
  ]);
  if (
    !selected[0].identifier.endsWith('exact-female') ||
    !selected[1].identifier.endsWith('exact-male')
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
  if (
    errorRate(['one', 'two'], ['one', 'too']) !== 0.5 ||
    sttModelsForOption('all').size !== 3 ||
    !sttModelsForOption('qwen3').has('qwen3-asr-0.6b-mlx-direct')
  ) {
    throw new Error('self-test: STT selection or error rate failed');
  }
  process.stdout.write('factorial local runner: self-test passed\n');
}
