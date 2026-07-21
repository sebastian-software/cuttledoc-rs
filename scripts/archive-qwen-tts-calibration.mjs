#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({
  options: {
    result: { type: 'string' },
    reference: { type: 'string' },
    'pcm16-wav': { type: 'string' },
    'asset-dir': { type: 'string' },
    'result-dir': { type: 'string' },
    purpose: { type: 'string' },
    disposition: { type: 'string' },
    summary: { type: 'string' },
  },
});

for (const option of [
  'result',
  'reference',
  'pcm16-wav',
  'asset-dir',
  'result-dir',
  'purpose',
  'disposition',
  'summary',
]) {
  if (!(values[option]?.length > 0)) {
    throw new Error(`--${option} is required`);
  }
}

const resultInputPath = resolve(values.result);
const referenceInputPath = resolve(values.reference);
const pcm16WavPath = resolve(values['pcm16-wav']);
const assetDirectory = withinRepo(values['asset-dir']);
const resultDirectory = withinRepo(values['result-dir']);
const audioPath = join(assetDirectory, 'audio.opus');
await access(audioPath);

const [runBytes, referenceBytes, pcm16Wav, audio, selection, modelManifest] =
  await Promise.all([
    readFile(resultInputPath),
    readFile(referenceInputPath),
    readFile(pcm16WavPath),
    readFile(audioPath),
    readJson('benchmarks/fixtures/synthetic-roundtrip-selection.json'),
    readJson('spikes/tts-calibration/qwen3-tts-1.7b-voicedesign-bf16.json'),
  ]);
const run = JSON.parse(runBytes.toString('utf8'));
const profile = modelManifest.calibration.profiles.find(
  (item) => item.id === run.candidate?.voice?.profile_id,
);
const selected = selection.sources
  .flatMap((source) => source.passages.map((passage) => ({ source, passage })))
  .find(({ passage }) => passage.id === run.input?.passage_id);
if (!profile || !selected || profile.passage_id !== selected.passage.id) {
  throw new Error('run does not identify one pinned Qwen calibration profile');
}
if (run.selection_revision !== selection.revision ||
    run.candidate?.model?.revision !== modelManifest.artifact.revision ||
    run.candidate?.runtime?.revision !== modelManifest.reference_runtime.revision ||
    run.input?.source_id !== selected.source.id ||
    run.input?.locale !== selected.source.locale ||
    run.input?.character_count !== selected.passage.character_count ||
    run.input?.text_sha256 !== selected.passage.spoken_sha256) {
  throw new Error('run differs from the pinned selection or model manifest');
}
if (sha256(referenceBytes) !== selected.passage.spoken_sha256 ||
    [...referenceBytes.toString('utf8')].length !== selected.passage.character_count) {
  throw new Error('reference differs from the pinned passage');
}
if (run.result?.asr_content_checks?.status !== 'complete' ||
    run.result.asr_content_checks.backends?.length !== 5) {
  throw new Error('run must contain all five ASR content checks');
}

const resultDisposition = values.disposition.startsWith('passed-')
  ? `${values.disposition}-listening-pending`
  : values.disposition;
run.conclusion = {
  ...run.conclusion,
  calibration_disposition: resultDisposition,
  next: values.disposition.startsWith('passed-')
    ? 'Complete listening review before promoting this voice and content cell.'
    : 'Keep this failed output as reproducible evidence and require a predeclared control before promotion.',
};
const archivedResult = Buffer.from(`${JSON.stringify(run, null, 2)}\n`, 'utf8');
const resultDigest = sha256(archivedResult);
const probe = ffprobe(audioPath);
const wers = run.result.asr_content_checks.backends.map(
  (item) => item.quality.wer,
);
const sourceCopyright = selected.source.kind === 'mediawiki'
  ? `${selected.source.attribution}`
  : '2026 Cuttledoc contributors';
const attribution = attributionMarkdown(
  selected.source,
  selected.passage,
  run,
  values.summary,
  pcm16Wav,
);
const manifest = {
  schema_version: '1.0.0',
  asset_id:
    `${selected.passage.id}.qwen3-tts-1.7b-voicedesign-warm.opus-64k-1`,
  purpose: values.purpose,
  license: 'CC-BY-SA-4.0',
  reference: {
    path: 'reference.txt',
    bytes: referenceBytes.length,
    characters: selected.passage.character_count,
    sha256: sha256(referenceBytes),
    spoken_text_sha256: selected.passage.spoken_sha256,
    source_id: selected.source.id,
    passage_id: selected.passage.id,
    attribution: 'ATTRIBUTION.md',
  },
  generation: {
    run_id: run.run_id,
    record_sha256: resultDigest,
    model: modelManifest.artifact.repository,
    model_revision: modelManifest.artifact.revision,
    runtime:
      `${modelManifest.reference_runtime.repository}@` +
      modelManifest.reference_runtime.revision,
    voice_profile: profile.id,
    voice_instruction: profile.instruction,
    seed: profile.seed,
    locale: profile.locale,
    lossless_f32le: {
      location: 'local-required',
      sample_rate_hz: run.result.audio.sample_rate_hz,
      channel_count: run.result.audio.channel_count,
      sample_count: run.result.audio.sample_count,
      bytes: run.result.audio.byte_count,
      sha256: run.result.audio.sha256,
    },
    lossless_pcm16_wav: {
      location: 'local-required',
      sha256: sha256(pcm16Wav),
    },
  },
  archive: {
    path: 'audio.opus',
    container: 'Ogg',
    codec: 'Opus',
    bytes: audio.length,
    sha256: sha256(audio),
    duration_ms: Number.parseFloat(probe.format.duration) * 1_000,
    decoded_sample_rate_hz: Number.parseInt(probe.streams[0].sample_rate, 10),
    channel_count: probe.streams[0].channels,
    target_bit_rate_bps: 64_000,
    container_bit_rate_bps: Number.parseInt(probe.format.bit_rate, 10),
    variable_bit_rate: true,
    application: 'audio',
    frame_duration_ms: 20,
    expected_packet_loss_percent: 0,
    encoder: `${ffmpegVersion()} with libopus`,
    deterministic_repeat_sha256_match: true,
    reproduction_script: 'scripts/encode-benchmark-opus.mjs',
  },
  codec_control: 'benchmarks/controls/opus-codec-qwen3-tts-de-1.json',
  calibration_finding: {
    disposition: values.disposition,
    receiver_count: 5,
    reached_max_tokens: run.result.termination.reached_max_tokens,
    observed_wer_range: [Math.min(...wers), Math.max(...wers)],
    summary: values.summary,
  },
  distribution: {
    git_repository: true,
    git_archive: false,
    cargo_package: false,
    npm_package: false,
  },
};

await mkdir(resultDirectory, { recursive: true });
for (const path of [
  join(resultDirectory, 'result.json'),
  join(assetDirectory, 'manifest.json'),
  join(assetDirectory, 'reference.txt'),
  join(assetDirectory, 'ATTRIBUTION.md'),
]) {
  try {
    await access(path);
    throw new Error(`refusing to overwrite existing archive file: ${path}`);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
}
await Promise.all([
  writeFile(join(resultDirectory, 'result.json'), archivedResult),
  writeFile(join(assetDirectory, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`, 'utf8'),
  writeFile(join(assetDirectory, 'reference.txt'), referenceBytes),
  writeFile(join(assetDirectory, 'ATTRIBUTION.md'), attribution, 'utf8'),
  writeFile(join(assetDirectory, 'audio.opus.license'),
    `SPDX-FileCopyrightText: ${sourceCopyright}\n` +
    'SPDX-FileCopyrightText: 2026 Cuttledoc contributors ' +
    '(generated sound recording and benchmark packaging)\n' +
    'SPDX-License-Identifier: CC-BY-SA-4.0\n', 'utf8'),
  writeFile(join(assetDirectory, 'reference.txt.license'),
    `SPDX-FileCopyrightText: ${sourceCopyright}\n` +
    'SPDX-License-Identifier: CC-BY-SA-4.0\n', 'utf8'),
]);

process.stdout.write(`${JSON.stringify({
  run_id: run.run_id,
  result_sha256: resultDigest,
  asset_id: manifest.asset_id,
  audio_sha256: manifest.archive.sha256,
  disposition: values.disposition,
}, null, 2)}\n`);

function withinRepo(path) {
  const result = resolve(repoRoot, path);
  if (result !== repoRoot && !result.startsWith(`${repoRoot}/`)) {
    throw new Error(`archive path escapes the repository: ${path}`);
  }
  return result;
}

async function readJson(path) {
  return JSON.parse(await readFile(join(repoRoot, path), 'utf8'));
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function commandOutput(command, args) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} failed: ${result.stderr || result.error?.message || result.status}`,
    );
  }
  return result.stdout;
}

function ffprobe(path) {
  return JSON.parse(commandOutput('ffprobe', [
    '-v',
    'error',
    '-show_entries',
    'format=duration,size,bit_rate:stream=codec_name,sample_rate,channels',
    '-of',
    'json',
    path,
  ]));
}

function ffmpegVersion() {
  return commandOutput('ffmpeg', ['-version'])
    .split('\n')[0]
    .split(' ')
    .slice(0, 3)
    .join(' ');
}

function attributionMarkdown(source, passage, run, summary, pcm16Wav) {
  const sourcePin = source.kind === 'mediawiki'
    ? `Exact revision: \`${source.revision_id}\``
    : `Exact source digest: \`${source.revision}\``;
  return `# Attribution and provenance

## Source text

- Work: “${source.title}”
- ${sourcePin}
- Revision: [permanent link](${source.revision_url})
- Source history: [history](${source.history_url})
- Source license: [CC BY-SA 4.0](${source.license_url})
- Passage selector and digest: \`${passage.id}\` in
  \`benchmarks/fixtures/synthetic-roundtrip-selection.json\`

The passage is materialized without an additional spoken-text transform.

## Generated recording

- Generator: \`${run.candidate.model.repository}\`
- Model revision: \`${run.candidate.model.revision}\`
- Model license: ${run.candidate.model.license}
- Runtime: \`${run.candidate.runtime.repository}\` ${run.candidate.runtime.version}
- Runtime revision: \`${run.candidate.runtime.revision}\`
- Runtime license: ${run.candidate.runtime.license}
- Official compute runtime: Apple MLX ${run.environment.packages.mlx}
- Voice profile: \`${run.candidate.voice.profile_id}\`, Seed ${run.candidate.voice.seed}
- Voice instruction: “${run.candidate.voice.instruction}”
- Generation record:
  \`benchmarks/raw/${run.run_id}/result.json\`

The lossless generated source is mono 24 kHz float PCM with SHA-256
\`${run.result.audio.sha256}\`. It was quantized to the PCM16 WAV with SHA-256
\`${sha256(pcm16Wav)}\` before the checked-in Ogg Opus encode.

${summary} Listening review remains required before treating this recording as
a voice-quality exemplar.

The generated recording and its source passage are redistributed under CC
BY-SA 4.0. Preserve this attribution, the license link, and the indication of
changes when redistributing either file.
`;
}
