#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({
  options: {
    'self-test': { type: 'boolean', default: false },
  },
});
const manifestDirectory = join(repoRoot, 'spikes/tts-calibration');
const manifestFiles = [
  'qwen3-tts-1.7b-voicedesign-bf16.json',
  'voxtral-tts-4b-mlx-bf16.json',
  'kugelaudio-0-open-bf16.json',
];
const manifests = await Promise.all(
  manifestFiles.map(async (file) => ({
    file,
    data: JSON.parse(await readFile(join(manifestDirectory, file), 'utf8')),
  })),
);
const plan = JSON.parse(await readFile(
  join(repoRoot, 'benchmarks/fixtures/synthetic-roundtrip-plan.json'),
  'utf8',
));

const failures = validate(manifests, plan);
if (values['self-test']) {
  const changed = structuredClone(manifests);
  changed[0].data.artifact.snapshot_bytes += 1;
  if (validate(changed, plan).length === 0) {
    failures.push('self-test failed to reject changed snapshot bytes');
  }
}
if (failures.length > 0) {
  for (const failure of failures) process.stderr.write(`- ${failure}\n`);
  process.exit(1);
}
process.stdout.write(
  `Validated ${manifests.length} TTS calibration manifests: ` +
    `${manifests.reduce((sum, item) => sum + item.data.artifact.snapshot_bytes, 0)} total snapshot bytes\n`,
);

function validate(items, acceptedPlan) {
  const errors = [];
  const runtimeRevision = '64e8416c303fb3b3463dab8eb4ebd78c55a87c1a';
  const expected = new Map([
    ['qwen3-tts-12hz-1.7b-voicedesign-mlx-bf16', {
      candidate: 'qwen3-tts-1.7b-voicedesign-mlx-audio',
      source: 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
      sourceRevision: '5ecdb67327fd37bb2e042aab12ff7391903235d3',
      repository: 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
      revision: '7d3824abff87e49756bb0f83fb5411de75d160c4',
      license: 'Apache-2.0',
      snapshotBytes: 4520194992,
      artifactCount: 14,
      role: 'required-generator',
      status: 'pinned-pending-execution',
      voiceMode: 'description',
      profiles: [
        'qwen-de-clear-documentary',
        'qwen-de-warm-podcast',
        'qwen-en-clear-documentary',
        'qwen-en-warm-podcast',
      ],
      planStatus: 'pinned-for-calibration',
    }],
    ['voxtral-4b-tts-2603-mlx-bf16', {
      candidate: 'voxtral-tts-4b-bf16-mlx-audio',
      source: 'mistralai/Voxtral-4B-TTS-2603',
      sourceRevision: 'b81be46c3777f88621676791b512bb01dc1cb970',
      repository: 'mlx-community/Voxtral-4B-TTS-2603-mlx-bf16',
      revision: 'dd85c02adbae551f5bb29ded35ee60ccdfb90927',
      license: 'CC-BY-NC-4.0',
      snapshotBytes: 8037606951,
      artifactCount: 27,
      role: 'required-generator',
      status: 'pinned-pending-execution',
      voiceMode: 'preset',
      profiles: [
        'voxtral-de-female',
        'voxtral-de-male',
        'voxtral-en-casual-female',
        'voxtral-en-casual-male',
      ],
      planStatus: 'pinned-for-calibration-reference-only',
    }],
    ['kugelaudio-0-open-bf16', {
      candidate: 'kugelaudio-0-open-mlx-audio',
      source: 'kugelaudio/kugelaudio-0-open',
      sourceRevision: '22d6ed9b8627394d4bf3e5af35285ad37c4d72b6',
      repository: 'kugelaudio/kugelaudio-0-open',
      revision: '22d6ed9b8627394d4bf3e5af35285ad37c4d72b6',
      license: 'MIT',
      snapshotBytes: 18688088739,
      artifactCount: 14,
      role: 'bounded-challenger',
      status: 'pinned-capability-limited-pending-execution',
      voiceMode: 'implicit-default',
      profiles: ['kugel-de-implicit-default'],
      planStatus: 'pinned-for-German-calibration-capability-limited',
    }],
  ]);
  const seenIds = new Set();
  for (const { file, data } of items) {
    const prefix = `${file}: `;
    const contract = expected.get(data.id);
    if (!contract || seenIds.has(data.id)) {
      errors.push(`${prefix}unknown or duplicate manifest id ${data.id}`);
      continue;
    }
    seenIds.add(data.id);
    if (data.$schema !== './model-manifest.schema.json' || data.task !== 'tts') {
      errors.push(`${prefix}schema and task must identify the TTS calibration contract`);
    }
    if (data.source?.repository !== contract.source ||
        data.source?.observed_revision !== contract.sourceRevision ||
        data.source?.license !== contract.license ||
        !data.source?.supported_languages?.includes('German')) {
      errors.push(`${prefix}source revision, license, or German capability differs`);
    }
    if (data.artifact?.repository !== contract.repository ||
        data.artifact?.revision !== contract.revision ||
        data.artifact?.license !== contract.license ||
        data.artifact?.snapshot_bytes !== contract.snapshotBytes) {
      errors.push(`${prefix}artifact identity or snapshot size differs`);
    }
    if (data.reference_runtime?.repository !== 'Blaizzy/mlx-audio' ||
        data.reference_runtime?.revision !== runtimeRevision ||
        data.reference_runtime?.version !== '0.4.5' ||
        data.reference_runtime?.license !== 'MIT') {
      errors.push(`${prefix}runtime must preserve the accepted mlx-audio 0.4.5 commit`);
    }
    const artifacts = data.artifact?.artifacts ?? [];
    const paths = new Set();
    let snapshotBytes = 0;
    let weightBytes = 0;
    for (const artifact of artifacts) {
      if (!(artifact.path?.length > 0) || paths.has(artifact.path)) {
        errors.push(`${prefix}artifact paths must be non-empty and unique`);
      }
      paths.add(artifact.path);
      if (!(Number.isInteger(artifact.bytes) && artifact.bytes > 0) ||
          !/^[0-9a-f]{64}$/.test(artifact.sha256 ?? '') ||
          !['downloaded-at-pinned-revision', 'hugging-face-lfs-metadata']
            .includes(artifact.digest_source)) {
        errors.push(`${prefix}${artifact.path}: invalid bytes, digest, or source`);
      }
      snapshotBytes += artifact.bytes ?? 0;
      if (artifact.path.endsWith('.safetensors')) weightBytes += artifact.bytes ?? 0;
    }
    if (artifacts.length !== contract.artifactCount ||
        snapshotBytes !== data.artifact?.snapshot_bytes ||
        weightBytes !== data.artifact?.weight_bytes) {
      errors.push(`${prefix}artifact count or byte totals do not reconcile`);
    }
    if (data.calibration?.role !== contract.role ||
        data.calibration?.status !== contract.status ||
        data.calibration?.voice_mode !== contract.voiceMode ||
        data.calibration?.sample_rate_hz !== 24_000 ||
        JSON.stringify(data.calibration?.profiles?.map((profile) => profile.id)) !==
          JSON.stringify(contract.profiles)) {
      errors.push(`${prefix}calibration role, voice mode, or profiles differ`);
    }
    for (const profile of data.calibration?.profiles ?? []) {
      if (profile.locale === 'de-DE' && profile.passage_id !== 'synthetic-de-origin') {
        errors.push(`${prefix}${profile.id}: German profile must use synthetic-de-origin`);
      }
      if (profile.locale === 'en-US' && profile.passage_id !== 'synthetic-en-reasoning') {
        errors.push(`${prefix}${profile.id}: English profile must use synthetic-en-reasoning`);
      }
      if (contract.voiceMode === 'description' &&
          (profile.voice !== null || !(profile.instruction?.length > 0))) {
        errors.push(`${prefix}${profile.id}: description profile requires only an instruction`);
      }
      if (contract.voiceMode === 'preset' &&
          (!(profile.voice?.length > 0) || profile.instruction !== null)) {
        errors.push(`${prefix}${profile.id}: preset profile requires only a voice id`);
      }
      if (contract.voiceMode === 'implicit-default' &&
          (profile.voice !== null || profile.instruction !== null)) {
        errors.push(`${prefix}${profile.id}: implicit-default profile cannot claim a voice`);
      }
    }
    const candidate = acceptedPlan.tts_candidates.find(
      (item) => item.id === contract.candidate,
    );
    if (!candidate ||
        !candidate.model.includes(`${contract.repository}@${contract.revision}`) ||
        candidate.source_revision !== runtimeRevision ||
        candidate.status !== contract.planStatus) {
      errors.push(`${prefix}synthetic plan does not reference the accepted artifact and runtime`);
    }
  }
  if (seenIds.size !== expected.size) {
    errors.push('the three accepted TTS calibration manifests must all be present');
  }
  return errors;
}
