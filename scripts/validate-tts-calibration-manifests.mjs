#!/usr/bin/env node

import { createHash } from 'node:crypto';
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
const selection = JSON.parse(await readFile(
  join(repoRoot, 'benchmarks/fixtures/synthetic-roundtrip-selection.json'),
  'utf8',
));
const qwenCalibrationRun = JSON.parse(await readFile(
  join(
    repoRoot,
    'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-de-clear-documentary.1/result.json',
  ),
  'utf8',
));
const qwenWarmCalibrationRun = JSON.parse(await readFile(
  join(
    repoRoot,
    'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-de-warm-podcast.1/result.json',
  ),
  'utf8',
));
const qwenEnglishCalibrationRun = JSON.parse(await readFile(
  join(
    repoRoot,
    'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-en-warm-podcast.1/result.json',
  ),
  'utf8',
));
const qwenNativeCalibrationRun = JSON.parse(await readFile(
  join(
    repoRoot,
    'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-de-warm-native.1/result.json',
  ),
  'utf8',
));
const qwenDialogueCalibrationRun = JSON.parse(await readFile(
  join(
    repoRoot,
    'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-de-warm-dialogue.1/result.json',
  ),
  'utf8',
));
const calibrationAssetDirectory = join(
  repoRoot,
  'benchmarks/assets/synthetic/de-DE/qwen3-tts-1.7b-voicedesign-clear/' +
    'synthetic-de-origin',
);
const qwenCalibrationAsset = JSON.parse(await readFile(
  join(calibrationAssetDirectory, 'manifest.json'),
  'utf8',
));
const [qwenCalibrationAudio, qwenCalibrationReference, qwenCalibrationAttribution] =
  await Promise.all([
    readFile(join(calibrationAssetDirectory, 'audio.opus')),
    readFile(join(calibrationAssetDirectory, 'reference.txt')),
    readFile(join(calibrationAssetDirectory, 'ATTRIBUTION.md'), 'utf8'),
  ]);
const warmCalibrationAssetDirectory = join(
  repoRoot,
  'benchmarks/assets/synthetic/de-DE/qwen3-tts-1.7b-voicedesign-warm/' +
    'synthetic-de-origin',
);
const qwenWarmCalibrationAsset = JSON.parse(await readFile(
  join(warmCalibrationAssetDirectory, 'manifest.json'),
  'utf8',
));
const [qwenWarmCalibrationAudio, qwenWarmCalibrationReference,
  qwenWarmCalibrationAttribution] = await Promise.all([
  readFile(join(warmCalibrationAssetDirectory, 'audio.opus')),
  readFile(join(warmCalibrationAssetDirectory, 'reference.txt')),
  readFile(join(warmCalibrationAssetDirectory, 'ATTRIBUTION.md'), 'utf8'),
]);
const englishCalibrationAssetDirectory = join(
  repoRoot,
  'benchmarks/assets/synthetic/en-US/qwen3-tts-1.7b-voicedesign-warm/' +
    'synthetic-en-reasoning',
);
const qwenEnglishCalibrationAsset = JSON.parse(await readFile(
  join(englishCalibrationAssetDirectory, 'manifest.json'),
  'utf8',
));
const [qwenEnglishCalibrationAudio, qwenEnglishCalibrationReference,
  qwenEnglishCalibrationAttribution] = await Promise.all([
  readFile(join(englishCalibrationAssetDirectory, 'audio.opus')),
  readFile(join(englishCalibrationAssetDirectory, 'reference.txt')),
  readFile(join(englishCalibrationAssetDirectory, 'ATTRIBUTION.md'), 'utf8'),
]);
const nativeCalibrationAssetDirectory = join(
  repoRoot,
  'benchmarks/assets/synthetic/de-DE/qwen3-tts-1.7b-voicedesign-warm/' +
    'synthetic-de-native',
);
const qwenNativeCalibrationAsset = JSON.parse(await readFile(
  join(nativeCalibrationAssetDirectory, 'manifest.json'),
  'utf8',
));
const [qwenNativeCalibrationAudio, qwenNativeCalibrationReference,
  qwenNativeCalibrationAttribution] = await Promise.all([
  readFile(join(nativeCalibrationAssetDirectory, 'audio.opus')),
  readFile(join(nativeCalibrationAssetDirectory, 'reference.txt')),
  readFile(join(nativeCalibrationAssetDirectory, 'ATTRIBUTION.md'), 'utf8'),
]);
const dialogueCalibrationAssetDirectory = join(
  repoRoot,
  'benchmarks/assets/synthetic/de-DE/qwen3-tts-1.7b-voicedesign-warm/' +
    'synthetic-de-dialogue',
);
const qwenDialogueCalibrationAsset = JSON.parse(await readFile(
  join(dialogueCalibrationAssetDirectory, 'manifest.json'),
  'utf8',
));
const [qwenDialogueCalibrationAudio, qwenDialogueCalibrationReference,
  qwenDialogueCalibrationAttribution] = await Promise.all([
  readFile(join(dialogueCalibrationAssetDirectory, 'audio.opus')),
  readFile(join(dialogueCalibrationAssetDirectory, 'reference.txt')),
  readFile(join(dialogueCalibrationAssetDirectory, 'ATTRIBUTION.md'), 'utf8'),
]);
const additionalQwenCalibrationSpecs = [
  {
    profileId: 'qwen-en-warm-native',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-en-warm-native.1',
    assetDirectory:
      'benchmarks/assets/synthetic/en-US/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-en-native',
    purpose: 'reproducible-passed-content-type-calibration',
    disposition: 'passed-native-content-gate',
    resultDisposition: 'passed-native-content-gate-listening-pending',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'passes the native-factual lexical gate',
  },
  {
    profileId: 'qwen-en-warm-dialogue',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-en-warm-dialogue.1',
    assetDirectory:
      'benchmarks/assets/synthetic/en-US/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-en-dialogue',
    purpose: 'reproducible-failed-content-type-calibration',
    disposition: 'failed-shared-repetition-and-truncation',
    resultDisposition: 'failed-shared-repetition-and-truncation',
    referencePathProven: false,
    reachedMaxTokens: true,
    attributionNeedle: 'reproducible failed generation control',
  },
  {
    profileId: 'qwen-es-warm-technical',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-es-warm-technical.1',
    assetDirectory:
      'benchmarks/assets/synthetic/es-419/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-es-technical',
    purpose: 'reproducible-passed-content-type-calibration',
    disposition: 'passed-technical-content-gate-receiver-spread',
    resultDisposition:
      'passed-technical-content-gate-receiver-spread-listening-pending',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'Whisper and Voxtral make two and three word edits',
  },
  {
    profileId: 'qwen-es-warm-native',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-es-warm-native.1',
    assetDirectory:
      'benchmarks/assets/synthetic/es-419/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-es-native',
    purpose: 'reproducible-passed-content-type-calibration',
    disposition: 'passed-native-content-gate-pronunciation-review-required',
    resultDisposition:
      'passed-native-content-gate-pronunciation-review-required-listening-pending',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'keeps that phrase open for pronunciation review',
  },
  {
    profileId: 'qwen-es-warm-dialogue',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-es-warm-dialogue.1',
    assetDirectory:
      'benchmarks/assets/synthetic/es-419/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-es-dialogue',
    purpose: 'reproducible-passed-content-type-calibration',
    disposition: 'passed-dialogue-content-gate-receiver-spread',
    resultDisposition:
      'passed-dialogue-content-gate-receiver-spread-listening-pending',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'Parakeet alone makes fifteen edits',
  },
  {
    profileId: 'qwen-fr-warm-technical',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-fr-warm-technical.1',
    assetDirectory:
      'benchmarks/assets/synthetic/fr-FR/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-fr-technical',
    purpose: 'reproducible-failed-content-type-calibration',
    disposition: 'failed-shared-mid-passage-truncation',
    resultDisposition: 'failed-shared-mid-passage-truncation',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'stops after the third of five technical list items',
  },
  {
    profileId: 'qwen-fr-warm-native',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-fr-warm-native.1',
    assetDirectory:
      'benchmarks/assets/synthetic/fr-FR/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-fr-native',
    purpose: 'reproducible-passed-content-type-calibration',
    disposition: 'passed-native-content-gate',
    resultDisposition: 'passed-native-content-gate-listening-pending',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'Voxtral is lexically exact',
  },
  {
    profileId: 'qwen-fr-warm-dialogue',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-fr-warm-dialogue.1',
    assetDirectory:
      'benchmarks/assets/synthetic/fr-FR/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-fr-dialogue',
    purpose: 'reproducible-failed-content-type-calibration',
    disposition: 'failed-shared-repetition-and-truncation',
    resultDisposition: 'failed-shared-repetition-and-truncation',
    referencePathProven: false,
    reachedMaxTokens: true,
    attributionNeedle: 'dialogue stops after “Jonas sourit”',
  },
  {
    profileId: 'qwen-pt-warm-technical',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-pt-warm-technical.1',
    assetDirectory:
      'benchmarks/assets/synthetic/pt-BR/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-pt-technical',
    purpose: 'reproducible-failed-content-type-calibration',
    disposition: 'failed-critical-technical-term-pronunciation',
    resultDisposition: 'failed-critical-technical-term-pronunciation',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'garble the embedded critical term “Agentic AI”',
  },
  {
    profileId: 'qwen-pt-warm-native',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-pt-warm-native.1',
    assetDirectory:
      'benchmarks/assets/synthetic/pt-BR/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-pt-native',
    purpose: 'reproducible-passed-content-type-calibration',
    disposition: 'passed-native-content-gate-receiver-spread',
    resultDisposition:
      'passed-native-content-gate-receiver-spread-listening-pending',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'Voxtral makes one word edit',
  },
  {
    profileId: 'qwen-pt-warm-dialogue',
    resultDirectory:
      'benchmarks/raw/phase5.qwen3-tts-1.7b-voicedesign.' +
      'qwen-pt-warm-dialogue.1',
    assetDirectory:
      'benchmarks/assets/synthetic/pt-BR/' +
      'qwen3-tts-1.7b-voicedesign-warm/synthetic-pt-dialogue',
    purpose: 'reproducible-passed-content-type-calibration',
    disposition: 'passed-dialogue-content-gate',
    resultDisposition: 'passed-dialogue-content-gate-listening-pending',
    referencePathProven: true,
    reachedMaxTokens: false,
    attributionNeedle: 'complete dialogue with one to four word edits',
  },
];
const additionalQwenCalibrationEvidence = await Promise.all(
  additionalQwenCalibrationSpecs.map(async (spec) => {
    const resultBytes = await readFile(join(repoRoot, spec.resultDirectory, 'result.json'));
    const assetDirectory = join(repoRoot, spec.assetDirectory);
    const [assetBytes, audio, reference, attribution] = await Promise.all([
      readFile(join(assetDirectory, 'manifest.json')),
      readFile(join(assetDirectory, 'audio.opus')),
      readFile(join(assetDirectory, 'reference.txt')),
      readFile(join(assetDirectory, 'ATTRIBUTION.md'), 'utf8'),
    ]);
    return {
      ...spec,
      resultBytes,
      run: JSON.parse(resultBytes.toString('utf8')),
      asset: JSON.parse(assetBytes.toString('utf8')),
      audio,
      reference,
      attribution,
    };
  }),
);

const failures = validate(manifests, plan);
failures.push(...validateQwenCalibrationRun(
  qwenCalibrationRun,
  manifests[0].data,
  selection,
));
failures.push(...validateQwenCalibrationAsset(
  qwenCalibrationAsset,
  qwenCalibrationAudio,
  qwenCalibrationReference,
  qwenCalibrationAttribution,
  qwenCalibrationRun,
));
failures.push(...validateQwenWarmCalibrationRun(
  qwenWarmCalibrationRun,
  manifests[0].data,
  selection,
));
failures.push(...validateQwenWarmCalibrationAsset(
  qwenWarmCalibrationAsset,
  qwenWarmCalibrationAudio,
  qwenWarmCalibrationReference,
  qwenWarmCalibrationAttribution,
  qwenWarmCalibrationRun,
));
failures.push(...validateQwenEnglishCalibrationRun(
  qwenEnglishCalibrationRun,
  manifests[0].data,
  selection,
));
failures.push(...validateQwenEnglishCalibrationAsset(
  qwenEnglishCalibrationAsset,
  qwenEnglishCalibrationAudio,
  qwenEnglishCalibrationReference,
  qwenEnglishCalibrationAttribution,
  qwenEnglishCalibrationRun,
));
failures.push(...validateQwenNativeCalibrationRun(
  qwenNativeCalibrationRun,
  manifests[0].data,
  selection,
));
failures.push(...validateQwenNativeCalibrationAsset(
  qwenNativeCalibrationAsset,
  qwenNativeCalibrationAudio,
  qwenNativeCalibrationReference,
  qwenNativeCalibrationAttribution,
  qwenNativeCalibrationRun,
));
failures.push(...validateQwenDialogueCalibrationRun(
  qwenDialogueCalibrationRun,
  manifests[0].data,
  selection,
));
failures.push(...validateQwenDialogueCalibrationAsset(
  qwenDialogueCalibrationAsset,
  qwenDialogueCalibrationAudio,
  qwenDialogueCalibrationReference,
  qwenDialogueCalibrationAttribution,
  qwenDialogueCalibrationRun,
));
for (const evidence of additionalQwenCalibrationEvidence) {
  failures.push(...validateAdditionalQwenCalibration(
    evidence,
    manifests[0].data,
    selection,
  ));
}
if (values['self-test']) {
  const changed = structuredClone(manifests);
  changed[0].data.artifact.snapshot_bytes += 1;
  if (validate(changed, plan).length === 0) {
    failures.push('self-test failed to reject changed snapshot bytes');
  }
  const changedRun = structuredClone(qwenCalibrationRun);
  changedRun.result.asr_content_checks.backends[0].quality.word_edits += 1;
  if (validateQwenCalibrationRun(
    changedRun,
    manifests[0].data,
    selection,
  ).length === 0) {
    failures.push('self-test failed to reject changed calibration WER evidence');
  }
  const changedAsset = structuredClone(qwenCalibrationAsset);
  changedAsset.archive.bytes += 1;
  if (validateQwenCalibrationAsset(
    changedAsset,
    qwenCalibrationAudio,
    qwenCalibrationReference,
    qwenCalibrationAttribution,
    qwenCalibrationRun,
  ).length === 0) {
    failures.push('self-test failed to reject changed calibration asset bytes');
  }
  const changedWarmRun = structuredClone(qwenWarmCalibrationRun);
  changedWarmRun.result.asr_content_checks.backends[0].quality.word_edits += 1;
  if (validateQwenWarmCalibrationRun(
    changedWarmRun,
    manifests[0].data,
    selection,
  ).length === 0) {
    failures.push('self-test failed to reject changed warm-profile WER evidence');
  }
  const changedWarmAsset = structuredClone(qwenWarmCalibrationAsset);
  changedWarmAsset.archive.bytes += 1;
  if (validateQwenWarmCalibrationAsset(
    changedWarmAsset,
    qwenWarmCalibrationAudio,
    qwenWarmCalibrationReference,
    qwenWarmCalibrationAttribution,
    qwenWarmCalibrationRun,
  ).length === 0) {
    failures.push('self-test failed to reject changed warm-profile asset bytes');
  }
  const changedEnglishRun = structuredClone(qwenEnglishCalibrationRun);
  changedEnglishRun.result.asr_content_checks.backends[0].quality.character_edits += 1;
  if (validateQwenEnglishCalibrationRun(
    changedEnglishRun,
    manifests[0].data,
    selection,
  ).length === 0) {
    failures.push('self-test failed to reject changed English-profile evidence');
  }
  const changedEnglishAsset = structuredClone(qwenEnglishCalibrationAsset);
  changedEnglishAsset.archive.bytes += 1;
  if (validateQwenEnglishCalibrationAsset(
    changedEnglishAsset,
    qwenEnglishCalibrationAudio,
    qwenEnglishCalibrationReference,
    qwenEnglishCalibrationAttribution,
    qwenEnglishCalibrationRun,
  ).length === 0) {
    failures.push('self-test failed to reject changed English-profile asset bytes');
  }
  const changedNativeRun = structuredClone(qwenNativeCalibrationRun);
  changedNativeRun.result.asr_content_checks.backends[0].quality.word_edits += 1;
  if (validateQwenNativeCalibrationRun(
    changedNativeRun,
    manifests[0].data,
    selection,
  ).length === 0) {
    failures.push('self-test failed to reject changed native-German evidence');
  }
  const changedNativeAsset = structuredClone(qwenNativeCalibrationAsset);
  changedNativeAsset.archive.bytes += 1;
  if (validateQwenNativeCalibrationAsset(
    changedNativeAsset,
    qwenNativeCalibrationAudio,
    qwenNativeCalibrationReference,
    qwenNativeCalibrationAttribution,
    qwenNativeCalibrationRun,
  ).length === 0) {
    failures.push('self-test failed to reject changed native-German asset bytes');
  }
  const changedDialogueRun = structuredClone(qwenDialogueCalibrationRun);
  changedDialogueRun.result.asr_content_checks.backends[0].quality.word_edits += 1;
  if (validateQwenDialogueCalibrationRun(
    changedDialogueRun,
    manifests[0].data,
    selection,
  ).length === 0) {
    failures.push('self-test failed to reject changed German-dialogue evidence');
  }
  const changedDialogueAsset = structuredClone(qwenDialogueCalibrationAsset);
  changedDialogueAsset.archive.bytes += 1;
  if (validateQwenDialogueCalibrationAsset(
    changedDialogueAsset,
    qwenDialogueCalibrationAudio,
    qwenDialogueCalibrationReference,
    qwenDialogueCalibrationAttribution,
    qwenDialogueCalibrationRun,
  ).length === 0) {
    failures.push('self-test failed to reject changed German-dialogue asset bytes');
  }
  const changedAdditional = structuredClone(additionalQwenCalibrationEvidence[0]);
  changedAdditional.run.result.asr_content_checks.backends[0].quality.word_edits += 1;
  if (validateAdditionalQwenCalibration(
    changedAdditional,
    manifests[0].data,
    selection,
  ).length === 0) {
    failures.push('self-test failed to reject changed additional Qwen evidence');
  }
}
if (failures.length > 0) {
  for (const failure of failures) process.stderr.write(`- ${failure}\n`);
  process.exit(1);
}
process.stdout.write(
  `Validated ${manifests.length} TTS calibration manifests: ` +
    `${manifests.reduce((sum, item) => sum + item.data.artifact.snapshot_bytes, 0)} ` +
    `total snapshot bytes, ${5 + additionalQwenCalibrationEvidence.length} ` +
    `measured calibration profiles, and ${5 + additionalQwenCalibrationEvidence.length} ` +
    'Opus assets\n',
);

function validateAdditionalQwenCalibration(evidence, manifest, acceptedSelection) {
  const errors = [];
  const prefix = `${evidence.profileId}: `;
  const profile = manifest.calibration.profiles.find(
    (item) => item.id === evidence.profileId,
  );
  const selected = acceptedSelection.sources
    .flatMap((source) => source.passages.map((passage) => ({ source, passage })))
    .find(({ passage }) => passage.id === profile?.passage_id);
  const { run, asset } = evidence;
  const resultDigest = createHash('sha256').update(evidence.resultBytes).digest('hex');
  const audioDigest = createHash('sha256').update(evidence.audio).digest('hex');
  const referenceDigest = createHash('sha256').update(evidence.reference).digest('hex');
  if (!profile || !selected ||
      run.schema_version !== '1.0.0' ||
      run.run_id !== `phase5.qwen3-tts-1.7b-voicedesign.${profile.id}.1` ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '') ||
      !acceptedSelectionRevision(run.selection_revision, acceptedSelection) ||
      run.purpose !== 'calibration') {
    errors.push(`${prefix}run identity or selected passage differs`);
    return errors;
  }
  if (run.candidate?.id !== 'qwen3-tts-1.7b-voicedesign-mlx-audio' ||
      run.candidate?.model?.repository !== manifest.artifact.repository ||
      run.candidate?.model?.revision !== manifest.artifact.revision ||
      run.candidate?.model?.snapshot_bytes !== manifest.artifact.snapshot_bytes ||
      run.candidate?.runtime?.revision !== manifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !== manifest.reference_runtime.version ||
      run.candidate?.voice?.profile_id !== profile.id ||
      run.candidate?.voice?.instruction !== profile.instruction ||
      run.candidate?.voice?.language !== profile.language ||
      run.candidate?.voice?.locale !== profile.locale ||
      run.candidate?.voice?.seed !== profile.seed) {
    errors.push(`${prefix}candidate, runtime, or profile differs`);
  }
  if (run.input?.passage_id !== selected.passage.id ||
      run.input?.source_id !== selected.source.id ||
      run.input?.locale !== selected.source.locale ||
      run.input?.character_count !== selected.passage.character_count ||
      run.input?.text_sha256 !== selected.passage.spoken_sha256 ||
      run.input?.license !== selected.source.license) {
    errors.push(`${prefix}input differs from the pinned passage`);
  }
  const expectedGeneration = {
    profile_id: profile.id,
    passage_id: profile.passage_id,
    locale: profile.locale,
    language: profile.language,
    voice: profile.voice,
    instruction: profile.instruction,
    seed: profile.seed,
    ...manifest.calibration.generation,
    sample_rate_hz: manifest.calibration.sample_rate_hz,
  };
  if (JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(expectedGeneration) ||
      run.procedure?.stream !== false ||
      !run.procedure?.command?.includes(
        'scripts/run-qwen3-tts-voicedesign-calibration.sh',
      ) ||
      !run.procedure?.model_verification?.includes('SHA-256')) {
    errors.push(`${prefix}generation contract differs`);
  }
  const audio = run.result?.audio;
  const timing = run.result?.timing;
  const expectedStatus = evidence.reachedMaxTokens ? 'partial' : 'measured';
  if (run.result?.status !== expectedStatus ||
      audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !== 24_000 ||
      audio?.channel_count !== 1 ||
      !(audio?.sample_count > 0) ||
      audio?.byte_count !== audio?.sample_count * 4 ||
      audio?.duration_ms !== audio?.sample_count / 24 ||
      !/^[0-9a-f]{64}$/.test(audio?.sha256 ?? '') ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.rms > 0 && audio.rms < 1) ||
      !(timing?.model_load_ms > 0) ||
      !(timing?.first_audio_ms > 0) ||
      !(timing?.complete_synthesis_ms >= timing?.first_audio_ms) ||
      timing?.output_count !== 1 ||
      Math.abs(timing?.real_time_factor -
        timing?.complete_synthesis_ms / audio?.duration_ms) > 1e-12 ||
      run.result?.termination?.reached_max_tokens !== evidence.reachedMaxTokens ||
      run.result?.termination?.configured_max_tokens !== 1_200 ||
      run.result?.resources?.model_snapshot_bytes !== manifest.artifact.snapshot_bytes ||
      !(run.result?.resources?.mlx_peak_memory_bytes > 0)) {
    errors.push(`${prefix}audio, timing, resources, or termination differs`);
  }
  const checks = run.result?.asr_content_checks;
  const normalized = checks?.normalized_audio;
  if (checks?.status !== 'complete' ||
      normalized?.sample_format !== 'f32le' ||
      normalized?.sample_rate_hz !== 16_000 ||
      normalized?.channel_count !== 1 ||
      normalized?.sample_count !== audio?.duration_ms * 16 ||
      normalized?.byte_count !== normalized?.sample_count * 4 ||
      normalized?.duration_ms !== audio?.duration_ms ||
      !/^[0-9a-f]{64}$/.test(normalized?.sha256 ?? '')) {
    errors.push(`${prefix}normalized PCM evidence differs`);
  }
  const expectedBackendIds = new Set([
    'whisper-large-v3-turbo-coreml-whispercpp',
    'parakeet-tdt-0.6b-v3-coreml',
    'qwen3-asr-0.6b-mlx-direct',
    'voxtral-realtime-4b-mlx-direct-2400ms',
    'apple-speechtranscriber',
  ]);
  const backends = checks?.backends ?? [];
  if (backends.length !== 5 ||
      new Set(backends.map((item) => item.backend?.id)).size !== 5 ||
      backends.some((item) => !expectedBackendIds.has(item.backend?.id))) {
    errors.push(`${prefix}must contain the five required ASR backends`);
  }
  for (const check of backends) {
    const quality = check.quality;
    const transcriptDigest = createHash('sha256')
      .update(check.transcript?.text ?? '')
      .digest('hex');
    if (check.transcript?.sha256 !== transcriptDigest ||
        !(quality?.reference_word_count > 0) ||
        !(quality?.hypothesis_word_count >= 0) ||
        !(quality?.word_edits >= 0) ||
        Math.abs(quality?.wer -
          quality?.word_edits / quality?.reference_word_count) > 1e-15 ||
        !(quality?.reference_character_count > 0) ||
        !(quality?.hypothesis_character_count >= 0) ||
        !(quality?.character_edits >= 0) ||
        Math.abs(quality?.cer -
          quality?.character_edits / quality?.reference_character_count) > 1e-15 ||
        !(check.timing?.complete_inference_ms > 0) ||
        Math.abs(check.timing?.real_time_factor -
          check.timing?.complete_inference_ms / audio?.duration_ms) > 1e-12) {
      errors.push(`${prefix}${check.backend?.id}: transcript or metrics differ`);
    }
  }
  if (checks?.comparison?.completed_backend_count !== 5 ||
      checks?.comparison?.expected_backend_count !== 5 ||
      checks?.comparison?.remaining_backends?.length !== 0 ||
      run.conclusion?.reference_path_proven !== evidence.referencePathProven ||
      run.conclusion?.calibration_disposition !== evidence.resultDisposition) {
    errors.push(`${prefix}comparison or conclusion differs`);
  }
  if (asset.schema_version !== '1.0.0' ||
      asset.asset_id !==
        `${selected.passage.id}.qwen3-tts-1.7b-voicedesign-warm.opus-64k-1` ||
      asset.purpose !== evidence.purpose ||
      asset.license !== 'CC-BY-SA-4.0' ||
      asset.generation?.run_id !== run.run_id ||
      asset.generation?.record_sha256 !== resultDigest ||
      asset.generation?.model_revision !== run.candidate.model.revision ||
      asset.generation?.voice_profile !== profile.id ||
      asset.generation?.voice_instruction !== profile.instruction ||
      asset.generation?.seed !== profile.seed ||
      asset.generation?.locale !== profile.locale) {
    errors.push(`${prefix}asset identity or generation provenance differs`);
  }
  if (asset.reference?.bytes !== evidence.reference.length ||
      asset.reference?.characters !== selected.passage.character_count ||
      asset.reference?.sha256 !== referenceDigest ||
      asset.reference?.spoken_text_sha256 !== selected.passage.spoken_sha256 ||
      asset.reference?.source_id !== selected.source.id ||
      asset.reference?.passage_id !== selected.passage.id) {
    errors.push(`${prefix}asset reference metadata differs`);
  }
  if (asset.generation?.lossless_f32le?.sample_count !== audio?.sample_count ||
      asset.generation?.lossless_f32le?.bytes !== audio?.byte_count ||
      asset.generation?.lossless_f32le?.sha256 !== audio?.sha256 ||
      asset.archive?.path !== 'audio.opus' ||
      asset.archive?.codec !== 'Opus' ||
      asset.archive?.bytes !== evidence.audio.length ||
      asset.archive?.sha256 !== audioDigest ||
      asset.archive?.target_bit_rate_bps !== 64_000 ||
      asset.archive?.variable_bit_rate !== true ||
      asset.archive?.application !== 'audio' ||
      asset.archive?.frame_duration_ms !== 20 ||
      asset.archive?.deterministic_repeat_sha256_match !== true ||
      asset.calibration_finding?.disposition !== evidence.disposition ||
      asset.calibration_finding?.receiver_count !== 5 ||
      !evidence.attribution.includes('CC BY-SA 4.0') ||
      !evidence.attribution.includes(evidence.attributionNeedle)) {
    errors.push(`${prefix}archive, finding, or attribution differs`);
  }
  return errors;
}

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
      status:
        'multilingual-content-type-expansion-complete-four-failures-' +
        'listening-pending',
      voiceMode: 'description',
      profiles: [
        'qwen-de-clear-documentary',
        'qwen-de-warm-podcast',
        'qwen-de-warm-native',
        'qwen-de-warm-dialogue',
        'qwen-en-clear-documentary',
        'qwen-en-warm-podcast',
        'qwen-en-warm-native',
        'qwen-en-warm-dialogue',
        'qwen-es-warm-technical',
        'qwen-es-warm-native',
        'qwen-es-warm-dialogue',
        'qwen-fr-warm-technical',
        'qwen-fr-warm-native',
        'qwen-fr-warm-dialogue',
        'qwen-pt-warm-technical',
        'qwen-pt-warm-native',
        'qwen-pt-warm-dialogue',
      ],
      planStatus:
        'multilingual-content-type-expansion-complete-four-failures-' +
        'listening-pending',
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
    const acceptedPassagesByLocale = new Map([
      ['de-DE', new Set([
        'synthetic-de-origin',
        'synthetic-de-native',
        'synthetic-de-dialogue',
      ])],
      ['en-US', new Set([
        'synthetic-en-reasoning',
        'synthetic-en-native',
        'synthetic-en-dialogue',
      ])],
      ['es-419', new Set([
        'synthetic-es-technical',
        'synthetic-es-native',
        'synthetic-es-dialogue',
      ])],
      ['fr-FR', new Set([
        'synthetic-fr-technical',
        'synthetic-fr-native',
        'synthetic-fr-dialogue',
      ])],
      ['pt-BR', new Set([
        'synthetic-pt-technical',
        'synthetic-pt-native',
        'synthetic-pt-dialogue',
      ])],
    ]);
    for (const profile of data.calibration?.profiles ?? []) {
      const acceptedPassages = acceptedPassagesByLocale.get(profile.locale);
      if (contract.candidate === 'qwen3-tts-1.7b-voicedesign-mlx-audio' &&
          (!acceptedPassages || !acceptedPassages.has(profile.passage_id))) {
        errors.push(`${prefix}${profile.id}: profile uses an unknown locale/content cell`);
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

function acceptedSelectionRevision(revision, selection) {
  return revision === selection.revision ||
    revision === 'synthetic-roundtrip-passages-1' ||
    revision === 'synthetic-roundtrip-passages-2' ||
    revision === 'synthetic-roundtrip-passages-3';
}

function validateQwenCalibrationRun(run, manifest, acceptedSelection) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign clear-profile result: ';
  const profile = manifest.calibration.profiles.find(
    (item) => item.id === 'qwen-de-clear-documentary',
  );
  const passage = acceptedSelection.sources
    .flatMap((source) => source.passages.map((item) => ({ source, item })))
    .find(({ item }) => item.id === profile.passage_id);
  if (run.schema_version !== '1.0.0' ||
      run.run_id !==
        'phase5.qwen3-tts-1.7b-voicedesign.qwen-de-clear-documentary.1' ||
      run.source_revision !== '6cbd92dd0464182e9458ffe7301be8a57b4bfdbe' ||
      !acceptedSelectionRevision(run.selection_revision, acceptedSelection) ||
      run.purpose !== 'calibration') {
    errors.push(`${prefix}identity or revisions differ`);
  }
  if (run.candidate?.id !== 'qwen3-tts-1.7b-voicedesign-mlx-audio' ||
      run.candidate?.model?.repository !== manifest.artifact.repository ||
      run.candidate?.model?.revision !== manifest.artifact.revision ||
      run.candidate?.model?.license !== manifest.artifact.license ||
      run.candidate?.model?.snapshot_bytes !== manifest.artifact.snapshot_bytes ||
      run.candidate?.runtime?.revision !== manifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !== manifest.reference_runtime.version ||
      run.candidate?.voice?.profile_id !== profile.id ||
      run.candidate?.voice?.instruction !== profile.instruction ||
      run.candidate?.voice?.seed !== profile.seed) {
    errors.push(`${prefix}candidate, runtime, or voice profile differs`);
  }
  if (!passage ||
      run.input?.passage_id !== passage.item.id ||
      run.input?.source_id !== passage.source.id ||
      run.input?.locale !== passage.source.locale ||
      run.input?.character_count !== passage.item.character_count ||
      run.input?.text_sha256 !== passage.item.spoken_sha256 ||
      run.input?.license !== passage.source.license) {
    errors.push(`${prefix}input differs from the pinned passage`);
  }
  const expectedGeneration = {
    profile_id: profile.id,
    passage_id: profile.passage_id,
    locale: profile.locale,
    language: profile.language,
    voice: profile.voice,
    instruction: profile.instruction,
    seed: profile.seed,
    ...manifest.calibration.generation,
    sample_rate_hz: manifest.calibration.sample_rate_hz,
  };
  if (JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(expectedGeneration) ||
      run.procedure?.stream !== false ||
      !run.procedure?.command?.includes(
        'scripts/run-qwen3-tts-voicedesign-calibration.sh',
      ) ||
      !run.procedure?.model_verification?.includes('SHA-256')) {
    errors.push(`${prefix}execution contract differs`);
  }

  const audio = run.result?.audio;
  if (run.result?.status !== 'measured' ||
      audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !== 24_000 ||
      audio?.channel_count !== 1 ||
      audio?.sample_count !== 1_253_760 ||
      audio?.byte_count !== audio.sample_count * 4 ||
      audio?.duration_ms !== 52_240 ||
      audio?.sha256 !==
        'af4e290315377c15f0235f75e07276fa9019f74ebb66b2d08e691944d7a1b19b' ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.rms > 0 && audio.rms < 1)) {
    errors.push(`${prefix}audio evidence differs`);
  }
  const timing = run.result?.timing;
  if (!(timing?.model_load_ms > 0) ||
      !(timing?.first_audio_ms > 0) ||
      !(timing?.complete_synthesis_ms >= timing?.first_audio_ms) ||
      timing?.output_count !== 1 ||
      timing?.token_count !== 653 ||
      Math.abs(
        timing?.real_time_factor -
          timing?.complete_synthesis_ms / audio?.duration_ms,
      ) > 1e-12 ||
      run.result?.termination?.reached_max_tokens !== false ||
      run.result?.termination?.configured_max_tokens !== 1_200 ||
      run.result?.resources?.model_snapshot_bytes !==
        manifest.artifact.snapshot_bytes ||
      !(run.result?.resources?.mlx_peak_memory_bytes > 0)) {
    errors.push(`${prefix}timing, resources, or termination differ`);
  }

  const checks = run.result?.asr_content_checks;
  if (checks?.status !== 'complete' ||
      checks?.normalized_audio?.sample_format !== 'f32le' ||
      checks?.normalized_audio?.sample_rate_hz !== 16_000 ||
      checks?.normalized_audio?.channel_count !== 1 ||
      checks?.normalized_audio?.sample_count !== 835_840 ||
      checks?.normalized_audio?.byte_count !== 3_343_360 ||
      checks?.normalized_audio?.duration_ms !== audio?.duration_ms ||
      checks?.normalized_audio?.sha256 !==
        '64d3718e313ae5b00e8c402c45b940c048ce6d338e3e37815b55ba74223fdcba') {
    errors.push(`${prefix}normalized PCM evidence differs`);
  }
  const expectedBackends = new Map([
    ['whisper-large-v3-turbo-coreml-whispercpp', [
      '449697880368d6ef88d09274bd28c6d6b0a784d900b9d068fd6d170c6a6190cc',
      5, 105, 31, 719,
    ]],
    ['parakeet-tdt-0.6b-v3-coreml', [
      'd74eaa3243027c447bb30114e49fc8bf38e9f482ef4ea84bd683d746bce01370',
      4, 105, 9, 699,
    ]],
    ['qwen3-asr-0.6b-mlx-direct', [
      '6802e150a53be15bcfc918373dc80e1f40edbcee818b1cf31e2013f968e800a2',
      7, 104, 27, 715,
    ]],
    ['voxtral-realtime-4b-mlx-direct-2400ms', [
      '709119e918575ad096cb90d6ee88469042b347aaec8f1d4ffd73a21b7a56acc5',
      5, 104, 8, 698,
    ]],
    ['apple-speechtranscriber', [
      'f84de7e8e5f21d50829bcbd7db20b1e4c4f29c4a0200f45a09a9e960cfd7f717',
      7, 104, 16, 692,
    ]],
  ]);
  const measuredBackends = checks?.backends ?? [];
  if (measuredBackends.length !== expectedBackends.size ||
      new Set(measuredBackends.map((item) => item.backend?.id)).size !==
        measuredBackends.length) {
    errors.push(`${prefix}must contain five unique ASR backends`);
  }
  for (const check of measuredBackends) {
    const expected = expectedBackends.get(check.backend?.id);
    const quality = check.quality;
    const digest = createHash('sha256')
      .update(check.transcript?.text ?? '')
      .digest('hex');
    if (!expected ||
        check.transcript?.sha256 !== digest ||
        check.transcript?.sha256 !== expected[0] ||
        quality?.reference_word_count !== 103 ||
        quality?.word_edits !== expected[1] ||
        quality?.hypothesis_word_count !== expected[2] ||
        Math.abs(quality?.wer - quality?.word_edits / 103) > 1e-15 ||
        quality?.reference_character_count !== 692 ||
        quality?.character_edits !== expected[3] ||
        quality?.hypothesis_character_count !== expected[4] ||
        Math.abs(quality?.cer - quality?.character_edits / 692) > 1e-15 ||
        !(check.timing?.complete_inference_ms > 0) ||
        Math.abs(
          check.timing?.real_time_factor -
            check.timing?.complete_inference_ms / audio?.duration_ms,
        ) > 1e-12) {
      errors.push(`${prefix}${check.backend?.id}: transcript or metrics differ`);
    }
  }
  if (checks?.comparison?.completed_backend_count !== 5 ||
      checks?.comparison?.expected_backend_count !== 5 ||
      checks?.comparison?.critical_findings?.shared_content_omission !== false ||
      checks?.comparison?.critical_findings?.shared_truncation !== false ||
      checks?.comparison?.critical_findings?.year_1962_recovered_by_any_backend !==
        false ||
      checks?.comparison?.critical_findings?.probable_tts_content_error !== true ||
      run.conclusion?.calibration_disposition !==
        'failed-critical-token-pronunciation' ||
      run.conclusion?.second_profile_required !== true) {
    errors.push(`${prefix}critical finding or disposition differs`);
  }
  return errors;
}

function validateQwenCalibrationAsset(
  asset,
  audio,
  reference,
  attribution,
  run,
) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign clear-profile asset: ';
  const audioDigest = createHash('sha256').update(audio).digest('hex');
  const referenceDigest = createHash('sha256').update(reference).digest('hex');
  if (asset.schema_version !== '1.0.0' ||
      asset.asset_id !==
        'synthetic-de-origin.qwen3-tts-1.7b-voicedesign-clear.opus-64k-1' ||
      asset.purpose !== 'reproducible-critical-token-calibration-failure' ||
      asset.license !== 'CC-BY-SA-4.0') {
    errors.push(`${prefix}identity or license differs`);
  }
  if (asset.reference?.bytes !== reference.length ||
      asset.reference?.characters !== run.input.character_count ||
      asset.reference?.sha256 !== referenceDigest ||
      asset.reference?.spoken_text_sha256 !== run.input.text_sha256 ||
      asset.reference?.source_id !== run.input.source_id ||
      asset.reference?.passage_id !== run.input.passage_id) {
    errors.push(`${prefix}reference metadata or bytes differ`);
  }
  if (asset.generation?.run_id !== run.run_id ||
      asset.generation?.model_revision !== run.candidate.model.revision ||
      asset.generation?.voice_profile !== run.candidate.voice.profile_id ||
      asset.generation?.voice_instruction !== run.candidate.voice.instruction ||
      asset.generation?.seed !== run.candidate.voice.seed ||
      asset.generation?.lossless_f32le?.sample_count !==
        run.result.audio.sample_count ||
      asset.generation?.lossless_f32le?.bytes !== run.result.audio.byte_count ||
      asset.generation?.lossless_f32le?.sha256 !== run.result.audio.sha256) {
    errors.push(`${prefix}generation provenance differs from the run`);
  }
  if (asset.archive?.path !== 'audio.opus' ||
      asset.archive?.codec !== 'Opus' ||
      asset.archive?.bytes !== audio.length ||
      asset.archive?.sha256 !== audioDigest ||
      asset.archive?.target_bit_rate_bps !== 64_000 ||
      asset.archive?.variable_bit_rate !== true ||
      asset.archive?.application !== 'audio' ||
      asset.archive?.frame_duration_ms !== 20 ||
      asset.archive?.deterministic_repeat_sha256_match !== true) {
    errors.push(`${prefix}archive metadata or bytes differ`);
  }
  if (asset.calibration_finding?.disposition !==
        run.conclusion.calibration_disposition ||
      asset.calibration_finding?.token !== '1962' ||
      asset.calibration_finding?.receiver_count !== 5 ||
      asset.calibration_finding?.receiver_recovery_count !== 0 ||
      !attribution.includes('CC BY-SA 4.0') ||
      !attribution.includes('reproducible calibration failure')) {
    errors.push(`${prefix}finding or attribution caveat differs`);
  }
  return errors;
}

function validateQwenWarmCalibrationRun(run, manifest, acceptedSelection) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign warm-profile result: ';
  const profile = manifest.calibration.profiles.find(
    (item) => item.id === 'qwen-de-warm-podcast',
  );
  const passage = acceptedSelection.sources
    .flatMap((source) => source.passages.map((item) => ({ source, item })))
    .find(({ item }) => item.id === profile.passage_id);
  if (run.schema_version !== '1.0.0' ||
      run.run_id !==
        'phase5.qwen3-tts-1.7b-voicedesign.qwen-de-warm-podcast.1' ||
      run.source_revision !== '82c044b2d04e3daf97a65530fa9b9b3dc90ee71f' ||
      !acceptedSelectionRevision(run.selection_revision, acceptedSelection) ||
      run.purpose !== 'calibration') {
    errors.push(`${prefix}identity or revisions differ`);
  }
  if (run.candidate?.id !== 'qwen3-tts-1.7b-voicedesign-mlx-audio' ||
      run.candidate?.model?.repository !== manifest.artifact.repository ||
      run.candidate?.model?.revision !== manifest.artifact.revision ||
      run.candidate?.model?.license !== manifest.artifact.license ||
      run.candidate?.model?.snapshot_bytes !== manifest.artifact.snapshot_bytes ||
      run.candidate?.runtime?.revision !== manifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !== manifest.reference_runtime.version ||
      run.candidate?.voice?.profile_id !== profile.id ||
      run.candidate?.voice?.instruction !== profile.instruction ||
      run.candidate?.voice?.seed !== profile.seed) {
    errors.push(`${prefix}candidate, runtime, or voice profile differs`);
  }
  if (!passage ||
      run.input?.passage_id !== passage.item.id ||
      run.input?.source_id !== passage.source.id ||
      run.input?.locale !== passage.source.locale ||
      run.input?.character_count !== passage.item.character_count ||
      run.input?.text_sha256 !== passage.item.spoken_sha256 ||
      run.input?.license !== passage.source.license) {
    errors.push(`${prefix}input differs from the pinned passage`);
  }
  const expectedGeneration = {
    profile_id: profile.id,
    passage_id: profile.passage_id,
    locale: profile.locale,
    language: profile.language,
    voice: profile.voice,
    instruction: profile.instruction,
    seed: profile.seed,
    ...manifest.calibration.generation,
    sample_rate_hz: manifest.calibration.sample_rate_hz,
  };
  if (JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(expectedGeneration) ||
      run.procedure?.stream !== false ||
      !run.procedure?.command?.includes(
        'scripts/run-qwen3-tts-voicedesign-calibration.sh',
      ) ||
      !run.procedure?.model_verification?.includes('SHA-256')) {
    errors.push(`${prefix}execution contract differs`);
  }

  const audio = run.result?.audio;
  if (run.result?.status !== 'measured' ||
      audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !== 24_000 ||
      audio?.channel_count !== 1 ||
      audio?.sample_count !== 1_344_000 ||
      audio?.byte_count !== audio.sample_count * 4 ||
      audio?.duration_ms !== 56_000 ||
      audio?.sha256 !==
        'd47fe6927ad81229a098f6ed91bb4c80b8379a826c95eeb8d53f682edf1a2e6c' ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.rms > 0 && audio.rms < 1)) {
    errors.push(`${prefix}audio evidence differs`);
  }
  const timing = run.result?.timing;
  if (!(timing?.model_load_ms > 0) ||
      !(timing?.first_audio_ms > 0) ||
      !(timing?.complete_synthesis_ms >= timing?.first_audio_ms) ||
      timing?.output_count !== 1 ||
      timing?.token_count !== 700 ||
      Math.abs(
        timing?.real_time_factor -
          timing?.complete_synthesis_ms / audio?.duration_ms,
      ) > 1e-12 ||
      run.result?.termination?.reached_max_tokens !== false ||
      run.result?.termination?.configured_max_tokens !== 1_200 ||
      run.result?.resources?.model_snapshot_bytes !==
        manifest.artifact.snapshot_bytes ||
      !(run.result?.resources?.mlx_peak_memory_bytes > 0)) {
    errors.push(`${prefix}timing, resources, or termination differ`);
  }

  const checks = run.result?.asr_content_checks;
  if (checks?.status !== 'complete' ||
      checks?.normalized_audio?.sample_format !== 'f32le' ||
      checks?.normalized_audio?.sample_rate_hz !== 16_000 ||
      checks?.normalized_audio?.channel_count !== 1 ||
      checks?.normalized_audio?.sample_count !== 896_000 ||
      checks?.normalized_audio?.byte_count !== 3_584_000 ||
      checks?.normalized_audio?.duration_ms !== audio?.duration_ms ||
      checks?.normalized_audio?.sha256 !==
        'ea4e03f9659576ea9f6dcbffbd9447b6c50a91f46b1044bd9d65bb5d6d3d08b1') {
    errors.push(`${prefix}normalized PCM evidence differs`);
  }
  const expectedBackends = new Map([
    ['whisper-large-v3-turbo-coreml-whispercpp', [
      '4b026aec9b4358023747f4775c31a4243ea1a24c0bc4da8c3c22034fb840319e',
      3, 104, 3, 695,
    ]],
    ['parakeet-tdt-0.6b-v3-coreml', [
      '232615d0f5b4a2707e05456d08cdbdce10422f52469193bf7500fb0ed511194a',
      6, 104, 7, 695,
    ]],
    ['qwen3-asr-0.6b-mlx-direct', [
      'c18e0fd42ba47a2b42a9a52b6ec97d8779ea8af93e648045b742ef339c676361',
      4, 104, 4, 696,
    ]],
    ['voxtral-realtime-4b-mlx-direct-2400ms', [
      '47e7206970197af15b60ca6e84cc959cbedc63a19f15254d15554deea89fbbbc',
      3, 104, 3, 695,
    ]],
    ['apple-speechtranscriber', [
      '9320736ea6bff083c308d68fea3db2622eb6fe4572bfa29f6e1b13179edc2ae0',
      1, 103, 1, 692,
    ]],
  ]);
  const measuredBackends = checks?.backends ?? [];
  if (measuredBackends.length !== expectedBackends.size ||
      new Set(measuredBackends.map((item) => item.backend?.id)).size !==
        measuredBackends.length) {
    errors.push(`${prefix}must contain five unique ASR backends`);
  }
  for (const check of measuredBackends) {
    const expected = expectedBackends.get(check.backend?.id);
    const quality = check.quality;
    const digest = createHash('sha256')
      .update(check.transcript?.text ?? '')
      .digest('hex');
    if (!expected ||
        check.transcript?.sha256 !== digest ||
        check.transcript?.sha256 !== expected[0] ||
        quality?.reference_word_count !== 103 ||
        quality?.word_edits !== expected[1] ||
        quality?.hypothesis_word_count !== expected[2] ||
        Math.abs(quality?.wer - quality?.word_edits / 103) > 1e-15 ||
        quality?.reference_character_count !== 692 ||
        quality?.character_edits !== expected[3] ||
        quality?.hypothesis_character_count !== expected[4] ||
        Math.abs(quality?.cer - quality?.character_edits / 692) > 1e-15 ||
        !(check.timing?.complete_inference_ms > 0) ||
        Math.abs(
          check.timing?.real_time_factor -
            check.timing?.complete_inference_ms / audio?.duration_ms,
        ) > 1e-12) {
      errors.push(`${prefix}${check.backend?.id}: transcript or metrics differ`);
    }
  }
  if (checks?.comparison?.completed_backend_count !== 5 ||
      checks?.comparison?.expected_backend_count !== 5 ||
      checks?.comparison?.critical_findings?.shared_content_omission !== false ||
      checks?.comparison?.critical_findings?.shared_truncation !== false ||
      checks?.comparison?.critical_findings?.year_1962_recovered_by_all_backends !==
        true ||
      checks?.comparison?.critical_findings?.probable_tts_content_error !== false ||
      run.conclusion?.calibration_disposition !==
        'passed-lexical-critical-token-gate' ||
      run.conclusion?.second_profile_required !== false ||
      run.conclusion?.listening_required !== true) {
    errors.push(`${prefix}critical finding or disposition differs`);
  }
  return errors;
}

function validateQwenWarmCalibrationAsset(
  asset,
  audio,
  reference,
  attribution,
  run,
) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign warm-profile asset: ';
  const audioDigest = createHash('sha256').update(audio).digest('hex');
  const referenceDigest = createHash('sha256').update(reference).digest('hex');
  if (asset.schema_version !== '1.0.0' ||
      asset.asset_id !==
        'synthetic-de-origin.qwen3-tts-1.7b-voicedesign-warm.opus-64k-1' ||
      asset.purpose !== 'reproducible-passed-calibration-profile' ||
      asset.license !== 'CC-BY-SA-4.0') {
    errors.push(`${prefix}identity or license differs`);
  }
  if (asset.reference?.bytes !== reference.length ||
      asset.reference?.characters !== run.input.character_count ||
      asset.reference?.sha256 !== referenceDigest ||
      asset.reference?.spoken_text_sha256 !== run.input.text_sha256 ||
      asset.reference?.source_id !== run.input.source_id ||
      asset.reference?.passage_id !== run.input.passage_id) {
    errors.push(`${prefix}reference metadata or bytes differ`);
  }
  if (asset.generation?.run_id !== run.run_id ||
      asset.generation?.model_revision !== run.candidate.model.revision ||
      asset.generation?.voice_profile !== run.candidate.voice.profile_id ||
      asset.generation?.voice_instruction !== run.candidate.voice.instruction ||
      asset.generation?.seed !== run.candidate.voice.seed ||
      asset.generation?.lossless_f32le?.sample_count !==
        run.result.audio.sample_count ||
      asset.generation?.lossless_f32le?.bytes !== run.result.audio.byte_count ||
      asset.generation?.lossless_f32le?.sha256 !== run.result.audio.sha256) {
    errors.push(`${prefix}generation provenance differs from the run`);
  }
  if (asset.archive?.path !== 'audio.opus' ||
      asset.archive?.codec !== 'Opus' ||
      asset.archive?.bytes !== audio.length ||
      asset.archive?.sha256 !== audioDigest ||
      asset.archive?.target_bit_rate_bps !== 64_000 ||
      asset.archive?.variable_bit_rate !== true ||
      asset.archive?.application !== 'audio' ||
      asset.archive?.frame_duration_ms !== 20 ||
      asset.archive?.deterministic_repeat_sha256_match !== true) {
    errors.push(`${prefix}archive metadata or bytes differ`);
  }
  if (asset.calibration_finding?.disposition !==
        run.conclusion.calibration_disposition ||
      asset.calibration_finding?.token !== '1962' ||
      asset.calibration_finding?.receiver_count !== 5 ||
      asset.calibration_finding?.receiver_recovery_count !== 5 ||
      !attribution.includes('CC BY-SA 4.0') ||
      !attribution.includes('passes the lexical calibration')) {
    errors.push(`${prefix}finding or attribution caveat differs`);
  }
  return errors;
}

function validateQwenEnglishCalibrationRun(run, manifest, acceptedSelection) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign English warm-profile result: ';
  const profile = manifest.calibration.profiles.find(
    (item) => item.id === 'qwen-en-warm-podcast',
  );
  const passage = acceptedSelection.sources
    .flatMap((source) => source.passages.map((item) => ({ source, item })))
    .find(({ item }) => item.id === profile.passage_id);
  if (run.schema_version !== '1.0.0' ||
      run.run_id !==
        'phase5.qwen3-tts-1.7b-voicedesign.qwen-en-warm-podcast.1' ||
      run.source_revision !== '6143f888fd4629e7f5bac6630646806d93c5ce9a' ||
      !acceptedSelectionRevision(run.selection_revision, acceptedSelection) ||
      run.purpose !== 'calibration') {
    errors.push(`${prefix}identity or revisions differ`);
  }
  if (run.candidate?.id !== 'qwen3-tts-1.7b-voicedesign-mlx-audio' ||
      run.candidate?.model?.repository !== manifest.artifact.repository ||
      run.candidate?.model?.revision !== manifest.artifact.revision ||
      run.candidate?.model?.snapshot_bytes !== manifest.artifact.snapshot_bytes ||
      run.candidate?.runtime?.revision !== manifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !== manifest.reference_runtime.version ||
      run.candidate?.voice?.profile_id !== profile.id ||
      run.candidate?.voice?.instruction !== profile.instruction ||
      run.candidate?.voice?.locale !== profile.locale ||
      run.candidate?.voice?.seed !== profile.seed) {
    errors.push(`${prefix}candidate, runtime, or voice profile differs`);
  }
  if (!passage ||
      run.input?.passage_id !== passage.item.id ||
      run.input?.source_id !== passage.source.id ||
      run.input?.locale !== passage.source.locale ||
      run.input?.character_count !== passage.item.character_count ||
      run.input?.text_sha256 !== passage.item.spoken_sha256 ||
      run.input?.license !== passage.source.license) {
    errors.push(`${prefix}input differs from the pinned passage`);
  }
  const expectedGeneration = {
    profile_id: profile.id,
    passage_id: profile.passage_id,
    locale: profile.locale,
    language: profile.language,
    voice: profile.voice,
    instruction: profile.instruction,
    seed: profile.seed,
    ...manifest.calibration.generation,
    sample_rate_hz: manifest.calibration.sample_rate_hz,
  };
  if (JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(expectedGeneration) ||
      run.procedure?.stream !== false ||
      !run.procedure?.command?.includes(
        'scripts/run-qwen3-tts-voicedesign-calibration.sh',
      ) ||
      !run.procedure?.model_verification?.includes('SHA-256')) {
    errors.push(`${prefix}execution contract differs`);
  }

  const audio = run.result?.audio;
  const timing = run.result?.timing;
  if (run.result?.status !== 'measured' ||
      audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !== 24_000 ||
      audio?.channel_count !== 1 ||
      audio?.sample_count !== 1_361_280 ||
      audio?.byte_count !== 5_445_120 ||
      audio?.duration_ms !== 56_720 ||
      audio?.sha256 !==
        '6ec75583ea12ac8151e047e0917511b2d3c6751375629c1e16724770325699f5' ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.rms > 0 && audio.rms < 1) ||
      !(timing?.model_load_ms > 0) ||
      !(timing?.first_audio_ms > 0) ||
      !(timing?.complete_synthesis_ms >= timing?.first_audio_ms) ||
      timing?.output_count !== 1 ||
      timing?.token_count !== 709 ||
      Math.abs(timing?.real_time_factor -
        timing?.complete_synthesis_ms / audio?.duration_ms) > 1e-12 ||
      run.result?.termination?.reached_max_tokens !== false ||
      run.result?.termination?.configured_max_tokens !== 1_200 ||
      run.result?.resources?.model_snapshot_bytes !==
        manifest.artifact.snapshot_bytes ||
      !(run.result?.resources?.mlx_peak_memory_bytes > 0)) {
    errors.push(`${prefix}audio, timing, resources, or termination differ`);
  }

  const checks = run.result?.asr_content_checks;
  if (checks?.status !== 'complete' ||
      checks?.normalized_audio?.sample_format !== 'f32le' ||
      checks?.normalized_audio?.sample_rate_hz !== 16_000 ||
      checks?.normalized_audio?.channel_count !== 1 ||
      checks?.normalized_audio?.sample_count !== 907_520 ||
      checks?.normalized_audio?.byte_count !== 3_630_080 ||
      checks?.normalized_audio?.duration_ms !== audio?.duration_ms ||
      checks?.normalized_audio?.sha256 !==
        '73bde213ff46e32c6b6ba49defd7cf4fe328fb4a2c07e6a14cd903635e3b79c3') {
    errors.push(`${prefix}normalized PCM evidence differs`);
  }
  const expectedBackends = new Map([
    ['whisper-large-v3-turbo-coreml-whispercpp', [
      'en', 'e16eebe08cc6456c6019ece2986e8393ff1c3be9de5f11ce6121277e55c15134',
    ]],
    ['parakeet-tdt-0.6b-v3-coreml', [
      'model-managed',
      'd597a30767e36bd118cd03feae2593fb39fb24cf083046437bf56c2ef394adbb',
    ]],
    ['qwen3-asr-0.6b-mlx-direct', [
      'en', 'd66a0a1bb9c4d1305863cabde172f26c1437efe64ca0bc77448dee74f825579e',
    ]],
    ['voxtral-realtime-4b-mlx-direct-2400ms', [
      'model-managed',
      'e16eebe08cc6456c6019ece2986e8393ff1c3be9de5f11ce6121277e55c15134',
    ]],
    ['apple-speechtranscriber', [
      'en-US', 'e16eebe08cc6456c6019ece2986e8393ff1c3be9de5f11ce6121277e55c15134',
    ]],
  ]);
  const measuredBackends = checks?.backends ?? [];
  if (measuredBackends.length !== 5 ||
      new Set(measuredBackends.map((item) => item.backend?.id)).size !== 5) {
    errors.push(`${prefix}must contain five unique ASR backends`);
  }
  for (const check of measuredBackends) {
    const expected = expectedBackends.get(check.backend?.id);
    const quality = check.quality;
    const digest = createHash('sha256')
      .update(check.transcript?.text ?? '')
      .digest('hex');
    if (!expected ||
        check.backend?.locale !== expected[0] ||
        check.transcript?.sha256 !== digest ||
        check.transcript?.sha256 !== expected[1] ||
        quality?.reference_word_count !== 117 ||
        quality?.hypothesis_word_count !== 119 ||
        quality?.word_edits !== 3 ||
        Math.abs(quality?.wer - 3 / 117) > 1e-15 ||
        quality?.reference_character_count !== 716 ||
        quality?.hypothesis_character_count !== 716 ||
        quality?.character_edits !== 0 ||
        quality?.cer !== 0 ||
        !(check.timing?.complete_inference_ms > 0) ||
        Math.abs(check.timing?.real_time_factor -
          check.timing?.complete_inference_ms / audio?.duration_ms) > 1e-12) {
      errors.push(`${prefix}${check.backend?.id}: transcript or metrics differ`);
    }
  }
  if (checks?.comparison?.completed_backend_count !== 5 ||
      checks?.comparison?.expected_backend_count !== 5 ||
      checks?.comparison?.critical_findings?.shared_content_omission !== false ||
      checks?.comparison?.critical_findings?.shared_truncation !== false ||
      checks?.comparison?.critical_findings
        ?.exact_normalized_character_content_by_all_backends !== true ||
      checks?.comparison?.critical_findings?.probable_tts_content_error !== false ||
      run.conclusion?.calibration_disposition !== 'passed-lexical-content-gate' ||
      run.conclusion?.full_matrix_promotion_ready !== false ||
      run.conclusion?.listening_required !== true) {
    errors.push(`${prefix}critical finding or disposition differs`);
  }
  return errors;
}

function validateQwenEnglishCalibrationAsset(
  asset,
  audio,
  reference,
  attribution,
  run,
) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign English warm-profile asset: ';
  const audioDigest = createHash('sha256').update(audio).digest('hex');
  const referenceDigest = createHash('sha256').update(reference).digest('hex');
  if (asset.schema_version !== '1.0.0' ||
      asset.asset_id !==
        'synthetic-en-reasoning.qwen3-tts-1.7b-voicedesign-warm.opus-64k-1' ||
      asset.purpose !== 'reproducible-passed-calibration-profile' ||
      asset.license !== 'CC-BY-SA-4.0' ||
      asset.reference?.bytes !== reference.length ||
      asset.reference?.characters !== run.input.character_count ||
      asset.reference?.sha256 !== referenceDigest ||
      asset.reference?.spoken_text_sha256 !== run.input.text_sha256 ||
      asset.reference?.source_id !== run.input.source_id ||
      asset.reference?.passage_id !== run.input.passage_id) {
    errors.push(`${prefix}identity, license, or reference differs`);
  }
  if (asset.generation?.run_id !== run.run_id ||
      asset.generation?.model_revision !== run.candidate.model.revision ||
      asset.generation?.voice_profile !== run.candidate.voice.profile_id ||
      asset.generation?.voice_instruction !== run.candidate.voice.instruction ||
      asset.generation?.seed !== run.candidate.voice.seed ||
      asset.generation?.locale !== 'en-US' ||
      asset.generation?.lossless_f32le?.sample_count !==
        run.result.audio.sample_count ||
      asset.generation?.lossless_f32le?.bytes !== run.result.audio.byte_count ||
      asset.generation?.lossless_f32le?.sha256 !== run.result.audio.sha256) {
    errors.push(`${prefix}generation provenance differs from the run`);
  }
  if (asset.archive?.path !== 'audio.opus' ||
      asset.archive?.codec !== 'Opus' ||
      asset.archive?.bytes !== audio.length ||
      asset.archive?.sha256 !== audioDigest ||
      asset.archive?.target_bit_rate_bps !== 64_000 ||
      asset.archive?.variable_bit_rate !== true ||
      asset.archive?.application !== 'audio' ||
      asset.archive?.frame_duration_ms !== 20 ||
      asset.archive?.deterministic_repeat_sha256_match !== true ||
      asset.calibration_finding?.disposition !==
        run.conclusion.calibration_disposition ||
      asset.calibration_finding?.receiver_count !== 5 ||
      asset.calibration_finding?.receiver_exact_normalized_character_count !== 5 ||
      !attribution.includes('CC BY-SA 4.0') ||
      !attribution.includes('lexical calibration gate')) {
    errors.push(`${prefix}archive, finding, or attribution caveat differs`);
  }
  return errors;
}

function validateQwenNativeCalibrationRun(run, manifest, acceptedSelection) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign native-German result: ';
  const profile = manifest.calibration.profiles.find(
    (item) => item.id === 'qwen-de-warm-native',
  );
  const passage = acceptedSelection.sources
    .flatMap((source) => source.passages.map((item) => ({ source, item })))
    .find(({ item }) => item.id === profile.passage_id);
  if (run.schema_version !== '1.0.0' ||
      run.run_id !==
        'phase5.qwen3-tts-1.7b-voicedesign.qwen-de-warm-native.1' ||
      run.source_revision !== 'ebaee27d51511b858e9a008b8d2bf939183d56f1' ||
      !acceptedSelectionRevision(run.selection_revision, acceptedSelection) ||
      run.purpose !== 'calibration') {
    errors.push(`${prefix}identity or revisions differ`);
  }
  if (run.candidate?.id !== 'qwen3-tts-1.7b-voicedesign-mlx-audio' ||
      run.candidate?.model?.revision !== manifest.artifact.revision ||
      run.candidate?.model?.snapshot_bytes !== manifest.artifact.snapshot_bytes ||
      run.candidate?.runtime?.revision !== manifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !== manifest.reference_runtime.version ||
      run.candidate?.voice?.profile_id !== profile.id ||
      run.candidate?.voice?.instruction !== profile.instruction ||
      run.candidate?.voice?.locale !== profile.locale ||
      run.candidate?.voice?.seed !== profile.seed) {
    errors.push(`${prefix}candidate, runtime, or voice profile differs`);
  }
  if (!passage ||
      run.input?.passage_id !== passage.item.id ||
      run.input?.source_id !== passage.source.id ||
      run.input?.locale !== passage.source.locale ||
      run.input?.character_count !== passage.item.character_count ||
      run.input?.text_sha256 !== passage.item.spoken_sha256 ||
      run.input?.license !== passage.source.license) {
    errors.push(`${prefix}input differs from the pinned passage`);
  }
  const expectedGeneration = {
    profile_id: profile.id,
    passage_id: profile.passage_id,
    locale: profile.locale,
    language: profile.language,
    voice: profile.voice,
    instruction: profile.instruction,
    seed: profile.seed,
    ...manifest.calibration.generation,
    sample_rate_hz: manifest.calibration.sample_rate_hz,
  };
  if (JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(expectedGeneration) ||
      run.procedure?.stream !== false ||
      !run.procedure?.model_verification?.includes('SHA-256')) {
    errors.push(`${prefix}execution contract differs`);
  }

  const audio = run.result?.audio;
  if (run.result?.status !== 'measured' ||
      audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !== 24_000 ||
      audio?.channel_count !== 1 ||
      audio?.sample_count !== 1_317_120 ||
      audio?.byte_count !== 5_268_480 ||
      audio?.duration_ms !== 54_880 ||
      audio?.sha256 !==
        '87d10c77bb42da5a20d39b3075045c41b3f24fad076cfef7b9cee07d36cecc47' ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.rms > 0 && audio.rms < 1) ||
      run.result?.timing?.token_count !== 686 ||
      !(run.result?.timing?.complete_synthesis_ms > 0) ||
      run.result?.termination?.reached_max_tokens !== false ||
      run.result?.termination?.configured_max_tokens !== 1_200 ||
      run.result?.resources?.model_snapshot_bytes !==
        manifest.artifact.snapshot_bytes) {
    errors.push(`${prefix}audio, timing, resources, or termination differ`);
  }

  const checks = run.result?.asr_content_checks;
  if (checks?.status !== 'complete' ||
      checks?.normalized_audio?.sample_format !== 'f32le' ||
      checks?.normalized_audio?.sample_rate_hz !== 16_000 ||
      checks?.normalized_audio?.sample_count !== 878_080 ||
      checks?.normalized_audio?.byte_count !== 3_512_320 ||
      checks?.normalized_audio?.duration_ms !== audio?.duration_ms ||
      checks?.normalized_audio?.sha256 !==
        'cef63fd06cb53c7e5b229cfba5533259597266a8dfa50a185361d1bb608a3948') {
    errors.push(`${prefix}normalized PCM evidence differs`);
  }
  const expectedBackends = new Map([
    ['whisper-large-v3-turbo-coreml-whispercpp', [
      '6c1567f93b2f33b9bcbe31933c8ac8e7b2fab200ee8dea64724532279e76a1c3',
      2, 103, 0, 665,
    ]],
    ['parakeet-tdt-0.6b-v3-coreml', [
      '7afee190f296501b51b2067cf5fabc9c991fcba448fd65160f96fa93927094bf',
      3, 102, 2, 663,
    ]],
    ['qwen3-asr-0.6b-mlx-direct', [
      '8651fd4b155379aa28dc5c4310b84d3dff08d2cefe098b51981f6151112f7e3d',
      3, 104, 8, 669,
    ]],
    ['voxtral-realtime-4b-mlx-direct-2400ms', [
      '6c1567f93b2f33b9bcbe31933c8ac8e7b2fab200ee8dea64724532279e76a1c3',
      2, 103, 0, 665,
    ]],
    ['apple-speechtranscriber', [
      '0bd72edd8c606364543affc3f5470454714d2dc08efde51836f67e1860180a51',
      3, 103, 2, 667,
    ]],
  ]);
  const measuredBackends = checks?.backends ?? [];
  if (measuredBackends.length !== expectedBackends.size ||
      new Set(measuredBackends.map((item) => item.backend?.id)).size !==
        measuredBackends.length) {
    errors.push(`${prefix}must contain five unique ASR backends`);
  }
  for (const check of measuredBackends) {
    const expected = expectedBackends.get(check.backend?.id);
    const quality = check.quality;
    const digest = createHash('sha256')
      .update(check.transcript?.text ?? '')
      .digest('hex');
    if (!expected ||
        check.transcript?.sha256 !== digest ||
        check.transcript?.sha256 !== expected[0] ||
        quality?.reference_word_count !== 104 ||
        quality?.word_edits !== expected[1] ||
        quality?.hypothesis_word_count !== expected[2] ||
        Math.abs(quality?.wer - quality?.word_edits / 104) > 1e-15 ||
        quality?.reference_character_count !== 665 ||
        quality?.character_edits !== expected[3] ||
        quality?.hypothesis_character_count !== expected[4] ||
        Math.abs(quality?.cer - quality?.character_edits / 665) > 1e-15 ||
        !(check.timing?.complete_inference_ms > 0) ||
        Math.abs(check.timing.real_time_factor -
          check.timing.complete_inference_ms / audio.duration_ms) > 1e-12) {
      errors.push(`${prefix}${check.backend?.id}: transcript or metrics differ`);
    }
  }
  if (checks?.comparison?.completed_backend_count !== 5 ||
      checks?.comparison?.expected_backend_count !== 5 ||
      checks?.comparison?.critical_findings?.shared_content_omission !== false ||
      checks?.comparison?.critical_findings?.shared_truncation !== false ||
      checks?.comparison?.critical_findings
        ?.critical_facts_recovered_by_all_backends !== true ||
      checks?.comparison?.critical_findings?.probable_tts_content_error !== false ||
      run.conclusion?.calibration_disposition !==
        'passed-native-German-content-gate' ||
      run.conclusion?.content_type !== 'de-native' ||
      run.conclusion?.listening_required !== true) {
    errors.push(`${prefix}critical finding or disposition differs`);
  }
  return errors;
}

function validateQwenNativeCalibrationAsset(
  asset,
  audio,
  reference,
  attribution,
  run,
) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign native-German asset: ';
  const audioDigest = createHash('sha256').update(audio).digest('hex');
  const referenceDigest = createHash('sha256').update(reference).digest('hex');
  if (asset.schema_version !== '1.0.0' ||
      asset.asset_id !==
        'synthetic-de-native.qwen3-tts-1.7b-voicedesign-warm.opus-64k-1' ||
      asset.purpose !== 'reproducible-passed-content-type-calibration' ||
      asset.license !== 'CC-BY-SA-4.0' ||
      asset.reference?.bytes !== reference.length ||
      asset.reference?.characters !== run.input.character_count ||
      asset.reference?.sha256 !== referenceDigest ||
      asset.reference?.spoken_text_sha256 !== run.input.text_sha256 ||
      asset.reference?.source_id !== run.input.source_id ||
      asset.reference?.passage_id !== run.input.passage_id) {
    errors.push(`${prefix}identity, license, or reference differs`);
  }
  if (asset.generation?.run_id !== run.run_id ||
      asset.generation?.model_revision !== run.candidate.model.revision ||
      asset.generation?.voice_profile !== run.candidate.voice.profile_id ||
      asset.generation?.voice_instruction !== run.candidate.voice.instruction ||
      asset.generation?.seed !== run.candidate.voice.seed ||
      asset.generation?.lossless_f32le?.sample_count !==
        run.result.audio.sample_count ||
      asset.generation?.lossless_f32le?.bytes !== run.result.audio.byte_count ||
      asset.generation?.lossless_f32le?.sha256 !== run.result.audio.sha256) {
    errors.push(`${prefix}generation provenance differs from the run`);
  }
  if (asset.archive?.path !== 'audio.opus' ||
      asset.archive?.codec !== 'Opus' ||
      asset.archive?.bytes !== audio.length ||
      asset.archive?.sha256 !== audioDigest ||
      asset.archive?.target_bit_rate_bps !== 64_000 ||
      asset.archive?.variable_bit_rate !== true ||
      asset.archive?.application !== 'audio' ||
      asset.archive?.frame_duration_ms !== 20 ||
      asset.archive?.deterministic_repeat_sha256_match !== true ||
      asset.calibration_finding?.disposition !==
        run.conclusion.calibration_disposition ||
      asset.calibration_finding?.receiver_count !== 5 ||
      asset.calibration_finding?.receiver_exact_normalized_character_count !== 2 ||
      asset.calibration_finding?.critical_facts_recovered_by_all_backends !== true ||
      !attribution.includes('CC BY-SA 4.0') ||
      !attribution.includes('native-German factual')) {
    errors.push(`${prefix}archive, finding, or attribution caveat differs`);
  }
  return errors;
}

function validateQwenDialogueCalibrationRun(run, manifest, acceptedSelection) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign German-dialogue result: ';
  const profile = manifest.calibration.profiles.find(
    (item) => item.id === 'qwen-de-warm-dialogue',
  );
  const passage = acceptedSelection.sources
    .flatMap((source) => source.passages.map((item) => ({ source, item })))
    .find(({ item }) => item.id === profile.passage_id);
  if (run.schema_version !== '1.0.0' ||
      run.run_id !==
        'phase5.qwen3-tts-1.7b-voicedesign.qwen-de-warm-dialogue.1' ||
      run.source_revision !== '8a1561681999eda4d51b9de126b2ec41393a7dac' ||
      !acceptedSelectionRevision(run.selection_revision, acceptedSelection) ||
      run.purpose !== 'calibration') {
    errors.push(`${prefix}identity or revisions differ`);
  }
  if (run.candidate?.id !== 'qwen3-tts-1.7b-voicedesign-mlx-audio' ||
      run.candidate?.model?.revision !== manifest.artifact.revision ||
      run.candidate?.model?.snapshot_bytes !== manifest.artifact.snapshot_bytes ||
      run.candidate?.runtime?.revision !== manifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !== manifest.reference_runtime.version ||
      run.candidate?.voice?.profile_id !== profile.id ||
      run.candidate?.voice?.instruction !== profile.instruction ||
      run.candidate?.voice?.locale !== profile.locale ||
      run.candidate?.voice?.seed !== profile.seed) {
    errors.push(`${prefix}candidate, runtime, or voice profile differs`);
  }
  if (!passage ||
      run.input?.passage_id !== passage.item.id ||
      run.input?.source_id !== passage.source.id ||
      run.input?.locale !== passage.source.locale ||
      run.input?.character_count !== passage.item.character_count ||
      run.input?.text_sha256 !== passage.item.spoken_sha256 ||
      run.input?.license !== passage.source.license) {
    errors.push(`${prefix}input differs from the pinned passage`);
  }
  const expectedGeneration = {
    profile_id: profile.id,
    passage_id: profile.passage_id,
    locale: profile.locale,
    language: profile.language,
    voice: profile.voice,
    instruction: profile.instruction,
    seed: profile.seed,
    ...manifest.calibration.generation,
    sample_rate_hz: manifest.calibration.sample_rate_hz,
  };
  if (JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(expectedGeneration) ||
      run.procedure?.stream !== false ||
      !run.procedure?.model_verification?.includes('SHA-256')) {
    errors.push(`${prefix}execution contract differs`);
  }

  const audio = run.result?.audio;
  if (run.result?.status !== 'measured' ||
      audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !== 24_000 ||
      audio?.channel_count !== 1 ||
      audio?.sample_count !== 1_365_120 ||
      audio?.byte_count !== 5_460_480 ||
      audio?.duration_ms !== 56_880 ||
      audio?.sha256 !==
        '6519ca5543caf434b48e90970eed0c3e97aa32a14efae76c069b124a3bf5608e' ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.rms > 0 && audio.rms < 1) ||
      run.result?.timing?.token_count !== 711 ||
      !(run.result?.timing?.complete_synthesis_ms > 0) ||
      run.result?.termination?.reached_max_tokens !== false ||
      run.result?.termination?.configured_max_tokens !== 1_200 ||
      run.result?.resources?.model_snapshot_bytes !==
        manifest.artifact.snapshot_bytes) {
    errors.push(`${prefix}audio, timing, resources, or termination differ`);
  }

  const checks = run.result?.asr_content_checks;
  if (checks?.status !== 'complete' ||
      checks?.normalized_audio?.sample_format !== 'f32le' ||
      checks?.normalized_audio?.sample_rate_hz !== 16_000 ||
      checks?.normalized_audio?.sample_count !== 910_080 ||
      checks?.normalized_audio?.byte_count !== 3_640_320 ||
      checks?.normalized_audio?.duration_ms !== audio?.duration_ms ||
      checks?.normalized_audio?.sha256 !==
        '122eba76f3c700796cda06810d14e0921a93abeb6b6844e8c2b9c19c5cd33c82') {
    errors.push(`${prefix}normalized PCM evidence differs`);
  }
  const expectedBackends = new Map([
    ['whisper-large-v3-turbo-coreml-whispercpp', [
      '12235e1edd638456ee057835bb3bc214d02c271a504441f6989598b2f67db8c7',
      1, 123, 8, 649,
    ]],
    ['parakeet-tdt-0.6b-v3-coreml', [
      '39db0f8da7fd960831952f98a800d80b023eed8abde85c872efb1006d3447f45',
      7, 120, 31, 630,
    ]],
    ['qwen3-asr-0.6b-mlx-direct', [
      '2f3a515c67d405218e7088853495542911f1e8abfea1ee65b1c172b5f8e2cd5b',
      1, 123, 1, 655,
    ]],
    ['voxtral-realtime-4b-mlx-direct-2400ms', [
      'bf2e4df8074c40fffc99c845fea624b9348bdc9dad64f393ea2a876c2f048132',
      2, 123, 17, 641,
    ]],
    ['apple-speechtranscriber', [
      '71ea803e7c86aed9bf87cc2874231b1773a659a60fe7d3757130644480414b78',
      1, 123, 8, 649,
    ]],
  ]);
  const measuredBackends = checks?.backends ?? [];
  if (measuredBackends.length !== expectedBackends.size ||
      new Set(measuredBackends.map((item) => item.backend?.id)).size !==
        measuredBackends.length) {
    errors.push(`${prefix}must contain five unique ASR backends`);
  }
  for (const check of measuredBackends) {
    const expected = expectedBackends.get(check.backend?.id);
    const quality = check.quality;
    const digest = createHash('sha256')
      .update(check.transcript?.text ?? '')
      .digest('hex');
    if (!expected ||
        check.transcript?.sha256 !== digest ||
        check.transcript?.sha256 !== expected[0] ||
        quality?.reference_word_count !== 123 ||
        quality?.word_edits !== expected[1] ||
        quality?.hypothesis_word_count !== expected[2] ||
        Math.abs(quality?.wer - quality?.word_edits / 123) > 1e-15 ||
        quality?.reference_character_count !== 655 ||
        quality?.character_edits !== expected[3] ||
        quality?.hypothesis_character_count !== expected[4] ||
        Math.abs(quality?.cer - quality?.character_edits / 655) > 1e-15 ||
        !(check.timing?.complete_inference_ms > 0) ||
        Math.abs(check.timing.real_time_factor -
          check.timing.complete_inference_ms / audio.duration_ms) > 1e-12) {
      errors.push(`${prefix}${check.backend?.id}: transcript or metrics differ`);
    }
  }
  if (checks?.comparison?.completed_backend_count !== 5 ||
      checks?.comparison?.expected_backend_count !== 5 ||
      checks?.comparison?.critical_findings?.shared_content_omission !== false ||
      checks?.comparison?.critical_findings?.shared_truncation !== false ||
      checks?.comparison?.critical_findings
        ?.dialogue_content_recovered_by_four_backends !== true ||
      checks?.comparison?.critical_findings
        ?.parakeet_receiver_specific_errors !== true ||
      checks?.comparison?.critical_findings?.probable_tts_content_error !== false ||
      run.conclusion?.calibration_disposition !==
        'passed-dialogue-content-gate' ||
      run.conclusion?.content_type !== 'de-dialogue' ||
      run.conclusion?.three_content_cells_complete !== true ||
      run.conclusion?.listening_required !== true) {
    errors.push(`${prefix}critical finding or disposition differs`);
  }
  return errors;
}

function validateQwenDialogueCalibrationAsset(
  asset,
  audio,
  reference,
  attribution,
  run,
) {
  const errors = [];
  const prefix = 'Qwen VoiceDesign German-dialogue asset: ';
  const audioDigest = createHash('sha256').update(audio).digest('hex');
  const referenceDigest = createHash('sha256').update(reference).digest('hex');
  if (asset.schema_version !== '1.0.0' ||
      asset.asset_id !==
        'synthetic-de-dialogue.qwen3-tts-1.7b-voicedesign-warm.opus-64k-1' ||
      asset.purpose !== 'reproducible-passed-content-type-calibration' ||
      asset.license !== 'CC-BY-SA-4.0' ||
      asset.reference?.bytes !== reference.length ||
      asset.reference?.characters !== run.input.character_count ||
      asset.reference?.sha256 !== referenceDigest ||
      asset.reference?.spoken_text_sha256 !== run.input.text_sha256 ||
      asset.reference?.source_id !== run.input.source_id ||
      asset.reference?.passage_id !== run.input.passage_id) {
    errors.push(`${prefix}identity, license, or reference differs`);
  }
  if (asset.generation?.run_id !== run.run_id ||
      asset.generation?.model_revision !== run.candidate.model.revision ||
      asset.generation?.voice_profile !== run.candidate.voice.profile_id ||
      asset.generation?.voice_instruction !== run.candidate.voice.instruction ||
      asset.generation?.seed !== run.candidate.voice.seed ||
      asset.generation?.lossless_f32le?.sample_count !==
        run.result.audio.sample_count ||
      asset.generation?.lossless_f32le?.bytes !== run.result.audio.byte_count ||
      asset.generation?.lossless_f32le?.sha256 !== run.result.audio.sha256) {
    errors.push(`${prefix}generation provenance differs from the run`);
  }
  if (asset.archive?.path !== 'audio.opus' ||
      asset.archive?.codec !== 'Opus' ||
      asset.archive?.bytes !== audio.length ||
      asset.archive?.sha256 !== audioDigest ||
      asset.archive?.target_bit_rate_bps !== 64_000 ||
      asset.archive?.variable_bit_rate !== true ||
      asset.archive?.application !== 'audio' ||
      asset.archive?.frame_duration_ms !== 20 ||
      asset.archive?.deterministic_repeat_sha256_match !== true ||
      asset.calibration_finding?.disposition !==
        run.conclusion.calibration_disposition ||
      asset.calibration_finding?.receiver_count !== 5 ||
      asset.calibration_finding?.receiver_count_at_or_below_two_word_edits !== 4 ||
      asset.calibration_finding?.parakeet_receiver_specific_errors !== true ||
      !attribution.includes('CC BY-SA 4.0') ||
      !attribution.includes('dialogue lexical gate')) {
    errors.push(`${prefix}archive, finding, or attribution caveat differs`);
  }
  return errors;
}
