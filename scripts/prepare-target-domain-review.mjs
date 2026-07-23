#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
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
    'normalized-dir': {
      type: 'string',
      default: join(repoRoot, 'artifacts/target-domain/normalized'),
    },
    'publisher-dir': {
      type: 'string',
      default: join(repoRoot, 'artifacts/target-domain/publisher-drafts'),
    },
    'output-dir': {
      type: 'string',
      default: join(repoRoot, 'artifacts/target-domain/review'),
    },
    split: { type: 'string', default: 'validation' },
    'source-group': { type: 'string' },
    'allow-test': { type: 'boolean', default: false },
    pdftotext: {
      type: 'string',
      default: process.env.PDFTOTEXT ?? 'pdftotext',
    },
  },
});

const command = positionals[0];
if (!['prepare', 'self-test'].includes(command)) {
  throw new Error(
    'usage: node scripts/prepare-target-domain-review.mjs ' +
      '<prepare|self-test> [--split validation|test|all] ' +
      '[--source-group ID] [--allow-test] [--normalized-dir PATH] ' +
      '[--publisher-dir PATH] [--output-dir PATH] [--pdftotext PATH]',
  );
}

try {
  if (command === 'self-test') {
    await runSelfTest();
  } else {
    const options = {
      manifestPath: resolve(values.manifest),
      planPath: resolve(values.plan),
      normalizedDirectory: resolve(values['normalized-dir']),
      publisherDirectory: resolve(values['publisher-dir']),
      outputDirectory: resolve(values['output-dir']),
      split: values.split,
      sourceGroupId: values['source-group'],
      allowTest: values['allow-test'],
      pdftotext: values.pdftotext,
    };
    const result = await prepareReviewBundles(options);
    process.stdout.write(
      `target-domain review: prepared ${result.bundleCount} ` +
        `${result.split} bundle(s); ${result.durationMs} ms exact audio\n`,
    );
  }
} catch (error) {
  process.stderr.write(`target-domain review: ${error.message}\n`);
  process.exitCode = 1;
}

async function prepareReviewBundles({
  manifestPath,
  planPath,
  normalizedDirectory,
  publisherDirectory,
  outputDirectory,
  split,
  sourceGroupId,
  allowTest,
  pdftotext,
}) {
  if (!['validation', 'test', 'all'].includes(split)) {
    throw new Error('--split must be validation, test, or all');
  }
  if ((split === 'test' || split === 'all') && !allowTest) {
    throw new Error(
      'test sources remain closed; pass --allow-test only after the ' +
        'validation configuration is frozen',
    );
  }

  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const plan = JSON.parse(await readFile(planPath, 'utf8'));
  validatePlanAndManifest(plan, manifest);
  const sourceGroups = manifest.cells.flatMap((cell) =>
    cell.source_groups.map((sourceGroup) => ({ cell, sourceGroup })),
  );
  const selected = sourceGroups.filter(({ sourceGroup }) =>
    (split === 'all' || sourceGroup.split === split) &&
    (!sourceGroupId || sourceGroup.id === sourceGroupId),
  );
  if (selected.length === 0) {
    throw new Error(
      sourceGroupId
        ? `source group ${sourceGroupId} is absent from the ${split} split`
        : `no source groups are assigned to the ${split} split`,
    );
  }
  if (!allowTest && selected.some(({ sourceGroup }) =>
    sourceGroup.split === 'test')) {
    throw new Error('test source selected without --allow-test');
  }

  await mkdir(outputDirectory, { recursive: true });
  let durationMs = 0;
  for (const { cell, sourceGroup } of selected) {
    await prepareBundle({
      cell,
      sourceGroup,
      corpus: manifest,
      plan,
      normalizedDirectory,
      publisherDirectory,
      outputDirectory,
      pdftotext,
    });
    durationMs += sourceGroup.selection.duration_ms;
  }
  return { bundleCount: selected.length, durationMs, split };
}

function validatePlanAndManifest(plan, manifest) {
  if (plan.schema_version !== '1.0.0' ||
      plan.revision !== 'target-domain-held-out-3' ||
      plan.purpose !== 'held-out') {
    throw new Error('expected target-domain-held-out-3 plan');
  }
  if (manifest.schema_version !== '1.0.0' ||
      manifest.plan_revision !== plan.revision ||
      manifest.purpose !== 'held-out') {
    throw new Error('target-domain corpus does not match the held-out plan');
  }
  const expectedBackends = [
    'apple-speechtranscriber',
    'whisper-large-v3-turbo-coreml-whispercpp',
    'qwen3-asr-0.6b-mlx-direct',
    'parakeet-tdt-0.6b-v3-coreml',
    'voxtral-realtime-4b-mlx-direct-2400ms',
  ];
  if (JSON.stringify(plan.candidate_backends) !==
      JSON.stringify(expectedBackends)) {
    throw new Error('target-domain plan does not contain the frozen five-backend order');
  }
}

async function prepareBundle({
  cell,
  sourceGroup,
  corpus,
  plan,
  normalizedDirectory,
  publisherDirectory,
  outputDirectory,
  pdftotext,
}) {
  const normalizedName = safeArtifactName(
    sourceGroup.normalized_audio.artifact_name,
  );
  const normalizedPath = join(normalizedDirectory, normalizedName);
  const normalizedBytes = await verifiedBytes(
    normalizedPath,
    sourceGroup.normalized_audio,
    `${sourceGroup.id} normalized audio`,
  );
  if (normalizedBytes.length % 4 !== 0) {
    throw new Error(`${sourceGroup.id}: normalized float PCM is truncated`);
  }

  const publisherName = safeArtifactName(
    sourceGroup.published_transcript.artifact_name,
  );
  const publisherPath = join(publisherDirectory, publisherName);
  await verifiedBytes(
    publisherPath,
    sourceGroup.published_transcript,
    `${sourceGroup.id} publisher transcript`,
  );
  const publisherText = extractPublisherText(pdftotext, publisherPath);
  const publisherTextBytes = Buffer.from(publisherText, 'utf8');
  const publisherTextSha256 = digest(publisherTextBytes);

  const bundleDirectory = join(outputDirectory, sourceGroup.id);
  await mkdir(bundleDirectory, { recursive: true });
  const playbackName = `${sourceGroup.id}.exact-audio.f32.wav`;
  const publisherTextName = `${sourceGroup.id}.publisher-draft.txt`;
  const reviewManifestName = 'review-manifest.json';
  const instructionsName = 'REVIEW.md';
  const playbackBytes = float32Wave(
    normalizedBytes,
    corpus.normalization.sample_rate_hz,
    corpus.normalization.channels,
  );

  await writeExact(join(bundleDirectory, playbackName), playbackBytes);
  await writeExact(
    join(bundleDirectory, publisherTextName),
    publisherTextBytes,
  );

  const reviewManifest = {
    schema_version: '1.0.0',
    bundle_id: `target-domain-review--${sourceGroup.id}`,
    corpus_revision: corpus.revision,
    plan_revision: plan.revision,
    source_group: {
      id: sourceGroup.id,
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
    },
    exact_audio: {
      normalized_artifact_name: normalizedName,
      normalized_sha256: sourceGroup.normalized_audio.sha256,
      normalized_bytes: sourceGroup.normalized_audio.bytes,
      sample_rate_hz: corpus.normalization.sample_rate_hz,
      channels: corpus.normalization.channels,
      sample_format: corpus.normalization.sample_format,
      playback_artifact_name: playbackName,
      playback_sha256: digest(playbackBytes),
      playback_bytes: playbackBytes.length,
      playback_encoding: 'IEEE float32 WAV with an exact byte-for-byte copy of the normalized f32le PCM in its data chunk',
    },
    publisher_draft: {
      source_artifact_name: publisherName,
      source_sha256: sourceGroup.published_transcript.sha256,
      source_bytes: sourceGroup.published_transcript.bytes,
      extracted_artifact_name: publisherTextName,
      extracted_sha256: publisherTextSha256,
      extracted_bytes: publisherTextBytes.length,
      status: 'publisher-draft-not-gold',
      use: 'alignment-and-transcription-aid-only',
    },
    asr_drafts: {
      status: 'pending',
      required_backends: plan.candidate_backends,
      rule: 'Every backend must receive the exact normalized SHA-256 above. ASR output is draft material and must never overwrite human gold.',
    },
    gold_review: {
      status: sourceGroup.gold_transcript.status,
      target_artifact_name: sourceGroup.gold_transcript.artifact_name,
      transcript_schema:
        'benchmarks/schema/target-domain-transcript.schema.json',
      independent_reviewer_required: true,
      complete_audio_listening_required: true,
      preserved_fields: plan.gold_policy.preserve,
      critical_content: plan.gold_policy.critical_content,
      promotion:
        'Automation may validate reviewer evidence but may not promote its own transcript to human-verified.',
    },
    split_policy: {
      current_split: sourceGroup.split,
      test_opened: sourceGroup.split === 'test',
      rule: plan.split_policy.test,
    },
    claim_limit:
      'This bundle prepares independent review. It is not human gold, does not support WER, and does not select an ASR or transcript-enhancement model.',
  };
  await writeExact(
    join(bundleDirectory, reviewManifestName),
    Buffer.from(`${JSON.stringify(reviewManifest, null, 2)}\n`),
  );
  await writeExact(
    join(bundleDirectory, instructionsName),
    Buffer.from(reviewInstructions(reviewManifest)),
  );
}

function extractPublisherText(executable, path) {
  const result = spawnSync(executable, ['-layout', path, '-'], {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `pdftotext failed for ${path}: ` +
        `${result.stderr?.trim() || result.error?.message || result.status}`,
    );
  }
  const text = result.stdout.replace(/\r\n?/g, '\n');
  if (text.trim().length === 0) {
    throw new Error(`${path}: pdftotext returned no text`);
  }
  return text.endsWith('\n') ? text : `${text}\n`;
}

function reviewInstructions(bundle) {
  const source = bundle.source_group;
  const audio = bundle.exact_audio;
  return `# Gold review: ${source.title}

This is a review bundle, not a gold transcript.

## Exact boundary

- Source group: \`${source.id}\`
- Split: \`${source.split}\`
- Source range: ${formatTime(source.source_range.start_ms)}–${formatTime(source.source_range.end_ms)}
- Duration: ${source.source_range.duration_ms} ms
- Normalized PCM SHA-256: \`${audio.normalized_sha256}\`
- Playback file: \`${audio.playback_artifact_name}\`

The WAV data chunk is an exact byte-for-byte copy of the digest-pinned mono
16 kHz float32 PCM. Listen to the complete file.

## Procedure

1. Use the publisher text only as a draft aid; it is lightly smoothed and covers
   more than the selected passage.
2. ASR drafts may suggest alignments but are not evidence of what was spoken.
3. Create \`${bundle.gold_review.target_artifact_name}\` using
   \`${bundle.gold_review.transcript_schema}\`.
4. Preserve verbatim wording, speaker turns, punctuation, capitalization,
   numbers, disfluencies, and uncertainty.
5. Label names and terms, numbers/dates/units, negation, and uncertain spans.
6. A reviewer other than the draft preparer must compare every turn against the
   complete exact-audio file and record their identity and UTC review time.
7. Do not open or use a test bundle to tune the validation configuration.

Automation may validate completed evidence; it cannot attest that a human
listened to the audio.
`;
}

function float32Wave(pcmBytes, sampleRate, channels) {
  if (!Buffer.isBuffer(pcmBytes) ||
      pcmBytes.length === 0 ||
      pcmBytes.length % (channels * 4) !== 0 ||
      !Number.isInteger(sampleRate) ||
      sampleRate <= 0 ||
      !Number.isInteger(channels) ||
      channels <= 0) {
    throw new Error('invalid float32 PCM for WAV wrapping');
  }
  if (pcmBytes.length > 0xffff_ffff - 36) {
    throw new Error('PCM is too large for a RIFF/WAVE container');
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0, 'ascii');
  header.writeUInt32LE(36 + pcmBytes.length, 4);
  header.write('WAVE', 8, 'ascii');
  header.write('fmt ', 12, 'ascii');
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(3, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * channels * 4, 28);
  header.writeUInt16LE(channels * 4, 32);
  header.writeUInt16LE(32, 34);
  header.write('data', 36, 'ascii');
  header.writeUInt32LE(pcmBytes.length, 40);
  return Buffer.concat([header, pcmBytes]);
}

async function verifiedBytes(path, expected, label) {
  const metadata = await stat(path);
  if (!metadata.isFile() || metadata.size !== expected.bytes) {
    throw new Error(
      `${label}: byte count ${metadata.size} != ${expected.bytes}`,
    );
  }
  const bytes = await readFile(path);
  const actualDigest = digest(bytes);
  if (actualDigest !== expected.sha256) {
    throw new Error(
      `${label}: SHA-256 ${actualDigest} != ${expected.sha256}`,
    );
  }
  return bytes;
}

async function writeExact(path, bytes) {
  try {
    const existing = await readFile(path);
    if (!existing.equals(bytes)) {
      throw new Error(`${path}: existing review artifact differs`);
    }
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    await writeFile(path, bytes, { flag: 'wx' });
  }
}

function safeArtifactName(value) {
  if (!(value?.length > 0) ||
      basename(value) !== value ||
      value === '.' ||
      value === '..') {
    throw new Error(`unsafe artifact name: ${value ?? '<missing>'}`);
  }
  return value;
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function formatTime(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds]
    .map((value) => String(value).padStart(2, '0'))
    .join(':');
}

async function runSelfTest() {
  const directory = await mkdtemp(join(tmpdir(), 'cuttledoc-review-'));
  try {
    const pcm = Buffer.alloc(16);
    pcm.writeFloatLE(0.25, 0);
    pcm.writeFloatLE(-0.25, 4);
    pcm.writeFloatLE(0.5, 8);
    pcm.writeFloatLE(-0.5, 12);
    const wave = float32Wave(pcm, 16_000, 1);
    if (wave.length !== pcm.length + 44 ||
        wave.toString('ascii', 0, 4) !== 'RIFF' ||
        wave.readUInt16LE(20) !== 3 ||
        wave.readUInt32LE(24) !== 16_000 ||
        wave.readUInt32LE(40) !== pcm.length ||
        !wave.subarray(44).equals(pcm)) {
      throw new Error('float WAV wrapper did not preserve exact PCM');
    }
    const path = join(directory, 'exact.bin');
    await writeExact(path, wave);
    await writeExact(path, wave);
    let rejected = false;
    try {
      await writeExact(path, Buffer.from('different'));
    } catch {
      rejected = true;
    }
    if (!rejected) {
      throw new Error('self-test accepted drift in an existing review artifact');
    }
    if (formatTime(7_700_000) !== '02:08:20') {
      throw new Error('time formatter is not deterministic');
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
  process.stdout.write('target-domain review preparation: self-test passed\n');
}
