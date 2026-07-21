#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import {
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultManifest = join(
  repoRoot,
  'benchmarks/fixtures/target-domain-corpus.json',
);

try {
  const options = parseArguments(process.argv.slice(2));
  if (options.selfTest) {
    await runSelfTest();
  } else {
    const manifest = JSON.parse(await readFile(options.manifest, 'utf8'));
    const result = await materializeCorpus({
      manifest,
      inputDirectory: options.inputDirectory,
      outputDirectory: options.outputDirectory,
      ffmpeg: options.ffmpeg,
    });
    process.stdout.write(
      `target-domain corpus: verified ${result.artifactCount} normalized ` +
      `artifact(s), ${result.durationMs} ms total\n`,
    );
  }
} catch (error) {
  process.stderr.write(`target-domain corpus: ${error.message}\n`);
  process.exitCode = 1;
}

function parseArguments(arguments_) {
  if (arguments_.length === 1 && arguments_[0] === '--self-test') {
    return { selfTest: true };
  }
  const options = {
    manifest: defaultManifest,
    ffmpeg: process.env.FFMPEG ?? 'ffmpeg',
  };
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === '--manifest') {
      options.manifest = resolve(arguments_[index + 1]);
      index += 1;
    } else if (argument === '--input-dir') {
      options.inputDirectory = resolve(arguments_[index + 1]);
      index += 1;
    } else if (argument === '--output-dir') {
      options.outputDirectory = resolve(arguments_[index + 1]);
      index += 1;
    } else if (argument === '--ffmpeg') {
      options.ffmpeg = arguments_[index + 1];
      index += 1;
    } else {
      throw new Error(`unknown argument: ${argument}`);
    }
  }
  if (!options.inputDirectory || !options.outputDirectory) {
    throw new Error(
      'usage: node scripts/materialize-target-domain-corpus.mjs ' +
      '--input-dir /absolute/path/to/imported-originals ' +
      '--output-dir /absolute/path/to/normalized-corpus ' +
      '[--manifest benchmarks/fixtures/target-domain-corpus.json] ' +
      '[--ffmpeg /absolute/path/to/ffmpeg]',
    );
  }
  return options;
}

async function materializeCorpus({
  manifest,
  inputDirectory,
  outputDirectory,
  ffmpeg,
}) {
  if (manifest.schema_version !== '1.0.0' || manifest.purpose !== 'held-out') {
    throw new Error('manifest must be a schema 1.0.0 held-out corpus');
  }
  await mkdir(outputDirectory, { recursive: true });
  let artifactCount = 0;
  let durationMs = 0;
  for (const cell of manifest.cells ?? []) {
    for (const sourceGroup of cell.source_groups ?? []) {
      await verifyRightsReview(sourceGroup);
      const sourceName = safeArtifactName(
        sourceGroup.source_audio?.artifact_name,
      );
      const sourcePath = join(inputDirectory, sourceName);
      await verifyArtifact(sourcePath, sourceGroup.source_audio, 'source');

      const outputName = safeArtifactName(
        sourceGroup.normalized_audio?.artifact_name,
      );
      const outputPath = join(outputDirectory, outputName);
      const temporaryPath = `${outputPath}.partial-${process.pid}`;
      if (await pathExists(outputPath)) {
        await verifyArtifact(
          outputPath,
          sourceGroup.normalized_audio,
          'normalized',
        );
      } else {
        await rm(temporaryPath, { force: true });
        try {
          await runFfmpeg({ ffmpeg, sourcePath, temporaryPath, sourceGroup });
          await verifyArtifact(
            temporaryPath,
            sourceGroup.normalized_audio,
            'normalized',
          );
          await rename(temporaryPath, outputPath);
        } finally {
          await rm(temporaryPath, { force: true });
        }
      }
      await writeProvenance(outputPath, sourceGroup, manifest.normalization);
      artifactCount += 1;
      durationMs += sourceGroup.normalized_audio.duration_ms;
    }
  }
  if (artifactCount === 0) {
    throw new Error('manifest contains no source groups');
  }
  return { artifactCount, durationMs };
}

async function verifyRightsReview(sourceGroup) {
  const reviewPath = resolve(repoRoot, sourceGroup.rights_review_path ?? '');
  if (!reviewPath.startsWith(`${repoRoot}/`)) {
    throw new Error(`${sourceGroup.id}: rights review escapes the repository`);
  }
  const review = JSON.parse(await readFile(reviewPath, 'utf8'));
  if (review.review_id !== sourceGroup.rights_review_id ||
      review.source_group_id !== sourceGroup.id ||
      review.source_candidate_id !== sourceGroup.source_candidate_id ||
      review.disposition !== 'accepted' ||
      review.acquisition?.status !== 'allowed') {
    throw new Error(`${sourceGroup.id}: rights review is not accepted or exact`);
  }
  const source = sourceGroup.source_audio;
  if (review.acquisition.artifact_name !== source.artifact_name ||
      review.acquisition.source_url !== source.source_url ||
      review.acquisition.expected_source_sha256 !== source.sha256) {
    throw new Error(`${sourceGroup.id}: source differs from its rights review`);
  }
}

async function runFfmpeg({ ffmpeg, sourcePath, temporaryPath, sourceGroup }) {
  const selection = sourceGroup.selection;
  const durationMs = selection.end_ms - selection.start_ms;
  if (durationMs !== selection.duration_ms ||
      durationMs !== sourceGroup.normalized_audio.duration_ms) {
    throw new Error(`${sourceGroup.id}: selection duration is inconsistent`);
  }
  const arguments_ = [
    '-v', 'error',
    '-nostdin',
    '-i', sourcePath,
    '-ss', seconds(selection.start_ms),
    '-t', seconds(durationMs),
    '-map', '0:a:0',
    '-ac', '1',
    '-ar', '16000',
    '-c:a', 'pcm_f32le',
    '-f', 'f32le',
    temporaryPath,
  ];
  await new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(ffmpeg, arguments_, { stdio: ['ignore', 'ignore', 'pipe'] });
    let standardError = '';
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => {
      standardError += chunk;
    });
    child.on('error', rejectPromise);
    child.on('close', (code) => {
      if (code === 0) {
        resolvePromise();
      } else {
        rejectPromise(new Error(
          `${sourceGroup.id}: ffmpeg exited ${code}: ${standardError.trim()}`,
        ));
      }
    });
  });
}

async function writeProvenance(outputPath, sourceGroup, normalization) {
  const provenance = {
    schema_version: '1.0.0',
    source_group_id: sourceGroup.id,
    rights_review_id: sourceGroup.rights_review_id,
    source_audio: sourceGroup.source_audio,
    selection: sourceGroup.selection,
    normalization,
    normalized_audio: sourceGroup.normalized_audio,
    attribution: sourceGroup.attribution,
  };
  const bytes = `${JSON.stringify(provenance, null, 2)}\n`;
  const path = `${outputPath}.provenance.json`;
  if (await pathExists(path)) {
    if (await readFile(path, 'utf8') !== bytes) {
      throw new Error(`${path}: existing provenance differs from the manifest`);
    }
  } else {
    await writeFile(path, bytes, { encoding: 'utf8', flag: 'wx' });
  }
}

async function verifyArtifact(path, expected, label) {
  const metadata = await stat(path);
  if (metadata.size !== expected.bytes) {
    throw new Error(
      `${path}: ${label} byte count ${metadata.size} != ${expected.bytes}`,
    );
  }
  const digest = await sha256File(path);
  if (digest !== expected.sha256) {
    throw new Error(
      `${path}: ${label} SHA-256 ${digest} != ${expected.sha256}`,
    );
  }
}

function safeArtifactName(value) {
  if (!(value?.length > 0) || basename(value) !== value ||
      value === '.' || value === '..') {
    throw new Error(`unsafe artifact name: ${value ?? '<missing>'}`);
  }
  return value;
}

function seconds(milliseconds) {
  return (milliseconds / 1000).toFixed(3);
}

async function sha256File(path) {
  return await new Promise((resolvePromise, rejectPromise) => {
    const hash = createHash('sha256');
    const stream = createReadStream(path);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', rejectPromise);
    stream.on('end', () => resolvePromise(hash.digest('hex')));
  });
}

async function pathExists(path) {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (error.code === 'ENOENT') return false;
    throw error;
  }
}

async function runSelfTest() {
  const directory = await mkdtemp(join(tmpdir(), 'cuttledoc-target-corpus-'));
  try {
    const path = join(directory, 'artifact.bin');
    const bytes = Buffer.from('target-domain corpus self-test\n');
    await writeFile(path, bytes);
    const expected = {
      bytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex'),
    };
    await verifyArtifact(path, expected, 'self-test');
    let rejected = false;
    try {
      await verifyArtifact(path, { ...expected, sha256: '0'.repeat(64) }, 'bad');
    } catch {
      rejected = true;
    }
    if (!rejected) {
      throw new Error('self-test failed to reject digest drift');
    }
    if (seconds(124_000) !== '124.000' ||
        safeArtifactName('fixture.f32le') !== 'fixture.f32le') {
      throw new Error('self-test helper behavior changed');
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
  process.stdout.write('target-domain corpus materializer: self-test passed\n');
}
