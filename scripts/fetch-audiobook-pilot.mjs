#!/usr/bin/env node

import { createHash } from 'node:crypto';
import {
  mkdir,
  readFile,
  rename,
  stat,
  unlink,
  writeFile,
} from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultManifestPath = join(
  repoRoot,
  'benchmarks/fixtures/audiobook-pilot.json',
);

const options = parseArguments(process.argv.slice(2));
const manifestPath = resolve(options.manifest ?? defaultManifestPath);
const outputDirectory = resolve(options.outputDirectory);
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const ffmpeg = process.env.FFMPEG ?? 'ffmpeg';

await mkdir(outputDirectory, { recursive: true });
await verifyDatasetRevisions(manifest.fixtures);

for (const fixture of manifest.fixtures) {
  const row = await fetchDatasetRow(fixture);
  verifyRow(fixture, row);

  const sourcePath = join(
    outputDirectory,
    `${fixture.language}-${fixture.row_id}.${fixture.source.extension}`,
  );
  const normalizedPath = join(
    outputDirectory,
    `${fixture.language}-${fixture.row_id}.${fixture.normalized.extension}`,
  );

  if (!(await artifactMatches(sourcePath, fixture.source))) {
    const asset = findAudioAsset(row);
    const bytes = Buffer.from(await fetchArrayBuffer(asset.src));
    verifyBuffer(fixture.id, 'source download', bytes, fixture.source);
    await writeAtomic(sourcePath, bytes);
  }

  if (!(await artifactMatches(normalizedPath, fixture.normalized))) {
    const temporaryPath = `${normalizedPath}.part`;
    await unlink(temporaryPath).catch((error) => {
      if (error.code !== 'ENOENT') throw error;
    });
    const argumentsForFixture = manifest.normalization.ffmpeg_arguments.map(
      (argument) => {
        if (argument === '<source>') return sourcePath;
        if (argument === '<normalized>') return temporaryPath;
        return argument;
      },
    );
    await run(ffmpeg, argumentsForFixture);
    await verifyFile(fixture.id, 'normalized audio', temporaryPath, fixture.normalized);
    await rename(temporaryPath, normalizedPath);
  }

  await verifyFile(fixture.id, 'source audio', sourcePath, fixture.source);
  await verifyFile(
    fixture.id,
    'normalized audio',
    normalizedPath,
    fixture.normalized,
  );
  process.stdout.write(`verified ${fixture.id}\n`);
}

process.stdout.write(
  `audiobook pilot: ${manifest.fixtures.length} fixtures verified in ` +
  `${outputDirectory}\n`,
);

function parseArguments(arguments_) {
  let manifest;
  let outputDirectory;
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === '--manifest') {
      manifest = arguments_[index + 1];
      index += 1;
    } else if (argument === '--output-dir') {
      outputDirectory = arguments_[index + 1];
      index += 1;
    } else {
      throw new Error(`unknown argument: ${argument}`);
    }
  }
  if (!outputDirectory) {
    throw new Error(
      'usage: node scripts/fetch-audiobook-pilot.mjs ' +
      '--output-dir /absolute/output/path [--manifest /absolute/manifest.json]',
    );
  }
  return { manifest, outputDirectory };
}

async function verifyDatasetRevisions(fixtures) {
  const datasets = new Map();
  for (const fixture of fixtures) {
    const revisions = datasets.get(fixture.dataset) ?? new Set();
    revisions.add(fixture.dataset_revision);
    datasets.set(fixture.dataset, revisions);
  }

  for (const [dataset, revisions] of datasets) {
    if (revisions.size !== 1) {
      throw new Error(`${dataset}: manifest contains multiple revisions`);
    }
    const [revision] = revisions;
    const metadata = await fetchJson(
      `https://huggingface.co/api/datasets/${dataset}/revision/${revision}`,
    );
    if (metadata.sha !== revision) {
      throw new Error(
        `${dataset}: requested revision ${revision}, API returned ${metadata.sha}`,
      );
    }
  }
}

async function fetchDatasetRow(fixture) {
  const url = new URL('https://datasets-server.huggingface.co/rows');
  url.searchParams.set('dataset', fixture.dataset);
  url.searchParams.set('config', fixture.config);
  url.searchParams.set('split', fixture.split);
  url.searchParams.set('offset', String(fixture.row_index));
  url.searchParams.set('length', '1');
  const payload = await fetchJson(url);
  const result = payload.rows?.[0];
  if (!result || result.row_idx !== fixture.row_index || !result.row) {
    throw new Error(
      `${fixture.id}: dataset server did not return row ${fixture.row_index}`,
    );
  }
  return result.row;
}

function verifyRow(fixture, row) {
  const actual = {
    row_id: requiredString(row.id, 'id'),
    speaker_id: requiredString(row.speaker_id, 'speaker_id'),
    chapter_id: requiredString(row.chapter_id, 'chapter_id'),
    original_path: row.original_path ?? null,
    begin_time: row.begin_time ?? null,
    end_time: row.end_time ?? null,
    reference_text: row.transcript ?? row.text,
  };
  for (const [field, expected] of Object.entries(actual)) {
    if (expected !== fixture[field]) {
      throw new Error(
        `${fixture.id}: ${field} changed; expected ` +
        `${JSON.stringify(fixture[field])}, received ${JSON.stringify(expected)}`,
      );
    }
  }
}

function requiredString(value, field) {
  if (typeof value !== 'string' && typeof value !== 'number') {
    throw new Error(`dataset row has no ${field}`);
  }
  return String(value);
}

function findAudioAsset(row) {
  const candidates = Array.isArray(row.audio) ? row.audio : [row.audio];
  const asset = candidates.find(
    (candidate) => candidate && typeof candidate.src === 'string',
  );
  if (!asset) throw new Error('dataset row has no downloadable audio asset');
  return asset;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      'user-agent': 'cuttledoc-benchmark-acquisition/1.0',
    },
  });
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url, {
    headers: {
      'user-agent': 'cuttledoc-benchmark-acquisition/1.0',
    },
  });
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  }
  return response.arrayBuffer();
}

async function artifactMatches(path, expected) {
  try {
    await verifyFile('cached fixture', path, path, expected);
    return true;
  } catch (error) {
    if (error.code === 'ENOENT') return false;
    if (error.message.includes('does not match manifest')) return false;
    throw error;
  }
}

async function verifyFile(fixtureId, label, path, expected) {
  const metadata = await stat(path);
  if (metadata.size !== expected.bytes) {
    throw new Error(
      `${fixtureId}: ${label} size ${metadata.size} does not match manifest ` +
      `${expected.bytes}`,
    );
  }
  const bytes = await readFile(path);
  verifyBuffer(fixtureId, label, bytes, expected);
}

function verifyBuffer(fixtureId, label, bytes, expected) {
  if (bytes.length !== expected.bytes) {
    throw new Error(
      `${fixtureId}: ${label} size ${bytes.length} does not match manifest ` +
      `${expected.bytes}`,
    );
  }
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (digest !== expected.sha256) {
    throw new Error(
      `${fixtureId}: ${label} digest ${digest} does not match manifest ` +
      `${expected.sha256}`,
    );
  }
}

async function writeAtomic(path, bytes) {
  const temporaryPath = `${path}.part`;
  await writeFile(temporaryPath, bytes);
  await rename(temporaryPath, path);
}

async function run(command, arguments_) {
  await new Promise((resolvePromise, reject) => {
    const child = spawn(command, arguments_, { stdio: 'inherit' });
    child.once('error', reject);
    child.once('exit', (code, signal) => {
      if (code === 0) {
        resolvePromise();
      } else {
        reject(
          new Error(
            `${command} failed with ${signal ? `signal ${signal}` : `code ${code}`}`,
          ),
        );
      }
    });
  });
}
