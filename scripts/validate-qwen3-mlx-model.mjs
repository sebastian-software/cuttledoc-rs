#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const manifestPath = join(
  repositoryRoot,
  'spikes/qwen3-mlx-direct/model-manifest.json',
);
const modelDirectory = parseArguments(process.argv.slice(2));
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));

const errors = [];
const paths = new Set();
for (const artifact of manifest.artifacts ?? []) {
  if (paths.has(artifact.path)) {
    errors.push(`duplicate artifact path: ${artifact.path}`);
    continue;
  }
  paths.add(artifact.path);
  if (!/^[0-9a-f]{64}$/.test(artifact.sha256 ?? '')) {
    errors.push(`${artifact.path}: sha256 must contain 64 lowercase hex digits`);
    continue;
  }
  const artifactPath = join(modelDirectory, artifact.path);
  let artifactStat;
  try {
    artifactStat = await stat(artifactPath);
  } catch (error) {
    errors.push(`${artifact.path}: ${error.message}`);
    continue;
  }
  if (!artifactStat.isFile()) {
    errors.push(`${artifact.path}: not a regular file`);
    continue;
  }
  if (artifactStat.size !== artifact.bytes) {
    errors.push(
      `${artifact.path}: ${artifactStat.size} bytes, expected ${artifact.bytes}`,
    );
    continue;
  }
  const digest = await sha256(artifactPath);
  if (digest !== artifact.sha256) {
    errors.push(`${artifact.path}: SHA-256 ${digest}, expected ${artifact.sha256}`);
  }
}

if (errors.length > 0) {
  for (const error of errors) console.error(error);
  process.exit(1);
}

console.log(
  JSON.stringify({
    status: 'ok',
    manifest: 'spikes/qwen3-mlx-direct/model-manifest.json',
    model_directory: modelDirectory,
    source_revision: manifest.source.revision,
    conversion_revision: manifest.conversion.revision,
    artifact_count: manifest.artifacts.length,
    artifact_bytes: manifest.artifacts.reduce(
      (total, artifact) => total + artifact.bytes,
      0,
    ),
  }),
);

function parseArguments(arguments_) {
  if (
    arguments_.length !== 2 ||
    arguments_[0] !== '--model-dir' ||
    !arguments_[1]
  ) {
    console.error(
      'usage: node scripts/validate-qwen3-mlx-model.mjs --model-dir MODEL_DIR',
    );
    process.exit(2);
  }
  return resolve(arguments_[1]);
}

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
