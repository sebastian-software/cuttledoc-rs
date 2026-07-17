#!/usr/bin/env node

import { createReadStream, createWriteStream } from 'node:fs';
import {
  readFile,
  mkdir,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, relative, resolve } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { parseArgs } from 'node:util';

const MODEL_REPOSITORY = 'FluidInference/parakeet-tdt-0.6b-v3-coreml';
const MODEL_REVISION = 'aed02740059203c4a87495924f685de3722ae9ce';
const MODEL_TREE_SHA256 =
  '586856dc1886f4fe3c9912b05a5db591f5ec3750b8fa90aa7934f5d62bf07b67';
const VAD_REPOSITORY = 'FluidInference/silero-vad-coreml';
const VAD_REVISION = 'b419383c55c110e2c9271fa6ee0ea83d03c70d96';
const VAD_ROOT = 'silero-vad-unified-v6.0.0.mlmodelc';
const VAD_TREE_SHA256 =
  'e9f462027fc42a88aaa8915d443543759e5f1e670a02ce7cd9737d44131ab66d';
const VOCABULARIES = new Set([
  'parakeet_vocab.json',
  'parakeet_v3_vocab.json',
  'tokens.txt',
  'vocab.txt',
]);

const { values } = parseArgs({
  options: {
    'model-dir': { type: 'string' },
    'vad-dir': { type: 'string' },
  },
});
if (!values['model-dir'] || !values['vad-dir']) {
  throw new Error('--model-dir and --vad-dir are required');
}
const modelDir = resolve(values['model-dir']);
const vadDir = resolve(values['vad-dir']);

const modelEntries = (await repositoryTree(MODEL_REPOSITORY, MODEL_REVISION))
  .filter(
    (entry) =>
      entry.type === 'file' &&
      (entry.path.includes('.mlmodelc/') || VOCABULARIES.has(entry.path)),
  )
  .sort((left, right) => left.path.localeCompare(right.path));
const vadEntries = (await repositoryTree(VAD_REPOSITORY, VAD_REVISION))
  .filter(
    (entry) =>
      entry.type === 'file' && entry.path.startsWith(`${VAD_ROOT}/`),
  )
  .sort((left, right) => left.path.localeCompare(right.path));

await materialize(MODEL_REPOSITORY, MODEL_REVISION, modelEntries, modelDir);
await materialize(VAD_REPOSITORY, VAD_REVISION, vadEntries, vadDir);
await materializeTokens(modelDir);

const modelTree = await treeDigest(modelDir);
if (modelTree.sha256 !== MODEL_TREE_SHA256) {
  throw new Error(
    `unexpected model tree SHA-256: ${modelTree.sha256}; use an empty --model-dir`,
  );
}
const vadTree = await treeDigest(vadDir);
if (vadTree.sha256 !== VAD_TREE_SHA256) {
  throw new Error(
    `unexpected VAD tree SHA-256: ${vadTree.sha256}; use an empty --vad-dir`,
  );
}

process.stdout.write(
  `${JSON.stringify(
    {
      model: {
        repository: MODEL_REPOSITORY,
        revision: MODEL_REVISION,
        license: 'CC-BY-4.0',
        tree: modelTree,
      },
      vad: {
        repository: VAD_REPOSITORY,
        revision: VAD_REVISION,
        license: 'MIT',
        root: VAD_ROOT,
        tree: vadTree,
      },
    },
    null,
    2,
  )}\n`,
);

async function repositoryTree(repository, revision) {
  const url =
    `https://huggingface.co/api/models/${repository}/tree/${revision}` +
    '?recursive=true&expand=false';
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`tree request failed (${response.status}): ${url}`);
  }
  return response.json();
}

async function materialize(repository, revision, entries, destinationRoot) {
  await mkdir(destinationRoot, { recursive: true });
  for (const entry of entries) {
    const destination = join(destinationRoot, entry.path);
    if (await entryMatches(destination, entry)) {
      continue;
    }
    await download(
      repositoryUrl(repository, revision, entry.path),
      destination,
    );
    if (!(await entryMatches(destination, entry))) {
      throw new Error(`digest mismatch after download: ${entry.path}`);
    }
  }
}

async function materializeTokens(destinationRoot) {
  const vocabulary = JSON.parse(
    await readFile(join(destinationRoot, 'parakeet_v3_vocab.json'), 'utf8'),
  );
  const maxIndex = Math.max(...Object.keys(vocabulary).map(Number));
  const tokens = new Array(maxIndex + 1).fill('');
  for (const [index, token] of Object.entries(vocabulary)) {
    tokens[Number(index)] = token;
  }
  await writeFile(join(destinationRoot, 'tokens.txt'), tokens.join('\n'));
}

async function entryMatches(path, entry) {
  try {
    if (entry.lfs?.oid) {
      return (await digest(path, 'sha256')) === entry.lfs.oid;
    }
    const size = (await stat(path)).size;
    const hash = createHash('sha1');
    hash.update(`blob ${size}\0`);
    for await (const chunk of createReadStream(path)) {
      hash.update(chunk);
    }
    return hash.digest('hex') === entry.oid;
  } catch {
    return false;
  }
}

function repositoryUrl(repository, revision, file) {
  return `https://huggingface.co/${repository}/resolve/${revision}/${file}`;
}

async function download(url, destination) {
  await mkdir(dirname(destination), { recursive: true });
  const temporary = `${destination}.part`;
  await rm(temporary, { force: true });
  process.stderr.write(`Downloading ${url}\n`);
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(`download failed (${response.status}): ${url}`);
  }
  try {
    await pipeline(
      Readable.fromWeb(response.body),
      createWriteStream(temporary, { flags: 'wx' }),
    );
    await rename(temporary, destination);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

async function treeDigest(root) {
  const files = await walk(root);
  const aggregate = createHash('sha256');
  let bytes = 0;
  for (const path of files.sort()) {
    const size = (await stat(path)).size;
    const sha256 = await digest(path, 'sha256');
    bytes += size;
    aggregate.update(`${sha256}  ${relative(root, path)}\n`);
  }
  return { bytes, files: files.length, sha256: aggregate.digest('hex') };
}

async function walk(directory, files = []) {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      await walk(path, files);
    } else if (entry.isFile()) {
      files.push(path);
    }
  }
  return files;
}

async function digest(path, algorithm) {
  const hash = createHash(algorithm);
  await pipeline(createReadStream(path), hash);
  return hash.digest('hex');
}
