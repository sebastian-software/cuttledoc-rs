#!/usr/bin/env node

import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, rename, rm, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, relative, resolve } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { parseArgs } from 'node:util';

const GGML_REPOSITORY = 'ggerganov/whisper.cpp';
const GGML_REVISION = '5359861c739e955e79d9a303bcbc70fb988958b1';
const GGML_FILE = 'ggml-large-v3-turbo.bin';
const GGML_SHA1 = '4af2b29d7ec73d781377bfd1758ca957a807e941';
const GGML_SHA256 =
  '1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69';

const COREML_REPOSITORY = 'sebastian-software/whisper-coreml-models';
const COREML_REVISION = 'dd3515371e6b560b63ec275abf020153a45caa60';
const COREML_ROOT = 'ggml-large-v3-turbo-encoder.mlmodelc';
const COREML_FILES = {
  'analytics/coremldata.bin':
    'a207b44aef13857e63631fd0e898c595f6147a8016b0b368d81bf8f84c9b3fbc',
  'coremldata.bin':
    '6518bbdf3ccd178e703f839c37f287995826ce6f9ba977057121c8497426bc90',
  'metadata.json':
    '9c7444439b70609a0bd8aad3d624f3d491d1e19beefbc748f7f22c6a2ea0e8ea',
  'model.mil':
    '7df4ffa5df99643d58d8b66ef7f61aeb320d29210785830a640fd4f7da9eccac',
  'weights/weight.bin':
    'fcc450fb244d55335f6df82a41558de1b07d44acaf67c7b7b3040da44f94bdd3',
};
const MODEL_TREE_SHA256 =
  'a4ce8c6307d111df0e08836120f6b14c7852f6ffddcba78f351cbfc436f4d3e0';

const { values } = parseArgs({
  options: {
    'model-dir': { type: 'string' },
  },
});
if (!values['model-dir']) {
  throw new Error('--model-dir is required');
}
const modelDir = resolve(values['model-dir']);
await mkdir(modelDir, { recursive: true });

const ggmlPath = join(modelDir, GGML_FILE);
if ((await existingDigest(ggmlPath, 'sha1')) !== GGML_SHA1) {
  await download(
    repositoryUrl(GGML_REPOSITORY, GGML_REVISION, GGML_FILE),
    ggmlPath,
  );
}
const ggmlSha1 = await digest(ggmlPath, 'sha1');
if (ggmlSha1 !== GGML_SHA1) {
  throw new Error(`unexpected ${GGML_FILE} SHA-1: ${ggmlSha1}`);
}
const ggmlSha256 = await digest(ggmlPath, 'sha256');
if (ggmlSha256 !== GGML_SHA256) {
  throw new Error(`unexpected ${GGML_FILE} SHA-256: ${ggmlSha256}`);
}

const coremlFiles = [];
for (const [relativePath, expectedSha256] of Object.entries(COREML_FILES)) {
  const repositoryPath = `${COREML_ROOT}/${relativePath}`;
  const destination = join(modelDir, repositoryPath);
  if ((await existingDigest(destination, 'sha256')) !== expectedSha256) {
    await download(
      repositoryUrl(COREML_REPOSITORY, COREML_REVISION, repositoryPath),
      destination,
    );
  }
  const sha256 = await digest(destination, 'sha256');
  if (sha256 !== expectedSha256) {
    throw new Error(`unexpected ${repositoryPath} SHA-256: ${sha256}`);
  }
  coremlFiles.push({
    path: repositoryPath,
    bytes: (await stat(destination)).size,
    sha256,
  });
}

const tree = await treeDigest(modelDir);
if (tree.sha256 !== MODEL_TREE_SHA256) {
  throw new Error(
    `unexpected model tree SHA-256: ${tree.sha256}; use an empty --model-dir`,
  );
}

process.stdout.write(
  `${JSON.stringify(
    {
      ggml: {
        repository: GGML_REPOSITORY,
        revision: GGML_REVISION,
        license: 'MIT',
        file: GGML_FILE,
        bytes: (await stat(ggmlPath)).size,
        sha1: ggmlSha1,
        sha256: ggmlSha256,
      },
      coreml: {
        repository: COREML_REPOSITORY,
        revision: COREML_REVISION,
        license: 'MIT',
        root: COREML_ROOT,
        files: coremlFiles,
      },
      tree,
    },
    null,
    2,
  )}\n`,
);

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

async function existingDigest(path, algorithm) {
  try {
    return await digest(path, algorithm);
  } catch {
    return null;
  }
}

async function digest(path, algorithm) {
  const hash = createHash(algorithm);
  await pipeline(createReadStream(path), hash);
  return hash.digest('hex');
}

async function treeDigest(root) {
  const files = [
    ggmlPath,
    ...Object.keys(COREML_FILES).map((path) =>
      join(root, COREML_ROOT, path),
    ),
  ].sort();
  const aggregate = createHash('sha256');
  let bytes = 0;
  for (const path of files) {
    const size = (await stat(path)).size;
    const sha256 = await digest(path, 'sha256');
    bytes += size;
    aggregate.update(`${sha256}  ${relative(root, path)}\n`);
  }
  return { bytes, files: files.length, sha256: aggregate.digest('hex') };
}
