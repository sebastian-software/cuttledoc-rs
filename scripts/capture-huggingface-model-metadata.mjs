#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    repository: { type: 'string' },
    revision: { type: 'string' },
    output: { type: 'string' },
  },
});

if (!values.repository || !values.revision) {
  throw new Error(
    'usage: node scripts/capture-huggingface-model-metadata.mjs ' +
      '--repository OWNER/MODEL --revision COMMIT [--output FILE]',
  );
}
if (!/^[0-9a-f]{40}$/.test(values.revision)) {
  throw new Error('--revision must be a lowercase 40-character commit SHA');
}

const repository = values.repository;
const revision = values.revision;
const apiUrl = `https://huggingface.co/api/models/${repository}?blobs=true`;
const response = await fetch(apiUrl);
if (!response.ok) {
  throw new Error(`Hugging Face API failed with HTTP ${response.status}`);
}
const model = await response.json();
if (model.sha !== revision) {
  throw new Error(
    `repository main moved: expected ${revision}, API reports ${model.sha}`,
  );
}

const artifacts = [];
for (const sibling of model.siblings ?? []) {
  const path = sibling.rfilename;
  if (!(path?.length > 0) || !Number.isInteger(sibling.size) || sibling.size < 0) {
    throw new Error(`invalid sibling metadata: ${JSON.stringify(sibling)}`);
  }
  if (sibling.lfs) {
    if (sibling.lfs.size !== sibling.size ||
        !/^[0-9a-f]{64}$/.test(sibling.lfs.sha256 ?? '')) {
      throw new Error(`${path}: invalid LFS metadata`);
    }
    artifacts.push({
      path,
      bytes: sibling.size,
      sha256: sibling.lfs.sha256,
      digest_source: 'hugging-face-lfs-metadata',
    });
    continue;
  }

  if (sibling.size > 32 * 1024 * 1024) {
    throw new Error(`${path}: refusing to hash a non-LFS file larger than 32 MiB`);
  }
  const encodedPath = path.split('/').map(encodeURIComponent).join('/');
  const artifactUrl =
    `https://huggingface.co/${repository}/resolve/${revision}/${encodedPath}`;
  const artifactResponse = await fetch(artifactUrl);
  if (!artifactResponse.ok) {
    throw new Error(`${path}: download failed with HTTP ${artifactResponse.status}`);
  }
  const bytes = Buffer.from(await artifactResponse.arrayBuffer());
  if (bytes.length !== sibling.size) {
    throw new Error(
      `${path}: API reports ${sibling.size} bytes, downloaded ${bytes.length}`,
    );
  }
  artifacts.push({
    path,
    bytes: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex'),
    digest_source: 'downloaded-at-pinned-revision',
  });
}

const result = {
  repository,
  revision,
  captured_at: new Date().toISOString(),
  last_modified: model.lastModified,
  gated: model.gated,
  library_name: model.library_name ?? null,
  pipeline_tag: model.pipeline_tag ?? null,
  license_tag:
    (model.tags ?? []).find((tag) => tag.startsWith('license:')) ?? null,
  artifacts,
  artifact_count: artifacts.length,
  weight_bytes: artifacts
    .filter((artifact) => artifact.path.endsWith('.safetensors'))
    .reduce((sum, artifact) => sum + artifact.bytes, 0),
  snapshot_bytes: artifacts.reduce((sum, artifact) => sum + artifact.bytes, 0),
};
const serialized = `${JSON.stringify(result, null, 2)}\n`;
if (values.output) {
  await writeFile(values.output, serialized);
  process.stderr.write(`Wrote ${values.output}\n`);
} else {
  process.stdout.write(serialized);
}
