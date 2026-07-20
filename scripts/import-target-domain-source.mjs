#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';

try {
  const options = parseArguments(process.argv.slice(2));
  if (options.selfTest) {
    await runSelfTest();
  } else {
    const reviewPath = resolve(options.review);
    const review = JSON.parse(await readFile(reviewPath, 'utf8'));
    const result = await importSource({
      review,
      sourcePath: resolve(options.source),
      outputDirectory: resolve(options.outputDirectory),
    });
    process.stdout.write(
      `target-domain source: verified ${result.reviewId} at ` +
      `${result.artifactPath}\n`,
    );
  }
} catch (error) {
  process.stderr.write(`target-domain source: ${error.message}\n`);
  process.exitCode = 1;
}

function parseArguments(arguments_) {
  if (arguments_.length === 1 && arguments_[0] === '--self-test') {
    return { selfTest: true };
  }
  const values = {};
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === '--review') {
      values.review = arguments_[index + 1];
      index += 1;
    } else if (argument === '--source') {
      values.source = arguments_[index + 1];
      index += 1;
    } else if (argument === '--output-dir') {
      values.outputDirectory = arguments_[index + 1];
      index += 1;
    } else {
      throw new Error(`unknown argument: ${argument}`);
    }
  }
  if (!values.review || !values.source || !values.outputDirectory) {
    throw new Error(
      'usage: node scripts/import-target-domain-source.mjs ' +
      '--review benchmarks/rights/<source-group>.json ' +
      '--source /absolute/path/to/source-audio ' +
      '--output-dir /absolute/path/to/corpus',
    );
  }
  return { ...values, selfTest: false };
}

async function importSource({ review, sourcePath, outputDirectory }) {
  validateImportableReview(review);
  const sourceMetadata = await stat(sourcePath);
  if (!sourceMetadata.isFile()) {
    throw new Error(`${sourcePath}: source must be a regular file`);
  }
  const actualDigest = await sha256File(sourcePath);
  if (actualDigest !== review.acquisition.expected_source_sha256) {
    throw new Error(
      `${review.review_id}: source SHA-256 mismatch; expected ` +
      `${review.acquisition.expected_source_sha256}, received ${actualDigest}`,
    );
  }

  await mkdir(outputDirectory, { recursive: true });
  const artifactPath = join(outputDirectory, review.acquisition.artifact_name);
  const provenancePath = `${artifactPath}.provenance.json`;
  const provenance = {
    schema_version: '1.0.0',
    review_id: review.review_id,
    source_candidate_id: review.source_candidate_id,
    source_group_id: review.source_group_id,
    source_url: review.acquisition.source_url,
    artifact_name: review.acquisition.artifact_name,
    source_sha256: actualDigest,
    redistribution: review.rights.redistribution,
  };
  const provenanceBytes = `${JSON.stringify(provenance, null, 2)}\n`;
  if (await pathExists(provenancePath)) {
    const existing = await readFile(provenancePath, 'utf8');
    if (existing !== provenanceBytes) {
      throw new Error(
        `${provenancePath}: existing provenance differs from accepted review`,
      );
    }
  }

  if (await pathExists(artifactPath)) {
    const existingDigest = await sha256File(artifactPath);
    if (existingDigest !== actualDigest) {
      throw new Error(
        `${artifactPath}: existing artifact does not match accepted review`,
      );
    }
  } else {
    await copyFile(sourcePath, artifactPath, constants.COPYFILE_EXCL);
    const copiedDigest = await sha256File(artifactPath);
    if (copiedDigest !== actualDigest) {
      await rm(artifactPath, { force: true });
      throw new Error(`${artifactPath}: copied artifact failed verification`);
    }
  }

  if (!(await pathExists(provenancePath))) {
    await writeFile(provenancePath, provenanceBytes, {
      encoding: 'utf8',
      flag: 'wx',
    });
  }

  return {
    reviewId: review.review_id,
    artifactPath,
    provenancePath,
  };
}

function validateImportableReview(review) {
  if (review.schema_version !== '1.0.0') {
    throw new Error('review schema_version must be 1.0.0');
  }
  if (review.disposition !== 'accepted') {
    throw new Error(
      `${review.review_id ?? '<unknown review>'}: acquisition is blocked; ` +
      `disposition is ${review.disposition ?? 'missing'}`,
    );
  }
  if (review.scope !== 'source-group' || !(review.source_group_id?.length > 0)) {
    throw new Error(
      `${review.review_id}: accepted acquisition requires a source-group review`,
    );
  }
  if (review.benchmark_role !== 'target-domain') {
    throw new Error(
      `${review.review_id}: only target-domain reviews may use this importer`,
    );
  }
  for (const field of [
    'audio',
    'transcript',
    'derived_clips',
    'commercial_benchmark_use',
  ]) {
    const decision = review.rights?.[field];
    if (decision?.status !== 'accepted' ||
        !(decision.basis?.length > 0) ||
        !Array.isArray(decision.evidence_urls) ||
        decision.evidence_urls.length === 0) {
      throw new Error(
        `${review.review_id}: rights.${field} lacks accepted evidence`,
      );
    }
  }
  if (!['local-only', 'allowed'].includes(review.rights?.redistribution)) {
    throw new Error(
      `${review.review_id}: redistribution must be local-only or allowed`,
    );
  }
  if (review.development_isolation?.status !== 'accepted' ||
      !Array.isArray(review.development_isolation.evidence) ||
      review.development_isolation.evidence.length === 0) {
    throw new Error(
      `${review.review_id}: development-set isolation is not accepted`,
    );
  }
  if (review.acquisition?.status !== 'allowed' ||
      review.acquisition?.method !== 'manual-local-file') {
    throw new Error(
      `${review.review_id}: acquisition must allow manual-local-file import`,
    );
  }
  if (!(review.acquisition.source_url?.length > 0)) {
    throw new Error(`${review.review_id}: acquisition.source_url is required`);
  }
  if (!/^[0-9a-f]{64}$/.test(
    review.acquisition.expected_source_sha256 ?? '',
  )) {
    throw new Error(
      `${review.review_id}: acquisition.expected_source_sha256 must be SHA-256`,
    );
  }
  const artifactName = review.acquisition.artifact_name;
  if (!(artifactName?.length > 0) ||
      artifactName !== basename(artifactName) ||
      artifactName === '.' ||
      artifactName === '..') {
    throw new Error(
      `${review.review_id}: acquisition.artifact_name must be a safe filename`,
    );
  }
  if (!Array.isArray(review.blockers) || review.blockers.length !== 0) {
    throw new Error(`${review.review_id}: accepted review must have no blockers`);
  }
}

async function sha256File(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
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
  const directory = await mkdtemp(join(tmpdir(), 'cuttledoc-source-import-'));
  try {
    const sourcePath = join(directory, 'source.wav');
    const outputDirectory = join(directory, 'output');
    const sourceBytes = Buffer.from('cuttledoc target-domain source self-test\n');
    const digest = createHash('sha256').update(sourceBytes).digest('hex');
    await writeFile(sourcePath, sourceBytes);
    const acceptedDecision = {
      status: 'accepted',
      basis: 'explicit-permission',
      evidence_urls: ['https://example.invalid/permission'],
      conditions: ['Local benchmark use only.'],
    };
    const acceptedReview = {
      schema_version: '1.0.0',
      review_id: 'self-test-source-group-1',
      source_candidate_id: 'cuttledoc-professional-podcast-gold',
      source_group_id: 'self-test-episode',
      scope: 'source-group',
      disposition: 'accepted',
      benchmark_role: 'target-domain',
      rights: {
        audio: acceptedDecision,
        transcript: acceptedDecision,
        derived_clips: acceptedDecision,
        commercial_benchmark_use: acceptedDecision,
        redistribution: 'local-only',
      },
      development_isolation: {
        status: 'accepted',
        evidence: ['Synthetic self-test source is absent from development data.'],
      },
      acquisition: {
        status: 'allowed',
        method: 'manual-local-file',
        source_url: 'https://example.invalid/source',
        expected_source_sha256: digest,
        artifact_name: 'self-test-source.wav',
      },
      blockers: [],
    };
    const imported = await importSource({
      review: acceptedReview,
      sourcePath,
      outputDirectory,
    });
    if (await sha256File(imported.artifactPath) !== digest) {
      throw new Error('self-test imported artifact digest changed');
    }
    await importSource({
      review: acceptedReview,
      sourcePath,
      outputDirectory,
    });
    const blockedReview = {
      ...acceptedReview,
      review_id: 'self-test-blocked-1',
      disposition: 'blocked',
    };
    let rejected = false;
    try {
      validateImportableReview(blockedReview);
    } catch {
      rejected = true;
    }
    if (!rejected) {
      throw new Error('self-test failed to reject a blocked review');
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
  process.stdout.write('target-domain source importer: self-test passed\n');
}
