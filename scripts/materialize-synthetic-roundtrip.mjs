#!/usr/bin/env node

import {
  access,
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultSelectionPath = join(
  repoRoot,
  'benchmarks/fixtures/synthetic-roundtrip-selection.json',
);

function sha256(text) {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

function parsePlainExtract(extract) {
  const paragraphs = [];
  const counts = new Map();
  let sectionPath = [];
  let lines = [];

  const flush = () => {
    const text = lines.join(' ').replace(/\s+/g, ' ').trim();
    lines = [];
    if (text.length === 0) return;
    const path = sectionPath.length > 0 ? [...sectionPath] : ['lead'];
    const key = path.join('\u001f');
    const paragraphIndex = counts.get(key) ?? 0;
    counts.set(key, paragraphIndex + 1);
    paragraphs.push({
      section_path: path,
      paragraph_index: paragraphIndex,
      text,
    });
  };

  for (const line of extract.replace(/\r\n/g, '\n').split('\n')) {
    const heading = line.match(/^(={2,6})\s*(.*?)\s*\1$/);
    if (heading) {
      flush();
      const level = heading[1].length - 2;
      sectionPath = sectionPath.slice(0, level);
      sectionPath[level] = heading[2];
    } else if (line.trim().length === 0) {
      flush();
    } else {
      lines.push(line.trim());
    }
  }
  flush();
  return paragraphs;
}

function validateSelection(selection) {
  const errors = [];
  if (selection.schema_version !== '1.0.0') {
    errors.push('schema_version must be 1.0.0');
  }
  if (selection.purpose !== 'diagnostic-materialization' ||
      selection.text_encoding !== 'utf-8' ||
      selection.paragraph_joiner !== '\n\n' ||
      selection.spoken_transform !== 'none' ||
      selection.generated_assets !==
        'lossless-local-opus-repository-after-rights-review') {
    errors.push('selection policy fields do not match the accepted diagnostic contract');
  }
  const sourceIds = new Set();
  const passageIds = new Set();
  const localeCounts = new Map();
  for (const source of selection.sources ?? []) {
    if (sourceIds.has(source.id)) errors.push(`duplicate source id: ${source.id}`);
    sourceIds.add(source.id);
    if (!['mediawiki', 'repository-authored'].includes(source.kind)) {
      errors.push(`${source.id}: unsupported source kind`);
    }
    if (source.kind === 'mediawiki' &&
        (!Number.isInteger(source.page_id) || source.page_id < 1 ||
         !Number.isInteger(source.revision_id) || source.revision_id < 1 ||
         !Number.isInteger(source.parent_revision_id) ||
           source.parent_revision_id < 1)) {
      errors.push(`${source.id}: page and revision ids must be positive integers`);
    }
    if (source.kind === 'repository-authored' &&
        (!(source.path?.length > 0) ||
         !/^[0-9a-f]{64}$/.test(source.revision ?? ''))) {
      errors.push(`${source.id}: repository path and SHA-256 revision are required`);
    }
    if (source.license !== 'CC-BY-SA-4.0' ||
        source.license_url !==
          'https://creativecommons.org/licenses/by-sa/4.0/') {
      errors.push(`${source.id}: license metadata must remain CC BY-SA 4.0`);
    }
    const requiredFields = [
      'revision_url',
      'history_url',
      'attribution',
      ...(source.kind === 'mediawiki' ? ['api_url', 'revision_timestamp'] : []),
    ];
    for (const field of requiredFields) {
      if (!(source[field]?.length > 0)) {
        errors.push(`${source.id}.${field} must be a non-empty string`);
      }
    }
    localeCounts.set(
      source.locale,
      (localeCounts.get(source.locale) ?? 0) + (source.passages?.length ?? 0),
    );
    for (const passage of source.passages ?? []) {
      if (passageIds.has(passage.id)) {
        errors.push(`duplicate passage id: ${passage.id}`);
      }
      passageIds.add(passage.id);
      if (!Array.isArray(passage.segments) || passage.segments.length === 0) {
        errors.push(`${passage.id}: segments must not be empty`);
      }
      for (const segment of passage.segments ?? []) {
        if (!Array.isArray(segment.section_path) ||
            segment.section_path.length === 0 ||
            segment.section_path.some(
              (part) => typeof part !== 'string' || part.length === 0,
            ) ||
            !Number.isInteger(segment.paragraph_index) ||
            segment.paragraph_index < 0) {
          errors.push(`${passage.id}: invalid segment selector`);
        }
      }
      if (!(Number.isInteger(passage.character_count) &&
          passage.character_count > 0)) {
        errors.push(`${passage.id}: character_count must be a positive integer`);
      }
      for (const field of ['verbatim_sha256', 'spoken_sha256']) {
        if (!/^[0-9a-f]{64}$/.test(passage[field] ?? '')) {
          errors.push(`${passage.id}.${field} must be a SHA-256 digest`);
        }
      }
      if (passage.verbatim_sha256 !== passage.spoken_sha256) {
        errors.push(`${passage.id}: no spoken transform permits divergent digests`);
      }
      if (!Array.isArray(passage.phenomena) ||
          passage.phenomena.length === 0 ||
          new Set(passage.phenomena).size !== passage.phenomena.length) {
        errors.push(`${passage.id}: phenomena must be a non-empty unique array`);
      }
    }
  }
  const minimumPassages = new Map([
    ['de-DE', 8],
    ['en-US', 5],
    ['es-419', 3],
    ['fr-FR', 3],
    ['pt-BR', 3],
  ]);
  for (const [locale, minimum] of minimumPassages) {
    if ((localeCounts.get(locale) ?? 0) < minimum) {
      errors.push(`${locale}: selection requires at least ${minimum} passages`);
    }
  }
  return errors;
}

function findParagraph(paragraphs, segment, passageId) {
  const paragraph = paragraphs.find(
    (candidate) => (
      candidate.paragraph_index === segment.paragraph_index &&
      candidate.section_path.length === segment.section_path.length &&
      candidate.section_path.every(
        (part, index) => part === segment.section_path[index],
      )
    ),
  );
  if (!paragraph) {
    throw new Error(
      `${passageId}: missing paragraph ` +
      `${segment.section_path.join(' > ')}[${segment.paragraph_index}]`,
    );
  }
  return paragraph.text;
}

async function fetchSource(source) {
  if (source.kind === 'repository-authored') {
    const sourcePath = resolve(repoRoot, source.path);
    if (sourcePath !== repoRoot && !sourcePath.startsWith(`${repoRoot}${sep}`)) {
      throw new Error(`${source.id}: repository source escapes the repository root`);
    }
    const text = await readFile(sourcePath, 'utf8');
    if (sha256(text) !== source.revision) {
      throw new Error(`${source.id}: repository source differs from pinned SHA-256`);
    }
    return parsePlainExtract(text);
  }
  const url = new URL(source.api_url);
  url.search = new URLSearchParams({
    action: 'query',
    prop: 'extracts|revisions',
    revids: String(source.revision_id),
    explaintext: '1',
    rvprop: 'ids|timestamp',
    format: 'json',
    formatversion: '2',
  });
  const response = await fetch(url, {
    headers: {
      'User-Agent': 'cuttledoc-rs synthetic benchmark materializer',
    },
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) {
    throw new Error(`${source.id}: MediaWiki returned HTTP ${response.status}`);
  }
  const payload = await response.json();
  const page = payload.query?.pages?.[0];
  const revision = page?.revisions?.[0];
  if (page?.pageid !== source.page_id ||
      page?.title !== source.title ||
      revision?.revid !== source.revision_id ||
      revision?.parentid !== source.parent_revision_id ||
      revision?.timestamp !== source.revision_timestamp) {
    throw new Error(`${source.id}: pinned page or revision metadata changed`);
  }
  if (!(page.extract?.length > 0)) {
    throw new Error(`${source.id}: MediaWiki returned no plain-text extract`);
  }
  return parsePlainExtract(page.extract);
}

function attributionMarkdown(selection) {
  const lines = [
    '# Attribution for synthetic roundtrip text',
    '',
    'The materialized passage files are derived from the pinned Wikimedia and',
    'repository sources below and are licensed under CC BY-SA 4.0. Wikimedia',
    'extraction removes page markup and references; no additional spoken-text',
    'transform is applied in this selection.',
    '',
  ];
  for (const source of selection.sources) {
    lines.push(
      `- ${source.attribution}. ` +
      `[Revision](${source.revision_url}); ` +
      `[history](${source.history_url}); ` +
      `[CC BY-SA 4.0](${source.license_url}).`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

async function materialize(selection, outputDirectory) {
  await access(dirname(outputDirectory));
  try {
    await access(outputDirectory);
    throw new Error(`output directory already exists: ${outputDirectory}`);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  const temporaryDirectory = `${outputDirectory}.tmp-${process.pid}`;
  await mkdir(temporaryDirectory);
  try {
    const outputManifest = {
      schema_version: selection.schema_version,
      selection_revision: selection.revision,
      generated_at: new Date().toISOString(),
      text_encoding: selection.text_encoding,
      spoken_transform: selection.spoken_transform,
      sources: [],
      passages: [],
    };
    for (const source of selection.sources) {
      const paragraphs = await fetchSource(source);
      outputManifest.sources.push({
        id: source.id,
        kind: source.kind,
        locale: source.locale,
        title: source.title,
        ...(source.kind === 'mediawiki'
          ? { revision_id: source.revision_id }
          : { path: source.path, revision: source.revision }),
        revision_url: source.revision_url,
        history_url: source.history_url,
        license: source.license,
        license_url: source.license_url,
        attribution: source.attribution,
      });
      for (const passage of source.passages) {
        const text = passage.segments
          .map((segment) => findParagraph(paragraphs, segment, passage.id))
          .join(selection.paragraph_joiner);
        const digest = sha256(text);
        const characterCount = [...text].length;
        if (digest !== passage.verbatim_sha256 ||
            digest !== passage.spoken_sha256 ||
            characterCount !== passage.character_count) {
          throw new Error(
            `${passage.id}: materialized text differs from pinned digest or length`,
          );
        }
        const file = `${passage.id}.txt`;
        await writeFile(join(temporaryDirectory, file), text, 'utf8');
        outputManifest.passages.push({
          id: passage.id,
          source_id: source.id,
          locale: source.locale,
          file,
          character_count: characterCount,
          verbatim_sha256: digest,
          spoken_sha256: digest,
          phenomena: passage.phenomena,
          ...(passage.content_type
            ? { content_type: passage.content_type }
            : {}),
        });
      }
    }
    await writeFile(
      join(temporaryDirectory, 'ATTRIBUTION.md'),
      attributionMarkdown(selection),
      'utf8',
    );
    await writeFile(
      join(temporaryDirectory, 'manifest.json'),
      `${JSON.stringify(outputManifest, null, 2)}\n`,
      'utf8',
    );
    await rename(temporaryDirectory, outputDirectory);
    return outputManifest;
  } catch (error) {
    await rm(temporaryDirectory, { recursive: true, force: true });
    throw error;
  }
}

function selfTest() {
  const extract = [
    'Lead text.',
    '',
    '== Section ==',
    'First line.',
    'Second line.',
    '',
    '=== Child ===',
    'Child text.',
  ].join('\n');
  const paragraphs = parsePlainExtract(extract);
  if (paragraphs.length !== 3 ||
      paragraphs[0].section_path[0] !== 'lead' ||
      paragraphs[1].text !== 'First line. Second line.' ||
      paragraphs[2].section_path.join('/') !== 'Section/Child') {
    throw new Error('plain extract parser self-test failed');
  }
  const invalid = {
    schema_version: '1.0.0',
    purpose: 'diagnostic-materialization',
    text_encoding: 'utf-8',
    paragraph_joiner: '\n\n',
    spoken_transform: 'none',
    generated_assets: 'invalid',
    sources: [],
  };
  if (validateSelection(invalid).length === 0) {
    throw new Error('selection validator self-test failed');
  }
  process.stdout.write('synthetic roundtrip materializer: self-test passed\n');
}

async function main() {
  if (process.argv.includes('--self-test')) {
    selfTest();
    return;
  }
  const selectionIndex = process.argv.indexOf('--selection');
  const outputIndex = process.argv.indexOf('--output-dir');
  if (outputIndex < 0 || !process.argv[outputIndex + 1]) {
    throw new Error(
      'usage: node scripts/materialize-synthetic-roundtrip.mjs ' +
      '[--selection <path>] --output-dir <new-directory>',
    );
  }
  const selectionPath = selectionIndex >= 0
    ? resolve(process.cwd(), process.argv[selectionIndex + 1])
    : defaultSelectionPath;
  const outputDirectory = resolve(process.cwd(), process.argv[outputIndex + 1]);
  const selection = JSON.parse(await readFile(selectionPath, 'utf8'));
  const errors = validateSelection(selection);
  if (errors.length > 0) throw new Error(errors.join('\n'));
  const result = await materialize(selection, outputDirectory);
  process.stdout.write(
    `synthetic roundtrip: materialized ${result.passages.length} passage(s) ` +
    `from ${result.sources.length} pinned source(s) at ` +
    `${outputDirectory}\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`synthetic roundtrip: ${error.message}\n`);
  process.exitCode = 1;
});
