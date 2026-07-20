#!/usr/bin/env node

import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const languageOrder = ['en', 'de', 'es', 'fr', 'pt'];
const { values } = parseArgs({
  options: {
    input: { type: 'string', multiple: true },
    output: { type: 'string' },
    check: { type: 'string' },
  },
});

const inputPaths = values.input?.map((path) => resolve(path)) ??
  await discoverAggregatePaths();
const report = await buildReport(inputPaths);

if (values.check) {
  const checkedPath = resolve(values.check);
  const checked = JSON.parse(await readFile(checkedPath, 'utf8'));
  if (JSON.stringify(checked) !== JSON.stringify(report)) {
    throw new Error(
      `${checkedPath} differs from deterministic ASR error analysis`,
    );
  }
  console.log(
    `ASR error analysis is current: ${checkedPath} ` +
    `(${report.candidates.length} candidates)`,
  );
} else if (values.output) {
  const outputPath = resolve(values.output);
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`wrote ${outputPath}`);
} else {
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

async function discoverAggregatePaths() {
  const rawDirectory = join(repoRoot, 'benchmarks/raw');
  const paths = [];
  for (const entry of await readdir(rawDirectory, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const path = join(rawDirectory, entry.name, 'result.json');
    try {
      const value = JSON.parse(await readFile(path, 'utf8'));
      if (typeof value.matrix_run_id === 'string' &&
          Array.isArray(value.results)) {
        paths.push(path);
      }
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  }
  return paths.sort();
}

async function buildReport(paths) {
  const sources = [];
  const candidates = [];
  for (const path of paths) {
    const bytes = await readFile(path);
    const aggregate = JSON.parse(bytes);
    if (!Array.isArray(aggregate.results) ||
        typeof aggregate.matrix_run_id !== 'string') {
      throw new Error(`${path}: expected a multilingual aggregate`);
    }
    sources.push({
      path: relativePath(path),
      sha256: createHash('sha256').update(bytes).digest('hex'),
      matrix_run_id: aggregate.matrix_run_id,
      source_revision: aggregate.source_revision,
      captured_at: aggregate.captured_at,
      candidate_id: aggregate.candidate.id,
    });
    candidates.push(analyzeCandidate(aggregate));
  }
  if (candidates.length === 0) {
    throw new Error('no multilingual aggregate inputs found');
  }

  return {
    schema_version: '1.0.0',
    analysis_id: 'phase0-multilingual-fleurs-error-analysis-1',
    source_captured_through: sources
      .map((source) => source.captured_at)
      .filter(Boolean)
      .sort()
      .at(-1) ?? null,
    source_artifacts: sources,
    normalization: {
      recorded: {
        words:
          'Unicode lowercase; retain letters and numbers; delete punctuation',
        source_expression:
          ".toLocaleLowerCase('en-US').replace(/[^\\p{L}\\p{N}\\s]/gu, '')",
        role:
          'Reproduces the immutable phase-0 WER exactly, including punctuation-boundary collisions.',
      },
      review: {
        words:
          'Unicode lowercase; remove apostrophes inside words; treat hyphens, dashes, and slashes as word boundaries; delete remaining punctuation',
        source_expression:
          ".toLocaleLowerCase('en-US').replace(/['’ʼ]/gu, '').replace(/[-‐‑‒–—/]/gu, ' ').replace(/[^\\p{L}\\p{N}\\s]/gu, '')",
        role:
          'Produces review alignments without turning T-Rex into trex or 25-30 into 2530.',
      },
    },
    dimensions: {
      source_dataset: 'FLEURS',
      source_domain: 'short-read',
      languages: languageOrder,
      caveat:
        'One source and two fixtures per language; error classes are integration evidence, not population estimates.',
    },
    review_contract: {
      classifications: [
        'surface-only',
        'word-boundary-or-orthographic',
        'benign-lexical-or-inflectional',
        'critical-content',
        'omission',
        'insertion-or-hallucination',
        'attribution-timing-or-merge',
      ],
      severities: ['benign', 'material', 'critical'],
      automatic_state: 'unreviewed',
      note:
        'Operations and risk hints are deterministic. Semantic classification remains an explicit human review step.',
    },
    surface_metrics: {
      available: false,
      reason:
        'The current FLEURS references are normalized recognition text rather than verified punctuation and capitalization gold.',
    },
    candidates,
  };
}

function analyzeCandidate(aggregate) {
  const fixtures = aggregate.results.map((result) => analyzeFixture(result));
  return {
    id: aggregate.candidate.id,
    model: aggregate.candidate.model,
    aggregate: {
      recorded: summarize(fixtures, 'recorded'),
      review: summarize(fixtures, 'review'),
    },
    by_language: Object.fromEntries(
      languageOrder.map((language) => [
        language,
        {
          recorded: summarize(
            fixtures.filter((fixture) => fixture.language === language),
            'recorded',
          ),
          review: summarize(
            fixtures.filter((fixture) => fixture.language === language),
            'review',
          ),
        },
      ]),
    ),
    fixtures,
  };
}

function analyzeFixture(result) {
  const recordedReference = recordedWords(result.reference_text);
  const recordedHypothesis = recordedWords(result.text);
  const recordedAlignment = align(recordedReference, recordedHypothesis);
  const recordedCounts = countOperations(recordedAlignment);
  const recordedWer = errorRate(
    recordedReference,
    recordedHypothesis,
    recordedCounts,
  );
  if (Math.abs(recordedWer - result.quality.wer) > 1e-12) {
    throw new Error(
      `${result.fixture_id}: alignment WER ${recordedWer} differs from ` +
      `recorded ${result.quality.wer}`,
    );
  }

  const reviewReference = reviewWords(result.reference_text);
  const reviewHypothesis = reviewWords(result.text);
  const reviewAlignment = align(reviewReference, reviewHypothesis);
  const operations = reviewAlignment
    .filter((item) => item.operation !== 'equal')
    .map((item, index) => ({
      operation_id: `${result.fixture_id}:E${String(index + 1).padStart(3, '0')}`,
      operation: item.operation,
      reference_index: item.reference_index,
      hypothesis_index: item.hypothesis_index,
      reference: item.reference,
      hypothesis: item.hypothesis,
      reference_context: context(
        reviewReference,
        item.reference_index,
        item.reference.length,
      ),
      hypothesis_context: context(
        reviewHypothesis,
        item.hypothesis_index,
        item.hypothesis.length,
      ),
      risk_hints: riskHints(item, result.requested_language),
      review: {
        classification: 'unreviewed',
        severity: 'unreviewed',
        note: null,
      },
    }));
  const reviewCounts = countOperations(reviewAlignment);
  return {
    fixture_id: result.fixture_id,
    language: result.requested_language,
    source_dataset: 'FLEURS',
    source_domain: 'short-read',
    reference_text: result.reference_text,
    hypothesis_text: result.text,
    recorded: {
      reference_word_count: recordedReference.length,
      hypothesis_word_count: recordedHypothesis.length,
      wer: recordedWer,
      counts: recordedCounts,
    },
    review: {
      reference_word_count: reviewReference.length,
      hypothesis_word_count: reviewHypothesis.length,
      wer: errorRate(reviewReference, reviewHypothesis, reviewCounts),
      counts: reviewCounts,
      operations,
    },
  };
}

function recordedWords(text) {
  return text
    .toLocaleLowerCase('en-US')
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .split(/\s+/)
    .filter(Boolean);
}

function reviewWords(text) {
  return text
    .toLocaleLowerCase('en-US')
    .replace(/['’ʼ]/gu, '')
    .replace(/[-‐‑‒–—/]/gu, ' ')
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .split(/\s+/)
    .filter(Boolean);
}

function errorRate(reference, hypothesis, counts) {
  if (reference.length === 0) return hypothesis.length === 0 ? 0 : 1;
  return counts.edits / reference.length;
}

function align(reference, hypothesis) {
  const rows = reference.length + 1;
  const columns = hypothesis.length + 1;
  const distance = Array.from(
    { length: rows },
    () => Array(columns).fill(0),
  );
  for (let row = 0; row < rows; row += 1) distance[row][0] = row;
  for (let column = 0; column < columns; column += 1) {
    distance[0][column] = column;
  }
  for (let row = 1; row < rows; row += 1) {
    for (let column = 1; column < columns; column += 1) {
      const substitution = distance[row - 1][column - 1] +
        (reference[row - 1] === hypothesis[column - 1] ? 0 : 1);
      distance[row][column] = Math.min(
        substitution,
        distance[row - 1][column] + 1,
        distance[row][column - 1] + 1,
      );
    }
  }

  const reversed = [];
  let row = reference.length;
  let column = hypothesis.length;
  while (row > 0 || column > 0) {
    if (row > 0 && column > 0 &&
        reference[row - 1] === hypothesis[column - 1] &&
        distance[row][column] === distance[row - 1][column - 1]) {
      reversed.push({
        operation: 'equal',
        reference_index: row - 1,
        hypothesis_index: column - 1,
        reference: [reference[row - 1]],
        hypothesis: [hypothesis[column - 1]],
      });
      row -= 1;
      column -= 1;
      continue;
    }
    if (row > 0 && column > 0 &&
        distance[row][column] === distance[row - 1][column - 1] + 1) {
      reversed.push({
        operation: 'substitution',
        reference_index: row - 1,
        hypothesis_index: column - 1,
        reference: [reference[row - 1]],
        hypothesis: [hypothesis[column - 1]],
      });
      row -= 1;
      column -= 1;
      continue;
    }
    if (row > 0 &&
        distance[row][column] === distance[row - 1][column] + 1) {
      reversed.push({
        operation: 'deletion',
        reference_index: row - 1,
        hypothesis_index: column,
        reference: [reference[row - 1]],
        hypothesis: [],
      });
      row -= 1;
      continue;
    }
    if (column > 0 &&
        distance[row][column] === distance[row][column - 1] + 1) {
      reversed.push({
        operation: 'insertion',
        reference_index: row,
        hypothesis_index: column - 1,
        reference: [],
        hypothesis: [hypothesis[column - 1]],
      });
      column -= 1;
      continue;
    }
    throw new Error(`alignment traceback failed at ${row},${column}`);
  }
  return reversed.reverse();
}

function countOperations(alignment) {
  const counts = {
    substitutions: 0,
    deletions: 0,
    insertions: 0,
    edits: 0,
  };
  for (const item of alignment) {
    if (item.operation === 'equal') continue;
    counts[`${item.operation}s`] += 1;
    counts.edits += 1;
  }
  return counts;
}

function context(tokens, index, length) {
  return {
    before: tokens.slice(Math.max(0, index - 3), index),
    focus: tokens.slice(index, index + length),
    after: tokens.slice(index + length, index + length + 3),
  };
}

function riskHints(item, language) {
  const hints = new Set();
  if (item.operation === 'deletion') hints.add('omission');
  if (item.operation === 'insertion') hints.add('insertion');
  if (item.operation === 'substitution') hints.add('lexical-substitution');
  const changed = [...item.reference, ...item.hypothesis];
  if (changed.some((token) => /\p{N}/u.test(token))) {
    hints.add('number-date-or-unit');
  }
  const negations = {
    en: new Set(['not', 'no', 'never', 'without']),
    de: new Set(['nicht', 'kein', 'keine', 'keinen', 'keinem', 'keiner', 'nie', 'ohne']),
    es: new Set(['no', 'nunca', 'jamás', 'sin']),
    fr: new Set(['ne', 'n', 'pas', 'jamais', 'sans', 'aucun', 'aucune']),
    pt: new Set(['não', 'nunca', 'jamais', 'sem', 'nenhum', 'nenhuma']),
  };
  if (changed.some((token) => negations[language]?.has(token))) {
    hints.add('negation');
  }
  return [...hints];
}

function summarize(fixtures, view) {
  const counts = fixtures.reduce(
    (total, fixture) => ({
      substitutions: total.substitutions + fixture[view].counts.substitutions,
      deletions: total.deletions + fixture[view].counts.deletions,
      insertions: total.insertions + fixture[view].counts.insertions,
      edits: total.edits + fixture[view].counts.edits,
    }),
    { substitutions: 0, deletions: 0, insertions: 0, edits: 0 },
  );
  const referenceWordCount = fixtures.reduce(
    (sum, fixture) => sum + fixture[view].reference_word_count,
    0,
  );
  return {
    fixture_count: fixtures.length,
    reference_word_count: referenceWordCount,
    macro_wer: fixtures.length === 0
      ? null
      : fixtures.reduce((sum, fixture) => sum + fixture[view].wer, 0) /
        fixtures.length,
    micro_wer: referenceWordCount === 0 ? null : counts.edits / referenceWordCount,
    counts,
    ...(view === 'review'
      ? { unreviewed_operation_count: counts.edits }
      : {}),
  };
}

function relativePath(path) {
  const prefix = `${repoRoot}/`;
  return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}
