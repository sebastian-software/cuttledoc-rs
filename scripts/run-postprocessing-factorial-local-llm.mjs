#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    'output-dir': {
      type: 'string',
      default: join(repoRoot, 'artifacts/postprocessing-factorial-local-plan-6'),
    },
    'text-dir': {
      type: 'string',
      default: '/private/tmp/cuttledoc-synthetic-passages-4-verify',
    },
    ledger: {
      type: 'string',
      default: join(repoRoot, 'benchmarks/postprocessing/factorial-cells.json'),
    },
    selection: {
      type: 'string',
      default: join(repoRoot, 'benchmarks/fixtures/synthetic-roundtrip-selection.json'),
    },
    screen: {
      type: 'string',
      default: join(repoRoot, 'benchmarks/postprocessing/local-llm-screen.json'),
    },
    'model-base-dir': {
      type: 'string',
      default: '/private/tmp/cuttledoc-textgen-models',
    },
    python: {
      type: 'string',
      default: join(repoRoot, 'spikes/text-generation-mlx-reference/.venv/bin/python'),
    },
    candidate: { type: 'string', default: 'all' },
    locale: { type: 'string' },
    limit: { type: 'string' },
    repetitions: { type: 'string' },
    force: { type: 'boolean', default: false },
  },
});

const command = positionals[0];
if (!['materialize', 'run', 'status', 'self-test'].includes(command)) {
  throw new Error(
    'usage: node scripts/run-postprocessing-factorial-local-llm.mjs ' +
    '<materialize|run|status|self-test> [options]',
  );
}

const paths = {
  output: resolve(values['output-dir']),
  text: resolve(values['text-dir']),
  ledger: resolve(values.ledger),
  selection: resolve(values.selection),
  screen: resolve(values.screen),
  modelBase: resolve(values['model-base-dir']),
  python: resolve(values.python),
};
paths.stt = join(paths.output, 'stt');
paths.audio = join(paths.output, 'audio');
paths.documents = join(paths.output, 'llm-documents');
paths.results = join(paths.output, 'llm-local-results');
paths.jobs = join(paths.output, 'llm-local-jobs.json');

if (command === 'self-test') {
  selfTest();
} else if (command === 'materialize') {
  await materializeDocuments();
  await printStatus();
} else if (command === 'run') {
  await runCandidates();
  await printStatus();
} else {
  await printStatus();
}

async function materializeDocuments() {
  const [ledger, selection, screen] = await Promise.all([
    readJson(paths.ledger),
    readJson(paths.selection),
    readJson(paths.screen),
  ]);
  validatePlan(ledger, screen);
  const selected = new Map(selection.sources.flatMap((source) =>
    source.passages.map((passage) => [passage.id, { source, passage }])));
  const sttUnits = new Map(ledger.stt_units.map((unit) => [unit.id, unit]));
  let completed = 0;
  let resumed = 0;
  const documents = ledger.llm_documents.filter((document) =>
    document.status === screen.scope.document_status &&
    screen.scope.tts_engines.includes(document.tts_engine) &&
    (!values.locale || document.locale === values.locale));
  const limit = parsePositiveInteger(values.limit, '--limit');
  const selectedDocuments = limit === null ? documents : documents.slice(0, limit);
  await mkdir(paths.documents, { recursive: true });
  for (const document of selectedDocuments) {
    const output = join(paths.documents, `${document.id}.json`);
    if (!values.force && await validDocument(output, document)) {
      resumed += 1;
      continue;
    }
    const sections = [];
    for (const section of document.sections) {
      const unit = sttUnits.get(section.stt_unit_id);
      if (!unit) throw new Error(`${section.stt_unit_id}: missing ledger unit`);
      const manifestPath = join(paths.stt, unit.id, 'manifest.json');
      const manifest = await readJson(manifestPath);
      const transcriptPath = join(paths.stt, unit.id, manifest.transcript.path);
      const transcriptBytes = await readFile(transcriptPath);
      const audioManifest = await readJson(join(
        paths.audio,
        unit.audio_unit_id,
        'manifest.json',
      ));
      if (
        manifest.id !== unit.id ||
        manifest.plan_revision !== screen.plan_revision ||
        sha256(transcriptBytes) !== manifest.transcript.sha256 ||
        manifest.source_audio.normalized_sha256 !==
          audioManifest.normalized_audio.sha256
      ) {
        throw new Error(`${unit.id}: invalid STT checkpoint`);
      }
      const source = selected.get(section.passage_id);
      if (!source) throw new Error(`${section.passage_id}: missing source`);
      const reference = (await readFile(
        join(paths.text, `${section.passage_id}.txt`),
        'utf8',
      )).trim();
      if (sha256(Buffer.from(reference)) !== source.passage.spoken_sha256) {
        throw new Error(`${section.passage_id}: reference digest drift`);
      }
      sections.push({
        id: section.section_id,
        content_type: section.content_type,
        passage_id: section.passage_id,
        source_id: source.source.id,
        source_license: source.source.license,
        transcript: manifest.transcript.text,
        transcript_sha256: manifest.transcript.sha256,
        reference,
        reference_sha256: source.passage.spoken_sha256,
        raw_wer: manifest.quality.wer,
        raw_cer: manifest.quality.cer,
        normalized_audio_sha256: manifest.source_audio.normalized_sha256,
      });
    }
    const modelSections = sections.map((section) => ({
      id: section.id,
      text: section.transcript,
    }));
    const modelInput = {
      language: languageName(document.locale),
      locale: document.locale,
      domain: 'clean synthetic professional-style technical prose, factual prose, and dialogue',
      asr_backend: document.stt_model,
      error_profile: 'Clean synthetic speech may still contain substitutions, omissions, insertions, word-boundary errors, punctuation loss, abbreviation expansion, or end-of-passage hallucination. These are possibilities, not proof that a span is wrong.',
      glossary: [],
      sections: modelSections,
    };
    const artifact = {
      schema_version: '1.0.0',
      id: document.id,
      screen_id: screen.id,
      plan_id: ledger.plan_id,
      plan_revision: ledger.plan_revision,
      source_selection_revision: ledger.source_selection_revision,
      captured_at: new Date().toISOString(),
      dimensions: {
        locale: document.locale,
        tts_engine: document.tts_engine,
        voice_slot_id: document.voice_slot_id,
        realization_id: document.realization_id,
        stt_model: document.stt_model,
      },
      model_input: modelInput,
      model_input_sha256: sha256(Buffer.from(JSON.stringify(modelInput))),
      evaluation: {
        reference_visible_to_model: false,
        sections: sections.map((section) => ({
          id: section.id,
          content_type: section.content_type,
          passage_id: section.passage_id,
          source_id: section.source_id,
          source_license: section.source_license,
          reference: section.reference,
          reference_sha256: section.reference_sha256,
          raw_wer: section.raw_wer,
          raw_cer: section.raw_cer,
          transcript_sha256: section.transcript_sha256,
          normalized_audio_sha256: section.normalized_audio_sha256,
        })),
      },
      provenance: {
        stt_unit_ids: document.sections.map((section) => section.stt_unit_id),
        materializer: 'scripts/run-postprocessing-factorial-local-llm.mjs',
      },
    };
    await atomicJson(output, artifact);
    completed += 1;
    process.stderr.write(
      `llm document: ${document.id} (${completed + resumed}/${selectedDocuments.length})\n`,
    );
  }
  process.stdout.write(
    `LLM documents completed ${completed}, resumed ${resumed}, selected ${selectedDocuments.length}\n`,
  );
}

async function runCandidates() {
  const [ledger, screen] = await Promise.all([
    readJson(paths.ledger),
    readJson(paths.screen),
  ]);
  validatePlan(ledger, screen);
  const limit = parsePositiveInteger(values.limit, '--limit');
  const repetitions = parsePositiveInteger(values.repetitions, '--repetitions');
  let documents = ledger.llm_documents.filter((document) =>
    document.status === screen.scope.document_status &&
    screen.scope.tts_engines.includes(document.tts_engine) &&
    (!values.locale || document.locale === values.locale));
  if (limit !== null) documents = documents.slice(0, limit);
  const jobs = [];
  for (const document of documents) {
    const documentPath = join(paths.documents, `${document.id}.json`);
    if (!await validDocument(documentPath, document)) {
      throw new Error(`${document.id}: materialize valid LLM documents first`);
    }
    jobs.push({ document_id: document.id, document_path: documentPath });
  }
  await atomicJson(paths.jobs, { schema_version: '1.0.0', jobs });
  const candidates = values.candidate === 'all'
    ? screen.candidates
    : screen.candidates.filter((candidate) => candidate.id === values.candidate);
  if (candidates.length === 0) {
    throw new Error(`Unknown local candidate: ${values.candidate}`);
  }
  for (const candidate of candidates) {
    const manifestPath = join(repoRoot, candidate.model_manifest_path);
    const modelDirectory = join(paths.modelBase, candidate.model_directory_name);
    const outputDirectory = join(paths.results, candidate.id);
    for (const required of [paths.python, manifestPath, modelDirectory]) {
      if (!await exists(required)) throw new Error(`Missing local input: ${required}`);
    }
    const arguments_ = [
      join(repoRoot, 'spikes/text-generation-mlx-reference/run_factorial_batch.py'),
      '--screen', paths.screen,
      '--manifest', manifestPath,
      '--model-dir', modelDirectory,
      '--prompt', join(repoRoot, screen.generation.prompt_path),
      '--jobs', paths.jobs,
      '--output-dir', outputDirectory,
      '--source-revision', gitRevision(),
    ];
    if (repetitions !== null) {
      arguments_.push('--repetitions', String(repetitions));
    }
    commandOutput(paths.python, arguments_, {
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: 'inherit',
    });
  }
}

async function printStatus() {
  const [ledger, screen] = await Promise.all([
    readJson(paths.ledger),
    readJson(paths.screen),
  ]);
  validatePlan(ledger, screen);
  const ready = ledger.llm_documents.filter((document) =>
    document.status === screen.scope.document_status &&
    screen.scope.tts_engines.includes(document.tts_engine));
  let materialized = 0;
  for (const document of ready) {
    if (await validDocument(join(paths.documents, `${document.id}.json`), document)) {
      materialized += 1;
    }
  }
  const candidateResults = [];
  for (const candidate of screen.candidates) {
    let completed = 0;
    let accepted = 0;
    let invalid = 0;
    for (const document of ready) {
      for (let repetition = 1;
        repetition <= screen.generation.repetitions;
        repetition += 1) {
        const path = join(
          paths.results,
          candidate.id,
          document.id,
          `repeat-${repetition}.json`,
        );
        try {
          const result = await readJson(path);
          if (
            result.candidate.manifest_id === candidate.id &&
            result.document.id === document.id &&
            result.repetition === repetition
          ) {
            completed += 1;
            if (result.output.mechanically_accepted) accepted += 1;
            else invalid += 1;
          }
        } catch (error) {
          if (error.code !== 'ENOENT') throw error;
        }
      }
    }
    candidateResults.push({
      candidate: candidate.id,
      expected_requests: ready.length * screen.generation.repetitions,
      completed_requests: completed,
      mechanically_accepted: accepted,
      invalid_or_capped: invalid,
    });
  }
  process.stdout.write(`${JSON.stringify({
    screen_id: screen.id,
    plan_revision: screen.plan_revision,
    ready_documents: ready.length,
    materialized_documents: materialized,
    blocked_documents: ledger.llm_documents.length - ready.length,
    candidate_results: candidateResults,
  }, null, 2)}\n`);
}

function validatePlan(ledger, screen) {
  if (
    ledger.plan_id !== screen.plan_id ||
    ledger.plan_revision !== screen.plan_revision ||
    screen.scope.document_count !== 120 ||
    screen.expected_counts.documents !== 120 ||
    screen.expected_counts.requests !== 720
  ) {
    throw new Error('Local LLM screen differs from the pinned factorial ledger');
  }
}

async function validDocument(path, expected) {
  try {
    const document = await readJson(path);
    return document.id === expected.id &&
      document.plan_revision === (await readJson(paths.screen)).plan_revision &&
      document.model_input.sections.length === 6 &&
      document.model_input.sections.every((section, index) =>
        section.id === expected.sections[index].section_id &&
        typeof section.text === 'string' && section.text.length > 0);
  } catch {
    return false;
  }
}

function languageName(locale) {
  return ({
    'de-DE': 'German',
    'en-US': 'English',
    'es-419': 'Spanish',
    'fr-FR': 'French',
    'pt-BR': 'Portuguese',
  })[locale] ?? locale;
}

function parsePositiveInteger(value, option) {
  if (value === undefined) return null;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${option} must be a positive integer`);
  }
  return parsed;
}

function gitRevision() {
  return commandOutput('git', ['rev-parse', 'HEAD']).trim();
}

function commandOutput(commandName, arguments_, options = {}) {
  const result = spawnSync(commandName, arguments_, {
    cwd: repoRoot,
    encoding: options.stdio === 'inherit' ? undefined : 'utf8',
    maxBuffer: 64 * 1024 * 1024,
    ...options,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `${commandName} ${arguments_.join(' ')} failed (${result.status})\n` +
      `${result.stdout ?? ''}${result.stderr ?? ''}`,
    );
  }
  return result.stdout ?? '';
}

async function atomicJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporary, path);
}

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function selfTest() {
  if (
    languageName('pt-BR') !== 'Portuguese' ||
    parsePositiveInteger(undefined, '--limit') !== null ||
    parsePositiveInteger('2', '--limit') !== 2
  ) {
    throw new Error('local LLM runner self-test failed');
  }
  process.stdout.write('factorial local LLM runner: self-test passed\n');
}
