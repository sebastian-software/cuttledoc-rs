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
    'summary-output': {
      type: 'string',
      default: join(
        repoRoot,
        'benchmarks/postprocessing/local-llm-screen-results.json',
      ),
    },
    'results-subdir': {
      type: 'string',
      default: 'llm-local-results',
    },
    'allow-partial': { type: 'boolean', default: false },
    force: { type: 'boolean', default: false },
  },
});

const command = positionals[0];
if (!['materialize', 'run', 'summarize', 'status', 'self-test'].includes(command)) {
  throw new Error(
    'usage: node scripts/run-postprocessing-factorial-local-llm.mjs ' +
    '<materialize|run|summarize|status|self-test> [options]',
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
paths.results = join(paths.output, values['results-subdir']);
paths.jobs = join(paths.output, `${values['results-subdir']}-jobs.json`);

if (command === 'self-test') {
  await selfTest();
} else if (command === 'materialize') {
  await materializeDocuments();
  await printStatus();
} else if (command === 'run') {
  await runCandidates();
  await printStatus();
} else if (command === 'summarize') {
  await summarizeResults();
} else {
  await printStatus();
}

async function materializeDocuments() {
  const [ledger, selection, screen] = await Promise.all([
    readJson(paths.ledger),
    readJson(paths.selection),
    readJson(paths.screen),
  ]);
  await validatePlan(ledger, screen);
  const selected = new Map(selection.sources.flatMap((source) =>
    source.passages.map((passage) => [passage.id, { source, passage }])));
  const sttUnits = new Map(ledger.stt_units.map((unit) => [unit.id, unit]));
  let completed = 0;
  let resumed = 0;
  const documents = ledger.llm_documents.filter((document) =>
    screenIncludesDocument(screen, document) &&
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
  await validatePlan(ledger, screen);
  const limit = parsePositiveInteger(values.limit, '--limit');
  const repetitions = parsePositiveInteger(values.repetitions, '--repetitions');
  let documents = ledger.llm_documents.filter((document) =>
    screenIncludesDocument(screen, document) &&
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
    if (values.force) arguments_.push('--force');
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
  await validatePlan(ledger, screen);
  const ready = ledger.llm_documents.filter((document) =>
    screenIncludesDocument(screen, document));
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

async function summarizeResults() {
  const [ledger, screen] = await Promise.all([
    readJson(paths.ledger),
    readJson(paths.screen),
  ]);
  await validatePlan(ledger, screen);
  const documents = ledger.llm_documents.filter((document) =>
    screenIncludesDocument(screen, document));
  const requests = [];
  const observations = [];
  const missing = [];
  const sourceRevisions = new Set();
  const capturedAt = [];
  for (const candidate of screen.candidates) {
    for (const expectedDocument of documents) {
      const documentPath = join(paths.documents, `${expectedDocument.id}.json`);
      if (!await validDocument(documentPath, expectedDocument)) {
        throw new Error(`${expectedDocument.id}: invalid materialized document`);
      }
      const document = await readJson(documentPath);
      for (let repetition = 1;
        repetition <= screen.generation.repetitions;
        repetition += 1) {
        const resultPath = join(
          paths.results,
          candidate.id,
          expectedDocument.id,
          `repeat-${repetition}.json`,
        );
        let result;
        try {
          result = await readJson(resultPath);
        } catch (error) {
          if (error.code === 'ENOENT') {
            missing.push({
              candidate: candidate.id,
              document_id: expectedDocument.id,
              repetition,
            });
            continue;
          }
          throw error;
        }
        if (
          result.candidate.manifest_id !== candidate.id ||
          result.document.id !== expectedDocument.id ||
          result.repetition !== repetition ||
          result.screen_id !== screen.id
        ) {
          throw new Error(`${resultPath}: result identity drift`);
        }
        sourceRevisions.add(result.source_revision);
        capturedAt.push(result.captured_at);
        const strictValid = result.output.parser.valid === true &&
          result.output.mechanically_accepted === true;
        const diagnostic = diagnosticSections(
          result.output.raw_text,
          document.model_input.sections.map((section) => section.id),
        );
        const outputSections = strictValid
          ? result.output.sections
          : diagnostic.sections;
        const formatViolation = !strictValid && diagnostic.valid;
        requests.push({
          candidate: candidate.id,
          document_id: expectedDocument.id,
          locale: document.dimensions.locale,
          tts_engine: document.dimensions.tts_engine,
          voice_slot_id: document.dimensions.voice_slot_id,
          realization_id: document.dimensions.realization_id,
          stt_model: document.dimensions.stt_model,
          repetition,
          mechanically_accepted: strictValid,
          format_violation_with_recoverable_sections: formatViolation,
          diagnostic_recovery_class: strictValid ? null : diagnostic.recovery,
          parser_error: result.output.parser.error,
          reached_token_limit: result.generation.reached_token_limit,
          raw_text_sha256: result.output.raw_text_sha256,
          prompt_tokens: result.measurements.prompt_tokens,
          generation_tokens: result.measurements.generation_tokens,
          complete_generation_ms: result.measurements.complete_generation_ms,
          generation_tokens_per_second:
            result.measurements.generation_tokens_per_second,
        });
        for (let index = 0;
          index < document.model_input.sections.length;
          index += 1) {
          const input = document.model_input.sections[index];
          const evaluation = document.evaluation.sections[index];
          const output = outputSections[index]?.text ?? null;
          const referenceWords = normalizedWords(evaluation.reference);
          const inputWords = normalizedWords(input.text);
          const rawDistance = editDistance(referenceWords, inputWords);
          const outputWords = output === null ? null : normalizedWords(output);
          const postDistance = outputWords === null
            ? null
            : editDistance(referenceWords, outputWords);
          const inputOutputDistance = outputWords === null
            ? null
            : editDistance(inputWords, outputWords);
          observations.push({
            candidate: candidate.id,
            document_id: expectedDocument.id,
            repetition,
            locale: document.dimensions.locale,
            tts_engine: document.dimensions.tts_engine,
            voice_slot_id: document.dimensions.voice_slot_id,
            realization_id: document.dimensions.realization_id,
            stt_model: document.dimensions.stt_model,
            section_id: input.id,
            content_type: evaluation.content_type,
            passage_id: evaluation.passage_id,
            source_id: evaluation.source_id,
            mechanically_accepted: strictValid,
            diagnostic_output_available: output !== null,
            format_violation_with_recoverable_sections: formatViolation,
            reference_word_count: referenceWords.length,
            input_word_count: inputWords.length,
            raw_word_edit_distance: rawDistance,
            raw_wer: rawDistance / referenceWords.length,
            post_word_edit_distance: postDistance,
            post_wer: strictValid && postDistance !== null
              ? postDistance / referenceWords.length
              : null,
            diagnostic_post_wer: postDistance === null
              ? null
              : postDistance / referenceWords.length,
            input_output_word_edit_distance: inputOutputDistance,
            input_output_word_edit_rate: inputOutputDistance === null
              ? null
              : inputOutputDistance / Math.max(1, inputWords.length),
            exact_text_unchanged: output === input.text,
            presentation_only_change: output !== null &&
              output !== input.text && inputOutputDistance === 0,
            improved: postDistance !== null && postDistance < rawDistance,
            regressed: postDistance !== null && postDistance > rawDistance,
            changed_correct_input_to_error: rawDistance === 0 &&
              postDistance !== null && postDistance > 0,
          });
        }
      }
    }
  }
  if (missing.length > 0 && !values['allow-partial']) {
    throw new Error(
      `${missing.length} local LLM request(s) are missing; ` +
      'use --allow-partial only for a diagnostic report',
    );
  }
  const repeatGroups = new Map();
  for (const request of requests) {
    const key = `${request.candidate}|${request.document_id}`;
    if (!repeatGroups.has(key)) repeatGroups.set(key, []);
    repeatGroups.get(key).push(request);
  }
  let completePairs = 0;
  let identicalPairs = 0;
  for (const group of repeatGroups.values()) {
    if (group.length !== screen.generation.repetitions) continue;
    completePairs += 1;
    if (new Set(group.map((request) => request.raw_text_sha256)).size === 1) {
      identicalPairs += 1;
    }
  }
  const report = {
    schema_version: '1.0.0',
    id: `${screen.id}-results-1`,
    screen_id: screen.id,
    plan_id: screen.plan_id,
    plan_revision: screen.plan_revision,
    evidence_source_revisions: [...sourceRevisions].sort(),
    evidence_captured_through: capturedAt.sort().at(-1) ?? null,
    status: missing.length === 0 ? 'complete' : 'partial',
    scope: {
      expected_documents: documents.length,
      candidates: screen.candidates.length,
      repetitions: screen.generation.repetitions,
      expected_requests: screen.expected_counts.requests,
      completed_requests: requests.length,
      missing_requests: missing.length,
      section_observations: observations.length,
    },
    contract_compliance: {
      mechanically_accepted_requests: requests.filter((request) =>
        request.mechanically_accepted).length,
      invalid_requests: requests.filter((request) =>
        !request.mechanically_accepted).length,
      recoverable_markdown_or_format_violations: requests.filter((request) =>
        request.format_violation_with_recoverable_sections).length,
      recoverable_markdown_fence_violations: requests.filter((request) =>
        request.diagnostic_recovery_class?.includes('markdown-fence')).length,
      recoverable_section_id_normalization_violations: requests.filter((request) =>
        request.diagnostic_recovery_class?.includes(
          'section-id-underscore-normalization',
        )).length,
      token_limit_failures: requests.filter((request) =>
        request.reached_token_limit).length,
    },
    repeat_stability: {
      expected_pairs: documents.length * screen.candidates.length,
      complete_pairs: completePairs,
      byte_identical_raw_output_pairs: identicalPairs,
    },
    exploratory_safety_gate_sensitivity: buildSafetyGateSensitivity(
      observations,
      [0.02, 0.05, 0.1, 0.2, 0.3],
    ),
    aggregates: {
      by_candidate: aggregateObservations(observations, ['candidate'], requests),
      by_candidate_and_locale: aggregateObservations(
        observations,
        ['candidate', 'locale'],
        requests,
      ),
      by_candidate_and_stt_model: aggregateObservations(
        observations,
        ['candidate', 'stt_model'],
        requests,
      ),
      by_candidate_and_tts_engine: aggregateObservations(
        observations,
        ['candidate', 'tts_engine'],
        requests,
      ),
      by_candidate_and_content_type: aggregateObservations(
        observations,
        ['candidate', 'content_type'],
        requests,
      ),
    },
    requests,
    observations,
    missing,
    limitations: [
      screen.claim_limit,
      'Strict contract failures are excluded from canonical postprocessed WER.',
      'Diagnostic WER for structurally recoverable fenced JSON remains noncompliant evidence and is never promoted to an accepted result.',
      'The edit-rate safety-gate sweep is evaluated on the same synthetic development observations and must not be treated as a selected product threshold.',
      'Synthetic references can identify restoration and regressions but do not replace held-out human-verified professional audio.',
    ],
  };
  validateSummaryReport(report, screen);
  const output = resolve(values['summary-output']);
  await atomicJson(output, report);
  process.stdout.write(
    `Wrote ${relative(repoRoot, output)}: ${requests.length}/` +
    `${screen.expected_counts.requests} requests, ${report.status}\n`,
  );
}

function diagnosticSections(rawText, expectedIds) {
  const trimmed = rawText.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/iu);
  const candidates = fenced
    ? [
        { text: trimmed, recovery: null },
        { text: fenced[1], recovery: 'markdown-fence' },
      ]
    : [{ text: trimmed, recovery: null }];
  for (const candidate of candidates) {
    try {
      const value = JSON.parse(candidate.text);
      if (!Array.isArray(value.sections)) continue;
      const ids = value.sections.map((section) => section?.id);
      if (
        ids.length === expectedIds.length &&
        ids.every((id, index) => id === expectedIds[index]) &&
        value.sections.every((section) => typeof section.text === 'string')
      ) {
        return {
          valid: true,
          sections: value.sections,
          recovery: candidate.recovery,
        };
      }
      const underscoreNormalized = ids.map((id) =>
        typeof id === 'string' ? id.replaceAll('_', '-') : id);
      if (
        ids.length === expectedIds.length &&
        underscoreNormalized.every((id, index) => id === expectedIds[index]) &&
        value.sections.every((section) => typeof section.text === 'string')
      ) {
        return {
          valid: true,
          sections: value.sections,
          recovery: [
            candidate.recovery,
            'section-id-underscore-normalization',
          ].filter(Boolean).join('+'),
        };
      }
    } catch {
      // Try the optional format-only recovery candidate.
    }
  }
  return { valid: false, sections: [], recovery: null };
}

function buildSafetyGateSensitivity(observations, thresholds) {
  const summarize = (group, threshold) => {
    const usePostprocessed = (item) =>
      item.mechanically_accepted &&
      item.post_word_edit_distance !== null &&
      item.input_output_word_edit_rate !== null &&
      item.input_output_word_edit_rate <= threshold;
    const referenceWords = group.reduce(
      (sum, item) => sum + item.reference_word_count,
      0,
    );
    const rawDistance = group.reduce(
      (sum, item) => sum + item.raw_word_edit_distance,
      0,
    );
    const gatedDistance = group.reduce(
      (sum, item) => sum + (
        usePostprocessed(item)
          ? item.post_word_edit_distance
          : item.raw_word_edit_distance
      ),
      0,
    );
    const postprocessed = group.filter(usePostprocessed);
    return {
      section_count: group.length,
      postprocessed_section_count: postprocessed.length,
      raw_fallback_section_count: group.length - postprocessed.length,
      micro_raw_wer: rawDistance / referenceWords,
      micro_gated_wer: gatedDistance / referenceWords,
      relative_error_change: rawDistance === 0
        ? null
        : (gatedDistance - rawDistance) / rawDistance,
      applied_improvement_count: postprocessed.filter((item) =>
        item.improved).length,
      applied_regression_count: postprocessed.filter((item) =>
        item.regressed).length,
      applied_correct_input_to_error_count: postprocessed.filter((item) =>
        item.changed_correct_input_to_error).length,
    };
  };
  return {
    policy:
      'Use postprocessed text only for a strictly valid request whose ' +
      'normalized input-to-output word edit rate is at or below the threshold; ' +
      'otherwise retain the raw section.',
    selection_status: 'exploratory-development-sensitivity-only',
    thresholds: thresholds.map((threshold) => {
      const byLocale = new Map();
      for (const observation of observations) {
        if (!byLocale.has(observation.locale)) {
          byLocale.set(observation.locale, []);
        }
        byLocale.get(observation.locale).push(observation);
      }
      return {
        maximum_input_output_word_edit_rate: threshold,
        overall: summarize(observations, threshold),
        by_locale: [...byLocale.entries()].map(([locale, group]) => ({
          locale,
          ...summarize(group, threshold),
        })).sort((left, right) => left.locale.localeCompare(right.locale)),
      };
    }),
  };
}

function validateSummaryReport(report, screen) {
  const errors = [];
  if (
    report.screen_id !== screen.id ||
    report.plan_id !== screen.plan_id ||
    report.plan_revision !== screen.plan_revision
  ) {
    errors.push('report identity differs from the screen');
  }
  if (
    report.status !== 'complete' ||
    report.scope.expected_documents !== screen.expected_counts.documents ||
    report.scope.expected_requests !== screen.expected_counts.requests ||
    report.scope.completed_requests !== screen.expected_counts.requests ||
    report.scope.missing_requests !== 0 ||
    report.scope.section_observations !==
      screen.expected_counts.section_outputs ||
    report.requests.length !== screen.expected_counts.requests ||
    report.observations.length !== screen.expected_counts.section_outputs ||
    report.missing.length !== 0
  ) {
    errors.push('complete report counts differ from the screen');
  }
  const acceptedRequests = report.requests.filter((request) =>
    request.mechanically_accepted).length;
  if (
    report.contract_compliance.mechanically_accepted_requests !==
      acceptedRequests ||
    report.contract_compliance.invalid_requests !==
      report.requests.length - acceptedRequests ||
    report.repeat_stability.expected_pairs !==
      screen.expected_counts.document_model_pairs ||
    report.repeat_stability.complete_pairs !==
      screen.expected_counts.document_model_pairs
  ) {
    errors.push('contract or repeat counts differ from request evidence');
  }
  for (const observation of report.observations) {
    const expectedRate = observation.input_output_word_edit_distance === null
      ? null
      : observation.input_output_word_edit_distance /
        Math.max(1, observation.input_word_count);
    if (
      observation.input_word_count < 1 ||
      (expectedRate === null
        ? observation.input_output_word_edit_rate !== null
        : Math.abs(
          expectedRate - observation.input_output_word_edit_rate,
        ) > 1e-15)
    ) {
      errors.push(`${observation.document_id}: invalid input/output edit rate`);
      break;
    }
  }
  const thresholds =
    report.exploratory_safety_gate_sensitivity?.thresholds?.map((entry) =>
      entry.maximum_input_output_word_edit_rate);
  const expectedSensitivity = Array.isArray(thresholds)
    ? buildSafetyGateSensitivity(report.observations, thresholds)
    : null;
  if (
    expectedSensitivity === null ||
    JSON.stringify(report.exploratory_safety_gate_sensitivity) !==
      JSON.stringify(expectedSensitivity)
  ) {
    errors.push('safety-gate sensitivity differs from section observations');
  }
  if (errors.length > 0) {
    throw new Error(`invalid local LLM summary: ${errors.join('; ')}`);
  }
}

function aggregateObservations(observations, fields, requests) {
  const groups = new Map();
  for (const observation of observations) {
    const key = fields.map((field) => observation[field]).join('|');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(observation);
  }
  return [...groups.values()].map((group) => {
    const identity = Object.fromEntries(fields.map((field) => [
      field,
      group[0][field],
    ]));
    const accepted = group.filter((item) => item.post_wer !== null);
    const diagnostic = group.filter((item) => item.diagnostic_post_wer !== null);
    const requestKey = (item) =>
      `${item.candidate}|${item.document_id}|${item.repetition}`;
    const requestKeys = new Set(group.map(requestKey));
    const acceptedRequestKeys = new Set(group.filter((item) =>
      item.mechanically_accepted).map(requestKey));
    const diagnosticRequestKeys = new Set(group.filter((item) =>
      item.diagnostic_output_available).map(requestKey));
    const requestByKey = new Map(requests.map((request) => [
      requestKey(request),
      request,
    ]));
    return {
      ...identity,
      request_count: requestKeys.size,
      accepted_request_count: acceptedRequestKeys.size,
      diagnostic_request_count: diagnosticRequestKeys.size,
      invalid_request_count: requestKeys.size - acceptedRequestKeys.size,
      token_limit_failure_count: [...requestKeys].filter((key) =>
        requestByKey.get(key)?.reached_token_limit).length,
      section_count: group.length,
      accepted_section_count: accepted.length,
      diagnostic_section_count: diagnostic.length,
      macro_mean_raw_wer: mean(group.map((item) => item.raw_wer)),
      macro_mean_accepted_raw_wer: accepted.length === 0
        ? null
        : mean(accepted.map((item) => item.raw_wer)),
      macro_mean_post_wer: accepted.length === 0
        ? null
        : mean(accepted.map((item) => item.post_wer)),
      macro_mean_diagnostic_raw_wer: diagnostic.length === 0
        ? null
        : mean(diagnostic.map((item) => item.raw_wer)),
      macro_mean_diagnostic_post_wer: diagnostic.length === 0
        ? null
        : mean(diagnostic.map((item) => item.diagnostic_post_wer)),
      improved_section_count: diagnostic.filter((item) => item.improved).length,
      regressed_section_count: diagnostic.filter((item) => item.regressed).length,
      changed_correct_input_to_error_count: diagnostic.filter((item) =>
        item.changed_correct_input_to_error).length,
      exact_text_unchanged_section_count: diagnostic.filter((item) =>
        item.exact_text_unchanged).length,
      presentation_only_change_count: diagnostic.filter((item) =>
        item.presentation_only_change).length,
    };
  }).sort((left, right) => fields.map((field) => left[field]).join('|')
    .localeCompare(fields.map((field) => right[field]).join('|')));
}

function screenDocumentStatuses(screen) {
  return screen.scope.document_statuses ??
    [screen.scope.document_status];
}

function screenIncludesDocument(screen, document) {
  return screenDocumentStatuses(screen).includes(document.status) &&
    screen.scope.tts_engines.includes(document.tts_engine);
}

async function validatePlan(ledger, screen) {
  const statuses = screenDocumentStatuses(screen);
  const documents = ledger.llm_documents.filter((document) =>
    screenIncludesDocument(screen, document));
  const expectedPairs = documents.length * screen.candidates.length;
  const expectedRequests = expectedPairs * screen.generation.repetitions;
  const expectedSections =
    expectedRequests * screen.scope.sections_per_document;
  if (
    ledger.plan_id !== screen.plan_id ||
    ledger.plan_revision !== screen.plan_revision ||
    !Array.isArray(statuses) ||
    statuses.length === 0 ||
    new Set(statuses).size !== statuses.length ||
    !Array.isArray(screen.scope.tts_engines) ||
    screen.scope.tts_engines.length === 0 ||
    screen.scope.document_count !== documents.length ||
    screen.expected_counts.documents !== documents.length ||
    screen.expected_counts.document_model_pairs !== expectedPairs ||
    screen.expected_counts.requests !== expectedRequests ||
    screen.expected_counts.section_outputs !== expectedSections
  ) {
    throw new Error('Local LLM screen differs from the pinned factorial ledger');
  }
  const promptBytes = await readFile(join(repoRoot, screen.generation.prompt_path));
  if (sha256(promptBytes) !== screen.generation.prompt_sha256) {
    throw new Error(`${screen.generation.prompt_path}: prompt digest drift`);
  }
  if (
    new Set(screen.candidates.map((candidate) => candidate.id)).size !==
      screen.candidates.length
  ) {
    throw new Error('Local LLM candidate ids must be unique');
  }
  const qualificationReports = new Map();
  for (const qualification of screen.scope.qualification_reports ?? []) {
    const path = join(repoRoot, qualification.path);
    const bytes = await readFile(path);
    if (sha256(bytes) !== qualification.sha256) {
      throw new Error(`${qualification.path}: qualification digest drift`);
    }
    const report = JSON.parse(bytes);
    if (
      report.plan_id !== screen.plan_id ||
      report.plan_revision !== screen.plan_revision ||
      report.qualification_slice?.tts_engine !== qualification.tts_engine ||
      report.disposition !== qualification.required_disposition
    ) {
      throw new Error(
        `${qualification.path}: qualification contract differs from screen`,
      );
    }
    qualificationReports.set(qualification.tts_engine, report);
  }
  for (const document of documents) {
    if (
      document.status !== 'ready' &&
      !qualificationReports.has(document.tts_engine)
    ) {
      throw new Error(
        `${document.id}: blocked document lacks a pinned qualification report`,
      );
    }
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

function normalizedWords(text) {
  return [...text.normalize('NFC').toLocaleLowerCase('und')]
    .map((character) => /[\p{L}\p{N}]/u.test(character) ? character : ' ')
    .join('')
    .split(/\s+/u)
    .filter(Boolean);
}

function editDistance(reference, hypothesis) {
  let previous = Array.from(
    { length: hypothesis.length + 1 },
    (_, index) => index,
  );
  for (let referenceIndex = 1;
    referenceIndex <= reference.length;
    referenceIndex += 1) {
    const current = [referenceIndex];
    for (let hypothesisIndex = 1;
      hypothesisIndex <= hypothesis.length;
      hypothesisIndex += 1) {
      current[hypothesisIndex] = Math.min(
        previous[hypothesisIndex] + 1,
        current[hypothesisIndex - 1] + 1,
        previous[hypothesisIndex - 1] +
          (reference[referenceIndex - 1] ===
            hypothesis[hypothesisIndex - 1] ? 0 : 1),
      );
    }
    previous = current;
  }
  return previous.at(-1);
}

function mean(values_) {
  return values_.reduce((sum, value) => sum + value, 0) / values_.length;
}

async function selfTest() {
  if (
    languageName('pt-BR') !== 'Portuguese' ||
    parsePositiveInteger(undefined, '--limit') !== null ||
    parsePositiveInteger('2', '--limit') !== 2 ||
    editDistance(['one', 'two'], ['one', 'too']) !== 1 ||
    normalizedWords('Über—Test').join('|') !== 'über|test' ||
    buildSafetyGateSensitivity([
      {
        locale: 'en-US',
        mechanically_accepted: true,
        reference_word_count: 2,
        raw_word_edit_distance: 1,
        post_word_edit_distance: 0,
        input_output_word_edit_rate: 0.5,
        improved: true,
        regressed: false,
        changed_correct_input_to_error: false,
      },
    ], [0.3]).thresholds[0].overall.raw_fallback_section_count !== 1
  ) {
    throw new Error('local LLM runner self-test failed');
  }
  const ledger = await readJson(join(
    repoRoot,
    'benchmarks/postprocessing/factorial-cells.json',
  ));
  for (const screenPath of [
    'benchmarks/postprocessing/local-llm-screen.json',
    'benchmarks/postprocessing/local-llm-gemma-contract-screen.json',
  ]) {
    await validatePlan(ledger, await readJson(join(repoRoot, screenPath)));
  }
  const contractScreen = await readJson(join(
    repoRoot,
    'benchmarks/postprocessing/local-llm-gemma-contract-screen.json',
  ));
  validateSummaryReport(
    await readJson(join(
      repoRoot,
      'benchmarks/postprocessing/local-llm-gemma-contract-results.json',
    )),
    contractScreen,
  );
  process.stdout.write('factorial local LLM runner: self-test passed\n');
}
