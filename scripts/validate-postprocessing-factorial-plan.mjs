#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const planPath = 'benchmarks/postprocessing/factorial-plan.json';
const matrixPath = 'benchmarks/postprocessing/factorial-cells.json';

async function readJson(path) {
  return JSON.parse(await readFile(resolve(repoRoot, path), 'utf8'));
}

function same(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function unique(values) {
  return values.length === new Set(values).size;
}

function near(actual, expected, tolerance = 1e-9) {
  return Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance;
}

function addError(errors, condition, message) {
  if (!condition) errors.push(message);
}

function selectionPassages(selection) {
  return new Map(selection.sources.flatMap((source) =>
    source.passages.map((passage) => [passage.id, { source, passage }])));
}

async function validatePlan(plan, selection, promptManifest) {
  const errors = [];
  const expectedLocales = ['de-DE', 'en-US', 'es-419', 'fr-FR', 'pt-BR'];
  const expectedContentTypes = ['technical', 'native-factual', 'dialogue'];
  const expectedTtsEngines = [
    'apple-avspeechsynthesizer',
    'qwen3-tts-1.7b-voicedesign-mlx-audio',
    'voxtral-tts-4b-bf16-mlx-audio',
  ];
  const expectedSttModels = [
    'whisper-large-v3-turbo-coreml-whispercpp',
    'qwen3-asr-0.6b-mlx-direct',
    'parakeet-tdt-0.6b-v3-coreml',
  ];
  const expectedLlmModels = [
    'openai/gpt-5.6-sol',
    'google/gemini-3.6-flash',
    'openai/gpt-5.6-terra',
    'moonshotai/kimi-k3',
    'anthropic/claude-sonnet-5',
  ];

  addError(errors, plan.schema_version === '1.0.0',
    'schema_version must be 1.0.0');
  addError(errors, plan.status ===
    'design-locked-native-review-and-voice-qualification-required',
  'status must retain native-review and voice-qualification gates');
  addError(errors, /^\d{4}-\d{2}-\d{2}$/.test(plan.evidence_date ?? ''),
    'evidence_date must be an ISO date');
  addError(errors, plan.source_selection.path ===
    'benchmarks/fixtures/synthetic-roundtrip-selection.json',
  'source selection path must remain pinned');
  addError(errors, plan.source_selection.revision === selection.revision,
    'source selection revision must match the selected corpus');
  addError(errors, same(plan.axes.locales, expectedLocales),
    'locales must preserve the five primary-language cells');
  addError(errors, same(plan.axes.content_types, expectedContentTypes),
    'content types must be technical, native-factual, and dialogue');
  addError(errors, plan.axes.passages_per_content_type === 2,
    'every content type must have exactly two independent passages');
  addError(errors, same(plan.axes.tts_engines.map((item) => item.id),
    expectedTtsEngines), 'TTS engines differ from the locked three-engine set');
  addError(errors, plan.axes.voices_per_tts_engine_and_locale === 2,
    'every TTS engine and locale must have exactly two voice slots');
  addError(errors, same(plan.axes.stt_models.map((item) => item.id),
    expectedSttModels), 'STT models differ from the locked three-model set');
  addError(errors, same(plan.axes.llm_models.map((item) => item.id),
    expectedLlmModels), 'LLM models differ from the locked five-model set');
  addError(errors, plan.axes.llm_repetitions === 2,
    'every LLM document/model pair must have two repetitions');

  const passages = selectionPassages(selection);
  const passageSlots = plan.passage_slots ?? [];
  addError(errors, passageSlots.length === 30,
    'the plan must contain 30 passage slots');
  addError(errors, unique(passageSlots.map((slot) => slot.id)),
    'passage slot ids must be unique');
  for (const locale of expectedLocales) {
    for (const contentType of expectedContentTypes) {
      const cells = passageSlots.filter((slot) =>
        slot.locale === locale && slot.content_type === contentType);
      addError(errors, cells.length === 2,
        `${locale}/${contentType} must contain two passage slots`);
      addError(errors, same(cells.map((slot) => slot.index), [1, 2]),
        `${locale}/${contentType} passage indexes must be [1,2]`);
    }
  }
  for (const slot of passageSlots) {
    addError(errors, expectedLocales.includes(slot.locale),
      `${slot.id}: unknown locale`);
    addError(errors, expectedContentTypes.includes(slot.content_type),
      `${slot.id}: unknown content type`);
    if (slot.status === 'materialized') {
      addError(errors, typeof slot.passage_id === 'string' &&
        passages.has(slot.passage_id),
      `${slot.id}: materialized passage must exist in the source selection`);
    } else if (slot.status === 'source-selection-required') {
      addError(errors, slot.passage_id === null,
        `${slot.id}: pending passage must not claim a passage id`);
    } else {
      errors.push(`${slot.id}: invalid passage status`);
    }
    const expectedLanguageReview = ['de-DE', 'en-US'].includes(slot.locale)
      ? 'not-required'
      : 'native-review-required';
    addError(errors, slot.language_review_status === expectedLanguageReview,
      `${slot.id}: language review status differs from the execution policy`);
  }
  const materializedCount = passageSlots.filter((slot) =>
    slot.status === 'materialized').length;
  const pendingCount = passageSlots.length - materializedCount;
  addError(errors, materializedCount === 30 && pendingCount === 0,
    'the locked plan must materialize all 30 passage slots');
  addError(errors, plan.source_selection.materialized_passage_slots ===
    materializedCount && plan.source_selection.pending_passage_slots ===
    pendingCount, 'source selection passage counts do not reconcile');

  const voiceSlots = plan.voice_slots ?? [];
  addError(errors, voiceSlots.length === 30,
    'the plan must contain 30 voice slots');
  addError(errors, unique(voiceSlots.map((slot) => slot.id)),
    'voice slot ids must be unique');
  for (const locale of expectedLocales) {
    for (const engine of expectedTtsEngines) {
      const cells = voiceSlots.filter((slot) =>
        slot.locale === locale && slot.tts_engine === engine);
      addError(errors, cells.length === 2,
        `${locale}/${engine} must contain two voice slots`);
      addError(errors, same(cells.map((slot) => slot.index), [1, 2]),
        `${locale}/${engine} voice indexes must be [1,2]`);
    }
  }
  for (const voice of voiceSlots) {
    addError(errors, voice.status === 'qualified' ||
      /(required|pending|inventory)/.test(voice.status),
    `${voice.id}: status must be qualified or explicitly gated`);
    if (voice.tts_engine === 'qwen3-tts-1.7b-voicedesign-mlx-audio') {
      addError(errors, typeof voice.selector === 'string' &&
        Number.isInteger(voice.voice_identity_seed),
      `${voice.id}: Qwen voice identity requires instruction and seed`);
    }
    if (voice.tts_engine === 'voxtral-tts-4b-bf16-mlx-audio') {
      addError(errors, typeof voice.selector === 'string' &&
        voice.voice_identity_seed === null,
      `${voice.id}: Voxtral voice must use a preset without identity seed`);
    }
    if (voice.tts_engine === 'apple-avspeechsynthesizer') {
      addError(errors, voice.selector === null &&
        voice.voice_identity_seed === null,
      `${voice.id}: Apple voice must resolve from the host inventory`);
    }
  }
  for (const locale of expectedLocales) {
    const seeds = voiceSlots.filter((slot) =>
      slot.locale === locale &&
      slot.tts_engine === 'qwen3-tts-1.7b-voicedesign-mlx-audio')
      .map((slot) => slot.voice_identity_seed);
    addError(errors, same(seeds, [0, 1]),
      `${locale}: Qwen voice identity seeds must be [0,1]`);
  }

  const policies = new Map(plan.generation_sampling.engine_policies.map(
    (policy) => [policy.tts_engine, policy]));
  addError(errors, policies.size === expectedTtsEngines.length &&
    expectedTtsEngines.every((id) => policies.has(id)),
  'every TTS engine must have exactly one generation policy');
  const expectedPolicies = {
    'apple-avspeechsynthesizer': { seeds: [], repetitions: 2 },
    'qwen3-tts-1.7b-voicedesign-mlx-audio': { seeds: [], repetitions: 2 },
    'voxtral-tts-4b-bf16-mlx-audio': { seeds: [0, 1], repetitions: 1 },
  };
  for (const [id, expected] of Object.entries(expectedPolicies)) {
    const policy = policies.get(id);
    addError(errors, same(policy?.seeds, expected.seeds) &&
      policy?.repetitions_per_seed_or_voice === expected.repetitions,
    `${id}: generation seed/repetition semantics changed`);
    addError(errors, (policy?.seeds.length || 1) *
      policy?.repetitions_per_seed_or_voice === 2,
    `${id}: primary sampling must produce two realizations`);
  }
  const control = plan.generation_sampling.same_seed_reproducibility_control;
  addError(errors,
    control.tts_engine === 'voxtral-tts-4b-bf16-mlx-audio' &&
      control.seed === 0 && control.additional_repetitions === 1 &&
      control.passage_slot_per_locale === 'technical-a' &&
      control.voices_per_locale === 2 &&
      control.postprocessing_required === false,
  'same-seed control must preserve one extra Voxtral seed-0 generation');

  addError(errors, plan.document_grouping.sections_per_document === 6,
    'each LLM document must contain six sections');
  addError(errors, same(plan.document_grouping.section_order, [
    'technical-a',
    'technical-b',
    'native-a',
    'native-b',
    'dialogue-a',
    'dialogue-b',
  ]), 'document section order changed');
  addError(errors, plan.document_grouping.hidden_reference_visible_to_llm ===
    false, 'the hidden reference must never be visible to the LLM');

  const prompt = promptManifest.prompts.find((candidate) =>
    candidate.id === plan.prompt_contract.id);
  addError(errors, Boolean(prompt),
    'prompt contract must exist in the prompt manifest');
  if (prompt) {
    addError(errors, prompt.path === plan.prompt_contract.path &&
      prompt.sha256 === plan.prompt_contract.sha256,
    'prompt contract metadata differs from the prompt manifest');
    const promptBytes = await readFile(resolve(repoRoot, prompt.path));
    const digest = createHash('sha256').update(promptBytes).digest('hex');
    addError(errors, digest === prompt.sha256,
      'prompt file SHA-256 differs from the pinned digest');
  }
  addError(errors,
    plan.prompt_contract.structured_output_contract ===
      'bounded-transcript-sections-local-diff-v2' &&
      plan.prompt_contract.model_authored_edit_ledger_required === false &&
      plan.prompt_contract.authoritative_diff ===
        'repository-local-input-output-diff' &&
      plan.prompt_contract.protected_span_gate === true &&
      plan.prompt_contract.harmful_change_review === true,
  'prompt contract must use the local authoritative diff and safety gates');

  for (const engine of plan.axes.tts_engines) {
    if (engine.manifest_path !== null) {
      const manifest = await readJson(engine.manifest_path);
      addError(errors, manifest.task === 'tts',
        `${engine.id}: referenced manifest must describe a TTS model`);
    }
  }
  for (const model of plan.axes.llm_models) {
    const manifest = await readJson(model.manifest_path);
    addError(errors, manifest.model?.requested_id === model.id,
      `${model.id}: candidate manifest requested model differs`);
    const requestPolicy = manifest.gateway?.request_policy;
    addError(errors,
      Array.isArray(requestPolicy?.only) && requestPolicy.only.length === 1 &&
      requestPolicy.allow_fallbacks === false &&
      requestPolicy.data_collection === 'deny' && requestPolicy.zdr === true,
    `${model.id}: candidate must remain provider-pinned, no-fallback, and ZDR`);
  }

  const execution = plan.execution_policy;
  for (const field of [
    'block_until_all_passage_slots_materialized',
    'block_until_all_voice_slots_qualified',
    'same_normalized_pcm_for_every_stt_model',
    'retain_lossless_audio_locally',
    'retain_reviewed_opus_fixture_in_repository',
    'retain_raw_stt_transcripts',
    'retain_every_llm_response',
    'randomize_llm_execution_order_with_recorded_seed',
    'require_provider_pin',
    'require_no_fallback',
    'require_zdr',
    'native_review_required_for_non_german_and_non_english_passages',
  ]) {
    addError(errors, execution[field] === true,
      `execution_policy.${field} must remain true`);
  }
  addError(errors, Number.isInteger(execution.llm_execution_order_seed),
    'LLM execution order seed must be an integer');

  const passageCount = expectedLocales.length * expectedContentTypes.length *
    plan.axes.passages_per_content_type;
  const voiceCount = expectedLocales.length * expectedTtsEngines.length *
    plan.axes.voices_per_tts_engine_and_locale;
  const primaryAudio = passageCount * expectedTtsEngines.length *
    plan.axes.voices_per_tts_engine_and_locale *
    plan.generation_sampling.primary_realizations_per_voice_and_passage;
  const controlAudio = expectedLocales.length * control.voices_per_locale *
    control.additional_repetitions;
  const primaryStt = primaryAudio * expectedSttModels.length;
  const controlStt = controlAudio * expectedSttModels.length;
  const documents = expectedLocales.length * expectedTtsEngines.length *
    plan.axes.voices_per_tts_engine_and_locale *
    plan.generation_sampling.primary_realizations_per_voice_and_passage *
    expectedSttModels.length;
  const modelRuns = documents * expectedLlmModels.length;
  const requests = modelRuns * plan.axes.llm_repetitions;
  const expectedCounts = {
    passage_slots: passageCount,
    materialized_passage_slots: materializedCount,
    pending_passage_slots: pendingCount,
    voice_slots: voiceCount,
    primary_audio_artifacts: primaryAudio,
    same_seed_control_audio_artifacts: controlAudio,
    total_audio_artifacts: primaryAudio + controlAudio,
    primary_stt_transcripts: primaryStt,
    same_seed_control_stt_transcripts: controlStt,
    total_stt_transcripts: primaryStt + controlStt,
    llm_documents: documents,
    llm_model_runs: modelRuns,
    llm_requests: requests,
    primary_section_level_outputs: modelRuns *
      plan.document_grouping.sections_per_document,
    section_level_repeat_observations: requests *
      plan.document_grouping.sections_per_document,
  };
  addError(errors, same(plan.expected_counts, expectedCounts),
    'expected_counts do not reconcile with the factorial axes');

  const cost = execution.cost_policy;
  const referenceCost = plan.axes.llm_models.reduce((sum, model) =>
    sum + model.reference_two_request_cost_usd, 0);
  addError(errors, near(cost.reference_two_request_cost_per_document_usd,
    referenceCost), 'reference per-document cost does not sum model costs');
  addError(errors, near(
    cost.projected_reference_cost_for_180_documents_usd,
    referenceCost * documents,
  ), 'projected reference cost does not equal documents times model costs');
  addError(errors,
    cost.require_dry_run_token_estimate === true &&
      cost.execute_in_complete_randomized_blocks === true &&
      cost.never_drop_failed_or_expensive_cells_after_observation === true &&
      cost.require_explicit_budget_approval_before_remote_execution === true,
  'remote cost controls must remain enabled');

  return errors;
}

function validateMatrix(matrix, plan, selection) {
  const errors = [];
  const expected = plan.expected_counts;
  addError(errors, matrix.schema_version === '1.0.0',
    'matrix schema_version must be 1.0.0');
  addError(errors, matrix.plan_id === plan.id &&
    matrix.plan_revision === plan.revision,
  'matrix must reference the current factorial plan');
  addError(errors, matrix.source_selection_revision === selection.revision,
    'matrix must reference the current source selection');
  addError(errors,
    matrix.generation?.generator ===
      'scripts/generate-postprocessing-factorial-matrix.mjs' &&
      matrix.generation?.deterministic === true &&
      matrix.generation?.llm_execution_order_seed ===
        plan.execution_policy.llm_execution_order_seed &&
      matrix.generation?.execution_blocks === plan.axes.llm_repetitions,
  'matrix generator and execution randomization metadata differ');

  const expectedMatrixCounts = {
    primary_audio_units: expected.primary_audio_artifacts,
    same_seed_control_audio_units: expected.same_seed_control_audio_artifacts,
    total_audio_units: expected.total_audio_artifacts,
    primary_stt_units: expected.primary_stt_transcripts,
    same_seed_control_stt_units: expected.same_seed_control_stt_transcripts,
    total_stt_units: expected.total_stt_transcripts,
    llm_documents: expected.llm_documents,
    llm_model_runs: expected.llm_model_runs,
    llm_requests: expected.llm_requests,
  };
  addError(errors, same(matrix.counts, expectedMatrixCounts),
    'matrix counts differ from the plan');

  const collections = [
    ['audio_units', matrix.audio_units, expected.total_audio_artifacts],
    ['stt_units', matrix.stt_units, expected.total_stt_transcripts],
    ['llm_documents', matrix.llm_documents, expected.llm_documents],
    ['llm_model_runs', matrix.llm_model_runs, expected.llm_model_runs],
    ['llm_requests_in_execution_order',
      matrix.llm_requests_in_execution_order, expected.llm_requests],
  ];
  for (const [name, items, count] of collections) {
    addError(errors, Array.isArray(items) && items.length === count,
      `${name} length differs from expected count`);
    addError(errors, unique((items ?? []).map((item) => item.id)),
      `${name} ids must be unique`);
  }

  const passageSlots = new Map(plan.passage_slots.map((slot) => [slot.id, slot]));
  const voiceSlots = new Map(plan.voice_slots.map((slot) => [slot.id, slot]));
  const audioUnits = new Map(matrix.audio_units.map((unit) => [unit.id, unit]));
  for (const audio of matrix.audio_units) {
    addError(errors, passageSlots.has(audio.passage_slot_id),
      `${audio.id}: unknown passage slot`);
    addError(errors, voiceSlots.has(audio.voice_slot_id),
      `${audio.id}: unknown voice slot`);
    const shouldBlock = audio.blockers.length > 0;
    addError(errors, audio.status === (shouldBlock ? 'blocked' : 'ready'),
      `${audio.id}: status and blockers disagree`);
  }
  addError(errors, matrix.audio_units.filter((unit) => unit.kind === 'primary')
    .length === expected.primary_audio_artifacts,
  'primary audio unit count differs');
  addError(errors, matrix.audio_units.filter((unit) =>
    unit.kind === 'same-seed-reproducibility-control').length ===
    expected.same_seed_control_audio_artifacts,
  'same-seed control audio unit count differs');

  const sttModels = new Set(plan.axes.stt_models.map((model) => model.id));
  const sttUnits = new Map(matrix.stt_units.map((unit) => [unit.id, unit]));
  for (const stt of matrix.stt_units) {
    const audio = audioUnits.get(stt.audio_unit_id);
    addError(errors, Boolean(audio), `${stt.id}: unknown audio unit`);
    addError(errors, sttModels.has(stt.stt_model),
      `${stt.id}: unknown STT model`);
    if (audio) {
      addError(errors, stt.status === audio.status &&
        same(stt.blockers, audio.blockers),
      `${stt.id}: readiness differs from its audio unit`);
    }
  }

  const documents = new Map(matrix.llm_documents.map((item) => [item.id, item]));
  for (const document of matrix.llm_documents) {
    addError(errors, document.sections.length === 6,
      `${document.id}: must contain six sections`);
    addError(errors, same(document.sections.map((section) =>
      section.section_id.split('-').slice(-2).join('-')),
    plan.document_grouping.section_order),
    `${document.id}: section order differs from the plan`);
    for (const section of document.sections) {
      const stt = sttUnits.get(section.stt_unit_id);
      addError(errors, Boolean(stt),
        `${document.id}: section references unknown STT unit`);
      if (stt) {
        addError(errors, stt.kind === 'primary' &&
          stt.passage_slot_id === section.section_id,
        `${document.id}: section/STT coordinates disagree`);
      }
    }
    addError(errors, document.status ===
      (document.blockers.length > 0 ? 'blocked' : 'ready'),
    `${document.id}: status and blockers disagree`);
  }

  const llmModels = new Set(plan.axes.llm_models.map((model) => model.id));
  const modelRuns = new Map(matrix.llm_model_runs.map((run) => [run.id, run]));
  const requests = new Map(matrix.llm_requests_in_execution_order.map(
    (request) => [request.id, request]));
  for (const run of matrix.llm_model_runs) {
    const document = documents.get(run.document_id);
    addError(errors, Boolean(document), `${run.id}: unknown document`);
    addError(errors, llmModels.has(run.llm_model),
      `${run.id}: unknown LLM model`);
    addError(errors, run.request_ids.length === plan.axes.llm_repetitions,
      `${run.id}: request count differs from LLM repetitions`);
    for (const requestId of run.request_ids) {
      const request = requests.get(requestId);
      addError(errors, Boolean(request) && request.llm_run_id === run.id,
        `${run.id}: request reference is missing or inconsistent`);
    }
    if (document) {
      addError(errors, run.status === document.status &&
        same(run.blockers, document.blockers),
      `${run.id}: readiness differs from its document`);
    }
  }

  const orderedRequests = matrix.llm_requests_in_execution_order;
  addError(errors, same(orderedRequests.map((request) =>
    request.global_execution_order),
  Array.from({ length: expected.llm_requests }, (_, index) => index + 1)),
  'global LLM execution order must be contiguous and match array order');
  const blockSize = expected.llm_model_runs;
  for (let block = 1; block <= plan.axes.llm_repetitions; block += 1) {
    const blockRequests = orderedRequests.filter((request) =>
      request.execution_block === block);
    addError(errors, blockRequests.length === blockSize,
      `execution block ${block} must contain every model run once`);
    addError(errors, same(blockRequests.map((request) =>
      request.block_execution_order),
    Array.from({ length: blockSize }, (_, index) => index + 1)),
    `execution block ${block} order must be contiguous`);
    addError(errors, unique(blockRequests.map((request) =>
      request.llm_run_id)),
    `execution block ${block} must not repeat a model run`);
  }
  for (const request of orderedRequests) {
    const run = modelRuns.get(request.llm_run_id);
    addError(errors, Boolean(run) && run.document_id === request.document_id &&
      run.llm_model === request.llm_model,
    `${request.id}: request coordinates differ from its model run`);
    addError(errors, request.repetition === request.execution_block,
      `${request.id}: repetition must equal its complete execution block`);
    if (run) {
      addError(errors, request.status === run.status &&
        same(request.blockers, run.blockers),
      `${request.id}: readiness differs from its model run`);
    }
  }
  return errors;
}

async function main() {
  const allowed = new Set(['--self-test']);
  for (const argument of process.argv.slice(2)) {
    if (!allowed.has(argument)) throw new Error(`Unknown argument: ${argument}`);
  }
  const [plan, matrix, selection, promptManifest] = await Promise.all([
    readJson(planPath),
    readJson(matrixPath),
    readJson('benchmarks/fixtures/synthetic-roundtrip-selection.json'),
    readJson('benchmarks/postprocessing/prompts/manifest.json'),
  ]);
  const errors = [
    ...await validatePlan(plan, selection, promptManifest),
    ...validateMatrix(matrix, plan, selection),
  ];
  if (errors.length > 0) {
    throw new Error(`Factorial benchmark validation failed:\n- ${errors.join('\n- ')}`);
  }

  if (process.argv.includes('--self-test')) {
    const changed = structuredClone(plan);
    changed.axes.passages_per_content_type = 1;
    const mutationErrors = await validatePlan(
      changed,
      selection,
      promptManifest,
    );
    if (!mutationErrors.some((error) =>
      error.includes('exactly two independent passages'))) {
      throw new Error('Self-test failed to reject a one-passage design');
    }
  }
  console.log('Validated factorial postprocessing plan and execution ledger');
}

await main();
