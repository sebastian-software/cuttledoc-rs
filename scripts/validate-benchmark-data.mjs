#!/usr/bin/env node

import { readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const schemaVersion = '1.0.0';

function at(object, path) {
  return path.split('.').reduce((value, key) => value?.[key], object);
}

function isMetric(value, integer = false) {
  return (
    value === null ||
    (typeof value === 'number' && Number.isFinite(value) && value >= 0 && (!integer || Number.isInteger(value)))
  );
}

function validateRun(run, fixtures) {
  const errors = [];
  const requiredStrings = [
    'run_id',
    'captured_at',
    'candidate.model.id',
    'candidate.model.revision',
    'candidate.model.quantization',
    'candidate.model.license',
    'candidate.runtime.id',
    'candidate.runtime.version',
    'candidate.runtime.boundary',
    'host.id',
    'host.chip',
    'host.os',
    'engineering_cost.conversion_effort',
    'engineering_cost.update_cost',
    'engineering_cost.packaging_cost',
    'reproduction.command',
  ];
  for (const path of requiredStrings) {
    if (typeof at(run, path) !== 'string' || at(run, path).length === 0) {
      errors.push(`${path} must be a non-empty string`);
    }
  }

  if (run.schema_version !== schemaVersion) errors.push(`schema_version must be ${schemaVersion}`);
  if (!/^[0-9a-f]{40}$/.test(run.source_revision ?? '')) errors.push('source_revision must be a 40-character Git SHA');
  if (run.candidate?.task !== 'asr') errors.push('candidate.task must be asr');
  if (run.host?.architecture !== 'arm64') errors.push('host.architecture must be arm64');
  if (!['ac', 'battery', 'unknown'].includes(run.host?.power_state)) errors.push('host.power_state is invalid');
  if (run.host?.memory_bytes !== null && !isMetric(run.host?.memory_bytes, true)) errors.push('host.memory_bytes must be a positive integer or null');
  if (run.fixture_id !== null && !fixtures.has(run.fixture_id)) errors.push(`fixture_id ${run.fixture_id} is not in the manifest`);

  const metricPaths = [
    ['measurements.quality.wer', false],
    ['measurements.quality.cer', false],
    ['measurements.timing.audio_duration_ms', false],
    ['measurements.timing.cold_load_ms', false],
    ['measurements.timing.warm_inference_ms', false],
    ['measurements.timing.real_time_factor', false],
    ['measurements.resources.peak_memory_bytes', true],
    ['measurements.resources.model_size_bytes', true],
    ['measurements.resources.binary_size_bytes', true],
    ['measurements.streaming.first_result_ms', false],
    ['measurements.streaming.update_count', true],
    ['measurements.streaming.volatile_update_count', true],
    ['measurements.streaming.final_update_count', true],
    ['measurements.streaming.revoke_count', true],
    ['measurements.energy.sample_rate_ms', true],
    ['measurements.energy.sample_count', true],
    ['measurements.energy.energy_joules', false],
  ];
  for (const [path, integer] of metricPaths) {
    if (!isMetric(at(run, path), integer)) errors.push(`${path} must be a non-negative metric or null`);
  }

  if (!['none', 'segment', 'word', 'unknown'].includes(run.measurements?.streaming?.timestamps)) errors.push('measurements.streaming.timestamps is invalid');
  if (!['powermetrics', 'external-meter', 'not-measured'].includes(run.measurements?.energy?.method)) errors.push('measurements.energy.method is invalid');
  if (!Array.isArray(run.reproduction?.raw_artifacts)) errors.push('reproduction.raw_artifacts must be an array');
  if (!Array.isArray(run.result?.observations)) errors.push('result.observations must be an array');

  const status = run.result?.status;
  if (!['measured', 'partial', 'blocked'].includes(status)) errors.push('result.status is invalid');
  if ((status === 'blocked' || status === 'partial') && !(run.result?.blocker?.length > 0)) errors.push(`${status} result requires a precise blocker`);
  if (status === 'measured') {
    const fixture = fixtures.get(run.fixture_id);
    if (!fixture || fixture.purpose !== 'quality') errors.push('measured result requires a quality fixture');
    if (!Number.isInteger(run.host?.memory_bytes) || run.host.memory_bytes <= 0) errors.push('measured result requires host.memory_bytes');
    for (const path of [
      'measurements.quality.wer',
      'measurements.quality.cer',
      'measurements.timing.audio_duration_ms',
      'measurements.timing.cold_load_ms',
      'measurements.timing.warm_inference_ms',
      'measurements.timing.real_time_factor',
      'measurements.resources.peak_memory_bytes',
      'measurements.resources.model_size_bytes',
    ]) {
      if (typeof at(run, path) !== 'number') errors.push(`measured result requires ${path}`);
    }
    if (run.reproduction.raw_artifacts.length === 0) errors.push('measured result requires at least one raw artifact');
  }

  if (run.measurements?.streaming?.supported) {
    if (typeof run.measurements.streaming.first_result_ms !== 'number') errors.push('streaming result requires first_result_ms');
    if (!(run.measurements.streaming.update_count > 0)) errors.push('streaming result requires update_count');
    if (!(run.measurements.streaming.final_update_count > 0)) errors.push('streaming result requires final_update_count');
  }

  return errors;
}

function validateManifest(manifest) {
  const errors = [];
  if (manifest.schema_version !== schemaVersion) errors.push(`manifest schema_version must be ${schemaVersion}`);
  if (!Array.isArray(manifest.fixtures) || manifest.fixtures.length === 0) errors.push('manifest must contain fixtures');
  const ids = new Set();
  for (const fixture of manifest.fixtures ?? []) {
    if (ids.has(fixture.id)) errors.push(`duplicate fixture id: ${fixture.id}`);
    ids.add(fixture.id);
    if (!['smoke', 'quality'].includes(fixture.purpose)) errors.push(`${fixture.id}: invalid purpose`);
    if (!(fixture.reference_text?.length > 0)) errors.push(`${fixture.id}: reference_text is required`);
    if (fixture.purpose === 'quality') {
      if (!/^[0-9a-f]{64}$/.test(fixture.sha256 ?? '')) errors.push(`${fixture.id}: quality fixture requires sha256`);
      if (!fixture.provenance?.redistributable && fixture.availability !== 'local-required') {
        errors.push(`${fixture.id}: non-redistributable quality fixture must be local-required`);
      }
    }
  }
  return { errors, ids };
}

function validateSourceCandidates(registry) {
  const errors = [];
  if (registry.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(registry.evidence_date ?? '')) {
    errors.push('evidence_date must be an ISO date');
  }
  if (!Array.isArray(registry.target_languages) ||
      registry.target_languages.length === 0 ||
      new Set(registry.target_languages).size !== registry.target_languages.length) {
    errors.push('target_languages must be a non-empty unique array');
  }
  if (!Array.isArray(registry.sources) || registry.sources.length === 0) {
    errors.push('sources must be a non-empty array');
  }
  const ids = new Set();
  const statuses = new Set([
    'active-diagnostic',
    'active-development-pilot',
    'selected-for-acquisition',
    'selected-pending-rights-review',
    'selected-as-domain-bridge',
    'acquisition-required',
  ]);
  for (const source of registry.sources ?? []) {
    if (!(source.id?.length > 0)) errors.push('source id must be a non-empty string');
    if (ids.has(source.id)) errors.push(`duplicate source id: ${source.id}`);
    ids.add(source.id);
    if (!statuses.has(source.status)) errors.push(`${source.id}: invalid status`);
    for (const field of ['domains', 'languages']) {
      if (!Array.isArray(source[field]) ||
          source[field].length === 0 ||
          new Set(source[field]).size !== source[field].length) {
        errors.push(`${source.id}: ${field} must be a non-empty unique array`);
      }
    }
    if (source.source_revision !== null &&
        !(source.source_revision?.length > 0)) {
      errors.push(`${source.id}: source_revision must be a string or null`);
    }
    for (const field of ['license', 'redistribution', 'decision']) {
      if (!(source[field]?.length > 0)) {
        errors.push(`${source.id}: ${field} must be a non-empty string`);
      }
    }
  }
  return errors;
}

function validateTargetDomainPlan(plan, sourceCandidates) {
  const errors = [];
  if (plan.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!(plan.revision?.length > 0)) {
    errors.push('revision must be a non-empty string');
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(plan.evidence_date ?? '')) {
    errors.push('evidence_date must be an ISO date');
  }
  if (!/^https:\/\/github\.com\/[^/]+\/[^/]+\/issues\/\d+$/.test(
    plan.tracking_issue ?? '',
  )) {
    errors.push('tracking_issue must be a GitHub issue URL');
  }
  if (plan.purpose !== 'held-out') {
    errors.push('purpose must be held-out');
  }

  const requiredBackends = [
    'apple-speechtranscriber',
    'whisper-large-v3-turbo-coreml-whispercpp',
    'qwen3-asr-0.6b-mlx-direct',
    'parakeet-tdt-0.6b-v3-coreml',
  ];
  if (!arrayEquals(plan.candidate_backends ?? [], requiredBackends)) {
    errors.push('candidate_backends must contain the four frozen candidates in execution order');
  }

  const sources = new Map(
    (sourceCandidates.sources ?? []).map((source) => [source.id, source]),
  );
  for (const sourceId of plan.excluded_development_sources ?? []) {
    const source = sources.get(sourceId);
    if (!source) {
      errors.push(`unknown excluded development source: ${sourceId}`);
    } else if (!['active-diagnostic', 'active-development-pilot'].includes(
      source.status,
    )) {
      errors.push(`${sourceId}: excluded source must remain diagnostic or development`);
    }
  }
  if (!arrayEquals(
    plan.excluded_development_sources ?? [],
    ['fleurs', 'multilingual-librispeech', 'librispeech-clean'],
  )) {
    errors.push('excluded_development_sources must preserve the known development inputs');
  }

  const alignedClip = plan.tracks?.aligned_clip;
  if (alignedClip?.minimum_total_duration_ms < 1_800_000) {
    errors.push('aligned_clip requires at least 30 minutes per cell');
  }
  if (alignedClip?.minimum_source_groups < 3) {
    errors.push('aligned_clip requires at least three source groups per cell');
  }
  if (alignedClip?.minimum_speakers < 3) {
    errors.push('aligned_clip requires at least three speakers per cell');
  }
  if (!(alignedClip?.alignment?.length > 0)) {
    errors.push('aligned_clip.alignment must be a non-empty string');
  }
  const longForm = plan.tracks?.long_form;
  if (longForm?.minimum_passage_duration_ms !== 300_000 ||
      longForm?.maximum_passage_duration_ms !== 900_000) {
    errors.push('long_form passage bounds must remain 5-15 minutes');
  }
  if (longForm?.minimum_passages_per_source_group < 1) {
    errors.push('long_form requires at least one passage per source group');
  }
  if (!Array.isArray(longForm?.required_behaviors) ||
      longForm.required_behaviors.length === 0 ||
      new Set(longForm.required_behaviors).size !==
        longForm.required_behaviors.length) {
    errors.push('long_form.required_behaviors must be a non-empty unique array');
  }

  for (const field of ['grouping_unit', 'development', 'validation', 'test']) {
    if (!(plan.split_policy?.[field]?.length > 0)) {
      errors.push(`split_policy.${field} must be a non-empty string`);
    }
  }
  if (plan.split_policy?.forbid_cross_split_derivatives !== true) {
    errors.push('split_policy must forbid cross-split derivatives');
  }
  if (plan.rights_policy?.default_redistribution !== 'denied' ||
      plan.rights_policy?.acquisition_gate !== 'accepted-rights-review' ||
      plan.rights_policy?.separate_audio_and_transcript_evidence !== true ||
      plan.rights_policy?.local_required_by_default !== true ||
      plan.rights_policy?.repository_license_is_not_audio_evidence !== true) {
    errors.push('rights_policy must preserve the deny-by-default acquisition gate');
  }
  if (!Array.isArray(plan.rights_policy?.accepted_bases) ||
      plan.rights_policy.accepted_bases.length === 0 ||
      new Set(plan.rights_policy.accepted_bases).size !==
        plan.rights_policy.accepted_bases.length) {
    errors.push('rights_policy.accepted_bases must be a non-empty unique array');
  }
  for (const field of ['draft_sources_only', 'preserve', 'critical_content']) {
    if (!Array.isArray(plan.gold_policy?.[field]) ||
        plan.gold_policy[field].length === 0 ||
        new Set(plan.gold_policy[field]).size !==
          plan.gold_policy[field].length) {
      errors.push(`gold_policy.${field} must be a non-empty unique array`);
    }
  }
  if (!(plan.gold_policy?.verification?.length > 0)) {
    errors.push('gold_policy.verification must be a non-empty string');
  }

  const expectedCells = sourceCandidates.target_languages.flatMap(
    (locale) => ['podcast', 'audiobook'].map((domain) => `${locale}/${domain}`),
  );
  const cells = plan.cells ?? [];
  const cellIds = cells.map((cell) => cell.id);
  if (!arrayEquals(cellIds, expectedCells)) {
    errors.push('cells must cover every target locale and domain in German-first order');
  }
  if (!arrayEquals(plan.priority_order ?? [], cellIds)) {
    errors.push('priority_order must exactly match cell order');
  }
  const seenCells = new Set();
  const excludedSources = new Set(plan.excluded_development_sources ?? []);
  for (const [index, cell] of cells.entries()) {
    if (seenCells.has(cell.id)) errors.push(`duplicate target-domain cell: ${cell.id}`);
    seenCells.add(cell.id);
    if (cell.priority !== index + 1) {
      errors.push(`${cell.id}: priority must match German-first cell order`);
    }
    if (cell.id !== `${cell.locale}/${cell.domain}`) {
      errors.push(`${cell.id}: id must match locale/domain`);
    }
    if (!['acquisition-required', 'rights-review', 'gold-review', 'ready', 'complete'].includes(
      cell.status,
    )) {
      errors.push(`${cell.id}: invalid status`);
    }
    if (!Array.isArray(cell.source_candidate_ids) ||
        cell.source_candidate_ids.length === 0 ||
        new Set(cell.source_candidate_ids).size !==
          cell.source_candidate_ids.length) {
      errors.push(`${cell.id}: source_candidate_ids must be a non-empty unique array`);
      continue;
    }
    for (const sourceId of cell.source_candidate_ids) {
      const source = sources.get(sourceId);
      if (!source) {
        errors.push(`${cell.id}: unknown source candidate ${sourceId}`);
        continue;
      }
      if (excludedSources.has(sourceId)) {
        errors.push(`${cell.id}: development source ${sourceId} cannot enter held-out cells`);
      }
      if (!source.domains.includes(cell.domain)) {
        errors.push(`${cell.id}: ${sourceId} does not cover ${cell.domain}`);
      }
      if (!source.languages.includes(cell.locale)) {
        errors.push(`${cell.id}: ${sourceId} does not cover ${cell.locale}`);
      }
    }
  }
  return errors;
}

function validateSyntheticRoundtripPlan(plan) {
  const errors = [];
  const uniqueStrings = (value) => (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => typeof item === 'string' && item.length > 0) &&
    new Set(value).size === value.length
  );
  const requiredBackends = [
    'apple-speechtranscriber',
    'whisper-large-v3-turbo-coreml-whispercpp',
    'qwen3-asr-0.6b-mlx-direct',
    'parakeet-tdt-0.6b-v3-coreml',
  ];
  const requiredCandidateIds = [
    'apple-avspeechsynthesizer',
    'qwen3-tts-0.6b-customvoice-mlx-audio',
    'voxtral-tts-4b-mlx-audio',
    'chatterbox-multilingual-mlx-audio',
    'qwen-audio-3.0-tts-plus-api',
  ];
  const requiredSourceIds = [
    'de-wikipedia-kuenstliche-intelligenz-268935951',
    'en-wikipedia-artificial-intelligence-1365114492',
  ];

  if (plan.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!(plan.revision?.length > 0)) {
    errors.push('revision must be a non-empty string');
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(plan.evidence_date ?? '')) {
    errors.push('evidence_date must be an ISO date');
  }
  if (plan.tracking_issue !==
      'https://github.com/sebastian-software/cuttledoc-rs/issues/13') {
    errors.push('tracking_issue must remain Phase 5 issue #13');
  }
  if (plan.purpose !== 'diagnostic') {
    errors.push('purpose must remain diagnostic');
  }
  if (plan.relationship_to_target_domain?.eligible_for_model_selection !== false ||
      plan.relationship_to_target_domain?.eligible_for_release_acceptance !== false) {
    errors.push('synthetic evidence must not select ASR models or satisfy release acceptance');
  }
  for (const field of ['required_comparison', 'reason']) {
    if (!(plan.relationship_to_target_domain?.[field]?.length > 0)) {
      errors.push(`relationship_to_target_domain.${field} must be a non-empty string`);
    }
  }

  if (!arrayEquals(plan.initial_scope?.locales ?? [], ['de-DE', 'en-US']) ||
      plan.initial_scope?.primary_locale !== 'de-DE') {
    errors.push('initial scope must preserve German-first de-DE and en-US cells');
  }
  if (plan.initial_scope?.topic !== 'artificial-intelligence') {
    errors.push('initial scope topic must remain artificial-intelligence');
  }
  if (plan.initial_scope?.minimum_passages_per_locale?.['de-DE'] < 6 ||
      plan.initial_scope?.minimum_passages_per_locale?.['en-US'] < 3) {
    errors.push('initial scope requires at least six German and three English passages');
  }
  if (plan.initial_scope?.minimum_passage_duration_ms !== 45_000 ||
      plan.initial_scope?.maximum_passage_duration_ms !== 90_000) {
    errors.push('initial passage duration must remain 45-90 seconds');
  }
  if (!uniqueStrings(plan.initial_scope?.required_text_phenomena)) {
    errors.push('required_text_phenomena must be a non-empty unique string array');
  }

  const sources = plan.text_sources ?? [];
  if (!arrayEquals(sources.map((source) => source.id), requiredSourceIds)) {
    errors.push('text_sources must preserve the pinned German and English Wikipedia revisions');
  }
  for (const source of sources) {
    for (const field of [
      'id',
      'locale',
      'title',
      'revision',
      'revision_url',
      'history_url',
      'license',
      'selection_status',
    ]) {
      if (!(source[field]?.length > 0)) {
        errors.push(`${source.id ?? '<source>'}.${field} must be a non-empty string`);
      }
    }
    for (const field of ['revision_url', 'history_url']) {
      try {
        if (new URL(source[field]).protocol !== 'https:') {
          errors.push(`${source.id}.${field} must use HTTPS`);
        }
      } catch {
        errors.push(`${source.id}.${field} must be an absolute URL`);
      }
    }
    if (source.license !== 'CC-BY-SA-4.0') {
      errors.push(`${source.id}: Wikipedia source license must be CC-BY-SA-4.0`);
    }
    if (!uniqueStrings(source.required_sections)) {
      errors.push(`${source.id}: required_sections must be a non-empty unique string array`);
    }
    const policy = source.materialization_policy;
    for (const field of [
      'pin_exact_revision',
      'record_verbatim_text_sha256',
      'record_spoken_text_sha256',
      'record_change_notice',
      'preserve_attribution',
    ]) {
      if (policy?.[field] !== true) {
        errors.push(`${source.id}: materialization_policy.${field} must be true`);
      }
    }
    if (policy?.generated_audio_location !== 'local-required' ||
        policy?.redistribution !==
          'blocked-until-CC-BY-SA-attribution-package-review') {
      errors.push(`${source.id}: generated assets must remain local and redistribution-blocked`);
    }
  }

  const candidates = plan.tts_candidates ?? [];
  const candidateIds = candidates.map((candidate) => candidate.id);
  if (!arrayEquals(candidateIds, requiredCandidateIds)) {
    errors.push('tts_candidates must preserve the selected system, local MLX, and remote controls');
  }
  const candidateIdSet = new Set(candidateIds);
  for (const candidate of candidates) {
    for (const field of [
      'id',
      'role',
      'runtime',
      'model',
      'integration',
      'source_url',
      'source_revision',
      'license',
      'status',
      'decision_after_pilot',
    ]) {
      if (!(candidate[field]?.length > 0)) {
        errors.push(`${candidate.id ?? '<candidate>'}.${field} must be a non-empty string`);
      }
    }
    try {
      if (new URL(candidate.source_url).protocol !== 'https:') {
        errors.push(`${candidate.id}.source_url must use HTTPS`);
      }
    } catch {
      errors.push(`${candidate.id}.source_url must be an absolute URL`);
    }
    if (!uniqueStrings(candidate.locales)) {
      errors.push(`${candidate.id}: locales must be a non-empty unique string array`);
    }
  }
  const mlxCandidates = candidates.filter(
    (candidate) => candidate.runtime?.startsWith('Blaizzy/mlx-audio'),
  );
  if (mlxCandidates.length !== 3 ||
      mlxCandidates.some(
        (candidate) => candidate.source_revision !==
          '64e8416c303fb3b3463dab8eb4ebd78c55a87c1a',
      )) {
    errors.push('local MLX candidates must pin the reviewed mlx-audio revision');
  }
  const remoteCandidate = candidates.find(
    (candidate) => candidate.id === 'qwen-audio-3.0-tts-plus-api',
  );
  if (!arrayEquals(remoteCandidate?.locales ?? [], ['en-US'])) {
    errors.push('Qwen-Audio-3.0-TTS-Plus must remain an English-only quality control');
  }

  const evidence = plan.external_evidence ?? [];
  if (evidence.length === 0) {
    errors.push('external_evidence must not be empty');
  }
  for (const item of evidence) {
    if (!candidateIdSet.has(item.candidate_id)) {
      errors.push(`external evidence names unknown candidate ${item.candidate_id}`);
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(item.observed_at ?? '')) {
      errors.push('external evidence observed_at must be an ISO date');
    }
    if (!(item.source?.length > 0) ||
        !(item.metric?.length > 0) ||
        !(item.scope?.length > 0) ||
        !Number.isFinite(item.value) ||
        !Number.isFinite(item.confidence_interval_plus_minus) ||
        !(Number.isInteger(item.sample_count) && item.sample_count > 0)) {
      errors.push('external evidence requires source, metric, scope, value, confidence, and samples');
    }
    try {
      if (new URL(item.source_url).protocol !== 'https:') {
        errors.push('external evidence source_url must use HTTPS');
      }
    } catch {
      errors.push('external evidence source_url must be an absolute URL');
    }
  }

  if (!arrayEquals(plan.asr_backends ?? [], requiredBackends)) {
    errors.push('asr_backends must contain the four frozen candidates in execution order');
  }
  const execution = plan.execution_policy;
  for (const field of [
    'retain_raw_tts_audio',
    'retain_raw_asr_output',
    'fixed_voice_per_candidate_and_locale',
    'fixed_generation_parameters',
    'require_fresh_process_cold_load',
    'forbid_asr_specific_text_changes',
  ]) {
    if (execution?.[field] !== true) {
      errors.push(`execution_policy.${field} must be true`);
    }
  }
  if (!(execution?.matrix?.length > 0) ||
      execution?.minimum_repetitions < 2) {
    errors.push('execution policy requires a matrix description and at least two repetitions');
  }
  for (const field of ['text_fidelity', 'synthesis']) {
    if (!uniqueStrings(plan.metrics?.[field])) {
      errors.push(`metrics.${field} must be a non-empty unique string array`);
    }
  }
  if (!(plan.metrics?.diagnosis?.length > 0)) {
    errors.push('metrics.diagnosis must be a non-empty string');
  }
  if (!uniqueStrings(plan.exit_criteria) || plan.exit_criteria.length < 7) {
    errors.push('exit_criteria must contain at least seven unique strings');
  }
  return errors;
}

function validateSyntheticRoundtripSelection(selection, plan) {
  const errors = [];
  if (selection.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (selection.plan_revision !== plan.revision) {
    errors.push('plan_revision must match the synthetic roundtrip plan');
  }
  if (selection.purpose !== 'diagnostic-materialization' ||
      selection.text_encoding !== 'utf-8' ||
      selection.paragraph_joiner !== '\n\n' ||
      selection.spoken_transform !== 'none' ||
      selection.generated_assets !== 'local-required') {
    errors.push('selection policy must preserve local diagnostic materialization');
  }
  const expectedSources = new Map(
    plan.text_sources.map((source) => [source.id, source]),
  );
  const sourceIds = new Set();
  const passageIds = new Set();
  const localeCounts = new Map();
  const coveredGermanPhenomena = new Set();
  for (const source of selection.sources ?? []) {
    if (sourceIds.has(source.id)) errors.push(`duplicate source id: ${source.id}`);
    sourceIds.add(source.id);
    const expected = expectedSources.get(source.id);
    if (!expected) {
      errors.push(`unknown source id: ${source.id}`);
    } else if (source.locale !== expected.locale ||
        source.title !== expected.title ||
        String(source.revision_id) !== expected.revision ||
        source.revision_url !== expected.revision_url ||
        source.history_url !== expected.history_url ||
        source.license !== expected.license) {
      errors.push(`${source.id}: source metadata differs from the accepted plan`);
    }
    if (!(Number.isInteger(source.page_id) && source.page_id > 0) ||
        !(Number.isInteger(source.revision_id) && source.revision_id > 0) ||
        !(Number.isInteger(source.parent_revision_id) &&
          source.parent_revision_id > 0) ||
        !/^\d{4}-\d{2}-\d{2}T/.test(source.revision_timestamp ?? '')) {
      errors.push(`${source.id}: invalid pinned MediaWiki metadata`);
    }
    for (const field of ['api_url', 'license_url']) {
      try {
        if (new URL(source[field]).protocol !== 'https:') {
          errors.push(`${source.id}.${field} must use HTTPS`);
        }
      } catch {
        errors.push(`${source.id}.${field} must be an absolute URL`);
      }
    }
    if (!(source.attribution?.length > 0)) {
      errors.push(`${source.id}: attribution must be a non-empty string`);
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
        errors.push(`${passage.id}: segments must be a non-empty array`);
      }
      for (const segment of passage.segments ?? []) {
        if (!Array.isArray(segment.section_path) ||
            segment.section_path.length === 0 ||
            segment.section_path.some(
              (part) => typeof part !== 'string' || part.length === 0,
            ) ||
            !(Number.isInteger(segment.paragraph_index) &&
              segment.paragraph_index >= 0)) {
          errors.push(`${passage.id}: invalid paragraph selector`);
        }
      }
      if (!(Number.isInteger(passage.character_count) &&
          passage.character_count > 0)) {
        errors.push(`${passage.id}: character_count must be a positive integer`);
      }
      if (!/^[0-9a-f]{64}$/.test(passage.verbatim_sha256 ?? '') ||
          passage.verbatim_sha256 !== passage.spoken_sha256) {
        errors.push(`${passage.id}: invalid or divergent text digests`);
      }
      if (!Array.isArray(passage.phenomena) ||
          passage.phenomena.length === 0 ||
          new Set(passage.phenomena).size !== passage.phenomena.length) {
        errors.push(`${passage.id}: phenomena must be a non-empty unique array`);
      }
      if (source.locale === 'de-DE') {
        for (const phenomenon of passage.phenomena ?? []) {
          coveredGermanPhenomena.add(phenomenon);
        }
      }
    }
  }
  if (sourceIds.size !== expectedSources.size ||
      [...expectedSources.keys()].some((id) => !sourceIds.has(id))) {
    errors.push('selection must cover every accepted text source exactly once');
  }
  for (const [locale, minimum] of Object.entries(
    plan.initial_scope.minimum_passages_per_locale,
  )) {
    if ((localeCounts.get(locale) ?? 0) < minimum) {
      errors.push(`${locale}: selection does not meet the minimum passage count`);
    }
  }
  for (const phenomenon of plan.initial_scope.required_text_phenomena) {
    if (!coveredGermanPhenomena.has(phenomenon)) {
      errors.push(`German selection does not cover required phenomenon: ${phenomenon}`);
    }
  }
  return errors;
}

function validateAppleTtsRun(run, selection) {
  const errors = [];
  if (run.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!(run.run_id?.length > 0) ||
      !/^\d{4}-\d{2}-\d{2}T/.test(run.captured_at ?? '') ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '')) {
    errors.push('run_id, captured_at, and source_revision must be pinned');
  }
  if (run.selection_revision !== selection.revision ||
      run.purpose !== 'development-diagnostic') {
    errors.push('run must reference the diagnostic passage selection');
  }
  if (run.candidate?.task !== 'tts' ||
      run.candidate?.id !== 'apple-avspeechsynthesizer' ||
      !(run.candidate?.runtime?.length > 0) ||
      !(run.candidate?.boundary?.length > 0) ||
      !(run.candidate?.model_delivery?.length > 0)) {
    errors.push('candidate must identify the Apple TTS task and boundary');
  }
  for (const field of ['identifier', 'name', 'language']) {
    if (!(run.candidate?.voice?.[field]?.length > 0)) {
      errors.push(`candidate.voice.${field} must be a non-empty string`);
    }
  }
  if (run.host?.architecture !== 'arm64' ||
      !(run.host?.memory_bytes > 0) ||
      !(run.host?.chip?.length > 0) ||
      !(run.host?.os?.length > 0) ||
      !(run.host?.execution_context?.length > 0)) {
    errors.push('host must record the Apple Silicon execution context');
  }

  const passages = new Map();
  for (const source of selection.sources) {
    for (const passage of source.passages) {
      passages.set(passage.id, { source, passage });
    }
  }
  const selected = passages.get(run.input?.passage_id);
  if (!selected) {
    errors.push(`unknown TTS input passage: ${run.input?.passage_id}`);
  } else if (run.input.source_id !== selected.source.id ||
      run.input.locale !== selected.source.locale ||
      run.input.character_count !== selected.passage.character_count ||
      run.input.text_sha256 !== selected.passage.spoken_sha256 ||
      run.input.license !== selected.source.license) {
    errors.push('TTS input metadata differs from the passage selection');
  }
  if (run.procedure?.repetitions !== 1 ||
      !(run.procedure?.voice_selection?.length > 0) ||
      !(run.procedure?.command?.length > 0) ||
      !(run.procedure?.raw_audio?.includes('local-required'))) {
    errors.push('procedure must preserve the one-run local diagnostic');
  }
  if (run.result?.status !== 'measured') {
    errors.push('result.status must be measured');
  }
  const audio = run.result?.audio;
  if (audio?.sample_format !== 'f32le' ||
      audio?.channel_count !== 1 ||
      !(Number.isInteger(audio?.sample_rate_hz) && audio.sample_rate_hz > 0) ||
      !(Number.isInteger(audio?.sample_count) && audio.sample_count > 0) ||
      audio?.byte_count !== audio?.sample_count * 4 ||
      !/^[0-9a-f]{64}$/.test(audio?.sha256 ?? '') ||
      audio?.non_finite_sample_count !== 0) {
    errors.push('audio must be finite digest-pinned mono f32 PCM');
  }
  const expectedDuration = audio?.sample_count * 1_000 / audio?.sample_rate_hz;
  if (!Number.isFinite(audio?.duration_ms) ||
      Math.abs(audio.duration_ms - expectedDuration) > 1e-9 ||
      !(audio.minimum_sample >= -1 && audio.minimum_sample <= 1) ||
      !(audio.maximum_sample >= -1 && audio.maximum_sample <= 1) ||
      !(audio.rms > 0 && audio.rms <= 1)) {
    errors.push('audio duration or waveform metrics diverge from PCM metadata');
  }
  const timing = run.result?.timing;
  const expectedRtf = timing?.complete_synthesis_ms / audio?.duration_ms;
  if (!(timing?.session_create_ms >= 0) ||
      !(timing?.first_audio_ms >= 0) ||
      !(timing?.complete_synthesis_ms >= timing?.first_audio_ms) ||
      !(Number.isInteger(timing?.chunk_count) && timing.chunk_count > 0) ||
      !Number.isFinite(timing?.real_time_factor) ||
      Math.abs(timing.real_time_factor - expectedRtf) > 1e-12) {
    errors.push('timing metrics or real-time factor are inconsistent');
  }
  for (const field of [
    'maximum_resident_set_size_bytes',
    'peak_memory_footprint_bytes',
    'swift_dylib_bytes',
    'rust_executable_bytes',
  ]) {
    if (!(Number.isInteger(run.result?.resources?.[field]) &&
        run.result.resources[field] > 0)) {
      errors.push(`resources.${field} must be a positive integer`);
    }
  }
  const lifecycle = run.result?.lifecycle;
  if (lifecycle?.busy_status !== 4 ||
      lifecycle?.cancelled_status !== 3 ||
      !(lifecycle?.cancel_to_return_ms >= 0) ||
      !(lifecycle?.shim_cancellation_latency_ms >= 0) ||
      !(lifecycle?.cancelled_call_elapsed_ms >= 0) ||
      lifecycle?.cancelled_call_sample_count !== 0) {
    errors.push('lifecycle must preserve stable busy and cancelled behavior');
  }
  if (run.environment_control?.voice_inventory_succeeded !== true ||
      run.environment_control?.synthesis_succeeded !== false ||
      run.environment_control?.first_audio_received !== false ||
      !(run.environment_control?.finding?.length > 0)) {
    errors.push('environment control must record the restricted-process failure');
  }
  if (run.conclusion?.foundation_proven !== true ||
      run.conclusion?.public_contract_accepted !== false ||
      !(run.conclusion?.next?.length > 0)) {
    errors.push('conclusion must keep the public TTS contract provisional');
  }
  return errors;
}

function validateQwen3TtsModelManifest(modelManifest, plan, selection) {
  const errors = [];
  const expectedId = 'qwen3-tts-12hz-0.6b-customvoice-mlx-bf16';
  const expectedCandidateId = 'qwen3-tts-0.6b-customvoice-mlx-audio';
  if (modelManifest.id !== expectedId || modelManifest.task !== 'tts') {
    errors.push('id and task must identify the Qwen3-TTS MLX reference');
  }
  for (const revision of [
    modelManifest.source?.observed_revision,
    modelManifest.conversion?.revision,
    modelManifest.reference_runtime?.revision,
  ]) {
    if (!/^[0-9a-f]{40}$/.test(revision ?? '')) {
      errors.push(`invalid pinned revision: ${revision}`);
    }
  }
  if (modelManifest.source?.license !== 'Apache-2.0' ||
      modelManifest.conversion?.license !== 'Apache-2.0' ||
      modelManifest.reference_runtime?.license !== 'MIT' ||
      !(modelManifest.source?.provenance_limit?.length > 0)) {
    errors.push('licenses and the conversion provenance limit must be explicit');
  }
  if (!modelManifest.source?.supported_languages?.includes('German') ||
      modelManifest.generation_contract?.locale !== 'de-DE' ||
      modelManifest.generation_contract?.language !== 'German' ||
      modelManifest.generation_contract?.cross_lingual !== true ||
      modelManifest.generation_contract?.instruction !== null ||
      modelManifest.generation_contract?.sample_rate_hz !== 24000) {
    errors.push(
      'generation contract must preserve the German cross-lingual diagnostic',
    );
  }

  const passage = selection.sources
    .flatMap((source) => source.passages)
    .find((item) => item.id === modelManifest.generation_contract?.passage_id);
  if (!passage) {
    errors.push('generation contract passage is not in the synthetic selection');
  }

  const artifactPaths = new Set();
  let artifactBytes = 0;
  for (const artifact of modelManifest.artifacts ?? []) {
    if (!(artifact.path?.length > 0) || artifactPaths.has(artifact.path)) {
      errors.push(`invalid or duplicate artifact path: ${artifact.path}`);
    }
    artifactPaths.add(artifact.path);
    if (!(Number.isInteger(artifact.bytes) && artifact.bytes > 0) ||
        !/^[0-9a-f]{64}$/.test(artifact.sha256 ?? '')) {
      errors.push(`${artifact.path}: invalid byte count or SHA-256`);
    }
    artifactBytes += artifact.bytes ?? 0;
  }
  for (const requiredPath of [
    'config.json',
    'model.safetensors',
    'speech_tokenizer/model.safetensors',
    'tokenizer_config.json',
  ]) {
    if (!artifactPaths.has(requiredPath)) {
      errors.push(`missing required model artifact: ${requiredPath}`);
    }
  }
  if (artifactBytes !== modelManifest.conversion?.snapshot_bytes) {
    errors.push('artifact bytes do not equal conversion.snapshot_bytes');
  }

  const candidate = plan.tts_candidates.find(
    (item) => item.id === expectedCandidateId,
  );
  if (!candidate ||
      !candidate.model.endsWith(`@${modelManifest.conversion?.revision}`) ||
      candidate.source_revision !== modelManifest.reference_runtime?.revision ||
      candidate.status !==
        'measured-reference-run-pending-listening') {
    errors.push('synthetic plan does not reference the pinned model and runtime');
  }
  return errors;
}

function validateVoxtralTtsModelManifest(modelManifest, plan, selection) {
  const errors = [];
  if (modelManifest.id !== 'voxtral-4b-tts-2603-mlx-4bit' ||
      modelManifest.task !== 'tts') {
    errors.push('id and task must identify the Voxtral TTS MLX reference');
  }
  for (const revision of [
    modelManifest.source?.observed_revision,
    modelManifest.conversion?.revision,
    modelManifest.reference_runtime?.revision,
  ]) {
    if (!/^[0-9a-f]{40}$/.test(revision ?? '')) {
      errors.push(`invalid pinned revision: ${revision}`);
    }
  }
  if (modelManifest.source?.license !== 'CC-BY-NC-4.0' ||
      modelManifest.conversion?.license !== 'CC-BY-NC-4.0' ||
      modelManifest.reference_runtime?.license !== 'MIT' ||
      !(modelManifest.source?.provenance_limit?.length > 0)) {
    errors.push('Voxtral licenses and conversion provenance must be explicit');
  }
  const contract = modelManifest.generation_contract;
  if (!modelManifest.source?.supported_languages?.includes('German') ||
      contract?.locale !== 'de-DE' ||
      contract?.speaker !== 'de_female' ||
      contract?.speaker_native_language !== 'German' ||
      contract?.cross_lingual !== false ||
      contract?.sample_rate_hz !== 24000) {
    errors.push('Voxtral must preserve the native German voice contract');
  }
  const passage = selection.sources
    .flatMap((source) => source.passages)
    .find((item) => item.id === contract?.passage_id);
  if (!passage) {
    errors.push('Voxtral generation passage is not in the selection');
  }

  const artifactPaths = new Set();
  let artifactBytes = 0;
  for (const artifact of modelManifest.artifacts ?? []) {
    if (!(artifact.path?.length > 0) || artifactPaths.has(artifact.path)) {
      errors.push(`invalid or duplicate Voxtral artifact: ${artifact.path}`);
    }
    artifactPaths.add(artifact.path);
    if (!(Number.isInteger(artifact.bytes) && artifact.bytes > 0) ||
        !/^[0-9a-f]{64}$/.test(artifact.sha256 ?? '')) {
      errors.push(`${artifact.path}: invalid byte count or SHA-256`);
    }
    artifactBytes += artifact.bytes ?? 0;
  }
  for (const requiredPath of [
    'config.json',
    'model.safetensors',
    'tekken.json',
    'voice_embedding/de_female.safetensors',
  ]) {
    if (!artifactPaths.has(requiredPath)) {
      errors.push(`missing required Voxtral artifact: ${requiredPath}`);
    }
  }
  if (artifactBytes !== modelManifest.conversion?.snapshot_bytes ||
      modelManifest.conversion?.weight_bytes !== 2509879373) {
    errors.push('Voxtral artifact bytes do not reconcile with the snapshot');
  }

  const candidate = plan.tts_candidates.find(
    (item) => item.id === 'voxtral-tts-4b-mlx-audio',
  );
  if (!candidate ||
      !candidate.model.endsWith(`@${modelManifest.conversion?.revision}`) ||
      candidate.source_revision !== modelManifest.reference_runtime?.revision ||
      candidate.status !==
        'measured-reference-run-pending-listening-and-controls' ||
      !candidate.license.includes('CC-BY-NC-4.0')) {
    errors.push('synthetic plan must pin Voxtral and keep it reference-only');
  }
  return errors;
}

function validateVoxtralRealtimeModelManifest(modelManifest) {
  const errors = [];
  if (modelManifest.id !==
        'voxtral-mini-4b-realtime-2602-mlx-4bit' ||
      modelManifest.task !== 'asr') {
    errors.push('id and task must identify the Voxtral Realtime MLX reference');
  }
  for (const revision of [
    modelManifest.source?.revision,
    modelManifest.conversion?.revision,
    modelManifest.reference_runtime?.revision,
  ]) {
    if (!/^[0-9a-f]{40}$/.test(revision ?? '')) {
      errors.push(`invalid pinned revision: ${revision}`);
    }
  }
  if (modelManifest.source?.license !== 'Apache-2.0' ||
      modelManifest.conversion?.license !== 'Apache-2.0' ||
      modelManifest.reference_runtime?.license !== 'MIT' ||
      modelManifest.reference_runtime?.version !== '0.4.5') {
    errors.push('Voxtral Realtime model and runtime licenses must be explicit');
  }
  const contract = modelManifest.inference_contract;
  if (contract?.sample_rate_hz !== 16000 ||
      contract?.sample_format !== 'f32le' ||
      contract?.channel_count !== 1 ||
      contract?.temperature !== 0 ||
      !arrayEquals(contract?.transcription_delay_ms ?? [], [480, 2400]) ||
      contract?.repetitions !== 2 ||
      contract?.warmup_runs !== 1 ||
      contract?.max_tokens !== 4096) {
    errors.push('inference contract must preserve both greedy delay controls');
  }
  if (modelManifest.source?.supported_languages?.length !== 13 ||
      !modelManifest.source.supported_languages.includes('German') ||
      !modelManifest.source.supported_languages.includes('French') ||
      !modelManifest.source.supported_languages.includes('Portuguese') ||
      modelManifest.architecture?.causal_audio_encoder !== true ||
      modelManifest.architecture?.adaptive_delay_conditioning !== true) {
    errors.push('multilingual causal streaming architecture must be recorded');
  }

  const artifactPaths = new Set();
  let artifactBytes = 0;
  for (const artifact of modelManifest.artifacts ?? []) {
    if (!(artifact.path?.length > 0) || artifactPaths.has(artifact.path)) {
      errors.push(`invalid or duplicate Voxtral artifact: ${artifact.path}`);
    }
    artifactPaths.add(artifact.path);
    if (!(Number.isInteger(artifact.bytes) && artifact.bytes > 0) ||
        !/^[0-9a-f]{64}$/.test(artifact.sha256 ?? '')) {
      errors.push(`${artifact.path}: invalid byte count or SHA-256`);
    }
    artifactBytes += artifact.bytes ?? 0;
  }
  for (const requiredPath of [
    'config.json',
    'model.safetensors',
    'model.safetensors.index.json',
    'tekken.json',
  ]) {
    if (!artifactPaths.has(requiredPath)) {
      errors.push(`missing required Voxtral artifact: ${requiredPath}`);
    }
  }
  if (artifactBytes !== modelManifest.conversion?.snapshot_bytes ||
      modelManifest.conversion?.weight_bytes !== 3133798126) {
    errors.push('Voxtral Realtime bytes do not reconcile with the snapshot');
  }
  return errors;
}

function validateVoxtralRealtimeDirectModelManifest(
  modelManifest,
  referenceManifest,
) {
  const errors = [];
  if (modelManifest.id !== referenceManifest.id ||
      modelManifest.task !== 'asr' ||
      modelManifest.source?.revision !== referenceManifest.source?.revision ||
      modelManifest.conversion?.revision !==
        referenceManifest.conversion?.revision) {
    errors.push('direct manifest must preserve the pinned Voxtral model');
  }
  if (modelManifest.source?.license !== 'Apache-2.0' ||
      modelManifest.conversion?.license !== 'Apache-2.0' ||
      modelManifest.official_runtime?.repository !== 'ml-explore/mlx' ||
      modelManifest.official_runtime?.revision !==
        '7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247' ||
      modelManifest.official_runtime?.version !== '0.32.0' ||
      modelManifest.official_runtime?.license !== 'MIT') {
    errors.push('direct runtime and licenses must pin official MLX v0.32.0');
  }

  const artifactPaths = new Set();
  let artifactBytes = 0;
  for (const artifact of modelManifest.artifacts ?? []) {
    if (!(artifact.path?.length > 0) || artifactPaths.has(artifact.path) ||
        !(Number.isInteger(artifact.bytes) && artifact.bytes > 0) ||
        !/^[0-9a-f]{64}$/.test(artifact.sha256 ?? '')) {
      errors.push(`invalid direct Voxtral artifact: ${artifact.path}`);
    }
    artifactPaths.add(artifact.path);
    artifactBytes += artifact.bytes ?? 0;
  }
  for (const requiredPath of [
    'config.json',
    'model.safetensors',
    'model.safetensors.index.json',
    'tekken.json',
  ]) {
    if (!artifactPaths.has(requiredPath)) {
      errors.push(`missing required direct Voxtral artifact: ${requiredPath}`);
    }
  }
  if (artifactBytes !== modelManifest.conversion?.snapshot_bytes ||
      modelManifest.conversion?.snapshot_bytes !==
        referenceManifest.conversion?.snapshot_bytes ||
      modelManifest.conversion?.weight_bytes !== 3133798126) {
    errors.push('direct Voxtral artifact bytes must reconcile with the pin');
  }

  const layout = modelManifest.expected_layout;
  if (layout?.tensor_count !== 1523 ||
      layout?.float32_tensors !== 300 ||
      layout?.float16_tensors !== 817 ||
      layout?.uint32_tensors !== 406 ||
      layout?.affine_4bit_modules !== 406) {
    errors.push('direct Voxtral tensor and quantization layout must stay pinned');
  }
  const batchContract = modelManifest.batch_contract;
  if (batchContract?.sample_rate_hz !== 16000 ||
      batchContract?.sample_format !== 'f32le' ||
      batchContract?.channel_count !== 1 ||
      batchContract?.input_semantics !== 'complete-audio-buffer' ||
      batchContract?.generation !== 'greedy' ||
      batchContract?.tokenizer !== 'tekken' ||
      batchContract?.output !== 'final-text-and-token-evidence' ||
      batchContract?.capabilities?.delay_conditioning !== true ||
      batchContract?.capabilities?.decoder !== true ||
      batchContract?.capabilities?.decoder_kv_cache !== true ||
      batchContract?.capabilities?.transcription !== true ||
      batchContract?.capabilities?.streaming_session !== false ||
      batchContract?.capabilities?.timestamps !== false) {
    errors.push('direct Voxtral batch transcription contract must stay explicit');
  }
  const contract = modelManifest.adapter_contract;
  if (contract?.sample_rate_hz !== 16000 ||
      contract?.sample_format !== 'f32le' ||
      contract?.channel_count !== 1 ||
      contract?.feed_semantics !== 'all-or-nothing' ||
      contract?.queue_capacity !== 'caller-configured-hard-bound' ||
      contract?.step_budget !== 'caller-configured-hard-bound' ||
      contract?.decode_budget !== 'caller-configured-hard-bound' ||
      contract?.end_of_audio !== 'explicit-close' ||
      contract?.output !== 'append-only-text-deltas-and-final-text' ||
      contract?.stable_statuses?.invalid_argument !== 1 ||
      contract?.stable_statuses?.cancelled !== 3 ||
      contract?.stable_statuses?.busy !== 4 ||
      contract?.stable_statuses?.backpressure !== 5 ||
      contract?.stable_statuses?.needs_audio !== 6 ||
      contract?.stable_statuses?.done !== 7 ||
      contract?.capabilities?.bounded_ingestion !== true ||
      contract?.capabilities?.backpressure !== true ||
      contract?.capabilities?.cancellation !== true ||
      contract?.capabilities?.mel_frontend !== true ||
      contract?.capabilities?.causal_conv_stem !== true ||
      contract?.capabilities?.causal_encoder !== true ||
      contract?.capabilities?.rotating_kv_cache !== true ||
      contract?.capabilities?.sliding_window_attention !== true ||
      contract?.capabilities?.adapter_projection !== true ||
      contract?.capabilities?.decoder !== true ||
      contract?.capabilities?.transcription !== true ||
      contract?.capabilities?.streaming_session !== true) {
    errors.push('direct Voxtral boundary contract must remain explicit and bounded');
  }
  return errors;
}

function validateVoxtralRealtimeDirectBoundary(
  run,
  modelManifest,
  audiobookManifest,
) {
  const errors = [];
  const fixture = audiobookManifest.fixtures.find(
    (candidate) => candidate.id === 'audiobook-de-135_82_000105',
  );
  if (run.schema_version !== schemaVersion ||
      !/^\d{4}-\d{2}-\d{2}$/.test(run.recorded_at ?? '') ||
      run.candidate?.id !== 'voxtral-mini-4b-realtime-mlx-direct' ||
      !run.candidate?.model?.endsWith(
        `@${modelManifest.conversion?.revision}`,
      ) ||
      !run.candidate?.runtime?.endsWith(
        `@${modelManifest.official_runtime?.revision}`,
      ) ||
      !run.candidate?.boundary?.includes('repository-owned')) {
    errors.push('direct boundary run must pin its model, runtime, and ownership');
  }
  if (!fixture ||
      run.fixture?.manifest_revision !== audiobookManifest.revision ||
      run.fixture?.id !== fixture.id ||
      run.fixture?.language !== fixture.language ||
      run.fixture?.gold_status !== fixture.gold_status ||
      run.fixture?.audio_sha256 !== fixture.normalized.sha256 ||
      run.fixture?.sample_rate_hz !== 16000 ||
      run.fixture?.sample_format !== 'f32le' ||
      run.fixture?.channel_count !== 1 ||
      run.fixture?.sample_count !== fixture.normalized.sample_count ||
      run.fixture?.duration_ms !== fixture.normalized.duration_ms) {
    errors.push('direct boundary fixture must match the pinned audiobook PCM');
  }

  const contract = run.contract;
  const expectedStepCount = Math.ceil(
    run.fixture?.sample_count /
      modelManifest.adapter_contract?.sample_rate_hz * 1000 /
      contract?.max_ingest_ms_per_step,
  );
  if (contract?.input_chunk_samples !== 1280 ||
      contract?.input_chunk_ms !== 80 ||
      contract?.queue_capacity_samples !== 10240 ||
      contract?.queue_capacity_ms !== 640 ||
      contract?.max_ingest_samples_per_step !== 5120 ||
      contract?.max_ingest_ms_per_step !== 320 ||
      contract?.total_fed_samples !== run.fixture?.sample_count ||
      contract?.total_ingested_samples !== run.fixture?.sample_count ||
      contract?.step_count !== expectedStepCount ||
      contract?.maximum_ingested_samples >
        contract?.max_ingest_samples_per_step ||
      !(contract?.backpressure_count > 0) ||
      contract?.stable_statuses?.invalid_argument !== 1 ||
      contract?.stable_statuses?.cancelled !== 3 ||
      contract?.stable_statuses?.backpressure !== 5 ||
      contract?.stable_statuses?.needs_audio !== 6 ||
      contract?.stable_statuses?.done !== 7) {
    errors.push('direct boundary run must preserve bounded ingestion and states');
  }
  if (!(contract?.repetitions?.length >= 2) ||
      contract.repetitions.some(
        (repetition) =>
          repetition.step_count !== expectedStepCount ||
          !(repetition.backpressure_count > 0) ||
          !(repetition.maximum_step_wall_ms >=
            repetition.maximum_mlx_elapsed_ms) ||
          !(repetition.maximum_step_wall_ms < 250),
      )) {
    errors.push('direct boundary repetitions must remain bounded below 250 ms');
  }

  const fingerprint = run.official_mlx_fingerprint;
  const calculatedRelativeError = Math.abs(
    fingerprint?.mlx_sum_squares - fingerprint?.cpu_expected_sum_squares,
  ) / fingerprint?.cpu_expected_sum_squares;
  if (!(fingerprint?.cpu_expected_sum_squares > 0) ||
      !(fingerprint?.mlx_sum_squares > 0) ||
      !(fingerprint?.relative_error <= 5e-5) ||
      Math.abs(fingerprint.relative_error - calculatedRelativeError) > 1e-12) {
    errors.push('direct official-MLX audio fingerprint must match its CPU control');
  }
  if (run.capabilities?.model_layout_validated !== true ||
      run.capabilities?.bounded_ingestion !== true ||
      run.capabilities?.backpressure !== true ||
      run.capabilities?.cancellation !== true ||
      run.capabilities?.transcription !== false ||
      !(run.artifacts?.shim_dylib_bytes > 0) ||
      !(run.artifacts?.rust_executable_bytes > 0) ||
      !(run.artifacts?.metallib_bytes > 0) ||
      run.artifacts?.model_safetensors_bytes !==
        modelManifest.conversion?.weight_bytes ||
      !run.limitations?.some((item) => item.includes('No transcript'))) {
    errors.push('direct boundary capabilities must not overclaim transcription');
  }
  return errors;
}

function validateVoxtralRealtimeDirectFrontend(
  run,
  oracle,
  modelManifest,
  audiobookManifest,
) {
  const errors = [];
  const fixture = audiobookManifest.fixtures.find(
    (candidate) => candidate.id === 'audiobook-de-135_82_000105',
  );
  if (run.schema_version !== schemaVersion ||
      !/^\d{4}-\d{2}-\d{2}T/.test(run.captured_at ?? '') ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '') ||
      run.issue !==
        'https://github.com/sebastian-software/cuttledoc-rs/issues/19' ||
      run.model?.source !== oracle.model?.source ||
      run.model?.conversion !== oracle.model?.conversion ||
      run.model?.artifact_sha256 !== oracle.model?.artifact_sha256 ||
      !run.toolchain?.mlx?.revision?.endsWith(
        modelManifest.official_runtime?.revision,
      )) {
    errors.push('direct frontend identity must pin source, model, and MLX');
  }
  if (!fixture ||
      run.fixture?.id !== fixture.id ||
      run.fixture?.pcm_sha256 !== fixture.normalized.sha256 ||
      run.fixture?.pcm_samples !== fixture.normalized.sample_count ||
      run.fixture?.sample_rate_hz !== 16000 ||
      run.fixture?.gold_status !== fixture.gold_status ||
      oracle.fixture?.id !== fixture.id ||
      oracle.fixture?.pcm_sha256 !== fixture.normalized.sha256 ||
      oracle.fixture?.pcm_samples !== fixture.normalized.sample_count) {
    errors.push('direct frontend must use the pinned audiobook fixture');
  }

  const result = run.result;
  const probe = result?.probe;
  if (result?.status !== 'ok' ||
      result?.reference_oracle !==
        'benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.frontend-480ms.json' ||
      probe?.status !== 'ok' ||
      probe?.boundary !== 'official-mlx-cpp' ||
      probe?.stage !== 'voxtral-audio-frontend' ||
      probe?.device !== 'gpu' ||
      probe?.pcm_samples !== fixture?.normalized.sample_count ||
      !(probe?.elapsed_ms > 0) ||
      !(probe?.peak_memory_bytes > 0)) {
    errors.push('direct frontend probe must record successful official-MLX GPU work');
  }
  for (const field of [
    'transcription_delay_ms',
    'delay_tokens',
    'left_pad_tokens',
    'left_pad_samples',
    'alignment_pad_samples',
    'right_pad_tokens',
    'right_pad_samples',
    'padded_samples',
  ]) {
    if (probe?.padding?.[field] !== oracle.padding?.[field]) {
      errors.push(`direct frontend padding.${field} differs from oracle`);
    }
  }
  if (probe?.mel_frames !== oracle.mel_frames ||
      probe?.front_truncation_frames !== oracle.front_truncation_frames ||
      probe?.capabilities?.mel_frontend !== true ||
      probe?.capabilities?.causal_conv_stem !== true ||
      probe?.capabilities?.causal_encoder !== false ||
      probe?.capabilities?.transcription !== false ||
      oracle.capabilities?.causal_encoder !== false ||
      oracle.capabilities?.transcription !== false) {
    errors.push('direct frontend shapes and capability ceiling must match oracle');
  }

  let maximumSampleAbsoluteError = 0;
  let maximumAggregateRelativeError = 0;
  for (const stage of [
    'mel_filters',
    'log_mel',
    'conv0_gelu',
    'conv1_pretrunc_gelu',
    'conv_stem',
  ]) {
    const expected = oracle.fingerprints?.[stage];
    const observed = probe?.fingerprints?.[stage];
    if (!expected || !observed ||
        !arrayEquals(expected.shape ?? [], observed.shape ?? []) ||
        !arrayEquals(
          expected.sample_indices ?? [],
          observed.sample_indices ?? [],
        ) ||
        expected.sample_values?.length !== observed.sample_values?.length) {
      errors.push(`${stage}: direct frontend fingerprint identity differs`);
      continue;
    }
    for (let index = 0; index < expected.sample_values.length; index += 1) {
      maximumSampleAbsoluteError = Math.max(
        maximumSampleAbsoluteError,
        Math.abs(observed.sample_values[index] - expected.sample_values[index]),
      );
    }
    for (const field of ['mean', 'stddev', 'minimum', 'maximum', 'l1']) {
      const relativeError = Math.abs(observed[field] - expected[field]) /
        Math.max(Math.abs(expected[field]), 1e-12);
      maximumAggregateRelativeError = Math.max(
        maximumAggregateRelativeError,
        relativeError,
      );
    }
  }
  const parity = result?.parity;
  if (parity?.status !== 'ok' ||
      parity?.sample_absolute_tolerance !== 2e-6 ||
      parity?.aggregate_relative_tolerance !== 2e-5 ||
      maximumSampleAbsoluteError > parity.sample_absolute_tolerance ||
      maximumAggregateRelativeError > parity.aggregate_relative_tolerance ||
      Math.abs(
        parity.maximum_sample_absolute_error - maximumSampleAbsoluteError,
      ) > 1e-15 ||
      Math.abs(
        parity.maximum_aggregate_relative_error -
          maximumAggregateRelativeError,
      ) > 1e-15) {
    errors.push('direct frontend parity must be derived within pinned tolerances');
  }
  if (result?.cpu_smoke?.status !== 'ok' ||
      result?.cpu_smoke?.mel_frames !== oracle.mel_frames ||
      result?.cpu_smoke?.conv_stem_frames !== 864 ||
      !(result?.cpu_smoke?.elapsed_ms > 0) ||
      !(result?.cpu_smoke?.peak_memory_bytes > 0) ||
      !(run.artifacts?.shim_dylib_bytes > 0) ||
      !(run.artifacts?.rust_probe_bytes > 0) ||
      !(run.artifacts?.mlx_metallib_bytes > 0) ||
      !run.limitations?.some((item) => item.includes('No token'))) {
    errors.push('direct frontend CPU/artifact evidence and limits must be explicit');
  }
  return errors;
}

function validateVoxtralRealtimeDirectEncoder(
  run,
  oracle,
  modelManifest,
  audiobookManifest,
) {
  const errors = [];
  const fixture = audiobookManifest.fixtures.find(
    (candidate) => candidate.id === 'audiobook-de-135_82_000105',
  );
  if (run.schema_version !== schemaVersion ||
      !/^\d{4}-\d{2}-\d{2}T/.test(run.captured_at ?? '') ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '') ||
      run.issue !==
        'https://github.com/sebastian-software/cuttledoc-rs/issues/19' ||
      run.model?.source !== oracle.model?.source ||
      run.model?.conversion !== oracle.model?.conversion ||
      run.model?.artifact_sha256 !== oracle.model?.artifact_sha256 ||
      run.toolchain?.mlx?.revision !==
        modelManifest.official_runtime?.revision) {
    errors.push('direct encoder identity must pin source, model, and MLX');
  }
  if (!fixture ||
      run.fixture?.id !== fixture.id ||
      run.fixture?.pcm_sha256 !== fixture.normalized.sha256 ||
      run.fixture?.pcm_samples !== fixture.normalized.sample_count ||
      run.fixture?.sample_rate_hz !== 16000 ||
      run.fixture?.gold_status !== fixture.gold_status ||
      oracle.fixture?.id !== fixture.id ||
      oracle.fixture?.pcm_sha256 !== fixture.normalized.sha256 ||
      oracle.fixture?.pcm_samples !== fixture.normalized.sample_count) {
    errors.push('direct encoder must use the pinned audiobook fixture');
  }

  const result = run.result;
  const probe = result?.probe;
  if (result?.status !== 'ok' ||
      result?.reference_oracle !==
        'benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.encoder-480ms.json' ||
      probe?.status !== 'ok' ||
      probe?.boundary !== 'official-mlx-cpp' ||
      probe?.stage !== 'voxtral-causal-encoder' ||
      probe?.device !== 'gpu' ||
      probe?.pcm_samples !== fixture?.normalized.sample_count ||
      probe?.transcription_delay_ms !== 480 ||
      probe?.delay_tokens !== 6 ||
      !(probe?.elapsed_ms > 0) ||
      !(probe?.peak_memory_bytes > 0)) {
    errors.push('direct encoder probe must record successful official-MLX GPU work');
  }
  for (const field of [
    'layers',
    'dimension',
    'attention_heads',
    'head_dimension',
    'sliding_window',
    'downsample_factor',
    'adapter_dimension',
  ]) {
    if (probe?.architecture?.[field] !== oracle.architecture?.[field]) {
      errors.push(`direct encoder architecture.${field} differs from oracle`);
    }
  }
  if (JSON.stringify(probe?.chunks) !== JSON.stringify(oracle.chunks)) {
    errors.push('direct encoder chunk and mask structure differs from oracle');
  }
  for (const field of [
    'layer_0_offset',
    'layer_0_size',
    'layer_0_materialized_key_frames',
    'layer_0_materialized_value_frames',
  ]) {
    if (probe?.cache?.[field] !== oracle.cache?.[field]) {
      errors.push(`direct encoder cache.${field} differs from oracle`);
    }
  }
  if (probe?.output?.encoded_frames !== oracle.output?.encoded_frames ||
      probe?.output?.adapter_frames !== oracle.output?.adapter_frames ||
      probe?.capabilities?.causal_encoder !== true ||
      probe?.capabilities?.rotating_kv_cache !== true ||
      probe?.capabilities?.sliding_window_attention !== true ||
      probe?.capabilities?.adapter_projection !== true ||
      probe?.capabilities?.decoder !== false ||
      probe?.capabilities?.transcription !== false ||
      modelManifest.adapter_contract?.capabilities?.causal_encoder !== true ||
      modelManifest.adapter_contract?.capabilities?.decoder !== true) {
    errors.push('direct encoder output and capability ceiling must match oracle');
  }

  let maximumSampleAbsoluteError = 0;
  let maximumAggregateRelativeError = 0;
  for (const stage of [
    'conv_stem',
    'chunk_0_layer_0',
    'chunk_0_layer_15',
    'chunk_0_layer_31',
    'chunk_0_norm',
    'chunk_1_layer_0',
    'chunk_1_layer_15',
    'chunk_1_layer_31',
    'chunk_1_norm',
    'encoded',
    'adapter_projection0_gelu',
    'adapter',
    'layer_0_cache_keys',
    'layer_0_cache_values',
  ]) {
    const expected = oracle.fingerprints?.[stage];
    const observed = probe?.fingerprints?.[stage];
    if (!expected || !observed ||
        !arrayEquals(expected.shape ?? [], observed.shape ?? []) ||
        !arrayEquals(
          expected.sample_indices ?? [],
          observed.sample_indices ?? [],
        ) ||
        expected.sample_values?.length !== observed.sample_values?.length) {
      errors.push(`${stage}: direct encoder fingerprint identity differs`);
      continue;
    }
    for (let index = 0; index < expected.sample_values.length; index += 1) {
      maximumSampleAbsoluteError = Math.max(
        maximumSampleAbsoluteError,
        Math.abs(observed.sample_values[index] - expected.sample_values[index]),
      );
    }
    for (const field of ['mean', 'stddev', 'minimum', 'maximum', 'l1']) {
      const relativeError = Math.abs(observed[field] - expected[field]) /
        Math.max(Math.abs(expected[field]), 1e-12);
      maximumAggregateRelativeError = Math.max(
        maximumAggregateRelativeError,
        relativeError,
      );
    }
  }
  const parity = result?.parity;
  if (parity?.status !== 'ok' ||
      parity?.sample_absolute_tolerance !== 5e-5 ||
      parity?.aggregate_relative_tolerance !== 1e-5 ||
      maximumSampleAbsoluteError > parity.sample_absolute_tolerance ||
      maximumAggregateRelativeError > parity.aggregate_relative_tolerance ||
      Math.abs(
        parity.maximum_sample_absolute_error - maximumSampleAbsoluteError,
      ) > 1e-15 ||
      Math.abs(
        parity.maximum_aggregate_relative_error -
          maximumAggregateRelativeError,
      ) > 1e-15) {
    errors.push('direct encoder parity must be derived within pinned tolerances');
  }
  if (result?.cpu_smoke?.status !== 'ok' ||
      result?.cpu_smoke?.encoded_frames !== oracle.output?.encoded_frames ||
      result?.cpu_smoke?.adapter_frames !== oracle.output?.adapter_frames ||
      result?.cpu_smoke?.cache_size !== oracle.cache?.layer_0_size ||
      result?.cpu_smoke?.materialized_cache_frames !==
        oracle.cache?.layer_0_materialized_key_frames ||
      !(result?.cpu_smoke?.elapsed_ms > 0) ||
      !(result?.cpu_smoke?.peak_memory_bytes > 0) ||
      !(run.artifacts?.shim_dylib_bytes > 0) ||
      !(run.artifacts?.rust_probe_bytes > 0) ||
      !(run.artifacts?.mlx_metallib_bytes > 0) ||
      !run.limitations?.some((item) => item.includes('No token'))) {
    errors.push('direct encoder CPU/artifact evidence and limits must be explicit');
  }
  return errors;
}

function validateVoxtralRealtimeDirectTranscription(
  run,
  oracle,
  modelManifest,
  audiobookManifest,
) {
  const errors = [];
  const fixture = audiobookManifest.fixtures.find(
    (candidate) => candidate.id === 'audiobook-de-135_82_000105',
  );
  if (run.schema_version !== schemaVersion ||
      !/^\d{4}-\d{2}-\d{2}T/.test(run.captured_at ?? '') ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '') ||
      run.issue !==
        'https://github.com/sebastian-software/cuttledoc-rs/issues/19' ||
      run.model?.source !== oracle.model?.source ||
      run.model?.conversion !== oracle.model?.conversion ||
      run.model?.artifact_sha256 !== oracle.model?.artifact_sha256 ||
      run.toolchain?.mlx?.revision !==
        modelManifest.official_runtime?.revision) {
    errors.push('direct transcription identity must pin source, model, and MLX');
  }
  if (!fixture ||
      run.fixture?.id !== fixture.id ||
      run.fixture?.pcm_sha256 !== fixture.normalized.sha256 ||
      run.fixture?.pcm_samples !== fixture.normalized.sample_count ||
      run.fixture?.sample_rate_hz !== 16000 ||
      run.fixture?.duration_ms !== fixture.normalized.duration_ms ||
      run.fixture?.gold_status !== fixture.gold_status ||
      oracle.fixture?.id !== fixture.id ||
      oracle.fixture?.pcm_sha256 !== fixture.normalized.sha256 ||
      oracle.fixture?.pcm_samples !== fixture.normalized.sample_count) {
    errors.push('direct transcription must use the pinned audiobook fixture');
  }

  const result = run.result;
  const probe = result?.probe;
  if (result?.status !== 'ok' ||
      result?.reference_oracle !==
        'benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.decoder-480ms.json' ||
      probe?.status !== 'ok' ||
      probe?.boundary !== 'official-mlx-cpp' ||
      probe?.stage !== 'voxtral-greedy-transcription' ||
      probe?.device !== 'gpu' ||
      probe?.pcm_samples !== fixture?.normalized.sample_count ||
      probe?.transcription_delay_ms !== 480 ||
      probe?.delay_tokens !== 6 ||
      !(probe?.elapsed_ms > 0) ||
      !(probe?.peak_memory_bytes > 0)) {
    errors.push('direct transcription must record successful official-MLX GPU work');
  }
  for (const field of [
    'layers',
    'dimension',
    'attention_heads',
    'kv_heads',
    'head_dimension',
    'hidden_dimension',
    'sliding_window',
    'vocabulary_size',
    'ada_bottleneck_dimension',
  ]) {
    if (probe?.architecture?.[field] !== oracle.architecture?.[field]) {
      errors.push(`direct transcription architecture.${field} differs from oracle`);
    }
  }
  if (JSON.stringify(probe?.prompt) !== JSON.stringify(oracle.prompt)) {
    errors.push('direct delay-conditioned prompt differs from oracle');
  }
  if (JSON.stringify(probe?.generation?.tokens) !==
        JSON.stringify(oracle.generation?.tokens) ||
      probe?.generation?.token_count !== oracle.generation?.token_count ||
      probe?.generation?.forward_steps !== oracle.generation?.forward_steps ||
      probe?.generation?.finish_reason !== oracle.generation?.finish_reason ||
      probe?.generation?.text !== oracle.generation?.text) {
    errors.push('direct generated tokens and text must exactly match the oracle');
  }
  if (JSON.stringify(probe?.cache) !== JSON.stringify(oracle.cache)) {
    errors.push('direct decoder cache state differs from oracle');
  }
  if (probe?.capabilities?.delay_conditioning !== true ||
      probe?.capabilities?.decoder !== true ||
      probe?.capabilities?.decoder_kv_cache !== true ||
      probe?.capabilities?.tekken_decode !== true ||
      probe?.capabilities?.greedy_transcription !== true ||
      probe?.capabilities?.streaming_session !== false ||
      modelManifest.batch_contract?.capabilities?.transcription !== true ||
      modelManifest.batch_contract?.capabilities?.streaming_session !== false ||
      modelManifest.adapter_contract?.capabilities?.transcription !== true) {
    errors.push('batch transcription must not overclaim the streaming session');
  }

  let maximumSampleAbsoluteError = 0;
  let maximumAggregateRelativeError = 0;
  for (const stage of [
    'adapter',
    'time_embedding',
    'ada_scale_layer_0',
    'ada_scale_layer_12',
    'ada_scale_layer_25',
    'prompt_text_embeddings',
    'prefix_embeddings',
    'prefill_layer_0',
    'prefill_layer_12',
    'prefill_layer_25',
    'prefill_norm',
    'prefill_logits',
    'decode_0_input',
    'decode_0_layer_0',
    'decode_0_layer_12',
    'decode_0_layer_25',
    'decode_0_norm',
    'decode_0_logits',
    'final_logits',
    'decoder_layer_0_cache_keys',
    'decoder_layer_0_cache_values',
  ]) {
    const expected = oracle.fingerprints?.[stage];
    const observed = probe?.fingerprints?.[stage];
    if (!expected || !observed ||
        !arrayEquals(expected.shape ?? [], observed.shape ?? []) ||
        !arrayEquals(
          expected.sample_indices ?? [],
          observed.sample_indices ?? [],
        ) ||
        expected.sample_values?.length !== observed.sample_values?.length) {
      errors.push(`${stage}: direct decoder fingerprint identity differs`);
      continue;
    }
    for (let index = 0; index < expected.sample_values.length; index += 1) {
      maximumSampleAbsoluteError = Math.max(
        maximumSampleAbsoluteError,
        Math.abs(observed.sample_values[index] - expected.sample_values[index]),
      );
    }
    for (const field of ['mean', 'stddev', 'minimum', 'maximum', 'l1']) {
      const relativeError = Math.abs(observed[field] - expected[field]) /
        Math.max(Math.abs(expected[field]), 1e-12);
      maximumAggregateRelativeError = Math.max(
        maximumAggregateRelativeError,
        relativeError,
      );
    }
  }
  const parity = result?.parity;
  if (parity?.status !== 'ok' ||
      parity?.exact_token_parity !== true ||
      parity?.exact_text_parity !== true ||
      parity?.sample_absolute_tolerance !== 5e-4 ||
      parity?.aggregate_relative_tolerance !== 5e-4 ||
      maximumSampleAbsoluteError > parity.sample_absolute_tolerance ||
      maximumAggregateRelativeError > parity.aggregate_relative_tolerance ||
      Math.abs(
        parity.maximum_sample_absolute_error - maximumSampleAbsoluteError,
      ) > 1e-15 ||
      Math.abs(
        parity.maximum_aggregate_relative_error -
          maximumAggregateRelativeError,
      ) > 1e-15) {
    errors.push('direct decoder parity must be exact and within tolerances');
  }
  const performance = result?.performance;
  const calculatedRtf = probe?.elapsed_ms / run.fixture?.duration_ms;
  if (performance?.inference_elapsed_ms !== probe?.elapsed_ms ||
      performance?.peak_memory_bytes !== probe?.peak_memory_bytes ||
      Math.abs(performance?.real_time_factor - calculatedRtf) > 1e-15 ||
      !(run.artifacts?.shim_dylib_bytes > 0) ||
      !(run.artifacts?.rust_executable_bytes > 0) ||
      !(run.artifacts?.mlx_metallib_bytes > 0) ||
      !run.limitations?.some((item) => item.includes('not WER')) ||
      !run.limitations?.some((item) => item.includes('does not yet execute'))) {
    errors.push('direct decoder performance, artifacts, and limits must be explicit');
  }
  return errors;
}

function validateVoxtralRealtimeAggregate(
  aggregate,
  modelManifest,
  delayMs,
) {
  const errors = [];
  const expectedId =
    `voxtral-mini-4b-realtime-mlx-reference-${delayMs}ms`;
  if (aggregate.candidate?.id !== expectedId ||
      !aggregate.candidate?.model?.includes(
        `@${modelManifest.conversion?.revision}`,
      ) ||
      !aggregate.candidate?.model?.includes(
        `@${modelManifest.source?.revision}`,
      ) ||
      !aggregate.candidate?.runtime?.includes(
        `@${modelManifest.reference_runtime?.revision}`,
      ) ||
      aggregate.procedure?.transcription_delay_ms !== delayMs ||
      aggregate.procedure?.live_input_streaming_measured !== false) {
    errors.push('candidate and procedure must pin the selected delay/runtime');
  }
  if ((aggregate.results ?? []).some(
    (result) =>
      result.streaming?.supported !== true ||
      result.streaming?.input_streaming_measured !== false ||
      result.streaming?.transcription_delay_ms !== delayMs ||
      result.repetitions?.length !==
        modelManifest.inference_contract?.repetitions ||
      result.repetitions.some(
        (sample) => sample.text !== result.repetitions[0]?.text,
      ) ||
      !(result.resources?.runtime_peak_memory_bytes > 0),
  )) {
    errors.push(
      'each result must be deterministic and distinguish available from measured streaming',
    );
  }
  return errors;
}

function validateVoxtralRealtimeStreamingRun(
  run,
  modelManifest,
  audiobookManifest,
  chunkMs,
) {
  const errors = [];
  const fixture = audiobookManifest.fixtures.find(
    (candidate) => candidate.id === 'audiobook-de-135_82_000105',
  );
  if (run.schema_version !== schemaVersion ||
      !/^\d{4}-\d{2}-\d{2}T/.test(run.captured_at ?? '') ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '')) {
    errors.push('streaming run identity must pin capture time and source');
  }
  if (run.candidate?.id !==
        'voxtral-mini-4b-realtime-mlx-reference' ||
      !run.candidate?.model?.includes(
        `@${modelManifest.conversion?.revision}`,
      ) ||
      !run.candidate?.model?.includes(
        `@${modelManifest.source?.revision}`,
      ) ||
      !run.candidate?.runtime?.includes(
        `@${modelManifest.reference_runtime?.revision}`,
      )) {
    errors.push('streaming candidate must pin the selected model and runtime');
  }
  if (!fixture ||
      run.fixture?.id !== fixture.id ||
      run.fixture?.language !== fixture.language ||
      run.fixture?.gold_status !== fixture.gold_status ||
      run.fixture?.reference_text !== fixture.reference_text ||
      run.fixture?.audio_sha256 !== fixture.normalized.sha256 ||
      run.fixture?.sample_rate_hz !==
        modelManifest.inference_contract?.sample_rate_hz ||
      run.fixture?.sample_count !== fixture.normalized.sample_count ||
      run.fixture?.duration_ms !== fixture.normalized.duration_ms) {
    errors.push('streaming fixture must match the pinned audiobook PCM');
  }
  if (run.procedure?.repetitions !== 2 ||
      !arrayEquals(
        run.procedure?.transcription_delays_ms ?? [],
        modelManifest.inference_contract?.transcription_delay_ms ?? [],
      ) ||
      run.procedure?.chunk_ms !== chunkMs ||
      run.procedure?.max_decode_tokens_per_step !== 16 ||
      run.procedure?.max_tokens !==
        modelManifest.inference_contract?.max_tokens ||
      run.procedure?.realtime_pacing !== true ||
      run.procedure?.first_chunk_available_after_capture !== true ||
      run.procedure?.producer_thread_independent_from_mlx_executor !== true ||
      run.procedure?.consumer_step_after_feed_signal !== true ||
      run.procedure?.feed_signals_may_coalesce_while_mlx_step_runs !== true ||
      run.procedure?.additional_step_calls_after_close_until_done !== true) {
    errors.push('streaming procedure must preserve the live-input contract');
  }
  for (const field of ['feed', 'step', 'close', 'done', 'generated']) {
    if (run.api_contract?.[field] !== true) {
      errors.push(`streaming API ${field} must remain available`);
    }
  }
  if (run.api_contract?.cancel !== false ||
      run.api_contract?.finalize_step !== false ||
      !run.api_contract?.close_semantics?.includes('not cancellation') ||
      run.abandonment_probe?.native_cancel_available !== false ||
      run.abandonment_probe?.done !== false ||
      !run.abandonment_probe?.claim?.includes(
        'not cooperative cancellation',
      )) {
    errors.push('streaming cancellation gap must remain explicit');
  }

  const expectedPairs = [
    '480:1',
    '480:2',
    '2400:1',
    '2400:2',
  ];
  const actualPairs = (run.runs ?? []).map(
    (result) =>
      `${result.transcription_delay_ms}:${result.repetition}`,
  );
  if (!arrayEquals(actualPairs, expectedPairs)) {
    errors.push('streaming runs must cover both delays and repetitions');
  }
  const expectedChunkCount = Math.ceil(
    fixture.normalized.sample_count /
    (chunkMs * modelManifest.inference_contract.sample_rate_hz / 1000),
  );
  for (const result of run.runs ?? []) {
    if (!(result.text?.length > 0) ||
        result.quality?.wer !== 0 ||
        result.quality?.cer !== 0 ||
        result.streaming?.input_streaming_measured !== true ||
        result.streaming?.append_only !== true ||
        result.streaming?.revoke_count !== 0 ||
        result.streaming?.audio_chunk_count !== expectedChunkCount ||
        result.streaming?.done !== true ||
        !(result.streaming?.generated_tokens > 0) ||
        result.events?.length !== result.streaming?.update_count ||
        result.step_trace?.length !== result.streaming?.step_call_count ||
        !(result.resources?.runtime_peak_memory_bytes > 0)) {
      errors.push(
        `${result.transcription_delay_ms}:${result.repetition}: ` +
        'invalid streaming result',
      );
      continue;
    }
    if ((result.events ?? []).some(
      (event, index) =>
        event.index !== index ||
        !(event.delta?.length > 0) ||
        !isMetric(event.elapsed_ms) ||
        !isMetric(event.audio_fed_before_step_ms) ||
        !isMetric(event.audio_fed_at_emit_ms),
    )) {
      errors.push(
        `${result.transcription_delay_ms}:${result.repetition}: ` +
        'invalid append trace',
      );
    }
    if ((result.step_trace ?? []).some(
      (step, index) =>
        step.index !== index ||
        !isMetric(step.started_ms) ||
        !isMetric(step.duration_ms) ||
        !isMetric(step.audio_fed_before_step_ms) ||
        !isMetric(step.audio_fed_after_step_ms) ||
        typeof step.producer_done_before_step !== 'boolean' ||
        typeof step.producer_done_after_step !== 'boolean' ||
        !(Number.isInteger(step.delta_count) && step.delta_count >= 0),
    )) {
      errors.push(
        `${result.transcription_delay_ms}:${result.repetition}: ` +
        'invalid executor step trace',
      );
    }
    if (!(result.timing?.first_append_ms > 0) ||
        result.timing?.first_stable_ms !==
          result.timing?.first_append_ms ||
        !(result.timing?.final_ms >= result.timing?.audio_close_ms) ||
        !(result.timing?.maximum_step_ms > 0) ||
        !(result.timing?.p95_feed_schedule_lateness_ms < 15)) {
      errors.push(
        `${result.transcription_delay_ms}:${result.repetition}: ` +
        'invalid streaming timing',
      );
    }
  }
  if (new Set((run.runs ?? []).map((result) => result.text)).size !== 1) {
    errors.push('streaming transcripts must be deterministic across lifecycle');
  }
  if ((run.batch_controls ?? []).length !== 2 ||
      run.batch_controls.some(
        (control) =>
          control.quality?.wer !== 0 ||
          control.quality?.cer !== 0 ||
          control.matches_stream_runs !== false,
      )) {
    errors.push(
      'batch controls must retain lexical parity and surface-form differences',
    );
  }
  return errors;
}

function validateQwen3TtsRun(run, selection, modelManifest) {
  const errors = [];
  if (run.schema_version !== schemaVersion ||
      !/^\d{4}-\d{2}-\d{2}T/.test(run.captured_at ?? '') ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '') ||
      run.selection_revision !== selection.revision ||
      run.purpose !== 'development-diagnostic') {
    errors.push('run identity must pin the source and diagnostic selection');
  }
  if (run.candidate?.id !== 'qwen3-tts-0.6b-customvoice-mlx-audio' ||
      run.candidate?.task !== 'tts' ||
      run.candidate?.model?.revision !== modelManifest.conversion.revision ||
      run.candidate?.model?.license !== modelManifest.conversion.license ||
      run.candidate?.model?.snapshot_bytes !==
        modelManifest.conversion.snapshot_bytes ||
      run.candidate?.runtime?.revision !==
        modelManifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !==
        modelManifest.reference_runtime.version ||
      run.candidate?.runtime?.license !== modelManifest.reference_runtime.license) {
    errors.push('candidate must match the pinned model and reference runtime');
  }
  if (run.candidate?.voice?.name !==
        modelManifest.generation_contract.speaker ||
      run.candidate?.voice?.language !==
        modelManifest.generation_contract.language ||
      run.candidate?.voice?.native_language !==
        modelManifest.generation_contract.speaker_native_language ||
      run.candidate?.voice?.cross_lingual !== true) {
    errors.push('voice must preserve the fixed cross-lingual contract');
  }
  if (run.host?.architecture !== 'arm64' ||
      !(run.host?.memory_bytes > 0) ||
      !(run.host?.chip?.length > 0) ||
      !(run.host?.os?.length > 0) ||
      !(run.host?.execution_context?.length > 0)) {
    errors.push('host must record the Apple Silicon execution context');
  }

  const selected = selection.sources
    .flatMap((source) => source.passages.map((passage) => ({
      source,
      passage,
    })))
    .find(({ passage }) => passage.id === run.input?.passage_id);
  if (!selected ||
      run.input?.source_id !== selected.source.id ||
      run.input?.locale !== selected.source.locale ||
      run.input?.character_count !== selected.passage.character_count ||
      run.input?.text_sha256 !== selected.passage.spoken_sha256 ||
      run.input?.license !== selected.source.license) {
    errors.push('TTS input metadata differs from the passage selection');
  }
  if (run.procedure?.repetitions !== 1 ||
      run.procedure?.stream !== false ||
      JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(modelManifest.generation_contract) ||
      !(run.procedure?.command?.includes(
        'scripts/run-qwen3-tts-mlx-reference.sh',
      )) ||
      !(run.procedure?.raw_audio?.includes('local-required')) ||
      !arrayEquals(
        run.procedure?.raw_artifacts ?? [],
        ['audio.f32le', 'audio.pcm16.wav', 'result.json'],
      )) {
    errors.push('procedure must preserve the pinned local reference run');
  }

  if (run.result?.status !== 'measured') {
    errors.push('result.status must be measured');
  }
  const audio = run.result?.audio;
  if (audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !==
        modelManifest.generation_contract.sample_rate_hz ||
      audio?.channel_count !== 1 ||
      !(Number.isInteger(audio?.sample_count) && audio.sample_count > 0) ||
      audio?.byte_count !== audio?.sample_count * 4 ||
      !/^[0-9a-f]{64}$/.test(audio?.sha256 ?? '') ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.minimum_sample >= -1 && audio.minimum_sample <= 1) ||
      !(audio?.maximum_sample >= -1 && audio.maximum_sample <= 1) ||
      !(audio?.rms > 0 && audio.rms <= 1)) {
    errors.push('audio must be finite digest-pinned mono f32 PCM');
  }
  const expectedDuration = audio?.sample_count * 1_000 / audio?.sample_rate_hz;
  if (!Number.isFinite(audio?.duration_ms) ||
      Math.abs(audio.duration_ms - expectedDuration) > 1e-9) {
    errors.push('audio duration diverges from PCM metadata');
  }

  const timing = run.result?.timing;
  const expectedRtf = timing?.complete_synthesis_ms / audio?.duration_ms;
  if (!(timing?.model_load_ms > 0) ||
      !(timing?.first_audio_ms > 0) ||
      !(timing?.complete_synthesis_ms >= timing?.first_audio_ms) ||
      !(Number.isInteger(timing?.output_count) && timing.output_count > 0) ||
      !(Number.isInteger(timing?.token_count) && timing.token_count > 0) ||
      !Number.isFinite(timing?.real_time_factor) ||
      Math.abs(timing.real_time_factor - expectedRtf) > 1e-12) {
    errors.push('timing metrics or real-time factor are inconsistent');
  }
  const runtimeOutput = run.result?.runtime_output ?? [];
  if (runtimeOutput.length !== timing?.output_count ||
      runtimeOutput.reduce((sum, output) => sum + output.sample_count, 0) !==
        audio?.sample_count ||
      runtimeOutput.reduce((sum, output) => sum + output.token_count, 0) !==
        timing?.token_count ||
      runtimeOutput.some(
        (output) =>
          output.sample_rate_hz !== audio?.sample_rate_hz ||
          output.is_streaming_chunk !== false,
      )) {
    errors.push('runtime output does not reconcile with audio and timing');
  }
  if (run.result?.termination?.reached_max_tokens !== false ||
      run.result?.termination?.configured_max_tokens !==
        modelManifest.generation_contract.max_tokens ||
      timing?.token_count >= run.result?.termination?.configured_max_tokens) {
    errors.push('generation must stop before the configured token limit');
  }
  if (!(run.result?.resources?.maximum_resident_set_size_bytes > 0) ||
      !(run.result?.resources?.mlx_peak_memory_bytes > 0) ||
      run.result?.resources?.model_snapshot_bytes !==
        modelManifest.conversion.snapshot_bytes) {
    errors.push('resource metrics must include process, MLX, and model size');
  }

  const contentChecks = run.result?.asr_content_checks;
  const normalizedAudio = contentChecks?.normalized_audio;
  if (contentChecks?.status !== 'complete' ||
      normalizedAudio?.sample_format !== 'f32le' ||
      normalizedAudio?.sample_rate_hz !== 16000 ||
      normalizedAudio?.channel_count !== 1 ||
      normalizedAudio?.byte_count !== normalizedAudio?.sample_count * 4 ||
      normalizedAudio?.duration_ms !== audio?.duration_ms ||
      !/^[0-9a-f]{64}$/.test(normalizedAudio?.sha256 ?? '')) {
    errors.push('content checks must pin the shared 16 kHz ASR input');
  }
  const expectedContentChecks = new Map([
    ['apple-speechtranscriber', [5, 104, 4, 694]],
    ['whisper-large-v3-turbo-coreml-whispercpp', [2, 104, 0, 692]],
    ['parakeet-tdt-0.6b-v3-coreml', [9, 100, 21, 692]],
    ['qwen3-asr-0.6b-mlx-direct', [2, 103, 4, 696]],
  ]);
  const measuredChecks = contentChecks?.backends ?? [];
  if (measuredChecks.length !== expectedContentChecks.size ||
      new Set(measuredChecks.map((check) => check.backend?.id)).size !==
        measuredChecks.length) {
    errors.push('content checks must contain four unique measured backends');
  }
  for (const check of measuredChecks) {
    const transcript = check.transcript;
    const transcriptDigest = createHash('sha256')
      .update(transcript?.text ?? '')
      .digest('hex');
    if (!(transcript?.text?.length > 0) ||
        transcript?.sha256 !== transcriptDigest ||
        !(transcript?.segment_count > 0) ||
        transcript?.update_count !==
          transcript?.final_update_count + transcript?.volatile_update_count ||
        transcript?.revoke_count !== 0) {
      errors.push(
        `${check.backend?.id ?? 'unknown'} transcript metadata is inconsistent`,
      );
    }
    const expected = expectedContentChecks.get(check.backend?.id);
    const quality = check.quality;
    if (!expected ||
        quality?.word_edits !== expected[0] ||
        quality?.reference_word_count !== 103 ||
        quality?.hypothesis_word_count !== expected[1] ||
        Math.abs(
          quality?.wer - quality?.word_edits / quality?.reference_word_count,
        ) > 1e-15 ||
        quality?.character_edits !== expected[2] ||
        quality?.reference_character_count !== 692 ||
        quality?.hypothesis_character_count !== expected[3] ||
        Math.abs(
          quality?.cer -
            quality?.character_edits / quality?.reference_character_count,
        ) > 1e-15) {
      errors.push(
        `${check.backend?.id ?? 'unknown'} WER or CER is not derivable`,
      );
    }
    const asrTiming = check.timing;
    if (!(asrTiming?.complete_inference_ms > 0) ||
        Math.abs(
          asrTiming?.real_time_factor -
            asrTiming?.complete_inference_ms / audio?.duration_ms,
        ) > 1e-15) {
      errors.push(`${check.backend?.id ?? 'unknown'} timing is inconsistent`);
    }
  }
  if (contentChecks?.comparison?.completed_backend_count !== 4 ||
      contentChecks?.comparison?.expected_backend_count !== 4 ||
      !arrayEquals(
        contentChecks?.comparison?.remaining_backends ?? [],
        [],
      )) {
    errors.push('content-check matrix must complete all four ASR backends');
  }
  if (run.conclusion?.reference_path_proven !== true ||
      run.conclusion?.public_contract_accepted !== false ||
      run.conclusion?.quality_decision_ready !== false ||
      !(run.conclusion?.next?.length > 0)) {
    errors.push('conclusion must keep the quality and public API provisional');
  }
  return errors;
}

function validateVoxtralTtsRun(run, selection, modelManifest) {
  const errors = [];
  if (run.schema_version !== schemaVersion ||
      !/^\d{4}-\d{2}-\d{2}T/.test(run.captured_at ?? '') ||
      !/^[0-9a-f]{40}$/.test(run.source_revision ?? '') ||
      run.selection_revision !== selection.revision ||
      run.purpose !== 'development-diagnostic') {
    errors.push('run identity must pin the source and diagnostic selection');
  }
  if (run.candidate?.id !== 'voxtral-tts-4b-mlx-audio' ||
      run.candidate?.task !== 'tts' ||
      run.candidate?.model?.revision !== modelManifest.conversion.revision ||
      run.candidate?.model?.license !== modelManifest.conversion.license ||
      run.candidate?.model?.snapshot_bytes !==
        modelManifest.conversion.snapshot_bytes ||
      run.candidate?.runtime?.revision !==
        modelManifest.reference_runtime.revision ||
      run.candidate?.runtime?.version !==
        modelManifest.reference_runtime.version ||
      run.candidate?.runtime?.license !== modelManifest.reference_runtime.license) {
    errors.push('candidate must match the pinned Voxtral model and runtime');
  }
  if (run.candidate?.voice?.name !==
        modelManifest.generation_contract.speaker ||
      run.candidate?.voice?.native_language !==
        modelManifest.generation_contract.speaker_native_language ||
      run.candidate?.voice?.cross_lingual !== false) {
    errors.push('voice must preserve the fixed native-German contract');
  }
  if (run.host?.architecture !== 'arm64' ||
      !(run.host?.memory_bytes > 0) ||
      !(run.host?.chip?.length > 0) ||
      !(run.host?.os?.length > 0) ||
      !(run.host?.execution_context?.length > 0)) {
    errors.push('host must record the Apple Silicon execution context');
  }

  const selected = selection.sources
    .flatMap((source) => source.passages.map((passage) => ({
      source,
      passage,
    })))
    .find(({ passage }) => passage.id === run.input?.passage_id);
  if (!selected ||
      run.input?.source_id !== selected.source.id ||
      run.input?.locale !== selected.source.locale ||
      run.input?.character_count !== selected.passage.character_count ||
      run.input?.text_sha256 !== selected.passage.spoken_sha256 ||
      run.input?.license !== selected.source.license) {
    errors.push('TTS input metadata differs from the passage selection');
  }
  if (run.procedure?.repetitions !== 1 ||
      run.procedure?.stream !== false ||
      JSON.stringify(run.procedure?.generation) !==
        JSON.stringify(modelManifest.generation_contract) ||
      !(run.procedure?.command?.includes(
        'scripts/run-voxtral-tts-mlx-reference.sh',
      )) ||
      !(run.procedure?.raw_audio?.includes('local-required')) ||
      !arrayEquals(
        run.procedure?.raw_artifacts ?? [],
        ['audio.f32le', 'audio.pcm16.wav', 'result.json'],
      )) {
    errors.push('procedure must preserve the pinned local Voxtral run');
  }

  const audio = run.result?.audio;
  if (run.result?.status !== 'measured' ||
      audio?.sample_format !== 'f32le' ||
      audio?.sample_rate_hz !==
        modelManifest.generation_contract.sample_rate_hz ||
      audio?.channel_count !== 1 ||
      !(Number.isInteger(audio?.sample_count) && audio.sample_count > 0) ||
      audio?.byte_count !== audio?.sample_count * 4 ||
      !/^[0-9a-f]{64}$/.test(audio?.sha256 ?? '') ||
      audio?.non_finite_sample_count !== 0 ||
      !(audio?.minimum_sample >= -1 && audio.minimum_sample <= 1) ||
      !(audio?.maximum_sample >= -1 && audio.maximum_sample <= 1) ||
      !(audio?.rms > 0 && audio.rms <= 1)) {
    errors.push('audio must be measured finite digest-pinned mono f32 PCM');
  }
  const expectedDuration = audio?.sample_count * 1_000 / audio?.sample_rate_hz;
  if (!Number.isFinite(audio?.duration_ms) ||
      Math.abs(audio.duration_ms - expectedDuration) > 1e-9) {
    errors.push('audio duration diverges from PCM metadata');
  }

  const timing = run.result?.timing;
  const expectedRtf = timing?.complete_synthesis_ms / audio?.duration_ms;
  if (!(timing?.model_load_ms > 0) ||
      !(timing?.first_audio_ms > 0) ||
      !(timing?.complete_synthesis_ms >= timing?.first_audio_ms) ||
      !(Number.isInteger(timing?.output_count) && timing.output_count > 0) ||
      !(Number.isInteger(timing?.token_count) && timing.token_count > 0) ||
      !Number.isFinite(timing?.real_time_factor) ||
      Math.abs(timing.real_time_factor - expectedRtf) > 1e-12) {
    errors.push('timing metrics or real-time factor are inconsistent');
  }
  const runtimeOutput = run.result?.runtime_output ?? [];
  if (runtimeOutput.length !== timing?.output_count ||
      runtimeOutput.reduce((sum, output) => sum + output.sample_count, 0) !==
        audio?.sample_count ||
      runtimeOutput.reduce((sum, output) => sum + output.token_count, 0) !==
        timing?.token_count ||
      runtimeOutput.some(
        (output) =>
          output.sample_rate_hz !== audio?.sample_rate_hz ||
          output.is_streaming_chunk !== false,
      )) {
    errors.push('runtime output does not reconcile with audio and timing');
  }
  if (run.result?.termination?.reached_max_tokens !== false ||
      run.result?.termination?.configured_max_tokens !==
        modelManifest.generation_contract.max_tokens ||
      timing?.token_count >= run.result?.termination?.configured_max_tokens) {
    errors.push('generation must stop before the configured token limit');
  }
  if (!(run.result?.resources?.maximum_resident_set_size_bytes > 0) ||
      !(run.result?.resources?.mlx_peak_memory_bytes > 0) ||
      run.result?.resources?.model_snapshot_bytes !==
        modelManifest.conversion.snapshot_bytes) {
    errors.push('resource metrics must include process, MLX, and model size');
  }

  const contentChecks = run.result?.asr_content_checks;
  const normalizedAudio = contentChecks?.normalized_audio;
  if (contentChecks?.status !== 'complete-with-level-control' ||
      normalizedAudio?.sample_format !== 'f32le' ||
      normalizedAudio?.sample_rate_hz !== 16000 ||
      normalizedAudio?.channel_count !== 1 ||
      normalizedAudio?.byte_count !== normalizedAudio?.sample_count * 4 ||
      normalizedAudio?.duration_ms !== audio?.duration_ms ||
      !/^[0-9a-f]{64}$/.test(normalizedAudio?.sha256 ?? '')) {
    errors.push('content checks must pin the shared 16 kHz ASR input');
  }
  const expectedContentChecks = new Map([
    ['apple-speechtranscriber', [18, 106, 45, 698]],
    ['whisper-large-v3-turbo-coreml-whispercpp', [9, 103, 32, 676]],
    ['parakeet-tdt-0.6b-v3-coreml', [64, 76, 306, 502]],
    ['qwen3-asr-0.6b-mlx-direct', [12, 104, 34, 703]],
  ]);
  const measuredChecks = contentChecks?.backends ?? [];
  if (measuredChecks.length !== expectedContentChecks.size ||
      new Set(measuredChecks.map((check) => check.backend?.id)).size !==
        measuredChecks.length) {
    errors.push('content checks must contain four unique measured backends');
  }
  for (const check of measuredChecks) {
    const transcript = check.transcript;
    const transcriptDigest = createHash('sha256')
      .update(transcript?.text ?? '')
      .digest('hex');
    if (!(transcript?.text?.length > 0) ||
        transcript?.sha256 !== transcriptDigest ||
        !(transcript?.segment_count > 0) ||
        transcript?.update_count !==
          transcript?.final_update_count + transcript?.volatile_update_count ||
        transcript?.revoke_count !== 0) {
      errors.push(
        `${check.backend?.id ?? 'unknown'} transcript metadata is inconsistent`,
      );
    }
    const expected = expectedContentChecks.get(check.backend?.id);
    const quality = check.quality;
    if (!expected ||
        quality?.word_edits !== expected[0] ||
        quality?.reference_word_count !== 103 ||
        quality?.hypothesis_word_count !== expected[1] ||
        Math.abs(
          quality?.wer - quality?.word_edits / quality?.reference_word_count,
        ) > 1e-15 ||
        quality?.character_edits !== expected[2] ||
        quality?.reference_character_count !== 692 ||
        quality?.hypothesis_character_count !== expected[3] ||
        Math.abs(
          quality?.cer -
            quality?.character_edits / quality?.reference_character_count,
        ) > 1e-15) {
      errors.push(
        `${check.backend?.id ?? 'unknown'} WER or CER is not derivable`,
      );
    }
    const asrTiming = check.timing;
    if (!(asrTiming?.complete_inference_ms > 0) ||
        Math.abs(
          asrTiming?.real_time_factor -
            asrTiming?.complete_inference_ms / audio?.duration_ms,
        ) > 1e-15) {
      errors.push(`${check.backend?.id ?? 'unknown'} timing is inconsistent`);
    }
  }

  const levelControl = contentChecks?.level_control;
  const controlAudio = levelControl?.audio;
  const expectedLevelControl = new Map([
    ['apple-speechtranscriber', [15, 47]],
    ['whisper-large-v3-turbo-coreml-whispercpp', [9, 32]],
    ['parakeet-tdt-0.6b-v3-coreml', [48, 179]],
    ['qwen3-asr-0.6b-mlx-direct', [13, 37]],
  ]);
  if (levelControl?.gain_db !== 12 ||
      controlAudio?.sample_format !== 'f32le' ||
      controlAudio?.sample_rate_hz !== 16000 ||
      controlAudio?.sample_count !== normalizedAudio?.sample_count ||
      controlAudio?.byte_count !== normalizedAudio?.byte_count ||
      controlAudio?.duration_ms !== normalizedAudio?.duration_ms ||
      controlAudio?.clipped_sample_count !== 0 ||
      !(controlAudio?.rms > 0) ||
      !/^[0-9a-f]{64}$/.test(controlAudio?.sha256 ?? '') ||
      levelControl?.backends?.length !== expectedLevelControl.size) {
    errors.push('level control must pin clipping-free +12 dB audio');
  }
  for (const check of levelControl?.backends ?? []) {
    const expected = expectedLevelControl.get(check.id);
    if (!expected ||
        check.word_edits !== expected[0] ||
        Math.abs(check.wer - check.word_edits / 103) > 1e-15 ||
        check.character_edits !== expected[1] ||
        Math.abs(check.cer - check.character_edits / 692) > 1e-15 ||
        !/^[0-9a-f]{64}$/.test(check.transcript_sha256 ?? '')) {
      errors.push(`${check.id ?? 'unknown'} level control is inconsistent`);
    }
  }
  if (contentChecks?.comparison?.completed_backend_count !== 4 ||
      contentChecks?.comparison?.expected_backend_count !== 4 ||
      !arrayEquals(
        contentChecks?.comparison?.remaining_backends ?? [],
        [],
      )) {
    errors.push('content-check matrix must complete all four ASR backends');
  }
  if (run.environment?.packages?.['mistral-common'] !== '1.10.0' ||
      run.conclusion?.reference_path_proven !== true ||
      run.conclusion?.public_contract_accepted !== false ||
      run.conclusion?.production_dependency_eligible !== false ||
      run.conclusion?.quality_decision_ready !== false ||
      !(run.conclusion?.next?.length > 0)) {
    errors.push('conclusion must keep licensing, quality, and API provisional');
  }
  return errors;
}

function validateSourceRightsReview(
  review,
  sourceCandidates,
  targetDomainPlan,
) {
  const errors = [];
  if (review.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  for (const field of [
    'review_id',
    'source_candidate_id',
    'review_authority',
    'benchmark_role',
  ]) {
    if (!(review[field]?.length > 0)) {
      errors.push(`${field} must be a non-empty string`);
    }
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(review.evidence_date ?? '')) {
    errors.push('evidence_date must be an ISO date');
  }
  if (!['candidate', 'source-group'].includes(review.scope)) {
    errors.push('scope is invalid');
  }
  if (review.scope === 'candidate' && review.source_group_id !== null) {
    errors.push('candidate review source_group_id must be null');
  }
  if (review.scope === 'source-group' &&
      !(review.source_group_id?.length > 0)) {
    errors.push('source-group review requires source_group_id');
  }
  if (!['draft', 'blocked', 'accepted'].includes(review.disposition)) {
    errors.push('disposition is invalid');
  }
  if (!['target-domain', 'professional-speech-bridge'].includes(
    review.benchmark_role,
  )) {
    errors.push('benchmark_role is invalid');
  }

  const source = sourceCandidates.sources.find(
    (candidate) => candidate.id === review.source_candidate_id,
  );
  if (!source) {
    errors.push(`unknown source candidate: ${review.source_candidate_id}`);
  }
  for (const [field, allowedValues] of [
    ['allowed_domains', source?.domains ?? []],
    ['allowed_locales', source?.languages ?? []],
  ]) {
    if (!Array.isArray(review[field]) ||
        review[field].length === 0 ||
        new Set(review[field]).size !== review[field].length) {
      errors.push(`${field} must be a non-empty unique array`);
      continue;
    }
    for (const value of review[field]) {
      if (!allowedValues.includes(value)) {
        errors.push(`${field} contains unsupported value: ${value}`);
      }
    }
  }
  if (!Array.isArray(review.evidence) || review.evidence.length === 0) {
    errors.push('evidence must be a non-empty array');
  }
  for (const [index, evidence] of (review.evidence ?? []).entries()) {
    for (const field of ['kind', 'url', 'finding']) {
      if (!(evidence[field]?.length > 0)) {
        errors.push(`evidence[${index}].${field} must be a non-empty string`);
      }
    }
    try {
      const url = new URL(evidence.url);
      if (url.protocol !== 'https:') {
        errors.push(`evidence[${index}].url must use HTTPS`);
      }
    } catch {
      errors.push(`evidence[${index}].url must be an absolute URL`);
    }
  }

  const rightFields = [
    'audio',
    'transcript',
    'derived_clips',
    'commercial_benchmark_use',
  ];
  const allowedBases = new Set(targetDomainPlan.rights_policy.accepted_bases);
  for (const field of rightFields) {
    const decision = review.rights?.[field];
    if (!['unverified', 'rejected', 'accepted'].includes(decision?.status)) {
      errors.push(`rights.${field}.status is invalid`);
    }
    if (decision?.basis !== null &&
        !allowedBases.has(decision?.basis)) {
      errors.push(`rights.${field}.basis is invalid`);
    }
    for (const listField of ['evidence_urls', 'conditions']) {
      if (!Array.isArray(decision?.[listField]) ||
          new Set(decision[listField]).size !== decision[listField].length) {
        errors.push(`rights.${field}.${listField} must be a unique array`);
      }
    }
    if (decision?.status === 'accepted' &&
        (!(decision.basis?.length > 0) ||
         decision.evidence_urls.length === 0)) {
      errors.push(`rights.${field} accepted without basis and evidence`);
    }
  }
  if (!['unverified', 'denied', 'local-only', 'allowed'].includes(
    review.rights?.redistribution,
  )) {
    errors.push('rights.redistribution is invalid');
  }

  if (!['unverified', 'rejected', 'accepted'].includes(
    review.development_isolation?.status,
  )) {
    errors.push('development_isolation.status is invalid');
  }
  if (!arrayEquals(
    review.development_isolation?.compared_source_ids ?? [],
    targetDomainPlan.excluded_development_sources,
  )) {
    errors.push('development_isolation must compare every excluded development source');
  }
  if (!Array.isArray(review.development_isolation?.evidence)) {
    errors.push('development_isolation.evidence must be an array');
  }
  if (!['blocked', 'allowed'].includes(review.acquisition?.status)) {
    errors.push('acquisition.status is invalid');
  }
  if (![null, 'manual-local-file'].includes(review.acquisition?.method)) {
    errors.push('acquisition.method is invalid');
  }
  if (!Array.isArray(review.blockers) || !Array.isArray(review.notes)) {
    errors.push('blockers and notes must be arrays');
  }

  if (review.disposition === 'accepted') {
    if (review.scope !== 'source-group') {
      errors.push('accepted review must be source-group scoped');
    }
    if (review.benchmark_role !== 'target-domain') {
      errors.push('accepted importer review must have target-domain role');
    }
    for (const field of rightFields) {
      if (review.rights?.[field]?.status !== 'accepted') {
        errors.push(`accepted review requires rights.${field}`);
      }
    }
    if (!['local-only', 'allowed'].includes(review.rights?.redistribution)) {
      errors.push('accepted review requires local-only or allowed redistribution');
    }
    if (review.development_isolation?.status !== 'accepted' ||
        review.development_isolation.evidence.length === 0) {
      errors.push('accepted review requires development isolation evidence');
    }
    if (review.acquisition?.status !== 'allowed' ||
        review.acquisition?.method !== 'manual-local-file' ||
        !(review.acquisition?.source_url?.length > 0) ||
        !/^[0-9a-f]{64}$/.test(
          review.acquisition?.expected_source_sha256 ?? '',
        ) ||
        !(review.acquisition?.artifact_name?.length > 0)) {
      errors.push('accepted review requires a complete manual-local-file acquisition');
    }
    if (review.blockers.length !== 0) {
      errors.push('accepted review must have no blockers');
    }
  } else if (review.disposition === 'blocked') {
    if (review.acquisition?.status !== 'blocked' ||
        review.acquisition?.method !== null ||
        review.acquisition?.expected_source_sha256 !== null ||
        review.acquisition?.artifact_name !== null) {
      errors.push('blocked review must not expose an acquisition path');
    }
    if (review.blockers.length === 0) {
      errors.push('blocked review requires precise blockers');
    }
  }
  return errors;
}

function validateAudiobookPilot(manifest) {
  const errors = [];
  if (manifest.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!(manifest.revision?.length > 0)) {
    errors.push('revision must be a non-empty string');
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(manifest.evidence_date ?? '')) {
    errors.push('evidence_date must be an ISO date');
  }
  if (manifest.purpose !== 'development') {
    errors.push('purpose must remain development until references are verified');
  }
  if (manifest.domain !== 'audiobook') {
    errors.push('domain must be audiobook');
  }
  if (manifest.normalization?.sample_rate_hz !== 16000 ||
      manifest.normalization?.channels !== 1 ||
      manifest.normalization?.sample_format !== 'float32le') {
    errors.push('normalization must be mono 16 kHz float32le');
  }

  const fixtures = manifest.fixtures ?? [];
  const ids = new Set();
  const rowKeys = new Set();
  const datasetRevisions = new Map();
  for (const fixture of fixtures) {
    if (ids.has(fixture.id)) errors.push(`duplicate fixture id: ${fixture.id}`);
    ids.add(fixture.id);
    const rowKey = [
      fixture.dataset,
      fixture.config,
      fixture.split,
      fixture.row_index,
    ].join(':');
    if (rowKeys.has(rowKey)) errors.push(`${fixture.id}: duplicate dataset row`);
    rowKeys.add(rowKey);
    if (!['de', 'en', 'es', 'fr', 'pt'].includes(fixture.language)) {
      errors.push(`${fixture.id}: unsupported language`);
    }
    if (fixture.locale !== null) {
      errors.push(`${fixture.id}: locale must stay unverified`);
    }
    if (fixture.gold_status !== 'dataset-transcript-unverified') {
      errors.push(`${fixture.id}: gold_status overstates the current review`);
    }
    if (!/^[0-9a-f]{40}$/.test(fixture.dataset_revision ?? '')) {
      errors.push(`${fixture.id}: dataset_revision must be a Git SHA`);
    }
    const knownRevision = datasetRevisions.get(fixture.dataset);
    if (knownRevision && knownRevision !== fixture.dataset_revision) {
      errors.push(`${fixture.dataset}: multiple revisions in one pilot`);
    }
    datasetRevisions.set(fixture.dataset, fixture.dataset_revision);
    if (!(fixture.reference_text?.length > 0)) {
      errors.push(`${fixture.id}: reference_text is required`);
    }
    if (fixture.license !== 'CC-BY-4.0' || !fixture.redistributable) {
      errors.push(`${fixture.id}: license and redistribution evidence changed`);
    }
    for (const [name, artifact] of [
      ['source', fixture.source],
      ['normalized', fixture.normalized],
    ]) {
      if (!/^[0-9a-f]{64}$/.test(artifact?.sha256 ?? '')) {
        errors.push(`${fixture.id}: ${name}.sha256 must be SHA-256`);
      }
      if (!Number.isInteger(artifact?.bytes) || artifact.bytes <= 0) {
        errors.push(`${fixture.id}: ${name}.bytes must be positive`);
      }
    }
    if (fixture.normalized?.bytes !== fixture.normalized?.sample_count * 4) {
      errors.push(`${fixture.id}: normalized byte and sample counts differ`);
    }
    const derivedDuration = fixture.normalized?.sample_count / 16;
    if (fixture.normalized?.duration_ms !== derivedDuration) {
      errors.push(`${fixture.id}: normalized duration differs from sample count`);
    }
  }
  if (fixtures.length === 0) errors.push('fixtures must be non-empty');

  const byLanguage = {};
  for (const fixture of fixtures) {
    const summary = byLanguage[fixture.language] ?? {
      fixture_count: 0,
      duration_ms: 0,
      speakers: new Set(),
      chapters: new Set(),
    };
    summary.fixture_count += 1;
    summary.duration_ms += fixture.normalized.duration_ms;
    summary.speakers.add(fixture.speaker_id);
    summary.chapters.add(fixture.chapter_id);
    byLanguage[fixture.language] = summary;
  }
  const derivedByLanguage = Object.fromEntries(
    Object.entries(byLanguage).map(([language, value]) => [
      language,
      {
        fixture_count: value.fixture_count,
        duration_ms: value.duration_ms,
        speaker_count: value.speakers.size,
        chapter_count: value.chapters.size,
      },
    ]),
  );
  const derivedSummary = {
    fixture_count: fixtures.length,
    duration_ms: fixtures.reduce(
      (sum, fixture) => sum + fixture.normalized.duration_ms,
      0,
    ),
    by_language: derivedByLanguage,
  };
  if (JSON.stringify(manifest.summary) !== JSON.stringify(derivedSummary)) {
    errors.push('summary does not match fixture metadata');
  }
  return errors;
}

function validatePostprocessingSnapshot(snapshot) {
  const errors = [];
  if (snapshot.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!(snapshot.snapshot_id?.length > 0)) {
    errors.push('snapshot_id must be a non-empty string');
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(snapshot.captured_at ?? '')) {
    errors.push('captured_at must be an ISO date');
  }
  if (!/^[0-9a-f]{40}$/.test(
    snapshot.provenance?.captured_from_source_head ?? '',
  )) {
    errors.push('provenance.captured_from_source_head must be a Git SHA');
  }

  const artifactIds = new Set();
  for (const artifact of snapshot.provenance?.artifacts ?? []) {
    if (artifactIds.has(artifact.experiment_id)) {
      errors.push(`duplicate provenance artifact: ${artifact.experiment_id}`);
    }
    artifactIds.add(artifact.experiment_id);
    for (const field of ['result_sha256', 'script_sha256']) {
      if (!/^[0-9a-f]{64}$/.test(artifact[field] ?? '')) {
        errors.push(`${artifact.experiment_id}: ${field} must be SHA-256`);
      }
    }
    if (!/^[0-9a-f]{40}$/.test(artifact.script_revision ?? '')) {
      errors.push(`${artifact.experiment_id}: script_revision must be a Git SHA`);
    }
    if (!Number.isInteger(artifact.result_record_count) ||
        artifact.result_record_count <= 0) {
      errors.push(`${artifact.experiment_id}: result_record_count must be positive`);
    }
  }

  const experimentIds = new Set();
  for (const experiment of snapshot.experiments ?? []) {
    if (experimentIds.has(experiment.id)) {
      errors.push(`duplicate postprocessing experiment: ${experiment.id}`);
    }
    experimentIds.add(experiment.id);
    if (!artifactIds.has(experiment.id)) {
      errors.push(`${experiment.id}: missing provenance artifact`);
    }
    if (!Array.isArray(experiment.languages) ||
        experiment.languages.length === 0 ||
        new Set(experiment.languages).size !== experiment.languages.length) {
      errors.push(`${experiment.id}: languages must be a non-empty unique array`);
    }
    if (!(experiment.normalization?.source_expression?.length > 0) ||
        !(experiment.normalization?.limitation?.length > 0)) {
      errors.push(`${experiment.id}: normalization evidence is required`);
    }

    const modelIds = new Set();
    for (const model of experiment.models ?? []) {
      if (modelIds.has(model.id)) {
        errors.push(`${experiment.id}: duplicate model ${model.id}`);
      }
      modelIds.add(model.id);
      if (!Number.isInteger(model.sample_count) || model.sample_count <= 0) {
        errors.push(`${experiment.id}/${model.id}: sample_count must be positive`);
      }
      if (!Number.isInteger(model.worsened_count) ||
          model.worsened_count < 0 ||
          model.worsened_count > model.sample_count) {
        errors.push(`${experiment.id}/${model.id}: worsened_count is invalid`);
      }
      for (const field of ['wer_before', 'wer_after']) {
        if (!isMetric(model[field]) || model[field] === null) {
          errors.push(`${experiment.id}/${model.id}: ${field} must be a metric`);
        }
      }
      const languageEntries = Object.entries(model.by_language ?? {});
      if (!arrayEquals(
        languageEntries.map(([language]) => language),
        experiment.languages,
      )) {
        errors.push(`${experiment.id}/${model.id}: language order or coverage differs`);
        continue;
      }
      const sampleCount = languageEntries.reduce(
        (sum, [, value]) => sum + value.sample_count,
        0,
      );
      const worsenedCount = languageEntries.reduce(
        (sum, [, value]) => sum + value.worsened_count,
        0,
      );
      if (sampleCount !== model.sample_count) {
        errors.push(`${experiment.id}/${model.id}: language sample counts differ`);
      }
      if (worsenedCount !== model.worsened_count) {
        errors.push(`${experiment.id}/${model.id}: language regression counts differ`);
      }
      for (const field of ['wer_before', 'wer_after']) {
        const weighted = languageEntries.reduce(
          (sum, [, value]) => sum + value[field] * value.sample_count,
          0,
        ) / sampleCount;
        if (Math.abs(weighted - model[field]) > 1e-12) {
          errors.push(`${experiment.id}/${model.id}: ${field} differs from languages`);
        }
      }
    }
    if (modelIds.size === 0) {
      errors.push(`${experiment.id}: models must be non-empty`);
    }
  }
  if (!arrayEquals([...artifactIds], [...experimentIds])) {
    errors.push('provenance artifact and experiment ids must match');
  }
  if (!Array.isArray(snapshot.conclusions) || snapshot.conclusions.length === 0) {
    errors.push('conclusions must be non-empty');
  }
  return errors;
}

async function validatePromptManifest(manifest) {
  const errors = [];
  if (manifest.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!(manifest.revision?.length > 0)) {
    errors.push('revision must be a non-empty string');
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(manifest.evidence_date ?? '')) {
    errors.push('evidence_date must be an ISO date');
  }
  for (const field of [
    'grouping_unit',
    'development',
    'validation',
    'test',
    'current_fleurs_disposition',
  ]) {
    if (!(manifest.split_policy?.[field]?.length > 0)) {
      errors.push(`split_policy.${field} must be a non-empty string`);
    }
  }
  if (!Array.isArray(manifest.prompts) || manifest.prompts.length === 0) {
    errors.push('prompts must be a non-empty array');
  }

  const ids = new Set();
  const paths = new Set();
  const modes = new Set([
    'historical-control',
    'surface-only',
    'bounded-lexical',
    'targeted-lexical',
  ]);
  for (const prompt of manifest.prompts ?? []) {
    if (ids.has(prompt.id)) errors.push(`duplicate prompt id: ${prompt.id}`);
    ids.add(prompt.id);
    if (paths.has(prompt.path)) {
      errors.push(`${prompt.id}: duplicate prompt path ${prompt.path}`);
    }
    paths.add(prompt.path);
    if (!modes.has(prompt.mode)) errors.push(`${prompt.id}: invalid mode`);
    if (!/^benchmarks\/postprocessing\/prompts\/[^/]+\.txt$/.test(
      prompt.path ?? '',
    )) {
      errors.push(`${prompt.id}: prompt path must stay in the prompt directory`);
      continue;
    }
    if (!/^[0-9a-f]{64}$/.test(prompt.sha256 ?? '')) {
      errors.push(`${prompt.id}: sha256 must be SHA-256`);
    }
    for (const field of ['source', 'output_contract']) {
      if (!(prompt[field]?.length > 0)) {
        errors.push(`${prompt.id}: ${field} must be a non-empty string`);
      }
    }
    for (const field of [
      'allowed_edit_classes',
      'context_fields',
      'mechanical_gates',
    ]) {
      if (!Array.isArray(prompt[field]) ||
          prompt[field].length === 0 ||
          new Set(prompt[field]).size !== prompt[field].length) {
        errors.push(`${prompt.id}: ${field} must be a non-empty unique array`);
      }
    }

    try {
      const promptBytes = await readFile(join(repoRoot, prompt.path));
      const digest = createHash('sha256').update(promptBytes).digest('hex');
      if (digest !== prompt.sha256) {
        errors.push(`${prompt.id}: prompt digest does not match ${prompt.path}`);
      }
      const promptText = promptBytes.toString('utf8');
      if (promptText.trim().length === 0) {
        errors.push(`${prompt.id}: prompt file must not be empty`);
      }
      if (prompt.mode !== 'historical-control') {
        for (const field of prompt.context_fields ?? []) {
          if (!promptText.includes(`{{${field}}}`)) {
            errors.push(`${prompt.id}: missing {{${field}}} placeholder`);
          }
        }
      }
    } catch (error) {
      if (error.code === 'ENOENT') {
        errors.push(`${prompt.id}: prompt file does not exist`);
      } else {
        throw error;
      }
    }
  }
  return errors;
}

function arrayEquals(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function nonNullMaximum(values) {
  const present = values.filter((value) => value !== null);
  return present.length === 0 ? null : Math.max(...present);
}

function aggregateSummary(results) {
  const languageGroups = {};
  for (const result of results) {
    const group = languageGroups[result.requested_language] ?? [];
    group.push(result);
    languageGroups[result.requested_language] = group;
  }
  return {
    macro_wer: mean(results.map((result) => result.quality.wer)),
    macro_cer: mean(results.map((result) => result.quality.cer)),
    mean_warm_inference_ms: mean(
      results.map((result) => result.timing.warm_inference_ms),
    ),
    mean_real_time_factor: mean(
      results.map((result) => result.timing.real_time_factor),
    ),
    maximum_peak_memory_bytes: Math.max(
      ...results.map((result) => result.resources.peak_memory_bytes),
    ),
    by_language: Object.fromEntries(
      Object.entries(languageGroups).map(([language, languageResults]) => [
        language,
        {
          fixture_count: languageResults.length,
          macro_wer: mean(
            languageResults.map((result) => result.quality.wer),
          ),
          macro_cer: mean(
            languageResults.map((result) => result.quality.cer),
          ),
        },
      ]),
    ),
  };
}

function validateAggregate(aggregate, manifest, path) {
  const errors = [];
  const qualityFixtures = manifest.fixtures
    .filter((fixture) => fixture.purpose === 'quality')
  const aggregateFixtures = qualityFixtures.length > 0
    ? qualityFixtures
    : manifest.fixtures;
  const qualityFixtureIds = aggregateFixtures.map((fixture) => fixture.id);
  if (aggregate.schema_version !== schemaVersion) {
    errors.push(`schema_version must be ${schemaVersion}`);
  }
  if (!(aggregate.matrix_run_id?.length > 0)) {
    errors.push('matrix_run_id must be a non-empty string');
  }
  if (!/^[0-9a-f]{40}$/.test(aggregate.source_revision ?? '')) {
    errors.push('source_revision must be a 40-character Git SHA');
  }
  if (aggregate.fixture_manifest_revision !== manifest.revision) {
    errors.push(`fixture_manifest_revision must be ${manifest.revision}`);
  }
  for (const field of [
    'candidate.id',
    'candidate.model',
    'candidate.runtime',
    'candidate.boundary',
    'candidate.license',
    'host.id',
    'host.chip',
    'host.os',
  ]) {
    if (!(at(aggregate, field)?.length > 0)) {
      errors.push(`${field} must be a non-empty string`);
    }
  }
  if (!arrayEquals(aggregate.procedure?.fixture_order ?? [], qualityFixtureIds)) {
    errors.push('procedure.fixture_order must equal the quality fixture order');
  }
  if (aggregate.procedure?.fixture_count !== qualityFixtureIds.length) {
    errors.push(`procedure.fixture_count must be ${qualityFixtureIds.length}`);
  }
  const resultIds = (aggregate.results ?? []).map((result) => result.fixture_id);
  if (!arrayEquals(resultIds, qualityFixtureIds)) {
    errors.push('result fixture order must equal the quality fixture order');
  }

  for (const result of aggregate.results ?? []) {
    if (!(result.text?.length > 0)) {
      errors.push(`${result.fixture_id}: text must be non-empty`);
    }
    for (const field of [
      'quality.wer',
      'quality.cer',
      'timing.audio_duration_ms',
      'timing.cold_load_ms',
      'timing.warm_inference_ms',
      'timing.real_time_factor',
      'resources.peak_memory_bytes',
      'resources.model_size_bytes',
      'resources.binary_size_bytes',
    ]) {
      if (!isMetric(at(result, field))) {
        errors.push(`${result.fixture_id}: ${field} must be a non-negative metric`);
      }
    }
    if (!Array.isArray(result.repetitions) ||
        result.repetitions.length !== aggregate.procedure?.repetitions) {
      errors.push(`${result.fixture_id}: repetition count does not match procedure`);
    }
    if (result.streaming?.supported &&
        result.streaming?.input_streaming_measured !== false &&
        !isMetric(result.streaming.first_result_ms)) {
      errors.push(`${result.fixture_id}: streaming requires first_result_ms`);
    }
    if (!['none', 'segment', 'word', 'unknown'].includes(result.streaming?.timestamps)) {
      errors.push(`${result.fixture_id}: invalid streaming timestamps`);
    }
  }

  if ((aggregate.results ?? []).length > 0) {
    const derived = aggregateSummary(aggregate.results);
    if (JSON.stringify(aggregate.summary) !== JSON.stringify(derived)) {
      errors.push('summary does not match the per-fixture results');
    }
  }
  return errors.map((error) => `${path}: ${error}`);
}

async function validateMatrix(matrix, manifest, path) {
  const errors = [];
  const qualityFixtures = manifest.fixtures.filter(
    (fixture) => fixture.purpose === 'quality',
  );
  const eligibleFixtures = qualityFixtures.length > 0
    ? qualityFixtures
    : manifest.fixtures;
  if (matrix.schema_version !== schemaVersion) errors.push(`schema_version must be ${schemaVersion}`);
  if (!(matrix.matrix_id?.length > 0)) errors.push('matrix_id must be a non-empty string');
  if (!/^[0-9a-f]{40}$/.test(matrix.source_revision ?? '')) errors.push('source_revision must be a 40-character Git SHA');
  if (matrix.fixture_manifest_revision !== manifest.revision) {
    errors.push(`fixture_manifest_revision must be ${manifest.revision}`);
  }
  if (!Array.isArray(matrix.fixture_ids) || matrix.fixture_ids.length === 0) {
    errors.push('fixture_ids must be a non-empty array');
  }
  if (new Set(matrix.fixture_ids ?? []).size !== (matrix.fixture_ids ?? []).length) {
    errors.push('fixture_ids must be unique');
  }
  for (const fixtureId of matrix.fixture_ids ?? []) {
    const fixture = eligibleFixtures.find((item) => item.id === fixtureId);
    if (!fixture) {
      errors.push(`${fixtureId}: matrix fixture is not eligible`);
    }
  }
  if (!arrayEquals(matrix.procedure?.fixture_order ?? [], matrix.fixture_ids ?? [])) {
    errors.push('procedure.fixture_order must equal fixture_ids');
  }

  const candidateIds = new Set();
  for (const candidate of matrix.candidates ?? []) {
    if (candidateIds.has(candidate.id)) errors.push(`duplicate candidate id: ${candidate.id}`);
    candidateIds.add(candidate.id);
    if (!(candidate.raw_artifact?.startsWith('benchmarks/raw/') && candidate.raw_artifact.endsWith('/result.json'))) {
      errors.push(`${candidate.id}: raw_artifact must name a checked-in benchmark result`);
      continue;
    }

    let raw;
    try {
      raw = await readJson(resolve(repoRoot, candidate.raw_artifact));
    } catch (error) {
      errors.push(`${candidate.id}: cannot read ${candidate.raw_artifact}: ${error.message}`);
      continue;
    }
    if (raw.schema_version !== schemaVersion) errors.push(`${candidate.id}: raw schema_version must be ${schemaVersion}`);
    if (raw.candidate?.id !== candidate.id) errors.push(`${candidate.id}: raw candidate id does not match`);
    const candidateSourceRevision =
      candidate.source_revision ?? matrix.source_revision;
    if (raw.source_revision !== candidateSourceRevision) {
      errors.push(`${candidate.id}: raw source_revision does not match`);
    }
    if (raw.fixture_manifest_revision !== matrix.fixture_manifest_revision) {
      errors.push(`${candidate.id}: raw fixture_manifest_revision does not match`);
    }
    const rawFixtureIds = (raw.results ?? []).map((result) => result.fixture_id);
    if (!arrayEquals(rawFixtureIds, matrix.fixture_ids)) errors.push(`${candidate.id}: raw fixture order does not match`);

    const rawMetrics = {
      macroWer: raw.summary?.macro_wer,
      macroCer: raw.summary?.macro_cer,
      meanCold: mean(raw.results.map((result) => result.timing.cold_load_ms)),
      minimumCold: Math.min(...raw.results.map((result) => result.timing.cold_load_ms)),
      maximumCold: Math.max(...raw.results.map((result) => result.timing.cold_load_ms)),
      meanWarm: raw.summary?.mean_warm_inference_ms,
      meanRtf: raw.summary?.mean_real_time_factor,
      maximumPeakMemory: raw.summary?.maximum_peak_memory_bytes,
      runtimePeakMemory: nonNullMaximum(raw.results.map((result) => result.resources.runtime_peak_memory_bytes)),
      modelSize: nonNullMaximum(raw.results.map((result) => result.resources.model_size_bytes)),
      binarySize: nonNullMaximum(raw.results.map((result) => result.resources.binary_size_bytes)),
      meanFirstResult: (() => {
        const values = raw.results
          .map((result) => result.streaming.first_result_ms)
          .filter((value) => value !== null);
        return values.length === 0 ? null : mean(values);
      })(),
      streamingSupported: raw.results.every((result) => result.streaming.supported),
      timestamps: [...new Set(raw.results.map((result) => result.streaming.timestamps))],
    };
    const comparisons = [
      ['quality.macro_wer', candidate.quality?.macro_wer, rawMetrics.macroWer],
      ['quality.macro_cer', candidate.quality?.macro_cer, rawMetrics.macroCer],
      ['timing.mean_cold_load_ms', candidate.timing?.mean_cold_load_ms, rawMetrics.meanCold],
      ['timing.minimum_cold_load_ms', candidate.timing?.minimum_cold_load_ms, rawMetrics.minimumCold],
      ['timing.maximum_cold_load_ms', candidate.timing?.maximum_cold_load_ms, rawMetrics.maximumCold],
      ['timing.mean_warm_inference_ms', candidate.timing?.mean_warm_inference_ms, rawMetrics.meanWarm],
      ['timing.mean_real_time_factor', candidate.timing?.mean_real_time_factor, rawMetrics.meanRtf],
      ['resources.maximum_peak_memory_bytes', candidate.resources?.maximum_peak_memory_bytes, rawMetrics.maximumPeakMemory],
      ['resources.runtime_peak_memory_bytes', candidate.resources?.runtime_peak_memory_bytes, rawMetrics.runtimePeakMemory],
      ['resources.model_size_bytes', candidate.resources?.model_size_bytes, rawMetrics.modelSize],
      ['resources.binary_size_bytes', candidate.resources?.binary_size_bytes, rawMetrics.binarySize],
      ['streaming.mean_first_result_ms', candidate.streaming?.mean_first_result_ms, rawMetrics.meanFirstResult],
      ['streaming.supported', candidate.streaming?.supported, rawMetrics.streamingSupported],
    ];
    for (const [field, actual, expected] of comparisons) {
      if (actual !== expected) errors.push(`${candidate.id}: ${field} does not match raw artifact`);
    }
    if (rawMetrics.timestamps.length !== 1 || candidate.streaming?.timestamps !== rawMetrics.timestamps[0]) {
      errors.push(`${candidate.id}: streaming.timestamps does not match raw artifact`);
    }
    if (JSON.stringify(candidate.quality?.by_language) !== JSON.stringify(raw.summary?.by_language)) {
      errors.push(`${candidate.id}: quality.by_language does not match raw artifact`);
    }
  }

  if ((matrix.candidates ?? []).length < 2) errors.push('matrix requires at least two candidates');
  if (!candidateIds.has(matrix.recommendation?.primary)) errors.push('recommendation.primary must name a candidate');
  if (!candidateIds.has(matrix.recommendation?.fallback)) errors.push('recommendation.fallback must name a candidate');
  if (matrix.recommendation?.primary === matrix.recommendation?.fallback) {
    errors.push('recommendation primary and fallback must differ');
  }
  for (const id of matrix.recommendation?.research ?? []) {
    if (!candidateIds.has(id)) errors.push(`recommendation research candidate does not exist: ${id}`);
  }

  return errors.map((error) => `${path}: ${error}`);
}

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

const manifestPath = join(repoRoot, 'benchmarks/fixtures/manifest.json');
const sourceCandidatesPath = join(
  repoRoot,
  'benchmarks/fixtures/source-candidates.json',
);
const targetDomainPlanPath = join(
  repoRoot,
  'benchmarks/fixtures/target-domain-plan.json',
);
const syntheticRoundtripPlanPath = join(
  repoRoot,
  'benchmarks/fixtures/synthetic-roundtrip-plan.json',
);
const syntheticRoundtripSelectionPath = join(
  repoRoot,
  'benchmarks/fixtures/synthetic-roundtrip-selection.json',
);
const appleTtsRunPath = join(
  repoRoot,
  'benchmarks/raw/phase5.apple-tts.synthetic-de-origin-1/result.json',
);
const qwen3TtsRunPath = join(
  repoRoot,
  'benchmarks/raw/phase5.qwen3-tts-0.6b-mlx-reference.synthetic-de-origin-1/result.json',
);
const voxtralTtsRunPath = join(
  repoRoot,
  'benchmarks/raw/phase5.voxtral-tts-4b-mlx-reference.synthetic-de-origin-1/result.json',
);
const qwen3TtsModelManifestPath = join(
  repoRoot,
  'spikes/qwen3-tts-mlx-reference/model-manifest.json',
);
const voxtralTtsModelManifestPath = join(
  repoRoot,
  'spikes/voxtral-tts-mlx-reference/model-manifest.json',
);
const voxtralRealtimeModelManifestPath = join(
  repoRoot,
  'spikes/voxtral-realtime-mlx-reference/model-manifest.json',
);
const voxtralRealtimeDirectModelManifestPath = join(
  repoRoot,
  'spikes/voxtral-realtime-mlx-direct/model-manifest.json',
);
const voxtralRealtimeDirectBoundaryPath = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-direct-boundary-1/result.json',
);
const voxtralRealtimeDirectFrontendOraclePath = join(
  repoRoot,
  'benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.frontend-480ms.json',
);
const voxtralRealtimeDirectFrontendPath = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-direct.frontend-480ms-1/result.json',
);
const voxtralRealtimeDirectEncoderOraclePath = join(
  repoRoot,
  'benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.encoder-480ms.json',
);
const voxtralRealtimeDirectEncoderPath = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-direct.encoder-480ms-1/result.json',
);
const voxtralRealtimeDirectTranscriptionOraclePath = join(
  repoRoot,
  'benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.decoder-480ms.json',
);
const voxtralRealtimeDirectTranscriptionPath = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-direct.transcription-480ms-1/result.json',
);
const voxtralRealtime480Path = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-reference-480ms.audiobook-pilot-1/result.json',
);
const voxtralRealtime2400Path = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-reference-2400ms.audiobook-pilot-1/result.json',
);
const voxtralRealtimeFleurs480Path = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-reference-480ms.multilingual-fleurs-10-1/result.json',
);
const voxtralRealtimeFleurs2400Path = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-reference-2400ms.multilingual-fleurs-10-1/result.json',
);
const voxtralRealtimeStreaming80Path = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-reference.streaming-80ms-1/result.json',
);
const voxtralRealtimeStreaming320Path = join(
  repoRoot,
  'benchmarks/raw/phase0.voxtral-realtime-mlx-reference.streaming-320ms-1/result.json',
);
const sourceRightsDirectory = join(repoRoot, 'benchmarks/rights');
const audiobookPilotPath = join(
  repoRoot,
  'benchmarks/fixtures/audiobook-pilot.json',
);
const postprocessingSnapshotPath = join(
  repoRoot,
  'benchmarks/postprocessing/cuttledoc-v2-snapshot.json',
);
const promptManifestPath = join(
  repoRoot,
  'benchmarks/postprocessing/prompts/manifest.json',
);
const schemaPaths = [
  join(repoRoot, 'benchmarks/schema/run.schema.json'),
  join(repoRoot, 'benchmarks/schema/fixture-manifest.schema.json'),
  join(repoRoot, 'benchmarks/schema/matrix.schema.json'),
  join(repoRoot, 'benchmarks/schema/source-candidates.schema.json'),
  join(repoRoot, 'benchmarks/schema/target-domain-plan.schema.json'),
  join(repoRoot, 'benchmarks/schema/synthetic-roundtrip-plan.schema.json'),
  join(repoRoot, 'benchmarks/schema/synthetic-roundtrip-selection.schema.json'),
  join(repoRoot, 'benchmarks/schema/tts-run.schema.json'),
  join(
    repoRoot,
    'spikes/qwen3-tts-mlx-reference/model-manifest.schema.json',
  ),
  join(
    repoRoot,
    'spikes/voxtral-realtime-mlx-reference/model-manifest.schema.json',
  ),
  join(
    repoRoot,
    'spikes/voxtral-realtime-mlx-direct/model-manifest.schema.json',
  ),
  join(repoRoot, 'benchmarks/schema/source-rights-review.schema.json'),
  join(repoRoot, 'benchmarks/schema/audiobook-pilot.schema.json'),
  join(repoRoot, 'benchmarks/schema/postprocessing-snapshot.schema.json'),
  join(repoRoot, 'benchmarks/schema/error-analysis.schema.json'),
  join(repoRoot, 'benchmarks/schema/postprocessing-prompts.schema.json'),
];
for (const path of schemaPaths) {
  const schema = await readJson(path);
  if (schema.$schema !== 'https://json-schema.org/draft/2020-12/schema') {
    throw new Error(`${path}: expected JSON Schema draft 2020-12`);
  }
}

const manifest = await readJson(manifestPath);
const manifestValidation = validateManifest(manifest);
const sourceCandidates = await readJson(sourceCandidatesPath);
const targetDomainPlan = await readJson(targetDomainPlanPath);
const syntheticRoundtripPlan = await readJson(syntheticRoundtripPlanPath);
const syntheticRoundtripSelection = await readJson(
  syntheticRoundtripSelectionPath,
);
const appleTtsRun = await readJson(appleTtsRunPath);
const qwen3TtsRun = await readJson(qwen3TtsRunPath);
const voxtralTtsRun = await readJson(voxtralTtsRunPath);
const qwen3TtsModelManifest = await readJson(qwen3TtsModelManifestPath);
const voxtralTtsModelManifest = await readJson(voxtralTtsModelManifestPath);
const voxtralRealtimeModelManifest = await readJson(
  voxtralRealtimeModelManifestPath,
);
const voxtralRealtimeDirectModelManifest = await readJson(
  voxtralRealtimeDirectModelManifestPath,
);
const voxtralRealtimeDirectBoundary = await readJson(
  voxtralRealtimeDirectBoundaryPath,
);
const voxtralRealtimeDirectFrontendOracle = await readJson(
  voxtralRealtimeDirectFrontendOraclePath,
);
const voxtralRealtimeDirectFrontend = await readJson(
  voxtralRealtimeDirectFrontendPath,
);
const voxtralRealtimeDirectEncoderOracle = await readJson(
  voxtralRealtimeDirectEncoderOraclePath,
);
const voxtralRealtimeDirectEncoder = await readJson(
  voxtralRealtimeDirectEncoderPath,
);
const voxtralRealtimeDirectTranscriptionOracle = await readJson(
  voxtralRealtimeDirectTranscriptionOraclePath,
);
const voxtralRealtimeDirectTranscription = await readJson(
  voxtralRealtimeDirectTranscriptionPath,
);
const sourceRightsPaths = (await readdir(sourceRightsDirectory))
  .filter((name) => name.endsWith('.json'))
  .sort()
  .map((name) => join(sourceRightsDirectory, name));
const sourceRightsReviews = await Promise.all(
  sourceRightsPaths.map((path) => readJson(path)),
);
const audiobookPilot = await readJson(audiobookPilotPath);
const voxtralRealtime480 = await readJson(voxtralRealtime480Path);
const voxtralRealtime2400 = await readJson(voxtralRealtime2400Path);
const voxtralRealtimeFleurs480 = await readJson(
  voxtralRealtimeFleurs480Path,
);
const voxtralRealtimeFleurs2400 = await readJson(
  voxtralRealtimeFleurs2400Path,
);
const voxtralRealtimeStreaming80 = await readJson(
  voxtralRealtimeStreaming80Path,
);
const voxtralRealtimeStreaming320 = await readJson(
  voxtralRealtimeStreaming320Path,
);
const aggregateManifests = new Map([
  [manifest.revision, manifest],
  [audiobookPilot.revision, audiobookPilot],
]);
const postprocessingSnapshot = await readJson(postprocessingSnapshotPath);
const promptManifest = await readJson(promptManifestPath);
const requestedRun = process.argv.indexOf('--run');
const runPaths = requestedRun >= 0
  ? [resolve(process.cwd(), process.argv[requestedRun + 1])]
  : (await readdir(join(repoRoot, 'benchmarks/runs')))
      .filter((name) => name.endsWith('.json'))
      .sort()
      .map((name) => join(repoRoot, 'benchmarks/runs', name));

const failures = manifestValidation.errors.map((error) => `${manifestPath}: ${error}`);
failures.push(
  ...validateSourceCandidates(sourceCandidates).map(
    (error) => `${sourceCandidatesPath}: ${error}`,
  ),
);
failures.push(
  ...validateTargetDomainPlan(targetDomainPlan, sourceCandidates).map(
    (error) => `${targetDomainPlanPath}: ${error}`,
  ),
);
failures.push(
  ...validateSyntheticRoundtripPlan(syntheticRoundtripPlan).map(
    (error) => `${syntheticRoundtripPlanPath}: ${error}`,
  ),
);
failures.push(
  ...validateSyntheticRoundtripSelection(
    syntheticRoundtripSelection,
    syntheticRoundtripPlan,
  ).map((error) => `${syntheticRoundtripSelectionPath}: ${error}`),
);
failures.push(
  ...validateAppleTtsRun(appleTtsRun, syntheticRoundtripSelection).map(
    (error) => `${appleTtsRunPath}: ${error}`,
  ),
);
failures.push(
  ...validateQwen3TtsModelManifest(
    qwen3TtsModelManifest,
    syntheticRoundtripPlan,
    syntheticRoundtripSelection,
  ).map((error) => `${qwen3TtsModelManifestPath}: ${error}`),
);
failures.push(
  ...validateVoxtralTtsModelManifest(
    voxtralTtsModelManifest,
    syntheticRoundtripPlan,
    syntheticRoundtripSelection,
  ).map((error) => `${voxtralTtsModelManifestPath}: ${error}`),
);
failures.push(
  ...validateVoxtralRealtimeModelManifest(
    voxtralRealtimeModelManifest,
  ).map((error) => `${voxtralRealtimeModelManifestPath}: ${error}`),
);
failures.push(
  ...validateVoxtralRealtimeDirectModelManifest(
    voxtralRealtimeDirectModelManifest,
    voxtralRealtimeModelManifest,
  ).map((error) => `${voxtralRealtimeDirectModelManifestPath}: ${error}`),
);
failures.push(
  ...validateQwen3TtsRun(
    qwen3TtsRun,
    syntheticRoundtripSelection,
    qwen3TtsModelManifest,
  ).map((error) => `${qwen3TtsRunPath}: ${error}`),
);
failures.push(
  ...validateVoxtralTtsRun(
    voxtralTtsRun,
    syntheticRoundtripSelection,
    voxtralTtsModelManifest,
  ).map((error) => `${voxtralTtsRunPath}: ${error}`),
);
failures.push(
  ...validateAggregate(
    voxtralRealtime480,
    audiobookPilot,
    voxtralRealtime480Path,
  ),
  ...validateVoxtralRealtimeAggregate(
    voxtralRealtime480,
    voxtralRealtimeModelManifest,
    480,
  ).map((error) => `${voxtralRealtime480Path}: ${error}`),
  ...validateAggregate(
    voxtralRealtime2400,
    audiobookPilot,
    voxtralRealtime2400Path,
  ),
  ...validateVoxtralRealtimeAggregate(
    voxtralRealtime2400,
    voxtralRealtimeModelManifest,
    2400,
  ).map((error) => `${voxtralRealtime2400Path}: ${error}`),
);
if (!(voxtralRealtime2400.summary?.macro_wer <
      voxtralRealtime480.summary?.macro_wer)) {
  failures.push(
    'Voxtral Realtime 2400 ms control must preserve its observed macro-WER gain',
  );
}
failures.push(
  ...validateAggregate(
    voxtralRealtimeFleurs480,
    manifest,
    voxtralRealtimeFleurs480Path,
  ),
  ...validateVoxtralRealtimeAggregate(
    voxtralRealtimeFleurs480,
    voxtralRealtimeModelManifest,
    480,
  ).map((error) => `${voxtralRealtimeFleurs480Path}: ${error}`),
  ...validateAggregate(
    voxtralRealtimeFleurs2400,
    manifest,
    voxtralRealtimeFleurs2400Path,
  ),
  ...validateVoxtralRealtimeAggregate(
    voxtralRealtimeFleurs2400,
    voxtralRealtimeModelManifest,
    2400,
  ).map((error) => `${voxtralRealtimeFleurs2400Path}: ${error}`),
);
if (!(voxtralRealtimeFleurs2400.summary?.macro_wer <
      voxtralRealtimeFleurs480.summary?.macro_wer)) {
  failures.push(
    'Voxtral Realtime 2400 ms FLEURS control must preserve its observed macro-WER gain',
  );
}
failures.push(
  ...validateVoxtralRealtimeStreamingRun(
    voxtralRealtimeStreaming80,
    voxtralRealtimeModelManifest,
    audiobookPilot,
    80,
  ).map((error) => `${voxtralRealtimeStreaming80Path}: ${error}`),
  ...validateVoxtralRealtimeStreamingRun(
    voxtralRealtimeStreaming320,
    voxtralRealtimeModelManifest,
    audiobookPilot,
    320,
  ).map((error) => `${voxtralRealtimeStreaming320Path}: ${error}`),
);
failures.push(
  ...validateVoxtralRealtimeDirectBoundary(
    voxtralRealtimeDirectBoundary,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).map((error) => `${voxtralRealtimeDirectBoundaryPath}: ${error}`),
);
failures.push(
  ...validateVoxtralRealtimeDirectFrontend(
    voxtralRealtimeDirectFrontend,
    voxtralRealtimeDirectFrontendOracle,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).map((error) => `${voxtralRealtimeDirectFrontendPath}: ${error}`),
);
failures.push(
  ...validateVoxtralRealtimeDirectEncoder(
    voxtralRealtimeDirectEncoder,
    voxtralRealtimeDirectEncoderOracle,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).map((error) => `${voxtralRealtimeDirectEncoderPath}: ${error}`),
);
failures.push(
  ...validateVoxtralRealtimeDirectTranscription(
    voxtralRealtimeDirectTranscription,
    voxtralRealtimeDirectTranscriptionOracle,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).map((error) => `${voxtralRealtimeDirectTranscriptionPath}: ${error}`),
);
const maximumRunningStep = (run) => Math.max(
  ...run.runs.map(
    (result) => result.timing.maximum_step_started_before_close_ms,
  ),
);
if (!(maximumRunningStep(voxtralRealtimeStreaming80) > 25_000) ||
    !(maximumRunningStep(voxtralRealtimeStreaming320) < 250) ||
    !voxtralRealtimeStreaming320.runs.every(
      (result) => result.timing.endpoint_finalization_ms < 1_000,
    )) {
  failures.push(
    'Voxtral streaming controls must preserve the measured 80 ms queue stall and stable 320 ms workaround',
  );
}
const rightsReviewIds = new Set();
const reviewedCandidateIds = new Set();
for (const [index, review] of sourceRightsReviews.entries()) {
  if (rightsReviewIds.has(review.review_id)) {
    failures.push(
      `${sourceRightsPaths[index]}: duplicate rights review id ${review.review_id}`,
    );
  }
  rightsReviewIds.add(review.review_id);
  reviewedCandidateIds.add(review.source_candidate_id);
  failures.push(
    ...validateSourceRightsReview(
      review,
      sourceCandidates,
      targetDomainPlan,
    ).map((error) => `${sourceRightsPaths[index]}: ${error}`),
  );
}
for (const source of sourceCandidates.sources) {
  if (source.status === 'selected-pending-rights-review' &&
      !reviewedCandidateIds.has(source.id)) {
    failures.push(
      `${sourceRightsDirectory}: missing rights review for ${source.id}`,
    );
  }
}
failures.push(
  ...validateAudiobookPilot(audiobookPilot).map(
    (error) => `${audiobookPilotPath}: ${error}`,
  ),
);
failures.push(
  ...validatePostprocessingSnapshot(postprocessingSnapshot).map(
    (error) => `${postprocessingSnapshotPath}: ${error}`,
  ),
);
failures.push(
  ...(await validatePromptManifest(promptManifest)).map(
    (error) => `${promptManifestPath}: ${error}`,
  ),
);
const runs = [];
for (const path of runPaths) {
  const run = await readJson(path);
  runs.push(run);
  failures.push(...validateRun(run, new Map(manifest.fixtures.map((fixture) => [fixture.id, fixture]))).map((error) => `${path}: ${error}`));
}

const matrixDirectory = join(repoRoot, 'benchmarks/matrices');
const matrixPaths = (await readdir(matrixDirectory))
  .filter((name) => name.endsWith('.json'))
  .sort()
  .map((name) => join(matrixDirectory, name));
const matrices = [];
for (const path of matrixPaths) {
  const matrix = await readJson(path);
  matrices.push(matrix);
  const matrixManifest = aggregateManifests.get(
    matrix.fixture_manifest_revision,
  );
  if (!matrixManifest) {
    failures.push(
      `${path}: unknown fixture manifest revision ` +
      `${matrix.fixture_manifest_revision}`,
    );
    continue;
  }
  failures.push(...await validateMatrix(matrix, matrixManifest, path));
}

const rawDirectory = join(repoRoot, 'benchmarks/raw');
const aggregatePaths = (await readdir(rawDirectory, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory())
  .map((entry) => join(rawDirectory, entry.name, 'result.json'))
  .sort();
const aggregates = [];
for (const path of aggregatePaths) {
  try {
    const raw = await readJson(path);
    if (typeof raw.matrix_run_id !== 'string') continue;
    aggregates.push(raw);
    const aggregateManifest = aggregateManifests.get(
      raw.fixture_manifest_revision,
    );
    if (!aggregateManifest) {
      failures.push(
        `${path}: unknown fixture manifest revision ` +
        `${raw.fixture_manifest_revision}`,
      );
      continue;
    }
    failures.push(...validateAggregate(raw, aggregateManifest, path));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
}

if (process.argv.includes('--self-test')) {
  const invalid = structuredClone(runs[0]);
  invalid.result.status = 'measured';
  invalid.result.blocker = null;
  invalid.fixture_id = null;
  if (validateRun(invalid, new Map(manifest.fixtures.map((fixture) => [fixture.id, fixture]))).length === 0) {
    failures.push('validator self-test failed to reject an incomplete measured run');
  }
  const invalidMatrix = structuredClone(matrices[0]);
  invalidMatrix.candidates[0].quality.macro_wer += 1;
  if ((await validateMatrix(invalidMatrix, manifest, '<matrix-self-test>')).length === 0) {
    failures.push('validator self-test failed to reject a matrix that diverges from its raw artifact');
  }
  const invalidAggregate = structuredClone(aggregates[0]);
  invalidAggregate.summary.macro_wer += 1;
  if (validateAggregate(invalidAggregate, manifest, '<aggregate-self-test>').length === 0) {
    failures.push('validator self-test failed to reject an aggregate with a divergent summary');
  }
  const invalidSourceCandidates = structuredClone(sourceCandidates);
  invalidSourceCandidates.sources.push(
    structuredClone(invalidSourceCandidates.sources[0]),
  );
  if (validateSourceCandidates(invalidSourceCandidates).length === 0) {
    failures.push('validator self-test failed to reject duplicate source ids');
  }
  const invalidTargetDomainPlan = structuredClone(targetDomainPlan);
  invalidTargetDomainPlan.cells[0].source_candidate_ids = ['fleurs'];
  if (validateTargetDomainPlan(
    invalidTargetDomainPlan,
    sourceCandidates,
  ).length === 0) {
    failures.push('validator self-test failed to reject a development source in a held-out cell');
  }
  const invalidSyntheticRoundtripPlan = structuredClone(
    syntheticRoundtripPlan,
  );
  invalidSyntheticRoundtripPlan.relationship_to_target_domain
    .eligible_for_model_selection = true;
  if (validateSyntheticRoundtripPlan(invalidSyntheticRoundtripPlan).length === 0) {
    failures.push('validator self-test failed to reject synthetic model-selection evidence');
  }
  const invalidSyntheticRoundtripSelection = structuredClone(
    syntheticRoundtripSelection,
  );
  invalidSyntheticRoundtripSelection.sources[0].passages[0]
    .verbatim_sha256 = '0'.repeat(64);
  if (validateSyntheticRoundtripSelection(
    invalidSyntheticRoundtripSelection,
    syntheticRoundtripPlan,
  ).length === 0) {
    failures.push('validator self-test failed to reject synthetic text digest drift');
  }
  const invalidTtsRun = structuredClone(appleTtsRun);
  invalidTtsRun.result.audio.byte_count += 4;
  if (validateAppleTtsRun(
    invalidTtsRun,
    syntheticRoundtripSelection,
  ).length === 0) {
    failures.push('validator self-test failed to reject divergent TTS audio metrics');
  }
  const invalidQwen3TtsModelManifest = structuredClone(
    qwen3TtsModelManifest,
  );
  invalidQwen3TtsModelManifest.artifacts[0].bytes += 1;
  if (validateQwen3TtsModelManifest(
    invalidQwen3TtsModelManifest,
    syntheticRoundtripPlan,
    syntheticRoundtripSelection,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject divergent Qwen3-TTS snapshot bytes',
    );
  }
  const invalidQwen3TtsRun = structuredClone(qwen3TtsRun);
  invalidQwen3TtsRun.result.asr_content_checks.backends[0]
    .quality.word_edits += 1;
  if (validateQwen3TtsRun(
    invalidQwen3TtsRun,
    syntheticRoundtripSelection,
    qwen3TtsModelManifest,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject divergent Qwen3-TTS content metrics',
    );
  }
  const invalidVoxtralTtsModelManifest = structuredClone(
    voxtralTtsModelManifest,
  );
  invalidVoxtralTtsModelManifest.artifacts[0].bytes += 1;
  if (validateVoxtralTtsModelManifest(
    invalidVoxtralTtsModelManifest,
    syntheticRoundtripPlan,
    syntheticRoundtripSelection,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject divergent Voxtral TTS snapshot bytes',
    );
  }
  const invalidVoxtralRealtimeModelManifest = structuredClone(
    voxtralRealtimeModelManifest,
  );
  invalidVoxtralRealtimeModelManifest.artifacts[0].bytes += 1;
  if (validateVoxtralRealtimeModelManifest(
    invalidVoxtralRealtimeModelManifest,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject divergent Voxtral Realtime bytes',
    );
  }
  const invalidVoxtralDirectModelManifest = structuredClone(
    voxtralRealtimeDirectModelManifest,
  );
  invalidVoxtralDirectModelManifest.batch_contract.capabilities
    .streaming_session = true;
  if (validateVoxtralRealtimeDirectModelManifest(
    invalidVoxtralDirectModelManifest,
    voxtralRealtimeModelManifest,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject a false direct Voxtral streaming capability',
    );
  }
  const invalidVoxtralDirectBoundary = structuredClone(
    voxtralRealtimeDirectBoundary,
  );
  invalidVoxtralDirectBoundary.contract.maximum_ingested_samples = 5121;
  if (validateVoxtralRealtimeDirectBoundary(
    invalidVoxtralDirectBoundary,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject an unbounded direct Voxtral step',
    );
  }
  const invalidVoxtralDirectFrontend = structuredClone(
    voxtralRealtimeDirectFrontend,
  );
  invalidVoxtralDirectFrontend.result.probe.capabilities.causal_encoder = true;
  if (validateVoxtralRealtimeDirectFrontend(
    invalidVoxtralDirectFrontend,
    voxtralRealtimeDirectFrontendOracle,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject a false direct Voxtral encoder capability',
    );
  }
  const invalidVoxtralDirectEncoder = structuredClone(
    voxtralRealtimeDirectEncoder,
  );
  invalidVoxtralDirectEncoder.result.probe.cache.layer_0_size += 1;
  if (validateVoxtralRealtimeDirectEncoder(
    invalidVoxtralDirectEncoder,
    voxtralRealtimeDirectEncoderOracle,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject a divergent direct Voxtral cache',
    );
  }
  const invalidVoxtralDirectTranscription = structuredClone(
    voxtralRealtimeDirectTranscription,
  );
  invalidVoxtralDirectTranscription.result.probe.generation.tokens[0] += 1;
  if (validateVoxtralRealtimeDirectTranscription(
    invalidVoxtralDirectTranscription,
    voxtralRealtimeDirectTranscriptionOracle,
    voxtralRealtimeDirectModelManifest,
    audiobookPilot,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject divergent direct Voxtral tokens',
    );
  }
  const invalidVoxtralRealtimeAggregate = structuredClone(
    voxtralRealtime480,
  );
  invalidVoxtralRealtimeAggregate.results[0].streaming
    .input_streaming_measured = true;
  if (validateVoxtralRealtimeAggregate(
    invalidVoxtralRealtimeAggregate,
    voxtralRealtimeModelManifest,
    480,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject false Voxtral streaming evidence',
    );
  }
  const invalidVoxtralStreaming = structuredClone(
    voxtralRealtimeStreaming320,
  );
  invalidVoxtralStreaming.runs[0].streaming.input_streaming_measured = false;
  if (validateVoxtralRealtimeStreamingRun(
    invalidVoxtralStreaming,
    voxtralRealtimeModelManifest,
    audiobookPilot,
    320,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject false Voxtral live-input evidence',
    );
  }
  const invalidVoxtralTtsRun = structuredClone(voxtralTtsRun);
  invalidVoxtralTtsRun.result.asr_content_checks.level_control.backends[0]
    .word_edits += 1;
  if (validateVoxtralTtsRun(
    invalidVoxtralTtsRun,
    syntheticRoundtripSelection,
    voxtralTtsModelManifest,
  ).length === 0) {
    failures.push(
      'validator self-test failed to reject divergent Voxtral level control',
    );
  }
  const invalidRightsReview = structuredClone(sourceRightsReviews[0]);
  invalidRightsReview.disposition = 'accepted';
  if (validateSourceRightsReview(
    invalidRightsReview,
    sourceCandidates,
    targetDomainPlan,
  ).length === 0) {
    failures.push('validator self-test failed to reject candidate-level rights acceptance');
  }
  const invalidAudiobookPilot = structuredClone(audiobookPilot);
  invalidAudiobookPilot.summary.duration_ms += 1;
  if (validateAudiobookPilot(invalidAudiobookPilot).length === 0) {
    failures.push('validator self-test failed to reject audiobook summary drift');
  }
  const invalidPostprocessing = structuredClone(postprocessingSnapshot);
  invalidPostprocessing.experiments[0].models[0].sample_count += 1;
  if (validatePostprocessingSnapshot(invalidPostprocessing).length === 0) {
    failures.push('validator self-test failed to reject divergent language counts');
  }
  const invalidPrompts = structuredClone(promptManifest);
  invalidPrompts.prompts[0].sha256 = '0'.repeat(64);
  if ((await validatePromptManifest(invalidPrompts)).length === 0) {
    failures.push('validator self-test failed to reject a prompt digest mismatch');
  }
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(
  `benchmark data: ${manifest.fixtures.length} fixture(s), ${runs.length} run record(s), ` +
  `${aggregates.length} aggregate(s), ${matrices.length} matrix record(s), ` +
  `${sourceCandidates.sources.length} source candidate(s), ` +
  `${targetDomainPlan.cells.length} target-domain cell(s), ` +
  `${syntheticRoundtripPlan.tts_candidates.length} synthetic TTS candidate(s), ` +
  `${syntheticRoundtripSelection.sources.flatMap((source) => source.passages).length} synthetic passage selector(s), ` +
  `3 measured TTS diagnostics, ` +
  `${sourceRightsReviews.length} source rights review(s), ` +
  `${audiobookPilot.fixtures.length} audiobook pilot fixture(s), ` +
  `1 direct Voxtral boundary record, 1 direct Voxtral frontend parity record, ` +
  `1 direct Voxtral encoder parity record, 1 direct Voxtral transcription parity record, ` +
  `1 postprocessing snapshot with ${postprocessingSnapshot.experiments.length} experiment(s), ` +
  `${promptManifest.prompts.length} prompt candidate(s), ` +
  `schema ${schemaVersion}`,
);
