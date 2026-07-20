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
const sourceRightsPaths = (await readdir(sourceRightsDirectory))
  .filter((name) => name.endsWith('.json'))
  .sort()
  .map((name) => join(sourceRightsDirectory, name));
const sourceRightsReviews = await Promise.all(
  sourceRightsPaths.map((path) => readJson(path)),
);
const audiobookPilot = await readJson(audiobookPilotPath);
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
  `${sourceRightsReviews.length} source rights review(s), ` +
  `${audiobookPilot.fixtures.length} audiobook pilot fixture(s), ` +
  `1 postprocessing snapshot with ${postprocessingSnapshot.experiments.length} experiment(s), ` +
  `${promptManifest.prompts.length} prompt candidate(s), ` +
  `schema ${schemaVersion}`,
);
