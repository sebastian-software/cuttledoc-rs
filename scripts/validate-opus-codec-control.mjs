#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultControl = resolve(
  repoRoot,
  'benchmarks/controls/opus-codec-qwen3-tts-de-1.json',
);
const { values } = parseArgs({
  options: {
    control: { type: 'string', default: defaultControl },
    'print-derived': { type: 'boolean', default: false },
    'self-test': { type: 'boolean', default: false },
  },
});

const controlPath = resolve(values.control);
const control = JSON.parse(await readFile(controlPath, 'utf8'));
const referencePath = resolve(repoRoot, control.fixture.reference_path);
const reference = (await readFile(referencePath, 'utf8')).trim();
const derived = deriveMetrics(control, reference);

if (values['print-derived']) {
  process.stdout.write(`${JSON.stringify(derived, null, 2)}\n`);
  process.exit(0);
}

const failures = await validate(control, reference, true);
if (values['self-test']) {
  const changed = structuredClone(control);
  changed.asr_backends[0].results['64k'].wer += 0.01;
  const expectedFailures = await validate(changed, reference, false);
  if (expectedFailures.length === 0) {
    failures.push('self-test failed to reject a changed WER');
  }
}

if (failures.length > 0) {
  for (const failure of failures) process.stderr.write(`- ${failure}\n`);
  process.exit(1);
}

process.stdout.write(
  `Validated ${control.control_id}: ${control.asr_backends.length} ASR backends, ` +
    `${control.variants.length} codec variants, selected ${control.decision.selected_variant}\n`,
);

async function validate(record, referenceText, checkFiles) {
  const errors = [];
  const variantIds = ['lossless', '48k', '64k', '96k'];
  const requiredBackends = new Set([
    'apple-speechtranscriber',
    'whisper-large-v3-turbo-coreml-whispercpp',
    'qwen3-asr-0.6b-mlx-direct',
    'parakeet-tdt-0.6b-v3-coreml',
    'voxtral-realtime-4b-mlx-direct-2400ms',
  ]);
  if (record.schema_version !== '1.0.0' ||
      record.control_id !== 'opus-codec-qwen3-tts-de-1') {
    errors.push('schema_version or control_id is invalid');
  }
  if (JSON.stringify(record.variants.map((item) => item.id)) !==
      JSON.stringify(variantIds)) {
    errors.push('variants must preserve lossless, 48k, 64k, and 96k order');
  }
  if (record.encoder?.application !== 'audio' ||
      record.encoder?.variable_bit_rate !== true ||
      record.encoder?.frame_duration_ms !== 20 ||
      record.encoder?.expected_packet_loss_percent !== 0 ||
      record.encoder?.deterministic_repeat !== true) {
    errors.push('encoder contract differs from the accepted deterministic speech settings');
  }
  const backendIds = new Set(record.asr_backends.map((backend) => backend.id));
  if (backendIds.size !== requiredBackends.size ||
      [...requiredBackends].some((id) => !backendIds.has(id))) {
    errors.push('control must contain the five frozen ASR backends exactly once');
  }

  const actualMetrics = deriveMetrics(record, referenceText);
  for (const backend of record.asr_backends) {
    if (JSON.stringify(Object.keys(backend.results)) !== JSON.stringify(variantIds)) {
      errors.push(`${backend.id}: results must preserve all four variants in order`);
      continue;
    }
    for (const variantId of variantIds) {
      const result = backend.results[variantId];
      if (!(result.transcript in record.transcripts)) {
        errors.push(`${backend.id}/${variantId}: unknown transcript id`);
        continue;
      }
      const actual = actualMetrics.backends[backend.id]?.[variantId];
      if (!nearlyEqual(result.wer, actual?.wer) ||
          !nearlyEqual(result.cer, actual?.cer)) {
        errors.push(
          `${backend.id}/${variantId}: stored WER/CER ${result.wer}/${result.cer} ` +
            `differs from ${actual?.wer}/${actual?.cer}`,
        );
      }
    }
  }

  const selected = record.variants.find(
    (variant) => variant.id === record.decision?.selected_variant,
  );
  if (record.decision?.selected_variant !== '64k' ||
      record.decision?.accepted !== true ||
      selected?.target_bit_rate_bps !== 64_000 ||
      record.decision?.selected_asset_sha256 !== selected?.artifact_sha256) {
    errors.push('decision must accept the digest-pinned 64k variant');
  }
  const observedIncrease = Math.max(
    ...record.asr_backends.map((backend) => (
      backend.results['64k'].wer - backend.results.lossless.wer
    )),
    0,
  );
  if (!nearlyEqual(
    record.decision?.maximum_observed_wer_increase,
    observedIncrease,
  ) || observedIncrease > record.decision?.maximum_allowed_wer_increase) {
    errors.push('selected codec exceeds or misstates the accepted WER-increase gate');
  }

  if (checkFiles) {
    const referenceBytes = await readFile(referencePath);
    if (digest(referenceBytes) !== record.fixture.reference_sha256) {
      errors.push('reference text digest differs from the control record');
    }
    const assetPath = resolve(repoRoot, record.decision.selected_asset);
    const assetBytes = await readFile(assetPath);
    if (digest(assetBytes) !== record.decision.selected_asset_sha256 ||
        assetBytes.length !== selected.bytes) {
      errors.push('selected Opus asset bytes differ from the control record');
    }
    const manifest = JSON.parse(await readFile(
      resolve(repoRoot, record.fixture.asset_manifest),
      'utf8',
    ));
    if (manifest.archive?.sha256 !== selected.artifact_sha256 ||
        manifest.archive?.bytes !== selected.bytes ||
        manifest.reference?.sha256 !== record.fixture.reference_sha256 ||
        manifest.codec_control !==
          'benchmarks/controls/opus-codec-qwen3-tts-de-1.json') {
      errors.push('asset manifest and codec control do not reconcile');
    }
  }
  return errors;
}

function deriveMetrics(record, referenceText) {
  const backends = {};
  for (const backend of record.asr_backends) {
    backends[backend.id] = {};
    for (const [variantId, result] of Object.entries(backend.results)) {
      const hypothesis = record.transcripts[result.transcript];
      if (typeof hypothesis !== 'string') continue;
      backends[backend.id][variantId] = {
        wer: errorRate(words(referenceText), words(hypothesis)),
        cer: errorRate(characters(referenceText), characters(hypothesis)),
      };
    }
  }
  return {
    reference_word_count: words(referenceText).length,
    reference_character_count: characters(referenceText).length,
    backends,
  };
}

function words(text) {
  return text
    .toLocaleLowerCase('en-US')
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .split(/\s+/)
    .filter(Boolean);
}

function characters(text) {
  return words(text).join('').split('');
}

function errorRate(referenceItems, hypothesisItems) {
  if (referenceItems.length === 0) return hypothesisItems.length === 0 ? 0 : 1;
  let previous = Array.from(
    { length: hypothesisItems.length + 1 },
    (_, index) => index,
  );
  for (let referenceIndex = 1; referenceIndex <= referenceItems.length; referenceIndex += 1) {
    const current = [referenceIndex];
    for (let hypothesisIndex = 1; hypothesisIndex <= hypothesisItems.length; hypothesisIndex += 1) {
      const substitution = previous[hypothesisIndex - 1] + (
        referenceItems[referenceIndex - 1] === hypothesisItems[hypothesisIndex - 1] ? 0 : 1
      );
      current[hypothesisIndex] = Math.min(
        previous[hypothesisIndex] + 1,
        current[hypothesisIndex - 1] + 1,
        substitution,
      );
    }
    previous = current;
  }
  return previous.at(-1) / referenceItems.length;
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function nearlyEqual(left, right) {
  return Number.isFinite(left) && Number.isFinite(right) &&
    Math.abs(left - right) <= 1e-15;
}
