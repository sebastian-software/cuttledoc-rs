#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({
  options: {
    'materialized-dir': { type: 'string' },
    'audio-dir': { type: 'string' },
    'asr-dir': { type: 'string' },
    output: { type: 'string' },
  },
});

for (const option of ['materialized-dir', 'audio-dir', 'asr-dir', 'output']) {
  if (!(values[option]?.length > 0)) throw new Error(`--${option} is required`);
}

const materializedDirectory = resolve(values['materialized-dir']);
const audioDirectory = resolve(values['audio-dir']);
const asrDirectory = resolve(values['asr-dir']);
const outputPath = resolve(values.output);
const selection = JSON.parse(await readFile(join(
  repoRoot,
  'benchmarks/fixtures/synthetic-roundtrip-selection.json',
), 'utf8'));

const plan = [
  ['synthetic-de-description', 'com.apple.voice.compact.de-DE.Anna', 'Anna', 22_050],
  ['synthetic-de-origin', 'com.apple.eloquence.de-DE.Eddy', 'Eddy', 16_000],
  ['synthetic-de-precursors', 'com.apple.eloquence.de-DE.Flo', 'Flo', 16_000],
  ['synthetic-de-turing-test', 'com.apple.eloquence.de-DE.Grandma', 'Grandma', 16_000],
  ['synthetic-de-methods', 'com.apple.eloquence.de-DE.Grandpa', 'Grandpa', 16_000],
  ['synthetic-de-applications', 'com.apple.eloquence.de-DE.Reed', 'Reed', 16_000],
  ['synthetic-de-native', 'com.apple.eloquence.de-DE.Rocko', 'Rocko', 16_000],
  ['synthetic-de-dialogue', 'com.apple.eloquence.de-DE.Sandy', 'Sandy', 16_000],
];
const selectedPassages = new Map(selection.sources.flatMap((source) =>
  source.passages.map((passage) => [passage.id, { source, passage }])));

const sections = [];
for (const [id, voiceIdentifier, voiceName, rawSampleRateHz] of plan) {
  const selected = selectedPassages.get(id);
  if (!selected) throw new Error(`${id} is absent from the pinned selection`);
  const [referenceBytes, rawAudio, asr] = await Promise.all([
    readFile(join(materializedDirectory, `${id}.txt`)),
    readFile(join(audioDirectory, `${id}.f32le`)),
    readFile(join(asrDirectory, `${id}.json`), 'utf8').then(JSON.parse),
  ]);
  const reference = referenceBytes.toString('utf8').trim();
  const transcript = asr.repetitions?.at(-1)?.text?.trim();
  if (sha256(referenceBytes) !== selected.passage.spoken_sha256 ||
      [...referenceBytes.toString('utf8')].length !==
        selected.passage.character_count) {
    throw new Error(`${id} reference differs from the pinned selection`);
  }
  if (asr.backend !== 'whisper' || !(transcript?.length > 0) ||
      asr.fixture?.reference?.trim() !== reference ||
      asr.fixture?.source_sha256 !== sha256(await readFile(
        join(audioDirectory, `${id}.16k.wav`),
      ))) {
    throw new Error(`${id} ASR evidence is incomplete or mismatched`);
  }
  if (rawAudio.length === 0 || rawAudio.length % 4 !== 0) {
    throw new Error(`${id} raw audio is not complete f32le PCM`);
  }
  const referenceWords = normalizedWords(reference).length;
  const rawEditDistance = editDistance(reference, transcript);
  sections.push({
    id,
    source_id: selected.source.id,
    source_title: selected.source.title,
    source_license: selected.source.license,
    source_revision: selected.source.revision_id ?? selected.source.revision,
    phenomena: selected.passage.phenomena,
    voice: {
      runtime: 'macOS AVFAudio AVSpeechSynthesizer',
      identifier: voiceIdentifier,
      name: voiceName,
      locale: 'de-DE',
      quality_raw_value: 1,
    },
    audio: {
      storage: 'local-required',
      raw_format: 'mono-f32le',
      raw_sample_rate_hz: rawSampleRateHz,
      raw_sample_count: rawAudio.length / 4,
      raw_sha256: sha256(rawAudio),
      normalized_format: 'mono-pcm16-wav',
      normalized_sample_rate_hz: 16_000,
      normalized_sha256: asr.fixture.source_sha256,
      duration_ms: asr.fixture.audio_duration_ms,
    },
    transcript,
    evaluation_reference: reference,
    protected_spans: [],
    diagnostics: {
      reference_word_count: referenceWords,
      raw_word_edit_distance: rawEditDistance,
      raw_wer: rawEditDistance / referenceWords,
    },
  });
}

const referenceWordCount = sections.reduce(
  (sum, section) => sum + section.diagnostics.reference_word_count,
  0,
);
const rawEditDistance = sections.reduce(
  (sum, section) => sum + section.diagnostics.raw_word_edit_distance,
  0,
);
const transcript = sections.map((section) => section.transcript).join('\n\n');
const evaluationReference = sections
  .map((section) => section.evaluation_reference)
  .join('\n\n');
const fixture = {
  schema_version: '1.0.0',
  id: 'issue20-de-synthetic-multipage-whisper-1',
  purpose: 'development-prompt-probe',
  language: 'German',
  locale: 'de-DE',
  domain: 'clean synthetic documentary prose and dialogue',
  asr_backend: 'whisper-large-v3-turbo-coreml-whispercpp',
  selection_revision: selection.revision,
  gold_status: 'synthetic-source-reference',
  development_only: true,
  transcript,
  evaluation_reference: evaluationReference,
  error_profile: 'This development round trip includes clean German system voices and may contain proper-name substitutions, English technical-term substitutions, spoken punctuation labels, abbreviation expansions, or end-of-passage hallucinations. These are possibilities, not proof that a span is wrong.',
  glossary: [],
  protected_spans: [],
  sections,
  aggregate: {
    section_count: sections.length,
    distinct_voice_count: new Set(sections.map((section) =>
      section.voice.identifier)).size,
    reference_word_count: referenceWordCount,
    raw_word_edit_distance: rawEditDistance,
    raw_micro_wer: rawEditDistance / referenceWordCount,
    audio_duration_ms: sections.reduce(
      (sum, section) => sum + section.audio.duration_ms,
      0,
    ),
  },
  inference_policy: {
    prompt_id: 'conservative-sections-v1',
    reference_visible_to_model: false,
    lexical_changes_allowed: true,
    section_ids_and_order_must_match: true,
    allowed_edit_classes: [
      'surface',
      'word-boundary',
      'contextual-misrecognition',
      'asr-inflection',
    ],
  },
  provenance: {
    selection_manifest: 'benchmarks/fixtures/synthetic-roundtrip-selection.json',
    materializer: 'scripts/materialize-postprocessing-long-form-fixture.mjs',
    tts_boundary: 'repository-owned Swift C ABI called from Rust',
    tts_generation: 'Eight explicit installed Apple voice identifiers at default rate, pitch, and volume.',
    asr_runner: 'scripts/run-legacy-asr-baseline.mjs',
    source_license: 'CC-BY-SA-4.0',
    audio_storage: 'The generated lossless audio remains local; exact raw and normalized digests bind every transcript to its input.',
    claim_limit: 'Development-only synthetic round-trip evidence. The exact source text measures restoration and regressions, but does not replace held-out human-verified professional audio.',
  },
};

await writeFile(outputPath, `${JSON.stringify(fixture, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  output: outputPath,
  section_count: sections.length,
  voice_count: fixture.aggregate.distinct_voice_count,
  reference_word_count: referenceWordCount,
  audio_duration_ms: fixture.aggregate.audio_duration_ms,
  raw_word_edit_distance: rawEditDistance,
  raw_micro_wer: fixture.aggregate.raw_micro_wer,
}, null, 2)}\n`);

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function normalizedWords(text) {
  return [...text.normalize('NFC').toLocaleLowerCase('und')]
    .map((character) => /[\p{L}\p{N}]/u.test(character) ? character : ' ')
    .join('')
    .split(/\s+/u)
    .filter(Boolean);
}

function editDistance(reference, hypothesis) {
  const referenceWords = normalizedWords(reference);
  const hypothesisWords = normalizedWords(hypothesis);
  let previous = Array.from(
    { length: hypothesisWords.length + 1 },
    (_, index) => index,
  );
  for (let referenceIndex = 1;
    referenceIndex <= referenceWords.length;
    referenceIndex += 1) {
    const current = [referenceIndex];
    for (let hypothesisIndex = 1;
      hypothesisIndex <= hypothesisWords.length;
      hypothesisIndex += 1) {
      current.push(Math.min(
        current.at(-1) + 1,
        previous[hypothesisIndex] + 1,
        previous[hypothesisIndex - 1] + Number(
          referenceWords[referenceIndex - 1] !==
            hypothesisWords[hypothesisIndex - 1],
        ),
      ));
    }
    previous = current;
  }
  return previous.at(-1);
}
