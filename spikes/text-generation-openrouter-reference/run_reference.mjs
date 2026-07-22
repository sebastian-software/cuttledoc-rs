#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { performance } from 'node:perf_hooks';

const sha256 = (value) => createHash('sha256').update(value).digest('hex');

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag?.startsWith('--') || value === undefined) {
      throw new Error(`invalid argument sequence near ${flag ?? '<end>'}`);
    }
    parsed[flag.slice(2)] = value;
  }
  for (const required of [
    'manifest',
    'experiment',
    'fixture',
    'prompt',
    'output',
    'source-revision',
  ]) {
    if (!(required in parsed)) throw new Error(`missing --${required}`);
  }
  return parsed;
}

function normalizedWords(text) {
  return [...text.normalize('NFC').toLocaleLowerCase('und')]
    .map((character) => /[\p{L}\p{N}]/u.test(character) ? character : ' ')
    .join('')
    .split(/\s+/u)
    .filter(Boolean);
}

function wordErrorRate(reference, hypothesis) {
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
  return referenceWords.length === 0 ? 0 : previous.at(-1) / referenceWords.length;
}

function wordDiff(source, target) {
  const sourceWords = normalizedWords(source);
  const targetWords = normalizedWords(target);
  const distances = Array.from(
    { length: sourceWords.length + 1 },
    () => Array(targetWords.length + 1).fill(0),
  );
  for (let index = 0; index <= sourceWords.length; index += 1) {
    distances[index][0] = index;
  }
  for (let index = 0; index <= targetWords.length; index += 1) {
    distances[0][index] = index;
  }
  for (let sourceIndex = 1; sourceIndex <= sourceWords.length; sourceIndex += 1) {
    for (let targetIndex = 1; targetIndex <= targetWords.length; targetIndex += 1) {
      distances[sourceIndex][targetIndex] = Math.min(
        distances[sourceIndex - 1][targetIndex] + 1,
        distances[sourceIndex][targetIndex - 1] + 1,
        distances[sourceIndex - 1][targetIndex - 1] + Number(
          sourceWords[sourceIndex - 1] !== targetWords[targetIndex - 1],
        ),
      );
    }
  }

  const operations = [];
  let sourceIndex = sourceWords.length;
  let targetIndex = targetWords.length;
  while (sourceIndex > 0 || targetIndex > 0) {
    if (sourceIndex > 0 && targetIndex > 0 &&
        sourceWords[sourceIndex - 1] === targetWords[targetIndex - 1] &&
        distances[sourceIndex][targetIndex] ===
          distances[sourceIndex - 1][targetIndex - 1]) {
      sourceIndex -= 1;
      targetIndex -= 1;
    } else if (sourceIndex > 0 && targetIndex > 0 &&
        distances[sourceIndex][targetIndex] ===
          distances[sourceIndex - 1][targetIndex - 1] + 1) {
      operations.push({
        type: 'replace',
        input_index: sourceIndex - 1,
        output_index: targetIndex - 1,
        input: sourceWords[sourceIndex - 1],
        output: targetWords[targetIndex - 1],
      });
      sourceIndex -= 1;
      targetIndex -= 1;
    } else if (sourceIndex > 0 &&
        distances[sourceIndex][targetIndex] ===
          distances[sourceIndex - 1][targetIndex] + 1) {
      operations.push({
        type: 'delete',
        input_index: sourceIndex - 1,
        output_index: targetIndex,
        input: sourceWords[sourceIndex - 1],
        output: null,
      });
      sourceIndex -= 1;
    } else {
      operations.push({
        type: 'insert',
        input_index: sourceIndex,
        output_index: targetIndex - 1,
        input: null,
        output: targetWords[targetIndex - 1],
      });
      targetIndex -= 1;
    }
  }
  operations.reverse();
  return {
    input_word_count: sourceWords.length,
    output_word_count: targetWords.length,
    edit_distance: distances.at(-1).at(-1),
    operations,
  };
}

function countSequence(words, sequence) {
  if (sequence.length === 0) return 0;
  let count = 0;
  for (let index = 0; index <= words.length - sequence.length; index += 1) {
    if (sequence.every((word, offset) => words[index + offset] === word)) count += 1;
  }
  return count;
}

function pairCounts(pairs) {
  const counts = new Map();
  for (const pair of pairs) {
    const key = JSON.stringify(pair);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

function equalCounts(left, right) {
  return left.size === right.size &&
    [...left].every(([key, value]) => right.get(key) === value);
}

function invalidOutput(text, edits, error) {
  return {
    text,
    reported_edits: edits,
    parser: { valid: false, error },
    audit: {
      lexical_edits_fully_reported: false,
      protected_spans_unchanged: false,
      reported_lexical_edit_count: null,
    },
  };
}

function parseOutput(rawText, fixture) {
  let parsed;
  try {
    parsed = JSON.parse(rawText);
  } catch (error) {
    return invalidOutput('', null, error.message);
  }
  if (parsed === null || Array.isArray(parsed) ||
      typeof parsed !== 'object' || typeof parsed.text !== 'string') {
    return invalidOutput('', null, 'output must be an object with a string text field');
  }
  const allowedClasses = new Set([
    'surface',
    'word-boundary',
    'contextual-misrecognition',
    'asr-inflection',
  ]);
  const validEdits = Array.isArray(parsed.edits) && parsed.edits.every((edit) =>
    edit !== null && !Array.isArray(edit) && typeof edit === 'object' &&
    typeof edit.original === 'string' &&
    typeof edit.replacement === 'string' &&
    allowedClasses.has(edit.class) &&
    typeof edit.reason === 'string' &&
    Number.isFinite(edit.confidence) &&
    edit.confidence >= 0 && edit.confidence <= 1,
  );
  if (!validEdits) {
    return invalidOutput(
      parsed.text,
      parsed.edits,
      'edits must match the bounded lexical audit schema',
    );
  }

  const diff = wordDiff(fixture.transcript, parsed.text);
  const operationPairs = pairCounts(diff.operations.map((operation) => [
    normalizedWords(operation.input ?? ''),
    normalizedWords(operation.output ?? ''),
  ]));
  const reportedPairs = pairCounts(parsed.edits
    .map((edit) => [
      normalizedWords(edit.original),
      normalizedWords(edit.replacement),
    ])
    .filter(([original, replacement]) =>
      JSON.stringify(original) !== JSON.stringify(replacement)));
  const inputWords = normalizedWords(fixture.transcript);
  const outputWords = normalizedWords(parsed.text);
  const protectedSpansUnchanged = (fixture.protected_spans ?? []).every((span) => {
    const words = normalizedWords(span);
    return countSequence(inputWords, words) === countSequence(outputWords, words);
  });
  return {
    text: parsed.text,
    reported_edits: parsed.edits,
    parser: { valid: true, error: null },
    audit: {
      lexical_edits_fully_reported: equalCounts(operationPairs, reportedPairs),
      protected_spans_unchanged: protectedSpansUnchanged,
      reported_lexical_edit_count: [...reportedPairs.values()]
        .reduce((sum, value) => sum + value, 0),
    },
  };
}

function renderPrompt(template, fixture, contract) {
  const renderValue = (value) => typeof value === 'string'
    ? value
    : JSON.stringify(value);
  const replacements = {
    '{{language}}': fixture.language,
    '{{locale}}': fixture.locale ?? 'unspecified',
    '{{domain}}': fixture.domain,
    '{{backend}}': fixture.asr_backend ?? 'unspecified',
    '{{error_profile}}': fixture.error_profile ??
      'No independent error profile supplied',
    '{{glossary}}': fixture.glossary ?? [],
    '{{protected_spans}}': fixture.protected_spans ?? [],
    '{{suspect_spans}}': fixture.suspect_spans ?? [],
    '{{transcript}}': fixture.transcript,
  };
  let rendered = template;
  for (const [marker, value] of Object.entries(replacements)) {
    rendered = rendered.replaceAll(marker, renderValue(value));
  }
  if (contract.render_mode !== 'template') {
    throw new Error(`unsupported render mode: ${contract.render_mode}`);
  }
  if (rendered.includes('{{') || rendered.includes('}}')) {
    throw new Error('prompt contains an unresolved template marker');
  }
  return rendered;
}

function correctionSchema() {
  return {
    type: 'object',
    additionalProperties: false,
    required: ['text', 'edits'],
    properties: {
      text: { type: 'string' },
      edits: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['original', 'replacement', 'class', 'reason', 'confidence'],
          properties: {
            original: { type: 'string' },
            replacement: { type: 'string' },
            class: {
              type: 'string',
              enum: [
                'surface',
                'word-boundary',
                'contextual-misrecognition',
                'asr-inflection',
              ],
            },
            reason: { type: 'string' },
            confidence: { type: 'number', minimum: 0, maximum: 1 },
          },
        },
      },
    },
  };
}

async function runRequest(manifest, contract, prompt, apiKey) {
  const request = manifest.request_defaults;
  const body = {
    model: manifest.model.requested_id,
    messages: [{ role: 'user', content: prompt }],
    response_format: {
      type: 'json_schema',
      json_schema: {
        name: 'bounded_transcript_correction_v1',
        strict: true,
        schema: correctionSchema(),
      },
    },
    provider: {
      only: [manifest.gateway.provider_slug],
      allow_fallbacks: false,
      require_parameters: true,
      data_collection: 'deny',
      zdr: true,
    },
  };
  body[request.token_limit_field] = contract.max_tokens;
  if (request.temperature !== null) body.temperature = request.temperature;
  if (request.seed !== null) body.seed = request.seed;
  if (request.reasoning !== null) body.reasoning = request.reasoning;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), contract.timeout_ms);
  const started = performance.now();
  let response;
  try {
    response = await fetch(manifest.gateway.endpoint, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
  const completed = performance.now();
  const responseText = await response.text();
  if (!response.ok) {
    throw new Error(
      `OpenRouter request failed with HTTP ${response.status}: ` +
      responseText.slice(0, 800),
    );
  }
  const envelope = JSON.parse(responseText);
  const content = envelope.choices?.[0]?.message?.content;
  if (typeof content !== 'string') {
    throw new Error('OpenRouter response has no string message content');
  }
  if (envelope.provider !== manifest.gateway.provider_name) {
    throw new Error(
      `provider pin failed: expected ${manifest.gateway.provider_name}, ` +
      `received ${envelope.provider ?? '<missing>'}`,
    );
  }
  if (!(manifest.model.allowed_response_ids ?? [manifest.model.requested_id])
    .includes(envelope.model)) {
    throw new Error(
      `model identity drift: requested ${manifest.model.requested_id}, ` +
      `received ${envelope.model ?? '<missing>'}`,
    );
  }
  return {
    text: content.trim(),
    evidence: {
      response_id: envelope.id,
      served_model: envelope.model,
      served_provider: envelope.provider,
      finish_reason: envelope.choices?.[0]?.finish_reason ?? null,
      client_observed_complete_ms: completed - started,
      usage: {
        prompt_tokens: envelope.usage?.prompt_tokens ?? null,
        completion_tokens: envelope.usage?.completion_tokens ?? null,
        total_tokens: envelope.usage?.total_tokens ?? null,
        cost_usd: envelope.usage?.cost ?? null,
        is_byok: envelope.usage?.is_byok ?? false,
      },
    },
  };
}

function assertSuffix(actual, expected, label) {
  if (!actual.replaceAll('\\', '/').endsWith(expected)) {
    throw new Error(`${label} does not match the pinned repository path`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!/^[0-9a-f]{40}$/.test(args['source-revision'])) {
    throw new Error('--source-revision must be a lowercase 40-character Git SHA');
  }
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY is not set');

  const [manifest, experiment, fixture, promptBytes] = await Promise.all([
    readFile(args.manifest, 'utf8').then(JSON.parse),
    readFile(args.experiment, 'utf8').then(JSON.parse),
    readFile(args.fixture, 'utf8').then(JSON.parse),
    readFile(args.prompt),
  ]);
  const contract = experiment.generation_contract;
  const resultContract = experiment.result_contract;
  assertSuffix(args.manifest, experiment.model_manifest_path, 'manifest');
  assertSuffix(args.fixture, resultContract.fixture_path, 'fixture');
  assertSuffix(args.prompt, contract.prompt_path, 'prompt');
  if (sha256(promptBytes) !== contract.prompt_sha256) {
    throw new Error('prompt SHA-256 drift');
  }
  if (fixture.inference_policy?.reference_visible_to_model !== false) {
    throw new Error('evaluation reference must remain hidden from the model');
  }

  const renderedPrompt = renderPrompt(promptBytes.toString('utf8'), fixture, contract);
  const first = await runRequest(manifest, contract, renderedPrompt, apiKey);
  const repeat = await runRequest(manifest, contract, renderedPrompt, apiKey);
  const parsed = parseOutput(first.text, fixture);
  const correctedText = parsed.text.trim();
  const lexicalDiff = wordDiff(fixture.transcript, correctedText);
  const nonempty = correctedText.length > 0;
  const lexicalInvariant = normalizedWords(correctedText).join('\n') ===
    normalizedWords(fixture.transcript).join('\n');
  const mechanicallyAccepted = nonempty && parsed.parser.valid &&
    parsed.audit.lexical_edits_fully_reported &&
    parsed.audit.protected_spans_unchanged;
  const requests = [first.evidence, repeat.evidence];
  const costs = requests.map((request) => request.usage.cost_usd);

  const result = {
    schema_version: '1.0.0',
    run_id: resultContract.run_id,
    captured_at: new Date().toISOString(),
    source_revision: args['source-revision'],
    purpose: resultContract.purpose,
    candidate: {
      task: 'text-generation',
      manifest_id: manifest.id,
      role: manifest.candidate_role,
      model: manifest.model,
      gateway: {
        name: manifest.gateway.name,
        requested_model: manifest.model.requested_id,
        provider_slug: manifest.gateway.provider_slug,
        provider_name: manifest.gateway.provider_name,
        served_model: first.evidence.served_model,
        served_provider: first.evidence.served_provider,
      },
    },
    fixture: {
      id: fixture.id,
      language: fixture.language,
      locale: fixture.locale,
      domain: fixture.domain,
      asr_backend: fixture.asr_backend,
      development_only: fixture.development_only,
      gold_status: fixture.gold_status,
      transcript_sha256: sha256(fixture.transcript),
      reference_visible_to_model: false,
    },
    procedure: {
      experiment_id: experiment.id,
      generation: contract,
      gateway_request: {
        token_limit_field: manifest.request_defaults.token_limit_field,
        response_format: manifest.request_defaults.response_format,
        temperature: manifest.request_defaults.temperature,
        seed: manifest.request_defaults.seed,
        reasoning: manifest.request_defaults.reasoning,
        provider: manifest.gateway.request_policy,
      },
      complete_generation_repetitions: 2,
      prompt_visible_fields: contract.context_fields,
      evaluation_reference_visible_to_model: false,
      execution_boundary: 'Remote OpenRouter API; no local model-runtime claim',
    },
    measurements: {
      requests,
      aggregate: {
        total_cost_usd: costs.every(Number.isFinite)
          ? costs.reduce((sum, cost) => sum + cost, 0)
          : null,
        total_prompt_tokens: requests.reduce(
          (sum, request) => sum + (request.usage.prompt_tokens ?? 0),
          0,
        ),
        total_completion_tokens: requests.reduce(
          (sum, request) => sum + (request.usage.completion_tokens ?? 0),
          0,
        ),
      },
      deterministic_repeat: {
        text_identical: first.text === repeat.text,
        requirement: 'observation-only; remote repeat equality is not an acceptance gate',
      },
      latency_claim_limit: 'Client-observed request duration includes gateway, queue, provider, and network time and is not comparable to local MLX measurements.',
    },
    output: {
      text: correctedText,
      text_sha256: sha256(correctedText),
      raw_text: first.text,
      raw_text_sha256: sha256(first.text),
      reported_edits: parsed.reported_edits,
      parser: parsed.parser,
      audit: parsed.audit,
      lexical_diff: lexicalDiff,
      gates: {
        policy_mode: contract.policy_mode,
        nonempty,
        case_and_punctuation_insensitive_lexical_invariant: lexicalInvariant,
        accepted: mechanicallyAccepted,
        quality_accepted: false,
      },
      diagnostic_wer: {
        input: wordErrorRate(fixture.evaluation_reference, fixture.transcript),
        output: wordErrorRate(fixture.evaluation_reference, correctedText),
        reference_status: fixture.gold_status,
        claim_limit: 'Development-only diagnostic against an unverified dataset transcript; not a quality-selection result.',
      },
    },
    conclusion: {
      remote_reference_executed: true,
      development_output_contract_accepted: mechanicallyAccepted,
      model_quality_selected: false,
      product_runtime_accepted: false,
      claim_limit: 'This run screens one development fixture and API contract. Human-verified target-domain audio remains required for model selection.',
    },
  };

  await mkdir(dirname(args.output), { recursive: true });
  await writeFile(args.output, `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify({
    run_id: result.run_id,
    model: result.candidate.gateway.served_model,
    provider: result.candidate.gateway.served_provider,
    accepted: mechanicallyAccepted,
    lexical_edits: lexicalDiff.edit_distance,
    diagnostic_wer: result.output.diagnostic_wer,
    total_cost_usd: result.measurements.aggregate.total_cost_usd,
  }));
}

await main();
