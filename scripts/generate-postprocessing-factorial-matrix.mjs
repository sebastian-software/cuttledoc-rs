#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultPlanPath = 'benchmarks/postprocessing/factorial-plan.json';
const defaultOutputPath = 'benchmarks/postprocessing/factorial-cells.json';

function parseArgs(argv) {
  const options = {
    planPath: defaultPlanPath,
    outputPath: defaultOutputPath,
    mode: 'print',
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--plan') {
      options.planPath = argv[++index];
    } else if (argument === '--output') {
      options.outputPath = argv[++index];
    } else if (argument === '--write') {
      options.mode = 'write';
    } else if (argument === '--check') {
      options.mode = 'check';
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }

  if (!options.planPath || !options.outputPath) {
    throw new Error('--plan and --output require paths');
  }
  return options;
}

async function readJson(path) {
  return JSON.parse(await readFile(resolve(repoRoot, path), 'utf8'));
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(values, seed) {
  const result = [...values];
  const random = mulberry32(seed);
  for (let index = result.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [result[index], result[target]] = [result[target], result[index]];
  }
  return result;
}

function unique(values) {
  return [...new Set(values)];
}

function passageIds(selection) {
  return new Set(
    selection.sources.flatMap((source) =>
      source.passages.map((passage) => passage.id)),
  );
}

function realizationSpecs(plan, engineId) {
  const policy = plan.generation_sampling.engine_policies.find(
    (candidate) => candidate.tts_engine === engineId,
  );
  if (!policy) throw new Error(`Missing generation policy for ${engineId}`);

  if (policy.seeds.length > 0) {
    return policy.seeds.flatMap((seed) =>
      Array.from(
        { length: policy.repetitions_per_seed_or_voice },
        (_, index) => ({
          id: `seed-${seed}-repeat-${index + 1}`,
          seed,
          repeat: index + 1,
        }),
      ));
  }

  return Array.from(
    { length: policy.repetitions_per_seed_or_voice },
    (_, index) => ({
      id: `repeat-${index + 1}`,
      seed: null,
      repeat: index + 1,
    }),
  );
}

function readinessForPassage(slot, knownPassageIds) {
  const blockers = [];
  if (slot.status !== 'materialized') {
    blockers.push(`passage-slot:${slot.status}`);
  }
  if (slot.passage_id === null) {
    blockers.push('passage-id:missing');
  } else if (!knownPassageIds.has(slot.passage_id)) {
    blockers.push('passage-id:not-in-source-selection');
  }
  return blockers;
}

function readinessForVoice(slot) {
  return slot.status === 'qualified' ? [] : [`voice-slot:${slot.status}`];
}

function unitStatus(blockers) {
  return blockers.length === 0 ? 'ready' : 'blocked';
}

function makePrimaryAudioUnits(plan, knownPassageIds) {
  const units = [];
  for (const passage of plan.passage_slots) {
    for (const engine of plan.axes.tts_engines) {
      const voices = plan.voice_slots.filter((voice) =>
        voice.locale === passage.locale && voice.tts_engine === engine.id);
      for (const voice of voices) {
        for (const realization of realizationSpecs(plan, engine.id)) {
          const blockers = unique([
            ...readinessForPassage(passage, knownPassageIds),
            ...readinessForVoice(voice),
          ]);
          units.push({
            id: `audio--${passage.id}--${voice.id}--${realization.id}`,
            kind: 'primary',
            locale: passage.locale,
            content_type: passage.content_type,
            passage_slot_id: passage.id,
            passage_id: passage.passage_id,
            tts_engine: engine.id,
            voice_slot_id: voice.id,
            realization_id: realization.id,
            generation_seed: realization.seed,
            generation_repeat: realization.repeat,
            status: unitStatus(blockers),
            blockers,
          });
        }
      }
    }
  }
  return units;
}

function makeSameSeedControlAudioUnits(plan, knownPassageIds) {
  const control = plan.generation_sampling.same_seed_reproducibility_control;
  const units = [];
  for (const locale of plan.axes.locales) {
    const suffix = control.passage_slot_per_locale;
    const passage = plan.passage_slots.find((slot) =>
      slot.locale === locale && slot.id.endsWith(suffix));
    if (!passage) throw new Error(`Missing ${locale} ${suffix} control passage`);
    const voices = plan.voice_slots.filter((voice) =>
      voice.locale === locale && voice.tts_engine === control.tts_engine);
    for (const voice of voices) {
      const blockers = unique([
        ...readinessForPassage(passage, knownPassageIds),
        ...readinessForVoice(voice),
      ]);
      units.push({
        id: `audio-control--${passage.id}--${voice.id}--seed-${control.seed}-repeat-${control.additional_repetitions + 1}`,
        kind: 'same-seed-reproducibility-control',
        locale,
        content_type: passage.content_type,
        passage_slot_id: passage.id,
        passage_id: passage.passage_id,
        tts_engine: control.tts_engine,
        voice_slot_id: voice.id,
        realization_id: `seed-${control.seed}-control-repeat-${control.additional_repetitions + 1}`,
        generation_seed: control.seed,
        generation_repeat: control.additional_repetitions + 1,
        status: unitStatus(blockers),
        blockers,
      });
    }
  }
  return units;
}

function makeSttUnits(plan, audioUnits) {
  return audioUnits.flatMap((audio) =>
    plan.axes.stt_models.map((model) => ({
      id: `stt--${audio.id}--${model.id}`,
      audio_unit_id: audio.id,
      kind: audio.kind,
      locale: audio.locale,
      content_type: audio.content_type,
      passage_slot_id: audio.passage_slot_id,
      stt_model: model.id,
      status: audio.status,
      blockers: audio.blockers,
    })),
  );
}

function makeDocuments(plan, primaryAudioUnits, primarySttUnits) {
  const audioByCoordinates = new Map(primaryAudioUnits.map((unit) => [
    [unit.passage_slot_id, unit.voice_slot_id, unit.realization_id].join('|'),
    unit,
  ]));
  const sttByCoordinates = new Map(primarySttUnits.map((unit) => [
    [unit.audio_unit_id, unit.stt_model].join('|'),
    unit,
  ]));
  const documents = [];

  for (const locale of plan.axes.locales) {
    const passageSlots = plan.document_grouping.section_order.map((suffix) => {
      const slot = plan.passage_slots.find((candidate) =>
        candidate.locale === locale && candidate.id.endsWith(suffix));
      if (!slot) throw new Error(`Missing ${locale} ${suffix} passage slot`);
      return slot;
    });
    for (const engine of plan.axes.tts_engines) {
      const voices = plan.voice_slots.filter((voice) =>
        voice.locale === locale && voice.tts_engine === engine.id);
      for (const voice of voices) {
        for (const realization of realizationSpecs(plan, engine.id)) {
          for (const sttModel of plan.axes.stt_models) {
            const sections = passageSlots.map((passage) => {
              const coordinates = [passage.id, voice.id, realization.id].join('|');
              const audio = audioByCoordinates.get(coordinates);
              if (!audio) throw new Error(`Missing audio unit for ${coordinates}`);
              const stt = sttByCoordinates.get(
                [audio.id, sttModel.id].join('|'),
              );
              if (!stt) throw new Error(`Missing STT unit for ${audio.id}`);
              return {
                section_id: passage.id,
                content_type: passage.content_type,
                passage_id: passage.passage_id,
                stt_unit_id: stt.id,
              };
            });
            const blockers = unique(sections.flatMap((section) =>
              sttByCoordinates.get([
                audioByCoordinates.get([
                  section.section_id,
                  voice.id,
                  realization.id,
                ].join('|')).id,
                sttModel.id,
              ].join('|')).blockers));
            documents.push({
              id: `document--${locale}--${voice.id}--${realization.id}--${sttModel.id}`,
              locale,
              tts_engine: engine.id,
              voice_slot_id: voice.id,
              realization_id: realization.id,
              stt_model: sttModel.id,
              sections,
              status: unitStatus(blockers),
              blockers,
            });
          }
        }
      }
    }
  }
  return documents;
}

function makeLlmUnits(plan, documents) {
  const modelRuns = [];
  const requests = [];
  for (const document of documents) {
    for (const model of plan.axes.llm_models) {
      const runId = `llm-run--${document.id}--${model.id.replaceAll('/', '--')}`;
      const requestIds = [];
      for (let repeat = 1; repeat <= plan.axes.llm_repetitions; repeat += 1) {
        const requestId = `${runId}--repeat-${repeat}`;
        requestIds.push(requestId);
        requests.push({
          id: requestId,
          llm_run_id: runId,
          document_id: document.id,
          llm_model: model.id,
          repetition: repeat,
          execution_block: repeat,
          status: document.status,
          blockers: document.blockers,
        });
      }
      modelRuns.push({
        id: runId,
        document_id: document.id,
        llm_model: model.id,
        request_ids: requestIds,
        status: document.status,
        blockers: document.blockers,
      });
    }
  }

  let globalOrder = 0;
  const orderedRequests = [];
  for (let repeat = 1; repeat <= plan.axes.llm_repetitions; repeat += 1) {
    const block = requests.filter((request) => request.repetition === repeat);
    const randomized = shuffled(
      block,
      plan.execution_policy.llm_execution_order_seed + repeat - 1,
    );
    randomized.forEach((request, index) => {
      globalOrder += 1;
      orderedRequests.push({
        ...request,
        block_execution_order: index + 1,
        global_execution_order: globalOrder,
      });
    });
  }
  const orders = new Map(orderedRequests.map((request) => [
    request.id,
    request.global_execution_order,
  ]));
  return {
    modelRuns: modelRuns.map((run) => ({
      ...run,
      global_execution_orders: run.request_ids.map((id) => orders.get(id)),
    })),
    requests: orderedRequests,
  };
}

async function buildMatrix(planPath) {
  const plan = await readJson(planPath);
  const selection = await readJson(plan.source_selection.path);
  const knownPassageIds = passageIds(selection);
  const primaryAudioUnits = makePrimaryAudioUnits(plan, knownPassageIds);
  const controlAudioUnits = makeSameSeedControlAudioUnits(
    plan,
    knownPassageIds,
  );
  const audioUnits = [...primaryAudioUnits, ...controlAudioUnits];
  const sttUnits = makeSttUnits(plan, audioUnits);
  const primarySttUnits = sttUnits.filter((unit) => unit.kind === 'primary');
  const documents = makeDocuments(plan, primaryAudioUnits, primarySttUnits);
  const llm = makeLlmUnits(plan, documents);

  return {
    $schema: '../schema/postprocessing-factorial-cells.schema.json',
    schema_version: '1.0.0',
    id: `${plan.id}-cells`,
    plan_id: plan.id,
    plan_revision: plan.revision,
    source_selection_revision: selection.revision,
    generation: {
      generator: 'scripts/generate-postprocessing-factorial-matrix.mjs',
      llm_execution_order_seed: plan.execution_policy.llm_execution_order_seed,
      execution_blocks: plan.axes.llm_repetitions,
      deterministic: true,
    },
    counts: {
      primary_audio_units: primaryAudioUnits.length,
      same_seed_control_audio_units: controlAudioUnits.length,
      total_audio_units: audioUnits.length,
      primary_stt_units: primarySttUnits.length,
      same_seed_control_stt_units: sttUnits.length - primarySttUnits.length,
      total_stt_units: sttUnits.length,
      llm_documents: documents.length,
      llm_model_runs: llm.modelRuns.length,
      llm_requests: llm.requests.length,
    },
    audio_units: audioUnits,
    stt_units: sttUnits,
    llm_documents: documents,
    llm_model_runs: llm.modelRuns,
    llm_requests_in_execution_order: llm.requests,
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const matrix = await buildMatrix(options.planPath);
  const serialized = `${JSON.stringify(matrix, null, 2)}\n`;
  const outputPath = resolve(repoRoot, options.outputPath);

  if (options.mode === 'write') {
    await writeFile(outputPath, serialized);
    console.log(`Wrote ${options.outputPath}`);
    return;
  }
  if (options.mode === 'check') {
    const current = await readFile(outputPath, 'utf8');
    if (current !== serialized) {
      throw new Error(
        `${options.outputPath} is stale; regenerate with --write`,
      );
    }
    console.log(`Validated deterministic ${options.outputPath}`);
    return;
  }
  process.stdout.write(serialized);
}

await main();
