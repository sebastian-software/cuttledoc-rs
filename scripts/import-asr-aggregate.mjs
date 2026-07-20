#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    input: { type: 'string' },
    output: { type: 'string' },
  },
});
if (!values.input || !values.output) {
  throw new Error(
    'usage: node scripts/import-asr-aggregate.mjs ' +
    '--input /tmp/result.json --output benchmarks/raw/<run>/result.json',
  );
}

const inputPath = resolve(values.input);
const outputPath = resolve(values.output);
const bytes = await readFile(inputPath);
const aggregate = JSON.parse(bytes.toString('utf8'));
const head = commandOutput('git', ['rev-parse', 'HEAD']).trim();

if (aggregate.schema_version !== '1.0.0') {
  throw new Error(`${inputPath}: unsupported schema version`);
}
if (!aggregate.matrix_run_id?.length || !aggregate.candidate?.id?.length) {
  throw new Error(`${inputPath}: incomplete aggregate identity`);
}
if (aggregate.source_revision !== head) {
  throw new Error(
    `${inputPath}: source revision ${aggregate.source_revision} ` +
    `does not equal current HEAD ${head}`,
  );
}
if (!Array.isArray(aggregate.results) ||
    aggregate.results.length !== aggregate.procedure?.fixture_count) {
  throw new Error(`${inputPath}: result count differs from procedure`);
}
if (!values.output.endsWith('/result.json')) {
  throw new Error('--output must end in /result.json');
}

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, bytes);
process.stdout.write(
  `imported ${aggregate.matrix_run_id} (${aggregate.candidate.id}) to ` +
  `${outputPath}\n`,
);

function commandOutput(command, arguments_) {
  const result = spawnSync(command, arguments_, {
    encoding: 'utf8',
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} failed with ${result.status}\n` +
      `${result.stdout}\n${result.stderr}`,
    );
  }
  return result.stdout;
}
