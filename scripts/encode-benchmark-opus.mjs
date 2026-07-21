#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    input: { type: 'string' },
    output: { type: 'string' },
    check: { type: 'boolean', default: false },
    ffmpeg: { type: 'string', default: process.env.FFMPEG ?? 'ffmpeg' },
    ffprobe: { type: 'string', default: process.env.FFPROBE ?? 'ffprobe' },
  },
});

if (!values.input || !values.output) {
  throw new Error(
    'usage: node scripts/encode-benchmark-opus.mjs --input LOSSLESS.wav --output AUDIO.opus [--check]',
  );
}

const inputPath = resolve(values.input);
const outputPath = resolve(values.output);
const tempDirectory = await mkdtemp(join(tmpdir(), 'cuttledoc-opus-'));
const firstPath = values.check
  ? join(tempDirectory, 'candidate.opus')
  : outputPath;
const repeatPath = join(tempDirectory, 'repeat.opus');

try {
  if (!values.check) await mkdir(dirname(outputPath), { recursive: true });
  encode(firstPath);
  encode(repeatPath);

  const first = await readFile(firstPath);
  const repeat = await readFile(repeatPath);
  if (!first.equals(repeat)) {
    throw new Error('repeated encoding was not byte-for-byte deterministic');
  }
  if (values.check) {
    const expected = await readFile(outputPath);
    if (!first.equals(expected)) {
      throw new Error(`${outputPath} differs from a fresh deterministic encode`);
    }
  }

  const probe = run(values.ffprobe, [
    '-v',
    'error',
    '-show_entries',
    'format=duration,size,bit_rate:stream=codec_name,sample_rate,channels',
    '-of',
    'json',
    firstPath,
  ]);
  const metadata = JSON.parse(probe.stdout);
  const stream = metadata.streams?.[0];
  if (stream?.codec_name !== 'opus' || stream.channels !== 1) {
    throw new Error('encoded artifact must be mono Opus');
  }

  const ffmpegVersion = run(values.ffmpeg, ['-version']).stdout.split('\n')[0];
  process.stdout.write(`${JSON.stringify({
    input: inputPath,
    output: outputPath,
    check: values.check,
    deterministic_repeat: true,
    sha256: digest(first),
    bytes: (await stat(firstPath)).size,
    duration_ms: Number.parseFloat(metadata.format.duration) * 1_000,
    container_bit_rate_bps: Number.parseInt(metadata.format.bit_rate, 10),
    codec: stream.codec_name,
    decoded_sample_rate_hz: Number.parseInt(stream.sample_rate, 10),
    channel_count: stream.channels,
    ffmpeg: ffmpegVersion,
    encoder: {
      container: 'Ogg',
      codec: 'libopus',
      application: 'audio',
      input_sample_rate_hz: 24_000,
      target_bit_rate_bps: 64_000,
      variable_bit_rate: true,
      frame_duration_ms: 20,
      expected_packet_loss_percent: 0,
      compression_level: 10,
      metadata: 'stripped',
      bitexact_flags: true,
    },
  }, null, 2)}\n`);
} finally {
  await rm(tempDirectory, { recursive: true, force: true });
}

function encode(path) {
  run(values.ffmpeg, [
    '-y',
    '-v',
    'error',
    '-i',
    inputPath,
    '-map_metadata',
    '-1',
    '-ac',
    '1',
    '-ar',
    '24000',
    '-c:a',
    'libopus',
    '-application',
    'audio',
    '-b:a',
    '64k',
    '-vbr',
    'on',
    '-frame_duration',
    '20',
    '-packet_loss',
    '0',
    '-compression_level',
    '10',
    '-fflags',
    '+bitexact',
    '-flags:a',
    '+bitexact',
    path,
  ]);
}

function run(command, args) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} failed: ${result.stderr || result.error?.message || `exit ${result.status}`}`,
    );
  }
  return result;
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}
