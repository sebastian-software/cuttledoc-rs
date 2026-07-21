import assert from 'node:assert/strict';
import { createContractStream, processPcm } from 'cuttledoc-node-spike';

const updates = [];
for await (const update of createContractStream('volatile_to_final')) {
  updates.push(update);
}

assert.deepEqual(updates.map(({ sequence }) => sequence), [1, 2, 3]);
assert.equal(updates.at(-1).stability, 'final');
assert.equal(updates.at(-1).text, 'hello_world');
assert.deepEqual(updates.at(-1), {
  sequence: 3,
  operation: 'replace',
  stability: 'final',
  affectedStartMs: 0,
  affectedEndMs: 1000,
  segmentEndMs: 1000,
  text: 'hello_world',
});

const revoke = [];
for await (const update of createContractStream('revoke_volatile')) {
  revoke.push(update);
}
assert.deepEqual(
  revoke.map(({ operation, affectedStartMs, affectedEndMs }) => ({
    operation,
    affectedStartMs,
    affectedEndMs,
  })),
  [
    { operation: 'replace', affectedStartMs: 0, affectedEndMs: 1000 },
    { operation: 'revoke', affectedStartMs: 0, affectedEndMs: 1000 },
  ],
);

const cancelled = createContractStream('finals_only');
assert.equal((await cancelled.next()).done, false);
assert.equal((await cancelled.return()).done, true);
assert.equal(cancelled.closed, true);
assert.equal((await cancelled.next()).done, true);

const pcm = Uint8Array.from([1, 2, 3, 4]);
const progress = [];
let eventLoopAdvanced = false;
const work = processPcm(pcm, {
  onProgress(value) {
    progress.push(value);
  },
});
pcm.fill(255);
setImmediate(() => {
  eventLoopAdvanced = true;
});
assert.deepEqual(await work, { byteLength: 4, checksum: 10, progressSteps: 4 });
assert.deepEqual(progress, [25, 50, 75, 100]);
assert.equal(eventLoopAdvanced, true);

await assert.rejects(
  processPcm(Uint8Array.of(1), { fail: true }),
  /simulated Rust transcription failure/,
);

const abortController = new AbortController();
const abortingWork = processPcm(new Uint8Array(1024), {
  signal: abortController.signal,
});
setTimeout(() => abortController.abort(), 6);
await assert.rejects(abortingWork, /abort|cancel/i);

console.log('ESM stream contract: ok');
