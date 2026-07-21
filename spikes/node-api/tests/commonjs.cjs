'use strict';

const assert = require('node:assert/strict');
const { createContractStream, processPcm } = require('cuttledoc-node-spike');

(async () => {
  const updates = [];
  for await (const update of createContractStream('volatile_to_final')) {
    updates.push(update);
  }

  assert.deepEqual(updates.map(({ sequence }) => sequence), [1, 2, 3]);
  assert.equal(updates.at(-1).stability, 'final');
  assert.equal(updates.at(-1).text, 'hello_world');

  const revoke = [];
  for await (const update of createContractStream('revoke_volatile')) {
    revoke.push(update);
  }
  assert.deepEqual(revoke.map(({ operation }) => operation), ['replace', 'revoke']);

  const cancelled = createContractStream('finals_only');
  assert.equal((await cancelled.next()).done, false);
  assert.equal((await cancelled.return()).done, true);
  assert.equal(cancelled.closed, true);
  assert.equal((await cancelled.next()).done, true);

  const progress = [];
  assert.deepEqual(
    await processPcm(Uint8Array.from([4, 3, 2, 1]), {
      onProgress(value) {
        progress.push(value);
      },
    }),
    { byteLength: 4, checksum: 10, progressSteps: 4 },
  );
  assert.deepEqual(progress, [25, 50, 75, 100]);

  await assert.rejects(
    processPcm(Uint8Array.of(1), { fail: true }),
    /simulated Rust transcription failure/,
  );

  console.log('CommonJS stream contract: ok');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
