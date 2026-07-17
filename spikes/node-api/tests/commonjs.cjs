'use strict';

const assert = require('node:assert/strict');
const { createContractStream } = require('cuttledoc-node-spike');

(async () => {
  const updates = [];
  for await (const update of createContractStream('volatile_to_final')) {
    updates.push(update);
  }

  assert.deepEqual(updates.map(({ sequence }) => sequence), [1, 2, 3]);
  assert.equal(updates.at(-1).stability, 'final');
  assert.equal(updates.at(-1).text, 'hello_world');

  const cancelled = createContractStream('finals_only');
  assert.equal((await cancelled.next()).done, false);
  assert.equal((await cancelled.return()).done, true);
  assert.equal(cancelled.closed, true);
  assert.equal((await cancelled.next()).done, true);

  console.log('CommonJS stream contract: ok');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
