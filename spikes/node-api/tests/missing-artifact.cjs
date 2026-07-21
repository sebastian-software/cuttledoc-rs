'use strict';

const assert = require('node:assert/strict');

assert.throws(
  () => require('cuttledoc-node-spike'),
  /could not load its darwin-arm64 prebuilt artifact/,
);

console.log('Missing artifact diagnostic: ok');
