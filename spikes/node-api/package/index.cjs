'use strict';

if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  throw new Error(
    `cuttledoc-node-spike only contains a darwin-arm64 artifact, received ${process.platform}-${process.arch}`,
  );
}

const { NativeContractStream } = require('./cuttledoc-node-spike.darwin-arm64.node');

class TranscriptionStream {
  #native;

  constructor(caseName) {
    this.#native = new NativeContractStream(caseName);
  }

  async next() {
    const value = this.#native.nextUpdate();
    if (value === null || value === undefined) {
      this.#native.close();
      return { done: true, value: undefined };
    }
    return { done: false, value };
  }

  async return() {
    this.#native.close();
    return { done: true, value: undefined };
  }

  get closed() {
    return this.#native.isClosed();
  }

  [Symbol.asyncIterator]() {
    return this;
  }
}

function createContractStream(caseName) {
  return new TranscriptionStream(caseName);
}

module.exports = { TranscriptionStream, createContractStream };
