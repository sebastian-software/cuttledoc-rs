'use strict';

if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  throw new Error(
    `cuttledoc-node-spike only contains a darwin-arm64 artifact, received ${process.platform}-${process.arch}`,
  );
}

let native;
try {
  native = require('./cuttledoc-node-spike.darwin-arm64.node');
} catch (cause) {
  throw new Error(
    'cuttledoc-node-spike could not load its darwin-arm64 prebuilt artifact; ' +
      'reinstall the package for this platform instead of rebuilding it locally',
    { cause },
  );
}

const { NativeContractStream, processPcm: nativeProcessPcm } = native;

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

function processPcm(pcm, options = {}) {
  if (!(pcm instanceof Uint8Array)) {
    throw new TypeError('pcm must be a Uint8Array');
  }
  if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) {
    throw new TypeError('signal must be an AbortSignal');
  }
  if (options.onProgress !== undefined && typeof options.onProgress !== 'function') {
    throw new TypeError('onProgress must be a function');
  }

  const ownedPcm = Buffer.from(pcm);
  const onProgress = options.onProgress ?? (() => {});
  return nativeProcessPcm(
    ownedPcm,
    options.fail === true,
    (error, progress) => {
      if (error) {
        throw error;
      }
      onProgress(progress);
    },
    options.signal,
  );
}

module.exports = { TranscriptionStream, createContractStream, processPcm };
