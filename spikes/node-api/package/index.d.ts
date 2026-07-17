export type Stability = 'volatile' | 'final';

export interface TranscriptionUpdate {
  sequence: number;
  operation: 'replace' | 'revoke';
  stability?: Stability;
  affectedStartMs: number;
  affectedEndMs: number;
  segmentStartMs?: number;
  segmentEndMs?: number;
  text?: string;
}

export declare class TranscriptionStream
  implements AsyncIterableIterator<TranscriptionUpdate>
{
  constructor(caseName: string);
  readonly closed: boolean;
  next(): Promise<IteratorResult<TranscriptionUpdate>>;
  return(): Promise<IteratorResult<TranscriptionUpdate>>;
  [Symbol.asyncIterator](): AsyncIterableIterator<TranscriptionUpdate>;
}

export declare function createContractStream(caseName: string): TranscriptionStream;
