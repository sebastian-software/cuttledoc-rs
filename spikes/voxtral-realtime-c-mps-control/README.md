# Voxtral pure-C/Metal-MPS comparison control

This directory records the deliberately bounded comparison requested by
issue #19. It does not introduce `antirez/voxtral.c` as a dependency and does
not repeat the multilingual quality evaluation already completed through the
repository-owned official-MLX adapter.

The control pins `antirez/voxtral.c` at
`134d366c24d20c64b614a3dcc8bda2a6922d077d`, builds its MPS target, verifies
the publisher's BF16 model artifacts, measures a fresh-process load/free
lifecycle, and transcribes the same development-exposed German fixture used
for the official-MLX implementation parity checks.

## Reproduction

Clone the source control outside the repository and detach it at the recorded
revision:

```sh
git clone https://github.com/antirez/voxtral.c.git \
  /private/tmp/cuttledoc-voxtral-c-134d366
git -C /private/tmp/cuttledoc-voxtral-c-134d366 checkout --detach \
  134d366c24d20c64b614a3dcc8bda2a6922d077d
```

Then run:

```sh
bash scripts/run-voxtral-c-mps-control.sh
```

The fetcher downloads 8,874,374,435 verified model bytes. Set
`CUTTLEDOC_VOXTRAL_C_SOURCE_DIR`, `CUTTLEDOC_VOXTRAL_C_MODEL_DIR`,
`CUTTLEDOC_VOXTRAL_PCM_FIXTURE`, or `CUTTLEDOC_VOXTRAL_C_BUILD_DIR` to use
different local locations without changing any revision or digest.

## Boundary finding

The MPS target has an exceptionally small executable and no non-system shared
library dependency. Its synchronous `feed()` call, however, performs all
available encoder and decoder work inline. The public API has no cancellation,
busy/reentrancy, queue-capacity, or backpressure operation, and the source has
no synchronization protecting a model context or stream. Cuttledoc would need
to own and maintain those semantics in a fork or an additional adapter.

The control is therefore useful implementation evidence, but it fails the
ADR-0005 production dependency gate at the pinned revision. The official MLX
core plus the repository-owned model adapter remains the accepted Voxtral
boundary.
