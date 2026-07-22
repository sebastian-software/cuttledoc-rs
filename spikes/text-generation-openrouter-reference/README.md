# Text-generation OpenRouter quality reference

This reference path screens larger hosted models before Cuttledoc invests in an
embedded MLX or Core ML port. It complements, but does not replace, the pinned
local MLX reference under `spikes/text-generation-mlx-reference/`.

The initial historical matrix held the German development fixture and
conservative prompt constant while varying four model/provider tuples:

| Role | Model | Pinned provider |
| --- | --- | --- |
| Large open-weight candidate | Qwen 3.5 122B-A10B | AtlasCloud FP8 |
| Multilingual open-weight candidate | Mistral Small 3.2 24B | Parasail BF16 |
| Frontier ceiling | GPT-5.6 Sol | Azure EU |
| Second frontier-family ceiling | Claude Sonnet 4.6 | Bedrock EU West 1 |

The 2026-07-22 development screen produced:

| Candidate | Mechanical result | Repeat | Two-request cost |
| --- | --- | --- | ---: |
| Qwen 3.5 122B-A10B | Accepted; corrected `schautete` to `schauderte` | Different | $0.0040482 |
| Mistral Small 3.2 24B | Rejected; reported `schaute` but did not apply it | Identical | $0.00017774 |
| GPT-5.6 Sol | Accepted; corrected `schautete` to `schauderte` | Different | $0.0346610 |
| Claude Sonnet 4.6 | Accepted; made only the target correction | Identical | $0.0124542 |

These are contract and capability observations on one inspected development
fixture, not a quality ranking. The three accepted outputs match its unverified
dataset reference, but only independent human gold can label edits beneficial
and estimate regression rates.

The multi-page quality screen adds current frontier controls without deleting
the reproducible historical records:

| Role | Model | Pinned provider | Routing status |
| --- | --- | --- | --- |
| Fast frontier candidate | Gemini 3.6 Flash | Google Vertex Global | ZDR; explicit low reasoning required |
| Frontier candidate | Kimi K3 | Moonshot AI INT4 | ZDR |
| GPT family control | GPT-5.6 Luna | Azure EU | ZDR |
| Current Anthropic control | Claude Sonnet 5 | Azure US East 2 | ZDR; replaces Sonnet 4.6 as the active control |
| Frontier candidate | Qwen3.7 Max | Alibaba | Blocked: no cataloged ZDR route; explicit privacy exception required |

## Multi-page follow-up

The quality follow-up uses eight sections, eight German Apple voices, 995
normalized reference words, 506.7 seconds of clean synthesized speech, and 60
raw Whisper word errors. It therefore supersedes the one-word fixture for
development-quality reasoning while retaining the earlier probe as a fast
contract smoke test.

| Candidate | Contract | Micro-WER | Improved / unchanged / regressed sections | Two-request cost |
| --- | --- | ---: | --- | ---: |
| Raw Whisper | n/a | 6.03% | n/a | n/a |
| Qwen 3.5 122B-A10B | Passed; unsupported changes to `5.0` and `2025` | 2.51% | 6 / 1 / 1 | $0.0254754 |
| GPT-5.6 Sol | Failed; one output change missing from its edit ledger | 0.50% | 6 / 2 / 0 | $0.2455266 |
| Claude Sonnet 4.6 (historical) | Passed | 3.12% | 5 / 3 / 0 | $0.1521399 |
| Gemini 3.6 Flash | Failed; one output change missing from its edit ledger, one regressed section | 0.90% | 6 / 1 / 1 | $0.0716760 |
| Kimi K3 | Passed | 2.31% | 6 / 2 / 0 | $0.1387416 |
| GPT-5.6 Luna | Failed; all six output changes missing from its edit ledger | 5.43% | 1 / 7 / 0 | $0.0294822 |
| Claude Sonnet 5 | Passed | 2.81% | 5 / 3 / 0 | $0.1449660 |
| Mistral Small 3.2 24B | Blocked twice by pinned Parasail endpoint HTTP 429 | — | — | not recorded |
| Qwen3.7 Max | Blocked because the only cataloged endpoint was not ZDR | — | — | not run |

The full methodology, per-error interpretation, and decision boundary are in
[`postprocessing-long-form-evaluation.md`](../../docs/postprocessing-long-form-evaluation.md).

Every request disables provider fallback, requires all requested parameters,
denies data collection, and requires a zero-data-retention route. The gateway
also enforces a strict JSON Schema. This is a pragmatic hosted-product
capability, so these runs are not directly identical to the local prompt-only
MLX structured-output attempts. The repository still parses the JSON, derives
the lexical diff independently, verifies that every lexical edit was reported,
and checks protected spans.

Qwen3.7 Max is deliberately not an implicit exception to that policy. Its
blocker record captures the 2026-07-22 catalog state. A one-time run may be
added only after explicit approval and must remain pinned to the public
synthetic fixture, deny provider data collection, and disable fallback.

The fixture's evaluation reference is never sent to the model. Its diagnostic
WER remains development-only because the dataset transcript is unverified. A
mechanically accepted result is not a selected quality winner.

Put the key in the ignored repository-root `.env.local` file:

```text
OPENROUTER_API_KEY=...
```

Run one pinned experiment from the repository root:

```sh
revision=$(git rev-parse HEAD)

node --env-file=.env.local \
  spikes/text-generation-openrouter-reference/run_reference.mjs \
  --manifest spikes/text-generation-openrouter-reference/candidates/qwen3.5-122b-a10b-atlas-cloud.json \
  --experiment spikes/text-generation-openrouter-reference/experiments/qwen3.5-122b-a10b.conservative-error-profile-v1.de-audiobook.json \
  --fixture benchmarks/postprocessing/fixtures/issue20-de-audiobook-whisper-conservative.json \
  --prompt benchmarks/postprocessing/prompts/conservative-error-profile-v1.txt \
  --output /tmp/qwen-openrouter-result.json \
  --source-revision "$revision"
```

Each run performs two complete requests. Equality is recorded only as an
observation: remote services cannot provide the artifact and execution
determinism required from the local MLX reference. Client-observed duration is
also recorded with an explicit prohibition on comparing it to local model
latency. Actual token use and OpenRouter-reported cost are retained.

Validate candidates, experiments, and any checked-in results without making a
network request:

```sh
node scripts/validate-text-generation-openrouter-reference.mjs --self-test
```

Neither the key nor authorization headers may appear in a manifest or result;
the validator rejects common secret fields and OpenRouter key prefixes.
