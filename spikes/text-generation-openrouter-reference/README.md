# Text-generation OpenRouter quality reference

This reference path screens larger hosted models before Cuttledoc invests in an
embedded MLX or Core ML port. It complements, but does not replace, the pinned
local MLX reference under `spikes/text-generation-mlx-reference/`.

The first matrix holds the German development fixture and conservative prompt
constant while varying four model/provider tuples:

| Role | Model | Pinned provider |
| --- | --- | --- |
| Large open-weight candidate | Qwen 3.5 122B-A10B | Alibaba |
| Multilingual open-weight candidate | Mistral Small 3.2 24B | Mistral |
| Frontier ceiling | GPT-5.6 Sol | OpenAI |
| Second frontier-family ceiling | Claude Sonnet 4.6 | Anthropic |

Every request disables provider fallback, requires all requested parameters,
denies data collection, and requires a zero-data-retention route. The gateway
also enforces a strict JSON Schema. This is a pragmatic hosted-product
capability, so these runs are not directly identical to the local prompt-only
MLX structured-output attempts. The repository still parses the JSON, derives
the lexical diff independently, verifies that every lexical edit was reported,
and checks protected spans.

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
  --manifest spikes/text-generation-openrouter-reference/candidates/qwen3.5-122b-a10b-alibaba.json \
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
