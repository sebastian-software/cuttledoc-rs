# Transcript postprocessing evaluation

**Status:** historical evidence imported; model-first Cuttledoc 3 bakeoff in
progress under issue #20.

**Evidence date:** 2026-07-22.

The Cuttledoc 2 correction path used a constrained transcript-proofreading
prompt and local Ollama models. The CLI default was **Gemma 3n E4B**
(`gemma3n:e4b`), not “Gamer.” The prompt allowed word-boundary, obvious
mishearing, grammar, punctuation, and capitalization fixes while forbidding
translation, summarization, rephrasing, and added content.

The historical data supports re-evaluating this stage. It does not support
enabling it unconditionally.

## What the historical runs show

The immutable aggregate import is
[`cuttledoc-v2-snapshot.json`](../benchmarks/postprocessing/cuttledoc-v2-snapshot.json).
The original result JSON files are local ignored artifacts in the sibling
Cuttledoc 2 checkout, so the snapshot records their SHA-256 digests and the
tracked benchmark-script revisions rather than claiming a producing Git
commit.

### TTS-generated input

The favorable experiment used two ElevenLabs-generated samples per language
and Parakeet raw transcripts.

| Model | Samples | WER before | WER after | Individual regressions |
| --- | ---: | ---: | ---: | ---: |
| `phi4:14b` | 10 | 5.63% | 2.80% | 0 |
| `mistral-nemo` | 10 | 5.63% | 3.23% | 0 |
| `gemma3n:e4b` | 10 | 5.63% | 3.33% | 0 |
| `gemma3n:e2b` | 10 | 5.63% | 3.58% | 1 |
| `qwen3:8b` | 10 | 5.63% | 11.67% | 3 |

Gemma 3n E4B is a credible candidate because it improved all ten of these
samples. The set is synthetic and contains only two samples per language,
however. It cannot establish podcast or audiobook safety.

### Real ASR output

The second experiment used ten FLEURS/Parakeet outputs per language. Its
checked-in script names Gemma 3n E4B, but the ignored result artifact contains
no completed Gemma rows.

| Model | Samples | WER before | WER after | Individual regressions |
| --- | ---: | ---: | ---: | ---: |
| `gpt-oss:20b` | 48 | 6.80% | 6.72% | 6 |
| `phi4:14b` | 50 | 6.78% | 14.59% | 21 |
| `qwen2.5:7b` | 50 | 6.78% | 22.02% | 23 |
| `mistral:7b` | 50 | 6.78% | 50.86% | 41 |

Even the nearly neutral `gpt-oss:20b` aggregate hides a strong language split:
German improved from 4.87% to 2.49%, while English worsened from 5.45% to
8.30% and Portuguese from 6.95% to 7.41%. Aggregate-only enablement would be
unsafe.

These real-run percentages are also not directly comparable to the new
Cuttledoc 3 WER. The old JavaScript normalizer used `[^\w\s]`, which treats
word characters as ASCII and removes accented letters before alignment. The
TTS script preserved only a bounded Latin accent range. Cuttledoc 3 uses
Unicode letters and must recompute every score from the retained raw text.

## Cuttledoc 3 decision

Postprocessing remains a first-class experiment under the text-generation
runtime work. Gemma 4 E2B is the initial embedded-size quality candidate, Qwen
3.5 0.8B is the small efficiency challenger, and SmolLM3 3B is the multilingual
product-size compromise. A hosted Qwen 3.5 122B-A10B candidate plus GPT and
Claude frontier controls now bound the quality that substantially stronger
models can provide. Gemma 3n E4B remains a historical Ollama control rather
than defining the new embedded runtime. Enhancement is a separate, optional
stage:

1. preserve the raw ASR transcript;
2. correct a copy with the exact model, quantization, prompt, and decoding
   settings pinned;
3. score raw and corrected text per language and domain with the same
   Unicode-normalized content metric;
4. score punctuation, sentence boundaries, and capitalization separately;
5. review every changed lexical token for names, numbers, units, dates,
   negation, terminology, omission, and hallucination; and
6. reject a candidate that introduces any critical semantic regression, even
   when aggregate WER improves.

Issue #20 first executes all three candidates through one pinned reference
layer on development data, then uses raw outputs from Apple, Whisper, Qwen,
Parakeet, and Voxtral on the audiobook and professional-podcast gold sets for
selection. Only the quality survivor proceeds to an official-MLX versus Core ML
product-runtime comparison. Synthetic TTS remains a smoke/control condition,
not the acceptance population. See
[`transcript-enhancement-model-bakeoff.md`](transcript-enhancement-model-bakeoff.md).

The first development matrix is now complete. Gemma produced no lexical change
under the surface and historical prompts but violated the structured output
contract by adding Markdown fences. Qwen made unsupported lexical rewrites
under both permissive prompts. SmolLM3 passed the structured contract with a
no-op but made one harmful edit under the surface-only prompt. These runs prove
the candidates and external gates execute; the single unverified fixture cannot
select a model. The selection gate remains human-verified, source-grouped German
podcast and audiobook data.

The hosted capability screen then ran the same hidden-reference conservative
fixture through provider-pinned, no-fallback ZDR routes. Qwen 3.5 122B-A10B,
GPT-5.6 Sol, and Claude Sonnet 4.6 each made the single reference-matching
lexical correction and passed the external audit. Mistral Small 3.2 24B
reported an edit it did not apply and was rejected. This is evidence that model
capability matters and that the conservative prompt can produce a useful edit;
it is not a quality ranking or a reason to ship a hosted provider. The strict
gateway schema also makes this a hosted product-capability result rather than a
directly identical replay of the prompt-only local MLX attempts.

The subsequent long-form screen retains those records as historical controls
and adds Gemini 3.6 Flash, Kimi K3, GPT-5.6 Luna, and Claude Sonnet 5. Sonnet 5,
not 4.6, is the active Anthropic comparison. Kimi and Sonnet 5 pass the complete
contract without a section regression; Gemini is more accurate but makes one
unsupported factual correction, and Luna barely improves the transcript.
Qwen3.7 Max reaches 1.51% without a section regression under an explicitly
authorized, consumed one-execution non-ZDR exception; its ledger omits four
repeated correction occurrences.

Prompt behavior is evaluated independently from model identity. The versioned
historical, surface-only, error-profile, and targeted-span candidates plus
their source-grouped development/validation/test discipline are defined in
[`postprocessing-prompt-evaluation.md`](postprocessing-prompt-evaluation.md).
