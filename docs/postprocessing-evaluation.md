# Transcript postprocessing evaluation

**Status:** historical evidence imported; Cuttledoc 3 acceptance benchmark
pending.

**Evidence date:** 2026-07-20.

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

Postprocessing remains a first-class experiment under the local
text-generation runtime work, with Gemma 3n E4B as the initial historical
candidate. It is a separate, optional stage:

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

The first meaningful rerun should use raw outputs from Apple, Whisper, and
Qwen on the audiobook and professional-podcast gold sets. Synthetic TTS remains
a smoke/control condition, not the acceptance population.

Prompt behavior is evaluated independently from model identity. The versioned
historical, surface-only, error-profile, and targeted-span candidates plus
their source-grouped development/validation/test discipline are defined in
[`postprocessing-prompt-evaluation.md`](postprocessing-prompt-evaluation.md).
