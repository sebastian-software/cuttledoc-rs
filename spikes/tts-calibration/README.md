# Current TTS calibration pins

This directory freezes the three July 2026 candidates selected before the
multi-voice synthetic roundtrip matrix:

- Qwen3-TTS 1.7B VoiceDesign BF16 as the required open multilingual generator;
- Voxtral 4B TTS BF16 as the required native-European-language control; and
- KugelAudio-0-Open BF16 as one bounded German challenger.

All three use the reviewed `Blaizzy/mlx-audio` 0.4.5 reference at commit
`64e8416c303fb3b3463dab8eb4ebd78c55a87c1a`. The manifests pin every Hub file
by byte count and SHA-256. LFS digests come from the immutable Hugging Face
object metadata; the fetcher verifies the downloaded bytes before accepting a
local snapshot.

Capture current metadata without downloading model weights:

```sh
node scripts/capture-huggingface-model-metadata.mjs \
  --repository OWNER/MODEL \
  --revision 40_CHARACTER_COMMIT
```

Materialize one accepted manifest:

```sh
bash scripts/fetch-tts-calibration-model.sh \
  spikes/tts-calibration/qwen3-tts-1.7b-voicedesign-bf16.json \
  /absolute/path/to/model
```

Run the first pinned Qwen VoiceDesign profile after materializing the passage
selection and model:

```sh
bash scripts/run-qwen3-tts-voicedesign-calibration.sh
```

The runner verifies every model file by byte count and SHA-256 before loading
it. Set `CUTTLEDOC_TTS_PROFILE` to select one of the 17 fixed Qwen profiles;
the default is `qwen-de-clear-documentary`.

Cross one completed local TTS result through the five required ASR backends:

```sh
node scripts/run-tts-calibration-asr.mjs \
  --result /absolute/path/to/result.json \
  --audio /absolute/path/to/audio.f32le \
  --reference /absolute/path/to/synthetic-de-origin.txt \
  --output /absolute/path/to/result-with-asr.json
```

The local default paths reproduce the reviewed host setup. Every backend path
has an explicit CLI override for another machine. Normalization starts at the
f32 synthesis master rather than the convenience PCM16 WAV, avoiding an
unnecessary quantization before the five receivers see the shared 16 kHz PCM.
Whisper, Qwen, and Apple receive the language or locale recorded by the TTS
run; the model-managed Parakeet and Voxtral paths are not forced to German.

After encoding the reviewed PCM16 WAV with
`scripts/encode-benchmark-opus.mjs`, archive a completed five-receiver result
with `scripts/archive-qwen-tts-calibration.mjs`. The archiver verifies the
selection, model/profile revisions, reference digest, and ASR completeness,
then writes the result record, asset manifest, exact reference, attribution,
and SPDX sidecars without overwriting an existing archive.

Both German Qwen profiles are checked in under `benchmarks/raw`. The
`qwen-de-clear-documentary` profile completed the passage but all five ASR
receivers failed at the spoken year `1962`. In contrast, all five receivers
recover `1962` from `qwen-de-warm-podcast`, with WER from 0.97% to 5.83%.
This localizes the failure to the designed voice/profile rather than the model
family. Retain the warm profile, reject the clear profile, perform listening
review, and use the warm profile for German.

The `qwen-en-warm-podcast` profile also passes the lexical gate. All five ASR
backends reproduce the exact normalized character sequence; its uniform 2.56%
WER is only the spoken expansion of the hyphenated `chains-of-thought` token.
The bounded German/English lexical calibration therefore accepts the warm
profiles. Two additional German profiles keep the same warm description while
changing the passage to native-factual prose and dialogue. The native-factual
cell passes all five receivers with complete critical-fact recovery and
1.92–2.88% WER. In the dialogue cell, four receivers make at most two edits;
Parakeet alone makes seven. The three content cells therefore pass the Qwen
lexical gate without letting embedded English terms dominate the comparison.
Listening remains the final Qwen gate before Voxtral and full-matrix
promotion.

The manifest now adds the same three cells for `en-US`, `es-419`, `fr-FR`,
and `pt-BR`. Each uses one fixed warm, conversational description within its
locale so the first comparison changes content rather than voice design.
Spanish and Portuguese are regional calibration proxies; their labels do not
turn a single synthesized voice or source edition into universal dialect
coverage.

The English native-factual cell passes: four receivers make one word edit,
Apple makes four, and every receiver recovers the complete passage and both
decades. The English dialogue does not pass. It repeats one earlier question
until the fixed 1,200-token limit and omits the final 32 reference words. Both
the positive and failed outputs are retained as digest-pinned Opus controls;
the remaining multilingual run must not average this generation failure away.

All three Spanish outputs finish normally. Whisper and Voxtral recover the two
factual passages nearly completely; the larger Parakeet, Qwen-ASR, and Apple
spreads are receiver-specific. Four receivers make one edit on the dialogue.
The native-factual phrase “a elegir” remains open for listening because two
otherwise strong receivers render it similarly as another word.

French yields one strong pass and two hard generation failures. The native-
factual cell is complete and reaches 0–4.72% WER. The technical cell silently
stops after three of five list items below the token limit; the dialogue stops
after “Jonas sourit” and repeats until the limit. These are retained as failed
controls rather than averaged into the successful French result.

The snapshots are large: Qwen is 4.52 GB, Voxtral is 8.04 GB, and KugelAudio
is 18.69 GB. Download only the candidate required by the current calibration
cell.

KugelAudio has a documented capability mismatch at the pinned runtime. Its Hub
card and `voices.json` name three presets whose `.pt` files are absent from the
snapshot, while the MLX implementation ignores the `voice` argument and uses
one implicit default voice. The challenger remains runnable without those
files, but it does not count as multi-voice coverage.
