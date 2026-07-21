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
it. Set `CUTTLEDOC_TTS_PROFILE` to select one of the four fixed Qwen profiles;
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

The snapshots are large: Qwen is 4.52 GB, Voxtral is 8.04 GB, and KugelAudio
is 18.69 GB. Download only the candidate required by the current calibration
cell.

KugelAudio has a documented capability mismatch at the pinned runtime. Its Hub
card and `voices.json` name three presets whose `.pt` files are absent from the
snapshot, while the MLX implementation ignores the `voice` argument and uses
one implicit default voice. The challenger remains runnable without those
files, but it does not count as multi-voice coverage.
