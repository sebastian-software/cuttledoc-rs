# Redistributable benchmark assets

This directory contains benchmark inputs that are intentionally committed to
Git so the same audio can be used without regenerating it. It is not part of
the Cuttledoc software distribution:

- the repository source code remains MIT-licensed unless a file says otherwise;
- every asset directory carries its own attribution, provenance, and license;
- generated speech derived from CC BY-SA text is distributed under CC BY-SA
  4.0 as the conservative repository policy; and
- `.gitattributes` excludes this directory from `git archive`, and future crate
  and npm package manifests must continue to exclude it explicitly.

The canonical committed speech format is mono Ogg Opus at a 64 kbit/s VBR
target, encoded with the `audio` application, 20 ms frames, and 0% expected
packet loss. Opus decoders expose 48 kHz because that is the codec's fixed
internal clock; the lossless TTS source for the first fixture is mono 24 kHz.
The original lossless generation output stays outside Git and is identified by
SHA-256 in each asset manifest. The first Qwen 0.6B fixture is the codec
selection control. The
[`qwen-de-clear-documentary`](synthetic/de-DE/qwen3-tts-1.7b-voicedesign-clear/synthetic-de-origin/manifest.json)
fixture preserves a current-model `1962` pronunciation failure for replay; it
is decision evidence, not an accepted quality exemplar. The
[`qwen-de-warm-podcast`](synthetic/de-DE/qwen3-tts-1.7b-voicedesign-warm/synthetic-de-origin/manifest.json)
fixture is the paired positive control: all five receivers recover `1962`,
although listening review remains open. The English
[`qwen-en-warm-podcast`](synthetic/en-US/qwen3-tts-1.7b-voicedesign-warm/synthetic-en-reasoning/manifest.json)
fixture provides the cross-language positive control with exact normalized
character content from all five receivers. The native-German
[`synthetic-de-native`](synthetic/de-DE/qwen3-tts-1.7b-voicedesign-warm/synthetic-de-native/manifest.json)
fixture provides the first content-type control: all five receivers recover
the full factual passage and its critical facts. The repository-authored
[`synthetic-de-dialogue`](synthetic/de-DE/qwen3-tts-1.7b-voicedesign-warm/synthetic-de-dialogue/manifest.json)
fixture completes the three-cell content control; four receivers make no more
than two word edits, while Parakeet records seven receiver-specific edits.
The English
[`synthetic-en-native`](synthetic/en-US/qwen3-tts-1.7b-voicedesign-warm/synthetic-en-native/manifest.json)
fixture is a second positive content control. The English
[`synthetic-en-dialogue`](synthetic/en-US/qwen3-tts-1.7b-voicedesign-warm/synthetic-en-dialogue/manifest.json)
fixture is intentionally retained as a failed control: Qwen repeats one
question until the token limit and omits the final 32 reference words.
The three `es-419` Qwen fixtures add technical, native-factual, and dialogue
controls. All finish normally; their per-receiver WER spread is retained
without collapsing it into one language-wide score.

This encoding is for clean, professionally produced speech. It is not a claim
that 64 kbit/s is transparent for music, acoustic analysis, or speech synthesis
quality review. Use the lossless local artifact for those tasks.

Create or verify the first artifact with the pinned command contract:

```sh
node scripts/encode-benchmark-opus.mjs \
  --input /absolute/path/to/audio.pcm16.wav \
  --output benchmarks/assets/synthetic/de-DE/qwen3-tts-0.6b-ryan/synthetic-de-origin/audio.opus

node scripts/encode-benchmark-opus.mjs \
  --input /absolute/path/to/audio.pcm16.wav \
  --output benchmarks/assets/synthetic/de-DE/qwen3-tts-0.6b-ryan/synthetic-de-origin/audio.opus \
  --check
```

Use the same command with the paths recorded in the VoiceDesign asset manifests
to reproduce or check the current-model fixtures.

See [LICENSE.md](LICENSE.md) for the directory-level licensing rule. The
adjacent `.license` sidecars are the machine-readable SPDX declarations for
files that cannot carry an inline header.
