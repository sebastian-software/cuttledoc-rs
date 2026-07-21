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
SHA-256 in the asset manifest.

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

See [LICENSE.md](LICENSE.md) for the directory-level licensing rule. The
adjacent `.license` sidecars are the machine-readable SPDX declarations for
files that cannot carry an inline header.
