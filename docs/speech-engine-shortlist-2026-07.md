# Speech engine shortlist — July 2026

**Status:** accepted calibration shortlist; every new runtime and model still
requires an immutable revision and artifact manifest before execution.

**Review date:** 2026-07-21.

## Decision

Do not start the full multi-voice matrix from the historical candidate list.
Run a small calibration gate first, then expand only the useful candidates to
all selected passages.

The primary workload is clean produced speech. German is evaluated first and
English remains a separate control. A default recommendation is decision
support, not a universal winner; every accepted ASR backend remains selectable.

## TTS generator shortlist

| Candidate | Disposition | Calibration role |
| --- | --- | --- |
| Apple `AVSpeechSynthesizer` | required | Two installed voices per locale provide the system baseline. Record voice identifier, quality, locale, host OS, and generated-audio digest because installed inventories vary by host. |
| Qwen3-TTS 1.7B VoiceDesign | required | Two description-pinned voices per locale provide an open multilingual family without third-party reference audio. The BF16 conversion and MLX runtime are now immutable and digest-pinned. |
| Voxtral 4B TTS BF16 | required, reference-only | Two native-locale presets per locale provide the strongest European-language hypothesis. Use BF16 to avoid treating the existing 4-bit result as a model-family verdict. The CC BY-NC 4.0 weights and voices keep artifacts local and prohibit production adoption without another license. |
| KugelAudio-0-Open | German challenger | Run its implicit German default voice on one passage first. It is German/European-focused and MIT-licensed, but the pinned snapshot lacks the advertised preset files and the current MLX path ignores voice selection. Its roughly 19 GB footprint does not justify full-matrix cost before calibration. |
| Chatterbox Multilingual V3 | deferred | Current upstream V3 supports German and 23+ languages, but multi-voice use depends on reference audio. Voice rights, prompt digests, and the current Apple-Silicon conversion must be resolved first. |
| Qwen-Audio-3.0-TTS-Plus API | optional English ceiling | Hosted, mutable, credentialed, and not German evidence. It may remain an English listening control but does not block the local matrix. |

The existing Qwen3-TTS 0.6B CustomVoice and Voxtral 4-bit results remain useful
preflight evidence. They do not define the final generator variants: Qwen's
published preset list has no native German speaker, while the first quantized
Voxtral run had materially weaker lexical fidelity. Qwen VoiceDesign and
Voxtral BF16 directly address those confounders.

Kokoro is excluded from the German-first comparison because its published
language set does not include German. Piper is excluded because the original
project is archived and individual voice licensing remains source-specific.
Large voice-cloning families are deferred unless a later decision specifically
needs their behavior.

## ASR receiver shortlist

The full matrix uses five already justified product candidates:

1. Apple SpeechTranscriber — zero bundled model and first-party system path.
2. Whisper large-v3-turbo through the existing CoreML/whisper.cpp path —
   mature multilingual robustness baseline.
3. Direct Qwen3-ASR 0.6B over official MLX — compact generative candidate with
   a repository-owned boundary.
4. Parakeet TDT 0.6B v3/CoreML — compact transducer with 25 European languages,
   timestamps, punctuation, and capitalization.
5. Direct Voxtral Mini 4B Realtime over official MLX — European-language and
   true-streaming candidate whose repository adapter now covers complete-buffer
   and live input. Use the 2,400 ms quality configuration for the clean
   long-form matrix and keep 480 ms as the latency control.

Two current models receive only a bounded qualification run:

- Qwen3-ASR 1.7B through a pinned MLX reference, to measure whether its
  publisher-reported quality gain justifies the larger product footprint; and
- Nemotron 3.5 ASR Streaming 0.6B through a pinned reference, because it is a
  new cache-aware 40-locale competitor to Parakeet. Its Open Model Data and
  Weights License and production boundary require review before promotion.

Canary 1B v2 is deferred: its translation capabilities do not serve the
current transcription workload, and Parakeet already represents the Granary/
NVIDIA offline family at lower model size.

## Calibration sequence

1. **Complete:** exact runtime/model revisions and artifact digests are pinned
   for Qwen VoiceDesign, Voxtral BF16, and KugelAudio without changing the
   historical measured pins.
2. **Complete for the first lexical calibration:** both original German Qwen
   VoiceDesign profiles and the warm English profile are materialized and
   measured. Retain the warm profiles: every receiver recovers German `1962`,
   and all five reproduce the exact normalized English character content.
   Reject the clear German profile.
3. **In progress:** the German text selection now separates code-switch,
   native-factual, and dialogue cells. The native-factual Qwen cell passes with
   complete critical-fact recovery from all five receivers and 1.92–2.88% WER.
   Run the dialogue with the same warm voice description before starting
   Voxtral. Qwen listening, Qwen3-ASR 1.7B, and Nemotron remain open for the
   bounded calibration set.
4. Reject a generator with repeated omissions, insertions, truncation,
   non-finite audio, unstable completion, or unjustified operational cost.
5. Freeze the surviving three-engine, six-voice-per-locale matrix and expand it
   to the remaining passages. Report every result by locale, TTS engine, and
   voice before any aggregate.

This sequence avoids multi-gigabyte downloads and dozens of redundant runs for
a candidate that fails one representative passage.

## Primary sources

- Apple AVSpeechSynthesizer and voice identity:
  https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer and
  https://developer.apple.com/documentation/avfaudio/avspeechsynthesisvoice/identifier
- Qwen3-TTS models and languages:
  https://github.com/QwenLM/Qwen3-TTS
- Voxtral TTS model, languages, and license:
  https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 and
  https://docs.mistral.ai/models/model-cards/voxtral-tts-26-03
- KugelAudio model, voices, resource requirement, and license:
  https://huggingface.co/kugelaudio/kugelaudio-0-open
- Chatterbox Multilingual V3:
  https://github.com/resemble-ai/chatterbox
- Apple SpeechTranscriber:
  https://developer.apple.com/documentation/speech/speechtranscriber
- Qwen3-ASR:
  https://github.com/QwenLM/Qwen3-ASR
- Voxtral Realtime:
  https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- Parakeet TDT 0.6B v3:
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Whisper large-v3-turbo:
  https://github.com/openai/whisper and
  https://huggingface.co/openai/whisper-large-v3-turbo
- Nemotron 3.5 ASR Streaming 0.6B:
  https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- Accepted shared MLX-Audio calibration runtime:
  https://github.com/Blaizzy/mlx-audio
