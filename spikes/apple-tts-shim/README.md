# Apple speech synthesis Swift shim spike (#13)

This is a narrow repository-owned C boundary over
`AVSpeechSynthesizer`, not the public Cuttledoc synthesis API. It validates
the ownership and lifecycle questions required by ADR-0009 before a stable
`SpeechSynthesisEngine` contract exists.

The boundary:

- inventories installed voices for a requested locale;
- creates one serial synthesis session with an explicit installed voice;
- accepts caller-owned UTF-8 and copies native output into mono f32 PCM;
- returns the native sample rate, timing, chunk count, and voice identity;
- rejects concurrent use with stable busy status `4`;
- supports cross-thread immediate cancellation with status `3`; and
- requires explicit destruction after synthesis or cancellation.

Session creation and complete synthesis run on the same serial thread. The
blocking C call pumps that thread's run loop because
`AVSpeechSynthesizer.write` delivers its buffers there. Cancellation may cross
threads: the accepted `stopSpeaking(.immediate)` call is the synchronous
observation boundary and unblocks the synthesis worker without waiting for a
delegate callback.

No `AVSpeechSynthesizer`, utterance, voice, or `AVAudioBuffer` crosses the C
ABI. Swift collects framework-owned buffers, downmixes them to mono, and
allocates one contiguous result. Rust copies that result and immediately calls
`cuttledoc_tts_free_audio`. Strings follow the same explicit ownership rule.

Materialize the pinned Wikipedia text first:

```sh
node scripts/materialize-synthetic-roundtrip.mjs \
  --output-dir /absolute/path/to/cuttledoc-synthetic-roundtrip
```

Then build and run the German voice inventory, one real passage, the busy
probe, and cooperative cancellation:

```sh
CUTTLEDOC_TTS_TEXT_DIR=/absolute/path/to/cuttledoc-synthetic-roundtrip \
  bash scripts/run-apple-tts-spike.sh
```

The macOS speech service must be reachable from the executable context. A
restricted process sandbox may enumerate voices but receive an immediate
platform cancellation before the first audio buffer; the same binary succeeds
in a normal host process. Product packaging therefore needs an explicit
clean-artifact execution check.

The first verified host run selected the installed compact German voice Anna
(`com.apple.voice.compact.de-DE.Anna`) and produced 55.145 seconds of mono
22,050 Hz f32 audio from `synthetic-de-origin`. Time to first audio was
0.930 seconds, complete synthesis took 1.383 seconds (RTF 0.0251), peak
footprint was 28.6 MB, busy returned status `4`, and cancellation returned
status `3` 0.202 ms after the Rust cancel call. These are development
measurements from one warm host run, not release thresholds.

The probe retains the platform's native voice sample rate. The later benchmark
runner must resample each candidate once into a shared digest-checked ASR PCM
artifact; ASR-specific transformations remain forbidden.
