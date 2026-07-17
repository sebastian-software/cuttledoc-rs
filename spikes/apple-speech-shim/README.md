# Apple Speech Swift shim spike (#11)

This is intentionally a tiny C ABI, not a general Swift binding layer:

```c
int32_t cuttledoc_speech_transcribe_file(const char *path, char **output);
void cuttledoc_speech_free_string(char *value);
```

The Swift implementation owns `SpeechAnalyzer`, `SpeechTranscriber`, asset
installation, the asynchronous result sequence, and error conversion. Rust owns
the call, converts the returned UTF-8 into a Rust `String`, and always returns
the allocated C string to the shim for release.

Run the checked-in Rust example and a temporary local AIFF fixture with:

```sh
bash scripts/run-apple-speech-spike.sh
```

The first run may download a system-managed Speech asset. The script is also an
executable-identity test: it runs from an unbundled Rust binary, not an app
bundle. Record the observed asset/identity behavior in the spike report; do not
generalize a success or failure into a product constraint without that evidence.

This first slice proves file transcription through Rust. It intentionally does
not yet claim PCM input, volatile result replacement, time ranges, confidence,
cancellation, locale inventory, or a production shim. Those are the remaining
acceptance work recorded in the report and must be implemented in the same
narrow boundary if this candidate advances.
