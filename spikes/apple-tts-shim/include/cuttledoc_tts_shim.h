#ifndef CUTTLEDOC_TTS_SHIM_H
#define CUTTLEDOC_TTS_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Returns installed AVSpeechSynthesizer voices as JSON. The caller owns the
 * returned string and releases it with cuttledoc_tts_free_string.
 */
int32_t cuttledoc_tts_voice_inventory(
    const char *locale,
    char **output,
    char **error_output
);

/*
 * Creates one serial synthesis session. A null or empty voice identifier
 * selects the installed system default for the requested locale.
 */
void *cuttledoc_tts_session_create(
    const char *locale,
    const char *voice_identifier,
    char **metadata,
    char **error_output
);

/*
 * Synthesizes UTF-8 text and returns Rust-owned-copyable mono f32 PCM in the
 * native voice sample rate. Create and synthesize must run on the same
 * runloop-owning serial worker; cancel may cross threads. Status 3 means
 * cancelled and status 4 means busy. The caller releases samples and strings
 * with the matching free functions.
 */
int32_t cuttledoc_tts_session_synthesize(
    void *handle,
    const char *text,
    float **samples,
    uint64_t *sample_count,
    uint32_t *sample_rate_hz,
    char **summary,
    char **error_output
);

void cuttledoc_tts_session_cancel(void *handle);
void cuttledoc_tts_session_destroy(void *handle);
void cuttledoc_tts_free_audio(float *samples);
void cuttledoc_tts_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
