#ifndef CUTTLEDOC_SPEECH_SHIM_H
#define CUTTLEDOC_SPEECH_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*cuttledoc_speech_update_callback)(
    void *context,
    const char *update_json
);

int32_t cuttledoc_speech_locale_inventory(char **output);

void *cuttledoc_speech_session_create(
    const char *locale,
    uint32_t sample_rate,
    cuttledoc_speech_update_callback callback,
    void *callback_context,
    char **metadata,
    char **error_output
);

/*
 * The shim copies samples before returning. Calls for one session must be
 * serialized. Status 4 means that the bounded queue rejected this chunk and
 * the caller may retry it; 3 means that the session is no longer active.
 */
int32_t cuttledoc_speech_session_push_pcm_f32(
    void *handle,
    const float *samples,
    uint32_t sample_count
);

int32_t cuttledoc_speech_session_finish(
    void *handle,
    char **summary,
    char **error_output
);

void cuttledoc_speech_session_cancel(void *handle);
void cuttledoc_speech_session_destroy(void *handle);

/* Bootstrap compatibility entry point retained by the spike. */
int32_t cuttledoc_speech_transcribe_file(const char *path, char **output);

void cuttledoc_speech_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
