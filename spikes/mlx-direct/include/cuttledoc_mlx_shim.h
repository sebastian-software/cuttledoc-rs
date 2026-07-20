#ifndef CUTTLEDOC_MLX_SHIM_H
#define CUTTLEDOC_MLX_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Task-level boundary for Whisper speech recognition. The handle owns all MLX
 * arrays and tokenizer data. Callers retain ownership of model path and audio
 * buffers. Returned JSON/error strings must be released with
 * cuttledoc_mlx_free_string.
 *
 * device_kind: 0 = CPU, 1 = GPU.
 */
void *cuttledoc_mlx_whisper_create(const char *model_directory,
                                   int32_t device_kind, char **error_out);

int32_t cuttledoc_mlx_whisper_describe(void *handle, char **json_out,
                                       char **error_out);

int32_t cuttledoc_mlx_whisper_transcribe(void *handle, const float *audio,
                                         size_t audio_len, char **json_out,
                                         char **error_out);

void cuttledoc_mlx_whisper_destroy(void *handle);
void cuttledoc_mlx_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
