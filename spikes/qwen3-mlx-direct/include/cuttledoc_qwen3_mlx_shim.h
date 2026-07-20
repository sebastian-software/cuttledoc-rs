#ifndef CUTTLEDOC_QWEN3_MLX_SHIM_H
#define CUTTLEDOC_QWEN3_MLX_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cuttledoc_qwen3_mlx_status {
  CUTTLEDOC_QWEN3_MLX_OK = 0,
  CUTTLEDOC_QWEN3_MLX_INVALID_ARGUMENT = 1,
  CUTTLEDOC_QWEN3_MLX_RUNTIME_ERROR = 2,
  CUTTLEDOC_QWEN3_MLX_CANCELLED = 3,
  CUTTLEDOC_QWEN3_MLX_BUSY = 4
} cuttledoc_qwen3_mlx_status;

/*
 * First vertical slice of the Qwen3-ASR task adapter.
 *
 * Loads the pinned model.safetensors through the official MLX C++ API and
 * validates the architecture and affine 8-bit tensor layout needed by the
 * repository-owned audio encoder and text decoder. The returned JSON and error
 * strings must be released with cuttledoc_qwen3_mlx_free_string.
 */
int32_t cuttledoc_qwen3_mlx_inspect_model(const char *model_directory,
                                          char **json_out,
                                          char **error_out);

/*
 * Runs the repository-owned 128-bin frontend and Qwen3 audio convolution
 * stack. device_kind is 0 for CPU or 1 for GPU. The returned JSON contains
 * compact numerical fingerprints for parity checks; no MLX object crosses the
 * ABI.
 */
int32_t cuttledoc_qwen3_mlx_probe_audio_frontend(
    const char *model_directory, const float *audio, size_t audio_len,
    int32_t device_kind, char **json_out, char **error_out);

/*
 * Extends the frontend probe through positional embeddings, ragged block
 * attention, all 18 audio transformer layers, and the final 1024-dimensional
 * projection.
 */
int32_t cuttledoc_qwen3_mlx_probe_audio_encoder(
    const char *model_directory, const float *audio, size_t audio_len,
    int32_t device_kind, char **json_out, char **error_out);

/*
 * Builds the Qwen ASR prompt, dequantizes the prompt token embeddings, and
 * replaces every audio-pad row with the corresponding encoder feature.
 */
int32_t cuttledoc_qwen3_mlx_probe_prompt_embeddings(
    const char *model_directory, const float *audio, size_t audio_len,
    const char *language, int32_t device_kind, char **json_out,
    char **error_out);

/*
 * Runs the 28-layer quantized text model through prompt prefill and two greedy
 * decode positions using a repository-owned KV cache.
 */
int32_t cuttledoc_qwen3_mlx_probe_decoder_prefill(
    const char *model_directory, const float *audio, size_t audio_len,
    const char *language, int32_t device_kind, char **json_out,
    char **error_out);

/*
 * Runs greedy generation through an end token and returns the decoded UTF-8
 * transcript plus the generated token sequence.
 */
int32_t cuttledoc_qwen3_mlx_transcribe(
    const char *model_directory, const float *audio, size_t audio_len,
    const char *language, int32_t device_kind, char **json_out,
    char **error_out);

/*
 * Reusable task-level boundary. The handle owns the official MLX model arrays;
 * callers retain input buffers and copy the returned JSON before freeing it.
 * One transcription may be active per handle. Cancellation is thread-safe and
 * observed after the current synchronous MLX graph or at the next decoder step.
 * The caller must await an active call before destroying its handle.
 */
void *cuttledoc_qwen3_mlx_session_create(
    const char *model_directory, int32_t device_kind, int32_t *status_out,
    char **error_out);

int32_t cuttledoc_qwen3_mlx_session_transcribe(
    void *handle, const float *audio, size_t audio_len, const char *language,
    char **json_out, char **error_out);

void cuttledoc_qwen3_mlx_session_cancel(void *handle);
void cuttledoc_qwen3_mlx_session_destroy(void *handle);

void cuttledoc_qwen3_mlx_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
