#ifndef CUTTLEDOC_VOXTRAL_MLX_SHIM_H
#define CUTTLEDOC_VOXTRAL_MLX_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cuttledoc_voxtral_mlx_status {
  CUTTLEDOC_VOXTRAL_MLX_OK = 0,
  CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT = 1,
  CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR = 2,
  CUTTLEDOC_VOXTRAL_MLX_CANCELLED = 3,
  CUTTLEDOC_VOXTRAL_MLX_BUSY = 4,
  CUTTLEDOC_VOXTRAL_MLX_BACKPRESSURE = 5,
  CUTTLEDOC_VOXTRAL_MLX_NEEDS_AUDIO = 6,
  CUTTLEDOC_VOXTRAL_MLX_DONE = 7
} cuttledoc_voxtral_mlx_status;

/*
 * Loads the pinned Voxtral safetensors with the official MLX C++ API and
 * validates the artifact, tensor layout, and critical architecture shapes.
 * Reported capabilities distinguish usable batch transcription from the
 * still-incomplete incremental streaming session.
 */
int32_t cuttledoc_voxtral_mlx_inspect_model(const char *model_directory,
                                             char **json_out,
                                             char **error_out);

/*
 * Runs the actual Voxtral offline-streaming pad, log-mel, and causal Conv1d
 * stem through official MLX and returns numerical fingerprints. This parity
 * gate still stops before the 32-layer causal encoder and decoder.
 */
int32_t cuttledoc_voxtral_mlx_probe_audio_frontend(
    const char *model_directory, const float *audio, size_t audio_len,
    int32_t transcription_delay_ms, int32_t device_kind, char **json_out,
    char **error_out);

/*
 * Runs all 32 causal encoder layers, the repository-owned rotating KV cache
 * and sliding-window mask, 4x downsampling, and the audio-language adapter.
 * The result is a numerical parity probe and still does not claim decoding.
 */
int32_t cuttledoc_voxtral_mlx_probe_causal_encoder(
    const char *model_directory, const float *audio, size_t audio_len,
    int32_t transcription_delay_ms, int32_t device_kind, char **json_out,
    char **error_out);

/*
 * Runs the repository-owned delay-conditioned Voxtral decoder over the direct
 * encoder output, greedily decodes Tekken tokens, and returns transcript and
 * numerical parity evidence. MLX supplies tensor execution only; the model
 * graph, cache semantics, generation loop, and tokenizer live in this shim.
 */
int32_t cuttledoc_voxtral_mlx_transcribe(
    const char *model_directory, const float *audio, size_t audio_len,
    int32_t transcription_delay_ms, size_t max_generated_tokens,
    int32_t device_kind, char **json_out, char **error_out);

/*
 * Creates the repository-owned incremental Voxtral session. The delay and
 * generation limits are fixed for the lifetime of the session.
 * max_pending_samples is the hard caller-visible queue capacity;
 * max_ingest_samples_per_step and max_decode_tokens_per_step bound the work
 * performed by one synchronous step.
 */
void *cuttledoc_voxtral_mlx_session_create(
    const char *model_directory, int32_t transcription_delay_ms,
    size_t max_generated_tokens, size_t max_decode_tokens_per_step,
    int32_t device_kind,
    size_t max_pending_samples, size_t max_ingest_samples_per_step,
    int32_t *status_out, char **error_out);

/*
 * Feed is thread-safe and all-or-nothing. BACKPRESSURE means the caller keeps
 * ownership of the complete input slice and may retry after a successful
 * step. No MLX work occurs in feed.
 */
int32_t cuttledoc_voxtral_mlx_session_feed(void *handle, const float *audio,
                                           size_t audio_len,
                                           char **error_out);

/* Marks end-of-audio. It is distinct from cancellation. */
int32_t cuttledoc_voxtral_mlx_session_close(void *handle, char **error_out);

/*
 * Removes at most max_ingest_samples_per_step from a fixed snapshot and
 * advances persistent mel, causal-convolution, encoder, downsampler, decoder,
 * tokenizer, and delta-text state through official MLX tensor operations.
 * NEEDS_AUDIO and DONE are normal states and also return JSON.
 */
int32_t cuttledoc_voxtral_mlx_session_step(void *handle, char **json_out,
                                           char **error_out);

/* Cooperative and thread-safe; observed before or after synchronous MLX work. */
void cuttledoc_voxtral_mlx_session_cancel(void *handle);
void cuttledoc_voxtral_mlx_session_destroy(void *handle);

void cuttledoc_voxtral_mlx_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
