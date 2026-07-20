#ifndef CUTTLEDOC_QWEN3_MLX_SHIM_H
#define CUTTLEDOC_QWEN3_MLX_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

void cuttledoc_qwen3_mlx_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
