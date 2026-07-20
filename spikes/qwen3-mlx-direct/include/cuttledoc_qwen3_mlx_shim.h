#ifndef CUTTLEDOC_QWEN3_MLX_SHIM_H
#define CUTTLEDOC_QWEN3_MLX_SHIM_H

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

void cuttledoc_qwen3_mlx_free_string(char *value);

#ifdef __cplusplus
}
#endif

#endif
