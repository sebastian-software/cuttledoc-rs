#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {

constexpr std::size_t kAudioFrameElements = 576;
constexpr std::size_t kProjectionOutputs = 4;

int32_t fail(const std::string& message, char** error_out) {
  if (error_out != nullptr) {
    *error_out = strdup(message.c_str());
  }
  return 1;
}

} // namespace

extern "C" int32_t cuttledoc_mlx_project_audio(
    const float* audio,
    std::size_t audio_len,
    const float* weights,
    std::size_t weights_len,
    float* output,
    std::size_t output_len,
    int32_t device_kind,
    char** error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (audio == nullptr || weights == nullptr || output == nullptr) {
    return fail("audio, weights, and output must be non-null", error_out);
  }
  if (audio_len != kAudioFrameElements) {
    return fail("audio must contain exactly 576 float samples", error_out);
  }
  if (weights_len != kAudioFrameElements * kProjectionOutputs) {
    return fail("weights must have shape [576, 4]", error_out);
  }
  if (output_len < kProjectionOutputs) {
    return fail("output must reserve four float values", error_out);
  }

  mx::Device device = mx::Device::cpu;
  if (device_kind == 1) {
    device = mx::Device::gpu;
  } else if (device_kind != 0) {
    return fail("device_kind must be 0 (CPU) or 1 (GPU)", error_out);
  }
  if (!mx::is_available(device)) {
    return fail("requested MLX device is not available", error_out);
  }

  try {
    // The C ABI borrows the caller's buffers only for this call. Copying into
    // C++ vectors establishes a clear ownership boundary before MLX lazily
    // captures the arrays in its graph.
    std::vector<float> owned_audio(audio, audio + audio_len);
    std::vector<float> owned_weights(weights, weights + weights_len);
    auto input = mx::array(owned_audio.begin(), {1, kAudioFrameElements});
    auto projection =
        mx::array(owned_weights.begin(), {kAudioFrameElements, kProjectionOutputs});

    // A dense projection plus activation is a representative audio-model
    // building block. It exercises input/weight materialization, lazy graph
    // evaluation, explicit CPU/GPU device selection, and copying a result back
    // across the C ABI without exposing an MLX array.
    auto scores = mx::tanh(mx::matmul(input, projection, device), device);
    mx::eval(scores);
    std::copy_n(scores.data<float>(), kProjectionOutputs, output);
    return 0;
  } catch (const std::exception& error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("MLX raised a non-standard exception", error_out);
  }
}

extern "C" void cuttledoc_mlx_free_string(char* value) {
  std::free(value);
}
