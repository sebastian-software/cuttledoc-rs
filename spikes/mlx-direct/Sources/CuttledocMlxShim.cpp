#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuttledoc_mlx_shim.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {

constexpr std::size_t kSampleRate = 16'000;
constexpr std::size_t kChunkSamples = 30 * kSampleRate;
constexpr std::size_t kFftSize = 400;
constexpr std::size_t kHopLength = 160;
constexpr std::size_t kMelFrames = 3'000;
constexpr std::size_t kAudioContext = 1'500;
constexpr std::size_t kAudioState = 384;
constexpr std::size_t kAudioHeads = 6;
constexpr std::size_t kAudioLayers = 4;
constexpr float kLayerNormEpsilon = 1e-5f;

std::mutex runtime_mutex;

int32_t fail(const std::string& message, char** error_out) {
  if (error_out != nullptr) {
    *error_out = strdup(message.c_str());
  }
  return 1;
}

void expect_shape(
    const mx::array& value,
    const mx::Shape& expected,
    const std::string& name) {
  if (value.shape() != expected) {
    std::ostringstream message;
    message << name << " has shape [";
    for (std::size_t index = 0; index < value.ndim(); ++index) {
      if (index != 0) {
        message << ", ";
      }
      message << value.shape(static_cast<int>(index));
    }
    message << "], expected [";
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (index != 0) {
        message << ", ";
      }
      message << expected[index];
    }
    message << "]";
    throw std::runtime_error(message.str());
  }
}

class WhisperEncoder {
 public:
  WhisperEncoder(std::filesystem::path model_directory, mx::Device device)
      : model_directory_(std::move(model_directory)), device_(device) {
    load_model();
  }

  const mx::Device& device() const {
    return device_;
  }

  const char* device_name() const {
    return device_ == mx::Device::cpu ? "cpu" : "gpu";
  }

  mx::Dtype compute_dtype() const {
    // The converted model stores FP16 weights. MLX Whisper's supported CPU
    // path keeps the log-mel input in FP32, which promotes the encoder graph
    // to FP32; the Metal path uses FP16 throughout.
    return device_ == mx::Device::cpu ? mx::float32 : mx::float16;
  }

  std::string load_description() const {
    std::ostringstream json;
    json << std::fixed << std::setprecision(3)
         << "{\"mlx_version\":\"" << mx::version() << "\","
         << "\"device\":\"" << device_name() << "\","
         << "\"loaded_tensors\":" << weights_.size() - 2 << ","
         << "\"load_peak_memory_bytes\":" << load_peak_memory_bytes_ << ","
         << "\"active_memory_bytes\":" << load_active_memory_bytes_ << ","
         << "\"cache_memory_bytes\":" << load_cache_memory_bytes_ << "}";
    return json.str();
  }

  std::string encode(const float* audio, std::size_t audio_len) const {
    if (audio == nullptr) {
      throw std::invalid_argument("audio must be non-null");
    }
    if (audio_len == 0 || audio_len > kChunkSamples) {
      throw std::invalid_argument(
          "audio must contain between 1 and 480000 mono f32 samples");
    }

    mx::set_default_device(device_);
    mx::reset_peak_memory();
    const auto started_at = std::chrono::steady_clock::now();

    auto encoded = encode_audio(log_mel(audio, audio_len));
    auto materialized =
        mx::contiguous(mx::astype(encoded, mx::float32, device_), true, device_);
    mx::eval(materialized);

    const auto finished_at = std::chrono::steady_clock::now();
    const auto inference_ms =
        std::chrono::duration<double, std::milli>(finished_at - started_at)
            .count();
    const auto* values = materialized.data<float>();
    const auto value_count = materialized.size();

    double sum = 0.0;
    double squared_sum = 0.0;
    double absolute_sum = 0.0;
    auto minimum = std::numeric_limits<float>::infinity();
    auto maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < value_count; ++index) {
      const auto value = values[index];
      sum += value;
      squared_sum += static_cast<double>(value) * value;
      absolute_sum += std::abs(value);
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
    const auto mean = sum / static_cast<double>(value_count);
    const auto variance =
        std::max(0.0, squared_sum / static_cast<double>(value_count) - mean * mean);

    std::ostringstream json;
    json << std::fixed << std::setprecision(9)
         << "{\"mlx_version\":\"" << mx::version() << "\","
         << "\"device\":\"" << device_name() << "\","
         << "\"input_samples\":" << audio_len << ","
         << "\"output_shape\":[1," << kAudioContext << "," << kAudioState
         << "],"
         << "\"inference_ms\":" << inference_ms << ","
         << "\"peak_memory_bytes\":" << mx::get_peak_memory() << ","
         << "\"active_memory_bytes\":" << mx::get_active_memory() << ","
         << "\"cache_memory_bytes\":" << mx::get_cache_memory() << ","
         << "\"fingerprint\":{"
         << "\"mean\":" << mean << ","
         << "\"stddev\":" << std::sqrt(variance) << ","
         << "\"minimum\":" << minimum << ","
         << "\"maximum\":" << maximum << ","
         << "\"l1\":" << absolute_sum << ","
         << "\"first_values\":[";
    for (std::size_t index = 0; index < 8; ++index) {
      if (index != 0) {
        json << ",";
      }
      json << values[index];
    }
    json << "]}}";
    return json.str();
  }

 private:
  const mx::array& weight(const std::string& name) const {
    return weights_.at(name);
  }

  void add_weight(
      const std::string& name,
      const mx::Shape& expected_shape) {
    const auto path = model_directory_ / (name + ".npy");
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("missing model tensor: " + path.string());
    }
    // MLX's NPY Load primitive evaluates on CPU only. Materialize there, then
    // copy the tensor to the requested device before the session evaluates
    // its owned weight set.
    auto tensor = mx::load(path.string(), mx::Device::cpu);
    if (device_ == mx::Device::gpu) {
      tensor = mx::copy(std::move(tensor), device_);
    }
    expect_shape(tensor, expected_shape, name);
    weights_.emplace(name, std::move(tensor));
  }

  void load_model() {
    mx::set_default_device(device_);

    add_weight("encoder.conv1.weight", {384, 3, 80});
    add_weight("encoder.conv1.bias", {384});
    add_weight("encoder.conv2.weight", {384, 3, 384});
    add_weight("encoder.conv2.bias", {384});

    for (std::size_t layer = 0; layer < kAudioLayers; ++layer) {
      const auto prefix = "encoder.blocks." + std::to_string(layer) + ".";
      add_weight(prefix + "attn.query.weight", {384, 384});
      add_weight(prefix + "attn.query.bias", {384});
      add_weight(prefix + "attn.key.weight", {384, 384});
      add_weight(prefix + "attn.value.weight", {384, 384});
      add_weight(prefix + "attn.value.bias", {384});
      add_weight(prefix + "attn.out.weight", {384, 384});
      add_weight(prefix + "attn.out.bias", {384});
      add_weight(prefix + "attn_ln.weight", {384});
      add_weight(prefix + "attn_ln.bias", {384});
      add_weight(prefix + "mlp1.weight", {1536, 384});
      add_weight(prefix + "mlp1.bias", {1536});
      add_weight(prefix + "mlp2.weight", {384, 1536});
      add_weight(prefix + "mlp2.bias", {384});
      add_weight(prefix + "mlp_ln.weight", {384});
      add_weight(prefix + "mlp_ln.bias", {384});
    }
    add_weight("encoder.ln_post.weight", {384});
    add_weight("encoder.ln_post.bias", {384});
    add_weight("mel_80", {80, 201});

    const auto log_timescale_increment =
        std::log(10'000.0f) / (static_cast<float>(kAudioState / 2) - 1.0f);
    auto inv_timescales = mx::exp(
        -log_timescale_increment *
            mx::arange(static_cast<int>(kAudioState / 2), mx::float32, device_),
        device_);
    auto scaled_time =
        mx::expand_dims(
            mx::arange(static_cast<int>(kAudioContext), mx::float32, device_),
            1,
            device_) *
        mx::expand_dims(inv_timescales, 0, device_);
    auto positional_embedding = mx::astype(
        mx::concatenate(
            {mx::sin(scaled_time, device_), mx::cos(scaled_time, device_)},
            1,
            device_),
        compute_dtype(),
        device_);
    weights_.emplace("__positional_embedding", std::move(positional_embedding));

    std::vector<mx::array> tensors;
    tensors.reserve(weights_.size());
    for (const auto& [name, tensor] : weights_) {
      static_cast<void>(name);
      tensors.push_back(tensor);
    }
    mx::eval(std::move(tensors));
    load_peak_memory_bytes_ = mx::get_peak_memory();
    load_active_memory_bytes_ = mx::get_active_memory();
    load_cache_memory_bytes_ = mx::get_cache_memory();
  }

  mx::array log_mel(const float* audio, std::size_t audio_len) const {
    std::vector<float> padded_audio(kChunkSamples + kFftSize, 0.0f);
    std::copy_n(audio, audio_len, padded_audio.begin() + kFftSize / 2);

    for (std::size_t index = 0; index < kFftSize / 2; ++index) {
      padded_audio[index] =
          padded_audio[kFftSize - index];
      padded_audio[kFftSize / 2 + kChunkSamples + index] =
          padded_audio[kFftSize / 2 + kChunkSamples - 2 - index];
    }

    std::vector<float> window(kFftSize);
    constexpr auto pi = 3.14159265358979323846;
    for (std::size_t index = 0; index < kFftSize; ++index) {
      window[index] =
          0.5f -
          0.5f * std::cos(
                     2.0 * pi * static_cast<double>(index) /
                     static_cast<double>(kFftSize));
    }

    auto audio_array =
        mx::array(padded_audio.begin(), {static_cast<int>(padded_audio.size())});
    auto frames = mx::as_strided(
        audio_array,
        {static_cast<int>(kMelFrames + 1), static_cast<int>(kFftSize)},
        {static_cast<int64_t>(kHopLength), 1},
        0,
        device_);
    auto window_array =
        mx::array(window.begin(), {static_cast<int>(window.size())});
    auto frequencies = mx::fft::rfft(
        frames * window_array, -1, mx::fft::FFTNorm::Backward, device_);
    frequencies = mx::slice(
        frequencies,
        {0, 0},
        {static_cast<int>(kMelFrames), static_cast<int>(kFftSize / 2 + 1)},
        {1, 1},
        device_);
    auto magnitudes = mx::square(mx::abs(frequencies, device_), device_);
    auto mel = mx::matmul(
        magnitudes, mx::transpose(weight("mel_80"), {1, 0}, device_), device_);
    auto log_spec = mx::log10(
        mx::maximum(mel, mx::array(1e-10f), device_), device_);
    log_spec = mx::maximum(log_spec, mx::max(log_spec, device_) - 8.0, device_);
    log_spec = (log_spec + 4.0) / 4.0;
    return mx::expand_dims(
        mx::astype(log_spec, compute_dtype(), device_),
        0,
        device_);
  }

  mx::array linear(
      const mx::array& input,
      const std::string& prefix,
      bool has_bias = true) const {
    if (has_bias) {
      return mx::addmm(
          weight(prefix + ".bias"),
          input,
          mx::transpose(weight(prefix + ".weight"), device_),
          1.0f,
          1.0f,
          device_);
    }
    return mx::matmul(
        input,
        mx::transpose(weight(prefix + ".weight"), device_),
        device_);
  }

  mx::array layer_norm(
      const mx::array& input,
      const std::string& prefix) const {
    return mx::fast::layer_norm(
        input,
        weight(prefix + ".weight"),
        weight(prefix + ".bias"),
        kLayerNormEpsilon,
        device_);
  }

  mx::array gelu(const mx::array& input) const {
    const auto one = mx::array(1.0f, input.dtype());
    const auto inverse_sqrt_two =
        mx::array(1.0f / std::sqrt(2.0f), input.dtype());
    const auto half = mx::array(0.5f, input.dtype());
    return mx::multiply(
        input,
        mx::multiply(
            one +
                mx::erf(
                    mx::multiply(input, inverse_sqrt_two, device_),
                    device_),
            half,
            device_),
        device_);
  }

  mx::array self_attention(
      const mx::array& input,
      const std::string& prefix) const {
    auto query = linear(input, prefix + ".query");
    auto key = linear(input, prefix + ".key", false);
    auto value = linear(input, prefix + ".value");
    const auto scale = mx::array(
        std::pow(
            static_cast<float>(kAudioState / kAudioHeads),
            -0.25f),
        query.dtype());

    query = mx::multiply(
        mx::transpose(
            mx::reshape(
                query,
                {1,
                 static_cast<int>(kAudioContext),
                 static_cast<int>(kAudioHeads),
                 static_cast<int>(kAudioState / kAudioHeads)},
                device_),
            {0, 2, 1, 3},
            device_),
        scale,
        device_);
    key = mx::multiply(
        mx::transpose(
            mx::reshape(
                key,
                {1,
                 static_cast<int>(kAudioContext),
                 static_cast<int>(kAudioHeads),
                 static_cast<int>(kAudioState / kAudioHeads)},
                device_),
            {0, 2, 3, 1},
            device_),
        scale,
        device_);
    value = mx::transpose(
        mx::reshape(
            value,
            {1,
             static_cast<int>(kAudioContext),
             static_cast<int>(kAudioHeads),
             static_cast<int>(kAudioState / kAudioHeads)},
            device_),
        {0, 2, 1, 3},
        device_);

    auto attention_weights = mx::softmax(
        mx::matmul(query, key, device_), -1, true, device_);
    auto attended = mx::matmul(attention_weights, value, device_);
    attended = mx::reshape(
        mx::transpose(attended, {0, 2, 1, 3}, device_),
        {1, static_cast<int>(kAudioContext), static_cast<int>(kAudioState)},
        device_);
    return linear(attended, prefix + ".out");
  }

  mx::array encode_audio(const mx::array& mel) const {
    auto encoded = gelu(
        mx::conv1d(mel, weight("encoder.conv1.weight"), 1, 1, 1, 1, device_) +
        weight("encoder.conv1.bias"));
    encoded = gelu(
        mx::conv1d(
            encoded,
            weight("encoder.conv2.weight"),
            2,
            1,
            1,
            1,
            device_) +
        weight("encoder.conv2.bias"));
    if (encoded.shape() != mx::Shape{1, 1500, 384}) {
      throw std::runtime_error("Whisper convolution produced an invalid shape");
    }
    encoded = encoded + weight("__positional_embedding");

    for (std::size_t layer = 0; layer < kAudioLayers; ++layer) {
      const auto prefix = "encoder.blocks." + std::to_string(layer);
      encoded =
          encoded +
          self_attention(
              layer_norm(encoded, prefix + ".attn_ln"),
              prefix + ".attn");
      const auto normalized = layer_norm(encoded, prefix + ".mlp_ln");
      encoded =
          encoded +
          linear(gelu(linear(normalized, prefix + ".mlp1")), prefix + ".mlp2");
    }
    encoded = layer_norm(encoded, "encoder.ln_post");
    return encoded;
  }

  std::filesystem::path model_directory_;
  mx::Device device_;
  std::unordered_map<std::string, mx::array> weights_;
  std::size_t load_peak_memory_bytes_ = 0;
  std::size_t load_active_memory_bytes_ = 0;
  std::size_t load_cache_memory_bytes_ = 0;
};

WhisperEncoder* as_encoder(void* handle) {
  if (handle == nullptr) {
    throw std::invalid_argument("encoder handle must be non-null");
  }
  return static_cast<WhisperEncoder*>(handle);
}

int32_t copy_json(
    const std::string& json,
    char** json_out,
    char** error_out) {
  if (json_out == nullptr) {
    return fail("json_out must be non-null", error_out);
  }
  *json_out = strdup(json.c_str());
  if (*json_out == nullptr) {
    return fail("could not allocate JSON result", error_out);
  }
  return 0;
}

} // namespace

extern "C" void* cuttledoc_mlx_whisper_encoder_create(
    const char* model_directory,
    int32_t device_kind,
    char** error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr) {
    fail("model_directory must be non-null", error_out);
    return nullptr;
  }

  mx::Device device = mx::Device::cpu;
  if (device_kind == 1) {
    device = mx::Device::gpu;
  } else if (device_kind != 0) {
    fail("device_kind must be 0 (CPU) or 1 (GPU)", error_out);
    return nullptr;
  }
  if (!mx::is_available(device)) {
    fail("requested MLX device is not available", error_out);
    return nullptr;
  }

  try {
    std::scoped_lock lock(runtime_mutex);
    mx::set_default_device(device);
    mx::clear_cache();
    mx::reset_peak_memory();
    return new WhisperEncoder(model_directory, device);
  } catch (const std::exception& error) {
    fail(error.what(), error_out);
  } catch (...) {
    fail("MLX raised a non-standard exception while loading the model", error_out);
  }
  return nullptr;
}

extern "C" int32_t cuttledoc_mlx_whisper_encoder_describe(
    void* handle,
    char** json_out,
    char** error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  try {
    std::scoped_lock lock(runtime_mutex);
    return copy_json(as_encoder(handle)->load_description(), json_out, error_out);
  } catch (const std::exception& error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("could not describe MLX encoder", error_out);
  }
}

extern "C" int32_t cuttledoc_mlx_whisper_encoder_encode(
    void* handle,
    const float* audio,
    std::size_t audio_len,
    char** json_out,
    char** error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  try {
    std::scoped_lock lock(runtime_mutex);
    auto* encoder = as_encoder(handle);
    mx::set_default_device(encoder->device());
    return copy_json(encoder->encode(audio, audio_len), json_out, error_out);
  } catch (const std::exception& error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("MLX raised a non-standard exception during encoding", error_out);
  }
}

extern "C" void cuttledoc_mlx_whisper_encoder_destroy(void* handle) {
  if (handle == nullptr) {
    return;
  }
  std::scoped_lock lock(runtime_mutex);
  delete static_cast<WhisperEncoder*>(handle);
  mx::clear_cache();
}

extern "C" void cuttledoc_mlx_free_string(char* value) {
  std::free(value);
}
