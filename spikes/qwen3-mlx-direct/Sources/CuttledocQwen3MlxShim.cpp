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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuttledoc_qwen3_mlx_shim.h"
#include "mlx/io.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {

constexpr std::uintmax_t kSafetensorsBytes = 1'006'229'426;
constexpr std::size_t kExpectedTensorCount = 1'005;
constexpr std::size_t kExpectedBfloat16Count = 808;
constexpr std::size_t kExpectedUint32Count = 197;
constexpr std::size_t kExpectedQuantizedModuleCount = 197;
constexpr std::size_t kFftSize = 400;
constexpr std::size_t kHopLength = 160;
constexpr std::size_t kMelBins = 128;
constexpr std::size_t kFrequencyBins = kFftSize / 2 + 1;
constexpr std::size_t kConvChunkFrames = 100;

std::mutex runtime_mutex;

int32_t fail(const std::string &message, char **error_out) {
  if (error_out != nullptr) {
    *error_out = strdup(message.c_str());
  }
  return 1;
}

std::string json_escape(std::string_view value) {
  std::ostringstream escaped;
  for (const auto character : value) {
    switch (character) {
    case '"':
      escaped << "\\\"";
      break;
    case '\\':
      escaped << "\\\\";
      break;
    case '\b':
      escaped << "\\b";
      break;
    case '\f':
      escaped << "\\f";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      if (static_cast<unsigned char>(character) < 0x20) {
        escaped << "\\u00";
        constexpr std::string_view digits = "0123456789abcdef";
        escaped << digits[(static_cast<unsigned char>(character) >> 4) & 0xf]
                << digits[static_cast<unsigned char>(character) & 0xf];
      } else {
        escaped << character;
      }
    }
  }
  return escaped.str();
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

std::string shape_string(const mx::array &value) {
  std::ostringstream shape;
  shape << "[";
  for (std::size_t index = 0; index < value.ndim(); ++index) {
    if (index != 0) {
      shape << ",";
    }
    shape << value.shape(static_cast<int>(index));
  }
  shape << "]";
  return shape.str();
}

void expect_shape(const std::unordered_map<std::string, mx::array> &weights,
                  const std::string &name, const mx::Shape &expected) {
  const auto found = weights.find(name);
  if (found == weights.end()) {
    throw std::runtime_error("missing required tensor: " + name);
  }
  if (found->second.shape() != expected) {
    std::ostringstream message;
    message << name << " has shape " << shape_string(found->second)
            << ", expected [";
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (index != 0) {
        message << ",";
      }
      message << expected[index];
    }
    message << "]";
    throw std::runtime_error(message.str());
  }
}

std::string inspect_model(const std::filesystem::path &model_directory) {
  const auto config_path = model_directory / "config.json";
  const auto preprocessor_path = model_directory / "preprocessor_config.json";
  const auto vocabulary_path = model_directory / "vocab.json";
  const auto merges_path = model_directory / "merges.txt";
  const auto safetensors_path = model_directory / "model.safetensors";
  for (const auto &path : {config_path, preprocessor_path, vocabulary_path,
                           merges_path, safetensors_path}) {
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("missing model artifact: " + path.string());
    }
  }
  const auto safetensors_bytes = std::filesystem::file_size(safetensors_path);
  if (safetensors_bytes != kSafetensorsBytes) {
    std::ostringstream message;
    message << "model.safetensors has " << safetensors_bytes
            << " bytes, expected " << kSafetensorsBytes;
    throw std::runtime_error(message.str());
  }

  const auto [weights, metadata] =
      mx::load_safetensors(safetensors_path.string(), mx::Device::cpu);
  if (weights.size() != kExpectedTensorCount) {
    std::ostringstream message;
    message << "model has " << weights.size() << " tensors, expected "
            << kExpectedTensorCount;
    throw std::runtime_error(message.str());
  }

  std::size_t bfloat16_count = 0;
  std::size_t uint32_count = 0;
  std::size_t quantized_module_count = 0;
  std::size_t scales_count = 0;
  std::size_t biases_count = 0;
  for (const auto &[name, value] : weights) {
    if (value.dtype() == mx::bfloat16) {
      ++bfloat16_count;
    } else if (value.dtype() == mx::uint32) {
      ++uint32_count;
    } else {
      throw std::runtime_error("unexpected dtype for tensor: " + name);
    }
    if (ends_with(name, ".scales")) {
      ++scales_count;
    }
    if (ends_with(name, ".biases")) {
      ++biases_count;
    }
    if (ends_with(name, ".weight")) {
      const auto prefix = name.substr(0, name.size() - std::strlen(".weight"));
      if (weights.contains(prefix + ".scales") &&
          weights.contains(prefix + ".biases")) {
        ++quantized_module_count;
      }
    }
  }
  if (bfloat16_count != kExpectedBfloat16Count ||
      uint32_count != kExpectedUint32Count ||
      quantized_module_count != kExpectedQuantizedModuleCount ||
      scales_count != kExpectedQuantizedModuleCount ||
      biases_count != kExpectedQuantizedModuleCount) {
    std::ostringstream message;
    message << "unexpected dtype/quantization layout: bfloat16="
            << bfloat16_count << ", uint32=" << uint32_count
            << ", quantized_modules=" << quantized_module_count
            << ", scales=" << scales_count << ", biases=" << biases_count;
    throw std::runtime_error(message.str());
  }

  expect_shape(weights, "audio_tower.conv2d1.weight", {480, 3, 3, 1});
  expect_shape(weights, "audio_tower.conv2d2.weight", {480, 3, 3, 480});
  expect_shape(weights, "audio_tower.conv2d3.weight", {480, 3, 3, 480});
  expect_shape(weights, "audio_tower.conv_out.weight", {896, 7680});
  expect_shape(weights, "audio_tower.layers.0.self_attn.q_proj.weight",
               {896, 896});
  expect_shape(weights, "audio_tower.layers.17.self_attn.q_proj.weight",
               {896, 896});
  expect_shape(weights, "audio_tower.proj2.weight", {1024, 896});
  expect_shape(weights, "model.embed_tokens.weight", {151936, 256});
  expect_shape(weights, "model.embed_tokens.scales", {151936, 16});
  expect_shape(weights, "model.embed_tokens.biases", {151936, 16});
  expect_shape(weights, "model.layers.0.self_attn.q_proj.weight", {2048, 256});
  expect_shape(weights, "model.layers.0.self_attn.q_proj.scales", {2048, 16});
  expect_shape(weights, "model.layers.27.mlp.down_proj.weight", {1024, 768});
  expect_shape(weights, "model.norm.weight", {1024});
  if (weights.contains("audio_tower.layers.18.self_attn.q_proj.weight") ||
      weights.contains("model.layers.28.self_attn.q_proj.weight")) {
    throw std::runtime_error("model contains layers beyond the pinned config");
  }

  std::ostringstream json;
  json << "{\"status\":\"ok\",\"boundary\":\"official-mlx-cpp\","
       << "\"mlx_version\":\"" << json_escape(mx::version()) << "\","
       << "\"model_directory\":\""
       << json_escape(model_directory.string()) << "\","
       << "\"model_artifact\":{\"file\":\"model.safetensors\","
       << "\"bytes\":" << safetensors_bytes << "},"
       << "\"weights\":{\"tensor_count\":" << weights.size()
       << ",\"metadata_entries\":" << metadata.size()
       << ",\"bfloat16_tensors\":" << bfloat16_count
       << ",\"uint32_tensors\":" << uint32_count
       << ",\"affine_8bit_modules\":" << quantized_module_count << "},"
       << "\"architecture\":{\"audio_encoder_layers\":18,"
       << "\"audio_state\":896,\"audio_output_dim\":1024,"
       << "\"text_decoder_layers\":28,\"text_hidden_size\":1024,"
       << "\"text_attention_heads\":16,\"text_kv_heads\":8,"
       << "\"vocabulary_size\":151936,\"quantization_bits\":8,"
       << "\"quantization_group_size\":64},"
       << "\"validated_shapes\":14}";
  return json.str();
}

double hertz_to_slaney_mel(double frequency) {
  constexpr double minimum_log_hertz = 1000.0;
  constexpr double minimum_log_mel = 15.0;
  const auto log_step = 27.0 / std::log(6.4);
  if (frequency >= minimum_log_hertz) {
    return minimum_log_mel +
           std::log(frequency / minimum_log_hertz) * log_step;
  }
  return 3.0 * frequency / 200.0;
}

double slaney_mel_to_hertz(double mel) {
  constexpr double minimum_log_hertz = 1000.0;
  constexpr double minimum_log_mel = 15.0;
  const auto log_step = std::log(6.4) / 27.0;
  if (mel >= minimum_log_mel) {
    return minimum_log_hertz *
           std::exp(log_step * (mel - minimum_log_mel));
  }
  return 200.0 * mel / 3.0;
}

std::vector<float> make_slaney_mel_filters() {
  constexpr std::size_t filter_points = kMelBins + 2;
  const auto minimum_mel = hertz_to_slaney_mel(0.0);
  const auto maximum_mel = hertz_to_slaney_mel(8000.0);
  std::vector<double> filter_frequencies(filter_points);
  for (std::size_t index = 0; index < filter_points; ++index) {
    const auto fraction =
        static_cast<double>(index) / static_cast<double>(filter_points - 1);
    filter_frequencies[index] =
        slaney_mel_to_hertz(minimum_mel +
                            fraction * (maximum_mel - minimum_mel));
  }

  std::vector<float> filters(kFrequencyBins * kMelBins, 0.0f);
  for (std::size_t frequency_index = 0; frequency_index < kFrequencyBins;
       ++frequency_index) {
    const auto frequency =
        8000.0 * static_cast<double>(frequency_index) /
        static_cast<double>(kFrequencyBins - 1);
    for (std::size_t mel_index = 0; mel_index < kMelBins; ++mel_index) {
      const auto lower = filter_frequencies[mel_index];
      const auto center = filter_frequencies[mel_index + 1];
      const auto upper = filter_frequencies[mel_index + 2];
      const auto down_slope = (frequency - lower) / (center - lower);
      const auto up_slope = (upper - frequency) / (upper - center);
      const auto triangle = std::max(0.0, std::min(down_slope, up_slope));
      const auto area_normalization = 2.0 / (upper - lower);
      filters[frequency_index * kMelBins + mel_index] =
          static_cast<float>(triangle * area_normalization);
    }
  }
  return filters;
}

std::string fingerprint_json(const mx::array &value, mx::Device device) {
  auto materialized =
      mx::contiguous(mx::astype(value, mx::float32, device), false, device);
  mx::eval(materialized);
  const auto *values = materialized.data<float>();
  const auto count = materialized.size();
  if (count == 0) {
    throw std::runtime_error("cannot fingerprint an empty MLX array");
  }

  const auto sum = std::accumulate(values, values + count, 0.0);
  const auto mean = sum / static_cast<double>(count);
  double squared_difference_sum = 0.0;
  double l1 = 0.0;
  auto minimum = std::numeric_limits<float>::infinity();
  auto maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < count; ++index) {
    const auto current = values[index];
    const auto difference = static_cast<double>(current) - mean;
    squared_difference_sum += difference * difference;
    l1 += std::abs(static_cast<double>(current));
    minimum = std::min(minimum, current);
    maximum = std::max(maximum, current);
  }

  std::vector<std::size_t> sample_indices{
      0,
      std::min<std::size_t>(1, count - 1),
      std::min<std::size_t>(2, count - 1),
      std::min<std::size_t>(7, count - 1),
      std::min<std::size_t>(31, count - 1),
      count / 3,
      (2 * count) / 3,
      count - 1,
  };
  std::sort(sample_indices.begin(), sample_indices.end());
  sample_indices.erase(
      std::unique(sample_indices.begin(), sample_indices.end()),
      sample_indices.end());

  std::ostringstream json;
  json << std::setprecision(17) << "{\"shape\":" << shape_string(materialized)
       << ",\"source_dtype\":\"float32\",\"mean\":" << mean
       << ",\"stddev\":"
       << std::sqrt(squared_difference_sum / static_cast<double>(count))
       << ",\"minimum\":" << minimum << ",\"maximum\":" << maximum
       << ",\"l1\":" << l1 << ",\"sample_indices\":[";
  for (std::size_t index = 0; index < sample_indices.size(); ++index) {
    if (index != 0) {
      json << ",";
    }
    json << sample_indices[index];
  }
  json << "],\"sample_values\":[";
  for (std::size_t index = 0; index < sample_indices.size(); ++index) {
    if (index != 0) {
      json << ",";
    }
    json << values[sample_indices[index]];
  }
  json << "]}";
  return json.str();
}

class Qwen3AudioFrontend {
public:
  Qwen3AudioFrontend(const std::filesystem::path &model_directory,
                     mx::Device device)
      : device_(device), mel_filters_(make_slaney_mel_filters()) {
    auto [loaded_weights, metadata] = mx::load_safetensors(
        (model_directory / "model.safetensors").string(), mx::Device::cpu);
    static_cast<void>(metadata);
    if (device_ == mx::Device::gpu) {
      for (auto &[name, value] : loaded_weights) {
        static_cast<void>(name);
        value = mx::copy(std::move(value), device_);
      }
    }
    weights_ = std::move(loaded_weights);
    for (const auto &[name, shape] :
         std::vector<std::pair<std::string, mx::Shape>>{
             {"audio_tower.conv2d1.weight", {480, 3, 3, 1}},
             {"audio_tower.conv2d1.bias", {480}},
             {"audio_tower.conv2d2.weight", {480, 3, 3, 480}},
             {"audio_tower.conv2d2.bias", {480}},
             {"audio_tower.conv2d3.weight", {480, 3, 3, 480}},
             {"audio_tower.conv2d3.bias", {480}},
             {"audio_tower.conv_out.weight", {896, 7680}},
         }) {
      expect_shape(weights_, name, shape);
    }
  }

  std::string probe(const float *audio, std::size_t audio_len) const {
    const auto started = std::chrono::steady_clock::now();
    const auto features = log_mel(audio, audio_len);
    std::vector<std::size_t> chunk_lengths;
    const auto chunks = make_chunks(features, chunk_lengths);

    auto hidden = mx::expand_dims(chunks, 3, device_);
    hidden = gelu(mx::conv2d(hidden, weight("audio_tower.conv2d1.weight"),
                             {2, 2}, {1, 1}, {1, 1}, 1, device_) +
                  weight("audio_tower.conv2d1.bias"));
    const auto conv2d1_fingerprint = fingerprint_json(hidden, device_);
    hidden = gelu(mx::conv2d(hidden, weight("audio_tower.conv2d2.weight"),
                             {2, 2}, {1, 1}, {1, 1}, 1, device_) +
                  weight("audio_tower.conv2d2.bias"));
    const auto conv2d2_fingerprint = fingerprint_json(hidden, device_);
    hidden = gelu(mx::conv2d(hidden, weight("audio_tower.conv2d3.weight"),
                             {2, 2}, {1, 1}, {1, 1}, 1, device_) +
                  weight("audio_tower.conv2d3.bias"));
    const auto conv2d3_fingerprint = fingerprint_json(hidden, device_);

    const auto batch_size = hidden.shape(0);
    const auto frequency = hidden.shape(1);
    const auto frames = hidden.shape(2);
    const auto channels = hidden.shape(3);
    hidden =
        mx::reshape(mx::transpose(hidden, {0, 2, 3, 1}, device_),
                    {batch_size, frames, channels * frequency}, device_);
    hidden =
        mx::matmul(hidden,
                   mx::transpose(weight("audio_tower.conv_out.weight"), device_),
                   device_);
    const auto conv_out_fingerprint = fingerprint_json(hidden, device_);

    std::ostringstream json;
    json << std::setprecision(17)
         << "{\"status\":\"ok\",\"boundary\":\"official-mlx-cpp\","
         << "\"stage\":\"qwen3-audio-frontend-conv\","
         << "\"device\":\""
         << (device_ == mx::Device::gpu ? "gpu" : "cpu") << "\","
         << "\"pcm_samples\":" << audio_len << ",\"feature_length\":"
         << features.shape(2) << ",\"chunk_lengths\":[";
    for (std::size_t index = 0; index < chunk_lengths.size(); ++index) {
      if (index != 0) {
        json << ",";
      }
      json << chunk_lengths[index];
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
    json << "],\"elapsed_ms\":" << elapsed << ",\"peak_memory_bytes\":"
         << mx::get_peak_memory() << ",\"fingerprints\":{\"input_features\":"
         << fingerprint_json(features, device_)
         << ",\"conv2d1\":" << conv2d1_fingerprint
         << ",\"conv2d2\":" << conv2d2_fingerprint
         << ",\"conv2d3\":" << conv2d3_fingerprint
         << ",\"conv_out\":" << conv_out_fingerprint << "}}";
    return json.str();
  }

private:
  const mx::array &weight(const std::string &name) const {
    const auto found = weights_.find(name);
    if (found == weights_.end()) {
      throw std::runtime_error("missing required tensor: " + name);
    }
    return found->second;
  }

  mx::array gelu(const mx::array &input) const {
    const auto one = mx::array(1.0f, input.dtype());
    const auto inverse_sqrt_two =
        mx::array(1.0f / std::sqrt(2.0f), input.dtype());
    const auto half = mx::array(0.5f, input.dtype());
    return mx::multiply(
        input,
        mx::multiply(
            one + mx::erf(mx::multiply(input, inverse_sqrt_two, device_),
                          device_),
            half, device_),
        device_);
  }

  mx::array log_mel(const float *audio, std::size_t audio_len) const {
    if (audio_len < kFftSize) {
      throw std::runtime_error(
          "audio must contain at least 400 mono float32 samples");
    }
    std::vector<float> padded_audio(audio_len + kFftSize, 0.0f);
    std::copy_n(audio, audio_len, padded_audio.begin() + kFftSize / 2);
    for (std::size_t index = 0; index < kFftSize / 2; ++index) {
      padded_audio[index] = padded_audio[kFftSize - index];
      padded_audio[kFftSize / 2 + audio_len + index] =
          padded_audio[kFftSize / 2 + audio_len - 2 - index];
    }

    std::vector<float> window(kFftSize);
    constexpr auto pi = 3.14159265358979323846;
    for (std::size_t index = 0; index < kFftSize; ++index) {
      window[index] = static_cast<float>(
          0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(index) /
                               static_cast<double>(kFftSize)));
    }

    const auto feature_frames = audio_len / kHopLength;
    auto audio_array = mx::array(
        padded_audio.begin(), {static_cast<int>(padded_audio.size())});
    auto frames = mx::as_strided(
        audio_array,
        {static_cast<int>(feature_frames + 1), static_cast<int>(kFftSize)},
        {static_cast<int64_t>(kHopLength), 1}, 0, device_);
    const auto window_array =
        mx::array(window.begin(), {static_cast<int>(window.size())});
    auto frequencies = mx::fft::rfft(frames * window_array, -1,
                                     mx::fft::FFTNorm::Backward, device_);
    frequencies = mx::slice(
        frequencies, {0, 0},
        {static_cast<int>(feature_frames), static_cast<int>(kFrequencyBins)},
        {1, 1}, device_);
    const auto magnitudes =
        mx::square(mx::abs(frequencies, device_), device_);
    const auto mel_filter_array =
        mx::array(mel_filters_.begin(),
                  {static_cast<int>(kFrequencyBins),
                   static_cast<int>(kMelBins)});
    auto log_spec =
        mx::log10(mx::maximum(mx::matmul(magnitudes, mel_filter_array, device_),
                              mx::array(1e-10f), device_),
                  device_);
    log_spec = mx::maximum(log_spec, mx::max(log_spec, device_) - 8.0, device_);
    log_spec = (log_spec + 4.0) / 4.0;
    return mx::expand_dims(mx::transpose(log_spec, {1, 0}, device_), 0,
                           device_);
  }

  mx::array make_chunks(const mx::array &features,
                        std::vector<std::size_t> &chunk_lengths) const {
    const auto feature_length = static_cast<std::size_t>(features.shape(2));
    std::vector<mx::array> chunks;
    for (std::size_t position = 0; position < feature_length;
         position += kConvChunkFrames) {
      const auto chunk_length =
          std::min(kConvChunkFrames, feature_length - position);
      auto chunk = mx::slice(
          features, {0, 0, static_cast<int>(position)},
          {1, static_cast<int>(kMelBins),
           static_cast<int>(position + chunk_length)},
          {1, 1, 1}, device_);
      chunk = mx::squeeze(chunk, 0, device_);
      if (chunk_length < kConvChunkFrames) {
        const auto padding =
            mx::zeros({static_cast<int>(kMelBins),
                       static_cast<int>(kConvChunkFrames - chunk_length)},
                      mx::float32, device_);
        chunk = mx::concatenate({chunk, padding}, 1, device_);
      }
      chunks.push_back(std::move(chunk));
      chunk_lengths.push_back(chunk_length);
    }
    return mx::stack(chunks, 0, device_);
  }

  mx::Device device_;
  std::unordered_map<std::string, mx::array> weights_;
  std::vector<float> mel_filters_;
};

mx::Device requested_device(int32_t device_kind) {
  if (device_kind == 0) {
    return mx::Device::cpu;
  }
  if (device_kind == 1) {
    return mx::Device::gpu;
  }
  throw std::runtime_error("device_kind must be 0 (CPU) or 1 (GPU)");
}

} // namespace

extern "C" int32_t
cuttledoc_qwen3_mlx_inspect_model(const char *model_directory, char **json_out,
                                  char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || json_out == nullptr) {
    return fail("model_directory and json_out must be non-null", error_out);
  }

  try {
    const std::lock_guard lock(runtime_mutex);
    const auto json = inspect_model(model_directory);
    *json_out = strdup(json.c_str());
    if (*json_out == nullptr) {
      return fail("could not allocate JSON result", error_out);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error.what(), error_out);
  }
}

extern "C" int32_t cuttledoc_qwen3_mlx_probe_audio_frontend(
    const char *model_directory, const float *audio, std::size_t audio_len,
    int32_t device_kind, char **json_out, char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || audio == nullptr || json_out == nullptr) {
    return fail("model_directory, audio, and json_out must be non-null",
                error_out);
  }

  try {
    const std::lock_guard lock(runtime_mutex);
    const auto device = requested_device(device_kind);
    if (!mx::is_available(device)) {
      return fail("requested MLX device is not available", error_out);
    }
    mx::set_default_device(device);
    mx::clear_cache();
    mx::reset_peak_memory();
    const Qwen3AudioFrontend frontend(model_directory, device);
    const auto json = frontend.probe(audio, audio_len);
    *json_out = strdup(json.c_str());
    if (*json_out == nullptr) {
      return fail("could not allocate JSON result", error_out);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("MLX raised a non-standard exception in the audio frontend",
                error_out);
  }
}

extern "C" void cuttledoc_qwen3_mlx_free_string(char *value) {
  std::free(value);
}
