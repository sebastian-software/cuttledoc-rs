#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuttledoc_voxtral_mlx_shim.h"
#include "cuttledoc_mlx_transformer.h"
#include "mlx/io.h"
#include "mlx/mlx.h"
#include <json.hpp>

namespace mx = mlx::core;
namespace mx_support = cuttledoc::mlx_support;
using json = nlohmann::json;

namespace {

constexpr std::uintmax_t kSafetensorsBytes = 3'133'798'126;
constexpr std::uintmax_t kIndexBytes = 118'632;
constexpr std::uintmax_t kConfigBytes = 1'513;
constexpr std::uintmax_t kTekkenBytes = 14'910'348;
constexpr std::size_t kExpectedTensorCount = 1'523;
constexpr std::size_t kExpectedFloat32Count = 300;
constexpr std::size_t kExpectedFloat16Count = 817;
constexpr std::size_t kExpectedUint32Count = 406;
constexpr std::size_t kExpectedQuantizedModuleCount = 406;
constexpr std::size_t kSampleRate = 16'000;
constexpr std::size_t kFftSize = 400;
constexpr std::size_t kHopLength = 160;
constexpr std::size_t kFrequencyBins = kFftSize / 2 + 1;
constexpr std::size_t kMelBins = 128;
constexpr std::size_t kRawAudioSamplesPerToken = 1'280;
constexpr std::size_t kLeftPadTokens = 32;
constexpr std::size_t kEncoderDimension = 1'280;
constexpr std::size_t kEncoderLayers = 32;
constexpr std::size_t kEncoderAttentionHeads = 32;
constexpr std::size_t kEncoderHeadDimension = 64;
constexpr std::size_t kEncoderSlidingWindow = 750;
constexpr std::size_t kDownsampleFactor = 4;
constexpr std::size_t kDecoderDimension = 3'072;
constexpr std::size_t kDecoderLayers = 26;
constexpr std::size_t kDecoderAttentionHeads = 32;
constexpr std::size_t kDecoderKeyValueHeads = 8;
constexpr std::size_t kDecoderHeadDimension = 128;
constexpr std::size_t kDecoderHiddenDimension = 9'216;
constexpr std::size_t kDecoderSlidingWindow = 8'192;
constexpr std::size_t kDecoderVocabularySize = 131'072;
constexpr std::size_t kAdaBottleneckDimension = 32;
constexpr int32_t kBosToken = 1;
constexpr int32_t kEosToken = 2;
constexpr int32_t kStreamingPadToken = 32;
constexpr std::size_t kTekkenSpecialTokens = 1'000;
constexpr int kQuantizationGroupSize = 64;
constexpr int kQuantizationBits = 4;
constexpr float kRmsNormEpsilon = 1e-5f;
constexpr float kRopeTheta = 1'000'000.0f;

std::mutex runtime_mutex;
std::optional<mx::Stream> runtime_cpu_stream;
std::optional<mx::Stream> runtime_gpu_stream;

int32_t fail_with_status(int32_t status, const std::string &message,
                         char **error_out) {
  if (error_out != nullptr) {
    *error_out = strdup(message.c_str());
  }
  return status;
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
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

std::string decode_base64(std::string_view encoded) {
  static constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::array<int, 256> values{};
  values.fill(-1);
  for (std::size_t index = 0; index < alphabet.size(); ++index) {
    values[static_cast<unsigned char>(alphabet[index])] =
        static_cast<int>(index);
  }

  std::string decoded;
  decoded.reserve(encoded.size() * 3 / 4);
  std::uint32_t accumulator = 0;
  int available_bits = -8;
  for (const auto character : encoded) {
    if (character == '=') {
      break;
    }
    const auto value = values[static_cast<unsigned char>(character)];
    if (value < 0) {
      throw std::runtime_error("invalid base64 in Tekken vocabulary");
    }
    accumulator = (accumulator << 6) | static_cast<std::uint32_t>(value);
    available_bits += 6;
    if (available_bits >= 0) {
      decoded.push_back(static_cast<char>(
          (accumulator >> available_bits) & static_cast<std::uint32_t>(0xff)));
      available_bits -= 8;
    }
  }
  return decoded;
}

std::string trim_ascii_whitespace(std::string value) {
  const auto is_whitespace = [](unsigned char character) {
    return character == ' ' || character == '\t' || character == '\n' ||
           character == '\r' || character == '\f' || character == '\v';
  };
  const auto first = std::find_if_not(value.begin(), value.end(), is_whitespace);
  const auto last = std::find_if_not(value.rbegin(), value.rend(), is_whitespace)
                        .base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

class TekkenTokenizer {
public:
  explicit TekkenTokenizer(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
      throw std::runtime_error("could not open Tekken vocabulary: " +
                               path.string());
    }
    const auto document = json::parse(input);
    const auto &config = document.at("config");
    const auto n_special =
        config.value("default_num_special_tokens", kTekkenSpecialTokens);
    const auto vocabulary_size =
        config.value("default_vocab_size", kDecoderVocabularySize);
    if (n_special != kTekkenSpecialTokens ||
        vocabulary_size != kDecoderVocabularySize) {
      throw std::runtime_error("Tekken vocabulary configuration drifted");
    }
    std::unordered_set<std::size_t> special_ids;
    for (const auto &special : document.at("special_tokens")) {
      special_ids.insert(special.at("rank").get<std::size_t>());
    }
    if (special_ids.size() != kTekkenSpecialTokens ||
        *std::max_element(special_ids.begin(), special_ids.end()) >=
            kTekkenSpecialTokens) {
      throw std::runtime_error("Tekken special-token layout drifted");
    }

    const auto regular_count = kDecoderVocabularySize - kTekkenSpecialTokens;
    const auto &vocabulary = document.at("vocab");
    if (vocabulary.size() < regular_count) {
      throw std::runtime_error("Tekken vocabulary is shorter than model vocab");
    }
    token_bytes_.reserve(regular_count);
    for (std::size_t index = 0; index < regular_count; ++index) {
      token_bytes_.push_back(decode_base64(
          vocabulary.at(index).at("token_bytes").get<std::string>()));
    }
  }

  std::string decode(const std::vector<int32_t> &token_ids) const {
    std::string output;
    for (const auto token_id : token_ids) {
      if (token_id < static_cast<int32_t>(kTekkenSpecialTokens) ||
          token_id >= static_cast<int32_t>(kDecoderVocabularySize)) {
        continue;
      }
      output += token_bytes_.at(
          static_cast<std::size_t>(token_id) - kTekkenSpecialTokens);
    }
    return output;
  }

private:
  std::vector<std::string> token_bytes_;
};

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
    throw std::runtime_error(name + " has shape " +
                             shape_string(found->second) +
                             ", expected a pinned architecture shape");
  }
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
    filter_frequencies[index] = slaney_mel_to_hertz(
        minimum_mel + fraction * (maximum_mel - minimum_mel));
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
  std::string source_dtype = "other";
  if (value.dtype() == mx::float32) {
    source_dtype = "float32";
  } else if (value.dtype() == mx::float16) {
    source_dtype = "float16";
  } else if (value.dtype() == mx::bfloat16) {
    source_dtype = "bfloat16";
  }
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
  json << std::setprecision(17) << "{\"shape\":"
       << shape_string(materialized) << ",\"source_dtype\":\"" << source_dtype
       << "\",\"mean\":" << mean << ",\"stddev\":"
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

struct LoadedModel {
  std::unordered_map<std::string, mx::array> weights;
  std::size_t float32_count{0};
  std::size_t float16_count{0};
  std::size_t uint32_count{0};
  std::size_t quantized_module_count{0};
};

LoadedModel load_and_validate_model(
    const std::filesystem::path &model_directory) {
  const auto config_path = model_directory / "config.json";
  const auto index_path = model_directory / "model.safetensors.index.json";
  const auto safetensors_path = model_directory / "model.safetensors";
  const auto tekken_path = model_directory / "tekken.json";
  for (const auto &path :
       {config_path, index_path, safetensors_path, tekken_path}) {
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("missing model artifact: " + path.string());
    }
  }
  if (std::filesystem::file_size(config_path) != kConfigBytes ||
      std::filesystem::file_size(index_path) != kIndexBytes ||
      std::filesystem::file_size(safetensors_path) != kSafetensorsBytes ||
      std::filesystem::file_size(tekken_path) != kTekkenBytes) {
    throw std::runtime_error("Voxtral artifact byte count does not match pin");
  }

  auto [weights, metadata] =
      mx::load_safetensors(safetensors_path.string(), mx::Device::cpu);
  (void)metadata;
  if (weights.size() != kExpectedTensorCount) {
    std::ostringstream message;
    message << "model has " << weights.size() << " tensors, expected "
            << kExpectedTensorCount;
    throw std::runtime_error(message.str());
  }

  LoadedModel loaded{std::move(weights)};
  for (const auto &[name, value] : loaded.weights) {
    if (value.dtype() == mx::float32) {
      ++loaded.float32_count;
    } else if (value.dtype() == mx::float16) {
      ++loaded.float16_count;
    } else if (value.dtype() == mx::uint32) {
      ++loaded.uint32_count;
    } else {
      throw std::runtime_error("unexpected dtype for tensor: " + name);
    }
    if (ends_with(name, ".weight")) {
      const auto prefix = name.substr(0, name.size() - 7);
      if (loaded.weights.contains(prefix + ".scales") &&
          loaded.weights.contains(prefix + ".biases")) {
        ++loaded.quantized_module_count;
      }
    }
  }
  if (loaded.float32_count != kExpectedFloat32Count ||
      loaded.float16_count != kExpectedFloat16Count ||
      loaded.uint32_count != kExpectedUint32Count ||
      loaded.quantized_module_count != kExpectedQuantizedModuleCount) {
    throw std::runtime_error("Voxtral dtype or affine 4-bit layout drifted");
  }

  expect_shape(loaded.weights, "encoder.conv_layers_0_conv.conv.weight",
               {1280, 3, 128});
  expect_shape(loaded.weights, "encoder.conv_layers_0_conv.conv.bias", {1280});
  expect_shape(loaded.weights, "encoder.conv_layers_1_conv.conv.weight",
               {1280, 3, 1280});
  expect_shape(loaded.weights, "encoder.conv_layers_1_conv.conv.bias", {1280});
  expect_shape(loaded.weights,
               "encoder.transformer_layers.0.attention.wq.weight",
               {2048, 160});
  expect_shape(loaded.weights,
               "encoder.transformer_layers.31.attention.wq.weight",
               {2048, 160});
  expect_shape(loaded.weights, "encoder.audio_language_projection_0.weight",
               {3072, 5120});
  expect_shape(loaded.weights, "encoder.audio_language_projection_2.weight",
               {3072, 3072});
  expect_shape(loaded.weights, "decoder.tok_embeddings.weight",
               {131072, 3072});
  expect_shape(loaded.weights, "decoder.layers.0.attention.wq.weight",
               {4096, 384});
  expect_shape(loaded.weights, "decoder.layers.25.attention.wq.weight",
               {4096, 384});
  expect_shape(
      loaded.weights,
      "decoder.layers.0.ada_rms_norm_t_cond.ada_down.weight", {32, 3072});
  expect_shape(loaded.weights,
               "decoder.layers.25.ada_rms_norm_t_cond.ada_up.weight",
               {3072, 32});
  expect_shape(loaded.weights, "decoder.norm.weight", {3072});
  return loaded;
}

std::string model_json(const LoadedModel &model) {
  std::ostringstream json;
  json << "{\"boundary\":\"repository-owned-official-mlx\","
       << "\"model\":\"voxtral-mini-4b-realtime-2602-mlx-4bit\","
       << "\"model_bytes\":" << kSafetensorsBytes << ","
       << "\"tensor_count\":" << model.weights.size() << ","
       << "\"dtype_counts\":{\"float32\":" << model.float32_count
       << ",\"float16\":" << model.float16_count << ",\"uint32\":"
       << model.uint32_count << "},\"affine_4bit_modules\":"
       << model.quantized_module_count
       << ",\"capabilities\":{\"bounded_ingestion\":true,"
       << "\"cancellation\":true,\"mel_frontend\":true,"
       << "\"causal_conv_stem\":true,\"causal_encoder\":true,"
       << "\"rotating_kv_cache\":true,"
       << "\"sliding_window_attention\":true,"
       << "\"adapter_projection\":true,\"delay_conditioning\":true,"
       << "\"decoder\":true,\"decoder_kv_cache\":true,"
       << "\"tekken_decode\":true,\"greedy_transcription\":true,"
       << "\"transcription\":true,\"streaming_session\":true}}";
  return json.str();
}

mx::array weight_on_device(const LoadedModel &model, const std::string &name,
                           mx::Device device) {
  const auto found = model.weights.find(name);
  if (found == model.weights.end()) {
    throw std::runtime_error("missing required tensor: " + name);
  }
  return device == mx::Device::gpu ? mx::copy(found->second, device)
                                   : found->second;
}

void materialize_weight_prefix_on_device(LoadedModel &model,
                                         std::string_view prefix,
                                         mx::Device device) {
  if (device == mx::Device::cpu) {
    return;
  }
  std::vector<mx::array> arrays;
  for (auto &[name, value] : model.weights) {
    if (name.starts_with(prefix)) {
      value = mx::copy(std::move(value), device);
      arrays.push_back(value);
    }
  }
  mx::eval(std::move(arrays));
}

std::size_t audio_token_count(std::size_t sample_count) {
  const auto mel_frames =
      sample_count % kHopLength == 0
          ? sample_count / kHopLength
          : static_cast<std::size_t>(std::ceil(
                static_cast<double>(sample_count) /
                    static_cast<double>(kHopLength) -
                1.0));
  return (mel_frames + 7) / 8;
}

mx::array precise_gelu(const mx::array &input, mx::Device device) {
  const auto one = mx::array(1.0f, input.dtype());
  const auto inverse_sqrt_two =
      mx::array(1.0f / std::sqrt(2.0f), input.dtype());
  const auto half = mx::array(0.5f, input.dtype());
  return mx::multiply(
      input,
      mx::multiply(one + mx::erf(mx::multiply(input, inverse_sqrt_two, device),
                                 device),
                   half, device),
      device);
}

struct FrontendPadding {
  std::size_t delay_tokens;
  std::size_t left_pad_samples;
  std::size_t alignment_pad_samples;
  std::size_t right_pad_tokens;
  std::size_t right_pad_samples;
  std::size_t padded_samples;
};

FrontendPadding frontend_padding(std::size_t audio_len,
                                 int32_t transcription_delay_ms) {
  const auto delay_samples = static_cast<std::size_t>(
      static_cast<double>(transcription_delay_ms) / 1000.0 *
      static_cast<double>(kSampleRate));
  const auto delay_tokens = audio_token_count(delay_samples);
  const auto alignment_pad_samples =
      (kRawAudioSamplesPerToken - audio_len % kRawAudioSamplesPerToken) %
      kRawAudioSamplesPerToken;
  const auto left_pad_samples = kLeftPadTokens * kRawAudioSamplesPerToken;
  const auto right_pad_tokens = delay_tokens + 1 + 10;
  const auto right_pad_samples =
      alignment_pad_samples + right_pad_tokens * kRawAudioSamplesPerToken;
  return {
      delay_tokens,
      left_pad_samples,
      alignment_pad_samples,
      right_pad_tokens,
      right_pad_samples,
      left_pad_samples + audio_len + right_pad_samples,
  };
}

struct FrontendComputation {
  FrontendPadding padding;
  mx::array mel_filters;
  mx::array log_mel;
  mx::array conv0;
  mx::array conv1_pretrunc;
  mx::array conv_stem;
  std::size_t front_truncation;
};

FrontendComputation compute_audio_frontend(
    const LoadedModel &model, const float *audio, std::size_t audio_len,
    int32_t transcription_delay_ms, mx::Device device) {
  if (audio_len < kFftSize) {
    throw std::invalid_argument(
        "audio must contain at least 400 mono float32 samples");
  }
  if (transcription_delay_ms <= 0) {
    throw std::invalid_argument("transcription delay must be positive");
  }
  const auto padding = frontend_padding(audio_len, transcription_delay_ms);

  std::vector<float> streaming_audio(padding.padded_samples, 0.0f);
  std::copy_n(audio, audio_len,
              streaming_audio.begin() +
                  static_cast<std::ptrdiff_t>(padding.left_pad_samples));

  std::vector<float> reflected_audio(padding.padded_samples + kFftSize, 0.0f);
  std::copy(streaming_audio.begin(), streaming_audio.end(),
            reflected_audio.begin() + static_cast<std::ptrdiff_t>(kFftSize / 2));
  for (std::size_t index = 0; index < kFftSize / 2; ++index) {
    reflected_audio[index] = reflected_audio[kFftSize - index];
    reflected_audio[kFftSize / 2 + padding.padded_samples + index] =
        reflected_audio[kFftSize / 2 + padding.padded_samples - 2 - index];
  }

  std::vector<float> window(kFftSize);
  constexpr auto pi = 3.14159265358979323846;
  for (std::size_t index = 0; index < kFftSize; ++index) {
    window[index] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(index) /
                             static_cast<double>(kFftSize)));
  }

  const auto mel_frames = padding.padded_samples / kHopLength;
  auto audio_array = mx::array(
      reflected_audio.begin(), {static_cast<int>(reflected_audio.size())});
  auto frames = mx::as_strided(
      audio_array,
      {static_cast<int>(mel_frames + 1), static_cast<int>(kFftSize)},
      {static_cast<int64_t>(kHopLength), 1}, 0, device);
  const auto window_array =
      mx::array(window.begin(), {static_cast<int>(window.size())});
  auto frequencies = mx::fft::rfft(frames * window_array, -1,
                                    mx::fft::FFTNorm::Backward, device);
  frequencies = mx::slice(
      frequencies, {0, 0},
      {static_cast<int>(mel_frames), static_cast<int>(kFrequencyBins)},
      {1, 1}, device);
  const auto magnitudes =
      mx::square(mx::abs(frequencies, device), device);
  const auto mel_filters = make_slaney_mel_filters();
  auto mel_filter_array =
      mx::array(mel_filters.begin(),
                {static_cast<int>(kFrequencyBins), static_cast<int>(kMelBins)});
  auto log_mel = mx::log10(
      mx::maximum(mx::matmul(magnitudes, mel_filter_array, device),
                  mx::array(1e-10f), device),
      device);
  log_mel = mx::maximum(log_mel, mx::array(-6.5f), device);
  log_mel = (log_mel + 4.0) / 4.0;
  log_mel = mx::transpose(log_mel, {1, 0}, device);
  if (log_mel.shape(1) % 2 != 0) {
    log_mel = mx::slice(
        log_mel, {0, 1},
        {static_cast<int>(kMelBins), log_mel.shape(1)}, {1, 1}, device);
  }

  auto conv0_input = mx::expand_dims(mx::transpose(log_mel, {1, 0}, device),
                                     0, device);
  conv0_input = mx::pad(
      conv0_input, std::vector<std::pair<int, int>>{{0, 0}, {2, 0}, {0, 0}},
      mx::array(0.0f, conv0_input.dtype()), "constant", device);
  auto conv0 = precise_gelu(
      mx::conv1d(
          conv0_input,
          weight_on_device(model, "encoder.conv_layers_0_conv.conv.weight",
                           device),
          1, 0, 1, 1, device) +
          weight_on_device(model, "encoder.conv_layers_0_conv.conv.bias",
                           device),
      device);

  auto conv1_input = mx::pad(
      conv0, std::vector<std::pair<int, int>>{{0, 0}, {1, 0}, {0, 0}},
      mx::array(0.0f, conv0.dtype()), "constant", device);
  auto conv1_pretrunc = mx::squeeze(
      precise_gelu(
          mx::conv1d(
              conv1_input,
              weight_on_device(model,
                               "encoder.conv_layers_1_conv.conv.weight",
                               device),
              2, 0, 1, 1, device) +
              weight_on_device(model, "encoder.conv_layers_1_conv.conv.bias",
                               device),
          device),
      0, device);
  const auto front_truncation =
      static_cast<std::size_t>(conv1_pretrunc.shape(0)) % kDownsampleFactor;
  auto conv_stem = conv1_pretrunc;
  if (front_truncation > 0) {
    conv_stem = mx::slice(
        conv1_pretrunc,
        {static_cast<int>(front_truncation), 0},
        {conv1_pretrunc.shape(0), static_cast<int>(kEncoderDimension)}, {1, 1},
        device);
  }

  return {padding,
          std::move(mel_filter_array),
          std::move(log_mel),
          std::move(conv0),
          std::move(conv1_pretrunc),
          std::move(conv_stem),
          front_truncation};
}

std::string probe_audio_frontend(const LoadedModel &model, const float *audio,
                                 std::size_t audio_len,
                                 int32_t transcription_delay_ms,
                                 mx::Device device) {
  const auto started = std::chrono::steady_clock::now();
  const auto frontend = compute_audio_frontend(
      model, audio, audio_len, transcription_delay_ms, device);
  const auto mel_filter_fingerprint =
      fingerprint_json(frontend.mel_filters, device);
  const auto log_mel_fingerprint =
      fingerprint_json(frontend.log_mel, device);
  const auto conv0_fingerprint = fingerprint_json(frontend.conv0, device);
  const auto conv1_pretrunc_fingerprint =
      fingerprint_json(frontend.conv1_pretrunc, device);
  const auto conv_stem_fingerprint =
      fingerprint_json(frontend.conv_stem, device);
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  std::ostringstream json;
  json << std::setprecision(17)
       << "{\"status\":\"ok\",\"boundary\":\"official-mlx-cpp\","
       << "\"stage\":\"voxtral-audio-frontend\",\"device\":\""
       << (device == mx::Device::gpu ? "gpu" : "cpu")
       << "\",\"pcm_samples\":" << audio_len
       << ",\"padding\":{\"transcription_delay_ms\":"
       << transcription_delay_ms << ",\"delay_tokens\":"
       << frontend.padding.delay_tokens << ",\"left_pad_tokens\":"
       << kLeftPadTokens << ",\"left_pad_samples\":"
       << frontend.padding.left_pad_samples
       << ",\"alignment_pad_samples\":"
       << frontend.padding.alignment_pad_samples
       << ",\"right_pad_tokens\":" << frontend.padding.right_pad_tokens
       << ",\"right_pad_samples\":" << frontend.padding.right_pad_samples
       << ",\"padded_samples\":" << frontend.padding.padded_samples
       << "},\"mel_frames\":" << frontend.log_mel.shape(1)
       << ",\"front_truncation_frames\":" << frontend.front_truncation
       << ",\"elapsed_ms\":" << elapsed_ms << ",\"peak_memory_bytes\":"
       << mx::get_peak_memory() << ",\"fingerprints\":{\"mel_filters\":"
       << mel_filter_fingerprint << ",\"log_mel\":" << log_mel_fingerprint
       << ",\"conv0_gelu\":" << conv0_fingerprint
       << ",\"conv1_pretrunc_gelu\":" << conv1_pretrunc_fingerprint
       << ",\"conv_stem\":" << conv_stem_fingerprint
       << "},\"capabilities\":{\"mel_frontend\":true,"
       << "\"causal_conv_stem\":true,\"causal_encoder\":false,"
       << "\"transcription\":false}}";
  return json.str();
}

struct EncoderChunkEvidence {
  std::size_t index;
  std::size_t start;
  std::size_t length;
  std::string mask_kind;
  mx::Shape mask_shape;
};

struct EncoderEvidence {
  std::vector<EncoderChunkEvidence> chunks;
  std::unordered_map<std::string, std::string> fingerprints;
  std::size_t cache_offset{0};
  std::size_t cache_size{0};
  std::size_t cache_materialized_frames{0};
};

struct EncoderResult {
  mx::array encoded;
  mx::array projection0;
  mx::array adapter;
};

class VoxtralCausalEncoder {
public:
  VoxtralCausalEncoder(const LoadedModel &model, mx::Device device)
      : model_(model), device_(device) {}

  std::vector<mx_support::RotatingKeyValueCache> make_cache() const {
    std::vector<mx_support::RotatingKeyValueCache> caches;
    caches.reserve(kEncoderLayers);
    for (std::size_t layer = 0; layer < kEncoderLayers; ++layer) {
      caches.emplace_back(kEncoderSlidingWindow, 0);
    }
    return caches;
  }

  mx::array encode_chunk(
      const mx::array &conv_chunk, std::size_t start_position,
      std::vector<mx_support::RotatingKeyValueCache> &caches) const {
    if (conv_chunk.ndim() != 2 ||
        conv_chunk.shape(1) != static_cast<int>(kEncoderDimension) ||
        conv_chunk.shape(0) == 0 || caches.size() != kEncoderLayers) {
      throw std::invalid_argument(
          "streaming encoder requires non-empty [frames, 1280] input and 32 caches");
    }
    const auto chunk_length =
        static_cast<std::size_t>(conv_chunk.shape(0));
    const auto mask = caches.front().make_mask(
        chunk_length, kEncoderSlidingWindow, device_, true);
    auto hidden = conv_chunk;
    for (std::size_t layer = 0; layer < kEncoderLayers; ++layer) {
      hidden = encoder_layer(hidden, start_position, layer, mask,
                             caches[layer]);
    }
    return rms_norm(hidden, "encoder.transformer_norm.weight");
  }

  mx::array project_aligned(const mx::array &encoded) const {
    if (encoded.ndim() != 2 ||
        encoded.shape(1) != static_cast<int>(kEncoderDimension) ||
        encoded.shape(0) == 0 ||
        encoded.shape(0) % static_cast<int>(kDownsampleFactor) != 0) {
      throw std::invalid_argument(
          "adapter projection requires aligned [frames, 1280] encoder output");
    }
    const auto adapter_frames =
        static_cast<std::size_t>(encoded.shape(0)) / kDownsampleFactor;
    auto downsampled = mx::reshape(
        encoded,
        {static_cast<int>(adapter_frames),
         static_cast<int>(kEncoderDimension * kDownsampleFactor)},
        device_);
    auto projection0 = precise_gelu(
        linear(downsampled, "encoder.audio_language_projection_0.weight"),
        device_);
    return linear(projection0,
                  "encoder.audio_language_projection_2.weight");
  }

  EncoderResult encode(const mx::array &conv_stem,
                       EncoderEvidence *evidence = nullptr) const {
    if (conv_stem.ndim() != 2 ||
        conv_stem.shape(1) != static_cast<int>(kEncoderDimension) ||
        conv_stem.shape(0) % static_cast<int>(kDownsampleFactor) != 0) {
      throw std::invalid_argument(
          "encoder input must be aligned [frames, 1280] conv output");
    }

    auto caches = make_cache();

    std::vector<mx::array> encoded_chunks;
    for (std::size_t chunk_start = 0;
         chunk_start < static_cast<std::size_t>(conv_stem.shape(0));
         chunk_start += kEncoderSlidingWindow) {
      const auto chunk_end = std::min(
          chunk_start + kEncoderSlidingWindow,
          static_cast<std::size_t>(conv_stem.shape(0)));
      const auto chunk_length = chunk_end - chunk_start;
      auto hidden = mx::slice(
          conv_stem, {static_cast<int>(chunk_start), 0},
          {static_cast<int>(chunk_end), static_cast<int>(kEncoderDimension)},
          {1, 1}, device_);
      const auto mask = caches.front().make_mask(
          chunk_length, kEncoderSlidingWindow, device_);
      const auto chunk_index = encoded_chunks.size();
      if (evidence != nullptr) {
        evidence->chunks.push_back(
            {chunk_index,
             chunk_start,
             chunk_length,
             mask.mode.empty() ? "array" : mask.mode,
             mask.array.has_value() ? mask.array->shape() : mx::Shape{}});
      }
      for (std::size_t layer = 0; layer < kEncoderLayers; ++layer) {
        hidden = encoder_layer(hidden, chunk_start, layer, mask, caches[layer]);
        if (evidence != nullptr &&
            (layer == 0 || layer == 15 || layer == 31)) {
          const auto name = "chunk_" + std::to_string(chunk_index) +
                            "_layer_" + std::to_string(layer);
          evidence->fingerprints.emplace(name,
                                         fingerprint_json(hidden, device_));
        }
      }
      hidden = rms_norm(hidden, "encoder.transformer_norm.weight");
      if (evidence != nullptr) {
        evidence->fingerprints.emplace(
            "chunk_" + std::to_string(chunk_index) + "_norm",
            fingerprint_json(hidden, device_));
      }
      encoded_chunks.push_back(std::move(hidden));
    }

    if (evidence != nullptr) {
      evidence->cache_offset = caches.front().offset();
      evidence->cache_size = caches.front().size();
      evidence->cache_materialized_frames =
          caches.front().materialized_size();
      evidence->fingerprints.emplace(
          "layer_0_cache_keys",
          fingerprint_json(caches.front().keys(), device_));
      evidence->fingerprints.emplace(
          "layer_0_cache_values",
          fingerprint_json(caches.front().values(), device_));
    }

    auto encoded = encoded_chunks.size() == 1
                       ? encoded_chunks.front()
                       : mx::concatenate(std::move(encoded_chunks), 0, device_);
    const auto adapter_frames =
        static_cast<std::size_t>(encoded.shape(0)) / kDownsampleFactor;
    auto downsampled = mx::reshape(
        encoded,
        {static_cast<int>(adapter_frames),
         static_cast<int>(kEncoderDimension * kDownsampleFactor)},
        device_);
    auto projection0 = precise_gelu(
        linear(downsampled, "encoder.audio_language_projection_0.weight"),
        device_);
    auto adapter =
        linear(projection0, "encoder.audio_language_projection_2.weight");
    if (evidence != nullptr) {
      evidence->fingerprints.emplace("encoded",
                                     fingerprint_json(encoded, device_));
      evidence->fingerprints.emplace(
          "adapter_projection0_gelu",
          fingerprint_json(projection0, device_));
      evidence->fingerprints.emplace("adapter",
                                     fingerprint_json(adapter, device_));
    }
    return {std::move(encoded), std::move(projection0), std::move(adapter)};
  }

private:
  const mx::array &weight(const std::string &name) const {
    const auto found = model_.weights.find(name);
    if (found == model_.weights.end()) {
      throw std::runtime_error("missing required tensor: " + name);
    }
    return found->second;
  }

  std::optional<mx::array> optional_weight(const std::string &name) const {
    const auto found = model_.weights.find(name);
    return found == model_.weights.end()
               ? std::nullopt
               : std::optional<mx::array>(found->second);
  }

  mx::array quantized_linear(const mx::array &input,
                             const std::string &prefix) const {
    return mx_support::affine_quantized_linear(
        input, weight(prefix + ".weight"), weight(prefix + ".scales"),
        weight(prefix + ".biases"), optional_weight(prefix + ".bias"),
        kQuantizationGroupSize, kQuantizationBits, device_);
  }

  mx::array linear(const mx::array &input,
                   const std::string &weight_name) const {
    return mx::matmul(input, mx::transpose(weight(weight_name), device_),
                      device_);
  }

  mx::array rms_norm(const mx::array &input,
                     const std::string &weight_name) const {
    return mx::fast::rms_norm(input, weight(weight_name), kRmsNormEpsilon,
                              device_);
  }

  mx::array encoder_attention(
      const mx::array &input, std::size_t rope_offset, std::size_t layer,
      const mx_support::AttentionMask &mask,
      mx_support::RotatingKeyValueCache &cache) const {
    const auto prefix = "encoder.transformer_layers." +
                        std::to_string(layer) + ".attention.";
    const auto sequence_length = input.shape(0);
    auto queries = quantized_linear(input, prefix + "wq");
    auto keys = quantized_linear(input, prefix + "wk");
    auto values = quantized_linear(input, prefix + "wv");
    queries = mx::transpose(
        mx::reshape(queries,
                    {1, sequence_length,
                     static_cast<int>(kEncoderAttentionHeads),
                     static_cast<int>(kEncoderHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    keys = mx::transpose(
        mx::reshape(keys,
                    {1, sequence_length,
                     static_cast<int>(kEncoderAttentionHeads),
                     static_cast<int>(kEncoderHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    values = mx::transpose(
        mx::reshape(values,
                    {1, sequence_length,
                     static_cast<int>(kEncoderAttentionHeads),
                     static_cast<int>(kEncoderHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    queries = mx::fast::rope(
        queries, static_cast<int>(kEncoderHeadDimension), true, kRopeTheta,
        1.0f, static_cast<int>(rope_offset), {}, device_);
    keys = mx::fast::rope(
        keys, static_cast<int>(kEncoderHeadDimension), true, kRopeTheta, 1.0f,
        static_cast<int>(rope_offset), {}, device_);
    auto [cached_keys, cached_values] =
        cache.update_and_fetch(keys, values, device_);
    auto output = mx::fast::scaled_dot_product_attention(
        queries, cached_keys, cached_values,
        1.0f / std::sqrt(static_cast<float>(kEncoderHeadDimension)),
        mask.mode, mask.array, {}, device_);
    output = mx::reshape(
        mx::transpose(output, {0, 2, 1, 3}, device_),
        {sequence_length,
         static_cast<int>(kEncoderAttentionHeads * kEncoderHeadDimension)},
        device_);
    return quantized_linear(output, prefix + "wo");
  }

  mx::array encoder_layer(
      const mx::array &input, std::size_t rope_offset, std::size_t layer,
      const mx_support::AttentionMask &mask,
      mx_support::RotatingKeyValueCache &cache) const {
    const auto prefix =
        "encoder.transformer_layers." + std::to_string(layer) + ".";
    auto hidden =
        input + encoder_attention(
                    rms_norm(input, prefix + "attention_norm.weight"),
                    rope_offset, layer, mask, cache);
    const auto normalized = rms_norm(hidden, prefix + "ffn_norm.weight");
    const auto gate =
        mx_support::silu(quantized_linear(normalized,
                                          prefix + "feed_forward_w1"),
                         device_);
    const auto up =
        quantized_linear(normalized, prefix + "feed_forward_w3");
    return hidden +
           quantized_linear(gate * up, prefix + "feed_forward_w2");
  }

  const LoadedModel &model_;
  mx::Device device_;
};

std::string probe_causal_encoder(LoadedModel model, const float *audio,
                                 std::size_t audio_len,
                                 int32_t transcription_delay_ms,
                                 mx::Device device) {
  const auto started = std::chrono::steady_clock::now();
  materialize_weight_prefix_on_device(model, "encoder.", device);
  const auto frontend = compute_audio_frontend(
      model, audio, audio_len, transcription_delay_ms, device);
  EncoderEvidence evidence;
  evidence.fingerprints.emplace(
      "conv_stem", fingerprint_json(frontend.conv_stem, device));
  const VoxtralCausalEncoder encoder(model, device);
  const auto result = encoder.encode(frontend.conv_stem, &evidence);
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  const std::vector<std::string> fingerprint_order{
      "conv_stem",
      "chunk_0_layer_0",
      "chunk_0_layer_15",
      "chunk_0_layer_31",
      "chunk_0_norm",
      "chunk_1_layer_0",
      "chunk_1_layer_15",
      "chunk_1_layer_31",
      "chunk_1_norm",
      "encoded",
      "adapter_projection0_gelu",
      "adapter",
      "layer_0_cache_keys",
      "layer_0_cache_values",
  };

  std::ostringstream json;
  json << std::setprecision(17)
       << "{\"status\":\"ok\",\"boundary\":\"official-mlx-cpp\","
       << "\"stage\":\"voxtral-causal-encoder\",\"device\":\""
       << (device == mx::Device::gpu ? "gpu" : "cpu")
       << "\",\"pcm_samples\":" << audio_len
       << ",\"transcription_delay_ms\":" << transcription_delay_ms
       << ",\"delay_tokens\":" << frontend.padding.delay_tokens
       << ",\"architecture\":{\"layers\":" << kEncoderLayers
       << ",\"dimension\":" << kEncoderDimension
       << ",\"attention_heads\":" << kEncoderAttentionHeads
       << ",\"head_dimension\":" << kEncoderHeadDimension
       << ",\"sliding_window\":" << kEncoderSlidingWindow
       << ",\"downsample_factor\":" << kDownsampleFactor
       << ",\"adapter_dimension\":" << kDecoderDimension
       << "},\"chunks\":[";
  for (std::size_t index = 0; index < evidence.chunks.size(); ++index) {
    if (index != 0) {
      json << ",";
    }
    const auto &chunk = evidence.chunks[index];
    json << "{\"index\":" << chunk.index << ",\"start\":" << chunk.start
         << ",\"length\":" << chunk.length << ",\"mask_kind\":\""
         << chunk.mask_kind << "\",\"mask_shape\":[";
    for (std::size_t dimension = 0; dimension < chunk.mask_shape.size();
         ++dimension) {
      if (dimension != 0) {
        json << ",";
      }
      json << chunk.mask_shape[dimension];
    }
    json << "]}";
  }
  json << "],\"cache\":{\"layer_0_offset\":" << evidence.cache_offset
       << ",\"layer_0_size\":" << evidence.cache_size
       << ",\"layer_0_materialized_key_frames\":"
       << evidence.cache_materialized_frames
       << ",\"layer_0_materialized_value_frames\":"
       << evidence.cache_materialized_frames
       << "},\"output\":{\"encoded_frames\":" << result.encoded.shape(0)
       << ",\"adapter_frames\":" << result.adapter.shape(0)
       << "},\"elapsed_ms\":" << elapsed_ms << ",\"peak_memory_bytes\":"
       << mx::get_peak_memory() << ",\"fingerprints\":{";
  for (std::size_t index = 0; index < fingerprint_order.size(); ++index) {
    if (index != 0) {
      json << ",";
    }
    const auto &name = fingerprint_order[index];
    const auto found = evidence.fingerprints.find(name);
    if (found == evidence.fingerprints.end()) {
      throw std::runtime_error("missing encoder fingerprint: " + name);
    }
    json << "\"" << name << "\":" << found->second;
  }
  json << "},\"capabilities\":{\"causal_encoder\":true,"
       << "\"rotating_kv_cache\":true,\"sliding_window_attention\":true,"
       << "\"adapter_projection\":true,\"decoder\":false,"
       << "\"transcription\":false}}";
  return json.str();
}

struct DecoderEvidence {
  std::unordered_map<std::string, std::string> fingerprints;
};

class VoxtralDecoder {
public:
  VoxtralDecoder(const LoadedModel &model, std::size_t delay_tokens,
                 mx::Device device)
      : model_(model), device_(device),
        time_embedding_(compute_time_embedding(delay_tokens)) {
    ada_scales_.reserve(kDecoderLayers);
    for (std::size_t layer = 0; layer < kDecoderLayers; ++layer) {
      const auto prefix = "decoder.layers." + std::to_string(layer) +
                          ".ada_rms_norm_t_cond.";
      auto hidden = precise_gelu(
          linear(time_embedding_, prefix + "ada_down.weight"), device_);
      ada_scales_.push_back(linear(hidden, prefix + "ada_up.weight"));
    }
    mx::eval(ada_scales_);
  }

  const mx::array &time_embedding() const { return time_embedding_; }

  const mx::array &ada_scale(std::size_t layer) const {
    return ada_scales_.at(layer);
  }

  mx::array embed_tokens(const std::vector<int32_t> &token_ids) const {
    if (token_ids.empty()) {
      throw std::invalid_argument("cannot embed an empty token sequence");
    }
    const auto ids = mx::array(
        token_ids.begin(), {static_cast<int>(token_ids.size())});
    return mx::take(weight("decoder.tok_embeddings.weight"), ids, 0,
                    device_);
  }

  mx::array embed_token(int32_t token_id) const {
    const auto ids = mx::array({token_id});
    return mx::squeeze(
        mx::take(weight("decoder.tok_embeddings.weight"), ids, 0, device_),
        0, device_);
  }

  std::vector<mx_support::RotatingKeyValueCache> make_cache() const {
    std::vector<mx_support::RotatingKeyValueCache> caches;
    caches.reserve(kDecoderLayers);
    for (std::size_t layer = 0; layer < kDecoderLayers; ++layer) {
      caches.emplace_back(kDecoderSlidingWindow, 0);
    }
    return caches;
  }

  mx::array forward(
      const mx::array &embeddings, std::size_t start_position,
      std::vector<mx_support::RotatingKeyValueCache> &caches,
      DecoderEvidence *evidence = nullptr,
      std::string_view evidence_prefix = {}) const {
    if (embeddings.ndim() != 2 ||
        embeddings.shape(1) != static_cast<int>(kDecoderDimension) ||
        caches.size() != kDecoderLayers) {
      throw std::invalid_argument(
          "decoder requires [tokens, 3072] embeddings and 26 caches");
    }
    auto hidden = embeddings;
    for (std::size_t layer = 0; layer < kDecoderLayers; ++layer) {
      hidden = decoder_layer(hidden, start_position, layer, caches[layer]);
      if (evidence != nullptr &&
          (layer == 0 || layer == 12 || layer == 25)) {
        evidence->fingerprints.emplace(
            std::string(evidence_prefix) + "_layer_" +
                std::to_string(layer),
            fingerprint_json(hidden, device_));
      }
    }
    hidden = rms_norm(hidden, "decoder.norm.weight");
    if (evidence != nullptr) {
      evidence->fingerprints.emplace(
          std::string(evidence_prefix) + "_norm",
          fingerprint_json(hidden, device_));
    }
    return hidden;
  }

  mx::array logits(const mx::array &hidden) const {
    return mx::matmul(
        hidden,
        mx::transpose(weight("decoder.tok_embeddings.weight"), device_),
        device_);
  }

private:
  const mx::array &weight(const std::string &name) const {
    const auto found = model_.weights.find(name);
    if (found == model_.weights.end()) {
      throw std::runtime_error("missing required tensor: " + name);
    }
    return found->second;
  }

  mx::array quantized_linear(const mx::array &input,
                             const std::string &prefix) const {
    return mx_support::affine_quantized_linear(
        input, weight(prefix + ".weight"), weight(prefix + ".scales"),
        weight(prefix + ".biases"), std::nullopt, kQuantizationGroupSize,
        kQuantizationBits, device_);
  }

  mx::array linear(const mx::array &input,
                   const std::string &weight_name) const {
    return mx::matmul(input, mx::transpose(weight(weight_name), device_),
                      device_);
  }

  mx::array rms_norm(const mx::array &input,
                     const std::string &weight_name) const {
    return mx::fast::rms_norm(input, weight(weight_name), kRmsNormEpsilon,
                              device_);
  }

  mx::array compute_time_embedding(std::size_t delay_tokens) const {
    constexpr auto half_dimension = kDecoderDimension / 2;
    auto inverse_frequency = mx::exp(
        (-static_cast<float>(std::log(10'000.0)) /
         static_cast<float>(half_dimension)) *
            mx::arange(static_cast<int>(half_dimension), mx::float32, device_),
        device_);
    const auto phase = static_cast<float>(delay_tokens) * inverse_frequency;
    return mx::concatenate(
        {mx::cos(phase, device_), mx::sin(phase, device_)}, 0, device_);
  }

  mx::array decoder_attention(
      const mx::array &input, std::size_t start_position, std::size_t layer,
      mx_support::RotatingKeyValueCache &cache) const {
    const auto prefix = "decoder.layers." + std::to_string(layer) +
                        ".attention.";
    const auto sequence_length = input.shape(0);
    auto queries = quantized_linear(input, prefix + "wq");
    auto keys = quantized_linear(input, prefix + "wk");
    auto values = quantized_linear(input, prefix + "wv");
    queries = mx::transpose(
        mx::reshape(queries,
                    {1, sequence_length,
                     static_cast<int>(kDecoderAttentionHeads),
                     static_cast<int>(kDecoderHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    keys = mx::transpose(
        mx::reshape(keys,
                    {1, sequence_length,
                     static_cast<int>(kDecoderKeyValueHeads),
                     static_cast<int>(kDecoderHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    values = mx::transpose(
        mx::reshape(values,
                    {1, sequence_length,
                     static_cast<int>(kDecoderKeyValueHeads),
                     static_cast<int>(kDecoderHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    queries = mx::fast::rope(
        queries, static_cast<int>(kDecoderHeadDimension), true, kRopeTheta,
        1.0f, static_cast<int>(start_position), {}, device_);
    keys = mx::fast::rope(
        keys, static_cast<int>(kDecoderHeadDimension), true, kRopeTheta, 1.0f,
        static_cast<int>(start_position), {}, device_);
    auto [cached_keys, cached_values] =
        cache.update_and_fetch(keys, values, device_);
    const auto mask = sequence_length == 1
                          ? mx_support::AttentionMask{"", std::nullopt}
                          : cache.make_mask(
                                static_cast<std::size_t>(sequence_length),
                                kDecoderSlidingWindow, device_);
    auto output = mx::fast::scaled_dot_product_attention(
        queries, cached_keys, cached_values,
        1.0f / std::sqrt(static_cast<float>(kDecoderHeadDimension)),
        mask.mode, mask.array, {}, device_);
    output = mx::reshape(
        mx::transpose(output, {0, 2, 1, 3}, device_),
        {sequence_length,
         static_cast<int>(kDecoderAttentionHeads * kDecoderHeadDimension)},
        device_);
    return quantized_linear(output, prefix + "wo");
  }

  mx::array decoder_layer(
      const mx::array &input, std::size_t start_position, std::size_t layer,
      mx_support::RotatingKeyValueCache &cache) const {
    const auto prefix =
        "decoder.layers." + std::to_string(layer) + ".";
    auto hidden =
        input + decoder_attention(
                    rms_norm(input, prefix + "attention_norm.weight"),
                    start_position, layer, cache);
    auto normalized = rms_norm(hidden, prefix + "ffn_norm.weight");
    normalized = normalized * (1.0f + ada_scales_.at(layer));
    const auto gate =
        mx_support::silu(quantized_linear(normalized,
                                          prefix + "feed_forward_w1"),
                         device_);
    const auto up =
        quantized_linear(normalized, prefix + "feed_forward_w3");
    return hidden +
           quantized_linear(gate * up, prefix + "feed_forward_w2");
  }

  const LoadedModel &model_;
  mx::Device device_;
  mx::array time_embedding_;
  std::vector<mx::array> ada_scales_;
};

int32_t greedy_token(const mx::array &logits, mx::Device device);

class StreamingMel {
public:
  explicit StreamingMel(mx::Device device)
      : device_(device), window_(0.0f), mel_filters_(0.0f) {
    std::vector<float> window(kFftSize);
    constexpr auto pi = 3.14159265358979323846;
    for (std::size_t index = 0; index < kFftSize; ++index) {
      window[index] = static_cast<float>(
          0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(index) /
                               static_cast<double>(kFftSize)));
    }
    window_ = mx::array(window.begin(), {static_cast<int>(window.size())});
    const auto filters = make_slaney_mel_filters();
    mel_filters_ = mx::array(
        filters.begin(),
        {static_cast<int>(kFrequencyBins), static_cast<int>(kMelBins)});
    if (device_ == mx::Device::gpu) {
      window_ = mx::copy(std::move(window_), device_);
      mel_filters_ = mx::copy(std::move(mel_filters_), device_);
    }
    mx::eval(window_, mel_filters_);
  }

  std::optional<mx::array> append(const float *samples,
                                  std::size_t sample_count) {
    if (closed_) {
      throw std::runtime_error("streaming mel is closed");
    }
    if (sample_count == 0) {
      return std::nullopt;
    }
    audio_.insert(audio_.end(), samples, samples + sample_count);
    return drain(false);
  }

  std::optional<mx::array> append(const std::vector<float> &samples) {
    return append(samples.data(), samples.size());
  }

  std::optional<mx::array> close() {
    if (closed_) {
      return std::nullopt;
    }
    closed_ = true;
    return drain(true);
  }

  std::size_t received_samples() const { return audio_.size(); }
  std::size_t emitted_frames() const { return next_frame_; }

private:
  std::optional<mx::array> drain(bool final) {
    const auto received = audio_.size();
    std::size_t maximum_frame = 0;
    if (final) {
      if (received < kHopLength) {
        return std::nullopt;
      }
      maximum_frame = received / kHopLength - 1;
    } else {
      if (received < kFftSize / 2) {
        return std::nullopt;
      }
      maximum_frame = (received - kFftSize / 2) / kHopLength;
    }
    if (next_frame_ > maximum_frame) {
      return std::nullopt;
    }

    const auto frame_count = maximum_frame - next_frame_ + 1;
    std::vector<float> windows(frame_count * kFftSize);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
      const auto global_frame = next_frame_ + frame;
      const auto start =
          static_cast<std::int64_t>(global_frame * kHopLength) -
          static_cast<std::int64_t>(kFftSize / 2);
      for (std::size_t offset = 0; offset < kFftSize; ++offset) {
        const auto raw_index = start + static_cast<std::int64_t>(offset);
        std::int64_t source_index = raw_index;
        if (raw_index < 0) {
          source_index = -raw_index;
        } else if (raw_index >= static_cast<std::int64_t>(received)) {
          if (!closed_) {
            throw std::runtime_error(
                "non-final streaming mel attempted right reflection");
          }
          source_index =
              2 * static_cast<std::int64_t>(received) - 2 - raw_index;
        }
        if (source_index < 0 ||
            source_index >= static_cast<std::int64_t>(received)) {
          throw std::runtime_error(
              "streaming mel reflection exceeded available audio");
        }
        windows[frame * kFftSize + offset] =
            audio_[static_cast<std::size_t>(source_index)];
      }
    }
    next_frame_ = maximum_frame + 1;

    auto frames = mx::array(
        windows.begin(),
        {static_cast<int>(frame_count), static_cast<int>(kFftSize)});
    if (device_ == mx::Device::gpu) {
      frames = mx::copy(std::move(frames), device_);
    }
    auto frequencies = mx::fft::rfft(
        frames * window_, -1, mx::fft::FFTNorm::Backward, device_);
    auto log_mel = mx::log10(
        mx::maximum(
            mx::matmul(mx::square(mx::abs(frequencies, device_), device_),
                       mel_filters_, device_),
            mx::array(1e-10f), device_),
        device_);
    log_mel = mx::maximum(log_mel, mx::array(-6.5f), device_);
    log_mel = (log_mel + 4.0) / 4.0;
    return mx::transpose(log_mel, {1, 0}, device_);
  }

  mx::Device device_;
  mx::array window_;
  mx::array mel_filters_;
  std::vector<float> audio_;
  std::size_t next_frame_{0};
  bool closed_{false};
};

class StreamingCausalConv1d {
public:
  StreamingCausalConv1d(mx::array weight, mx::array bias, std::size_t stride,
                        mx::Device device)
      : weight_(std::move(weight)), bias_(std::move(bias)), stride_(stride),
        keep_(static_cast<std::size_t>(weight_.shape(1)) - stride),
        device_(device) {}

  mx::array step(const mx::array &input) {
    if (input.ndim() != 2) {
      throw std::invalid_argument(
          "streaming causal convolution requires [frames, channels]");
    }
    if (input.shape(0) == 0) {
      return empty(input.dtype());
    }
    auto context = input;
    if (!initialized_) {
      if (keep_ > 0) {
        context = mx::concatenate(
            {mx::zeros({static_cast<int>(keep_), input.shape(1)},
                       input.dtype(), device_),
             input},
            0, device_);
      }
      initialized_ = true;
    } else if (state_.has_value()) {
      context = mx::concatenate({state_.value(), input}, 0, device_);
    }

    const auto kernel = static_cast<std::size_t>(weight_.shape(1));
    if (static_cast<std::size_t>(context.shape(0)) < kernel) {
      state_ = context;
      return empty(input.dtype());
    }
    auto output = mx::squeeze(
        mx::conv1d(mx::expand_dims(context, 0, device_), weight_,
                   static_cast<int>(stride_), 0, 1, 1, device_) +
            bias_,
        0, device_);
    const auto output_frames = static_cast<std::size_t>(output.shape(0));
    const auto leftover =
        static_cast<std::size_t>(context.shape(0)) - output_frames * stride_;
    if (keep_ == 0 || leftover == 0) {
      state_.reset();
    } else {
      const auto state_frames = std::min(keep_, leftover);
      state_ = mx::slice(
          context,
          {context.shape(0) - static_cast<int>(state_frames), 0},
          {context.shape(0), context.shape(1)}, {1, 1}, device_);
    }
    return output;
  }

private:
  mx::array empty(mx::Dtype dtype) const {
    return mx::zeros({0, weight_.shape(0)}, dtype, device_);
  }

  mx::array weight_;
  mx::array bias_;
  std::size_t stride_;
  std::size_t keep_;
  mx::Device device_;
  std::optional<mx::array> state_;
  bool initialized_{false};
};

class StreamingConvStem {
public:
  StreamingConvStem(const LoadedModel &model, mx::Device device)
      : first_(model.weights.at("encoder.conv_layers_0_conv.conv.weight"),
               model.weights.at("encoder.conv_layers_0_conv.conv.bias"), 1,
               device),
        second_(model.weights.at("encoder.conv_layers_1_conv.conv.weight"),
                model.weights.at("encoder.conv_layers_1_conv.conv.bias"), 2,
                device),
        device_(device) {}

  mx::array step(const mx::array &mel) {
    if (mel.ndim() != 2 || mel.shape(0) != static_cast<int>(kMelBins)) {
      throw std::invalid_argument(
          "streaming convolution requires [128, frames] log-mel input");
    }
    auto hidden = mx::transpose(mel, {1, 0}, device_);
    hidden = precise_gelu(first_.step(hidden), device_);
    hidden = precise_gelu(second_.step(hidden), device_);
    return hidden;
  }

private:
  StreamingCausalConv1d first_;
  StreamingCausalConv1d second_;
  mx::Device device_;
};

class StreamingVoxtralEncoder {
public:
  StreamingVoxtralEncoder(const LoadedModel &model, mx::Device device)
      : encoder_(model, device), caches_(encoder_.make_cache()),
        device_(device) {}

  mx::array step(const mx::array &conv_chunk) {
    if (conv_chunk.shape(0) == 0) {
      return empty_adapter(conv_chunk.dtype());
    }
    auto encoded = encoder_.encode_chunk(conv_chunk, position_, caches_);
    position_ += static_cast<std::size_t>(conv_chunk.shape(0));
    if (pending_.has_value()) {
      encoded = mx::concatenate({pending_.value(), encoded}, 0, device_);
    }
    const auto frames = static_cast<std::size_t>(encoded.shape(0));
    const auto usable = frames - frames % kDownsampleFactor;
    if (usable == 0) {
      pending_ = encoded;
      return empty_adapter(encoded.dtype());
    }
    auto aligned = mx::slice(
        encoded, {0, 0},
        {static_cast<int>(usable), static_cast<int>(kEncoderDimension)},
        {1, 1}, device_);
    if (usable < frames) {
      pending_ = mx::slice(
          encoded, {static_cast<int>(usable), 0},
          {static_cast<int>(frames), static_cast<int>(kEncoderDimension)},
          {1, 1}, device_);
    } else {
      pending_.reset();
    }
    auto adapter = encoder_.project_aligned(aligned);
    mx::eval(adapter);
    return adapter;
  }

  std::size_t position() const { return position_; }
  std::size_t cache_size() const { return caches_.front().size(); }

private:
  mx::array empty_adapter(mx::Dtype dtype) const {
    return mx::zeros({0, static_cast<int>(kDecoderDimension)}, dtype,
                     device_);
  }

  VoxtralCausalEncoder encoder_;
  std::vector<mx_support::RotatingKeyValueCache> caches_;
  mx::Device device_;
  std::optional<mx::array> pending_;
  std::size_t position_{0};
};

struct StreamingAdvance {
  bool did_work;
  bool done;
  std::size_t adapter_frames_added;
  std::size_t tokens_added;
  std::string text_delta;
};

class VoxtralStreamingModel {
public:
  VoxtralStreamingModel(const LoadedModel &model,
                        const std::filesystem::path &model_directory,
                        int32_t transcription_delay_ms,
                        std::size_t max_generated_tokens, mx::Device device)
      : device_(device),
        delay_tokens_(delay_token_count(transcription_delay_ms)),
        prompt_length_(1 + kLeftPadTokens + delay_tokens_),
        max_generated_tokens_(max_generated_tokens), mel_(device),
        conv_stem_(model, device), encoder_(model, device),
        decoder_(model, delay_tokens_, device),
        tokenizer_(model_directory / "tekken.json") {
    if (max_generated_tokens_ == 0) {
      throw std::invalid_argument("max generated tokens must be positive");
    }
  }

  StreamingAdvance advance(const std::vector<float> &audio,
                            bool flush_close,
                            std::size_t max_decode_tokens) {
    const auto previous_adapter_frames = adapter_frames_;
    const auto previous_token_count = generated_.size();
    const auto previous_text = text_;
    bool did_work = false;
    if (!audio.empty()) {
      seed_left_pad();
      process_mel(mel_.append(audio));
      did_work = true;
    }
    if (flush_close && !close_flushed_) {
      seed_left_pad();
      close_flushed_ = true;
      const auto right_pad_tokens = delay_tokens_ + 1 + 10;
      std::vector<float> right_pad(
          right_pad_tokens * kRawAudioSamplesPerToken, 0.0f);
      process_mel(mel_.append(right_pad));
      process_mel(mel_.close());
      did_work = true;
    }
    if (!prefilled_ && adapter_frames_ >= prompt_length_) {
      prefill();
      did_work = true;
    }
    if (!prefilled_ && close_flushed_) {
      done_ = true;
    }
    if (prefilled_ && !done_ && max_decode_tokens > 0) {
      did_work = decode_some(max_decode_tokens) || did_work;
    }
    return {
        did_work,
        done_,
        adapter_frames_ - previous_adapter_frames,
        generated_.size() - previous_token_count,
        text_.substr(previous_text.size()),
    };
  }

  bool done() const { return done_; }
  bool close_flushed() const { return close_flushed_; }
  bool prefilled() const { return prefilled_; }
  std::size_t adapter_frames() const { return adapter_frames_; }
  std::size_t generated_tokens() const { return generated_.size(); }
  std::size_t decoder_position() const { return decoder_position_; }
  std::size_t encoder_position() const { return encoder_.position(); }
  std::size_t encoder_cache_size() const { return encoder_.cache_size(); }
  std::size_t decoder_cache_size() const {
    return decoder_caches_.empty() ? 0 : decoder_caches_.front().size();
  }
  std::size_t mel_frames() const { return mel_.emitted_frames(); }
  const std::string &text() const { return text_; }

private:
  static std::size_t delay_token_count(int32_t transcription_delay_ms) {
    if (transcription_delay_ms <= 0) {
      throw std::invalid_argument("transcription delay must be positive");
    }
    const auto delay_samples = static_cast<std::size_t>(
        static_cast<double>(transcription_delay_ms) / 1000.0 *
        static_cast<double>(kSampleRate));
    return audio_token_count(delay_samples);
  }

  void seed_left_pad() {
    if (left_pad_seeded_) {
      return;
    }
    left_pad_seeded_ = true;
    std::vector<float> left_pad(kLeftPadTokens * kRawAudioSamplesPerToken,
                                0.0f);
    process_mel(mel_.append(left_pad));
  }

  void process_mel(std::optional<mx::array> mel) {
    if (!mel.has_value() || mel->shape(1) == 0) {
      return;
    }
    auto conv = conv_stem_.step(mel.value());
    if (conv.shape(0) == 0) {
      return;
    }
    auto adapter = encoder_.step(conv);
    if (adapter.shape(0) == 0) {
      return;
    }
    if (adapter_.has_value()) {
      adapter_ =
          mx::concatenate({adapter_.value(), adapter}, 0, device_);
    } else {
      adapter_ = adapter;
    }
    mx::eval(adapter_.value());
    adapter_frames_ = static_cast<std::size_t>(adapter_->shape(0));
  }

  mx::array adapter_at(std::size_t position) const {
    if (!adapter_.has_value() || position >= adapter_frames_) {
      throw std::out_of_range("decoder position exceeds adapter frames");
    }
    return mx::take(adapter_.value(), static_cast<int>(position), 0,
                    device_);
  }

  void prefill() {
    std::vector<int32_t> prompt_ids{kBosToken};
    prompt_ids.insert(prompt_ids.end(), kLeftPadTokens + delay_tokens_,
                      kStreamingPadToken);
    auto prefix = mx::slice(
                      adapter_.value(), {0, 0},
                      {static_cast<int>(prompt_length_),
                       static_cast<int>(kDecoderDimension)},
                      {1, 1}, device_) +
                  decoder_.embed_tokens(prompt_ids);
    decoder_caches_ = decoder_.make_cache();
    auto hidden = decoder_.forward(prefix, 0, decoder_caches_);
    auto logits = decoder_.logits(
        mx::take(hidden, hidden.shape(0) - 1, 0, device_));
    next_token_ = greedy_token(logits, device_);
    decoder_position_ = prompt_length_;
    prefilled_ = true;
  }

  bool decode_some(std::size_t maximum_tokens) {
    bool did_work = false;
    for (std::size_t index = 0; index < maximum_tokens && !done_; ++index) {
      if (adapter_frames_ <= decoder_position_ && !close_flushed_) {
        break;
      }
      const auto token = next_token_;
      if (adapter_frames_ <= decoder_position_) {
        generated_.push_back(token);
        update_text();
        done_ = true;
        did_work = true;
        break;
      }

      auto input = adapter_at(decoder_position_) +
                   decoder_.embed_token(token);
      auto hidden = decoder_.forward(mx::expand_dims(input, 0, device_),
                                     decoder_position_, decoder_caches_);
      auto logits = decoder_.logits(mx::squeeze(hidden, 0, device_));
      const auto following_token = greedy_token(logits, device_);
      generated_.push_back(token);
      update_text();
      did_work = true;
      if (token == kEosToken ||
          generated_.size() >= max_generated_tokens_) {
        done_ = true;
        break;
      }
      next_token_ = following_token;
      ++decoder_position_;
    }
    return did_work;
  }

  void update_text() {
    std::vector<int32_t> text_tokens;
    text_tokens.reserve(generated_.size());
    for (const auto token : generated_) {
      if (token != kEosToken) {
        text_tokens.push_back(token);
      }
    }
    text_ = trim_ascii_whitespace(tokenizer_.decode(text_tokens));
  }

  mx::Device device_;
  std::size_t delay_tokens_;
  std::size_t prompt_length_;
  std::size_t max_generated_tokens_;
  StreamingMel mel_;
  StreamingConvStem conv_stem_;
  StreamingVoxtralEncoder encoder_;
  VoxtralDecoder decoder_;
  TekkenTokenizer tokenizer_;
  std::optional<mx::array> adapter_;
  std::vector<mx_support::RotatingKeyValueCache> decoder_caches_;
  std::vector<int32_t> generated_;
  std::string text_;
  std::size_t adapter_frames_{0};
  std::size_t decoder_position_{0};
  int32_t next_token_{kStreamingPadToken};
  bool left_pad_seeded_{false};
  bool close_flushed_{false};
  bool prefilled_{false};
  bool done_{false};
};

int32_t greedy_token(const mx::array &logits, mx::Device device) {
  const auto token = mx::argmax(logits, -1, false, device);
  mx::eval(token);
  return token.item<int32_t>();
}

std::string transcribe_greedy(LoadedModel model,
                              const std::filesystem::path &model_directory,
                              const float *audio, std::size_t audio_len,
                              int32_t transcription_delay_ms,
                              std::size_t max_generated_tokens,
                              mx::Device device) {
  if (max_generated_tokens == 0) {
    throw std::invalid_argument("max_generated_tokens must be positive");
  }
  const auto started = std::chrono::steady_clock::now();
  materialize_weight_prefix_on_device(model, "encoder.", device);
  materialize_weight_prefix_on_device(model, "decoder.", device);
  const auto frontend = compute_audio_frontend(
      model, audio, audio_len, transcription_delay_ms, device);
  const VoxtralCausalEncoder encoder(model, device);
  auto encoder_result = encoder.encode(frontend.conv_stem);
  auto adapter = encoder_result.adapter;
  mx::eval(adapter);

  const VoxtralDecoder decoder(model, frontend.padding.delay_tokens, device);
  DecoderEvidence evidence;
  evidence.fingerprints.emplace("adapter",
                                fingerprint_json(adapter, device));
  evidence.fingerprints.emplace(
      "time_embedding", fingerprint_json(decoder.time_embedding(), device));
  for (const auto layer : {0U, 12U, 25U}) {
    evidence.fingerprints.emplace(
        "ada_scale_layer_" + std::to_string(layer),
        fingerprint_json(decoder.ada_scale(layer), device));
  }

  std::vector<int32_t> prompt_ids{ kBosToken };
  prompt_ids.insert(prompt_ids.end(),
                    kLeftPadTokens + frontend.padding.delay_tokens,
                    kStreamingPadToken);
  auto prompt_text_embeddings = decoder.embed_tokens(prompt_ids);
  auto prefix_embeddings =
      mx::slice(adapter, {0, 0},
                {static_cast<int>(prompt_ids.size()),
                 static_cast<int>(kDecoderDimension)},
                {1, 1}, device) +
      prompt_text_embeddings;
  evidence.fingerprints.emplace(
      "prompt_text_embeddings",
      fingerprint_json(prompt_text_embeddings, device));
  evidence.fingerprints.emplace(
      "prefix_embeddings", fingerprint_json(prefix_embeddings, device));

  auto caches = decoder.make_cache();
  auto hidden =
      decoder.forward(prefix_embeddings, 0, caches, &evidence, "prefill");
  auto logits = decoder.logits(
      mx::take(hidden, hidden.shape(0) - 1, 0, device));
  evidence.fingerprints.emplace("prefill_logits",
                                fingerprint_json(logits, device));

  std::vector<int32_t> generated;
  auto next_token = greedy_token(logits, device);
  auto last_logits = logits;
  std::size_t forward_steps = 0;
  bool stopped = false;
  std::string_view finish_reason = "audio_end";
  for (std::size_t position = prompt_ids.size();
       position < static_cast<std::size_t>(adapter.shape(0)); ++position) {
    const auto token = next_token;
    generated.push_back(token);
    if (token == kEosToken || generated.size() >= max_generated_tokens) {
      stopped = true;
      finish_reason = token == kEosToken ? "eos" : "max_tokens";
      break;
    }

    auto decoder_input =
        mx::take(adapter, static_cast<int>(position), 0, device) +
        decoder.embed_token(token);
    const auto capture = forward_steps == 0;
    if (capture) {
      evidence.fingerprints.emplace(
          "decode_0_input", fingerprint_json(decoder_input, device));
    }
    hidden = decoder.forward(mx::expand_dims(decoder_input, 0, device),
                             position, caches,
                             capture ? &evidence : nullptr,
                             capture ? "decode_0" : "");
    logits = decoder.logits(mx::squeeze(hidden, 0, device));
    if (capture) {
      evidence.fingerprints.emplace("decode_0_logits",
                                    fingerprint_json(logits, device));
    }
    last_logits = logits;
    next_token = greedy_token(logits, device);
    ++forward_steps;
  }
  if (!stopped) {
    generated.push_back(next_token);
  }
  evidence.fingerprints.emplace("final_logits",
                                fingerprint_json(last_logits, device));
  evidence.fingerprints.emplace(
      "decoder_layer_0_cache_keys",
      fingerprint_json(caches.front().keys(), device));
  evidence.fingerprints.emplace(
      "decoder_layer_0_cache_values",
      fingerprint_json(caches.front().values(), device));

  auto text_tokens = generated;
  if (!text_tokens.empty() && text_tokens.back() == kEosToken) {
    text_tokens.pop_back();
  }
  const TekkenTokenizer tokenizer(model_directory / "tekken.json");
  const auto text = trim_ascii_whitespace(tokenizer.decode(text_tokens));
  if (text.empty()) {
    throw std::runtime_error("direct Voxtral decoder emitted no text");
  }
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  const std::vector<std::string> fingerprint_order{
      "adapter",
      "time_embedding",
      "ada_scale_layer_0",
      "ada_scale_layer_12",
      "ada_scale_layer_25",
      "prompt_text_embeddings",
      "prefix_embeddings",
      "prefill_layer_0",
      "prefill_layer_12",
      "prefill_layer_25",
      "prefill_norm",
      "prefill_logits",
      "decode_0_input",
      "decode_0_layer_0",
      "decode_0_layer_12",
      "decode_0_layer_25",
      "decode_0_norm",
      "decode_0_logits",
      "final_logits",
      "decoder_layer_0_cache_keys",
      "decoder_layer_0_cache_values",
  };

  std::ostringstream output;
  output << std::setprecision(17)
         << "{\"status\":\"ok\",\"boundary\":\"official-mlx-cpp\","
         << "\"stage\":\"voxtral-greedy-transcription\",\"device\":\""
         << (device == mx::Device::gpu ? "gpu" : "cpu")
         << "\",\"pcm_samples\":" << audio_len
         << ",\"transcription_delay_ms\":" << transcription_delay_ms
         << ",\"delay_tokens\":" << frontend.padding.delay_tokens
         << ",\"architecture\":{\"layers\":" << kDecoderLayers
         << ",\"dimension\":" << kDecoderDimension
         << ",\"attention_heads\":" << kDecoderAttentionHeads
         << ",\"kv_heads\":" << kDecoderKeyValueHeads
         << ",\"head_dimension\":" << kDecoderHeadDimension
         << ",\"hidden_dimension\":" << kDecoderHiddenDimension
         << ",\"sliding_window\":" << kDecoderSlidingWindow
         << ",\"vocabulary_size\":" << kDecoderVocabularySize
         << ",\"ada_bottleneck_dimension\":" << kAdaBottleneckDimension
         << "},\"prompt\":{\"bos_token_id\":" << kBosToken
         << ",\"streaming_pad_token_id\":" << kStreamingPadToken
         << ",\"eos_token_id\":" << kEosToken << ",\"token_ids\":[";
  for (std::size_t index = 0; index < prompt_ids.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    output << prompt_ids[index];
  }
  output << "],\"length\":" << prompt_ids.size()
         << ",\"adapter_frames\":" << adapter.shape(0)
         << "},\"generation\":{\"tokens\":[";
  for (std::size_t index = 0; index < generated.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    output << generated[index];
  }
  output << "],\"token_count\":" << generated.size()
         << ",\"forward_steps\":" << forward_steps
         << ",\"finish_reason\":\"" << finish_reason
         << "\",\"text\":\"" << json_escape(text)
         << "\"},\"cache\":{\"layer_0_offset\":"
         << caches.front().offset() << ",\"layer_0_size\":"
         << caches.front().size() << ",\"layer_0_state_frames\":"
         << caches.front().materialized_size()
         << "},\"elapsed_ms\":" << elapsed_ms
         << ",\"peak_memory_bytes\":" << mx::get_peak_memory()
         << ",\"fingerprints\":{";
  for (std::size_t index = 0; index < fingerprint_order.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    const auto &name = fingerprint_order[index];
    const auto found = evidence.fingerprints.find(name);
    if (found == evidence.fingerprints.end()) {
      throw std::runtime_error("missing decoder fingerprint: " + name);
    }
    output << "\"" << name << "\":" << found->second;
  }
  output << "},\"capabilities\":{\"delay_conditioning\":true,"
         << "\"decoder\":true,\"decoder_kv_cache\":true,"
         << "\"tekken_decode\":true,\"greedy_transcription\":true,"
         << "\"streaming_session\":false}}";
  return output.str();
}

mx::Device requested_device(int32_t device_kind) {
  if (device_kind == 0) {
    return mx::Device::cpu;
  }
  if (device_kind == 1) {
    return mx::Device::gpu;
  }
  throw std::invalid_argument("device_kind must be 0 (CPU) or 1 (GPU)");
}

mx::Stream reusable_thread_unsafe_stream(mx::Device device) {
  auto &stream =
      device == mx::Device::cpu ? runtime_cpu_stream : runtime_gpu_stream;
  if (!stream.has_value()) {
    stream = mx::new_thread_unsafe_stream(device);
  }
  return stream.value();
}

struct VoxtralMlxSession {
  VoxtralMlxSession(const std::filesystem::path &model_directory,
                    int32_t requested_delay_ms,
                    std::size_t requested_max_generated_tokens,
                    std::size_t requested_max_decode_tokens_per_step,
                    mx::Device requested_device,
                    mx::Stream requested_cpu_stream,
                    mx::Stream requested_device_stream,
                    std::size_t requested_max_pending,
                    std::size_t requested_step_budget)
      : model_directory(model_directory),
        transcription_delay_ms(requested_delay_ms),
        max_generated_tokens(requested_max_generated_tokens),
        max_decode_tokens_per_step(requested_max_decode_tokens_per_step),
        device(requested_device), cpu_stream(requested_cpu_stream),
        device_stream(requested_device_stream),
        max_pending_samples(requested_max_pending),
        max_ingest_samples_per_step(requested_step_budget),
        model(load_and_validate_model(model_directory)) {
    materialize_weight_prefix_on_device(model, "encoder.", device);
    materialize_weight_prefix_on_device(model, "decoder.", device);
    streaming = std::make_unique<VoxtralStreamingModel>(
        model, model_directory, transcription_delay_ms, max_generated_tokens,
        device);
  }

  std::filesystem::path model_directory;
  int32_t transcription_delay_ms;
  std::size_t max_generated_tokens;
  std::size_t max_decode_tokens_per_step;
  mx::Device device;
  mx::Stream cpu_stream;
  mx::Stream device_stream;
  std::size_t max_pending_samples;
  std::size_t max_ingest_samples_per_step;
  LoadedModel model;
  std::unique_ptr<VoxtralStreamingModel> streaming;
  std::deque<float> pending;
  std::size_t total_fed_samples{0};
  std::size_t total_ingested_samples{0};
  std::size_t step_count{0};
  bool audio_closed{false};
  bool done{false};
  std::atomic<bool> cancel_requested{false};
  std::mutex state_mutex;
  std::mutex operation_mutex;
};

std::string state_json(const VoxtralMlxSession &session,
                       std::string_view state, std::size_t ingested_samples,
                       std::size_t pending_samples,
                       const StreamingAdvance &advance, double elapsed_ms) {
  const auto &streaming = *session.streaming;
  std::ostringstream json;
  json << std::setprecision(17) << "{\"state\":\"" << state << "\","
       << "\"ingested_samples\":" << ingested_samples << ","
       << "\"pending_samples\":" << pending_samples << ","
       << "\"total_fed_samples\":" << session.total_fed_samples << ","
       << "\"total_ingested_samples\":" << session.total_ingested_samples
       << ",\"step_count\":" << session.step_count << ","
       << "\"max_ingest_samples_per_step\":"
       << session.max_ingest_samples_per_step << ","
       << "\"max_decode_tokens_per_step\":"
       << session.max_decode_tokens_per_step << ","
       << "\"transcription_delay_ms\":" << session.transcription_delay_ms
       << ",\"max_generated_tokens\":" << session.max_generated_tokens << ","
       << "\"adapter_frames_added\":" << advance.adapter_frames_added << ","
       << "\"tokens_added\":" << advance.tokens_added << ","
       << "\"mel_frames\":" << streaming.mel_frames() << ","
       << "\"encoder_position\":" << streaming.encoder_position() << ","
       << "\"encoder_cache_size\":" << streaming.encoder_cache_size() << ","
       << "\"adapter_frames\":" << streaming.adapter_frames() << ","
       << "\"decoder_position\":" << streaming.decoder_position() << ","
       << "\"decoder_cache_size\":" << streaming.decoder_cache_size() << ","
       << "\"generated_tokens\":" << streaming.generated_tokens() << ","
       << "\"text_delta\":\"" << json_escape(advance.text_delta) << "\","
       << "\"text\":\"" << json_escape(streaming.text()) << "\","
       << "\"mlx_elapsed_ms\":" << elapsed_ms << ","
       << "\"peak_memory_bytes\":" << mx::get_peak_memory() << ","
       << "\"audio_closed\":" << (session.audio_closed ? "true" : "false")
       << ",\"close_flushed\":"
       << (streaming.close_flushed() ? "true" : "false")
       << ",\"prefilled\":" << (streaming.prefilled() ? "true" : "false")
       << ",\"transcription_implemented\":true,"
       << "\"streaming_session\":true}";
  return json.str();
}

int32_t return_json(int32_t status, const std::string &json, char **json_out,
                    char **error_out) {
  if (json_out == nullptr) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            "json_out must be non-null", error_out);
  }
  *json_out = strdup(json.c_str());
  if (*json_out == nullptr) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR,
                            "could not allocate JSON result", error_out);
  }
  return status;
}

} // namespace

extern "C" int32_t cuttledoc_voxtral_mlx_inspect_model(
    const char *model_directory, char **json_out, char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || json_out == nullptr) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            "model_directory and json_out must be non-null",
                            error_out);
  }
  try {
    const std::lock_guard lock(runtime_mutex);
    const auto model = load_and_validate_model(model_directory);
    return return_json(CUTTLEDOC_VOXTRAL_MLX_OK, model_json(model), json_out,
                       error_out);
  } catch (const std::exception &error) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR, error.what(),
                            error_out);
  }
}

extern "C" int32_t cuttledoc_voxtral_mlx_probe_audio_frontend(
    const char *model_directory, const float *audio, std::size_t audio_len,
    int32_t transcription_delay_ms, int32_t device_kind, char **json_out,
    char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || audio == nullptr || audio_len == 0 ||
      json_out == nullptr) {
    return fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
        "model_directory, non-empty audio, and json_out are required",
        error_out);
  }
  try {
    const std::lock_guard lock(runtime_mutex);
    const auto device = requested_device(device_kind);
    if (!mx::is_available(device)) {
      throw std::runtime_error("requested MLX device is not available");
    }
    mx::set_default_device(device);
    const auto cpu_stream = reusable_thread_unsafe_stream(mx::Device::cpu);
    mx::set_default_stream(cpu_stream);
    const auto device_stream =
        device == mx::Device::cpu ? cpu_stream
                                  : reusable_thread_unsafe_stream(device);
    mx::set_default_stream(device_stream);
    mx::clear_cache();
    mx::reset_peak_memory();
    const auto model = load_and_validate_model(model_directory);
    return return_json(
        CUTTLEDOC_VOXTRAL_MLX_OK,
        probe_audio_frontend(model, audio, audio_len, transcription_delay_ms,
                             device),
        json_out, error_out);
  } catch (const std::invalid_argument &error) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            error.what(), error_out);
  } catch (const std::exception &error) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR, error.what(),
                            error_out);
  } catch (...) {
    return fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR,
        "MLX raised a non-standard exception in the audio frontend",
        error_out);
  }
}

extern "C" int32_t cuttledoc_voxtral_mlx_probe_causal_encoder(
    const char *model_directory, const float *audio, std::size_t audio_len,
    int32_t transcription_delay_ms, int32_t device_kind, char **json_out,
    char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || audio == nullptr || audio_len == 0 ||
      json_out == nullptr) {
    return fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
        "model_directory, non-empty audio, and json_out are required",
        error_out);
  }
  try {
    const std::lock_guard lock(runtime_mutex);
    const auto device = requested_device(device_kind);
    if (!mx::is_available(device)) {
      throw std::runtime_error("requested MLX device is not available");
    }
    mx::set_default_device(device);
    const auto cpu_stream = reusable_thread_unsafe_stream(mx::Device::cpu);
    mx::set_default_stream(cpu_stream);
    const auto device_stream =
        device == mx::Device::cpu ? cpu_stream
                                  : reusable_thread_unsafe_stream(device);
    mx::set_default_stream(device_stream);
    mx::clear_cache();
    mx::reset_peak_memory();
    auto model = load_and_validate_model(model_directory);
    return return_json(
        CUTTLEDOC_VOXTRAL_MLX_OK,
        probe_causal_encoder(std::move(model), audio, audio_len,
                             transcription_delay_ms, device),
        json_out, error_out);
  } catch (const std::invalid_argument &error) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            error.what(), error_out);
  } catch (const std::exception &error) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR, error.what(),
                            error_out);
  } catch (...) {
    return fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR,
        "MLX raised a non-standard exception in the causal encoder",
        error_out);
  }
}

extern "C" int32_t cuttledoc_voxtral_mlx_transcribe(
    const char *model_directory, const float *audio, std::size_t audio_len,
    int32_t transcription_delay_ms, std::size_t max_generated_tokens,
    int32_t device_kind, char **json_out, char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || audio == nullptr || audio_len == 0 ||
      max_generated_tokens == 0 || json_out == nullptr) {
    return fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
        "model_directory, non-empty audio, positive max_generated_tokens, "
        "and json_out are required",
        error_out);
  }
  try {
    const std::lock_guard lock(runtime_mutex);
    const auto device = requested_device(device_kind);
    if (!mx::is_available(device)) {
      throw std::runtime_error("requested MLX device is not available");
    }
    mx::set_default_device(device);
    const auto cpu_stream = reusable_thread_unsafe_stream(mx::Device::cpu);
    mx::set_default_stream(cpu_stream);
    const auto device_stream =
        device == mx::Device::cpu ? cpu_stream
                                  : reusable_thread_unsafe_stream(device);
    mx::set_default_stream(device_stream);
    mx::clear_cache();
    mx::reset_peak_memory();
    auto model = load_and_validate_model(model_directory);
    return return_json(
        CUTTLEDOC_VOXTRAL_MLX_OK,
        transcribe_greedy(std::move(model), model_directory, audio, audio_len,
                          transcription_delay_ms, max_generated_tokens,
                          device),
        json_out, error_out);
  } catch (const std::invalid_argument &error) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            error.what(), error_out);
  } catch (const std::exception &error) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR, error.what(),
                            error_out);
  } catch (...) {
    return fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR,
        "MLX raised a non-standard exception during transcription",
        error_out);
  }
}

extern "C" void *cuttledoc_voxtral_mlx_session_create(
    const char *model_directory, int32_t transcription_delay_ms,
    std::size_t max_generated_tokens,
    std::size_t max_decode_tokens_per_step, int32_t device_kind,
    std::size_t max_pending_samples,
    std::size_t max_ingest_samples_per_step, int32_t *status_out,
    char **error_out) {
  if (status_out != nullptr) {
    *status_out = CUTTLEDOC_VOXTRAL_MLX_OK;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || status_out == nullptr ||
      transcription_delay_ms <= 0 || max_generated_tokens == 0 ||
      max_decode_tokens_per_step == 0 ||
      max_pending_samples == 0 || max_ingest_samples_per_step == 0 ||
      max_ingest_samples_per_step > max_pending_samples) {
    if (status_out != nullptr) {
      *status_out = CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT;
    }
    fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
        "model, status, positive delay and generation limits, positive queue "
        "capacity, and an ingest budget no larger than capacity are required",
        error_out);
    return nullptr;
  }
  try {
    const std::lock_guard lock(runtime_mutex);
    const auto device = requested_device(device_kind);
    if (!mx::is_available(device)) {
      throw std::runtime_error("requested MLX device is not available");
    }
    mx::set_default_device(device);
    const auto cpu_stream = reusable_thread_unsafe_stream(mx::Device::cpu);
    mx::set_default_stream(cpu_stream);
    const auto device_stream =
        device == mx::Device::cpu ? cpu_stream
                                  : reusable_thread_unsafe_stream(device);
    mx::set_default_stream(device_stream);
    mx::clear_cache();
    mx::reset_peak_memory();
    return new VoxtralMlxSession(model_directory, transcription_delay_ms,
                                 max_generated_tokens,
                                 max_decode_tokens_per_step, device, cpu_stream,
                                 device_stream, max_pending_samples,
                                 max_ingest_samples_per_step);
  } catch (const std::invalid_argument &error) {
    *status_out = CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT;
    fail_with_status(*status_out, error.what(), error_out);
  } catch (const std::exception &error) {
    *status_out = CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR;
    fail_with_status(*status_out, error.what(), error_out);
  }
  return nullptr;
}

extern "C" int32_t cuttledoc_voxtral_mlx_session_feed(
    void *handle, const float *audio, std::size_t audio_len,
    char **error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (handle == nullptr || audio == nullptr || audio_len == 0) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            "handle and non-empty audio are required",
                            error_out);
  }
  auto *session = static_cast<VoxtralMlxSession *>(handle);
  if (session->cancel_requested.load(std::memory_order_relaxed)) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_CANCELLED,
                            "streaming session is cancelled", error_out);
  }
  const std::lock_guard lock(session->state_mutex);
  if (session->cancel_requested.load(std::memory_order_relaxed)) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_CANCELLED,
                            "streaming session is cancelled", error_out);
  }
  if (session->audio_closed || session->done) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            "cannot feed a closed streaming session",
                            error_out);
  }
  if (audio_len > session->max_pending_samples - session->pending.size()) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_BACKPRESSURE,
                            "bounded audio queue is full", error_out);
  }
  session->pending.insert(session->pending.end(), audio, audio + audio_len);
  session->total_fed_samples += audio_len;
  return CUTTLEDOC_VOXTRAL_MLX_OK;
}

extern "C" int32_t cuttledoc_voxtral_mlx_session_close(void *handle,
                                                        char **error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (handle == nullptr) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            "handle is required", error_out);
  }
  auto *session = static_cast<VoxtralMlxSession *>(handle);
  if (session->cancel_requested.load(std::memory_order_relaxed)) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_CANCELLED,
                            "streaming session is cancelled", error_out);
  }
  const std::lock_guard lock(session->state_mutex);
  if (session->cancel_requested.load(std::memory_order_relaxed)) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_CANCELLED,
                            "streaming session is cancelled", error_out);
  }
  session->audio_closed = true;
  return CUTTLEDOC_VOXTRAL_MLX_OK;
}

extern "C" int32_t cuttledoc_voxtral_mlx_session_step(
    void *handle, char **json_out, char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (handle == nullptr || json_out == nullptr) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
                            "handle and json_out are required", error_out);
  }
  auto *session = static_cast<VoxtralMlxSession *>(handle);
  std::unique_lock operation(session->operation_mutex, std::try_to_lock);
  if (!operation.owns_lock()) {
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_BUSY,
                            "streaming step is already active", error_out);
  }

  if (session->cancel_requested.load(std::memory_order_relaxed)) {
    const std::lock_guard lock(session->state_mutex);
    session->pending.clear();
    session->done = true;
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_CANCELLED,
                            "streaming session cancelled before MLX step",
                            error_out);
  }

  std::vector<float> snapshot;
  bool audio_closed = false;
  {
    const std::lock_guard lock(session->state_mutex);
    if (session->done) {
      const StreamingAdvance advance{false, true, 0, 0, ""};
      return return_json(
          CUTTLEDOC_VOXTRAL_MLX_DONE,
          state_json(*session, "done", 0, session->pending.size(), advance,
                     0.0),
          json_out,
          error_out);
    }
    const auto count = std::min(session->max_ingest_samples_per_step,
                                session->pending.size());
    snapshot.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      snapshot.push_back(session->pending.front());
      session->pending.pop_front();
    }
    audio_closed = session->audio_closed;
  }

  const auto started = std::chrono::steady_clock::now();
  StreamingAdvance advance{false, false, 0, 0, ""};
  try {
    const std::lock_guard lock(runtime_mutex);
    mx::set_default_device(session->device);
    mx::set_default_stream(session->cpu_stream);
    mx::set_default_stream(session->device_stream);
    bool flush_close = false;
    {
      const std::lock_guard state_lock(session->state_mutex);
      flush_close = audio_closed && session->pending.empty() &&
                    !session->streaming->close_flushed();
    }
    advance = session->streaming->advance(
        snapshot, flush_close, session->max_decode_tokens_per_step);
  } catch (const std::exception &error) {
    const std::lock_guard lock(session->state_mutex);
    session->pending.clear();
    session->done = true;
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_RUNTIME_ERROR, error.what(),
                            error_out);
  }
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  if (session->cancel_requested.load(std::memory_order_relaxed)) {
    const std::lock_guard lock(session->state_mutex);
    session->pending.clear();
    session->done = true;
    return fail_with_status(CUTTLEDOC_VOXTRAL_MLX_CANCELLED,
                            "streaming session cancelled after MLX step",
                            error_out);
  }

  std::size_t pending_samples = 0;
  std::string json;
  int32_t status = CUTTLEDOC_VOXTRAL_MLX_OK;
  {
    const std::lock_guard lock(session->state_mutex);
    session->total_ingested_samples += snapshot.size();
    ++session->step_count;
    pending_samples = session->pending.size();
    if (advance.done) {
      session->done = true;
      status = CUTTLEDOC_VOXTRAL_MLX_DONE;
      json = state_json(*session, "done", snapshot.size(), pending_samples,
                        advance, elapsed_ms);
    } else if (!advance.did_work && snapshot.empty() &&
               !session->audio_closed) {
      status = CUTTLEDOC_VOXTRAL_MLX_NEEDS_AUDIO;
      json = state_json(*session, "needs_audio", 0, pending_samples, advance,
                        elapsed_ms);
    } else {
      json = state_json(*session, "progress", snapshot.size(), pending_samples,
                        advance, elapsed_ms);
    }
  }
  return return_json(status, json, json_out, error_out);
}

extern "C" void cuttledoc_voxtral_mlx_session_cancel(void *handle) {
  if (handle != nullptr) {
    static_cast<VoxtralMlxSession *>(handle)->cancel_requested.store(
        true, std::memory_order_relaxed);
  }
}

extern "C" void cuttledoc_voxtral_mlx_session_destroy(void *handle) {
  if (handle == nullptr) {
    return;
  }
  auto *session = static_cast<VoxtralMlxSession *>(handle);
  session->cancel_requested.store(true, std::memory_order_relaxed);
  session->operation_mutex.lock();
  session->operation_mutex.unlock();
  const std::lock_guard lock(runtime_mutex);
  delete session;
  mx::clear_cache();
}

extern "C" void cuttledoc_voxtral_mlx_free_string(char *value) {
  std::free(value);
}
