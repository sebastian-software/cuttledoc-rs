#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuttledoc_voxtral_mlx_shim.h"
#include "mlx/io.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

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
constexpr std::size_t kDownsampleFactor = 4;

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
       << "\"cancellation\":true,\"transcription\":false}}";
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

std::string probe_audio_frontend(const LoadedModel &model, const float *audio,
                                 std::size_t audio_len,
                                 int32_t transcription_delay_ms,
                                 mx::Device device) {
  if (audio_len < kFftSize) {
    throw std::invalid_argument(
        "audio must contain at least 400 mono float32 samples");
  }
  if (transcription_delay_ms <= 0) {
    throw std::invalid_argument("transcription delay must be positive");
  }
  const auto started = std::chrono::steady_clock::now();
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
  const auto mel_filter_array =
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

  const auto mel_filter_fingerprint =
      fingerprint_json(mel_filter_array, device);
  const auto log_mel_fingerprint = fingerprint_json(log_mel, device);
  const auto conv0_fingerprint = fingerprint_json(conv0, device);
  const auto conv1_pretrunc_fingerprint =
      fingerprint_json(conv1_pretrunc, device);
  const auto conv_stem_fingerprint = fingerprint_json(conv_stem, device);
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
       << padding.delay_tokens << ",\"left_pad_tokens\":" << kLeftPadTokens
       << ",\"left_pad_samples\":" << padding.left_pad_samples
       << ",\"alignment_pad_samples\":" << padding.alignment_pad_samples
       << ",\"right_pad_tokens\":" << padding.right_pad_tokens
       << ",\"right_pad_samples\":" << padding.right_pad_samples
       << ",\"padded_samples\":" << padding.padded_samples
       << "},\"mel_frames\":" << log_mel.shape(1)
       << ",\"front_truncation_frames\":" << front_truncation
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
                    mx::Device requested_device,
                    mx::Stream requested_cpu_stream,
                    mx::Stream requested_device_stream,
                    std::size_t requested_max_pending,
                    std::size_t requested_step_budget)
      : device(requested_device), cpu_stream(requested_cpu_stream),
        device_stream(requested_device_stream),
        max_pending_samples(requested_max_pending),
        max_ingest_samples_per_step(requested_step_budget),
        model(load_and_validate_model(model_directory)) {}

  mx::Device device;
  mx::Stream cpu_stream;
  mx::Stream device_stream;
  std::size_t max_pending_samples;
  std::size_t max_ingest_samples_per_step;
  LoadedModel model;
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
                       std::size_t pending_samples, double energy,
                       double elapsed_ms) {
  std::ostringstream json;
  json << std::setprecision(17) << "{\"state\":\"" << state << "\","
       << "\"ingested_samples\":" << ingested_samples << ","
       << "\"pending_samples\":" << pending_samples << ","
       << "\"total_fed_samples\":" << session.total_fed_samples << ","
       << "\"total_ingested_samples\":" << session.total_ingested_samples
       << ",\"step_count\":" << session.step_count << ","
       << "\"max_ingest_samples_per_step\":"
       << session.max_ingest_samples_per_step << ","
       << "\"mlx_sum_squares\":" << energy << ","
       << "\"mlx_elapsed_ms\":" << elapsed_ms << ","
       << "\"audio_closed\":" << (session.audio_closed ? "true" : "false")
       << ",\"transcription_implemented\":false}";
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

extern "C" void *cuttledoc_voxtral_mlx_session_create(
    const char *model_directory, int32_t device_kind,
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
      max_pending_samples == 0 || max_ingest_samples_per_step == 0 ||
      max_ingest_samples_per_step > max_pending_samples) {
    if (status_out != nullptr) {
      *status_out = CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT;
    }
    fail_with_status(
        CUTTLEDOC_VOXTRAL_MLX_INVALID_ARGUMENT,
        "model, status, positive queue capacity, and a step budget no larger "
        "than capacity are required",
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
    return new VoxtralMlxSession(model_directory, device, cpu_stream,
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
  {
    const std::lock_guard lock(session->state_mutex);
    if (session->done) {
      return return_json(
          CUTTLEDOC_VOXTRAL_MLX_DONE,
          state_json(*session, "done", 0, 0, 0.0, 0.0), json_out,
          error_out);
    }
    if (session->pending.empty()) {
      if (session->audio_closed) {
        session->done = true;
        return return_json(
            CUTTLEDOC_VOXTRAL_MLX_DONE,
            state_json(*session, "done", 0, 0, 0.0, 0.0), json_out,
            error_out);
      }
      return return_json(
          CUTTLEDOC_VOXTRAL_MLX_NEEDS_AUDIO,
          state_json(*session, "needs_audio", 0, 0, 0.0, 0.0), json_out,
          error_out);
    }
    const auto count = std::min(session->max_ingest_samples_per_step,
                                session->pending.size());
    snapshot.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      snapshot.push_back(session->pending.front());
      session->pending.pop_front();
    }
  }

  const auto started = std::chrono::steady_clock::now();
  double energy = 0.0;
  try {
    const std::lock_guard lock(runtime_mutex);
    mx::set_default_device(session->device);
    mx::set_default_stream(session->cpu_stream);
    mx::set_default_stream(session->device_stream);
    auto input = mx::array(snapshot.begin(),
                           {static_cast<int>(snapshot.size())});
    input = mx::copy(std::move(input), session->device);
    const auto squared = mx::square(input, session->device_stream);
    energy = static_cast<double>(
        mx::sum(squared, false, session->device_stream).item<float>());
  } catch (const std::exception &error) {
    const std::lock_guard lock(session->state_mutex);
    for (auto sample = snapshot.rbegin(); sample != snapshot.rend(); ++sample) {
      session->pending.push_front(*sample);
    }
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
  {
    const std::lock_guard lock(session->state_mutex);
    session->total_ingested_samples += snapshot.size();
    ++session->step_count;
    pending_samples = session->pending.size();
    json = state_json(*session, "progress", snapshot.size(), pending_samples,
                      energy, elapsed_ms);
  }
  return return_json(CUTTLEDOC_VOXTRAL_MLX_OK, json, json_out, error_out);
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
