#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
constexpr std::size_t kTextContext = 448;
constexpr std::size_t kTextState = 384;
constexpr std::size_t kTextHeads = 6;
constexpr std::size_t kTextLayers = 4;
constexpr std::size_t kVocabularySize = 51'865;
constexpr float kLayerNormEpsilon = 1e-5f;

constexpr int32_t kEndOfText = 50'257;
constexpr int32_t kStartOfTranscript = 50'258;
constexpr int32_t kEnglish = 50'259;
constexpr int32_t kTranscribe = 50'359;
constexpr int32_t kTimestampBegin = 50'364;
constexpr int32_t kMergeableTokenCount = 50'257;
constexpr std::size_t kMaximumDecodedTokens = 224;

std::mutex runtime_mutex;

struct TranscriptSegment {
  double start_seconds;
  double end_seconds;
  std::string text;
};

struct DecodedTranscript {
  std::string text;
  std::vector<int32_t> tokens;
  std::vector<TranscriptSegment> segments;
};

std::string decode_base64(std::string_view encoded) {
  static constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string decoded;
  decoded.reserve(encoded.size() * 3 / 4);
  uint32_t accumulator = 0;
  int bits = 0;
  for (const auto character : encoded) {
    if (character == '=') {
      break;
    }
    const auto position = alphabet.find(character);
    if (position == std::string_view::npos) {
      throw std::runtime_error("invalid base64 token in multilingual.tiktoken");
    }
    accumulator = (accumulator << 6) | static_cast<uint32_t>(position);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      decoded.push_back(static_cast<char>((accumulator >> bits) & 0xff));
    }
  }
  return decoded;
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
        escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<int>(static_cast<unsigned char>(character))
                << std::dec;
      } else {
        escaped << character;
      }
    }
  }
  return escaped.str();
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

int32_t fail(const std::string &message, char **error_out) {
  if (error_out != nullptr) {
    *error_out = strdup(message.c_str());
  }
  return 1;
}

void expect_shape(const mx::array &value, const mx::Shape &expected,
                  const std::string &name) {
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

  const mx::Device &device() const { return device_; }

  const char *device_name() const {
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
    json << std::fixed << std::setprecision(3) << "{\"mlx_version\":\""
         << mx::version() << "\","
         << "\"device\":\"" << device_name() << "\","
         << "\"loaded_tensors\":" << weights_.size() - 2 << ","
         << "\"load_peak_memory_bytes\":" << load_peak_memory_bytes_ << ","
         << "\"active_memory_bytes\":" << load_active_memory_bytes_ << ","
         << "\"cache_memory_bytes\":" << load_cache_memory_bytes_ << "}";
    return json.str();
  }

  std::string encode(const float *audio, std::size_t audio_len) const {
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
    auto materialized = mx::contiguous(
        mx::astype(encoded, mx::float32, device_), true, device_);
    mx::eval(materialized);

    const auto finished_at = std::chrono::steady_clock::now();
    const auto inference_ms =
        std::chrono::duration<double, std::milli>(finished_at - started_at)
            .count();
    const auto *values = materialized.data<float>();
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
    const auto variance = std::max(
        0.0, squared_sum / static_cast<double>(value_count) - mean * mean);

    std::ostringstream json;
    json << std::fixed << std::setprecision(9) << "{\"mlx_version\":\""
         << mx::version() << "\","
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

  std::string transcribe(const float *audio, std::size_t audio_len) const {
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
    mx::eval(encoded);
    const auto encoder_finished_at = std::chrono::steady_clock::now();
    const auto transcript =
        decode_audio(encoded, static_cast<double>(audio_len) / kSampleRate);
    const auto finished_at = std::chrono::steady_clock::now();

    const auto encoder_ms = std::chrono::duration<double, std::milli>(
                                encoder_finished_at - started_at)
                                .count();
    const auto decoder_ms = std::chrono::duration<double, std::milli>(
                                finished_at - encoder_finished_at)
                                .count();
    const auto inference_ms =
        std::chrono::duration<double, std::milli>(finished_at - started_at)
            .count();

    std::ostringstream json;
    json << std::fixed << std::setprecision(3) << "{\"mlx_version\":\""
         << mx::version() << "\","
         << "\"device\":\"" << device_name() << "\","
         << "\"language\":\"en\","
         << "\"input_samples\":" << audio_len << ","
         << "\"audio_duration_seconds\":"
         << static_cast<double>(audio_len) / kSampleRate << ","
         << "\"encoder_ms\":" << encoder_ms << ","
         << "\"decoder_ms\":" << decoder_ms << ","
         << "\"inference_ms\":" << inference_ms << ","
         << "\"peak_memory_bytes\":" << mx::get_peak_memory() << ","
         << "\"active_memory_bytes\":" << mx::get_active_memory() << ","
         << "\"cache_memory_bytes\":" << mx::get_cache_memory() << ","
         << "\"text\":\"" << json_escape(transcript.text) << "\","
         << "\"tokens\":[";
    for (std::size_t index = 0; index < transcript.tokens.size(); ++index) {
      if (index != 0) {
        json << ",";
      }
      json << transcript.tokens[index];
    }
    json << "],\"segments\":[";
    for (std::size_t index = 0; index < transcript.segments.size(); ++index) {
      if (index != 0) {
        json << ",";
      }
      const auto &segment = transcript.segments[index];
      json << "{\"start_seconds\":" << segment.start_seconds
           << ",\"end_seconds\":" << segment.end_seconds << ",\"text\":\""
           << json_escape(segment.text) << "\"}";
    }
    json << "]}";
    return json.str();
  }

private:
  const mx::array &weight(const std::string &name) const {
    return weights_.at(name);
  }

  void add_weight(const std::string &name, const mx::Shape &expected_shape) {
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

    add_weight(
        "decoder.token_embedding.weight",
        {static_cast<int>(kVocabularySize), static_cast<int>(kTextState)});
    add_weight("decoder.positional_embedding",
               {static_cast<int>(kTextContext), static_cast<int>(kTextState)});
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
      const auto prefix = "decoder.blocks." + std::to_string(layer) + ".";
      add_weight(prefix + "attn.query.weight", {384, 384});
      add_weight(prefix + "attn.query.bias", {384});
      add_weight(prefix + "attn.key.weight", {384, 384});
      add_weight(prefix + "attn.value.weight", {384, 384});
      add_weight(prefix + "attn.value.bias", {384});
      add_weight(prefix + "attn.out.weight", {384, 384});
      add_weight(prefix + "attn.out.bias", {384});
      add_weight(prefix + "attn_ln.weight", {384});
      add_weight(prefix + "attn_ln.bias", {384});
      add_weight(prefix + "cross_attn.query.weight", {384, 384});
      add_weight(prefix + "cross_attn.query.bias", {384});
      add_weight(prefix + "cross_attn.key.weight", {384, 384});
      add_weight(prefix + "cross_attn.value.weight", {384, 384});
      add_weight(prefix + "cross_attn.value.bias", {384});
      add_weight(prefix + "cross_attn.out.weight", {384, 384});
      add_weight(prefix + "cross_attn.out.bias", {384});
      add_weight(prefix + "cross_attn_ln.weight", {384});
      add_weight(prefix + "cross_attn_ln.bias", {384});
      add_weight(prefix + "mlp1.weight", {1536, 384});
      add_weight(prefix + "mlp1.bias", {1536});
      add_weight(prefix + "mlp2.weight", {384, 1536});
      add_weight(prefix + "mlp2.bias", {384});
      add_weight(prefix + "mlp_ln.weight", {384});
      add_weight(prefix + "mlp_ln.bias", {384});
    }
    add_weight("decoder.ln.weight", {384});
    add_weight("decoder.ln.bias", {384});
    add_weight("mel_80", {80, 201});
    load_tokenizer();

    const auto log_timescale_increment =
        std::log(10'000.0f) / (static_cast<float>(kAudioState / 2) - 1.0f);
    auto inv_timescales = mx::exp(
        -log_timescale_increment *
            mx::arange(static_cast<int>(kAudioState / 2), mx::float32, device_),
        device_);
    auto scaled_time =
        mx::expand_dims(
            mx::arange(static_cast<int>(kAudioContext), mx::float32, device_),
            1, device_) *
        mx::expand_dims(inv_timescales, 0, device_);
    auto positional_embedding =
        mx::astype(mx::concatenate({mx::sin(scaled_time, device_),
                                    mx::cos(scaled_time, device_)},
                                   1, device_),
                   compute_dtype(), device_);
    weights_.emplace("__positional_embedding", std::move(positional_embedding));

    std::vector<mx::array> tensors;
    tensors.reserve(weights_.size());
    for (const auto &[name, tensor] : weights_) {
      static_cast<void>(name);
      tensors.push_back(tensor);
    }
    mx::eval(std::move(tensors));
    load_peak_memory_bytes_ = mx::get_peak_memory();
    load_active_memory_bytes_ = mx::get_active_memory();
    load_cache_memory_bytes_ = mx::get_cache_memory();
  }

  void load_tokenizer() {
    const auto path = model_directory_ / "multilingual.tiktoken";
    std::ifstream input(path);
    if (!input) {
      throw std::runtime_error("missing tokenizer vocabulary: " +
                               path.string());
    }
    token_bytes_.assign(kMergeableTokenCount, {});
    std::string encoded;
    int32_t rank = -1;
    std::size_t count = 0;
    while (input >> encoded >> rank) {
      if (rank < 0 || rank >= kMergeableTokenCount) {
        throw std::runtime_error("invalid tokenizer rank");
      }
      token_bytes_[static_cast<std::size_t>(rank)] = decode_base64(encoded);
      ++count;
    }
    if (count != kMergeableTokenCount) {
      throw std::runtime_error(
          "expected 50257 mergeable tokenizer entries, found " +
          std::to_string(count));
    }
  }

  mx::array log_mel(const float *audio, std::size_t audio_len) const {
    std::vector<float> padded_audio(kChunkSamples + kFftSize, 0.0f);
    std::copy_n(audio, audio_len, padded_audio.begin() + kFftSize / 2);

    for (std::size_t index = 0; index < kFftSize / 2; ++index) {
      padded_audio[index] = padded_audio[kFftSize - index];
      padded_audio[kFftSize / 2 + kChunkSamples + index] =
          padded_audio[kFftSize / 2 + kChunkSamples - 2 - index];
    }

    std::vector<float> window(kFftSize);
    constexpr auto pi = 3.14159265358979323846;
    for (std::size_t index = 0; index < kFftSize; ++index) {
      window[index] =
          0.5f - 0.5f * std::cos(2.0 * pi * static_cast<double>(index) /
                                 static_cast<double>(kFftSize));
    }

    auto audio_array = mx::array(padded_audio.begin(),
                                 {static_cast<int>(padded_audio.size())});
    auto frames = mx::as_strided(
        audio_array,
        {static_cast<int>(kMelFrames + 1), static_cast<int>(kFftSize)},
        {static_cast<int64_t>(kHopLength), 1}, 0, device_);
    auto window_array =
        mx::array(window.begin(), {static_cast<int>(window.size())});
    auto frequencies = mx::fft::rfft(frames * window_array, -1,
                                     mx::fft::FFTNorm::Backward, device_);
    frequencies = mx::slice(
        frequencies, {0, 0},
        {static_cast<int>(kMelFrames), static_cast<int>(kFftSize / 2 + 1)},
        {1, 1}, device_);
    auto magnitudes = mx::square(mx::abs(frequencies, device_), device_);
    auto mel = mx::matmul(
        magnitudes, mx::transpose(weight("mel_80"), {1, 0}, device_), device_);
    auto log_spec =
        mx::log10(mx::maximum(mel, mx::array(1e-10f), device_), device_);
    log_spec = mx::maximum(log_spec, mx::max(log_spec, device_) - 8.0, device_);
    log_spec = (log_spec + 4.0) / 4.0;
    return mx::expand_dims(mx::astype(log_spec, compute_dtype(), device_), 0,
                           device_);
  }

  mx::array linear(const mx::array &input, const std::string &prefix,
                   bool has_bias = true) const {
    if (has_bias) {
      return mx::addmm(weight(prefix + ".bias"), input,
                       mx::transpose(weight(prefix + ".weight"), device_), 1.0f,
                       1.0f, device_);
    }
    return mx::matmul(input, mx::transpose(weight(prefix + ".weight"), device_),
                      device_);
  }

  mx::array layer_norm(const mx::array &input,
                       const std::string &prefix) const {
    return mx::fast::layer_norm(input, weight(prefix + ".weight"),
                                weight(prefix + ".bias"), kLayerNormEpsilon,
                                device_);
  }

  mx::array gelu(const mx::array &input) const {
    const auto one = mx::array(1.0f, input.dtype());
    const auto inverse_sqrt_two =
        mx::array(1.0f / std::sqrt(2.0f), input.dtype());
    const auto half = mx::array(0.5f, input.dtype());
    return mx::multiply(
        input,
        mx::multiply(one +
                         mx::erf(mx::multiply(input, inverse_sqrt_two, device_),
                                 device_),
                     half, device_),
        device_);
  }

  mx::array self_attention(const mx::array &input,
                           const std::string &prefix) const {
    auto query = linear(input, prefix + ".query");
    auto key = linear(input, prefix + ".key", false);
    auto value = linear(input, prefix + ".value");
    const auto scale = mx::array(
        std::pow(static_cast<float>(kAudioState / kAudioHeads), -0.25f),
        query.dtype());

    query = mx::multiply(
        mx::transpose(mx::reshape(query,
                                  {1, static_cast<int>(kAudioContext),
                                   static_cast<int>(kAudioHeads),
                                   static_cast<int>(kAudioState / kAudioHeads)},
                                  device_),
                      {0, 2, 1, 3}, device_),
        scale, device_);
    key = mx::multiply(
        mx::transpose(mx::reshape(key,
                                  {1, static_cast<int>(kAudioContext),
                                   static_cast<int>(kAudioHeads),
                                   static_cast<int>(kAudioState / kAudioHeads)},
                                  device_),
                      {0, 2, 3, 1}, device_),
        scale, device_);
    value =
        mx::transpose(mx::reshape(value,
                                  {1, static_cast<int>(kAudioContext),
                                   static_cast<int>(kAudioHeads),
                                   static_cast<int>(kAudioState / kAudioHeads)},
                                  device_),
                      {0, 2, 1, 3}, device_);

    auto attention_weights =
        mx::softmax(mx::matmul(query, key, device_), -1, true, device_);
    auto attended = mx::matmul(attention_weights, value, device_);
    attended = mx::reshape(
        mx::transpose(attended, {0, 2, 1, 3}, device_),
        {1, static_cast<int>(kAudioContext), static_cast<int>(kAudioState)},
        device_);
    return linear(attended, prefix + ".out");
  }

  mx::array encode_audio(const mx::array &mel) const {
    auto encoded = gelu(
        mx::conv1d(mel, weight("encoder.conv1.weight"), 1, 1, 1, 1, device_) +
        weight("encoder.conv1.bias"));
    encoded = gelu(mx::conv1d(encoded, weight("encoder.conv2.weight"), 2, 1, 1,
                              1, device_) +
                   weight("encoder.conv2.bias"));
    if (encoded.shape() != mx::Shape{1, 1500, 384}) {
      throw std::runtime_error("Whisper convolution produced an invalid shape");
    }
    encoded = encoded + weight("__positional_embedding");

    for (std::size_t layer = 0; layer < kAudioLayers; ++layer) {
      const auto prefix = "encoder.blocks." + std::to_string(layer);
      encoded =
          encoded + self_attention(layer_norm(encoded, prefix + ".attn_ln"),
                                   prefix + ".attn");
      const auto normalized = layer_norm(encoded, prefix + ".mlp_ln");
      encoded = encoded + linear(gelu(linear(normalized, prefix + ".mlp1")),
                                 prefix + ".mlp2");
    }
    encoded = layer_norm(encoded, "encoder.ln_post");
    return encoded;
  }

  mx::array text_attention(const mx::array &input, const mx::array &source,
                           const std::string &prefix, bool causal) const {
    const auto query_context = input.shape(1);
    const auto key_context = source.shape(1);
    auto query = linear(input, prefix + ".query");
    auto key = linear(source, prefix + ".key", false);
    auto value = linear(source, prefix + ".value");
    const auto scale =
        mx::array(std::pow(static_cast<float>(kTextState / kTextHeads), -0.25f),
                  query.dtype());

    query = mx::multiply(
        mx::transpose(
            mx::reshape(query,
                        {1, query_context, static_cast<int>(kTextHeads),
                         static_cast<int>(kTextState / kTextHeads)},
                        device_),
            {0, 2, 1, 3}, device_),
        scale, device_);
    key = mx::multiply(
        mx::transpose(mx::reshape(key,
                                  {1, key_context, static_cast<int>(kTextHeads),
                                   static_cast<int>(kTextState / kTextHeads)},
                                  device_),
                      {0, 2, 3, 1}, device_),
        scale, device_);
    value =
        mx::transpose(mx::reshape(value,
                                  {1, key_context, static_cast<int>(kTextHeads),
                                   static_cast<int>(kTextState / kTextHeads)},
                                  device_),
                      {0, 2, 1, 3}, device_);

    auto scores = mx::matmul(query, key, device_);
    if (causal) {
      std::vector<float> mask(
          static_cast<std::size_t>(query_context * query_context), 0.0f);
      for (int row = 0; row < query_context; ++row) {
        for (int column = row + 1; column < query_context; ++column) {
          mask[static_cast<std::size_t>(row * query_context + column)] =
              -std::numeric_limits<float>::infinity();
        }
      }
      auto mask_array = mx::array(mask.begin(), {query_context, query_context});
      scores = scores + mx::astype(mask_array, scores.dtype(), device_);
    }
    const auto attention_weights = mx::softmax(scores, -1, true, device_);
    auto attended = mx::matmul(attention_weights, value, device_);
    attended =
        mx::reshape(mx::transpose(attended, {0, 2, 1, 3}, device_),
                    {1, query_context, static_cast<int>(kTextState)}, device_);
    return linear(attended, prefix + ".out");
  }

  mx::array decoder_logits(const std::vector<int32_t> &tokens,
                           const mx::array &audio_features) const {
    const auto context = static_cast<int>(tokens.size());
    if (context == 0 || context > static_cast<int>(kTextContext)) {
      throw std::runtime_error("Whisper decoder token context is invalid");
    }
    auto token_ids = mx::array(tokens.begin(), {1, context});
    auto decoded = mx::take(weight("decoder.token_embedding.weight"), token_ids,
                            0, device_);
    decoded =
        decoded + mx::slice(weight("decoder.positional_embedding"), {0, 0},
                            {context, static_cast<int>(kTextState)}, device_);

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
      const auto prefix = "decoder.blocks." + std::to_string(layer);
      const auto self_normalized = layer_norm(decoded, prefix + ".attn_ln");
      decoded = decoded + text_attention(self_normalized, self_normalized,
                                         prefix + ".attn", true);
      const auto cross_normalized =
          layer_norm(decoded, prefix + ".cross_attn_ln");
      decoded = decoded + text_attention(cross_normalized, audio_features,
                                         prefix + ".cross_attn", false);
      const auto mlp_normalized = layer_norm(decoded, prefix + ".mlp_ln");
      decoded = decoded + linear(gelu(linear(mlp_normalized, prefix + ".mlp1")),
                                 prefix + ".mlp2");
    }
    decoded = layer_norm(decoded, "decoder.ln");
    auto logits = mx::matmul(
        decoded,
        mx::transpose(weight("decoder.token_embedding.weight"), device_),
        device_);
    return mx::reshape(
        mx::slice(logits, {0, context - 1, 0},
                  {1, context, static_cast<int>(kVocabularySize)}, {1, 1, 1},
                  device_),
        {static_cast<int>(kVocabularySize)}, device_);
  }

  int32_t select_next_token(const mx::array &logits,
                            const std::vector<int32_t> &generated) const {
    auto materialized =
        mx::contiguous(mx::astype(logits, mx::float32, device_), true, device_);
    if (device_ == mx::Device::gpu) {
      materialized = mx::copy(std::move(materialized), mx::Device::cpu);
    }
    mx::eval(materialized);
    const auto *values = materialized.data<float>();
    auto selected = -1;
    auto selected_logit = -std::numeric_limits<float>::infinity();
    const auto consider = [&](int32_t begin, int32_t end) {
      for (auto token = begin; token < end; ++token) {
        if (values[token] > selected_logit) {
          selected = token;
          selected_logit = values[token];
        }
      }
    };

    if (generated.empty()) {
      consider(kTimestampBegin, kTimestampBegin + 51);
      return selected;
    }

    const auto last = generated.back();
    if (last >= kTimestampBegin) {
      const auto penultimate_is_timestamp =
          generated.size() < 2 ||
          generated[generated.size() - 2] >= kTimestampBegin;
      if (penultimate_is_timestamp) {
        consider(0, kEndOfText);
      } else {
        consider(kEndOfText, kEndOfText + 1);
        consider(std::max(last, kTimestampBegin),
                 static_cast<int32_t>(kVocabularySize));
      }
      return selected;
    }

    auto latest_timestamp = kTimestampBegin;
    for (auto iterator = generated.rbegin(); iterator != generated.rend();
         ++iterator) {
      if (*iterator >= kTimestampBegin) {
        latest_timestamp = *iterator;
        break;
      }
    }

    consider(0, kEndOfText + 1);
    const auto best_text_token = selected;
    const auto best_text_logit = selected_logit;

    auto best_timestamp_token = -1;
    auto best_timestamp_logit = -std::numeric_limits<float>::infinity();
    double timestamp_exp_sum = 0.0;
    for (auto token = latest_timestamp;
         token < static_cast<int32_t>(kVocabularySize); ++token) {
      if (values[token] > best_timestamp_logit) {
        best_timestamp_token = token;
        best_timestamp_logit = values[token];
      }
    }
    for (auto token = latest_timestamp;
         token < static_cast<int32_t>(kVocabularySize); ++token) {
      timestamp_exp_sum +=
          std::exp(static_cast<double>(values[token] - best_timestamp_logit));
    }
    const auto timestamp_log_probability =
        static_cast<double>(best_timestamp_logit) + std::log(timestamp_exp_sum);
    if (timestamp_log_probability > best_text_logit) {
      return best_timestamp_token;
    }
    return best_text_token;
  }

  std::string decode_text_tokens(const std::vector<int32_t> &tokens) const {
    std::string decoded;
    for (const auto token : tokens) {
      if (token >= 0 && token < kMergeableTokenCount) {
        decoded += token_bytes_[static_cast<std::size_t>(token)];
      }
    }
    return decoded;
  }

  DecodedTranscript decode_audio(const mx::array &audio_features,
                                 double audio_duration_seconds) const {
    std::vector<int32_t> context = {
        kStartOfTranscript,
        kEnglish,
        kTranscribe,
    };
    std::vector<int32_t> generated;
    generated.reserve(kMaximumDecodedTokens);
    for (std::size_t index = 0; index < kMaximumDecodedTokens; ++index) {
      const auto next =
          select_next_token(decoder_logits(context, audio_features), generated);
      if (next < 0) {
        throw std::runtime_error("Whisper decoder found no valid next token");
      }
      context.push_back(next);
      generated.push_back(next);
      if (next == kEndOfText) {
        break;
      }
    }
    if (generated.empty() || generated.back() != kEndOfText) {
      throw std::runtime_error(
          "Whisper decoder reached its token limit before end-of-text");
    }

    std::vector<TranscriptSegment> segments;
    std::vector<int32_t> text_tokens;
    double segment_start = 0.0;
    for (const auto token : generated) {
      if (token >= kTimestampBegin) {
        const auto timestamp =
            std::min(audio_duration_seconds,
                     static_cast<double>(token - kTimestampBegin) * 0.02);
        if (!text_tokens.empty()) {
          auto text = trim(decode_text_tokens(text_tokens));
          if (!text.empty()) {
            segments.push_back(
                {segment_start, std::max(segment_start, timestamp), text});
          }
          text_tokens.clear();
        }
        segment_start = timestamp;
      } else if (token >= 0 && token < kMergeableTokenCount) {
        text_tokens.push_back(token);
      }
    }
    if (!text_tokens.empty()) {
      auto text = trim(decode_text_tokens(text_tokens));
      if (!text.empty()) {
        segments.push_back({segment_start, audio_duration_seconds, text});
      }
    }

    std::string text;
    for (const auto &segment : segments) {
      if (!text.empty()) {
        text += " ";
      }
      text += segment.text;
    }
    return {std::move(text), std::move(generated), std::move(segments)};
  }

  std::filesystem::path model_directory_;
  mx::Device device_;
  std::unordered_map<std::string, mx::array> weights_;
  std::vector<std::string> token_bytes_;
  std::size_t load_peak_memory_bytes_ = 0;
  std::size_t load_active_memory_bytes_ = 0;
  std::size_t load_cache_memory_bytes_ = 0;
};

WhisperEncoder *as_encoder(void *handle) {
  if (handle == nullptr) {
    throw std::invalid_argument("encoder handle must be non-null");
  }
  return static_cast<WhisperEncoder *>(handle);
}

int32_t copy_json(const std::string &json, char **json_out, char **error_out) {
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

extern "C" void *cuttledoc_mlx_whisper_create(const char *model_directory,
                                              int32_t device_kind,
                                              char **error_out) {
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
  } catch (const std::exception &error) {
    fail(error.what(), error_out);
  } catch (...) {
    fail("MLX raised a non-standard exception while loading the model",
         error_out);
  }
  return nullptr;
}

extern "C" int32_t cuttledoc_mlx_whisper_describe(void *handle, char **json_out,
                                                  char **error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  try {
    std::scoped_lock lock(runtime_mutex);
    return copy_json(as_encoder(handle)->load_description(), json_out,
                     error_out);
  } catch (const std::exception &error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("could not describe MLX encoder", error_out);
  }
}

extern "C" int32_t cuttledoc_mlx_whisper_transcribe(void *handle,
                                                    const float *audio,
                                                    std::size_t audio_len,
                                                    char **json_out,
                                                    char **error_out) {
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  try {
    std::scoped_lock lock(runtime_mutex);
    auto *encoder = as_encoder(handle);
    mx::set_default_device(encoder->device());
    return copy_json(encoder->transcribe(audio, audio_len), json_out,
                     error_out);
  } catch (const std::exception &error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("MLX raised a non-standard exception during transcription",
                error_out);
  }
}

extern "C" void cuttledoc_mlx_whisper_destroy(void *handle) {
  if (handle == nullptr) {
    return;
  }
  std::scoped_lock lock(runtime_mutex);
  delete static_cast<WhisperEncoder *>(handle);
  mx::clear_cache();
}

extern "C" void cuttledoc_mlx_free_string(char *value) { std::free(value); }
