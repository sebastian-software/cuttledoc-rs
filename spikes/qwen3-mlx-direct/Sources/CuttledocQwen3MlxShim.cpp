#include <algorithm>
#include <array>
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
#include <numeric>
#include <optional>
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
constexpr std::size_t kAudioState = 896;
constexpr std::size_t kAudioHeads = 14;
constexpr std::size_t kAudioHeadDimension = kAudioState / kAudioHeads;
constexpr std::size_t kAudioEncoderLayers = 18;
constexpr float kAudioLayerNormEpsilon = 1e-5f;
constexpr int32_t kImStartToken = 151644;
constexpr int32_t kImEndToken = 151645;
constexpr int32_t kAudioStartToken = 151669;
constexpr int32_t kAudioEndToken = 151670;
constexpr int32_t kAudioPadToken = 151676;
constexpr int32_t kAsrTextToken = 151704;

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
  std::string source_dtype = "other";
  if (value.dtype() == mx::float32) {
    source_dtype = "float32";
  } else if (value.dtype() == mx::bfloat16) {
    source_dtype = "bfloat16";
  } else if (value.dtype() == mx::float16) {
    source_dtype = "float16";
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
  json << std::setprecision(17) << "{\"shape\":" << shape_string(materialized)
       << ",\"source_dtype\":\"" << source_dtype << "\",\"mean\":" << mean
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

void append_utf8(std::string &output, std::uint32_t codepoint) {
  if (codepoint <= 0x7f) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ff) {
    output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
}

int hexadecimal_digit(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  throw std::runtime_error("invalid hexadecimal digit in vocab.json");
}

std::uint32_t parse_json_code_unit(std::string_view input,
                                   std::size_t &position) {
  if (position + 4 > input.size()) {
    throw std::runtime_error("truncated Unicode escape in vocab.json");
  }
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < 4; ++index) {
    value = (value << 4) |
            static_cast<std::uint32_t>(hexadecimal_digit(input[position++]));
  }
  return value;
}

std::string parse_json_string(std::string_view input, std::size_t &position) {
  if (position >= input.size() || input[position++] != '"') {
    throw std::runtime_error("expected JSON string in vocab.json");
  }
  std::string output;
  while (position < input.size()) {
    const auto current = input[position++];
    if (current == '"') {
      return output;
    }
    if (current != '\\') {
      output.push_back(current);
      continue;
    }
    if (position >= input.size()) {
      throw std::runtime_error("truncated escape in vocab.json");
    }
    switch (const auto escaped = input[position++]) {
    case '"':
    case '\\':
    case '/':
      output.push_back(escaped);
      break;
    case 'b':
      output.push_back('\b');
      break;
    case 'f':
      output.push_back('\f');
      break;
    case 'n':
      output.push_back('\n');
      break;
    case 'r':
      output.push_back('\r');
      break;
    case 't':
      output.push_back('\t');
      break;
    case 'u': {
      auto codepoint = parse_json_code_unit(input, position);
      if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
        if (position + 2 > input.size() || input[position] != '\\' ||
            input[position + 1] != 'u') {
          throw std::runtime_error("unpaired high surrogate in vocab.json");
        }
        position += 2;
        const auto low = parse_json_code_unit(input, position);
        if (low < 0xdc00 || low > 0xdfff) {
          throw std::runtime_error("invalid low surrogate in vocab.json");
        }
        codepoint =
            0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
      }
      append_utf8(output, codepoint);
      break;
    }
    default:
      throw std::runtime_error("unsupported escape in vocab.json");
    }
  }
  throw std::runtime_error("unterminated JSON string in vocab.json");
}

void skip_json_whitespace(std::string_view input, std::size_t &position) {
  while (position < input.size() &&
         (input[position] == ' ' || input[position] == '\n' ||
          input[position] == '\r' || input[position] == '\t')) {
    ++position;
  }
}

std::unordered_map<std::string, int32_t>
load_vocabulary(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("missing tokenizer vocabulary: " + path.string());
  }
  const std::string input((std::istreambuf_iterator<char>(stream)),
                          std::istreambuf_iterator<char>());
  std::size_t position = 0;
  skip_json_whitespace(input, position);
  if (position >= input.size() || input[position++] != '{') {
    throw std::runtime_error("vocab.json must contain an object");
  }
  std::unordered_map<std::string, int32_t> vocabulary;
  while (true) {
    skip_json_whitespace(input, position);
    if (position < input.size() && input[position] == '}') {
      ++position;
      break;
    }
    const auto token = parse_json_string(input, position);
    skip_json_whitespace(input, position);
    if (position >= input.size() || input[position++] != ':') {
      throw std::runtime_error("expected ':' in vocab.json");
    }
    skip_json_whitespace(input, position);
    if (position >= input.size() || input[position] < '0' ||
        input[position] > '9') {
      throw std::runtime_error("expected integer token id in vocab.json");
    }
    int32_t token_id = 0;
    while (position < input.size() && input[position] >= '0' &&
           input[position] <= '9') {
      token_id = token_id * 10 + (input[position++] - '0');
    }
    if (!vocabulary.emplace(token, token_id).second) {
      throw std::runtime_error("duplicate token in vocab.json");
    }
    skip_json_whitespace(input, position);
    if (position < input.size() && input[position] == ',') {
      ++position;
      continue;
    }
    if (position < input.size() && input[position] == '}') {
      ++position;
      break;
    }
    throw std::runtime_error("expected ',' or '}' in vocab.json");
  }
  if (vocabulary.size() != 151'643) {
    throw std::runtime_error("unexpected base vocabulary size: " +
                             std::to_string(vocabulary.size()));
  }
  return vocabulary;
}

std::string pair_key(const std::string &left, const std::string &right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back('\0');
  key.append(right);
  return key;
}

class QwenTokenizer {
public:
  explicit QwenTokenizer(const std::filesystem::path &model_directory)
      : vocabulary_(load_vocabulary(model_directory / "vocab.json")),
        byte_symbols_(make_byte_symbols()) {
    std::ifstream merges(model_directory / "merges.txt");
    if (!merges) {
      throw std::runtime_error("missing tokenizer merges: " +
                               (model_directory / "merges.txt").string());
    }
    std::string line;
    std::size_t rank = 0;
    while (std::getline(merges, line)) {
      if (line.empty() || line.starts_with("#version:")) {
        continue;
      }
      const auto separator = line.find(' ');
      if (separator == std::string::npos || separator == 0 ||
          separator + 1 >= line.size()) {
        throw std::runtime_error("invalid tokenizer merge line");
      }
      merge_ranks_[pair_key(line.substr(0, separator),
                            line.substr(separator + 1))] = rank++;
    }
    if (rank != 151'387 || merge_ranks_.size() != 151'387) {
      throw std::runtime_error(
          "unexpected tokenizer merge layout: lines=" + std::to_string(rank) +
          ", unique=" + std::to_string(merge_ranks_.size()));
    }
  }

  std::vector<int32_t> build_asr_prompt(std::size_t audio_tokens,
                                        std::string_view language) const {
    if (language.empty()) {
      throw std::runtime_error("language must be non-empty");
    }
    std::vector<int32_t> tokens;
    tokens.reserve(audio_tokens + 18);
    tokens.push_back(kImStartToken);
    append_piece(tokens, "system");
    append_piece(tokens, "\n");
    tokens.push_back(kImEndToken);
    append_piece(tokens, "\n");
    tokens.push_back(kImStartToken);
    append_piece(tokens, "user");
    append_piece(tokens, "\n");
    tokens.push_back(kAudioStartToken);
    tokens.insert(tokens.end(), audio_tokens, kAudioPadToken);
    tokens.push_back(kAudioEndToken);
    tokens.push_back(kImEndToken);
    append_piece(tokens, "\n");
    tokens.push_back(kImStartToken);
    append_piece(tokens, "assistant");
    append_piece(tokens, "\n");
    append_piece(tokens, "language");
    append_piece(tokens, " " + std::string(language));
    tokens.push_back(kAsrTextToken);
    return tokens;
  }

private:
  static std::array<std::string, 256> make_byte_symbols() {
    std::vector<int> bytes;
    for (int value = '!'; value <= '~'; ++value) {
      bytes.push_back(value);
    }
    for (int value = 0xa1; value <= 0xac; ++value) {
      bytes.push_back(value);
    }
    for (int value = 0xae; value <= 0xff; ++value) {
      bytes.push_back(value);
    }
    std::vector<std::uint32_t> codepoints(bytes.begin(), bytes.end());
    std::array<bool, 256> included{};
    for (const auto value : bytes) {
      included[static_cast<std::size_t>(value)] = true;
    }
    std::uint32_t extra = 0;
    for (int value = 0; value < 256; ++value) {
      if (!included[static_cast<std::size_t>(value)]) {
        bytes.push_back(value);
        codepoints.push_back(256 + extra++);
      }
    }
    std::array<std::string, 256> symbols;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      append_utf8(symbols[static_cast<std::size_t>(bytes[index])],
                  codepoints[index]);
    }
    return symbols;
  }

  void append_piece(std::vector<int32_t> &output,
                    const std::string &piece) const {
    const auto cached = cache_.find(piece);
    if (cached != cache_.end()) {
      output.insert(output.end(), cached->second.begin(), cached->second.end());
      return;
    }
    std::vector<std::string> word;
    word.reserve(piece.size());
    for (const auto byte : piece) {
      word.push_back(byte_symbols_[static_cast<unsigned char>(byte)]);
    }
    while (word.size() > 1) {
      auto best_rank = std::numeric_limits<std::size_t>::max();
      std::string best_pair;
      for (std::size_t index = 0; index + 1 < word.size(); ++index) {
        const auto key = pair_key(word[index], word[index + 1]);
        const auto found = merge_ranks_.find(key);
        if (found != merge_ranks_.end() && found->second < best_rank) {
          best_rank = found->second;
          best_pair = key;
        }
      }
      if (best_pair.empty()) {
        break;
      }
      std::vector<std::string> merged;
      for (std::size_t index = 0; index < word.size();) {
        if (index + 1 < word.size() &&
            pair_key(word[index], word[index + 1]) == best_pair) {
          merged.push_back(word[index] + word[index + 1]);
          index += 2;
        } else {
          merged.push_back(word[index++]);
        }
      }
      word = std::move(merged);
    }
    std::vector<int32_t> encoded;
    encoded.reserve(word.size());
    for (const auto &symbol : word) {
      const auto found = vocabulary_.find(symbol);
      if (found == vocabulary_.end()) {
        throw std::runtime_error("BPE result is absent from vocab.json");
      }
      encoded.push_back(found->second);
    }
    cache_.emplace(piece, encoded);
    output.insert(output.end(), encoded.begin(), encoded.end());
  }

  std::unordered_map<std::string, int32_t> vocabulary_;
  std::unordered_map<std::string, std::size_t> merge_ranks_;
  std::array<std::string, 256> byte_symbols_;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

class Qwen3AudioEncoder {
public:
  Qwen3AudioEncoder(const std::filesystem::path &model_directory,
                    mx::Device device)
      : model_directory_(model_directory), device_(device),
        mel_filters_(make_slaney_mel_filters()) {
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
             {"audio_tower.ln_post.weight", {896}},
             {"audio_tower.ln_post.bias", {896}},
             {"audio_tower.proj1.weight", {896, 896}},
             {"audio_tower.proj1.bias", {896}},
             {"audio_tower.proj2.weight", {1024, 896}},
             {"audio_tower.proj2.bias", {1024}},
             {"model.embed_tokens.weight", {151936, 256}},
             {"model.embed_tokens.scales", {151936, 16}},
             {"model.embed_tokens.biases", {151936, 16}},
         }) {
      expect_shape(weights_, name, shape);
    }
    for (std::size_t layer = 0; layer < kAudioEncoderLayers; ++layer) {
      const auto prefix =
          "audio_tower.layers." + std::to_string(layer) + ".";
      for (const auto &[suffix, shape] :
           std::vector<std::pair<std::string, mx::Shape>>{
               {"self_attn.q_proj.weight", {896, 896}},
               {"self_attn.q_proj.bias", {896}},
               {"self_attn.k_proj.weight", {896, 896}},
               {"self_attn.k_proj.bias", {896}},
               {"self_attn.v_proj.weight", {896, 896}},
               {"self_attn.v_proj.bias", {896}},
               {"self_attn.out_proj.weight", {896, 896}},
               {"self_attn.out_proj.bias", {896}},
               {"self_attn_layer_norm.weight", {896}},
               {"self_attn_layer_norm.bias", {896}},
               {"fc1.weight", {3584, 896}},
               {"fc1.bias", {3584}},
               {"fc2.weight", {896, 3584}},
               {"fc2.bias", {896}},
               {"final_layer_norm.weight", {896}},
               {"final_layer_norm.bias", {896}},
           }) {
        expect_shape(weights_, prefix + suffix, shape);
      }
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

  std::string probe_encoder(const float *audio,
                            std::size_t audio_len) const {
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

    hidden = hidden + mx::expand_dims(positional_embedding(frames), 0, device_);
    std::vector<mx::array> valid_chunks;
    std::vector<std::size_t> post_convolution_lengths;
    std::size_t aftercnn_length = 0;
    for (std::size_t index = 0; index < chunk_lengths.size(); ++index) {
      const auto valid_length = (chunk_lengths[index] + 7) / 8;
      post_convolution_lengths.push_back(valid_length);
      aftercnn_length += valid_length;
      auto chunk = mx::slice(
          hidden, {static_cast<int>(index), 0, 0},
          {static_cast<int>(index + 1), static_cast<int>(valid_length),
           static_cast<int>(kAudioState)},
          {1, 1, 1}, device_);
      valid_chunks.push_back(mx::squeeze(chunk, 0, device_));
    }
    hidden = mx::concatenate(std::move(valid_chunks), 0, device_);
    const auto encoder_input_fingerprint = fingerprint_json(hidden, device_);

    const auto attention_window = static_cast<std::size_t>(frames) * 8;
    std::vector<std::size_t> attention_windows{0};
    for (std::size_t position = 0; position < aftercnn_length;) {
      position = std::min(position + attention_window, aftercnn_length);
      attention_windows.push_back(position);
    }
    const auto attention_mask =
        make_attention_mask(aftercnn_length, attention_windows);

    hidden = mx::expand_dims(hidden, 0, device_);
    std::optional<std::string> layer_zero_fingerprint;
    std::optional<std::string> layer_final_fingerprint;
    for (std::size_t layer = 0; layer < kAudioEncoderLayers; ++layer) {
      hidden = encoder_layer(hidden, layer, attention_mask);
      if (layer == 0) {
        layer_zero_fingerprint = fingerprint_json(hidden, device_);
      } else if (layer == kAudioEncoderLayers - 1) {
        layer_final_fingerprint = fingerprint_json(hidden, device_);
      }
    }

    hidden = mx::squeeze(hidden, 0, device_);
    hidden = layer_norm(hidden, "audio_tower.ln_post");
    hidden = gelu(linear(hidden, "audio_tower.proj1"));
    hidden = linear(hidden, "audio_tower.proj2");
    const auto audio_features_fingerprint = fingerprint_json(hidden, device_);

    std::ostringstream json;
    json << std::setprecision(17)
         << "{\"status\":\"ok\",\"boundary\":\"official-mlx-cpp\","
         << "\"stage\":\"qwen3-audio-encoder\","
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
    json << "],\"post_convolution_lengths\":[";
    for (std::size_t index = 0; index < post_convolution_lengths.size();
         ++index) {
      if (index != 0) {
        json << ",";
      }
      json << post_convolution_lengths[index];
    }
    json << "],\"aftercnn_length\":" << aftercnn_length
         << ",\"attention_windows\":[";
    for (std::size_t index = 0; index < attention_windows.size(); ++index) {
      if (index != 0) {
        json << ",";
      }
      json << attention_windows[index];
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
         << ",\"conv_out\":" << conv_out_fingerprint
         << ",\"encoder_input\":" << encoder_input_fingerprint
         << ",\"encoder_layer_0\":" << layer_zero_fingerprint.value()
         << ",\"encoder_layer_17\":" << layer_final_fingerprint.value()
         << ",\"audio_features\":" << audio_features_fingerprint << "}}";
    return json.str();
  }

  std::string probe_prompt(const float *audio, std::size_t audio_len,
                           std::string_view language) const {
    const auto started = std::chrono::steady_clock::now();
    const auto audio_features = encode_audio_features(audio, audio_len);
    const auto audio_token_count =
        static_cast<std::size_t>(audio_features.shape(0));
    const QwenTokenizer tokenizer(model_directory_);
    const auto token_ids =
        tokenizer.build_asr_prompt(audio_token_count, language);
    const auto token_array =
        mx::array(token_ids.begin(), {1, static_cast<int>(token_ids.size())});

    const auto quantized_weights =
        mx::take(weight("model.embed_tokens.weight"), token_array, 0, device_);
    const auto scales =
        mx::take(weight("model.embed_tokens.scales"), token_array, 0, device_);
    const auto biases =
        mx::take(weight("model.embed_tokens.biases"), token_array, 0, device_);
    const auto token_embeddings = mx::dequantize(
        quantized_weights, scales, biases, 64, 8, "affine", std::nullopt,
        std::nullopt, device_);
    const auto audio_features_bfloat16 =
        mx::astype(audio_features, token_embeddings.dtype(), device_);

    const auto flat_embeddings =
        mx::squeeze(token_embeddings, 0, device_);
    std::vector<mx::array> merged_rows;
    merged_rows.reserve(token_ids.size());
    std::vector<std::size_t> audio_token_indices;
    std::size_t audio_index = 0;
    for (std::size_t token_index = 0; token_index < token_ids.size();
         ++token_index) {
      if (token_ids[token_index] == kAudioPadToken) {
        if (audio_index >= audio_token_count) {
          throw std::runtime_error(
              "prompt contains more audio tokens than encoder features");
        }
        merged_rows.push_back(mx::take(audio_features_bfloat16,
                                       static_cast<int>(audio_index++), 0,
                                       device_));
        audio_token_indices.push_back(token_index);
      } else {
        merged_rows.push_back(mx::take(flat_embeddings,
                                       static_cast<int>(token_index), 0,
                                       device_));
      }
    }
    if (audio_index != audio_token_count) {
      throw std::runtime_error(
          "prompt contains fewer audio tokens than encoder features");
    }
    const auto inputs_embeds =
        mx::expand_dims(mx::stack(merged_rows, 0, device_), 0, device_);

    const auto token_embeddings_fingerprint =
        fingerprint_json(token_embeddings, device_);
    const auto audio_features_fingerprint =
        fingerprint_json(audio_features_bfloat16, device_);
    const auto inputs_embeds_fingerprint =
        fingerprint_json(inputs_embeds, device_);
    const auto elapsed =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();

    std::ostringstream json;
    json << std::setprecision(17)
         << "{\"status\":\"ok\",\"boundary\":\"official-mlx-cpp\","
         << "\"stage\":\"qwen3-prompt-embeddings\","
         << "\"device\":\""
         << (device_ == mx::Device::gpu ? "gpu" : "cpu") << "\","
         << "\"pcm_samples\":" << audio_len << ",\"language\":\""
         << json_escape(language) << "\",\"num_audio_tokens\":"
         << audio_token_count << ",\"prompt_length\":" << token_ids.size()
         << ",\"token_ids\":[";
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
      if (index != 0) {
        json << ",";
      }
      json << token_ids[index];
    }
    json << "],\"audio_token_indices\":[";
    for (std::size_t index = 0; index < audio_token_indices.size(); ++index) {
      if (index != 0) {
        json << ",";
      }
      json << audio_token_indices[index];
    }
    json << "],\"elapsed_ms\":" << elapsed << ",\"peak_memory_bytes\":"
         << mx::get_peak_memory()
         << ",\"fingerprints\":{\"token_embeddings\":"
         << token_embeddings_fingerprint
         << ",\"audio_features_bfloat16\":" << audio_features_fingerprint
         << ",\"inputs_embeds\":" << inputs_embeds_fingerprint << "}}";
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

  mx::array linear(const mx::array &input, const std::string &prefix) const {
    return mx::addmm(weight(prefix + ".bias"), input,
                     mx::transpose(weight(prefix + ".weight"), device_), 1.0f,
                     1.0f, device_);
  }

  mx::array layer_norm(const mx::array &input,
                       const std::string &prefix) const {
    return mx::fast::layer_norm(input, weight(prefix + ".weight"),
                                weight(prefix + ".bias"),
                                kAudioLayerNormEpsilon, device_);
  }

  mx::array positional_embedding(int sequence_length) const {
    const auto log_timescale_increment = static_cast<float>(
        std::log(10'000.0) / (static_cast<double>(kAudioState / 2) - 1.0));
    const auto inverse_timescales =
        mx::exp(-log_timescale_increment *
                    mx::arange(static_cast<int>(kAudioState / 2), mx::float32,
                               device_),
                device_);
    const auto scaled_time =
        mx::expand_dims(
            mx::arange(sequence_length, mx::float32, device_), 1, device_) *
        mx::expand_dims(inverse_timescales, 0, device_);
    return mx::concatenate(
        {mx::sin(scaled_time, device_), mx::cos(scaled_time, device_)}, 1,
        device_);
  }

  mx::array make_attention_mask(
      std::size_t sequence_length,
      const std::vector<std::size_t> &attention_windows) const {
    std::vector<float> values(sequence_length * sequence_length, -1e9f);
    for (std::size_t window = 0; window + 1 < attention_windows.size();
         ++window) {
      for (auto row = attention_windows[window];
           row < attention_windows[window + 1]; ++row) {
        for (auto column = attention_windows[window];
             column < attention_windows[window + 1]; ++column) {
          values[row * sequence_length + column] = 0.0f;
        }
      }
    }
    return mx::array(values.begin(),
                     {1, 1, static_cast<int>(sequence_length),
                      static_cast<int>(sequence_length)});
  }

  mx::array self_attention(const mx::array &input, std::size_t layer,
                           const mx::array &attention_mask) const {
    const auto prefix =
        "audio_tower.layers." + std::to_string(layer) + ".self_attn";
    const auto batch_size = input.shape(0);
    const auto sequence_length = input.shape(1);
    auto query =
        linear(input, prefix + ".q_proj") *
        (1.0f / std::sqrt(static_cast<float>(kAudioHeadDimension)));
    auto key = linear(input, prefix + ".k_proj");
    auto value = linear(input, prefix + ".v_proj");
    query = mx::transpose(
        mx::reshape(query,
                    {batch_size, sequence_length,
                     static_cast<int>(kAudioHeads),
                     static_cast<int>(kAudioHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    key = mx::transpose(
        mx::reshape(key,
                    {batch_size, sequence_length,
                     static_cast<int>(kAudioHeads),
                     static_cast<int>(kAudioHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    value = mx::transpose(
        mx::reshape(value,
                    {batch_size, sequence_length,
                     static_cast<int>(kAudioHeads),
                     static_cast<int>(kAudioHeadDimension)},
                    device_),
        {0, 2, 1, 3}, device_);
    auto output = mx::fast::scaled_dot_product_attention(
        query, key, value, 1.0f, "", attention_mask, {}, device_);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}, device_),
                         {batch_size, sequence_length,
                          static_cast<int>(kAudioState)},
                         device_);
    return linear(output, prefix + ".out_proj");
  }

  mx::array encoder_layer(const mx::array &input, std::size_t layer,
                          const mx::array &attention_mask) const {
    const auto prefix =
        "audio_tower.layers." + std::to_string(layer) + ".";
    auto hidden =
        input + self_attention(layer_norm(input, prefix + "self_attn_layer_norm"),
                               layer, attention_mask);
    return hidden +
           linear(gelu(linear(layer_norm(hidden, prefix + "final_layer_norm"),
                              prefix + "fc1")),
                  prefix + "fc2");
  }

  mx::array encode_audio_features(const float *audio,
                                  std::size_t audio_len) const {
    const auto features = log_mel(audio, audio_len);
    std::vector<std::size_t> chunk_lengths;
    const auto chunks = make_chunks(features, chunk_lengths);
    auto hidden = mx::expand_dims(chunks, 3, device_);
    hidden = gelu(mx::conv2d(hidden, weight("audio_tower.conv2d1.weight"),
                             {2, 2}, {1, 1}, {1, 1}, 1, device_) +
                  weight("audio_tower.conv2d1.bias"));
    hidden = gelu(mx::conv2d(hidden, weight("audio_tower.conv2d2.weight"),
                             {2, 2}, {1, 1}, {1, 1}, 1, device_) +
                  weight("audio_tower.conv2d2.bias"));
    hidden = gelu(mx::conv2d(hidden, weight("audio_tower.conv2d3.weight"),
                             {2, 2}, {1, 1}, {1, 1}, 1, device_) +
                  weight("audio_tower.conv2d3.bias"));
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
    hidden = hidden + mx::expand_dims(positional_embedding(frames), 0, device_);

    std::vector<mx::array> valid_chunks;
    std::size_t sequence_length = 0;
    for (std::size_t index = 0; index < chunk_lengths.size(); ++index) {
      const auto valid_length = (chunk_lengths[index] + 7) / 8;
      sequence_length += valid_length;
      auto chunk = mx::slice(
          hidden, {static_cast<int>(index), 0, 0},
          {static_cast<int>(index + 1), static_cast<int>(valid_length),
           static_cast<int>(kAudioState)},
          {1, 1, 1}, device_);
      valid_chunks.push_back(mx::squeeze(chunk, 0, device_));
    }
    hidden = mx::concatenate(std::move(valid_chunks), 0, device_);

    const auto attention_window = static_cast<std::size_t>(frames) * 8;
    std::vector<std::size_t> attention_windows{0};
    for (std::size_t position = 0; position < sequence_length;) {
      position = std::min(position + attention_window, sequence_length);
      attention_windows.push_back(position);
    }
    const auto attention_mask =
        make_attention_mask(sequence_length, attention_windows);
    hidden = mx::expand_dims(hidden, 0, device_);
    for (std::size_t layer = 0; layer < kAudioEncoderLayers; ++layer) {
      hidden = encoder_layer(hidden, layer, attention_mask);
    }
    hidden = mx::squeeze(hidden, 0, device_);
    hidden = layer_norm(hidden, "audio_tower.ln_post");
    hidden = gelu(linear(hidden, "audio_tower.proj1"));
    return linear(hidden, "audio_tower.proj2");
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

  std::filesystem::path model_directory_;
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
    const Qwen3AudioEncoder frontend(model_directory, device);
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

extern "C" int32_t cuttledoc_qwen3_mlx_probe_audio_encoder(
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
    const Qwen3AudioEncoder encoder(model_directory, device);
    const auto json = encoder.probe_encoder(audio, audio_len);
    *json_out = strdup(json.c_str());
    if (*json_out == nullptr) {
      return fail("could not allocate JSON result", error_out);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("MLX raised a non-standard exception in the audio encoder",
                error_out);
  }
}

extern "C" int32_t cuttledoc_qwen3_mlx_probe_prompt_embeddings(
    const char *model_directory, const float *audio, std::size_t audio_len,
    const char *language, int32_t device_kind, char **json_out,
    char **error_out) {
  if (json_out != nullptr) {
    *json_out = nullptr;
  }
  if (error_out != nullptr) {
    *error_out = nullptr;
  }
  if (model_directory == nullptr || audio == nullptr || language == nullptr ||
      json_out == nullptr) {
    return fail(
        "model_directory, audio, language, and json_out must be non-null",
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
    const Qwen3AudioEncoder model(model_directory, device);
    const auto json = model.probe_prompt(audio, audio_len, language);
    *json_out = strdup(json.c_str());
    if (*json_out == nullptr) {
      return fail("could not allocate JSON result", error_out);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error.what(), error_out);
  } catch (...) {
    return fail("MLX raised a non-standard exception in prompt embedding",
                error_out);
  }
}

extern "C" void cuttledoc_qwen3_mlx_free_string(char *value) {
  std::free(value);
}
