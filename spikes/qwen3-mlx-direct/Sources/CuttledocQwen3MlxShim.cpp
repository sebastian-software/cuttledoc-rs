#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

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

extern "C" void cuttledoc_qwen3_mlx_free_string(char *value) {
  std::free(value);
}
