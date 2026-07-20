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
  expect_shape(loaded.weights, "encoder.conv_layers_1_conv.conv.weight",
               {1280, 3, 1280});
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
