#include "cuttledoc_mlx_transformer.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;

namespace cuttledoc::mlx_support {

RotatingKeyValueCache::RotatingKeyValueCache(std::size_t max_size,
                                             std::size_t keep)
    : max_size_(max_size), keep_(keep) {
  if (max_size == 0 || keep >= max_size) {
    throw std::invalid_argument(
        "rotating KV cache requires 0 <= keep < max_size");
  }
}

std::size_t RotatingKeyValueCache::offset() const { return offset_; }

std::size_t RotatingKeyValueCache::size() const {
  return std::min(offset_, max_size_);
}

std::size_t RotatingKeyValueCache::materialized_size() const {
  return keys_.has_value() ? static_cast<std::size_t>(keys_->shape(2)) : 0;
}

AttentionMask RotatingKeyValueCache::make_mask(
    std::size_t query_length, std::size_t window_size, mx::Device device,
    bool force_array) const {
  if (query_length == 0 || window_size == 0) {
    throw std::invalid_argument(
        "attention mask requires positive query and window sizes");
  }
  if (query_length == 1 && !force_array) {
    return {"", std::nullopt};
  }

  const auto mask_offset = std::min(max_size_ - 1, offset_);
  if (!force_array && mask_offset + query_length <= window_size) {
    return {"causal", std::nullopt};
  }

  const auto key_length = mask_offset + query_length;
  std::vector<std::uint8_t> mask(query_length * key_length, 0);
  for (std::size_t row = 0; row < query_length; ++row) {
    const auto query_position = mask_offset + row;
    for (std::size_t column = 0; column < key_length; ++column) {
      const auto causal = query_position >= column;
      const auto inside_window = query_position < column + window_size;
      mask[row * key_length + column] = causal && inside_window ? 1 : 0;
    }
  }
  auto mask_array = mx::array(
      mask.begin(),
      {static_cast<int>(query_length), static_cast<int>(key_length)},
      mx::bool_);
  if (device == mx::Device::gpu) {
    mask_array = mx::copy(std::move(mask_array), device);
  }
  return {"", std::move(mask_array)};
}

mx::array RotatingKeyValueCache::trim_front(const mx::array &value,
                                            std::size_t trim,
                                            mx::Device device) const {
  if (trim == 0) {
    return value;
  }
  const auto length = static_cast<std::size_t>(value.shape(2));
  if (keep_ + trim > length) {
    throw std::runtime_error("rotating KV cache trim exceeds stored context");
  }
  auto recent = mx::slice(
      value, {0, 0, static_cast<int>(keep_ + trim), 0},
      {value.shape(0), value.shape(1), value.shape(2), value.shape(3)},
      {1, 1, 1, 1}, device);
  if (keep_ == 0) {
    return recent;
  }
  auto prefix = mx::slice(
      value, {0, 0, 0, 0},
      {value.shape(0), value.shape(1), static_cast<int>(keep_),
       value.shape(3)},
      {1, 1, 1, 1}, device);
  return mx::concatenate({std::move(prefix), std::move(recent)}, 2, device);
}

std::pair<mx::array, mx::array> RotatingKeyValueCache::update_and_fetch(
    const mx::array &keys, const mx::array &values, mx::Device device) {
  if (keys.ndim() != 4 || values.ndim() != 4 ||
      keys.shape(0) != values.shape(0) ||
      keys.shape(1) != values.shape(1) ||
      keys.shape(2) != values.shape(2)) {
    throw std::invalid_argument(
        "KV cache updates require matching [batch, heads, time, dim] arrays");
  }
  const auto input_length = static_cast<std::size_t>(keys.shape(2));
  if (input_length == 0) {
    throw std::invalid_argument("KV cache cannot append an empty update");
  }

  if (!keys_.has_value()) {
    keys_ = keys;
    values_ = values;
  } else {
    const auto stored_length = materialized_size();
    std::size_t trim = 0;
    if (input_length == 1) {
      if (stored_length + 1 > max_size_) {
        trim = stored_length + 1 - max_size_;
      }
    } else if (stored_length >= max_size_) {
      // Match mlx-lm's multi-token update: retain max_size - 1 old tokens so
      // the first new query sees a full window, then append the complete chunk.
      trim = stored_length - max_size_ + 1;
    }
    auto old_keys = trim_front(keys_.value(), trim, device);
    auto old_values = trim_front(values_.value(), trim, device);
    keys_ = mx::concatenate({std::move(old_keys), keys}, 2, device);
    values_ = mx::concatenate({std::move(old_values), values}, 2, device);
  }
  offset_ += input_length;
  return {keys_.value(), values_.value()};
}

const mx::array &RotatingKeyValueCache::keys() const {
  if (!keys_.has_value()) {
    throw std::runtime_error("KV cache keys are empty");
  }
  return keys_.value();
}

const mx::array &RotatingKeyValueCache::values() const {
  if (!values_.has_value()) {
    throw std::runtime_error("KV cache values are empty");
  }
  return values_.value();
}

mx::array affine_quantized_linear(
    const mx::array &input, const mx::array &packed_weight,
    const mx::array &scales, const mx::array &quantization_biases,
    const std::optional<mx::array> &output_bias, int group_size, int bits,
    mx::Device device) {
  auto output = mx::quantized_matmul(
      input, packed_weight, scales, quantization_biases, true, group_size,
      bits, "affine", device);
  if (output_bias.has_value()) {
    output = output + output_bias.value();
  }
  return output;
}

mx::array silu(const mx::array &input, mx::Device device) {
  return input * mx::sigmoid(input, device);
}

} // namespace cuttledoc::mlx_support
