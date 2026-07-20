#ifndef CUTTLEDOC_MLX_TRANSFORMER_H
#define CUTTLEDOC_MLX_TRANSFORMER_H

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "mlx/mlx.h"

namespace cuttledoc::mlx_support {

struct AttentionMask {
  std::string mode;
  std::optional<mlx::core::array> array;
};

/*
 * Repository-owned temporal/cache semantics over MLX arrays. Multi-token
 * updates preserve max_size context for every query in the incoming chunk,
 * matching mlx-lm RotatingKVCache without depending on mlx-lm or Python.
 */
class RotatingKeyValueCache {
public:
  explicit RotatingKeyValueCache(std::size_t max_size,
                                 std::size_t keep = 0);

  std::size_t offset() const;
  std::size_t size() const;
  std::size_t materialized_size() const;

  AttentionMask make_mask(std::size_t query_length,
                          std::size_t window_size,
                          mlx::core::Device device,
                          bool force_array = false) const;

  std::pair<mlx::core::array, mlx::core::array>
  update_and_fetch(const mlx::core::array &keys,
                   const mlx::core::array &values,
                   mlx::core::Device device);

  const mlx::core::array &keys() const;
  const mlx::core::array &values() const;

private:
  mlx::core::array trim_front(const mlx::core::array &value,
                              std::size_t trim,
                              mlx::core::Device device) const;

  std::size_t max_size_;
  std::size_t keep_;
  std::size_t offset_{0};
  std::optional<mlx::core::array> keys_;
  std::optional<mlx::core::array> values_;
};

mlx::core::array affine_quantized_linear(
    const mlx::core::array &input, const mlx::core::array &packed_weight,
    const mlx::core::array &scales, const mlx::core::array &quantization_biases,
    const std::optional<mlx::core::array> &output_bias, int group_size,
    int bits, mlx::core::Device device);

mlx::core::array silu(const mlx::core::array &input,
                      mlx::core::Device device);

} // namespace cuttledoc::mlx_support

#endif
