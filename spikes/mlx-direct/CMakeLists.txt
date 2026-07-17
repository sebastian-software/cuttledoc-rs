cmake_minimum_required(VERSION 3.25)

project(cuttledoc_mlx_direct_spike LANGUAGES CXX)

if(NOT DEFINED MLX_SOURCE_DIR)
  message(FATAL_ERROR "Set MLX_SOURCE_DIR to a pinned official MLX checkout")
endif()

if(NOT EXISTS "${MLX_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "MLX_SOURCE_DIR is not an MLX source checkout: ${MLX_SOURCE_DIR}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The direct-official-MLX spike deliberately builds the upstream source instead
# of pulling a Rust wrapper or mlx-c into the product boundary.
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_SAFETENSORS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${MLX_SOURCE_DIR}" mlx-upstream EXCLUDE_FROM_ALL)

add_library(cuttledoc_mlx_shim SHARED Sources/CuttledocMlxShim.cpp)
target_link_libraries(cuttledoc_mlx_shim PRIVATE mlx)
target_compile_options(cuttledoc_mlx_shim PRIVATE -Wall -Wextra -Wpedantic)

# MLX resolves its default Metal kernels from mlx.metallib at runtime. Make the
# artifact an explicit build dependency of the shim so the spike cannot appear
# to link successfully while omitting the file needed for GPU execution.
if(TARGET mlx-metallib)
  add_dependencies(cuttledoc_mlx_shim mlx-metallib)
  add_custom_command(
    TARGET cuttledoc_mlx_shim
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/mlx-upstream/mlx/backend/metal/kernels/mlx.metallib"
      "$<TARGET_FILE_DIR:cuttledoc_mlx_shim>/mlx.metallib"
    COMMENT "Colocating mlx.metallib with the Rust executable")
endif()
