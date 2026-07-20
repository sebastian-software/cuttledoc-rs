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
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${MLX_SOURCE_DIR}" mlx-upstream EXCLUDE_FROM_ALL)

add_library(cuttledoc_mlx_shim SHARED Sources/CuttledocMlxShim.cpp)
target_link_libraries(cuttledoc_mlx_shim PRIVATE mlx)
target_include_directories(cuttledoc_mlx_shim PUBLIC include)
target_compile_options(cuttledoc_mlx_shim PRIVATE -Wall -Wextra -Wpedantic)

add_library(
  cuttledoc_qwen3_mlx_shim
  SHARED
  ../qwen3-mlx-direct/Sources/CuttledocQwen3MlxShim.cpp)
target_link_libraries(cuttledoc_qwen3_mlx_shim PRIVATE mlx)
target_include_directories(
  cuttledoc_qwen3_mlx_shim
  PUBLIC
  ../qwen3-mlx-direct/include)
target_compile_options(
  cuttledoc_qwen3_mlx_shim
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic)

add_library(
  cuttledoc_voxtral_mlx_shim
  SHARED
  ../voxtral-realtime-mlx-direct/Sources/CuttledocVoxtralMlxShim.cpp
  Sources/CuttledocMlxTransformer.cpp)
target_link_libraries(cuttledoc_voxtral_mlx_shim PRIVATE mlx)
target_include_directories(
  cuttledoc_voxtral_mlx_shim
  PUBLIC
  ../voxtral-realtime-mlx-direct/include
  include)
FetchContent_GetProperties(json)
target_include_directories(
  cuttledoc_voxtral_mlx_shim
  PRIVATE
  "${json_SOURCE_DIR}/single_include/nlohmann")
target_compile_options(
  cuttledoc_voxtral_mlx_shim
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic)

# MLX resolves its default Metal kernels from mlx.metallib at runtime. Make the
# artifact an explicit build dependency of the shim so the spike cannot appear
# to link successfully while omitting the file needed for GPU execution.
if(TARGET mlx-metallib)
  foreach(
    shim_target
    IN ITEMS
    cuttledoc_mlx_shim
    cuttledoc_qwen3_mlx_shim
    cuttledoc_voxtral_mlx_shim)
    add_dependencies(${shim_target} mlx-metallib)
    add_custom_command(
      TARGET ${shim_target}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/mlx-upstream/mlx/backend/metal/kernels/mlx.metallib"
        "$<TARGET_FILE_DIR:${shim_target}>/mlx.metallib"
      COMMENT "Colocating mlx.metallib with ${shim_target}")
  endforeach()
endif()
