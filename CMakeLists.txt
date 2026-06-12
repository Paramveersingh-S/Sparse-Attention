cmake_minimum_required(VERSION 3.18)
project(sparse_attn LANGUAGES CXX CUDA)

# ---- C++ / CUDA standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ---- CUDA architectures ----
# sm_75  = Turing  (RTX 2080 Ti, T4)
# sm_80  = Ampere  (A100, RTX 3090)
# sm_86  = Ampere  (RTX 3080)
# sm_89  = Ada     (RTX 4090)
# sm_90  = Hopper  (H100)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# ---- Compiler flags ----
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -fPIC")

# ---- Include directories ----
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- Find CUDA ----
find_package(CUDAToolkit REQUIRED)

# ---- Source files ----
set(CUDA_SOURCES
    # Future: add .cu files here when building CUDA extensions
    # src/sparse_decode_chunked_kv.cu
    # src/sparse_merge_partial.cu
)

set(CXX_SOURCES
    # Future: C++ binding files
)

# ---- Build shared library (for Python extension) ----
if(CUDA_SOURCES)
    add_library(sparse_attn_cuda SHARED
        ${CUDA_SOURCES}
        ${CXX_SOURCES}
    )

    target_include_directories(sparse_attn_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(sparse_attn_cuda
        CUDA::cudart
        CUDA::cublas
    )

    # Install
    install(TARGETS sparse_attn_cuda
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
    install(DIRECTORY include/
            DESTINATION include/sparse_attn)
endif()

# ---- Print configuration ----
message(STATUS "==============================================")
message(STATUS "sparse_attn CMake Configuration")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDAToolkit:        ${CUDAToolkit_VERSION}")
message(STATUS "==============================================")
