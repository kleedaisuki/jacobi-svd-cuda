cmake_minimum_required(VERSION 3.24)

project(jacobi_svd_cuda LANGUAGES NONE)

set(JACOBI_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build")

option(JACOBI_USE_CMAKE_CUDA_LANGUAGE
       "Use CMake native CUDA language mode (requires VS CUDA toolset on Windows)."
       OFF)

set(JACOBI_SOURCES
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/pipeline.cpp"
    "${CMAKE_SOURCE_DIR}/src/io.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels.cu"
)

if(WIN32)
    set(JACOBI_EXE_PATH "${JACOBI_OUTPUT_DIR}/jacobi-svd-cuda.exe")
else()
    set(JACOBI_EXE_PATH "${JACOBI_OUTPUT_DIR}/jacobi-svd-cuda")
endif()

if(JACOBI_USE_CMAKE_CUDA_LANGUAGE)
    enable_language(CXX)
    enable_language(CUDA)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${JACOBI_OUTPUT_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${JACOBI_OUTPUT_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${JACOBI_OUTPUT_DIR}")

    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${config}" config_upper)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${JACOBI_OUTPUT_DIR}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${JACOBI_OUTPUT_DIR}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${JACOBI_OUTPUT_DIR}")
    endforeach()

    add_executable(jacobi-svd-cuda ${JACOBI_SOURCES})
    target_include_directories(jacobi-svd-cuda PRIVATE "${CMAKE_SOURCE_DIR}/include")
    set_target_properties(jacobi-svd-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    find_program(JACOBI_NVCC_EXECUTABLE
                 NAMES nvcc
                 HINTS "$ENV{CUDA_PATH}/bin"
                 REQUIRED)

    set(JACOBI_NVCC_ARGS
        -std=c++20
        -I"${CMAKE_SOURCE_DIR}/include"
    )

    if(WIN32)
        list(APPEND JACOBI_NVCC_ARGS -Xcompiler=/utf-8)
    endif()

    add_custom_command(
        OUTPUT "${JACOBI_EXE_PATH}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${JACOBI_OUTPUT_DIR}"
        COMMAND "${JACOBI_NVCC_EXECUTABLE}" ${JACOBI_NVCC_ARGS} ${JACOBI_SOURCES} -o "${JACOBI_EXE_PATH}"
        DEPENDS ${JACOBI_SOURCES}
        VERBATIM
        COMMENT "Building jacobi-svd-cuda via nvcc driver mode"
    )

    add_custom_target(jacobi-svd-cuda ALL DEPENDS "${JACOBI_EXE_PATH}")
endif()
