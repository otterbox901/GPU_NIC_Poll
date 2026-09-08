cmake_minimum_required(VERSION 3.24)
project(gpu_nic_poll LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(GNP_SIMULATION  "Build the software packet-arrival simulator" ON)
option(GNP_ENABLE_CUDA "Use the CUDA backend when a CUDA compiler exists" ON)
option(GNP_BUILD_TESTS "Build unit tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(gnp_cuda)   # -> GNP_WITH_CUDA

# ---------------------------------------------------------------------------
# gnp_core: everything except main(), so the tests can link against it.
# ---------------------------------------------------------------------------
set(GNP_SOURCES
    src/host/setup.cpp
    src/host/metrics.cpp)

if(GNP_SIMULATION OR GNP_BUILD_TESTS)
    # sim_publish() is shared by the injector and the ring-protocol unit tests.
    list(APPEND GNP_SOURCES src/host/sim_inject.cpp)
endif()

if(GNP_WITH_CUDA)
    list(APPEND GNP_SOURCES src/gpu/poll_kernel.cu src/gpu/utils.cu)
else()
    list(APPEND GNP_SOURCES src/gpu/poll_cpu.cpp)
endif()

add_library(gnp_core STATIC ${GNP_SOURCES})

target_include_directories(gnp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Threads REQUIRED)
target_link_libraries(gnp_core PUBLIC Threads::Threads)

if(GNP_SIMULATION)
    target_compile_definitions(gnp_core PUBLIC GNP_SIMULATION=1)
endif()

if(GNP_WITH_CUDA)
    target_compile_definitions(gnp_core PUBLIC GNP_WITH_CUDA=1)
    set_target_properties(gnp_core PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    # Link cudart explicitly: `gnp` itself has no .cu sources, so it is linked by
    # the C++ driver and cannot be relied on to pick the runtime up implicitly.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(gnp_core PUBLIC CUDA::cudart)
    # -lineinfo keeps the poll loop attributable in Nsight Compute.
    target_compile_options(gnp_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# ---------------------------------------------------------------------------
# gnp: the driver executable
# ---------------------------------------------------------------------------
add_executable(gnp src/host/main.cpp)
target_link_libraries(gnp PRIVATE gnp_core)

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
if(GNP_BUILD_TESTS)
    enable_testing()
    add_executable(test_ring tests/test_ring.cpp)
    target_link_libraries(test_ring PRIVATE gnp_core)
    add_test(NAME ring_protocol COMMAND test_ring)
endif()

message(STATUS "gnp: simulation=${GNP_SIMULATION} cuda=${GNP_WITH_CUDA} tests=${GNP_BUILD_TESTS}")
