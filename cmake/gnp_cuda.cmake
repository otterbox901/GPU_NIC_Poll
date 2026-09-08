# cmake/gnp_cuda.cmake
#
# Decides which polling backend gets compiled. Sets GNP_WITH_CUDA in the
# caller's scope (include() does not create a new scope).

include(CheckLanguage)

set(GNP_WITH_CUDA OFF)

if(NOT GNP_ENABLE_CUDA)
    message(STATUS "gnp: CUDA disabled by GNP_ENABLE_CUDA=OFF")
    return()
endif()

check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING
        "gnp: no CUDA compiler found - falling back to the CPU poller.\n"
        "     The ring, simulator, metrics and tests all still work, but the\n"
        "     persistent kernel is not built. Install the CUDA Toolkit and\n"
        "     reconfigure to get the real backend.")
    return()
endif()

# nvcc refuses host compilers it does not recognise, and distro GCC routinely
# runs ahead of the newest CUDA release. Pick an older g++ if one is installed.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    foreach(_ver 14 13 12 11)
        find_program(_gnp_host_cxx NAMES "g++-${_ver}" "g++${_ver}")
        if(_gnp_host_cxx)
            set(CMAKE_CUDA_HOST_COMPILER "${_gnp_host_cxx}" CACHE FILEPATH "nvcc host compiler")
            message(STATUS "gnp: using ${_gnp_host_cxx} as the nvcc host compiler")
            break()
        endif()
    endforeach()
endif()

# Must be set before enable_language(CUDA): that is when the default is captured.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(GNP_WITH_CUDA ON)

message(STATUS "gnp: CUDA backend enabled (arch=${CMAKE_CUDA_ARCHITECTURES})")
