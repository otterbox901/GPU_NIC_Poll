//
// src/gpu/utils.cu - CUDA backend plumbing: device probe, shared memory, clocks.
//

#include <cuda_runtime.h>

#include <cstdio>
#include <new>

#include "gnp/common.hpp"
#include "gnp/gpu_poll.hpp"

namespace gnp {
namespace {

int  g_device = 0;
bool g_initialised = false;
const char* g_mem_strategy = "uninitialised";

/// Spins until the host raises `flag`, then records the GPU clock.
///
/// Launching a kernel just to read %globaltimer would fold ~10 us of launch
/// latency into the offset. Having the kernel already resident when the host
/// takes its timestamp cuts that down to one PCIe hop.
__global__ void gnp_clock_probe(volatile unsigned int* flag, unsigned long long* out) {
    while (*flag == 0) {
    }
    *out = device_now_ns();
}

}  // namespace

const char* backend_name() { return "cuda"; }

const char* backend_memory_strategy() { return g_mem_strategy; }

bool backend_init(bool verbose) {
    if (g_initialised) return true;

    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        std::fprintf(stderr, "[gnp] no CUDA device available: %s\n",
                     e == cudaSuccess ? "count is 0" : cudaGetErrorString(e));
        return false;
    }

    GNP_CUDA_CHECK(cudaSetDevice(g_device));
    g_mem_strategy = "pinned-mapped host (coherent)";

    if (verbose) {
        cudaDeviceProp p{};
        GNP_CUDA_CHECK(cudaGetDeviceProperties(&p, g_device));
        std::printf("[gnp] device %d: %s (sm_%d%d, %d SMs)\n", g_device, p.name, p.major, p.minor,
                    p.multiProcessorCount);
        std::printf("[gnp] shared-memory strategy: %s\n", g_mem_strategy);
        std::printf("[gnp] note: 1 block x 1 thread poller (CQ is strictly ordered)\n");
    }

    g_initialised = true;
    return true;
}

void* backend_alloc_shared(size_t bytes, bool write_combined) {
    void* p = nullptr;
    unsigned flags = cudaHostAllocMapped;
    if (write_combined) flags |= cudaHostAllocWriteCombined;
    GNP_CUDA_CHECK(cudaHostAlloc(&p, bytes, flags));
    return p;
}

void backend_free_shared(void* p) {
    if (p) cudaFreeHost(p);
}

void* backend_alloc_host(size_t bytes) {
    return ::operator new(bytes, std::align_val_t(64), std::nothrow);
}

void backend_free_host(void* p) {
    if (p) ::operator delete(p, std::align_val_t(64));
}

int64_t backend_clock_offset_ns() {
    constexpr int kSamples = 16;

    volatile unsigned int* flag = nullptr;
    unsigned long long* gpu_ns = nullptr;
    GNP_CUDA_CHECK(cudaHostAlloc((void**)&flag, sizeof(unsigned int), cudaHostAllocMapped));
    GNP_CUDA_CHECK(cudaHostAlloc((void**)&gpu_ns, sizeof(unsigned long long), cudaHostAllocMapped));

    int64_t best_offset = 0;
    uint64_t best_window = ~0ull;

    for (int i = 0; i < kSamples; ++i) {
        *flag = 0;
        *gpu_ns = 0;
        gnp_clock_probe<<<1, 1>>>(flag, gpu_ns);
        GNP_CUDA_CHECK(cudaGetLastError());

        // Let the probe reach its spin loop before we start the stopwatch.
        for (volatile int spin = 0; spin < 200000; ++spin) {
        }

        const uint64_t t0 = host_now_ns();
        *flag = 1;
        GNP_CUDA_CHECK(cudaDeviceSynchronize());
        const uint64_t t1 = host_now_ns();

        const uint64_t window = t1 - t0;
        if (window < best_window) {
            best_window = window;
            // Midpoint of the host window is our best guess at the instant the
            // GPU sampled its own clock.
            best_offset = static_cast<int64_t>(t0 + window / 2) - static_cast<int64_t>(*gpu_ns);
        }
    }

    cudaFreeHost((void*)flag);
    cudaFreeHost(gpu_ns);
    return best_offset;
}

}  // namespace gnp
