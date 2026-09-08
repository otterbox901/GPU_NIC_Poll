//
// src/gpu/utils.cu - CUDA backend plumbing: device probe, shared memory, clocks.
//

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>

#include "gnp/common.hpp"
#include "gnp/gpu_poll.hpp"
#include "gnp/ring.hpp"

namespace gnp {
namespace {

int  g_device = 0;
bool g_initialised = false;
const char* g_mem_strategy = "uninitialised";

cudaStream_t g_copy_stream = nullptr;

/// Spins until the host raises `flag`, then records the GPU clock.
__global__ void gnp_clock_probe(volatile unsigned int* flag, unsigned long long* out) {
    while (*flag == 0) {
    }
    *out = device_now_ns();
}

// NOTE: H2D flush uses the DMA copy engine (cudaMemcpyAsync), not a compute
// kernel. A persistent poller would starve a flush kernel on this GPU; the copy
// engine runs concurrently with the SM poll loop.

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
    GNP_CUDA_CHECK(cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking));
    g_mem_strategy = "device CQ + pinned staging (DMA H2D flush)";

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

bool backend_alloc_ring(size_t bytes, CompletionDesc** host_out, CompletionDesc** device_out) {
    void* host = nullptr;
    void* device = nullptr;
    GNP_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
    GNP_CUDA_CHECK(cudaMalloc(&device, bytes));
    GNP_CUDA_CHECK(cudaMemset(device, 0, bytes));
    std::memset(host, 0, bytes);
    *host_out = static_cast<CompletionDesc*>(host);
    *device_out = static_cast<CompletionDesc*>(device);
    return true;
}

void backend_free_ring(CompletionDesc* host, CompletionDesc* device) {
    if (host) cudaFreeHost(host);
    if (device) cudaFree(device);
}

void backend_flush_descs(CompletionDesc* host, CompletionDesc* device, uint32_t capacity,
                         uint64_t start_idx, uint32_t count) {
    if (!host || !device || count == 0 || host == device) return;

    // DMA copy engine runs concurrently with the persistent poller. Prefer one
    // contiguous memcpy per non-wrapping span. status is the last field of each
    // CQE, so address-ordered DMA publishes the owner bit after the payload.
    const uint32_t mask = capacity - 1;
    uint32_t slot = static_cast<uint32_t>(start_idx) & mask;
    uint32_t left = count;

    while (left > 0) {
        const uint32_t span = (slot + left <= capacity) ? left : (capacity - slot);
        const size_t bytes = static_cast<size_t>(span) * sizeof(CompletionDesc);
        GNP_CUDA_CHECK(cudaMemcpyAsync(device + slot, host + slot, bytes, cudaMemcpyHostToDevice,
                                       g_copy_stream));
        slot = (slot + span) & mask;
        left -= span;
    }
}

void backend_flush_wait() {
    if (g_copy_stream) GNP_CUDA_CHECK(cudaStreamSynchronize(g_copy_stream));
}

void backend_fini_copy() {
    if (g_copy_stream) {
        cudaStreamSynchronize(g_copy_stream);
        cudaStreamDestroy(g_copy_stream);
        g_copy_stream = nullptr;
    }
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

        for (volatile int spin = 0; spin < 200000; ++spin) {
        }

        const uint64_t t0 = host_now_ns();
        *flag = 1;
        GNP_CUDA_CHECK(cudaDeviceSynchronize());
        const uint64_t t1 = host_now_ns();

        const uint64_t window = t1 - t0;
        if (window < best_window) {
            best_window = window;
            best_offset = static_cast<int64_t>(t0 + window / 2) - static_cast<int64_t>(*gpu_ns);
        }
    }

    cudaFreeHost((void*)flag);
    cudaFreeHost(gpu_ns);
    return best_offset;
}

}  // namespace gnp
