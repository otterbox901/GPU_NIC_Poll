//
// src/gpu/poll_kernel.cu - the persistent completion-queue poller.
//
// This is the heart of the project: a long-running CUDA kernel that watches the
// completion ring from an SM and never asks the CPU for anything. The only
// host interaction is stop_flag + publish_limit, checked on the idle path.
//

#include <cuda_runtime.h>

#include "gnp/common.hpp"
#include "gnp/gpu_poll.hpp"
#include "gnp/metrics.hpp"
#include "gnp/ring.hpp"

namespace gnp {
namespace {

/// Publish the consumer watermark this often. Every entry would put a PCIe
/// write on the critical path; every 64 keeps the producer's free-slot estimate
/// fresh enough while staying off the fast path.
constexpr unsigned long long kPublishMask = 63ull;

cudaStream_t g_stream = nullptr;

}  // namespace

/// Poll the ring until the host asks us to stop *and* we have reached the
/// publish limit. Single-threaded on purpose: a CQ is consumed in order.
__global__ void gnp_poll_kernel(CompletionRing ring, RingControl* ctrl, PollStats* stats,
                                long long clock_offset_ns, unsigned long long max_run_ns,
                                unsigned int idle_backoff_ns) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // volatile => every load goes past the non-coherent L1. Producer stores
    // arrive from outside this SM (host mapped memory over PCIe).
    volatile CompletionDesc* descs = ring.descs;
    volatile unsigned int* stop = &ctrl->stop_flag;
    volatile unsigned long long* publish_limit = &ctrl->publish_limit;

    const unsigned long long t_start = device_now_ns();

    PollStats s = {};
    s.lat_min_ns = ~0ull;

    unsigned long long idx = 0;
    unsigned int next_id = 0;
    bool have_prev = false;
    bool saw_stop = false;

    for (;;) {
        const unsigned int slot = ring_slot(ring, idx);
        const unsigned int want = ring_expected_owner(ring, idx);

        // System-scoped acquire was needed when the CQ lived in host-mapped
        // memory. The CQ is now device-resident; a GPU-scope acquire is enough
        // and much cheaper on the critical path.
        unsigned int st;
        asm volatile("ld.acquire.gpu.u32 %0, [%1];"
                     : "=r"(st)
                     : "l"(&descs[slot].status)
                     : "memory");

        if (desc_ready(st, want)) {
            const unsigned int len = descs[slot].byte_len;
            const unsigned int pid = descs[slot].packet_id;
            const unsigned long long post_ns = descs[slot].post_ns;

            const long long now_host = static_cast<long long>(device_now_ns()) + clock_offset_ns;
            long long lat = now_host - static_cast<long long>(post_ns);
            if (lat < 0) {
                lat = 0;
                ++s.clamped;
            }
            const unsigned long long ulat = static_cast<unsigned long long>(lat);

            ++s.packets;
            s.bytes += len;
            s.lat_sum_ns += ulat;
            if (ulat < s.lat_min_ns) s.lat_min_ns = ulat;
            if (ulat > s.lat_max_ns) s.lat_max_ns = ulat;
            if (have_prev && pid != next_id) ++s.gaps;
            next_id = pid + 1;
            have_prev = true;

            ++idx;

            if ((idx & kPublishMask) == 0) {
                ctrl->consumed = idx;
                *stats = s;
                // Host must see consumed/stats; system fence only on this path.
                __threadfence_system();
                if (device_now_ns() - t_start > max_run_ns) break;
            }
        } else {
            ++s.idle_spins;

            // Drain protocol: stop alone is not enough. Host publishes the final
            // produced count first; we only retire once idx has caught it.
            if (*stop) {
                if (!saw_stop) saw_stop = true;
                else ++s.drain_spins;
                __threadfence_system();
                if (idx >= *publish_limit) break;
            }

            if ((s.idle_spins & 1023ull) == 0 && device_now_ns() - t_start > max_run_ns) break;
#if __CUDA_ARCH__ >= 700
            if (idle_backoff_ns) __nanosleep(idle_backoff_ns);
#endif
        }
    }

    ctrl->consumed = idx;
    s.run_ns = device_now_ns() - t_start;
    *stats = s;
    __threadfence_system();
}

bool backend_launch_poller(const CompletionRing& ring, RingControl* ctrl, PollStats* stats,
                           int64_t clock_offset_ns, const RunConfig& cfg) {
    if (!g_stream) {
        GNP_CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
    }

    const unsigned long long max_run_ns =
        (static_cast<unsigned long long>(cfg.duration_ms) + 5000ull) * 1000000ull;

    gnp_poll_kernel<<<1, 1, 0, g_stream>>>(ring, ctrl, stats, clock_offset_ns, max_run_ns,
                                           cfg.idle_backoff_ns);

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[gnp] poller launch failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool backend_wait_poller() {
    if (!g_stream) return true;
    const cudaError_t e = cudaStreamSynchronize(g_stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[gnp] poller failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

void backend_shutdown() {
    backend_fini_copy();
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
}

}  // namespace gnp
