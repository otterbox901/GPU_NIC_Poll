//
// src/gpu/poll_kernel.cu - the persistent completion-queue poller.
//
// This is the heart of the project: a long-running CUDA kernel that watches the
// completion ring from an SM and never asks the CPU for anything. The only
// host interaction is a single stop flag, checked on the idle path.
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

/// Poll the ring until the host asks us to stop.
///
/// Single-threaded on purpose. A completion queue is consumed strictly in
/// order, so exactly one lane can own the head; additional lanes would only
/// contend on the same descriptor. Payload *processing* is where extra threads
/// belong, and that is deliberately out of scope for v1.
__global__ void gnp_poll_kernel(CompletionRing ring, RingControl* ctrl, PollStats* stats,
                                long long clock_offset_ns, unsigned long long max_run_ns,
                                unsigned int idle_backoff_ns) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // volatile => every load goes past the non-coherent L1. The producer's
    // stores arrive from outside this SM, so cached reads would spin forever.
    volatile CompletionDesc* descs = ring.descs;
    volatile unsigned int* stop = &ctrl->stop_flag;

    const unsigned long long t_start = device_now_ns();

    PollStats s = {};
    s.lat_min_ns = ~0ull;

    unsigned long long idx = 0;  // monotonic consumer index
    unsigned int next_id = 0;
    bool have_prev = false;

    for (;;) {
        const unsigned int slot = ring_slot(ring, idx);
        const unsigned int want = ring_expected_owner(ring, idx);

        if (desc_ready(descs[slot].status, want)) {
            // The producer published `status` last. Keep the payload loads from
            // floating above it.
            __threadfence_system();

            const unsigned int len = descs[slot].byte_len;
            const unsigned int pid = descs[slot].packet_id;
            const unsigned long long post_ns = descs[slot].post_ns;

            // Translate the GPU clock onto the host epoch so the difference is
            // meaningful. Residual calibration error can make this negative.
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
                __threadfence_system();
                // Safety net only: never trust the host to still be alive.
                if (device_now_ns() - t_start > max_run_ns) break;
            }
        } else {
            ++s.idle_spins;
            // Checking the stop flag only here guarantees we drain the ring
            // first: the host sets it after the producer has already quiesced.
            if (*stop) break;
            if ((s.idle_spins & 1023ull) == 0 && device_now_ns() - t_start > max_run_ns) break;
#if __CUDA_ARCH__ >= 700
            if (idle_backoff_ns) __nanosleep(idle_backoff_ns);
#endif
        }
    }

    ctrl->consumed = idx;
    *stats = s;
    __threadfence_system();
}

bool backend_launch_poller(const CompletionRing& ring, RingControl* ctrl, PollStats* stats,
                           int64_t clock_offset_ns, const RunConfig& cfg) {
    if (!g_stream) {
        GNP_CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
    }

    // Hard ceiling on kernel lifetime, well past the requested duration.
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
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
}

}  // namespace gnp
