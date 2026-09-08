//
// src/gpu/poll_cpu.cpp - CPU-thread stand-in for the persistent CUDA kernel.
//
// Compiled ONLY when CMake could not find a CUDA compiler. It runs the exact
// same owner-bit poll loop on a spinning host thread, which lets the ring, the
// simulator and the metrics path be developed and tested without nvcc.
//

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>

#include "gnp/common.hpp"
#include "gnp/gpu_poll.hpp"

namespace gnp {
namespace {

constexpr unsigned long long kPublishMask = 63ull;

std::thread g_poller;

/// Mirror of gnp_poll_kernel(). Kept structurally identical on purpose.
void poll_loop(CompletionRing ring, RingControl* ctrl, PollStats* stats,
               unsigned long long max_run_ns, unsigned int idle_backoff_ns) {
    auto status_of = [](CompletionDesc* d) {
        return reinterpret_cast<std::atomic<uint32_t>*>(&d->status);
    };

    const uint64_t t_start = host_now_ns();

    PollStats s = {};
    s.lat_min_ns = ~0ull;

    unsigned long long idx = 0;
    uint32_t next_id = 0;
    bool have_prev = false;
    bool saw_stop = false;

    auto* stop = reinterpret_cast<std::atomic<uint32_t>*>(&ctrl->stop_flag);
    auto* publish_limit =
        reinterpret_cast<std::atomic<unsigned long long>*>(&ctrl->publish_limit);

    for (;;) {
        CompletionDesc* d = &ring.descs[ring_slot(ring, idx)];
        const uint32_t want = ring_expected_owner(ring, idx);

        if (desc_ready(status_of(d)->load(std::memory_order_acquire), want)) {
            const uint32_t len = d->byte_len;
            const uint32_t pid = d->packet_id;
            const uint64_t post_ns = d->post_ns;

            long long lat = static_cast<long long>(host_now_ns()) - static_cast<long long>(post_ns);
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
                reinterpret_cast<std::atomic<unsigned long long>*>(&ctrl->consumed)
                    ->store(idx, std::memory_order_release);
                *stats = s;
                if (host_now_ns() - t_start > max_run_ns) break;
            }
        } else {
            ++s.idle_spins;
            if (stop->load(std::memory_order_acquire)) {
                if (!saw_stop) saw_stop = true;
                else ++s.drain_spins;
                if (idx >= publish_limit->load(std::memory_order_acquire)) break;
            }
            if ((s.idle_spins & 1023ull) == 0 && host_now_ns() - t_start > max_run_ns) break;
            if (idle_backoff_ns) std::this_thread::yield();
        }
    }

    reinterpret_cast<std::atomic<unsigned long long>*>(&ctrl->consumed)
        ->store(idx, std::memory_order_release);
    *stats = s;
}

}  // namespace

const char* backend_name() { return "cpu-fallback"; }

const char* backend_memory_strategy() { return "host heap (cpu-fallback)"; }

bool backend_init(bool verbose) {
    if (verbose) {
        std::printf("[gnp] no CUDA compiler at build time; polling on a host thread\n");
        std::printf("[gnp] shared-memory strategy: %s\n", backend_memory_strategy());
    }
    return true;
}

void* backend_alloc_shared(size_t bytes, bool /*write_combined*/) {
    return ::operator new(bytes, std::align_val_t(256), std::nothrow);
}

void backend_free_shared(void* p) {
    if (p) ::operator delete(p, std::align_val_t(256));
}

void* backend_alloc_host(size_t bytes) {
    return ::operator new(bytes, std::align_val_t(64), std::nothrow);
}

void backend_free_host(void* p) {
    if (p) ::operator delete(p, std::align_val_t(64));
}

int64_t backend_clock_offset_ns() { return 0; }

bool backend_launch_poller(const CompletionRing& ring, RingControl* ctrl, PollStats* stats,
                           int64_t /*clock_offset_ns*/, const RunConfig& cfg) {
    const unsigned long long max_run_ns =
        (static_cast<unsigned long long>(cfg.duration_ms) + 5000ull) * 1000000ull;
    g_poller = std::thread(poll_loop, ring, ctrl, stats, max_run_ns, cfg.idle_backoff_ns);
    return true;
}

bool backend_wait_poller() {
    if (g_poller.joinable()) g_poller.join();
    return true;
}

void backend_shutdown() {}

}  // namespace gnp
