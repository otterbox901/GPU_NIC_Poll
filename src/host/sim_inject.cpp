//
// src/host/sim_inject.cpp - stand-in for the NIC.
//
// A host thread that publishes completion descriptors at a paced rate. When
// real hardware arrives this file is what gets replaced: the ring, the owner-bit
// protocol and the poller all stay exactly as they are.
//

#include <atomic>
#include <cstring>
#include <thread>

#include "gnp/common.hpp"
#include "gnp/ring.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace gnp {
namespace {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

}  // namespace

void sim_publish(const CompletionRing& ring, uint64_t idx, uint64_t payload_offset,
                 uint32_t byte_len, uint32_t packet_id, uint64_t post_ns) {
    CompletionDesc* d = &ring.descs[ring_slot(ring, idx)];

    d->payload_offset = payload_offset;
    d->byte_len = byte_len;
    d->packet_id = packet_id;
    d->post_ns = post_ns;
    d->reserved = 0;

    // Full barrier, not just a release fence. On x86 a release fence is only a
    // compiler barrier, which is not enough here: these stores may land in
    // write-combining PCIe-mapped memory, where the CPU is free to reorder them.
    // A seq_cst fence emits a real mfence and drains the WC buffers.
    std::atomic_thread_fence(std::memory_order_seq_cst);

    reinterpret_cast<std::atomic<uint32_t>*>(&d->status)
        ->store(ring_expected_owner(ring, idx), std::memory_order_relaxed);
}

struct Simulator {
    std::thread thread;
    std::atomic<bool> stop{false};
    SimStats stats;
};

namespace {

void inject_loop(Simulator* sim, CompletionRing ring, RingControl* ctrl, uint8_t* arena,
                 RunConfig cfg) {
    const uint32_t burst = cfg.burst ? cfg.burst : 1u;
    // Nanoseconds between bursts. 0 means "as fast as the CPU can go".
    const uint64_t interval_ns =
        cfg.target_pps ? (1000000000ull * burst) / cfg.target_pps : 0ull;

    const auto* consumed =
        reinterpret_cast<const std::atomic<unsigned long long>*>(&ctrl->consumed);

    uint64_t produced = 0;
    uint64_t overruns = 0;
    sim->stats.start_ns = host_now_ns();
    uint64_t next_ns = sim->stats.start_ns;

    while (!sim->stop.load(std::memory_order_relaxed)) {
        if (cfg.max_packets && produced >= cfg.max_packets) break;

        if (interval_ns) {
            // Busy-wait: sleep_for cannot pace anywhere near microsecond periods.
            while (host_now_ns() < next_ns) {
                if (sim->stop.load(std::memory_order_relaxed)) goto done;
                cpu_relax();
            }
            next_ns += interval_ns;
        }

        for (uint32_t b = 0; b < burst; ++b) {
            if (cfg.max_packets && produced >= cfg.max_packets) break;

            // Back-pressure. The consumer publishes its index every 64 entries,
            // so this watermark is slightly stale - harmless, it only ever makes
            // us more conservative.
            while (produced - consumed->load(std::memory_order_acquire) >= ring.capacity) {
                ++overruns;
                if (sim->stop.load(std::memory_order_relaxed)) goto done;
                cpu_relax();
            }

            const uint64_t offset =
                static_cast<uint64_t>(ring_slot(ring, produced)) * cfg.payload_bytes;

            // Simulated DMA. We write a 16-byte header rather than the whole
            // payload: the producer has to keep up with target_pps, and pushing
            // full packets over PCIe uncached would make the injector, not the
            // poller, the bottleneck.
            const uint32_t id = static_cast<uint32_t>(produced);
            std::memcpy(arena + offset, &id, sizeof(id));

            sim_publish(ring, produced, offset, cfg.payload_bytes, id, host_now_ns());
            ++produced;
        }
    }

done:
    sim->stats.produced = produced;
    sim->stats.overruns = overruns;
    sim->stats.end_ns = host_now_ns();
}

}  // namespace

Simulator* sim_start(const CompletionRing& ring, RingControl* ctrl, uint8_t* arena,
                     size_t arena_bytes, const RunConfig& cfg) {
    if (!ring.descs || !ctrl || !arena) return nullptr;
    if (arena_bytes < static_cast<size_t>(ring.capacity) * cfg.payload_bytes) return nullptr;

    auto* sim = new Simulator();
    sim->thread = std::thread(inject_loop, sim, ring, ctrl, arena, cfg);
    return sim;
}

void sim_stop(Simulator* sim, SimStats& out) {
    if (!sim) return;
    sim->stop.store(true, std::memory_order_relaxed);
    if (sim->thread.joinable()) sim->thread.join();
    out = sim->stats;
    delete sim;
}

}  // namespace gnp
