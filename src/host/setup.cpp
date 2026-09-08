//
// src/host/setup.cpp - allocate and tear down everything one run needs.
//

#include <chrono>
#include <cstdio>
#include <cstring>

#include "gnp/common.hpp"
#include "gnp/gpu_poll.hpp"
#include "gnp/metrics.hpp"
#include "gnp/ring.hpp"

namespace gnp {

uint64_t host_now_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
            .count());
}

void stats_reset(PollStats& s) {
    std::memset(&s, 0, sizeof(s));
    s.lat_min_ns = ~0ull;
}

namespace {

uint32_t log2_exact(uint32_t v) {
    uint32_t n = 0;
    while ((1u << n) < v) ++n;
    return n;
}

}  // namespace

bool session_create(const RunConfig& cfg, Session& out) {
    if (!is_power_of_two(cfg.ring_capacity)) {
        std::fprintf(stderr, "[gnp] ring capacity %u is not a power of two\n", cfg.ring_capacity);
        return false;
    }
    if (cfg.payload_bytes == 0) {
        std::fprintf(stderr, "[gnp] payload size must be non-zero\n");
        return false;
    }
    if (!backend_init(cfg.verbose)) return false;

    out.ring.capacity = cfg.ring_capacity;
    out.ring.mask = cfg.ring_capacity - 1;
    out.ring.shift = log2_exact(cfg.ring_capacity);

    const size_t desc_bytes = static_cast<size_t>(cfg.ring_capacity) * sizeof(CompletionDesc);
    CompletionDesc* device_descs = nullptr;
    if (!backend_alloc_ring(desc_bytes, &out.host_descs, &device_descs)) {
        std::fprintf(stderr, "[gnp] ring allocation failed\n");
        session_destroy(out);
        return false;
    }
    out.ring.descs = device_descs;

    out.ctrl = static_cast<RingControl*>(backend_alloc_shared(sizeof(RingControl), false));
    out.stats = static_cast<PollStats*>(backend_alloc_shared(sizeof(PollStats), false));

    out.arena_bytes = static_cast<size_t>(cfg.ring_capacity) * cfg.payload_bytes;
    out.arena = static_cast<uint8_t*>(backend_alloc_host(out.arena_bytes));

    if (!out.host_descs || !out.ring.descs || !out.ctrl || !out.stats || !out.arena) {
        std::fprintf(stderr, "[gnp] allocation failed\n");
        session_destroy(out);
        return false;
    }

    std::memset(out.ctrl, 0, sizeof(RingControl));
    out.ctrl->publish_limit = ~0ull;
    std::memset(out.arena, 0, out.arena_bytes);
    stats_reset(*out.stats);

    out.clock_offset_ns = backend_clock_offset_ns();

    if (cfg.verbose) {
        std::printf("[gnp] memory: %s\n", backend_memory_strategy());
        std::printf("[gnp] ring: %u entries (%zu KiB), arena %zu KiB (host-only), clock offset %+lld ns\n",
                    cfg.ring_capacity, desc_bytes / 1024, out.arena_bytes / 1024,
                    static_cast<long long>(out.clock_offset_ns));
        std::printf("[gnp] ring pointers: host_staging=%p device_cq=%p%s\n",
                    static_cast<void*>(out.host_descs), static_cast<void*>(out.ring.descs),
                    out.host_descs == out.ring.descs ? " (alias)" : "");
    }
    return true;
}

void session_destroy(Session& s) {
    backend_free_host(s.arena);
    backend_free_shared(s.stats);
    backend_free_shared(s.ctrl);
    backend_free_ring(s.host_descs, s.ring.descs);
    s = Session{};
    backend_shutdown();
}

}  // namespace gnp
