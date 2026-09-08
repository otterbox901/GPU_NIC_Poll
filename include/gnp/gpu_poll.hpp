#pragma once
//
// gnp/gpu_poll.hpp - the polling backend, plus the session that owns its memory.
//
// Exactly one backend is compiled in:
//   * src/gpu/poll_kernel.cu + src/gpu/utils.cu  when a CUDA compiler was found
//   * src/gpu/poll_cpu.cpp                       otherwise
// Both implement the free functions below, so no host .cpp ever includes a
// CUDA header.
//

#include "gnp/common.hpp"
#include "gnp/metrics.hpp"
#include "gnp/ring.hpp"

namespace gnp {

/// "cuda" or "cpu-fallback". Printed in the report.
const char* backend_name();

/// Select/probe the device. Returns false if the backend is unusable.
bool backend_init(bool verbose);

/// Control/stats: memory both sides can access (pinned mapped on CUDA).
void* backend_alloc_shared(size_t bytes, bool write_combined = false);
void  backend_free_shared(void* p);

/// Completion ring: host staging (producer) + device-resident CQ (poller).
/// On the CPU backend both pointers are equal. On CUDA, `host_out` is pinned
/// staging and `device_out` is cudaMalloc'd GPU memory the SM polls locally.
bool backend_alloc_ring(size_t bytes, CompletionDesc** host_out, CompletionDesc** device_out);
void backend_free_ring(CompletionDesc* host, CompletionDesc* device);

/// Copy `count` completed CQEs from host staging into the device ring, starting
/// at monotonic index `start_idx`. Status is published last per slot so the
/// poller never sees a torn descriptor. No-op when host == device.
void backend_flush_descs(CompletionDesc* host, CompletionDesc* device, uint32_t capacity,
                         uint64_t start_idx, uint32_t count);

/// Wait for outstanding flushes (call after the producer joins, before stop).
void backend_flush_wait();

/// Tear down the H2D copy stream (called from backend_shutdown).
void backend_fini_copy();

/// Host-only allocation (payload arena). The poller never touches packet bytes.
void* backend_alloc_host(size_t bytes);
void  backend_free_host(void* p);

/// Short string describing the active shared-memory strategy (for --verbose).
const char* backend_memory_strategy();

/// Nanoseconds to add to a device timestamp to express it on the host_now_ns()
/// epoch. Zero for the CPU fallback.
int64_t backend_clock_offset_ns();

/// Launch the persistent poller. `ring.descs` must be the device-side pointer.
bool backend_launch_poller(const CompletionRing& ring, RingControl* ctrl, PollStats* stats,
                           int64_t clock_offset_ns, const RunConfig& cfg);

bool backend_wait_poller();

void backend_shutdown();

// --- session ----------------------------------------------------------------

struct Session {
    CompletionRing ring;                 ///< .descs = device pointer (poller)
    CompletionDesc* host_descs = nullptr; ///< producer staging (equals ring.descs on CPU)
    RingControl*   ctrl   = nullptr;
    PollStats*     stats  = nullptr;
    uint8_t*       arena  = nullptr;
    size_t         arena_bytes = 0;
    int64_t        clock_offset_ns = 0;
};

bool session_create(const RunConfig& cfg, Session& out);
void session_destroy(Session& s);

}  // namespace gnp
