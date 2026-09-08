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

/// Allocate memory both the producer (CPU/NIC) and the consumer (SM) can reach.
/// Under CUDA this is managed memory pinned to the device, so host stores travel
/// over PCIe into GPU DRAM - the same direction a real NIC DMA would take.
void* backend_alloc_shared(size_t bytes);
void  backend_free_shared(void* p);

/// Nanoseconds to add to a device timestamp to express it on the host_now_ns()
/// epoch. Zero for the CPU fallback.
int64_t backend_clock_offset_ns();

/// Launch the persistent poller. Returns immediately; the poller runs until
/// RingControl::stop_flag is set.
bool backend_launch_poller(const CompletionRing& ring, RingControl* ctrl, PollStats* stats,
                           int64_t clock_offset_ns, const RunConfig& cfg);

/// Block until the poller has retired.
bool backend_wait_poller();

void backend_shutdown();

// --- session ----------------------------------------------------------------

/// Everything one run allocates. Created by setup.cpp.
struct Session {
    CompletionRing ring;
    RingControl*   ctrl   = nullptr;
    PollStats*     stats  = nullptr;
    uint8_t*       arena  = nullptr;   ///< simulated packet payload buffer
    size_t         arena_bytes = 0;
    int64_t        clock_offset_ns = 0;
};

bool session_create(const RunConfig& cfg, Session& out);
void session_destroy(Session& s);

}  // namespace gnp
