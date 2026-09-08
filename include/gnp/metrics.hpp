#pragma once
//
// gnp/metrics.hpp - counters the poller fills in, and the end-of-run report.
//

#include "gnp/common.hpp"
#include "gnp/ring.hpp"

namespace gnp {

/// Written by the poller (device or CPU fallback), read by the host at teardown.
///
/// `unsigned long long` throughout so the fields stay atomicAdd-compatible if
/// the poll loop is ever widened to more than one thread.
struct PollStats {
    unsigned long long packets;        ///< descriptors observed
    unsigned long long bytes;          ///< sum of byte_len
    unsigned long long idle_spins;     ///< poll iterations that found nothing
    unsigned long long lat_sum_ns;     ///< sum of sampled publish -> detect latency
    unsigned long long lat_min_ns;
    unsigned long long lat_max_ns;
    unsigned long long lat_samples;    ///< how many packets contributed to lat_*
    unsigned long long gaps;           ///< packet_id discontinuities
    unsigned long long clamped;        ///< latencies clamped to 0 by clock skew
    unsigned long long drain_spins;    ///< idle spins after stop while catching publish_limit
    unsigned long long run_ns;         ///< device-side elapsed (%globaltimer) while poller ran
};

void stats_reset(PollStats& s);

/// Print the end-of-run summary to stdout.
void report(const RunConfig& cfg, const PollStats& poll, const SimStats& sim,
            const char* backend, int64_t clock_offset_ns);

}  // namespace gnp
