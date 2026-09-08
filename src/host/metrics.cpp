//
// src/host/metrics.cpp - end-of-run reporting.
//

#include <cinttypes>
#include <cstdio>

#include "gnp/metrics.hpp"

namespace gnp {
namespace {

void rule() { std::printf("  --------------------------------------------------\n"); }

void row_u64(const char* label, unsigned long long v, const char* unit = "") {
    std::printf("  %-28s %14llu %s\n", label, v, unit);
}

void row_f64(const char* label, double v, const char* unit) {
    std::printf("  %-28s %14.3f %s\n", label, v, unit);
}

}  // namespace

void report(const RunConfig& cfg, const PollStats& poll, const SimStats& sim, const char* backend,
            int64_t clock_offset_ns) {
    const double elapsed_s =
        sim.end_ns > sim.start_ns ? (sim.end_ns - sim.start_ns) / 1e9 : 0.0;

    std::printf("\n=== gpu-nic-poll run summary ===\n");
    std::printf("  backend: %s   ring: %u   payload: %u B   target: %llu pps\n\n", backend,
                cfg.ring_capacity, cfg.payload_bytes,
                static_cast<unsigned long long>(cfg.target_pps));

    std::printf("  producer (simulated NIC)\n");
    rule();
    row_u64("descriptors published", sim.produced);
    row_u64("ring-full stalls", sim.overruns);
    row_f64("elapsed", elapsed_s, "s");
    if (elapsed_s > 0.0) {
        row_f64("achieved rate", sim.produced / elapsed_s / 1e6, "Mpps");
    }

    std::printf("\n  consumer (SM poller)\n");
    rule();
    row_u64("packets observed", poll.packets);
    row_u64("bytes observed", poll.bytes);
    row_u64("packet-id gaps", poll.gaps);
    row_u64("idle poll iterations", poll.idle_spins);
    if (poll.drain_spins) {
        row_u64("post-stop drain spins", poll.drain_spins);
    }
    if (poll.packets) {
        row_f64("idle spins per packet", static_cast<double>(poll.idle_spins) / poll.packets, "");
    }
    if (elapsed_s > 0.0) {
        row_f64("goodput", poll.bytes * 8.0 / elapsed_s / 1e9, "Gb/s");
    }

    const long long missed =
        static_cast<long long>(sim.produced) - static_cast<long long>(poll.packets);
    if (missed != 0) {
        std::printf("  %-28s %14lld %s\n", "NOT observed", missed,
                    "(poller stopped before drain?)");
    }

    std::printf("\n  detection latency (publish -> SM observes)\n");
    rule();
    if (poll.packets) {
        row_f64("min", poll.lat_min_ns / 1000.0, "us");
        row_f64("mean", (poll.lat_sum_ns / static_cast<double>(poll.packets)) / 1000.0, "us");
        row_f64("max", poll.lat_max_ns / 1000.0, "us");
        if (poll.clamped) {
            row_u64("clamped to zero", poll.clamped, "(clock-offset error)");
        }
    } else {
        std::printf("  %-28s %14s\n", "no packets observed", "-");
    }
    std::printf("  cross-clock offset applied: %+lld ns (calibrated, ~2 us accurate)\n",
                static_cast<long long>(clock_offset_ns));
    std::printf("\n");
}

}  // namespace gnp
