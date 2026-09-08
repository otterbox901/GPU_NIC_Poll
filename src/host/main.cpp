//
// src/host/main.cpp - set up, launch the poller, then get out of the way.
//
// The whole point of the project is that this thread does nothing during
// steady-state receive. It allocates, launches, sleeps, and reports.
//

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "gnp/common.hpp"
#include "gnp/gpu_poll.hpp"
#include "gnp/metrics.hpp"
#include "gnp/ring.hpp"

namespace {

void usage(const char* argv0) {
    std::printf(
        "usage: %s [options]\n"
        "  --ring N          completion-ring entries, power of two (default 1024)\n"
        "  --size B          simulated packet size in bytes (default 1024)\n"
        "  --pps R           target injection rate, 0 = unpaced (default 100000)\n"
        "  --packets N       stop after N packets, 0 = until duration (default 0)\n"
        "  --duration MS     how long to run (default 2000)\n"
        "  --burst N         descriptors published back-to-back (default 1)\n"
        "  --backoff NS      relax the poll loop when idle, 0 = pure spin (default 0)\n"
        "  --verbose         print device and allocation details\n"
        "  --help\n",
        argv0);
}

/// Returns false if the value is missing or unparseable.
bool take_u64(int argc, char** argv, int& i, uint64_t& out) {
    if (i + 1 >= argc) return false;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(argv[++i], &end, 10);
    if (end == argv[i] || *end != '\0') return false;
    out = v;
    return true;
}

bool parse_args(int argc, char** argv, gnp::RunConfig& cfg, bool& want_help) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        uint64_t v = 0;
        bool ok = true;

        if (a == "--help" || a == "-h") {
            want_help = true;
            return true;
        } else if (a == "--verbose") {
            cfg.verbose = true;
        } else if (a == "--ring") {
            ok = take_u64(argc, argv, i, v);
            cfg.ring_capacity = static_cast<uint32_t>(v);
        } else if (a == "--size") {
            ok = take_u64(argc, argv, i, v);
            cfg.payload_bytes = static_cast<uint32_t>(v);
        } else if (a == "--pps") {
            ok = take_u64(argc, argv, i, v);
            cfg.target_pps = v;
        } else if (a == "--packets") {
            ok = take_u64(argc, argv, i, v);
            cfg.max_packets = v;
        } else if (a == "--duration") {
            ok = take_u64(argc, argv, i, v);
            cfg.duration_ms = static_cast<uint32_t>(v);
        } else if (a == "--burst") {
            ok = take_u64(argc, argv, i, v);
            cfg.burst = static_cast<uint32_t>(v);
        } else if (a == "--backoff") {
            ok = take_u64(argc, argv, i, v);
            cfg.idle_backoff_ns = static_cast<uint32_t>(v);
        } else {
            std::fprintf(stderr, "[gnp] unknown option: %s\n", a.c_str());
            return false;
        }

        if (!ok) {
            std::fprintf(stderr, "[gnp] bad or missing value for %s\n", a.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    gnp::RunConfig cfg;
    bool want_help = false;
    if (!parse_args(argc, argv, cfg, want_help)) return 2;
    if (want_help) {
        usage(argv[0]);
        return 0;
    }

#if !defined(GNP_SIMULATION)
    std::fprintf(stderr,
                 "[gnp] built with GNP_SIMULATION=OFF and no hardware producer exists yet\n");
    return 1;
#else
    gnp::Session session;
    if (!gnp::session_create(cfg, session)) return 1;

    // 1. The poller goes first, so it is already spinning when packets appear.
    if (!gnp::backend_launch_poller(session.ring, session.ctrl, session.stats,
                                    session.clock_offset_ns, cfg)) {
        gnp::session_destroy(session);
        return 1;
    }

    // 2. Then the producer. In the hardware build this is the NIC's Rx queue.
    gnp::Simulator* sim = gnp::sim_start(session.ring, session.ctrl, session.arena,
                                         session.arena_bytes, cfg);
        if (!sim) {
        std::fprintf(stderr, "[gnp] failed to start the simulator\n");
        reinterpret_cast<std::atomic<unsigned long long>*>(&session.ctrl->publish_limit)
            ->store(0ull, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        reinterpret_cast<std::atomic<uint32_t>*>(&session.ctrl->stop_flag)
            ->store(1u, std::memory_order_release);
        gnp::backend_wait_poller();
        gnp::session_destroy(session);
        return 1;
    }

    std::printf("[gnp] polling on %s backend for %u ms...\n", gnp::backend_name(),
                cfg.duration_ms);

    // 3. Steady state. This is the interesting part: the host is asleep while
    //    the SM does all the receive-side work.
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.duration_ms));

    gnp::SimStats sim_stats;
    gnp::sim_stop(sim, sim_stats);

    // 4. Retire the poller. publish_limit first, then stop_flag: the kernel only
    //    leaves the idle path once idx has caught the final produced count, so a
    //    stop that becomes visible before the last CQE cannot truncate the drain.
    reinterpret_cast<std::atomic<unsigned long long>*>(&session.ctrl->publish_limit)
        ->store(sim_stats.produced, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    reinterpret_cast<std::atomic<uint32_t>*>(&session.ctrl->stop_flag)
        ->store(1u, std::memory_order_release);

    const bool ok = gnp::backend_wait_poller();

    gnp::report(cfg, *session.stats, sim_stats, gnp::backend_name(), session.clock_offset_ns);
    gnp::session_destroy(session);
    return ok ? 0 : 1;
#endif
}
