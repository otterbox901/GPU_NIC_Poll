//
// tests/test_ring.cpp - the owner-bit protocol, exercised on plain host memory.
//
// No GPU and no CUDA toolkit needed: the index math in ring.hpp is the same
// code the kernel runs, so testing it here catches real protocol bugs.
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gnp/common.hpp"
#include "gnp/ring.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

gnp::CompletionRing make_ring(std::vector<gnp::CompletionDesc>& storage, uint32_t capacity,
                              uint32_t shift) {
    storage.assign(capacity, gnp::CompletionDesc{});
    std::memset(storage.data(), 0, capacity * sizeof(gnp::CompletionDesc));
    gnp::CompletionRing r;
    r.descs = storage.data();
    r.capacity = capacity;
    r.mask = capacity - 1;
    r.shift = shift;
    return r;
}

void test_layout() {
    CHECK(sizeof(gnp::CompletionDesc) == 32);
    CHECK(alignof(gnp::CompletionDesc) == 32);
    CHECK(gnp::is_power_of_two(1024));
    CHECK(!gnp::is_power_of_two(1000));
    CHECK(!gnp::is_power_of_two(0));
}

void test_index_math() {
    std::vector<gnp::CompletionDesc> storage;
    const gnp::CompletionRing r = make_ring(storage, 8, 3);

    CHECK(gnp::ring_slot(r, 0) == 0);
    CHECK(gnp::ring_slot(r, 7) == 7);
    CHECK(gnp::ring_slot(r, 8) == 0);
    CHECK(gnp::ring_slot(r, 19) == 3);

    // Pass 0 expects owner 1, pass 1 expects 0, pass 2 expects 1 again.
    CHECK(gnp::ring_expected_owner(r, 0) == 1);
    CHECK(gnp::ring_expected_owner(r, 7) == 1);
    CHECK(gnp::ring_expected_owner(r, 8) == 0);
    CHECK(gnp::ring_expected_owner(r, 15) == 0);
    CHECK(gnp::ring_expected_owner(r, 16) == 1);
}

/// A freshly zeroed ring must look empty for the whole first pass, otherwise the
/// poller would report phantom packets the moment it starts.
void test_zeroed_ring_is_empty() {
    std::vector<gnp::CompletionDesc> storage;
    const gnp::CompletionRing r = make_ring(storage, 8, 3);
    for (uint64_t i = 0; i < r.capacity; ++i) {
        CHECK(!gnp::desc_ready(r.descs[gnp::ring_slot(r, i)].status,
                               gnp::ring_expected_owner(r, i)));
    }
}

/// Publish then consume across several wraps, mirroring the kernel's loop.
void test_publish_consume_wraps() {
    constexpr uint32_t kCap = 8;
    constexpr uint64_t kTotal = kCap * 5 + 3;  // deliberately not a whole number of passes

    std::vector<gnp::CompletionDesc> storage;
    const gnp::CompletionRing r = make_ring(storage, kCap, 3);

    uint64_t consumed = 0;
    for (uint64_t produced = 0; produced < kTotal; ++produced) {
        // Stale entry from the previous pass must not read as ready yet.
        CHECK(!gnp::desc_ready(r.descs[gnp::ring_slot(r, produced)].status,
                               gnp::ring_expected_owner(r, produced)));

        gnp::sim_publish(r, produced, produced * 64, 512, static_cast<uint32_t>(produced),
                         gnp::host_now_ns());

        // Consumer catches up one entry at a time, exactly like the kernel.
        CHECK(gnp::desc_ready(r.descs[gnp::ring_slot(r, consumed)].status,
                              gnp::ring_expected_owner(r, consumed)));
        const gnp::CompletionDesc& d = r.descs[gnp::ring_slot(r, consumed)];
        CHECK(d.packet_id == static_cast<uint32_t>(consumed));
        CHECK(d.byte_len == 512);
        CHECK(d.payload_offset == consumed * 64);
        ++consumed;
    }
    CHECK(consumed == kTotal);
}

/// Fill the ring completely before draining it: catches an expected-owner bug
/// that a lock-step producer/consumer would hide.
void test_full_ring_then_drain() {
    constexpr uint32_t kCap = 16;
    std::vector<gnp::CompletionDesc> storage;
    const gnp::CompletionRing r = make_ring(storage, kCap, 4);

    for (uint64_t pass = 0; pass < 3; ++pass) {
        const uint64_t base = pass * kCap;
        for (uint32_t i = 0; i < kCap; ++i) {
            gnp::sim_publish(r, base + i, i * 128, 64, static_cast<uint32_t>(base + i),
                             gnp::host_now_ns());
        }
        for (uint32_t i = 0; i < kCap; ++i) {
            const uint64_t idx = base + i;
            CHECK(gnp::desc_ready(r.descs[gnp::ring_slot(r, idx)].status,
                                  gnp::ring_expected_owner(r, idx)));
            CHECK(r.descs[gnp::ring_slot(r, idx)].packet_id == static_cast<uint32_t>(idx));
        }
        // The next unpublished index must still read as empty.
        const uint64_t next = base + kCap;
        CHECK(!gnp::desc_ready(r.descs[gnp::ring_slot(r, next)].status,
                               gnp::ring_expected_owner(r, next)));
    }
}

}  // namespace

int main() {
    test_layout();
    test_index_math();
    test_zeroed_ring_is_empty();
    test_publish_consume_wraps();
    test_full_ring_then_drain();

    if (g_failures) {
        std::fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all ring-protocol checks passed\n");
    return 0;
}
