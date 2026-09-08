# gpu-nic-poll — design

## Problem

In a conventional kernel-bypass receive path the CPU burns a core spinning on a
completion queue. Every arrival costs a poll loop iteration on the host, and the
data then has to be copied or mapped to wherever it is actually consumed.

If the consumer is a GPU, all of that is wasted motion. The NIC can DMA straight
into GPU memory (GPUDirect / PeerDirect), and the completion ring can live there
too — so an SM can do the polling itself and the CPU can stay idle.

This prototype builds the polling half of that, with a software producer standing
in for the NIC.

## Structure

```
producer                    ring (GPU memory)              consumer
─────────────────────       ────────────────────────       ─────────────────
sim_inject.cpp        ──►   CompletionRing            ◄──  poll_kernel.cu
(host thread, today)        capacity × 32 B CQE            persistent kernel
                            + RingControl                  1 block, 1 thread
NIC via DOCA/GDAKI    ──►   + payload arena           ◄──  (poll_cpu.cpp when
(later)                                                     nvcc is absent)
```

The ring is the seam. Swapping the producer for real hardware should not require
touching the consumer at all.

## Arrival detection: the owner bit

The consumer cannot be told that a packet arrived — there is no doorbell it can
receive and no interrupt it can take. So arrival has to be inferable from the
descriptor memory alone.

Real NIC completion queues solve this with an **owner (phase) bit**, and this
prototype copies that design:

- `CompletionDesc::status` bit 0 is the owner bit.
- The producer flips the expected value every time it wraps the ring.
- At index `i` the consumer expects `((i / capacity) & 1) ^ 1`.
- A zeroed ring therefore reads as empty for the whole first pass — pass 0
  expects owner `1`, and freshly allocated memory holds `0`.

The payoff is that the consumer never needs a producer index. It reads one
32-bit word, compares one bit, and knows whether slot `i` holds a new entry. No
shared counter, no atomics, no host round trip.

`ring_slot`, `ring_expected_owner` and `desc_ready` are `GNP_HD` inline
functions in `ring.hpp`, so the kernel, the CPU fallback and the unit tests all
run the *same* index math. A protocol bug cannot hide in one copy.

## Publication ordering

The producer must not let the owner bit become visible before the payload
fields. `sim_publish()` writes `payload_offset`, `byte_len`, `packet_id`,
`post_ns`, then a barrier, then `status`.

The barrier is `memory_order_seq_cst`, not `release`, and that is deliberate. On
x86 a release fence is only a compiler barrier — correct for write-back memory,
but these stores may land in write-combining PCIe-mapped memory, where the CPU
reorders freely. A seq_cst fence emits a real `mfence` and drains the WC
buffers.

On the consumer side, `__threadfence_system()` after the `status` load keeps the
payload loads from floating above it.

## Memory placement

`backend_alloc_shared()` uses **pinned, mapped host memory** (`cudaHostAlloc` +
`cudaHostAllocMapped`). The CPU producer writes into local DRAM; the SM polls
those cache lines over PCIe. That is the opposite of the eventual NIC topology
(DMA into GPU DRAM), but it is the right model while the producer is a host
thread: `cudaMallocManaged` with a GPU-preferred location thrash-migrates pages
under bidirectional touch and shows up as multi-millisecond detection latency.

The payload arena is ordinary host memory — the poller never reads packet bytes.

When a real NIC arrives, swap the allocator to device memory (or DOCA GPUNetIO
buffers). The owner-bit protocol and the poll kernel stay the same.

Device-side loads are `volatile`. The GPU L1 is not coherent with writes
arriving from outside the SM, so a cached read would spin forever on a stale
value.

## Why one thread

`gnp_poll_kernel` runs as `<<<1, 1>>>`. A completion queue is consumed strictly
in order, so exactly one lane can own the head; extra lanes would contend on the
same descriptor and buy nothing. Payload *processing* is where width belongs,
and that is out of scope for v1 — see "Next steps".

## Stopping

`RingControl::publish_limit` and `stop_flag` are host→device. After the producer
quiesces, the host stores the final produced count into `publish_limit`, fences,
then raises `stop_flag`. The kernel checks these **only on the idle path**, and
only retires once `idx >= publish_limit`. That closes the race where `stop`
becomes visible before the last CQE status bit — which previously showed up as
`published == observed + 1` on small rings.

There is also a hard `max_run_ns` ceiling (requested duration + 5 s) checked
every 1024 idle spins and every 64 packets, so a crashed host cannot wedge the
GPU indefinitely.

`RingControl::consumed` goes the other way, device→host, updated every 64
entries. The producer uses it for back-pressure. It is deliberately stale — a
lagging watermark only ever makes the producer more conservative.

## Latency measurement

`%globaltimer` and `std::chrono::steady_clock` do not share an epoch, so the kernel is
handed a precomputed `clock_offset_ns`.

Calibration (`backend_clock_offset_ns`) launches a probe kernel that spins on a
host flag, waits for it to become resident, then times the flag raise. Having
the kernel already spinning is the point: launching a kernel just to read the
timer would fold ~10 µs of launch latency into the offset. The best of 16
samples lands within roughly ±2 µs.

The residual bias is one PCIe hop and it *under*-reports latency slightly, since
the GPU observes the flag after the host timestamp. Latencies that come out
negative are clamped to zero and counted in `PollStats::clamped` — a non-zero
count there means the calibration, not the poller, needs attention.

## Non-goals for v1

No DPDK, no kernel module, no userspace network stack, no multi-queue, no RSS,
no packet parsing, no payload processing, no reliability layer.

## Next steps, roughly in order

1. **Payload processing width.** Keep lane 0 as the CQ head, hand descriptors to
   the rest of the warp for checksum/parse work.
2. **Occupancy honesty.** A persistent kernel holds an SM forever. Measure what
   that costs a co-resident compute kernel.
3. **Multi-queue.** One ring per block, so `N` rings poll independently.
4. **Real hardware.** Replace `sim_inject.cpp` with DOCA GPUNetIO / GDAKI. The
   ring, the owner-bit protocol and the kernel should not need to change.
