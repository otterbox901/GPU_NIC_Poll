# GPU-NIC-Poll

Moving steady-state receive-side completion-queue polling off the CPU and onto
GPU SMs.

A `CompletionRing` lives in GPU memory. A persistent CUDA kernel polls it from an
SM and counts arrivals. The host allocates, launches, sleeps, and reports — it
does no per-packet work at all.

Real NIC hardware is not wired up yet, so a host thread stands in for it. See
[`docs/design.md`](docs/design.md) for the protocol and the reasoning.

## Build

```bash
cmake --preset sim
cmake --build --preset sim
ctest --preset sim
```

CUDA is **auto-detected**. If `nvcc` is found you get the real persistent kernel
(`src/gpu/poll_kernel.cu`); if not, CMake warns and builds an equivalent poll
loop on a host thread (`src/gpu/poll_cpu.cpp`) so the ring, simulator, metrics
and tests still work. Which one is active is printed at configure time
(`gnp: simulation=ON cuda=ON/OFF ...`) and in every run summary.

The CUDA backend keeps the completion ring in **pinned mapped host memory** so
the CPU producer writes locally and the SM polls over PCIe. That avoids
`cudaMallocManaged` page-migration thrashing while the producer is still a host
thread; a real NIC will DMA into device memory later without changing the
poller.

## Run

```bash
./build/sim/gnp --verbose                       # defaults: 1024-entry ring, 100 kpps, 2 s
./build/sim/gnp --pps 0 --ring 64               # unpaced, exercises back-pressure
./build/sim/gnp --packets 50000 --burst 16      # fixed count, bursty arrivals
./build/sim/gnp --help
```

| flag | meaning | default |
|---|---|---|
| `--ring N` | completion-ring entries, power of two | 1024 |
| `--size B` | simulated packet size | 1024 |
| `--pps R` | target injection rate, `0` = unpaced | 100000 |
| `--packets N` | stop after N packets, `0` = use duration | 0 |
| `--duration MS` | how long to run | 2000 |
| `--burst N` | descriptors published back-to-back | 1 |
| `--backoff NS` | relax the poll loop when idle, `0` = pure spin | 0 |
| `--verbose` | device, allocation and clock-offset details | off |

## Reading the output

The two numbers that matter for correctness are **`packet-id gaps`** and the
difference between `descriptors published` and `packets observed`. Both must be
zero. A gap means the poller mis-sequenced the ring; a shortfall means it
stopped before draining.

`idle spins per packet` is the interesting performance number — it is how many
times the SM read a descriptor and found nothing, per useful packet. At low
rates it is large by construction, which is exactly the cost this design accepts
in exchange for leaving the CPU alone.

## Installing CUDA (Fedora)

Only the NVIDIA driver is required at runtime, but `nvcc` is needed to build the
real kernel:

```bash
sudo dnf install cuda-toolkit
```

Two things commonly bite on a current Fedora:

- **Host compiler too new.** `nvcc` rejects host compilers it does not
  recognise, and Fedora's GCC runs ahead of every CUDA release. `cmake/gnp_cuda.cmake`
  looks for `g++-14`/`13`/`12`/`11` automatically — install one
  (`sudo dnf install gcc14-c++`) or pass
  `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14`.
- **Display-GPU watchdog.** If the GPU also drives your desktop, a persistent
  kernel will freeze the display for as long as it runs and may be killed by the
  X/Wayland watchdog after a few seconds. Keep `--duration` short (the 2 s
  default is deliberate), or run on a GPU that is not driving a display.

## Layout

```
include/gnp/
  common.hpp      RunConfig, clocks, GNP_HD, GNP_CUDA_CHECK
  ring.hpp        CompletionDesc / CompletionRing / RingControl, owner-bit math,
                  producer API
  metrics.hpp     PollStats, report()
  gpu_poll.hpp    backend interface + Session
src/host/
  main.cpp        argument parsing and orchestration
  setup.cpp       allocation, teardown, host_now_ns()
  sim_inject.cpp  the stand-in NIC — this is what real hardware replaces
  metrics.cpp     end-of-run summary
src/gpu/
  poll_kernel.cu  the persistent poller
  utils.cu        device probe, shared allocation, clock calibration
  poll_cpu.cpp    CPU fallback, built only when nvcc is missing
tests/
  test_ring.cpp   owner-bit protocol on plain host memory, no GPU needed
```
