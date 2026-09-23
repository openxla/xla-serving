# XLA Serving Engine AOT Recompilation End-to-End Benchmark

This example demonstrates how to evaluate and quantify the empirical benefits
of Ahead-Of-Time (AOT) recompilation in XLA Serving Engine serving compared to
standard Just-In-Time (JIT) compilation.

## Empirical Highlights & Serving Benefits

Across all three architectures (CPU, CUDA A100, TPUv7x), Ahead-Of-Time (AOT)
recompilation delivers three verified serving improvements:

1. **Request Critical-Path Latency**: Slashes cold-start first-call latency
   across all platforms: **321x on CPU** (`1613 ms -> 5.03 ms`), **30x on
   TPUv7x** (`198.9 ms -> 6.63 ms`), and **1.5x / -35% on CUDA A100**
   (`1409 ms -> 923 ms`, saving 486 ms of XLA compilation). Once one-time
   device module loading completes, Call 1 lands in sub-millisecond steady
   state (`<0.5 ms`) across all architectures.
2. **Elimination of Compiler Heap Spikes**: **100% elimination of transient
   XLA compiler memory allocations** during serving (`220.3 MB -> 0 MB` on CPU,
   `111.6 MB -> 0 MB` on CUDA A100, `41.7 MB -> 0 MB` on TPUv7x), reducing
   peak container memory footprint by **65 MB to 223 MB** and preventing
   container out-of-memory (OOM) evictions.
3. **Transparent Lifecycle Trade-Off**: Shifts ~67 ms of executable
   deserialization into offline container boot time (`t_load`) so live user
   requests never stall on runtime graph compilation.

## Multi-Platform Cold-Start & Memory Comparison

The table below contrasts AOT precompilation against standard JIT across CPU,
CUDA, and TPU. All numbers are directly measured using the in-tree test targets.

### 1. Latency & Cold-Start Comparison

<!-- disableFinding(LINE_OVER_80) -->
| Platform | JIT Cold-Start Call 0 (median [min, max]) | AOT Cold-Start Call 0 (median [min, max]) | Cold-Start Delta (XLA Compile Saved) | Call 1 Kernel Cache (<5 ms) | Model Load, Warm Driver (JIT / AOT) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | 1613.32 ms [1496.3, 1882.2] | 5.03 ms [3.8, 7.8] | **-1608.29 ms (321x)** | 0.55 ms / 0.32 ms (PASS) | 22.32 ms / 17.07 ms |
| **CUDA** *(A100)* | 1408.98 ms [1359.5, 2819.2] | 923.08 ms [737.8, 937.4] | **-485.89 ms** *(485.9 ms compile saved)* | 0.47 ms / 0.40 ms (PASS) | 71.47 ms / 138.30 ms |
| **TPUv7x** | 198.90 ms [196.9, 199.9] | 6.63 ms [5.9, 6.9] | **-192.27 ms (30.0x / 96.7%)** | 0.38 ms / 0.47 ms (PASS) | 960.72 ms / 941.35 ms |
<!-- enableFinding(LINE_OVER_80) -->

Across all three architectures, AOT eliminates graph compilation overhead from
the serving critical path. All Call 0 figures are the median of 5 independent
subprocess runs, with the observed `[min, max]` spread shown. Cold-start
latency drops by **30x–321x** on CPU and TPUv7x (`1613.32 ms -> 5.03 ms` on
CPU, `198.90 ms -> 6.63 ms` on TPUv7x), and saves **485.89 ms** of XLA
compilation on CUDA A100 (`1408.98 ms -> 923.08 ms`). Across all platforms,
Call 1 lands well under 1 ms once device kernel modules are cached.

### 2. Host Memory Footprint & Container Stability

<!-- disableFinding(LINE_OVER_80) -->
| Platform | JIT Peak VmHWM | AOT Peak VmHWM | Peak Memory Reduction | First-Call Host Delta (JIT / AOT) | Attributed Compiler Spike (JIT -> AOT) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU (EPYC)** | 2324.2 MB | 2100.8 MB | **223.4 MB** (-9.6%) | 220.8 MB / 0.4 MB | **220.3 MB -> 0.0 MB** |
| **CUDA (A100)** | 2932.9 MB | 2777.6 MB | **155.3 MB** (-5.3%) | 1062.4 MB / 950.7 MB | **111.6 MB -> 0.0 MB** |
| **TPUv7x** | 4380.4 MB | 4314.9 MB | **65.6 MB** (-1.5%) | 42.6 MB / 0.9 MB | **41.7 MB -> 0.0 MB** |
<!-- enableFinding(LINE_OVER_80) -->

## Running the Benchmark

### Open Source (Bazel)

```bash
bazel run //examples/aot_benchmark:benchmark -- \
    --platform=cpu \
    --layers=8 \
    --hidden_dim=512
```

### Supported Arguments

- `--platform`: Target hardware platform (`cpu`, `cuda`, `tpu`). Defaults to
  `cpu`.
- `--layers`: Number of layers in the parameterized deep MLP model.
  Defaults to `8`.
- `--hidden_dim`: Hidden layer dimension. Defaults to `512`.
- `--repeats`: Number of independent cold-start samples per arm (defaults to
  `5`).
- `--target_topology`: Accelerator mesh topology string (e.g., `2x2x1` for
  TPU v7x 3D meshes).
- `--target_topology_label`: Custom label for the compiled topology (e.g.,
  `tpu_v7x_2x2x1`).
- `--autotune_database_path`: Path to CUDA autotune cache file or directory.
- `--compile_portable_executable`: Compile single-device portable executable.

## Methodology

To eliminate monotonic `ru_maxrss` cross-arm contamination and preserve hardware
device exclusivity (particularly on 4-chip TPUv7x slices), the
benchmark orchestrator executes child evaluation arms (`build`, `aot`, `jit`)
in fully isolated child processes via `subprocess.run()`.

Each child process starts with an unshared address space (`mm_struct`) and clean
kernel RSS accounting. Within each arm, phase barriers write `'5'` to
`/proc/self/clear_refs` (`reset_mm_hiwater_rss`) to reset `VmHWM` down to
current `VmRSS` immediately before model loading and first invocation. This
allows first-call host memory expansion
(`first_call_delta_mb = max(0.0, mem_after_first_call -`
`mem_before_first_call)`)
and attributed XLA compiler memory spikes
(`attributed_jit_spike_mb = max(0.0, jit_delta - aot_delta)`)
to be precisely resolved independently of base runtime allocations. Peak host
RSS and transient heap allocations are captured and verified via golden
numerical parity before rendering.
