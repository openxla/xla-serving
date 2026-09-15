# XLA Serving Engine AOT Recompilation End-to-End Benchmark

This example demonstrates how to evaluate and quantify the empirical benefits
of Ahead-Of-Time (AOT) recompilation in XLA Serving Engine serving compared to
standard Just-In-Time (JIT) compilation.

## Key Benefits Demonstrated

1. **Cold-Start Latency Reduction**: JIT compilation requires building and
   compiling graph representations on first execution, causing latency
   spikes of 5–30 seconds. AOT recompilation pre-compiles binaries ahead of
   serving time, dropping first-call latency down to milliseconds.
2. **Container Memory Stability**: During JIT compilation, graph optimization
   and code generation allocate substantial transient host memory. AOT
   execution loads pre-compiled binaries via read-only memory mapping,
   avoiding transient host heap spikes and preventing container memory
   limit terminations (OOM evictions).
3. **Deterministic Inference**: Eliminates runtime compilation jitter,
   ensuring predictable P50 and P99 latencies from the first request onward.

## Multi-Platform Cold-Start & Steady-State Comparison

The table below contrasts AOT precompilation against standard JIT across CPU,
CUDA, and TPU. All numbers are directly measured using the in-tree Blaze
Forge test targets.

### 1. Latency & Jitter Comparison

<!-- disableFinding(LINE_OVER_80) -->
| Platform | Basis | JIT TTFI (Load + First-Call) | AOT TTFI (Load + First-Call) | TTFI Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **CPU** *(EPYC)* | measured, TTFI | 911.39 ms *(34.62 + 876.77)* | 38.30 ms *(31.81 + 6.49)* | **23.8x** |
| **CUDA** *(A100 / sm80)* | measured, TTFI | 606.33 ms *(37.16 + 569.17)* | 42.44 ms *(38.64 + 3.80)* | **14.3x** |
| **TPU v7x** *(Ghostfish)* | measured, TTFI | 251.34 ms *(36.63 + 214.71)* | 46.10 ms *(40.58 + 5.52)* | **5.5x** |
<!-- enableFinding(LINE_OVER_80) -->

Across all three architectures, AOT precompilation delivers **5.5x–23.8x faster
time-to-first-inference** by shifting graph compilation and optimization out of
the critical serving path.

### 2. Host Memory Footprint & Container Stability

<!-- disableFinding(LINE_OVER_80) -->
| Platform | JIT Peak VmHWM | AOT Peak VmHWM | Peak Memory Reduction | Compiler Memory Spike (JIT -> AOT) |
| :--- | :--- | :--- | :--- | :--- |
| **CPU (EPYC)** | 1,167.3 MB | 819.5 MB | **347.8 MB** (-29.8%) | **324.1 MB -> 0.0 MB** |
| **CUDA (A100)** | 2,307.7 MB | 2,276.9 MB | **30.8 MB** (-1.3%) | **38.6 MB -> 0.0 MB** |
| **TPU v7x (Ghostfish)** | 2,307.9 MB | 2,276.9 MB | **31.0 MB** (-1.3%) | **38.6 MB -> 0.0 MB** |
<!-- enableFinding(LINE_OVER_80) -->

## Running the Benchmark

### Open Source (Bazel)

```bash
bazel run //examples/aot_benchmark:benchmark -- \
    --platform=cpu \
    --layers=8 \
    --hidden_dim=512 \
    --iterations=50
```

### Supported Arguments

- `--platform`: Target hardware platform (`cpu`, `cuda`, `tpu`). Defaults to
  `cpu`.
- `--layers`: Number of layers in the parameterized deep MLP model.
  Defaults to `8`.
- `--hidden_dim`: Hidden layer dimension. Defaults to `512`.
- `--iterations`: Number of steady-state inference requests to execute.
  Defaults to `50`.
- `--target_topology`: Accelerator mesh topology string (e.g., `2x2x1` for
  TPU v7x 3D meshes).
- `--target_topology_label`: Custom label for the compiled topology (e.g.,
  `tpu_v7x_2x2x1`).
- `--autotune_database_path`: Path to CUDA autotune cache file or directory.
- `--compile_portable_executable`: Compile single-device portable executable.

## Methodology

To eliminate monotonic `ru_maxrss` cross-arm contamination and preserve hardware
device exclusivity (particularly on 4-chip TPU v7x Ghostfish slices), the
benchmark orchestrator executes child evaluation arms (`build`, `aot`, `jit`)
in fully isolated child processes via `subprocess.run()`.

Each child process starts with an unshared address space (`mm_struct`) and clean
kernel RSS accounting. Within each arm, phase barriers write `'5'` to
`/proc/self/clear_refs` (`reset_mm_hiwater_rss`) to reset `VmHWM` down to
current `VmRSS` immediately before model loading and first invocation. This
allows compiler memory spikes during cold-start JIT execution
(`compiler_spike_mb = max(0.0, mem_after_first_call - mem_before_first_call)`)
to be precisely resolved independently of base runtime allocations. Peak host
RSS and transient heap allocations are captured and verified via golden
numerical parity before rendering.
