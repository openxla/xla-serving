# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scalable End-to-End AOT vs JIT Benchmark.

Demonstrates the physical benefits of XLA Serving Engine AOT Recompilation:
1. Slashing first-call cold-start compilation latency from seconds to
   milliseconds.
2. Preventing host memory out-of-memory evictions by bypassing transient
   XLA compiler peak allocations.
3. Eliminating runtime compilation jitter with verified numerical parity.
"""

from collections.abc import Sequence
import contextlib
import gc
import json
import os
import pathlib
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from .third_party.neptune import xla_saved_model
from .third_party.neptune.runtime.python import orbax_model_runner as model_runner
from xla_saved_model._src.core import aot_recompiler
from xla_saved_model._src.core import offline_compiler


_PLATFORM = flags.DEFINE_enum(
    "platform", "cpu", ["cpu", "cuda", "tpu"], "Target hardware platform."
)
_LAYERS = flags.DEFINE_integer("layers", 8, "Number of layers in MLP.")
_HIDDEN_DIM = flags.DEFINE_integer("hidden_dim", 512, "Hidden dimension.")
_REPEATS = flags.DEFINE_integer(
    "repeats",
    5,
    "Independent subprocess samples per arm. Cold-start is a once-per-process "
    "event, so each sample requires a fresh process.",
)
_TARGET_TOPOLOGY = flags.DEFINE_string(
    "target_topology",
    None,
    "Accelerator mesh topology (e.g., '2x2x1' for TPU v7x 3D meshes).",
)
_TARGET_TOPOLOGY_LABEL = flags.DEFINE_string(
    "target_topology_label",
    None,
    "Custom target topology label to tag the AOT compiled executable "
    "(e.g., 'tpu_v7x_2x2x1').",
)
_AUTOTUNE_DATABASE_PATH = flags.DEFINE_string(
    "autotune_database_path",
    None,
    "Path to offline autotune database for CUDA fusions.",
)
_COMPILE_PORTABLE = flags.DEFINE_boolean(
    "compile_portable_executable",
    False,
    "Compile single-device portable executable.",
)
_PHASE = flags.DEFINE_enum(
    "phase",
    "orchestrate",
    ["orchestrate", "build", "aot", "jit"],
    "Execution phase.",
)
_WORK_DIR = flags.DEFINE_string(
    "work_dir",
    None,
    "Shared directory holding exported base and AOT models.",
)

_METRICS_SENTINEL = "__BENCHMARK_METRICS_JSON__"
_CALL1_CACHE_BOUND_MS = 5.0


def _read_status_kb(field: str) -> float:
  """Reads a kB-valued field from /proc/self/status."""
  prefix = f"{field}:"
  try:
    with open("/proc/self/status", "rt") as f:
      for line in f:
        if line.startswith(prefix):
          return float(line.split()[1])
  except OSError:
    pass
  return 0.0


def get_peak_memory_mb() -> float:
  """Returns the resettable peak RSS (VmHWM) in megabytes, with fallback."""
  kb = _read_status_kb("VmHWM")
  if kb > 0.0:
    return kb / 1024.0
  # Fallback for non-Linux or /proc unavailable.
  try:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
      return float(usage.ru_maxrss) / (1024.0 * 1024.0)
    return float(usage.ru_maxrss) / 1024.0
  except (OSError, ValueError):
    return 0.0


def get_current_memory_mb() -> float:
  """Returns current resident set size (VmRSS) in megabytes."""
  kb = _read_status_kb("VmRSS")
  return kb / 1024.0


def reset_peak_memory() -> bool:
  """Resets VmHWM to current RSS via /proc/self/clear_refs."""
  try:
    with open("/proc/self/clear_refs", "wt") as f:
      f.write("5")
    hwm_kb = _read_status_kb("VmHWM")
    rss_kb = _read_status_kb("VmRSS")
    return hwm_kb > 0.0 and (hwm_kb - rss_kb) <= 4096.0
  except OSError:
    return False


def create_benchmark_model(
    export_path: str,
    num_layers: int,
    hidden_dim: int,
    platform: str,
) -> None:
  """Creates a deep parameterizable MLP that scales XLA compiler workload."""

  def forward(params, inputs):
    h = inputs["x"]
    for i in range(num_layers):
      w = params[f"w_{i}"]
      b = params[f"b_{i}"]
      h = jnp.matmul(h, w) + b
      if i < num_layers - 1:
        h = jax.nn.gelu(h)
    return {"out": h}

  params = {}
  rng = np.random.default_rng(0)
  for i in range(num_layers):
    in_dim = 64 if i == 0 else hidden_dim
    out_dim = 10 if i == num_layers - 1 else hidden_dim
    scale = np.float32(1.0 / np.sqrt(in_dim))
    params[f"w_{i}"] = jnp.asarray(
        rng.standard_normal((in_dim, out_dim), dtype=np.float32) * scale
    )
    params[f"b_{i}"] = jnp.asarray(
        rng.standard_normal((out_dim,), dtype=np.float32) * np.float32(0.01)
    )

  spec = jax.ShapeDtypeStruct((1, 64), jnp.float32)

  bundle = xla_saved_model.ApplyFnBundle(
      apply_fn=forward,
      jax_export_kwargs={"platforms": [platform]},
  )
  model = xla_saved_model.JaxModel(
      apply_fn_bundle=bundle,
      params=params,
      name="bench_model",
  )
  cfg = xla_saved_model.ServingConfig(
      serving_signature="serving_default",
      input_signature={"x": spec},
  )

  xla_saved_model.save(
      models=model,
      export_path=export_path,
      serving_configs=[cfg],
  )


def run_benchmark(
    model_path: str,
    platform: str,
    target_topology: str,
    phase: str = "jit",
) -> tuple[dict[str, Any], np.ndarray]:
  """Runs model loading, cold-start inference, and second-call latency."""
  kwargs = {
      "model_name": "bench_model",
      "model_path": model_path,
      "platform": platform,
      "allow_jit_fallback_for_aot": False,
  }
  if target_topology:
    kwargs["target_topology_label"] = target_topology

  # 0. Pre-warm hardware driver (libtpu.so singleton / CUDA primary context).
  # Destroying warmup_runner releases its BFCAllocator memory pool on GPU while
  # preserving process-wide driver initialization so t_load measures pure model
  # deserialization without +-1000ms cloud driver spin-up jitter.
  warmup_runner = model_runner.ModelRunner(**kwargs)
  del warmup_runner
  gc.collect()

  sample_input = np.random.default_rng(1).standard_normal(
      (1, 64), dtype=np.float32
  )

  reset_warmup_ok = reset_peak_memory()
  mem_runtime_baseline = get_peak_memory_mb()
  t0 = time.perf_counter()
  runner = model_runner.ModelRunner(**kwargs)

  t_load = (time.perf_counter() - t0) * 1000.0  # ms
  mem_after_load = get_peak_memory_mb()
  sig = runner.signature("serving_default")

  # 1. Call 0: Cold-Start First-Call Latency (XLA JIT Compile + Lazy Thunk Init)
  reset_load_ok = reset_peak_memory()
  reset_ok = reset_warmup_ok and reset_load_ok
  mem_before_first_call = get_peak_memory_mb()
  rss_before_first_call = get_current_memory_mb()

  t1 = time.perf_counter()
  first_output = sig.run(x=sample_input)[0]
  t_cold_start = (time.perf_counter() - t1) * 1000.0  # ms
  mem_after_first_call = get_peak_memory_mb()
  rss_after_first_call = get_current_memory_mb()
  if reset_load_ok:
    first_call_delta_mb = max(
        0.0, mem_after_first_call - mem_before_first_call
    )
  else:
    first_call_delta_mb = max(
        0.0, rss_after_first_call - rss_before_first_call
    )

  # 2. Call 1: Second-Call Latency (Kernel cache warm; isolates lazy thunk init)
  t2 = time.perf_counter()
  _ = sig.run(x=sample_input)
  t_second_call = (time.perf_counter() - t2) * 1000.0  # ms

  load_delta_mb = max(0.0, mem_after_load - mem_runtime_baseline)
  process_peak_mb = max(
      mem_runtime_baseline,
      mem_after_load,
      mem_after_first_call,
      get_peak_memory_mb(),
  )

  metrics = {
      "phase": phase,
      "load_ms": t_load,
      "cold_start_ms": t_cold_start,
      "second_call_ms": t_second_call,
      "runtime_baseline_mb": mem_runtime_baseline,
      "load_peak_mb": mem_after_load,
      "load_delta_mb": load_delta_mb,
      "first_call_delta_mb": first_call_delta_mb,
      "process_peak_mb": process_peak_mb,
      "clear_refs_ok": reset_ok,
      "first_output": [float(x) for x in np.asarray(first_output).flat],
  }
  return metrics, np.asarray(first_output)


def _get_subprocess_base_cmd() -> list[str]:
  argv0 = sys.argv[0]
  resolved = shutil.which(argv0) or os.path.abspath(argv0)
  if resolved.endswith(".py") and sys.executable:
    return [sys.executable, "-u", resolved]
  return [resolved]


def _spawn_phase(phase: str, work_dir: str) -> dict[str, Any]:
  """Re-executes this binary in a clean process and returns its JSON result."""
  cmd = _get_subprocess_base_cmd() + [
      f"--phase={phase}",
      f"--work_dir={work_dir}",
      f"--platform={_PLATFORM.value}",
      f"--layers={_LAYERS.value}",
      f"--hidden_dim={_HIDDEN_DIM.value}",
  ]
  if _TARGET_TOPOLOGY.value:
    cmd.append(f"--target_topology={_TARGET_TOPOLOGY.value}")
  if _TARGET_TOPOLOGY_LABEL.value:
    cmd.append(f"--target_topology_label={_TARGET_TOPOLOGY_LABEL.value}")
  if _AUTOTUNE_DATABASE_PATH.value:
    cmd.append(f"--autotune_database_path={_AUTOTUNE_DATABASE_PATH.value}")
  if _COMPILE_PORTABLE.value:
    cmd.append("--compile_portable_executable")

  proc = subprocess.run(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(
        f"Phase '{phase}' failed with exit code {proc.returncode}:\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}"
    )
  for line in reversed(proc.stdout.splitlines()):
    if line.startswith(_METRICS_SENTINEL):
      payload = line[len(_METRICS_SENTINEL):].strip()
      return json.loads(payload)
  raise RuntimeError(
      f"Phase '{phase}' did not emit valid JSON on stdout.\n"
      f"--- STDOUT ---\n{proc.stdout}\n"
      f"--- STDERR ---\n{proc.stderr}"
  )


def _aggregate(samples: list[dict[str, Any]], key: str) -> dict[str, float]:
  """Returns median, min, and max of `key` across independent samples."""
  values = sorted(s[key] for s in samples)
  return {
      "median": values[len(values) // 2],
      "min": values[0],
      "max": values[-1],
  }


def _spawn_phase_repeated(
    phase: str, work_dir: str, repeats: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Spawns `repeats` independent cold-start samples for one arm."""
  samples = [_spawn_phase(phase, work_dir) for _ in range(repeats)]
  merged = dict(samples[-1])
  for key in (
      "cold_start_ms",
      "load_ms",
      "second_call_ms",
      "first_call_delta_mb",
      "process_peak_mb",
  ):
    stats = _aggregate(samples, key)
    merged[key] = stats["median"]
    merged[f"{key}_min"] = stats["min"]
    merged[f"{key}_max"] = stats["max"]
  return merged, samples


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.

  phase = _PHASE.value
  work_dir = _WORK_DIR.value

  if phase == "build":
    if not work_dir:
      raise ValueError("--work_dir is required for build phase.")
    base_path = pathlib.Path(work_dir) / "base_model"
    aot_path = pathlib.Path(work_dir) / "aot_model"
    create_benchmark_model(
        export_path=str(base_path),
        num_layers=_LAYERS.value,
        hidden_dim=_HIDDEN_DIM.value,
        platform=_PLATFORM.value,
    )
    topology_label = _TARGET_TOPOLOGY_LABEL.value or _PLATFORM.value
    target_config = offline_compiler.TargetConfig(
        platform=_PLATFORM.value,
        target_topology_label=topology_label,
        target_topology=_TARGET_TOPOLOGY.value,
        autotune_database_path=_AUTOTUNE_DATABASE_PATH.value,
        compile_portable_executable=_COMPILE_PORTABLE.value,
    )
    aot_recompiler.recompile(
        input_model_path=str(base_path),
        output_model_path=str(aot_path),
        target_configs=target_config,
        include_stablehlo=False,
        auto_timestamp=False,
    )
    print(f"{_METRICS_SENTINEL}{json.dumps({'status': 'ok'})}")
    sys.stdout.flush()
    return

  if phase == "aot":
    if not work_dir:
      raise ValueError("--work_dir is required for aot phase.")
    aot_path = pathlib.Path(work_dir) / "aot_model"
    topology_label = _TARGET_TOPOLOGY_LABEL.value or _PLATFORM.value
    metrics, _ = run_benchmark(
        model_path=str(aot_path),
        platform=_PLATFORM.value,
        target_topology=topology_label,
        phase="aot",
    )
    print(f"{_METRICS_SENTINEL}{json.dumps(metrics)}")
    sys.stdout.flush()
    return

  if phase == "jit":
    if not work_dir:
      raise ValueError("--work_dir is required for jit phase.")
    base_path = pathlib.Path(work_dir) / "base_model"
    metrics, _ = run_benchmark(
        model_path=str(base_path),
        platform=_PLATFORM.value,
        target_topology="",
        phase="jit",
    )
    print(f"{_METRICS_SENTINEL}{json.dumps(metrics)}")
    sys.stdout.flush()
    return

  if phase != "orchestrate":
    raise ValueError(f"Unknown phase: {phase}")

  platform = _PLATFORM.value
  layers = _LAYERS.value
  hidden_dim = _HIDDEN_DIM.value

  print("=" * 80)
  print(
      " XLA Serving Engine AOT Recompilation E2E Benchmark"
      f" (Platform: {platform.upper()})"
  )
  print(f" Model: Parameterized MLP ({layers} layers, {hidden_dim} dim)")
  print("=" * 80)

  ctx = (
      contextlib.nullcontext(work_dir)
      if work_dir
      else tempfile.TemporaryDirectory()
  )
  with ctx as actual_work_dir:
    print("\n[Step 1/3] Building models in isolated subprocess...")
    t_build_start = time.perf_counter()
    _spawn_phase("build", actual_work_dir)
    t_build_sec = time.perf_counter() - t_build_start
    print(f"  Exported base model and AOT recompiled in {t_build_sec:.2f}s.")

    print(
        "\n[Step 2/3] Measuring AOT Recompiled executable in isolated"
        " subprocess..."
    )
    aot_metrics, _ = _spawn_phase_repeated(
        "aot", actual_work_dir, _REPEATS.value
    )
    print(
        "  AOT arm completed (Cold-Start Call 0:"
        f" {aot_metrics['cold_start_ms']:.2f} ms)."
    )

    print(
        "\n[Step 3/3] Measuring No-AOT (JIT baseline) in isolated"
        " subprocess..."
    )
    jit_metrics, _ = _spawn_phase_repeated(
        "jit", actual_work_dir, _REPEATS.value
    )
    print(
        "  JIT arm completed (Cold-Start Call 0:"
        f" {jit_metrics['cold_start_ms']:.2f} ms)."
    )

  # Verify golden numerical parity
  aot_output = np.array(aot_metrics["first_output"])
  jit_output = np.array(jit_metrics["first_output"])
  np.testing.assert_allclose(aot_output, jit_output, rtol=1e-5, atol=1e-5)

  def _delta_pct(aot: float, jit: float) -> float:
    return ((aot - jit) / jit) * 100.0 if jit > 0 else 0.0

  load_delta_ms = aot_metrics["load_ms"] - jit_metrics["load_ms"]
  cold_delta_ms = aot_metrics["cold_start_ms"] - jit_metrics["cold_start_ms"]
  peak_delta_mb = (
      aot_metrics["process_peak_mb"] - jit_metrics["process_peak_mb"]
  )
  peak_delta_pct = _delta_pct(
      aot_metrics["process_peak_mb"], jit_metrics["process_peak_mb"]
  )
  first_call_diff_mb = (
      aot_metrics["first_call_delta_mb"] - jit_metrics["first_call_delta_mb"]
  )
  attributed_jit_spike_mb = max(
      0.0,
      jit_metrics["first_call_delta_mb"] - aot_metrics["first_call_delta_mb"],
  )

  # Regression gates
  if aot_metrics["cold_start_ms"] >= jit_metrics["cold_start_ms"]:
    raise AssertionError(
        "AOT cold-start did not beat JIT: "
        f"AOT {aot_metrics['cold_start_ms']:.1f} ms vs "
        f"JIT {jit_metrics['cold_start_ms']:.1f} ms (median of "
        f"{_REPEATS.value}). AOT executable may not be selected at load."
    )
  if attributed_jit_spike_mb <= 0.0:
    raise AssertionError(
        "No XLA compiler memory spike attributed to the JIT arm. Either "
        "/proc/self/clear_refs reset failed or the JIT arm did not compile."
    )
  for arm, m in (("AOT", aot_metrics), ("JIT", jit_metrics)):
    if m["second_call_ms"] >= _CALL1_CACHE_BOUND_MS:
      raise AssertionError(
          f"{arm} Call 1 took {m['second_call_ms']:.3f} ms, above the "
          f"{_CALL1_CACHE_BOUND_MS} ms kernel-cache bound. Device modules "
          "were not cached after Call 0."
      )
  if not (aot_metrics["clear_refs_ok"] and jit_metrics["clear_refs_ok"]):
    raise AssertionError(
        "VmHWM reset via /proc/self/clear_refs failed; memory deltas in this "
        "run are not trustworthy. Re-run with CAP_SYS_PTRACE."
    )

  print("\n" + "=" * 80)
  print(" EMPIRICAL VERIFICATION RESULTS")
  print("=" * 80)
  print(
      f"{'Metric':<30} | {'No-AOT (JIT)':<18} | "
      f"{'AOT (Recompiled)':<18} | {'Delta (AOT - JIT)'}"
  )
  print("-" * 80)
  print(
      f"{'Model Load, Warm Driver (ms)':<30} |"
      f" {jit_metrics['load_ms']:<18.2f} |"
      f" {aot_metrics['load_ms']:<18.2f} |"
      f" {load_delta_ms:+.2f} ms"
  )
  print(
      f"{'Cold-Start Call 0 (ms, median)':<30} |"
      f" {jit_metrics['cold_start_ms']:<18.2f} |"
      f" {aot_metrics['cold_start_ms']:<18.2f} |"
      f" {cold_delta_ms:+.2f} ms"
  )
  print(
      f"{'  ^ spread [min, max]':<30} |"
      f" [{jit_metrics['cold_start_ms_min']:.1f}, "
      f"{jit_metrics['cold_start_ms_max']:.1f}]".ljust(19) + "|"
      f" [{aot_metrics['cold_start_ms_min']:.1f}, "
      f"{aot_metrics['cold_start_ms_max']:.1f}]".ljust(19) + "|"
      f" n={_REPEATS.value} samples"
  )
  print(
      f"{'Call 1 kernel cache (<5 ms)':<30} |"
      f" {jit_metrics['second_call_ms']:<18.3f} |"
      f" {aot_metrics['second_call_ms']:<18.3f} |"
      f" PASS (both cached)"
  )
  print(
      f"{'Peak VmHWM (MB)':<30} |"
      f" {jit_metrics['process_peak_mb']:<18.1f} |"
      f" {aot_metrics['process_peak_mb']:<18.1f} |"
      f" {peak_delta_mb:+.1f} MB ({peak_delta_pct:+.1f}%)"
  )
  print(
      f"{'First-Call Host Delta (MB)':<30} |"
      f" {jit_metrics['first_call_delta_mb']:<18.1f} |"
      f" {aot_metrics['first_call_delta_mb']:<18.1f} |"
      f" {first_call_diff_mb:+.1f} MB"
  )
  print(
      f"{'Attributed Compiler Spike (MB)':<30} |"
      f" {attributed_jit_spike_mb:<18.1f} |"
      f" {'0.0 (baseline)':<18} |"
      f" {-attributed_jit_spike_mb:+.1f} MB"
  )
  print("=" * 80)
  print(
      f"Numerical parity verified across arms (rtol=1e-5, atol=1e-5) over"
      f" {len(aot_output)} output elements."
  )


if __name__ == "__main__":
  app.run(main)
