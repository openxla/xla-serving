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

"""Samples host and accelerator resource usage of a server process tree.

Started by `run_benchmark.sh` next to every load-generator run and stopped with
SIGINT/SIGTERM. The summary JSON is rewritten atomically on every sample so a
kill at any point leaves a valid file:

  {
    "pid": ..., "samples": N, "wall_s": ..., "cpu_s": ...,      # user+sys of
    "avg_cpu_cores": ...,                                        # the tree
    "peak_rss_mb": ..., "avg_rss_mb": ...,                       # see below
    "memory_metric": "rss" | "pss",
    "num_threads_max": ..., "num_processes_max": ...,
    "gpu": {"util_avg_pct": ..., "util_max_pct": ..., "mem_used_max_mb": ...}
  }

CPU and memory are summed over the root process and all its descendants, so a
multi-process server (the `py-mp*` arms: device owner + N frontends) is
charged for all of them. For such trees memory is the proportional set size
(PSS) rather than RSS, otherwise pages shared between the processes (the
shared-memory rings) would be counted once per process; the key names stay
`*_rss_mb` for report compatibility and `memory_metric` says which it was.

GPU numbers come from `nvidia-smi` (if present and `--accelerator=cuda`).
TPU utilisation is not sampled (there is no stable CLI for it); HBM usage can be
read from the server logs instead.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import psutil

_PID = flags.DEFINE_integer('pid', None, 'Root PID of the server.')
_OUTPUT = flags.DEFINE_string('output', None, 'Summary JSON path.')
_INTERVAL_S = flags.DEFINE_float('interval_s', 0.5, 'Sampling interval.')
_ACCELERATOR = flags.DEFINE_enum(
    'accelerator',
    'none',
    ['none', 'cpu', 'cuda', 'tpu'],
    'Accelerator to sample ("cpu" is an alias of "none" so the value of'
    ' run_benchmark.sh --platform can be forwarded verbatim).',
)
_GPU_INDEX = flags.DEFINE_integer('gpu_index', 0, 'nvidia-smi GPU index.')


def _tree(root: psutil.Process) -> list[psutil.Process]:
  try:
    return [root] + root.children(recursive=True)
  except psutil.NoSuchProcess:
    return []


def _cpu_and_memory(
    procs: Sequence[psutil.Process],
) -> tuple[float, int, int, str]:
  """Returns (cpu seconds, memory bytes, threads, memory metric name)."""
  cpu = 0.0
  mem = 0
  threads = 0
  metric = 'pss' if len(procs) > 1 else 'rss'
  for p in procs:
    try:
      with p.oneshot():
        t = p.cpu_times()
        cpu += t.user + t.system
        threads += p.num_threads()
        if metric == 'pss':
          try:
            mem += p.memory_full_info().pss
          except (psutil.AccessDenied, AttributeError):
            mem += p.memory_info().rss
            metric = 'rss'
        else:
          mem += p.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
      continue
  return cpu, mem, threads, metric


def _gpu_sample(index: int) -> tuple[float, float] | None:
  if not shutil.which('nvidia-smi'):
    return None
  try:
    out = subprocess.run(
        [
            'nvidia-smi',
            f'--id={index}',
            '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    util, mem = out.split(',')
    return float(util), float(mem)
  except (subprocess.SubprocessError, ValueError):
    return None


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if _PID.value is None or not _OUTPUT.value:
    raise app.UsageError('--pid and --output are required')
  root = psutil.Process(_PID.value)
  stop = False

  def _on_signal(signum, frame):  # pylint: disable=unused-argument
    nonlocal stop
    stop = True

  signal.signal(signal.SIGINT, _on_signal)
  signal.signal(signal.SIGTERM, _on_signal)

  procs = _tree(root)
  cpu0, _, _, _ = _cpu_and_memory(procs)
  t0 = time.monotonic()
  peak_mem = 0
  mem_sum = 0
  threads_max = 0
  procs_max = 0
  metric = 'rss'
  gpu_utils: list[float] = []
  gpu_mem_max = 0.0
  samples = 0
  summary: dict[str, Any] = {}

  while not stop:
    time.sleep(_INTERVAL_S.value)
    procs = _tree(root)
    if not procs:
      logging.warning('Process %d exited; stopping monitor', _PID.value)
      break
    cpu, mem, threads, metric = _cpu_and_memory(procs)
    samples += 1
    peak_mem = max(peak_mem, mem)
    mem_sum += mem
    threads_max = max(threads_max, threads)
    procs_max = max(procs_max, len(procs))
    if _ACCELERATOR.value == 'cuda':
      g = _gpu_sample(_GPU_INDEX.value)
      if g is not None:
        gpu_utils.append(g[0])
        gpu_mem_max = max(gpu_mem_max, g[1])
    wall = time.monotonic() - t0
    summary = {
        'pid': _PID.value,
        'samples': samples,
        'wall_s': wall,
        'cpu_s': cpu - cpu0,
        'avg_cpu_cores': (cpu - cpu0) / wall if wall > 0 else 0.0,
        'peak_rss_mb': peak_mem / 2**20,
        'avg_rss_mb': mem_sum / samples / 2**20,
        'memory_metric': metric,
        'num_threads_max': threads_max,
        'num_processes_max': procs_max,
        'server_cpu_affinity': (
            sorted(root.cpu_affinity())
            if hasattr(root, 'cpu_affinity')
            else None
        ),
    }
    if gpu_utils:
      summary['gpu'] = {
          'util_avg_pct': sum(gpu_utils) / len(gpu_utils),
          'util_max_pct': max(gpu_utils),
          'mem_used_max_mb': gpu_mem_max,
      }
    tmp = _OUTPUT.value + '.tmp'
    with open(tmp, 'w') as f:
      json.dump(summary, f, indent=2)
    os.replace(tmp, _OUTPUT.value)

  logging.info('Monitor summary: %s', json.dumps(summary))
  sys.exit(0)


if __name__ == '__main__':
  app.run(main)
