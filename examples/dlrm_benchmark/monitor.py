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

import dataclasses
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence

from absl import app
from absl import flags
from absl import logging
import psutil


@dataclasses.dataclass(frozen=True)
class _Flags:
  """Command-line flags; only registered when run as a binary."""

  pid: flags.FlagHolder[int | None]
  output: flags.FlagHolder[str | None]
  interval_s: flags.FlagHolder[float]
  accelerator: flags.FlagHolder[str]
  gpu_index: flags.FlagHolder[int]


def _define_flags() -> _Flags:
  """Registers flags lazily so tests can import `monitor` alongside other tools."""
  return _Flags(
      pid=flags.DEFINE_integer('pid', None, 'Root PID of the server.'),
      output=flags.DEFINE_string('output', None, 'Summary JSON path.'),
      interval_s=flags.DEFINE_float('interval_s', 0.5, 'Sampling interval.'),
      accelerator=flags.DEFINE_enum(
          'accelerator',
          'none',
          ['none', 'cpu', 'cuda', 'tpu'],
          'Accelerator to sample ("cpu" is an alias of "none" so the value of'
          ' run_benchmark.sh --platform can be forwarded verbatim).',
      ),
      gpu_index=flags.DEFINE_integer('gpu_index', 0, 'nvidia-smi GPU index.'),
  )


def _tree(root: psutil.Process) -> list[psutil.Process]:
  try:
    return [root] + root.children(recursive=True)
  except psutil.NoSuchProcess:
    return []


class _TreeSampler:
  """Accumulates CPU deltas per `(pid, create_time)` across a process tree.

  Tracking deltas per process incarnation ensures that child processes exiting
  mid-run never subtract their lifetime CPU from subsequent samples (which
  could otherwise undercount or make `cpu_s` negative).
  """

  def __init__(
      self,
      root: Any,
      tree_fn: Callable[[Any], Sequence[Any]] = _tree,
  ):
    self._root = root
    self._tree_fn = tree_fn
    self._prev_cpu: dict[tuple[int, float], float] = {}
    self.cpu_s = 0.0
    self.samples = 0
    self.peak_mem = 0
    self.mem_sum = 0
    self.threads_max = 0
    self.procs_max = 0
    self.metric = 'rss'
    for p in self._tree_fn(self._root):
      try:
        with p.oneshot():
          key = (p.pid, p.create_time())
          t = p.cpu_times()
          self._prev_cpu[key] = t.user + t.system
      except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

  def sample(self) -> bool:
    """Takes one sample of the process tree; returns False if root exited."""
    procs = self._tree_fn(self._root)
    if not procs:
      return False
    mem = 0
    threads = 0
    metric = 'pss' if len(procs) > 1 else 'rss'
    for p in procs:
      try:
        with p.oneshot():
          key = (p.pid, p.create_time())
          t = p.cpu_times()
          cur_cpu = t.user + t.system
          prev_cpu = self._prev_cpu.get(key, 0.0)
          if cur_cpu > prev_cpu:
            self.cpu_s += cur_cpu - prev_cpu
          self._prev_cpu[key] = cur_cpu
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
    self.samples += 1
    self.peak_mem = max(self.peak_mem, mem)
    self.mem_sum += mem
    self.threads_max = max(self.threads_max, threads)
    self.procs_max = max(self.procs_max, len(procs))
    self.metric = metric
    return True


def _gpu_sample(index: int) -> tuple[float, float] | None:
  """Queries `(gpu_util_pct, mem_used_mb)` for GPU `index` via `nvidia-smi`."""
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


def _cpu_affinity(proc: Any) -> list[int] | None:
  if not hasattr(proc, 'cpu_affinity'):
    return None
  try:
    return sorted(proc.cpu_affinity())
  except (psutil.Error, OSError):
    return None


def main(argv: Sequence[str], f: _Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if f.pid.value is None or not f.output.value:
    raise app.UsageError('--pid and --output are required')
  if f.interval_s.value <= 0:
    raise app.UsageError('--interval_s must be > 0')
  try:
    root = psutil.Process(f.pid.value)
  except psutil.NoSuchProcess:
    logging.warning('Process %d already exited; stopping monitor', f.pid.value)
    sys.exit(0)
  out_path = os.path.abspath(f.output.value)
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  stop_event = threading.Event()

  def _on_signal(signum, frame):  # pylint: disable=unused-argument
    stop_event.set()

  signal.signal(signal.SIGINT, _on_signal)
  signal.signal(signal.SIGTERM, _on_signal)

  sampler = _TreeSampler(root)
  t0 = time.monotonic()
  gpu_utils: list[float] = []
  gpu_mem_max = 0.0
  summary: dict[str, Any] = {}

  while not stop_event.is_set():
    stop_event.wait(f.interval_s.value)
    if stop_event.is_set() and sampler.samples > 0:
      break
    if not sampler.sample():
      logging.warning('Process %d exited; stopping monitor', f.pid.value)
      break
    if f.accelerator.value == 'cuda':
      g = _gpu_sample(f.gpu_index.value)
      if g is not None:
        gpu_utils.append(g[0])
        gpu_mem_max = max(gpu_mem_max, g[1])
    wall = time.monotonic() - t0
    summary = {
        'pid': f.pid.value,
        'samples': sampler.samples,
        'wall_s': wall,
        'cpu_s': sampler.cpu_s,
        'avg_cpu_cores': sampler.cpu_s / wall if wall > 0 else 0.0,
        'peak_rss_mb': sampler.peak_mem / 2**20,
        'avg_rss_mb': sampler.mem_sum / sampler.samples / 2**20,
        'memory_metric': sampler.metric,
        'num_threads_max': sampler.threads_max,
        'num_processes_max': sampler.procs_max,
        'server_cpu_affinity': _cpu_affinity(root),
    }
    if gpu_utils:
      summary['gpu'] = {
          'util_avg_pct': sum(gpu_utils) / len(gpu_utils),
          'util_max_pct': max(gpu_utils),
          'mem_used_max_mb': gpu_mem_max,
      }
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as out_f:
      json.dump(summary, out_f, indent=2)
    os.replace(tmp, out_path)

  logging.info('Monitor summary: %s', json.dumps(summary))
  sys.exit(0)


if __name__ == '__main__':
  app.run(functools.partial(main, f=_define_flags()))
