# Copyright 2026 Arm Ltd. and affiliates
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

"""Measure FP32 DCN v2 throughput with one ModelRunner per process."""

import argparse
import json
import math
import multiprocessing
import time

import numpy as np


def load_inputs(path):
  """Loads batches and FP32 references from an inputs.npz prepared at export."""
  with np.load(path, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files if name != 'predictions'}
    expected = archive['predictions']
  if len(expected) == 0:
    raise ValueError('Input archive must contain at least one batch')
  batches = [
      {name: value[i] for name, value in arrays.items()} for i in range(len(expected))
  ]
  return batches, expected


def check_predictions(actual, expected):
  """Checks a batch against its FP32 reference."""
  actual = np.asarray(actual)
  if (
      actual.shape != expected.shape
      or actual.dtype != np.float32
      or not np.isfinite(actual).all()
  ):
    raise ValueError('Prediction shape, dtype or finiteness check failed')
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=False)


def summarize(results, start):
  """Uses common start through the last completed ModelRunner call."""
  requests = sum(count for count, _, _ in results)
  rows = sum(count * batch_size for count, batch_size, _ in results)
  elapsed = max(end for _, _, end in results) - start
  if any(count <= 0 for count, _, _ in results) or elapsed <= 0:
    raise ValueError(
        'Every worker must complete timed requests; use a longer duration'
    )
  return {
      'concurrency': len(results),
      'batch_size': results[0][1],
      'completed_requests': requests,
      'completed_rows': rows,
      'elapsed_seconds': elapsed,
      'requests_per_second': requests / elapsed,
      'rows_per_second': rows / elapsed,
  }


def _worker(connection, args):
  try:
    # Spawn, not fork: each process owns its native runtime and Python GIL.
    from xla_saved_model_runner import model_runner

    batches, expected = load_inputs(args.inputs)
    runner = model_runner.ModelRunner(model_path=args.model_path)
    signature = runner.signature(args.signature_name)
    for i in range(len(batches) + args.warmup):
      item = i % len(batches)
      actual = signature.run_with_dict_return(**batches[item])['predictions']
      check_predictions(actual, expected[item])
    connection.send(len(batches))  # Ready after loading and untimed validation.
    start = connection.recv()
    completed = 0
    while time.monotonic() < start + args.duration:
      item = completed % len(batches)
      signature.run_with_dict_return(**batches[item])
      completed += 1
    connection.send((completed, expected.shape[1], time.monotonic()))
  finally:
    connection.close()


def _receive(connection, timeout):
  if not connection.poll(timeout):
    raise TimeoutError('ModelRunner worker timed out')
  return connection.recv()  # EOF also fails the run if a native worker exits.


def run_benchmark(args):
  if args.concurrency < 1 or args.warmup < 0:
    raise ValueError('Use positive concurrency and nonnegative warmup')
  if not all(
      math.isfinite(value) and value > 0
      for value in (args.duration, args.timeout)
  ):
    raise ValueError('Use finite positive duration and timeout')
  context = multiprocessing.get_context('spawn')
  workers = []
  try:
    for _ in range(args.concurrency):
      parent, child = context.Pipe()
      process = context.Process(target=_worker, args=(child, args))
      process.start()
      child.close()
      workers.append((process, parent))
    batch_counts = [
        _receive(connection, args.timeout) for _, connection in workers
    ]
    start = time.monotonic()
    for _, connection in workers:
      connection.send(start)
    results = [
        _receive(connection, args.duration + args.timeout)
        for _, connection in workers
    ]
    for process, _ in workers:
      process.join(args.timeout)
      if process.exitcode is None:
        raise TimeoutError('ModelRunner worker did not exit within timeout')
      if process.exitcode != 0:
        raise RuntimeError(f'ModelRunner worker exit code: {process.exitcode}')
    return {
        **summarize(results, start),
        'num_input_batches': batch_counts[0],
        'configuration': vars(args).copy(),
    }
  finally:
    # On failures, never leave a sibling waiting for its start signal or result.
    for process, connection in workers:
      connection.close()
      if process.is_alive():
        process.kill()
    for process, _ in workers:
      process.join()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model_path', required=True, help='Exported bundle path.')
  parser.add_argument('--inputs', required=True, help='Prepared inputs.npz.')
  parser.add_argument('--signature_name', default='serving_default')
  parser.add_argument('--concurrency', type=int, default=1)
  parser.add_argument('--duration', type=float, default=10, help='Timed seconds.')
  parser.add_argument(
      '--warmup', type=int, default=20, help='Extra untimed requests per worker.'
  )
  parser.add_argument(
      '--timeout', type=float, default=60, help='Worker timeout in seconds.'
  )
  print(json.dumps(run_benchmark(parser.parse_args()), indent=2))


if __name__ == '__main__':
  main()
