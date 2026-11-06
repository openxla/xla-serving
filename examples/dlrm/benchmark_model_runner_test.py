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

"""Tests for ModelRunner throughput accounting and shared input helpers."""

import argparse
import multiprocessing
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from examples.dlrm import benchmark_model_runner


class FakeRunner:

  def __init__(self, model_path):
    self.calls = 0

  def signature(self, name):
    assert name == 'serving_default'
    return self

  def run_with_dict_return(self, **inputs):
    self.calls += 1
    assert len(inputs) == 27
    return {'predictions': inputs['dense_features'].mean(axis=1)}


def fake_worker(connection, args):
  module = types.SimpleNamespace(
      model_runner=types.SimpleNamespace(ModelRunner=FakeRunner)
  )
  with mock.patch.dict(sys.modules, {'xla_saved_model_runner': module}):
    benchmark_model_runner._worker(connection, args)


def stalled_worker(connection, args):
  connection.recv()  # Wait for start without announcing readiness.


class BenchmarkTest(unittest.TestCase):

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.path = Path(directory.name) / 'inputs.npz'
    arrays = {str(i): np.zeros((2, 3, 2), dtype=np.int32) for i in range(26)}
    arrays['dense_features'] = np.arange(78, dtype=np.float32).reshape(2, 3, 13)
    self.expected = arrays['dense_features'].mean(axis=2)
    np.savez(self.path, **arrays, predictions=self.expected)
    self.args = argparse.Namespace(
        inputs=str(self.path),
        model_path='fake',
        concurrency=2,
        duration=1,
        warmup=2,
        timeout=10,
        signature_name='serving_default',
    )

  # Reject malformed, nonfinite, and numerically wrong predictions.
  def test_invalid_predictions_fail(self):
    expected = self.expected[0]
    for actual in (
        expected[:, None],
        expected.astype(np.float64),
        np.full_like(expected, np.nan),
        expected + 1,
    ):
      with self.subTest(actual=actual), self.assertRaises(
          (ValueError, AssertionError)
      ):
        benchmark_model_runner.check_predictions(actual, expected)

  def test_empty_input_archive_fails_before_worker_loop(self):
    with np.load(self.path) as archive:
      arrays = {name: archive[name][:0] for name in archive.files}
    np.savez(self.path, **arrays)
    with self.assertRaisesRegex(ValueError, 'at least one batch'):
      benchmark_model_runner.load_inputs(self.path)

  # Count completed rows over the shared start-to-final-completion interval.
  def test_counts_use_common_start_and_last_completion(self):
    result = benchmark_model_runner.summarize([(4, 3, 11), (5, 3, 12)], 10)
    self.assertEqual(result['completed_requests'], 9)
    self.assertEqual(result['completed_rows'], 27)
    self.assertEqual(result['batch_size'], 3)
    self.assertEqual(result['elapsed_seconds'], 2)
    self.assertEqual(result['rows_per_second'], 13.5)
    with self.assertRaisesRegex(ValueError, 'Every worker'):
      benchmark_model_runner.summarize([(0, 3, 11), (5, 3, 12)], 10)

  # Validate before timing; count completed calls.
  def test_model_runner_validates_before_timing(self):
    runner = FakeRunner('fake')
    module = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(ModelRunner=lambda **_: runner)
    )
    connection = mock.Mock()
    connection.recv.return_value = 100.0
    with (
        mock.patch.dict(sys.modules, {'xla_saved_model_runner': module}),
        mock.patch.object(
            benchmark_model_runner.time,
            'monotonic',
            side_effect=[100.1, 100.2, 101.1, 101.2],
        ),
        mock.patch.object(
            benchmark_model_runner,
            'check_predictions',
            wraps=benchmark_model_runner.check_predictions,
        ) as checks,
    ):
      benchmark_model_runner._worker(connection, self.args)
    self.assertEqual(runner.calls, 2 + self.args.warmup + 2)
    self.assertEqual(checks.call_count, 2 + self.args.warmup)
    self.assertEqual(
        connection.send.call_args_list,
        [mock.call(2), mock.call((2, 3, 101.2))],
    )
    connection.close.assert_called_once()

  # Two spawned owners must report valid row counts and exit cleanly.
  def test_two_spawned_model_runners(self):
    before = {child.pid for child in multiprocessing.active_children()}
    with mock.patch.object(benchmark_model_runner, '_worker', fake_worker):
      result = benchmark_model_runner.run_benchmark(self.args)
    self.assertEqual(result['concurrency'], 2)
    self.assertEqual(result['batch_size'], 3)
    self.assertEqual(result['num_input_batches'], 2)
    self.assertEqual(result['configuration'], vars(self.args))
    self.assertGreater(result['completed_requests'], 0)
    self.assertEqual(result['completed_rows'], result['completed_requests'] * 3)
    self.assertGreaterEqual(result['elapsed_seconds'], self.args.duration)
    self.assertEqual(
        {child.pid for child in multiprocessing.active_children()}, before
    )

  # A readiness timeout must terminate every spawned worker.
  def test_timeout_cleans_up_spawned_workers(self):
    before = {child.pid for child in multiprocessing.active_children()}
    self.args.timeout = 0.1
    with (
        mock.patch.object(benchmark_model_runner, '_worker', stalled_worker),
        self.assertRaises(TimeoutError),
    ):
      benchmark_model_runner.run_benchmark(self.args)
    self.assertEqual(
        {child.pid for child in multiprocessing.active_children()}, before
    )


if __name__ == '__main__':
  unittest.main()
