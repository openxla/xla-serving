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

"""Tests for loadgen, parity_check, monitor, and run_benchmark.sh."""

from __future__ import annotations

from concurrent import futures
import contextlib
import dataclasses
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
import grpc
import numpy as np

from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import export_benchmark_model as export
from examples.dlrm_benchmark import loadgen
from examples.dlrm_benchmark import parity_check
from examples.dlrm_benchmark import python_server
from xla_saved_model.protos.grpc import predict_pb2

try:
  from examples.dlrm_benchmark import monitor  # pylint: disable=g-import-not-at-top
except ImportError:
  monitor = None

FLAGS = flags.FLAGS

_BATCH_SIZES = (1, 4, 16)


def _test_srcdir() -> str:
  return (
      FLAGS.test_srcdir
      if FLAGS.is_parsed()
      else os.environ.get('TEST_SRCDIR', '')
  )


def _loadgen_command(pysrc_dir: str) -> tuple[list[str], str | None]:
  pkg = loadgen.__package__
  srcdir = _test_srcdir()
  if srcdir and pkg:
    binary = os.path.join(srcdir, *pkg.split('.'), 'loadgen')
    if os.path.isfile(binary) and os.access(binary, os.X_OK):
      return [binary], None
  return [sys.executable, '-m', f'{pkg}.loadgen'], pysrc_dir


def _run_benchmark_script() -> str:
  local = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'run_benchmark.sh'
  )
  if os.path.isfile(local):
    return local
  pkg = loadgen.__package__ or ''
  srcdir = _test_srcdir()
  if srcdir and pkg:
    candidate = os.path.join(srcdir, *pkg.split('.'), 'run_benchmark.sh')
    if os.path.isfile(candidate):
      return candidate
  raise FileNotFoundError('Cannot locate run_benchmark.sh')


class HarnessServerTestCase(absltest.TestCase):
  """Shared small exported weights + isolated `pysrc` root for subprocesses."""

  tmpdir: str
  weights_dir: str
  pysrc_dir: str
  config: common.ModelConfig

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tmpdir = tempfile.mkdtemp(prefix='dlrm_harness_test_')
    cls.addClassCleanup(shutil.rmtree, cls.tmpdir, True)
    cls.weights_dir = os.path.join(cls.tmpdir, 'weights')
    cls.pysrc_dir = os.path.join(cls.tmpdir, 'pysrc')
    os.makedirs(cls.pysrc_dir, exist_ok=True)
    examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.symlink(examples_dir, os.path.join(cls.pysrc_dir, 'examples'))
    python_server.restrict_jax_platforms('cpu')
    cls.config = export.export_model(
        embedding_size=8,
        max_vocab_size=20,
        batch_sizes=_BATCH_SIZES,
        baseline_weights_dir=cls.weights_dir,
    )

  def _start_server(self, *, warmup: bool = False, **kwargs) -> int:
    server, servicer, batcher = python_server.build_server(
        weights_dir=self.weights_dir,
        port=0,
        device='cpu',
        num_threads=2,
        warmup=warmup,
        **kwargs,
    )
    server.start()
    self.addCleanup(server.stop, 0)
    if batcher is not None:
      self.addCleanup(batcher.stop)
    return servicer.port


class LoadgenTest(HarnessServerTestCase):

  def test_closed_loop_and_open_loop_end_to_end(self):
    port = self._start_server(dynamic_batching=True, warmup=True)
    worker_cmd, worker_cwd = _loadgen_command(self.pysrc_dir)

    closed = loadgen.run_load(
        server=f'localhost:{port}',
        config=self.config,
        request_batch_size=4,
        concurrency=2,
        qps=0.0,
        duration_s=1.0,
        warmup_s=0.2,
        num_workers=2,
        channels_per_worker=2,
        num_distinct_requests=4,
        timeout_s=15.0,
        max_inflight=64,
        seed=7,
        model_name=common.DEFAULT_MODEL_NAME,
        labels={'arm': 'py-batch'},
        worker_cmd=worker_cmd,
        worker_cwd=worker_cwd,
    )
    self.assertEqual(closed['mode'], 'closed_loop')
    self.assertGreater(closed['requests'], 0)
    self.assertGreaterEqual(closed['requests_sent_total'], closed['requests'])
    self.assertEqual(closed['errors'], 0)
    self.assertEqual(closed['connections'], 2)  # 2 workers x min(2, 1 slot)
    self.assertAlmostEqual(closed['qps'], closed['requests'] / 1.0)
    self.assertTrue(np.isfinite(closed['latency_ms']['p99']))

    open_res = loadgen.run_load(
        server=f'localhost:{port}',
        config=self.config,
        request_batch_size=1,
        concurrency=0,
        qps=20.0,
        duration_s=0.5,
        warmup_s=0.1,
        num_workers=1,
        channels_per_worker=2,
        num_distinct_requests=4,
        timeout_s=15.0,
        max_inflight=64,
        seed=8,
        model_name=common.DEFAULT_MODEL_NAME,
        labels={'arm': 'py-batch'},
        worker_cmd=worker_cmd,
        worker_cwd=worker_cwd,
    )
    self.assertEqual(open_res['mode'], 'open_loop')
    self.assertGreater(open_res['requests'], 0)
    self.assertEqual(open_res['errors'], 0)
    self.assertEqual(open_res['connections'], 2)

  def test_measurement_window_excludes_warmup_and_drain_tail(self):
    synthetic_worker = loadgen.WorkerResult(
        worker=0,
        # 103.0 is warmup (<105), 105..115 are measured, 115.4 is post-deadline
        # drain (>115) and has a huge tail latency that must not leak in.
        completed_at=[103.0, 105.0, 110.0, 115.0, 115.4],
        latencies_s=[0.01, 0.02, 0.04, 0.06, 0.90],
        error_at=[104.0, 108.0, 115.2],
        errors=3,
        overload_drops=3,
        overload_drops_total=7,
        sent=10,
        cpu_s=0.05,
        error_samples=['sample_err'],
        connections=4,
        error_codes=['CANCELLED', 'UNAVAILABLE', 'DEADLINE_EXCEEDED'],
    )
    with mock.patch.object(
        loadgen,
        '_run_workers',
        return_value=(100.0, 105.0, 115.0, 116.0, [synthetic_worker]),
    ):
      res = loadgen.run_load(
          server='localhost:8500',
          config=self.config,
          request_batch_size=16,
          concurrency=0,
          qps=50.0,
          duration_s=10.0,
          warmup_s=5.0,
          num_workers=1,
          num_distinct_requests=4,
          timeout_s=30.0,
          max_inflight=64,
          seed=1,
          model_name=common.DEFAULT_MODEL_NAME,
          labels={'arm': 'xse-batch'},
      )
    self.assertEqual(res['requests'], 3)
    self.assertAlmostEqual(res['qps'], 0.3)
    self.assertAlmostEqual(res['examples_per_s'], 4.8)
    self.assertAlmostEqual(res['latency_ms']['max'], 60.0)
    self.assertEqual(res['errors'], 1)
    self.assertEqual(res['errors_total'], 3)
    self.assertEqual(res['error_codes'], {'UNAVAILABLE': 1})
    self.assertEqual(res['overload_drops'], 3)
    self.assertEqual(res['overload_drops_total'], 7)
    # 0.05 CPU-s across 10 sent requests = 5.0 CPU-s per 1,000 requests.
    self.assertAlmostEqual(res['client_cpu_per_1k_requests_s'], 5.0)

  def test_invalid_parameters_rejected(self):
    base: dict[str, Any] = dict(
        server='localhost:8500',
        config=self.config,
        request_batch_size=1,
        concurrency=1,
        qps=0.0,
        duration_s=1.0,
        warmup_s=0.0,
        num_workers=1,
        num_distinct_requests=1,
        timeout_s=5.0,
        max_inflight=8,
        seed=0,
        model_name=common.DEFAULT_MODEL_NAME,
        labels={},
    )
    with self.assertRaisesRegex(ValueError, 'duration_s'):
      loadgen.run_load(**(base | {'duration_s': 0.0}))
    with self.assertRaisesRegex(ValueError, 'warmup_s'):
      loadgen.run_load(**(base | {'warmup_s': -1.0}))
    with self.assertRaisesRegex(ValueError, 'num_workers'):
      loadgen.run_load(**(base | {'num_workers': 0}))
    with self.assertRaisesRegex(ValueError, 'num_distinct_requests'):
      loadgen.run_load(**(base | {'num_distinct_requests': 0}))
    with self.assertRaisesRegex(ValueError, 'timeout_s'):
      loadgen.run_load(**(base | {'timeout_s': 0.0}))
    with self.assertRaisesRegex(ValueError, 'concurrency'):
      loadgen.run_load(**(base | {'concurrency': 2, 'qps': 10.0}))
    with self.assertRaisesRegex(ValueError, 'concurrency and qps'):
      loadgen.run_load(**(base | {'concurrency': -1, 'qps': 10.0}))
    with self.assertRaisesRegex(ValueError, 'request_batch_size'):
      loadgen.run_load(**(base | {'request_batch_size': 0}))
    with self.assertRaisesRegex(ValueError, 'request_batch_size'):
      loadgen.run_load(**(base | {'request_batch_size': 32}))


class ParityCheckTest(HarnessServerTestCase):

  def _start_custom_server(self, fn) -> int:
    def handle(request: predict_pb2.PredictRequest, context):
      del context
      feats = common.parse_predict_request(request)
      batch = common.validate_features(self.config, feats)
      return common.make_predict_response(
          {common.OUTPUT_NAME: fn(batch)}, request
      )

    service, method = common.PREDICT_METHOD.strip('/').rsplit('/', 1)
    handler = grpc.method_handlers_generic_handler(
        service,
        {
            method: grpc.unary_unary_rpc_method_handler(
                handle,
                request_deserializer=predict_pb2.PredictRequest.FromString,
                response_serializer=predict_pb2.PredictResponse.SerializeToString,
            )
        },
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2), handlers=(handler,)
    )
    port = common.add_local_port(server, 0)
    server.start()
    self.addCleanup(server.stop, 0)
    return port

  def test_parity_pass_and_failure_detection(self):
    port_a = self._start_server(dynamic_batching=False)
    port_b = self._start_server(dynamic_batching=True)
    out_json = os.path.join(
        self.create_tempdir().full_path, 'nested', 'dir', 'parity.json'
    )

    rep_local = parity_check.check_parity(
        server_a=f'localhost:{port_a}',
        weights_dir=self.weights_dir,
        batch_sizes=(1, 4),
        output_path=out_json,
    )
    self.assertTrue(rep_local['ok'])
    self.assertTrue(os.path.isfile(out_json))
    with open(out_json) as f:
      self.assertTrue(json.load(f)['ok'])

    rep_pair = parity_check.check_parity(
        server_a=f'localhost:{port_a}',
        server_b=f'localhost:{port_b}',
        weights_dir=self.weights_dir,
        batch_sizes=(1, 4, 16),
        output_path=out_json,
    )
    self.assertTrue(rep_pair['ok'])

    wrong_port = self._start_custom_server(
        lambda b: np.full((b,), 99.0, dtype=np.float32)
    )
    rep_wrong = parity_check.check_parity(
        server_a=f'localhost:{port_a}',
        server_b=f'localhost:{wrong_port}',
        weights_dir=self.weights_dir,
        batch_sizes=(1, 4),
    )
    self.assertFalse(rep_wrong['ok'])

    nan_port = self._start_custom_server(
        lambda b: np.full((b,), np.nan, dtype=np.float32)
    )
    rep_nan = parity_check.check_parity(
        server_a=f'localhost:{port_a}',
        server_b=f'localhost:{nan_port}',
        weights_dir=self.weights_dir,
        batch_sizes=(1,),
    )
    self.assertFalse(rep_nan['ok'])
    self.assertFalse(rep_nan['checks'][0]['finite'])

    with self.assertRaisesRegex(ValueError, 'batch_sizes'):
      parity_check.check_parity(
          server_a=f'localhost:{port_a}',
          weights_dir=self.weights_dir,
          batch_sizes=(),
      )
    with self.assertRaisesRegex(ValueError, 'atol'):
      parity_check.check_parity(
          server_a=f'localhost:{port_a}',
          weights_dir=self.weights_dir,
          atol=-1.0,
      )

  def test_unreachable_server_times_out_and_closes_channel(self):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.bind(('127.0.0.1', 0))
      closed_port = s.getsockname()[1]
    stubs: list[Any] = []
    real_make_channel = common.make_channel

    def capture_channel(*args, **kwargs):
      ch = real_make_channel(*args, **kwargs)
      self.addCleanup(ch.close)
      stubs.append(ch.unary_unary(common.PREDICT_METHOD))
      return ch

    with mock.patch.object(
        common, 'make_channel', side_effect=capture_channel
    ):
      with self.assertRaises(grpc.FutureTimeoutError):
        parity_check.check_parity(
            server_a=f'127.0.0.1:{closed_port}',
            weights_dir=self.weights_dir,
            batch_sizes=(1,),
            timeout_s=0.05,
        )
    self.assertLen(stubs, 1)
    with self.assertRaisesRegex(ValueError, 'closed'):
      stubs[0](b'', timeout=0.05)


@dataclasses.dataclass
class _FakeTimes:
  user: float
  system: float


@dataclasses.dataclass
class _FakeMem:
  rss: int
  pss: int


class _FakeProc:

  def __init__(
      self,
      pid: int,
      created: float,
      user: float,
      system: float = 0.0,
      threads: int = 2,
      rss: int = 10 * 2**20,
      pss: int = 6 * 2**20,
  ):
    self.pid = pid
    self._created = created
    self.user = user
    self.system = system
    self._threads = threads
    self._mem = _FakeMem(rss=rss, pss=pss)

  def oneshot(self):
    return contextlib.nullcontext()

  def create_time(self) -> float:
    return self._created

  def cpu_times(self) -> _FakeTimes:
    return _FakeTimes(self.user, self.system)

  def num_threads(self) -> int:
    return self._threads

  def memory_info(self) -> _FakeMem:
    return self._mem

  def memory_full_info(self) -> _FakeMem:
    return self._mem


@absltest.skipIf(
    monitor is None and not os.environ.get('TEST_SRCDIR'),
    'psutil is not installed',
)
class MonitorTest(absltest.TestCase):

  def test_live_process_sample(self):
    assert monitor is not None
    sampler = monitor._TreeSampler(monitor.psutil.Process(os.getpid()))
    self.assertTrue(sampler.sample())
    self.assertEqual(sampler.samples, 1)
    self.assertGreaterEqual(sampler.cpu_s, 0.0)
    self.assertGreater(sampler.peak_mem, 0)
    self.assertEqual(sampler.mem_sum, sampler.peak_mem)
    self.assertGreaterEqual(sampler.threads_max, 1)

  def test_exiting_children_never_subtract_accumulated_cpu(self):
    assert monitor is not None
    root = _FakeProc(pid=100, created=10.0, user=1.0, system=0.5)
    child1 = _FakeProc(pid=101, created=11.0, user=2.0, system=0.0)
    current = [root, child1]
    sampler = monitor._TreeSampler(root, tree_fn=lambda _: list(current))

    # Sample 1: root +0.5s, child1 +2.5s, newly spawned child2 +1.0s => 4.0s.
    root.user = 1.5
    child1.user = 4.0
    child1.system = 0.5
    child2 = _FakeProc(pid=102, created=12.0, user=0.75, system=0.25)
    current = [root, child1, child2]
    self.assertTrue(sampler.sample())
    self.assertAlmostEqual(sampler.cpu_s, 4.0)
    self.assertEqual(sampler.metric, 'pss')
    self.assertEqual(sampler.procs_max, 3)
    self.assertEqual(sampler.peak_mem, 18 * 2**20)
    self.assertEqual(sampler.mem_sum, 18 * 2**20)
    self.assertEqual(sampler.threads_max, 6)

    # Sample 2: both children exit; root adds +0.25s => 4.25s (not negative).
    root.user = 1.75
    current = [root]
    self.assertTrue(sampler.sample())
    self.assertAlmostEqual(sampler.cpu_s, 4.25)
    self.assertEqual(sampler.metric, 'rss')
    self.assertEqual(sampler.procs_max, 3)
    self.assertEqual(sampler.peak_mem, 18 * 2**20)
    self.assertEqual(sampler.mem_sum, 28 * 2**20)
    self.assertEqual(sampler.threads_max, 6)

    # Sample 3: OS reuses PID 101 with a new create_time (+0.3s) => 4.55s.
    reused = _FakeProc(pid=101, created=25.0, user=0.3)
    current = [root, reused]
    self.assertTrue(sampler.sample())
    self.assertAlmostEqual(sampler.cpu_s, 4.55)
    self.assertEqual(sampler.mem_sum, 40 * 2**20)


class RunBenchmarkDryRunTest(absltest.TestCase):

  def _dry_run(
      self, *args: str, extra_env: dict[str, str] | None = None
  ) -> tuple[subprocess.CompletedProcess[str], str]:
    work_dir = os.path.join(self.create_tempdir().full_path, 'work')
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith('TPU_') and k != 'LIBTPU_INIT_ARGS'
    }
    if extra_env:
      env.update(extra_env)
    proc = subprocess.run(
        [
            'bash',
            _run_benchmark_script(),
            '--dry_run',
            f'--work_dir={work_dir}',
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    self.assertEqual(
        proc.returncode,
        0,
        f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}',
    )
    with open(os.path.join(work_dir, 'results', 'env.txt')) as f:
      env_txt = f.read()
    return proc, env_txt

  def test_cpu_dry_run_matches_jce_slots_to_num_threads(self):
    proc, env_txt = self._dry_run(
        '--platform', 'cpu', '--num_threads', '2', '--num_processes', '2'
    )
    self.assertIn('jce_max_cpu_slots: 2', env_txt)
    self.assertIn('--jce_max_cpu_slots=2', proc.stderr)
    self.assertIn('--thread_pool_num_threads=2', proc.stderr)

  def test_tpu_dry_run_derives_single_host_topology_when_metadata_missing(self):
    proc, env_txt = self._dry_run(
        '--platform',
        'tpu',
        '--xse_image',
        'us-docker.pkg.dev/xla-serving-engine/xla-serving/tpu-jax-only:nightly',
        '--server_cpus',
        '0-7',
        '--client_cpus',
        '8-23',
        '--num_threads',
        '8',
        extra_env={'DLRM_BENCHMARK_FAKE_METADATA': '404:ct6e-standard-1t'},
    )
    self.assertIn('tpu_env_source: derived(ct6e-standard-1t)', env_txt)
    self.assertIn('jce_max_cpu_slots: 8', env_txt)
    self.assertIn('--jce_max_cpu_slots=8', proc.stderr)
    self.assertIn('--cpuset-cpus 0-7', proc.stderr)
    self.assertIn('TPU_ACCELERATOR_TYPE=v6e-1', proc.stderr)
    self.assertIn('TPU_CHIPS_PER_HOST_BOUNDS=1,1,1', proc.stderr)

  def test_cuda_dry_run_configures_gpu_container_flags(self):
    proc, _ = self._dry_run(
        '--platform',
        'cuda',
        '--xse_image',
        'us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda12:nightly',
    )
    self.assertIn('--gpus all', proc.stderr)
    self.assertIn('PJRT_DEVICE=cuda', proc.stderr)

  def test_invalid_cli_flags_rejected(self):
    script = _run_benchmark_script()
    for bad_args, expected in (
        (['--platform', 'cpu', '--arms='], '--arms must not be empty'),
        (['--platform', 'cpu', '--num_threads', '0'], 'positive integer'),
        (['--platform', 'invalid'], '--platform must be'),
    ):
      proc = subprocess.run(
          ['bash', script, '--dry_run', *bad_args],
          capture_output=True,
          text=True,
          timeout=30,
          check=False,
      )
      self.assertNotEqual(proc.returncode, 0, bad_args)
      self.assertIn(expected, proc.stderr)


if __name__ == '__main__':
  absltest.main()
