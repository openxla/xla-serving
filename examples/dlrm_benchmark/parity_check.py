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

r"""Asserts that two servers (or a server and the local reference) agree.

Run before any timing so the benchmark never compares two different functions.
Typical uses:

  # XSE vs. Python baseline, both running:
  parity_check --server_a=localhost:8500 --server_b=localhost:8501 \
      --weights_dir=/work/baseline_weights

  # A single server vs. the in-process numpy+JAX reference:
  parity_check --server_a=localhost:8500 --weights_dir=/work/baseline_weights \
      --reference_device=cpu

Exit code is non-zero if `max|a - b| > --atol` for any tested batch size.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import sys
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import grpc
import numpy as np

from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import python_server
from xla_saved_model.protos.grpc import predict_pb2


@dataclasses.dataclass(frozen=True)
class _Flags:
  """Command-line flags; only registered when run as a binary."""

  server_a: flags.FlagHolder[str]
  server_b: flags.FlagHolder[str | None]
  weights_dir: flags.FlagHolder[str | None]
  reference_device: flags.FlagHolder[str]
  model_name: flags.FlagHolder[str]
  batch_sizes: flags.FlagHolder[list[str]]
  atol: flags.FlagHolder[float]
  seed: flags.FlagHolder[int]
  timeout_s: flags.FlagHolder[float]
  output: flags.FlagHolder[str | None]


def _define_flags() -> _Flags:
  """Registers flags lazily so tests can import `parity_check`."""
  return _Flags(
      server_a=flags.DEFINE_string('server_a', 'localhost:8500', 'host:port.'),
      server_b=flags.DEFINE_string(
          'server_b',
          None,
          'Optional second server; if unset use local reference.',
      ),
      weights_dir=flags.DEFINE_string(
          'weights_dir', None, 'Baseline weights (config + reference weights).'
      ),
      reference_device=flags.DEFINE_enum(
          'reference_device',
          'cpu',
          ['cpu', 'cuda', 'tpu'],
          'Local reference device.',
      ),
      model_name=flags.DEFINE_string(
          'model_name',
          common.DEFAULT_MODEL_NAME,
          'Model name sent to servers.',
      ),
      batch_sizes=flags.DEFINE_list(
          'batch_sizes', ['1', '16', '128'], 'Request batch sizes to check.'
      ),
      atol=flags.DEFINE_float(
          'atol', 1e-4, 'Max allowed |a-b| on predictions.'
      ),
      seed=flags.DEFINE_integer('seed', 42, 'Seed for the parity batches.'),
      timeout_s=flags.DEFINE_float('timeout_s', 120.0, 'Per-RPC deadline.'),
      output=flags.DEFINE_string('output', None, 'Optional JSON report path.'),
  )


def _open_stub(
    server: str, timeout_s: float = 60.0
) -> tuple[grpc.Channel, Any]:
  """Opens a gRPC channel to `server` and returns `(channel, unary_stub)`."""
  channel = common.make_channel(
      server, options=python_server.grpc_server_options()
  )
  try:
    grpc.channel_ready_future(channel).result(timeout=timeout_s)
  except Exception:
    channel.close()
    raise
  stub = channel.unary_unary(
      common.PREDICT_METHOD,
      request_serializer=predict_pb2.PredictRequest.SerializeToString,
      response_deserializer=predict_pb2.PredictResponse.FromString,
  )
  return channel, stub


def check_parity(
    *,
    server_a: str,
    weights_dir: str,
    server_b: str | None = None,
    reference_device: str = 'cpu',
    model_name: str = common.DEFAULT_MODEL_NAME,
    batch_sizes: Sequence[int] = (1, 16, 128),
    atol: float = 1e-4,
    seed: int = 42,
    timeout_s: float = 120.0,
    output_path: str | None = None,
) -> dict[str, Any]:
  """Compares `server_a` vs. `server_b` (or local reference) across batches."""
  if not batch_sizes or any(int(b) < 1 for b in batch_sizes):
    raise ValueError(
        f'batch_sizes must be non-empty positive integers, got {batch_sizes}'
    )
  if atol < 0:
    raise ValueError(f'atol must be >= 0, got {atol}')
  if timeout_s <= 0:
    raise ValueError(f'timeout_s must be > 0, got {timeout_s}')
  config = common.ModelConfig.load(weights_dir)
  channels: list[grpc.Channel] = []
  try:
    ch_a, stub_a = _open_stub(server_a, timeout_s=timeout_s)
    channels.append(ch_a)
    if server_b:
      ch_b, stub_b = _open_stub(server_b, timeout_s=timeout_s)
      channels.append(ch_b)
      name_b = server_b

      def predict_b(feats):
        return common.predictions_from_response(
            stub_b(
                common.make_predict_request(feats, model_name),
                timeout=timeout_s,
            )
        )

    else:
      # The server under test owns the accelerator while this runs: make sure
      # the in-process reference only initialises the backend it was asked for.
      python_server.restrict_jax_platforms(reference_device)
      model = python_server.DlrmBaseline(weights_dir, device=reference_device)
      name_b = f'local reference ({reference_device})'
      predict_b = model.predict

    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        'server_a': server_a,
        'server_b': name_b,
        'checks': [],
    }
    ok = True
    for b in batch_sizes:
      feats = common.generate_features(config, int(b), rng)
      a = common.predictions_from_response(
          stub_a(
              common.make_predict_request(feats, model_name),
              timeout=timeout_s,
          )
      )
      ref = np.asarray(predict_b(feats))
      if a.shape != ref.shape:
        ok = False
        logging.error(
            'batch %d: shape mismatch %s vs %s', b, a.shape, ref.shape
        )
        report['checks'].append(
            {'batch': int(b), 'ok': False, 'reason': 'shape'}
        )
        continue
      diff = float(np.max(np.abs(a - ref)))
      finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(ref)))
      passed = finite and diff <= atol
      ok &= passed
      logging.info(
          'batch %4d: max|a-b| = %.3e  (atol %.1e)  %s',
          b,
          diff,
          atol,
          'OK' if passed else 'FAIL',
      )
      report['checks'].append({
          'batch': int(b),
          'ok': passed,
          'max_abs_diff': diff,
          'finite': finite,
          'sample_a': a[:4].tolist(),
          'sample_b': ref[:4].tolist(),
      })
    report['ok'] = ok
  finally:
    for ch in channels:
      ch.close()

  if output_path:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as out_f:
      json.dump(report, out_f, indent=2)
  return report


def main(argv: Sequence[str], f: _Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if not f.weights_dir.value:
    raise app.UsageError('--weights_dir is required')
  batch_sizes = [int(x) for x in f.batch_sizes.value if x.strip()]
  if not batch_sizes or any(b < 1 for b in batch_sizes):
    raise app.UsageError('--batch_sizes must be non-empty positive integers')
  if f.atol.value < 0:
    raise app.UsageError('--atol must be >= 0')
  if f.timeout_s.value <= 0:
    raise app.UsageError('--timeout_s must be > 0')
  report = check_parity(
      server_a=f.server_a.value,
      weights_dir=f.weights_dir.value,
      server_b=f.server_b.value,
      reference_device=f.reference_device.value,
      model_name=f.model_name.value,
      batch_sizes=batch_sizes,
      atol=f.atol.value,
      seed=f.seed.value,
      timeout_s=f.timeout_s.value,
      output_path=f.output.value,
  )
  ok = report['ok']
  print(
      f"PARITY {'OK' if ok else 'FAILED'}: {report['server_a']} vs"
      f" {report['server_b']}; "
      + ', '.join(
          f"B={c['batch']}:{c.get('max_abs_diff', float('nan')):.2e}"
          for c in report['checks']
      ),
      flush=True,
  )
  if not ok:
    sys.exit(1)


if __name__ == '__main__':
  app.run(functools.partial(main, f=_define_flags()))
