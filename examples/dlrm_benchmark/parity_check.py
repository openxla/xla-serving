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

import json
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

_SERVER_A = flags.DEFINE_string('server_a', 'localhost:8500', 'host:port.')
_SERVER_B = flags.DEFINE_string(
    'server_b', None, 'Optional second server; if unset use local reference.'
)
_WEIGHTS_DIR = flags.DEFINE_string(
    'weights_dir', None, 'Baseline weights (config + reference weights).'
)
_REFERENCE_DEVICE = flags.DEFINE_enum(
    'reference_device', 'cpu', ['cpu', 'cuda', 'tpu'], 'Local reference device.'
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', common.DEFAULT_MODEL_NAME, 'Model name sent to servers.'
)
_BATCH_SIZES = flags.DEFINE_list(
    'batch_sizes', ['1', '16', '128'], 'Request batch sizes to check.'
)
_ATOL = flags.DEFINE_float('atol', 1e-4, 'Max allowed |a-b| on predictions.')
_SEED = flags.DEFINE_integer('seed', 42, 'Seed for the parity batches.')
_TIMEOUT_S = flags.DEFINE_float('timeout_s', 120.0, 'Per-RPC deadline.')
_OUTPUT = flags.DEFINE_string('output', None, 'Optional JSON report path.')


def _stub(server: str):
  channel = common.make_channel(
      server,
      options=[('grpc.max_receive_message_length', 64 * 1024 * 1024)],
  )
  grpc.channel_ready_future(channel).result(timeout=60)
  return channel.unary_unary(
      common.PREDICT_METHOD,
      request_serializer=predict_pb2.PredictRequest.SerializeToString,
      response_deserializer=predict_pb2.PredictResponse.FromString,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if not _WEIGHTS_DIR.value:
    raise app.UsageError('--weights_dir is required')
  config = common.ModelConfig.load(_WEIGHTS_DIR.value)

  stub_a = _stub(_SERVER_A.value)
  if _SERVER_B.value:
    stub_b = _stub(_SERVER_B.value)
    name_b = _SERVER_B.value

    def predict_b(feats):
      return common.predictions_from_response(
          stub_b(
              common.make_predict_request(feats, _MODEL_NAME.value),
              timeout=_TIMEOUT_S.value,
          )
      )

  else:
    # The server under test owns the accelerator while this runs: make sure the
    # in-process reference only initialises the backend it was asked for.
    python_server.restrict_jax_platforms(_REFERENCE_DEVICE.value)
    model = python_server.DlrmBaseline(
        _WEIGHTS_DIR.value, device=_REFERENCE_DEVICE.value
    )
    name_b = f'local reference ({_REFERENCE_DEVICE.value})'
    predict_b = model.predict

  rng = np.random.default_rng(_SEED.value)
  report: dict[str, Any] = {
      'server_a': _SERVER_A.value,
      'server_b': name_b,
      'checks': [],
  }
  ok = True
  for b in [int(x) for x in _BATCH_SIZES.value]:
    feats = common.generate_features(config, b, rng)
    a = common.predictions_from_response(
        stub_a(
            common.make_predict_request(feats, _MODEL_NAME.value),
            timeout=_TIMEOUT_S.value,
        )
    )
    ref = predict_b(feats)
    if a.shape != ref.shape:
      ok = False
      logging.error('batch %d: shape mismatch %s vs %s', b, a.shape, ref.shape)
      report['checks'].append({'batch': b, 'ok': False, 'reason': 'shape'})
      continue
    diff = float(np.max(np.abs(a - ref)))
    finite = bool(np.all(np.isfinite(a)))
    passed = finite and diff <= _ATOL.value
    ok &= passed
    logging.info(
        'batch %4d: max|a-b| = %.3e  (atol %.1e)  %s',
        b,
        diff,
        _ATOL.value,
        'OK' if passed else 'FAIL',
    )
    report['checks'].append({
        'batch': b,
        'ok': passed,
        'max_abs_diff': diff,
        'finite': finite,
        'sample_a': a[:4].tolist(),
        'sample_b': np.asarray(ref)[:4].tolist(),
    })
  report['ok'] = ok
  if _OUTPUT.value:
    with open(_OUTPUT.value, 'w') as f:
      json.dump(report, f, indent=2)
  print(
      f"PARITY {'OK' if ok else 'FAILED'}: {_SERVER_A.value} vs {name_b}; "
      + ', '.join(
          f"B={c['batch']}:{c.get('max_abs_diff', float('nan')):.2e}"
          for c in report['checks']
      ),
      flush=True,
  )
  if not ok:
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
