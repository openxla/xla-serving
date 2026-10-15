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

r"""Generates a checkpoint for testing.

Usage:
blaze run \
  third_party/neptune/runtime/data_loader/testdata:gen_checkpoint \
  -- \
  --enable_sharding=false

export XLA_FLAGS="--xla_force_host_platform_device_count=4"
blaze run \
  third_party/neptune/runtime/data_loader/testdata:gen_checkpoint \
  -- \
  --enable_sharding=true
"""

from collections.abc import Sequence
import logging
import os
import shutil

from absl import app
from absl import flags
import jax
from jax.experimental.array_serialization import serialization
import jax.numpy as jnp
import jax.sharding as jsharding
from jaxtyping import PyTree
import numpy as np
import orbax.checkpoint as ocp

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/ckpt',
    'The path to the output directory.',
)

_ENABLE_SHARDING = flags.DEFINE_bool(
    'enable_sharding',
    False,
    'Whether to enable sharding.',
)

_NUM_PARAMS = flags.DEFINE_integer(
    'num_params',
    2,
    'The number of parameters to store.',
)

_SERIALIZATION_METHOD = flags.DEFINE_enum(
    'serialization_method',
    'ocdbt',
    ['jax', 'ocdbt'],
    'The serialization method to use.',
)


def _create_params(num_params: int, sizes: Sequence[int]) -> PyTree:
  """Create parameters pytree and transfer to jax devices()[0].

  E.g.
  _create_params(2, [4, 8]) returns
  {
    'param_0': jnp.arange(0, 4, dtype=jnp.float32),
    'param_1': jnp.arange(4, 12, dtype=jnp.float32),
  }
  _create_params(3, [2, 4]) returns
  {
    'param_0': jnp.arange(0, 2, dtype=jnp.float32),
    'param_1': jnp.arange(2, 6, dtype=jnp.float32),
    'param_2': jnp.arange(6, 10, dtype=jnp.float32),
  }

  Args:
    num_params: The number of parameters to create.
    sizes: The size of each parameter. When the length of sizes is less than
      num_params, the last element of sizes will be used for the rest of
      parameters.

  Returns:
    Param pytree resides on device
  """
  params = {}
  start = 0
  for i in range(num_params):
    stop = start + sizes[i] if i < len(sizes) else sizes[-1]
    params.update({f'param_{i}': jnp.arange(start, stop, dtype=jnp.float32)})
    start = stop
  return params


def _shard_params(params: PyTree) -> PyTree:
  """Reshape each parameter to [2, -1] and shard it across {'x':2, 'y':2}."""
  mesh = jsharding.Mesh(np.array(jax.devices()).reshape(2, 2), ('x', 'y'))
  params_sharding = jsharding.NamedSharding(
      mesh, jsharding.PartitionSpec('x', 'y')
  )
  params = jax.tree_util.tree_map(lambda p: p.reshape(2, -1), params)
  return jax.device_put(params, params_sharding)


def _jax_serialization(output_dir: str) -> None:
  """Generates a checkpoint using jax serialization: each parameter is stored in a separate directory."""
  manager = serialization.GlobalAsyncCheckpointManager()
  params = _create_params(num_params=_NUM_PARAMS.value, sizes=[4, 8])
  if _ENABLE_SHARDING.value:
    logging.info('sharding params %s', params)
    params = _shard_params(params)
  else:
    params = jax.device_put(params, jax.devices()[0])

  params_flat, _ = jax.tree_util.tree_flatten(params)
  tspecs = [
      serialization.get_tensorstore_spec(f'{output_dir}/params/{i}')
      for i in range(len(params_flat))
  ]

  manager.serialize(params_flat, tspecs, on_commit_callback=lambda: None)
  manager.wait_until_finished()


def _ocdbt_serialization(output_dir: str) -> None:
  """Generates a checkpoint using ocdbt serialization: all parameters are stored in a single directory."""
  handler = ocp.StandardCheckpointHandler()
  checkpointer = ocp.Checkpointer(handler)
  params = _create_params(num_params=_NUM_PARAMS.value, sizes=[4, 8])
  if _ENABLE_SHARDING.value:
    logging.info('sharding params %s', params)
    params = _shard_params(params)
  else:
    params = jax.device_put(params, jax.devices()[0])
  checkpointer.save(output_dir, params)


def main(argv: Sequence[str]) -> None:
  del argv
  output_dir = os.path.abspath(_OUTPUT_DIR.value)
  os.makedirs(output_dir, exist_ok=True)
  shutil.rmtree(output_dir)

  logging.info('Generating checkpoint to %s', output_dir)

  if _SERIALIZATION_METHOD.value == 'jax':
    _jax_serialization(output_dir)
  elif _SERIALIZATION_METHOD.value == 'ocdbt':
    _ocdbt_serialization(output_dir)


if __name__ == '__main__':
  app.run(main)
