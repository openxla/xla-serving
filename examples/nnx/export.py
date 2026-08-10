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

"""Neptune export for an NNX recommender model."""

import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp

from examples.nnx import model as nnx_model
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config


def export_model(export_path: str, dtype: jnp.dtype = jnp.float32):
  """Exports the NNX recommender model using Neptune."""

  rngs = nnx.Rngs(0)
  _, model = nnx_model.build_model(rngs)

  if dtype != jnp.float32:
    for _, module in nnx.iter_modules(model):
      if isinstance(module, (nnx.Linear, nnx.LayerNorm, nnx.Embed)):
        module.dtype = dtype

  model_graph, model_params, model_rest_state = nnx.split(
      model, nnx.filterlib.Not(nnx.RngState), ...
  )

  if dtype != jnp.float32:
    model_params = jax.tree_util.tree_map(
        lambda x: x.astype(dtype)
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        model_params,
    )

  def apply_fn(params, inputs):
    inference_model = nnx.merge(
        model_graph, params, model_rest_state, copy=True
    )
    return inference_model(inputs)

  input_signature = {
      'feature_1': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
      'feature_2': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
      'feature_3': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
      'feature_4': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
  }

  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'ckpt')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, model_params)
    ckpt_params = checkpointer.restore(ckpt_path)

  main_lib.save(
      jax_model.JaxModel(
          apply_fn_bundle=jax_model.ApplyFnBundle(
              apply_fn=apply_fn,
              input_polymorphic_shapes={
                  'feature_1': 'b, _',
                  'feature_2': 'b, _',
                  'feature_3': 'b, _',
                  'feature_4': 'b, _',
              },
              jax_export_kwargs={'platforms': ['cpu', 'tpu']},
          ),
          params=ckpt_params,
      ),
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
          )
      ],
  )
