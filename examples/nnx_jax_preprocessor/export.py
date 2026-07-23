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

"""Neptune export for an NNX recommender model with JAX preprocessor."""

import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from examples.nnx import model as nnx_model
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.processors import jax_processor
from neptune_model._src.processors import processor


def export_model(export_path: str):
  """Exports the NNX recommender model using Neptune."""

  rngs = nnx.Rngs(0)
  embedding_layer, model = nnx_model.build_model(
      rngs, separate_embedding_layer=True
  )

  # 1. Main model.
  model_graph, model_params, model_rest_state = nnx.split(
      model, nnx.filterlib.Not(nnx.RngState), ...
  )

  def apply_fn(params, inputs):
    inference_model = nnx.merge(
        model_graph, params, model_rest_state, copy=True
    )
    return inference_model(inputs)

  # 2. Preprocessor for the embedding layer.
  emb_graph, emb_params, emb_rest_state = nnx.split(
      embedding_layer, nnx.filterlib.Not(nnx.RngState), ...
  )

  def emb_apply_fn(params, inputs):
    inference_emb = nnx.merge(emb_graph, params, emb_rest_state, copy=True)
    return inference_emb(inputs)

  input_signature = {
      'feature_1': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
      'feature_2': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
      'feature_3': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
      'feature_4': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.int32),
  }

  # Simulate saving and restoring the main model params so that we get
  # model params in the same format as they would be after loading from a
  # checkpoint.
  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'ckpt')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, model_params)
    model_params = checkpointer.restore(ckpt_path)

  # Same as above, but for the embedding layer params.
  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'ckpt_emb')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, emb_params)
    emb_params = checkpointer.restore(ckpt_path)

  emb_jax_model = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=emb_apply_fn,
          input_polymorphic_shapes={
              'feature_1': 'b, _',
              'feature_2': 'b, _',
              'feature_3': 'b, _',
              'feature_4': 'b, _',
          },
          jax_export_kwargs={'platforms': ['cpu']},
      ),
      params=emb_params,
  )

  main_lib.save(
      export_path=export_path,
      models=jax_model.JaxModel(
          apply_fn_bundle=jax_model.ApplyFnBundle(
              apply_fn=apply_fn,
              input_polymorphic_shapes={
                  'feature_1': 'b, _, _',
                  'feature_2': 'b, _, _',
                  'feature_3': 'b, _, _',
                  'feature_4': 'b, _, _',
              },
              jax_export_kwargs={'platforms': ['cpu', 'tpu']},
          ),
          params=model_params,
      ),
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
              processing=processor.SequentialProcessing(
                  preprocessors=[
                      jax_processor.JaxProcessor(
                          name='embedding_layer',
                          model=emb_jax_model,
                          checkpoint_path='preprocessor_ckpt',
                          weights_name='preprocessor_weights',
                      )
                  ],
                  postprocessors=[],
              ),
          )
      ],
  )
