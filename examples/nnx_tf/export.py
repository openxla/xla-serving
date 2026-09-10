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
import orbax.checkpoint as ocp
import tensorflow as tf

from examples.nnx_tf import model as nnx_tf_model
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.export import tensorflow_model


@tf.function
def simple_postprocessor(inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
  """A simple TF postprocessor that adds 10.0 to the predictions."""
  pred = inputs['pred']
  return {'pred': pred + 10.0}


def export_model(export_path: str):
  """Exports the NNX recommender model using Neptune."""

  rngs = nnx.Rngs(0)
  embedding_layer, model = nnx_tf_model.build_model(rngs)

  model_graph, model_params, model_rest_state = nnx.split(
      model, nnx.filterlib.Not(nnx.RngState), ...
  )

  def apply_fn(params, inputs):
    inference_model = nnx.merge(
        model_graph, params, model_rest_state, copy=True
    )
    return inference_model(inputs)

  input_signature = {
      'feature_1': tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
      'feature_2': tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
      'feature_3': tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
      'feature_4': tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
  }

  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'ckpt')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, model_params)
    model_params = checkpointer.restore(ckpt_path)

  tf_proc_model = tensorflow_model.TensorflowModel(
      name='tf_model',
      tf_functions={
          'preprocessing': embedding_layer,
          'postprocessing': simple_postprocessor,
      },
      trackable_resources=[embedding_layer],
  )
  jm = jax_model.JaxModel(
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
      name='jax_model',
  )

  main_lib.save(
      [tf_proc_model, jm],
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
              orchestration=serving_config.Orchestration(
                  model_calls=[
                      serving_config.ModelCall(
                          model_name='tf_model',
                          method_key='preprocessing',
                          role=orchestration_pb2.ROLE_PREPROCESSOR,
                      ),
                      serving_config.ModelCall(
                          model_name='jax_model',
                          method_key='serving_default',
                          role=orchestration_pb2.ROLE_MODEL,
                      ),
                      serving_config.ModelCall(
                          model_name='tf_model',
                          method_key='postprocessing',
                          role=orchestration_pb2.ROLE_POSTPROCESSOR,
                      ),
                  ]
              ),
          )
      ],
  )
