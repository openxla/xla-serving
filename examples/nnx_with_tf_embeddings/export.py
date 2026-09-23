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

"""XLA SavedModel export for NNX recommender with TPUEmbeddingForServing."""

import os
import tempfile

from absl import flags
from flax import nnx
import orbax.checkpoint as ocp
import tensorflow as tf

from examples.nnx_with_tf_embeddings import model as model_lib
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.export import tensorflow_model

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'num_request_features', 2, 'Number of request features.'
)

flags.DEFINE_integer(
    'num_candidate_features', 2, 'Number of candidate features.'
)


def export_model(export_path: str):
  """Exports the NNX recommender model using XLA SavedModel."""

  rngs = nnx.Rngs(0)
  embedding_layer, model = model_lib.build_model(
      rngs, FLAGS.num_request_features, FLAGS.num_candidate_features
  )

  model_graph, model_params, model_rest_state = nnx.split(
      model, nnx.filterlib.Not(nnx.RngState), ...
  )

  def apply_fn(params, inputs):
    inference_model = nnx.merge(
        model_graph, params, model_rest_state, copy=True
    )
    return inference_model(inputs)
  input_signature = {
      f'request_feature_{i+1}': tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
      for i in range(FLAGS.num_request_features)
  }
  input_signature.update({
      f'candidate_feature_{i+1}': tf.TensorSpec(
          shape=(None, 1), dtype=tf.int32
      )
      for i in range(FLAGS.num_candidate_features)
  })
  input_polyshape = {
      f'request_feature_{i+1}': 'b, _, _'
      for i in range(FLAGS.num_request_features)
  }
  input_polyshape.update({
      f'candidate_feature_{i+1}': 'b, _, _'
      for i in range(FLAGS.num_candidate_features)
  })

  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'ckpt')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, model_params)
    model_params = checkpointer.restore(ckpt_path)

  pre_model = tensorflow_model.TensorflowModel(
      name='tf_model',
      tf_functions={'preprocessing': embedding_layer},
      trackable_resources=[embedding_layer],
  )
  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=apply_fn,
          input_polymorphic_shapes={
              **input_polyshape,
          },
          jax_export_kwargs={'platforms': ['cpu', 'cuda']},
      ),
      params=model_params,
      name='jax_model',
  )

  main_lib.save(
      [pre_model, jm],
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
                      ),
                      serving_config.ModelCall(
                          model_name='jax_model', method_key='serving_default'
                      ),
                  ]
              ),
          )
      ],
  )
