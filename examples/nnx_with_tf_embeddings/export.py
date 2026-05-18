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

"""Neptune export for NNX recommender model with TPUEmbeddingForServing layer."""

import os
import tempfile

from absl import app
from absl import flags
from flax import nnx
import orbax.checkpoint as ocp
import tensorflow as tf

from examples.nnx_with_tf_embeddings import model as model_lib
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.processors import processor
from neptune_model._src.processors import tf_processor

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'export_path', '/tmp/neptune_tf_embeddings', 'Path to export model.'
)
flags.DEFINE_integer(
    'num_request_features', 2, 'Number of request features.'
)

flags.DEFINE_integer(
    'num_candidate_features', 2, 'Number of candidate features.'
)


def export_model(export_path: str):
  """Exports the NNX recommender model using Neptune."""

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

  main_lib.save(
      export_path=export_path,
      model=jax_model.JaxModel(
          apply_fn_bundle=jax_model.ApplyFnBundle(
              apply_fn=apply_fn,
              input_polymorphic_shapes={
                  **input_polyshape,
              },
              jax_export_kwargs={'platforms': ['cpu', 'cuda']},
          ),
          params=model_params,
      ),
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
              processing=processor.SequentialProcessing(
                  preprocessors=[
                      tf_processor.TfProcessor(
                          name='embedding_layer',
                          fn=embedding_layer
                          ,
                      )
                  ],
                  options=processor.ProcessingOptions(
                      tf_trackable_resources=[embedding_layer]
                  ),
              ),
          )
      ],
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  export_model(FLAGS.export_path)


if __name__ == '__main__':
  app.run(main)
