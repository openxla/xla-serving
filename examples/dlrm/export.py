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

"""Neptune export for DLRM recommender model with TPUEmbeddingForServing."""

import os
import tempfile

from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf

from examples.dlrm import model as model_lib
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.export import tensorflow_model

FLAGS = flags.FLAGS

_NUM_DENSE_FEATURES = flags.DEFINE_integer(
    'num_dense_features', 13, 'Number of dense features.'
)


def get_synthetic_batch(batch_size, valency=None, vocab_sizes=None):
  """Returns a synthetic batch of data.

  Args:
    batch_size: The size of the batch to generate.
    valency: A list of sizes for the multi-valent features.
    vocab_sizes: A list of sizes for the vocabularies of the sparse features.

  Our features dict is an inputs of dictionary - {
      'label': (0,1)
      'dense_features': {
          '1': 0.123,
          '2': 0.456,
          ...
          '13': 0.987,
      },
      'sparse_features': {
          '1': sparse_tensor,
          '2': sparse_tensor,
          ...
          '26': sparse_tensor,
      },
  }
  """
  data = {}

  data['clicked'] = np.random.randint(0, 2, size=(batch_size,), dtype=np.int64)
  data['dense_features'] = np.random.uniform(
      0.0, 0.9, size=(batch_size, _NUM_DENSE_FEATURES.value)
  ).astype(np.float32)

  sparse_features = {}

  for i, rename_me in enumerate(valency):
    sparse_features[str(i)] = np.random.randint(
        low=0,
        high=vocab_sizes[i],
        size=(batch_size, rename_me),
    )

  data['sparse_features'] = sparse_features
  return data


def export_model(export_path: str):
  """Exports the DLRM recommender model using Neptune."""

  embedding_layer, model = model_lib.build_model()
  synthetic_batch_1 = get_synthetic_batch(
      batch_size=1,
      valency=model_lib.VALENCY,
      vocab_sizes=model_lib.VOCAB_SIZES,
  )
  synthetic_batch_64 = get_synthetic_batch(
      batch_size=64,
      valency=model_lib.VALENCY,
      vocab_sizes=model_lib.VOCAB_SIZES,
  )
  logging.info('synthetic_batch: %s', synthetic_batch_1)
  warmup_examples = {
      'serving_default': [
          synthetic_batch_1,
          synthetic_batch_64,
      ]
  }

  sparse_features = {}
  for i in range(len(model_lib.VALENCY)):
    sparse_features[str(i)] = tf.convert_to_tensor(
        synthetic_batch_1['sparse_features'][str(i)]
    )
  # TODO: Enable after Sparse/Ragged support is added.
  # Convert sparse features from dense to sparse tensors so that combiner.
  # for f, tensor in sparse_features.items():
  #   sparse_features[f] = tf.sparse.from_dense(tensor)
  logging.info('sparse_features: %s', sparse_features)
  tf_sparse_embeddings = embedding_layer(sparse_features)
  logging.info('tf_sparse_embeddings: %s', tf_sparse_embeddings)
  # This reduction is not needed when inputs are sparse tensors, TPU Embedding
  # for serving layer automatically combines the embeddings for the multi-valent
  # features (via parameter combiner="sum").
  # TODO: Remove this reduction after Sparse/Ragged support is
  # added.
  jax_sparse_embeddings = jax.tree.map(
      lambda x: jnp.sum(jnp.array(x.numpy()), axis=1), tf_sparse_embeddings
  )

  model_params = model.init(
      jax.random.PRNGKey(0),
      jnp.array(synthetic_batch_1['dense_features']),
      jax_sparse_embeddings,
  )

  def apply_fn(variables, inputs):
    predictions = model.apply(
        variables,
        inputs['dense_features'],
        inputs['sparse_embeddings'],
    )
    return {'predictions': predictions}

  @tf.function
  def serving_preprocessor(inputs):
    sparse_features = {
        str(i): inputs[str(i)] for i in range(len(model_lib.VOCAB_SIZES))
    }
    sparse_embeddings = embedding_layer(sparse_features)
    # TODO: Remove this reduction after Sparse/Ragged support is
    # added.
    summed_embeddings = {
        k: tf.reduce_sum(v, axis=1) for k, v in sparse_embeddings.items()
    }
    return {
        'dense_features': inputs['dense_features'],
        'sparse_embeddings': summed_embeddings,
    }

  input_signature = {
      str(i): tf.TensorSpec(shape=(None, model_lib.VALENCY[i]), dtype=tf.int32)
      for i in range(len(model_lib.VALENCY))
  }
  input_signature['dense_features'] = tf.TensorSpec(
      shape=(None, _NUM_DENSE_FEATURES.value), dtype=tf.float32
  )

  input_polyshape = {
      'dense_features': 'b, _',
      'sparse_embeddings': {
          str(i): 'b, _' for i in range(len(model_lib.VOCAB_SIZES))
      },
  }

  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'ckpt')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, model_params)
    model_params = checkpointer.restore(ckpt_path)

  pre_model = tensorflow_model.TensorflowModel(
      name='tf_model',
      tf_functions={'preprocessing': serving_preprocessor},
      trackable_resources=[embedding_layer],
  )
  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=apply_fn,
          input_polymorphic_shapes=input_polyshape,
          jax_export_kwargs={'platforms': ['cpu', 'cuda']},
          input_polymorphic_shape_symbol_values={
              'b': [1, 64],
          },
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
      warmup_examples=warmup_examples,
  )
