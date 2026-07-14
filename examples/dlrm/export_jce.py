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
from typing import Any, Sequence

from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import orbax.checkpoint as ocp

from neptune_data.cpu_embedding.python import embedding_utils
from neptune_data.kernels.cpu_embedding import cpu_embedding_module
from neptune_data.ops import packing
from neptune_data.processor.proto import transform_config_pb2
from neptune_data.processor.python import composite_processor_pybind
from neptune_data.resource_manager.python import resource_manager_pybind
from examples.dlrm import model_jce as model_lib
from neptune_model._src import assets2npt
from neptune_model._src.core import shlo_type
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.processors import processor
from neptune_model.protos import orchestration_pb2


def get_synthetic_batch(
    batch_size, valency=None, vocab_sizes=None, num_dense_features: int = 13
):
  """Returns a synthetic batch of data.

  Args:
    batch_size: The size of the batch to generate.
    valency: A list of sizes for the multi-valent features.
    vocab_sizes: A list of sizes for the vocabularies of the sparse features.
    num_dense_features: The number of dense features.

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
      0.0, 0.9, size=(batch_size, num_dense_features)
  ).astype(np.float32)

  sparse_features = {}

  for i, val in enumerate(valency):
    sparse_features[str(i)] = np.random.randint(
        low=0,
        high=vocab_sizes[i],
        size=(batch_size, val),
    )

  data['sparse_features'] = sparse_features
  return data


def export_model(
    export_path: str,
    use_capped_vocab: bool = False,
    embedding_size: int = 128,
    global_batch_size: int = 2,
    num_dense_features: int = 13,
):
  """Exports the DLRM recommender model using Neptune."""

  # Initialize synthetic weights and JCE resource for serving lookups
  if use_capped_vocab:
    vocab_sizes = [min(v, 100) for v in model_lib.VOCAB_SIZES]
  else:
    vocab_sizes = model_lib.VOCAB_SIZES

  table_data = {}
  combine_modes = {}
  feature_specs = []
  for i, size in enumerate(vocab_sizes):
    table_name = str(i)
    feature_name = str(i)
    table_data[table_name] = np.random.uniform(
        -0.05, 0.05, size=(size, embedding_size)
    ).astype(np.float32)
    combine_modes[table_name] = 'SUM'
    feature_specs.append({
        'name': feature_name,
        'table_name': table_name,
    })

  jce_resource = cpu_embedding_module.CpuEmbeddingResource(
      table_data=table_data,
      combine_modes=combine_modes,
      feature_specs=feature_specs,
  )

  model = model_lib.build_model(embedding_size=embedding_size)
  synthetic_batch = get_synthetic_batch(
      global_batch_size,
      model_lib.VALENCY,
      vocab_sizes,
      num_dense_features=num_dense_features,
  )
  logging.info('synthetic_batch: %s', synthetic_batch)

  # Generate mock JAX sparse embeddings arrays directly instead of calling TF
  jax_sparse_embeddings = {
      str(i): jnp.zeros(
          (global_batch_size, embedding_size),
          dtype=jnp.float32,
      )
      for i in range(len(model_lib.VOCAB_SIZES))
  }

  model_params = model.init(
      jax.random.PRNGKey(0),
      jnp.array(synthetic_batch['dense_features']),
      jax_sparse_embeddings,
  )

  def apply_fn(variables, inputs):
    # Unpack the 27 positional input tensors emitted by the preprocessor:
    # inputs[0] to inputs[25] are the sparse embeddings.
    # inputs[26] is the dense passthrough features.
    sparse_embeddings = {
        str(i): inputs[i] for i in range(len(model_lib.VOCAB_SIZES))
    }
    dense_features = inputs[len(model_lib.VOCAB_SIZES)]
    predictions = model.apply(
        variables,
        dense_features,
        sparse_embeddings,
    )
    return {'predictions': predictions}

  # Build the 3-transform pipeline to map arbitrary OBM inputs into strict JCE signatures:
  # 1. PackJceInputsModule: Assembles flat inputs into JCE's 4-vector shape using alphabetical indices.
  # 2. CpuEmbeddingModule: Executes the actual high-performance embedding lookup.
  # 3. UnpackTensorArrayModule: Scatters lookup outputs back to separate positional leaf tensors.

  input_signature = {
      str(i): jax.ShapeDtypeStruct(
          shape=(1, model_lib.VALENCY[i]), dtype=jnp.int32
      )
      for i in range(len(model_lib.VALENCY))
  }
  input_signature['dense_features'] = jax.ShapeDtypeStruct(
      shape=(1, num_dense_features), dtype=jnp.float32
  )

  composite_preprocessor = embedding_utils.build_embedding_lookup(
      name='jce_preprocessor',
      input_signature=input_signature,
      feature_specs=feature_specs,
      jce_resource=jce_resource,
      embedding_dim=embedding_size,
      dense_lookup=True,
  )

  input_polyshape = ['b, _'] * (len(model_lib.VOCAB_SIZES) + 1)

  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'ckpt')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, model_params)
    model_params = checkpointer.restore(ckpt_path)

  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=apply_fn,
          input_polymorphic_shapes=input_polyshape,
          jax_export_kwargs={'platforms': ['cpu']},
          input_polymorphic_shape_symbol_values={
              'b': [1, 64, 512, 4096],
          },
      ),
      params=model_params,
      name='jax_model',
  )

  main_lib.save(
      models=[composite_preprocessor, jm],
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
              orchestration=serving_config.Orchestration(
                  model_calls=[
                      serving_config.ModelCall(
                          model_name='jce_preprocessor',
                          method_key='serving_default',
                          role=orchestration_pb2.ROLE_PREPROCESSOR,
                      ),
                      serving_config.ModelCall(
                          model_name='jax_model',
                          method_key='serving_default',
                          role=orchestration_pb2.ROLE_MODEL,
                      ),
                  ]
              ),
          )
      ],
  )
