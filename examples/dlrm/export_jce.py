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

from .third_party.neptune.data.ops import cpu_embedding_module
from .third_party.neptune.data.ops import pack_jce_inputs_module
from .third_party.neptune.data.ops import unpack_tensor_array_module
from examples.dlrm import model_jce as model_lib
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.processors import processor
from neptune_model._src.processors.google import composite_processor


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
      'z_dense_features': {
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
  data['z_dense_features'] = np.random.uniform(
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
      jnp.array(synthetic_batch['z_dense_features']),
      jax_sparse_embeddings,
  )

  def apply_fn(variables, inputs):
    predictions = model.apply(
        variables,
        inputs['z_dense_features'],
        inputs['sparse_embeddings'],
    )
    return {'predictions': predictions}

  jce_module = cpu_embedding_module.CpuEmbeddingModule(
      resource=jce_resource,
      feature_names=[str(i) for i in range(len(model_lib.VOCAB_SIZES))],
      output_feature_names=[
          f'sparse_embeddings/{i}' for i in range(len(model_lib.VOCAB_SIZES))
      ],
      output_shapes=[[None, model_lib.EMBEDDING_SIZE]]
      * len(model_lib.VOCAB_SIZES),
      dense_lookup=True,
      passthrough_feature_names=['z_dense_features'],
  )

  # Calculate the input signature here so we can pass it consistently
  input_signature = {
      str(i): jax.ShapeDtypeStruct(
          shape=(None, model_lib.VALENCY[i]), dtype=jnp.int32
      )
      for i in range(len(model_lib.VALENCY))
  }
  input_signature['z_dense_features'] = jax.ShapeDtypeStruct(
      shape=(None, num_dense_features), dtype=jnp.float32
  )

  dict_keys = sorted(
      [str(i) for i in range(len(model_lib.VOCAB_SIZES))] + ['z_dense_features']
  )
  id_indices = [
      dict_keys.index(str(i)) for i in range(len(model_lib.VOCAB_SIZES))
  ]
  pass_indices = [dict_keys.index('z_dense_features')]

  pack_module = pack_jce_inputs_module.PackJceInputsModule(
      num_inputs=len(dict_keys),
      id_indices=id_indices,
      rp_indices=[],
      weight_indices=[],
      pass_indices=pass_indices,
  )

  unpack_module = unpack_tensor_array_module.UnpackTensorArrayModule(
      num_outputs=len(model_lib.VOCAB_SIZES) + 1,  # 26 sparse + 1 dense
      output_signature=jce_module.get_output_signature(input_signature),
  )

  composite_preprocessor = composite_processor.CompositeProcessor(
      name='jce_preprocessor',
      transforms=[pack_module, jce_module, unpack_module],
  )

  input_polyshape = {
      'z_dense_features': 'b, _',
      'sparse_embeddings': {
          str(i): 'b, _' for i in range(len(model_lib.VOCAB_SIZES))
      },
  }

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
              input_polymorphic_shapes=input_polyshape,
              jax_export_kwargs={'platforms': ['cpu']},
          ),
          params=model_params,
      ),
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
              processing=processor.SequentialProcessing(
                  preprocessors=[
                      composite_preprocessor,
                  ],
              ),
          )
      ],
  )
