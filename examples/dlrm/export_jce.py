# Copyright 2026 Google LLC
# Copyright 2026 Arm Ltd. and affiliates
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

"""XLA SavedModel export for DLRM with native CPU embedding."""

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

from xla_serving_data.cpu_embedding.python import embedding_utils
from xla_serving_data.kernels.cpu_embedding import cpu_embedding_module
from xla_serving_data.ops import packing
from xla_serving_data.processor.proto import transform_config_pb2
from xla_serving_data.processor.python import composite_processor_pybind
from xla_serving_data.resource_manager.python import resource_manager_pybind
from examples.dlrm import model_jce as model_lib
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src import assets2xsm
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.processors import processor


def get_synthetic_batch(
    batch_size: int,
    valency: Sequence[int] | None = None,
    vocab_sizes: Sequence[int] | None = None,
    num_dense_features: int = 13,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
  """Returns click labels, FP32 dense features and int32 sparse feature arrays.

  All arrays have batch_size rows; sparse feature i has valency[i] columns.
  Pass a seeded rng for reproducible draws.
  """
  if valency is None:
    valency = model_lib.VALENCY
  if vocab_sizes is None:
    vocab_sizes = model_lib.VOCAB_SIZES
  if rng is None:
    rng = np.random.default_rng()

  data = {}

  data['clicked'] = rng.integers(0, 2, size=(batch_size,), dtype=np.int64)
  data['dense_features'] = rng.uniform(
      0.0, 0.9, size=(batch_size, num_dense_features)
  ).astype(np.float32)

  sparse_features = {}

  for i, val in enumerate(valency):
    sparse_features[str(i)] = rng.integers(
        low=0,
        high=vocab_sizes[i],
        size=(batch_size, val),
        dtype=np.int32,
    )

  data['sparse_features'] = sparse_features
  return data


def export_model(
    export_path: str,
    use_capped_vocab: bool = False,
    embedding_size: int = 128,
    global_batch_size: int = 2,
    num_dense_features: int = 13,
    *,
    seed: int = 0,
    vocab_size_cap: int | None = None,
    benchmark_inputs: str | None = None,
    num_input_batches: int = 8,
):
  """Exports the DLRM recommender model using XLA SavedModel."""
  if benchmark_inputs is not None and os.path.exists(benchmark_inputs):
    raise FileExistsError(benchmark_inputs)
  rng = np.random.default_rng(seed)

  # Initialize synthetic weights and JCE resource for serving lookups
  if use_capped_vocab:
    vocab_sizes = [min(v, 100) for v in model_lib.VOCAB_SIZES]
  elif vocab_size_cap is not None:
    vocab_sizes = [min(v, vocab_size_cap) for v in model_lib.VOCAB_SIZES]
  else:
    vocab_sizes = model_lib.VOCAB_SIZES

  table_data = {}
  combine_modes = {}
  feature_specs = []
  for i, size in enumerate(vocab_sizes):
    table_name = str(i)
    feature_name = str(i)
    table_data[table_name] = rng.uniform(
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
      rng=rng,
  )
  logging.info('Initializing synthetic DCN v2, batch size %d', global_batch_size)

  # Generate mock JAX sparse embeddings arrays directly instead of calling TF
  jax_sparse_embeddings = {
      str(i): jnp.zeros(
          (global_batch_size, embedding_size),
          dtype=jnp.float32,
      )
      for i in range(len(model_lib.VOCAB_SIZES))
  }

  model_params = model.init(
      jax.random.PRNGKey(seed),
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
              'b': (
                  [global_batch_size]
                  if benchmark_inputs is not None
                  else [1, 64, 512, 4096]
              ),
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

  if benchmark_inputs is not None:
    batch = get_synthetic_batch(
        global_batch_size * num_input_batches,
        model_lib.VALENCY,
        vocab_sizes,
        num_dense_features,
        rng=np.random.default_rng(np.random.SeedSequence([seed, 1])),
    )
    inputs = {**batch['sparse_features'], 'dense_features': batch['dense_features']}
    inputs = {
        name: array.reshape(num_input_batches, global_batch_size, -1)
        for name, array in inputs.items()
    }
    predict = jax.jit(model.apply)
    predictions = []
    for index in range(num_input_batches):
      pooled = {
          name: table[inputs[name][index]].sum(axis=1, dtype=np.float32)
          for name, table in table_data.items()
      }
      predictions.append(np.asarray(predict(
          model_params, inputs['dense_features'][index], pooled
      ), dtype=np.float32))
    with open(benchmark_inputs, 'xb') as stream:
      np.savez(stream, **inputs, predictions=np.stack(predictions))
