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

"""Neptune model with a TF preprocessor that uses tf.RaggedTensor and tf.SparseTensor."""

import jax.numpy as jnp
import tensorflow as tf

from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.export import tensorflow_model


@tf.function
def first_preprocessor(
    inputs: dict[str, tf.Tensor],
) -> tuple[tf.RaggedTensor, tf.SparseTensor]:
  """A TF preprocessor that takes and returns composite tensors."""
  ragged_input = inputs["ragged_input"]
  if not isinstance(ragged_input, tf.RaggedTensor):
    raise TypeError(
        "Expected a tf.RaggedTensor for 'ragged_input', got"
        f" {type(ragged_input).__name__}."
    )
  sparse_input = inputs["sparse_input"]
  if not isinstance(sparse_input, tf.SparseTensor):
    raise TypeError(
        "Expected a tf.SparseTensor for 'sparse_input', got"
        f" {type(sparse_input).__name__}."
    )
  return ragged_input * 2, tf.SparseTensor(
      indices=sparse_input.indices,
      values=sparse_input.values * 3,
      dense_shape=sparse_input.dense_shape,
  )


@tf.function
def second_preprocessor(
    ragged_comps: tuple[tf.Tensor, tf.Tensor],
    sparse_comps: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> dict[str, tf.Tensor]:
  """A TF preprocessor that takes the composite tensors components."""
  ragged = tf.RaggedTensor.from_row_splits(
      values=ragged_comps[0], row_splits=ragged_comps[1]
  )
  sparse = tf.SparseTensor(
      indices=sparse_comps[0],
      values=sparse_comps[1],
      dense_shape=sparse_comps[2],
  )
  ragged_sum = tf.reduce_sum(ragged, axis=1, keepdims=True)
  sparse_sum = tf.sparse.reduce_sum(sparse, axis=1, keepdims=True)
  combined = ragged_sum + tf.cast(sparse_sum, ragged_sum.dtype)
  combined.set_shape([None, 1])
  return {"processed_input": tf.cast(combined, tf.float32)}


def apply_fn(params, inputs):
  """Trivial JAX model: scale the (batch, 1) preprocessed feature."""
  return {"pred": inputs["processed_input"] * params["w"]}


def export_model(export_path: str):
  """Exports a minimal pipeline with ragged and sparse input features."""
  input_signature = {
      "ragged_input": tf.RaggedTensorSpec(
          shape=(None, None), dtype=tf.int32, ragged_rank=1
      ),
      "sparse_input": tf.SparseTensorSpec(
          shape=(None, None), dtype=tf.int32
      ),
  }

  tf_proc_model = tensorflow_model.TensorflowModel(
      name="tf_model",
      tf_functions={
          "first_preprocessor": first_preprocessor,
          "second_preprocessor": second_preprocessor,
      },
  )
  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=apply_fn,
          input_polymorphic_shapes={"processed_input": "b, _"},
          jax_export_kwargs={"platforms": ["cpu"]},
      ),
      params={"w": jnp.asarray(2.0, dtype=jnp.float32)},
      name="jax_model",
  )

  main_lib.save(
      [tf_proc_model, jm],
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature="serving_default",
              input_signature=input_signature,
              orchestration=serving_config.Orchestration(
                  model_calls=[
                      serving_config.ModelCall(
                          model_name="tf_model",
                          method_key="first_preprocessor",
                          role=orchestration_pb2.ROLE_PREPROCESSOR,
                      ),
                      serving_config.ModelCall(
                          model_name="tf_model",
                          method_key="second_preprocessor",
                          role=orchestration_pb2.ROLE_PREPROCESSOR,
                      ),
                      serving_config.ModelCall(
                          model_name="jax_model",
                          method_key="serving_default",
                          role=orchestration_pb2.ROLE_MODEL,
                      ),
                  ]
              ),
          )
      ],
  )
