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

"""XLA SavedModel export example using the math and bitwise data ops.

This example shows two ways to use the elementwise transforms exposed by
`data.ops.math` and `data.ops.bitwise`:

* Eagerly on host buffers, e.g. `math.log1p(buffer)` (see `compute_features`).
* As nodes of a `CompositeProcessor` graph that runs as the preprocessor of an
  exported XLA SavedModel (see `build_preprocessor` and `export_model`).

The preprocessor turns raw request features into model features:

  log_revenue     = log1p(price * quantity)
  user_item_cross = user_id ^ item_id

The JAX model then scores each example as

  pred = W * log_revenue + cross_weights[user_item_cross % NUM_BUCKETS] + B
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xla_serving_data.kernels import numerical_transforms_module
from xla_serving_data.ops import bitwise
from xla_serving_data.ops import math
from xla_serving_data.processor.proto import transform_config_pb2
from xla_serving_data.processor.python import composite_processor_pybind
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config

NUM_BUCKETS = 16
WEIGHT = 2.0
BIAS = 0.5

# Raw request features. Each feature has shape [batch_size, 1].
INPUT_SIGNATURE = {
    'item_id': jax.ShapeDtypeStruct(shape=(1, 1), dtype=jnp.int32),
    'price': jax.ShapeDtypeStruct(shape=(1, 1), dtype=jnp.float32),
    'quantity': jax.ShapeDtypeStruct(shape=(1, 1), dtype=jnp.float32),
    'user_id': jax.ShapeDtypeStruct(shape=(1, 1), dtype=jnp.int32),
}

# Model features produced by the preprocessor, in output order.
FEATURE_SIGNATURE = [
    jax.ShapeDtypeStruct(shape=(1, 1), dtype=jnp.float32),  # log_revenue
    jax.ShapeDtypeStruct(shape=(1, 1), dtype=jnp.int32),  # user_item_cross
]


def cross_weights() -> np.ndarray:
  """Returns the per-bucket weights of the user/item cross feature."""
  return np.arange(NUM_BUCKETS, dtype=np.float32) / NUM_BUCKETS


def _to_host_buffer(x: np.ndarray) -> composite_processor_pybind.HostBuffer:
  return composite_processor_pybind.HostBuffer.from_dlpack(
      np.ascontiguousarray(x)
  )


def compute_features(
    price: np.ndarray,
    quantity: np.ndarray,
    user_id: np.ndarray,
    item_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes the model features eagerly with the math and bitwise ops.

  Each op runs its C++ transform immediately on host buffers. This is handy to
  debug the preprocessing logic before wiring it into a graph.

  Args:
    price: float32 array of shape [batch_size, 1].
    quantity: float32 array of shape [batch_size, 1].
    user_id: int32 array of shape [batch_size, 1].
    item_id: int32 array of shape [batch_size, 1].

  Returns:
    A `(log_revenue, user_item_cross)` tuple of NumPy arrays.
  """
  revenue = math.multiply(_to_host_buffer(price), _to_host_buffer(quantity))
  log_revenue = math.log1p(revenue)
  user_item_cross = bitwise.bitwise_xor(
      _to_host_buffer(user_id), _to_host_buffer(item_id)
  )
  return np.from_dlpack(log_revenue), np.from_dlpack(user_item_cross)


def _to_pytree_proto(tree: Any) -> transform_config_pb2.PyTreeProto:
  """Converts a dict/list tree of `jax.ShapeDtypeStruct`s to a PyTreeProto."""
  proto = transform_config_pb2.PyTreeProto()
  if isinstance(tree, Mapping):
    proto.type = transform_config_pb2.PyTreeProto.DICT
    for key in sorted(tree):
      proto.dict_keys.append(key)
      proto.children.append(_to_pytree_proto(tree[key]))
  elif isinstance(tree, list):
    proto.type = transform_config_pb2.PyTreeProto.LIST
    for child in tree:
      proto.children.append(_to_pytree_proto(child))
  else:
    proto.type = transform_config_pb2.PyTreeProto.LEAF
    # Every dim of these signatures is known, so every size is set.
    proto.leaf_value.partial_shape.SetInParent()
    for size in tree.shape:
      proto.leaf_value.partial_shape.dims.add(size=size)
    proto.leaf_value.dtype = np.dtype(tree.dtype).name
  return proto


def build_preprocessor() -> main_lib.CompositeModel:
  """Builds the preprocessing graph out of math and bitwise transforms."""
  graph = composite_processor_pybind.ExecutionGraph()
  edge = composite_processor_pybind.GraphEdge

  # Graph inputs are the leaves of INPUT_SIGNATURE, ordered by sorted key.
  input_index = {key: i for i, key in enumerate(sorted(INPUT_SIGNATURE))}

  revenue = graph.add_node(
      'revenue',
      numerical_transforms_module.MultiplyTransform().transform,
      [
          edge.graph_input(input_index['price']),
          edge.graph_input(input_index['quantity']),
      ],
  )
  log_revenue = graph.add_node(
      'log_revenue',
      numerical_transforms_module.Log1pTransform().transform,
      [edge.node_output(revenue, 0)],
  )
  user_item_cross = graph.add_node(
      'user_item_cross',
      numerical_transforms_module.BitwiseXorTransform().transform,
      [
          edge.graph_input(input_index['user_id']),
          edge.graph_input(input_index['item_id']),
      ],
  )
  graph.set_outputs([
      edge.node_output(log_revenue, 0),
      edge.node_output(user_item_cross, 0),
  ])

  io_structure = transform_config_pb2.GraphIOStructureProto()
  io_structure.input_structure.CopyFrom(_to_pytree_proto(INPUT_SIGNATURE))
  io_structure.output_structure.CopyFrom(_to_pytree_proto(FEATURE_SIGNATURE))
  graph.io_structure = io_structure

  return main_lib.CompositeModel.from_graph(
      name='math_preprocessor', graph=graph, assets={}
  )


def export_model(export_path: str) -> None:
  """Exports the preprocessor and a JAX model as an XLA SavedModel."""

  def apply_fn(params, inputs):
    log_revenue, user_item_cross = inputs
    bucket = jnp.mod(user_item_cross, NUM_BUCKETS)
    cross_weight = jnp.take(params['cross_weights'], bucket)
    pred = params['w'] * log_revenue + cross_weight + params['b']
    return {'pred': pred}

  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=apply_fn,
          input_polymorphic_shapes=['b, _', 'b, _'],
          jax_export_kwargs={'platforms': ['cpu']},
      ),
      params={
          'w': jnp.array(WEIGHT, dtype=jnp.float32),
          'b': jnp.array(BIAS, dtype=jnp.float32),
          'cross_weights': jnp.asarray(cross_weights()),
      },
      name='jax_model',
  )

  main_lib.save(
      [build_preprocessor(), jm],
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=INPUT_SIGNATURE,
              orchestration=serving_config.Orchestration(
                  model_calls=[
                      serving_config.ModelCall(
                          model_name='math_preprocessor',
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
