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

"""Generic JAX Data preprocessor execution graph builder for CPU embedding lookups."""

from typing import Any, Sequence

import jax
import jax.numpy as jnp

from data.kernels.cpu_embedding import cpu_embedding_module
from data.ops import packing
from data.processor.proto import transform_config_pb2
from data.processor.python import composite_processor_pybind
from data.processor.python import processor
from neptune_model._src.processors import composite_processor


def build_embedding_lookup(
    name: str,
    input_signature: dict[str, jax.ShapeDtypeStruct],
    feature_specs: Sequence[dict[str, Any]],
    jce_resource: Any,
    embedding_dim: int | None = None,
    dense_lookup: bool = True,
) -> composite_processor.CompositeProcessor:
  """Builds a generic preprocessor CompositeProcessor for CPU embedding lookups.

  Args:
    name: Preprocessor name (e.g. 'jce_preprocessor').
    input_signature: Dictionary mapping feature name to ShapeDtypeStruct.
    feature_specs: List of feature specifications containing 'name',
      'table_name', and optional 'max_sequence_length'.
    jce_resource: CpuEmbeddingResource containing embedding tables.
    embedding_dim: Optional embedding vector dimension. Defaults to None
      (auto-inferred from jce_resource.embedding_dim).
    dense_lookup: If True, uses the fast-path dense lookup with empty row
      pointers & weights. If False, packs row pointers and weights.

  Returns:
    A CompositeProcessor instance wrapping the ExecutionGraph DAG.
  """
  if embedding_dim is None:
    embedding_dim = jce_resource.embedding_dim

  ordered_keys = sorted(input_signature.keys())
  feature_names = [f['name'] for f in feature_specs]
  sparse_indices = [
      ordered_keys.index(fn) for fn in feature_names if fn in ordered_keys
  ]
  passthrough_indices = [
      idx for idx in range(len(ordered_keys)) if idx not in sparse_indices
  ]

  # 1. Prepare HostBuffer Packing Modules
  pack_ids_module = packing.PackHostBufferVectorModule(
      input_count=len(sparse_indices)
  )

  if dense_lookup:
    pack_rps_module = packing.PackHostBufferVectorModule(input_count=0)
    pack_weights_module = packing.PackHostBufferVectorModule(input_count=0)
  else:
    pack_rps_module = packing.PackHostBufferVectorModule(
        input_count=len(sparse_indices)
    )
    pack_weights_module = packing.PackHostBufferVectorModule(
        input_count=len(sparse_indices)
    )

  # 2. Derive output shapes for Unpack Module
  unpack_output_shapes = []
  for f in feature_specs:
    seq_len = f.get('max_sequence_length', 0)
    if seq_len > 0:
      unpack_output_shapes.append([None, seq_len, embedding_dim])
    else:
      unpack_output_shapes.append([None, embedding_dim])

  jce_module = cpu_embedding_module.CpuEmbeddingModule(
      resource=jce_resource,
      feature_names=feature_names,
      output_feature_names=feature_names,
      output_shapes=unpack_output_shapes,
      dense_lookup=dense_lookup,
  )

  unpack_module = packing.UnpackHostBufferVectorModule(
      output_count=len(feature_specs),
      output_shapes=unpack_output_shapes,
  )

  # Initialize modules
  pack_ids_module.initialize(None)
  pack_rps_module.initialize(None)
  pack_weights_module.initialize(None)
  unpack_module.initialize(None)

  # 3. Construct ExecutionGraph DAG
  graph = composite_processor_pybind.ExecutionGraph()

  # Node 0: Pack IDs
  input_edges_ids = [
      composite_processor_pybind.GraphEdge.graph_input(idx)
      for idx in sparse_indices
  ]
  node_0_idx = graph.add_node(
      'pack_ids', pack_ids_module.transform, input_edges_ids
  )

  # Node 1: Pack RPs
  if dense_lookup:
    node_1_idx = graph.add_node('pack_rps', pack_rps_module.transform, [])
  else:
    input_edges_rps = [
        composite_processor_pybind.GraphEdge.graph_input(idx)
        for idx in sparse_indices
    ]
    node_1_idx = graph.add_node(
        'pack_rps', pack_rps_module.transform, input_edges_rps
    )

  # Node 2: Pack Weights
  if dense_lookup:
    node_2_idx = graph.add_node(
        'pack_weights', pack_weights_module.transform, []
    )
  else:
    input_edges_weights = [
        composite_processor_pybind.GraphEdge.graph_input(idx)
        for idx in sparse_indices
    ]
    node_2_idx = graph.add_node(
        'pack_weights', pack_weights_module.transform, input_edges_weights
    )

  # Node 3: JCE Lookup
  input_edges_jce = [
      composite_processor_pybind.GraphEdge.node_output(node_0_idx, 0),
      composite_processor_pybind.GraphEdge.node_output(node_1_idx, 0),
      composite_processor_pybind.GraphEdge.node_output(node_2_idx, 0),
  ]
  node_3_idx = graph.add_node(
      'jce_lookup', jce_module.transform, input_edges_jce
  )

  # Node 4: Unpack
  input_edges_unpack = [
      composite_processor_pybind.GraphEdge.node_output(node_3_idx, 0)
  ]
  node_4_idx = graph.add_node(
      'unpack', unpack_module.transform, input_edges_unpack
  )

  output_edges = [
      composite_processor_pybind.GraphEdge.node_output(node_4_idx, i)
      for i in range(len(feature_specs))
  ] + [
      composite_processor_pybind.GraphEdge.graph_input(idx)
      for idx in passthrough_indices
  ]
  graph.set_outputs(output_edges)

  # 4. Construct Output Signature & IO Structure for ExecutionGraph

  output_signature = []
  for f in feature_specs:
    seq_len = f.get('max_sequence_length', 0)
    if seq_len > 0:
      output_signature.append(
          jax.ShapeDtypeStruct(
              shape=(1, seq_len, embedding_dim), dtype=jnp.float32
          )
      )
    else:
      output_signature.append(
          jax.ShapeDtypeStruct(shape=(1, embedding_dim), dtype=jnp.float32)
      )

  for idx in passthrough_indices:
    key = ordered_keys[idx]
    output_signature.append(input_signature[key])

  io_structure = transform_config_pb2.GraphIOStructureProto()
  io_structure.input_structure.CopyFrom(
      processor._tree_to_proto(input_signature)
  )
  io_structure.output_structure.CopyFrom(
      processor._tree_to_proto(output_signature)
  )
  graph.io_structure = io_structure

  return composite_processor.CompositeProcessor.from_graph(
      name=name,
      graph=graph,
      assets=jce_resource.export_assets(),
  )
