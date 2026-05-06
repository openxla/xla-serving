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

"""Pipeline: pre-processor + model-function + post-processor."""

from collections.abc import Sequence

import jax
import jaxtyping

from neptune_model._src.core import constants
from neptune_model._src.core import shlo_type
from neptune_model._src.core import type_proto_utils
from neptune_model.protos import manifest_pb2
from neptune_model.protos import orchestration_pb2


def _calculate_named_signature(
    signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
) -> Sequence[tuple[str, shlo_type.ShloTensorSpec]]:
  """Calculates a named signature from a PyTree of ShloTensorSpecs.

  It flattens the given PyTree signature and attaches a name to each leaf
  tensor spec. A leaf's name is determined by `leaf.name` if present, or
  the key/name associated with the last step of the JAX PyTree path to the
  leaf (e.g., a dictionary key or attribute name), if available.

  Args:
    signature: A PyTree of ShloTensorSpec.

  Returns:
    A sequence of (name, ShloTensorSpec) pairs.

  Raises:
    ValueError: If any leaf's name cannot be determined or if names are not
    unique across all leaves.
  """
  named_signature = []
  leaf_keys = set()
  # We use jax.tree_util.tree_leaves_with_path to handle PyTrees because it
  # supports custom node types registered with JAX. If other frameworks' custom
  # tree nodes need to be supported in the future, this traversal logic may
  # need to be redesigned.
  for path, leaf in jax.tree_util.tree_leaves_with_path(signature):
    if name := getattr(leaf, "name", None):
      leaf_key = name
    elif path:
      last_path_element = path[-1]
      if hasattr(last_path_element, "key"):
        leaf_key = str(last_path_element.key)
      elif hasattr(last_path_element, "name"):
        leaf_key = str(last_path_element.name)
      else:
        raise ValueError(
            "The signature of the orchestration pipeline must have not None"
            " names."
        )
    else:
      raise ValueError(
          "The signature of the orchestration pipeline must have not None"
          " names."
      )
    if leaf_key in leaf_keys:
      raise ValueError(
          "The signature of the orchestration pipeline must have unique"
          f" names. Got duplicate key: {leaf_key}"
      )
    named_signature.append((leaf_key, leaf))
    leaf_keys.add(leaf_key)
  return named_signature


def to_named_signature_proto(
    input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    output_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
) -> orchestration_pb2.NamedSignature:
  """Calculates named signatures and converts them to a NamedSignature proto.

  This function attempts to calculate named signatures from the input and
  output PyTrees by calling `_calculate_named_signature`. If calculation
  is successful for an input or output, the corresponding field in the
  `NamedSignature` proto is populated. If calculation fails (e.g., due to
  missing or duplicate names), this function will raise a ValueError.

  Args:
    input_signature: A PyTree of ShloTensorSpec for the input signature.
    output_signature: A PyTree of ShloTensorSpec for the output signature.

  Returns:
    A NamedSignature proto.

  Raises:
    ValueError: If named signatures could not be calculated for inputs or
    outputs.
  """
  named_input_signature = _calculate_named_signature(input_signature)
  named_output_signature = _calculate_named_signature(output_signature)
  named_signature_proto = orchestration_pb2.NamedSignature()
  if named_input_signature:
    named_signature_proto.inputs.SetInParent()
    named_signature_proto.inputs.named_tensor_types.extend([
        orchestration_pb2.NamedTensorType(
            name=name,
            tensor_type=type_proto_utils.to_tensor_type_proto(spec),
        )
        for name, spec in named_input_signature
    ])
  if named_output_signature:
    named_signature_proto.outputs.SetInParent()
    named_signature_proto.outputs.named_tensor_types.extend([
        orchestration_pb2.NamedTensorType(
            name=name,
            tensor_type=type_proto_utils.to_tensor_type_proto(spec),
        )
        for name, spec in named_output_signature
    ])
  return named_signature_proto


def serialize_pipelines_proto(
    proto: orchestration_pb2.Pipelines,
) -> manifest_pb2.UnstructuredData:
  return manifest_pb2.UnstructuredData(
      inlined_bytes=proto.SerializeToString(),
      mime_type=constants.ORCHESTRATION_MULTI_PIPELINE_MIME_TYPE,
      version=constants.ORCHESTRATION_MULTI_PIPELINE_VERSION,
  )
