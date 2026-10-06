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

"""Composite processor for XSE."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import os
from typing import Any

import jax
import jaxtyping

from xla_serving_data.processor.proto import transform_config_pb2
from xla_serving_data.processor.python import composite_processor_pybind
from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src import assets2xsm
from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.core import saveable
from xla_saved_model._src.core import serializable_function
from xla_saved_model._src.core import type_conversions
from xla_saved_model._src.core import unstructured_data
from xla_saved_model._src.processors import processor


def _tree_to_proto(tree: Any) -> transform_config_pb2.PyTreeProto:
  proto = transform_config_pb2.PyTreeProto()
  if tree is None:
    proto.type = transform_config_pb2.PyTreeProto.NONE
  elif isinstance(tree, dict):
    proto.type = transform_config_pb2.PyTreeProto.DICT
    for k, v in sorted(tree.items()):
      proto.dict_keys.append(k)
      proto.children.append(_tree_to_proto(v))
  elif isinstance(tree, list):
    proto.type = transform_config_pb2.PyTreeProto.LIST
    for v in tree:
      proto.children.append(_tree_to_proto(v))
  elif isinstance(tree, tuple):
    proto.type = transform_config_pb2.PyTreeProto.TUPLE
    for v in tree:
      proto.children.append(_tree_to_proto(v))
  else:
    proto.type = transform_config_pb2.PyTreeProto.LEAF
    proto.leaf_value.shape.extend(tree.shape)
    proto.leaf_value.dtype = (
        str(tree.dtype.name)
        if getattr(tree, "dtype", None)
        else str(tree.dtype)
    )
  return proto


def _proto_to_tree(proto: transform_config_pb2.PyTreeProto) -> Any:
  if proto.type == transform_config_pb2.PyTreeProto.NONE:
    return None
  elif proto.type == transform_config_pb2.PyTreeProto.DICT:
    return {
        k: _proto_to_tree(c) for k, c in zip(proto.dict_keys, proto.children)
    }
  elif proto.type == transform_config_pb2.PyTreeProto.LIST:
    return [_proto_to_tree(c) for c in proto.children]
  elif proto.type == transform_config_pb2.PyTreeProto.TUPLE:
    return tuple([_proto_to_tree(c) for c in proto.children])
  elif proto.type == transform_config_pb2.PyTreeProto.LEAF:
    return jax.ShapeDtypeStruct(
        shape=tuple(proto.leaf_value.shape), dtype=proto.leaf_value.dtype
    )
  else:
    raise ValueError(f"Unknown node type: {proto.type}")


def _propagate_pybind_signature(
    composite_processor: composite_processor_pybind.CompositeProcessor,
    output_structure_proto: transform_config_pb2.PyTreeProto,
    input_signature: jaxtyping.PyTree,
) -> jaxtyping.PyTree:
  flat_inputs, _ = jax.tree_util.tree_flatten(input_signature)
  pybind_inputs = []
  for leaf in flat_inputs:
    if isinstance(leaf, (list, tuple)):
      specs = [
          composite_processor_pybind.HostBufferSpec(
              item.dtype, list(item.shape)
          )
          for item in leaf
      ]
      pybind_inputs.append(
          composite_processor_pybind.ShapeInferenceValue(specs)
      )
    else:
      spec = composite_processor_pybind.HostBufferSpec(
          leaf.dtype, list(leaf.shape)
      )
      pybind_inputs.append(composite_processor_pybind.ShapeInferenceValue(spec))

  context = composite_processor_pybind.ShapeInferenceContext()
  context.inputs = pybind_inputs
  composite_processor.propagate_signature(context)

  def _to_shape_dtype_struct(val):
    if val.is_spec_list():
      return [
          jax.ShapeDtypeStruct(shape=spec.shape, dtype=spec.dtype)
          for spec in val.spec_list()
      ]
    else:
      spec = val.spec()
      return jax.ShapeDtypeStruct(shape=spec.shape, dtype=spec.dtype)

  flat_outputs = [_to_shape_dtype_struct(val) for val in context.outputs]
  output_structure = _proto_to_tree(output_structure_proto)
  _, output_treedef = jax.tree_util.tree_flatten(output_structure)
  return jax.tree_util.tree_unflatten(output_treedef, flat_outputs)


# Keep in sync with runtime/xla_serving_data/composite_processor_function_handler.h
COMPOSITE_PROCESSOR_MIME_TYPE = (
    "application/protobuf; type=xla_serving.data.composite.CompositeProcessorConfig"
)
COMPOSITE_PROCESSOR_VERSION = "0.0.1"
SERVING_DEFAULT_KEY = "serving_default"


@dataclasses.dataclass(frozen=True)
class PreparedCompositeProcessor(processor.ProcessorDefaults):
  """Composite processor that is prepared for export."""

  name: str
  asset_map: assets2xsm.AssetMap
  _input_signature: jaxtyping.PyTree
  _output_signature: jaxtyping.PyTree
  config: transform_config_pb2.CompositeProcessorConfig

  @property
  def input_signature(self) -> jaxtyping.PyTree:
    return self._input_signature

  @property
  def output_signature(self) -> jaxtyping.PyTree:
    return self._output_signature

  def to_xsm_saveable(self) -> saveable.Saveable:
    serialized_config = manifest_pb2.UnstructuredData(
        inlined_bytes=self.config.SerializeToString(),
        mime_type=COMPOSITE_PROCESSOR_MIME_TYPE,
        version=COMPOSITE_PROCESSOR_VERSION,
    )
    return serializable_function.SerializableFunction(
        input_signature=self.input_signature,
        output_signature=self.output_signature,
        body=unstructured_data.UnstructuredDataWithExtName(
            proto=serialized_config,
            subfolder=self.name,
            ext_name="pb",
        ),
    )

  def _save_assets(
      self,
      export_path: str,
  ) -> Mapping[str, persistence_lib.GlobalSupplemental]:
    assets2xsm.save_assets(self.asset_map, export_path)
    return {}

  def on_persist(
      self,
  ) -> (
      Callable[
          [str],
          Mapping[str, persistence_lib.GlobalSupplemental],
      ]
      | None
  ):
    return self._save_assets


class CompositeProcessor:
  """Composite processor for XSE."""

  def __init__(self, name: str, transforms: Sequence[Any]):
    self._name = name
    self._transforms = transforms

    if not self._name:
      raise ValueError("CompositeProcessor name must not be empty")

    if not transforms:
      raise ValueError("Expected at least one transform to be passed in")

    graph = composite_processor_pybind.ExecutionGraph()
    asset_paths: list[str] = []

    for i, t in enumerate(transforms):
      node_name = f"{t.transform.name}_{i}"
      graph.add_transform(node_name, t.transform)
      asset_paths.extend(t.export_assets().values())

    composite_processor = composite_processor_pybind.CompositeProcessor.create(
        graph
    )
    self._composite_processor = composite_processor
    self._processor_config = composite_processor.to_proto()
    self._asset_map = assets2xsm.AssetMap(self._name)
    for asset_path in asset_paths:
      self._asset_map.add_asset(asset_path)

    self._rewrite_asset_paths()

  @property
  def name(self) -> str:
    return self._name

  def method_keys(self) -> Sequence[str]:
    """Returns the supported method keys for the composite processor."""
    return [SERVING_DEFAULT_KEY]

  def _get_output_signature(
      self, input_signature: jaxtyping.PyTree
  ) -> jaxtyping.PyTree:
    """Computes the output signature by chaining the transforms."""
    intermediate_sig = input_signature
    for t in self._transforms:
      intermediate_sig = t.get_output_signature(intermediate_sig)
    return intermediate_sig

  def _rewrite_asset_paths(self) -> None:
    """Rewrites the asset paths in the processor config.

    The processor may use its own assets. When it runs in the XSE runtime,
    these assets will be loaded from the model export directory, so they need
    to be copied there during export. We use assets2xsm to copy the assets and
    need to update the processor config to point to the new asset paths.
    """
    for resource in self._processor_config.resources:
      for asset in resource.assets:
        if asset.path in self._asset_map.assets:
          asset.path = os.path.join(
              self._asset_map.bundle_name,
              self._asset_map.assets[asset.path],
          )

  @classmethod
  def from_graph(
      cls,
      name: str,
      graph: composite_processor_pybind.ExecutionGraph,
      assets: dict[str, str],
  ) -> "CompositeProcessor":
    """Constructs a CompositeProcessor from an ExecutionGraph."""
    instance = cls.__new__(cls)
    instance._name = name
    instance._transforms = None  # pyrefly: ignore[bad-assignment]

    pybind_processor = composite_processor_pybind.CompositeProcessor.create(
        graph
    )
    instance._composite_processor = pybind_processor
    instance._processor_config = pybind_processor.to_proto()
    instance._asset_map = assets2xsm.AssetMap(name)
    for asset_path in assets.values():
      instance._asset_map.add_asset(asset_path)

    instance._rewrite_asset_paths()
    return instance

  def prepare(
      self,
      input_signature: jaxtyping.PyTree,
      **kwargs: Any,
  ) -> PreparedCompositeProcessor:
    """Prepares the processor for export."""
    del kwargs  # unused

    if self._transforms is None:
      jd_output_signature = _propagate_pybind_signature(
          self._composite_processor,
          self._processor_config.io_structure.output_structure,
          input_signature,
      )
    else:
      jd_output_signature = self._get_output_signature(input_signature)

    xsm_input_signature = type_conversions.to_shlo_tensor_spec(input_signature)
    xsm_output_signature = type_conversions.to_shlo_tensor_spec(
        jd_output_signature
    )

    return PreparedCompositeProcessor(
        self._name,
        self._asset_map,
        xsm_input_signature,
        xsm_output_signature,
        self._processor_config,
    )
