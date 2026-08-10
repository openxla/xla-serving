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

"""Composite processor for Neptune."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import os
from typing import Any

import jax
import jaxtyping
import numpy as np

from neptune_data.processor.proto import transform_config_pb2
from neptune_data.processor.python import composite_processor_pybind
from neptune_model._src import assets2npt
from neptune_model._src.core import persistence_lib
from neptune_model._src.core import saveable
from neptune_model._src.core import serializable_function
from neptune_model._src.core import shlo_type
from neptune_model._src.core import unstructured_data
from neptune_model._src.processors import processor
from neptune_model.protos import manifest_pb2

# //neptune/runtime/neptune_data/composite_processor_function_handler.h
COMPOSITE_PROCESSOR_MIME_TYPE = (
    "application/protobuf; type=neptune.data.composite.CompositeProcessorConfig"
)
COMPOSITE_PROCESSOR_VERSION = "0.0.1"
SERVING_DEFAULT_KEY = "serving_default"


def _convert_jd_type_to_shlo_type(dtype: Any) -> shlo_type.ShloDType:
  if isinstance(dtype, shlo_type.ShloDType):
    return dtype
  if not isinstance(dtype, np.dtype):
    raise ValueError(
        f"Expected a numpy.dtype or ShloDType, got {dtype!r} of type"
        f" {type(dtype)}"
    )
  if dtype.kind in ("U", "S", "O"):
    return shlo_type.ShloDType.str
  return shlo_type.np_dtype_to_shlo_dtype(dtype)


@dataclasses.dataclass(frozen=True)
class PreparedCompositeProcessor(processor.ProcessorDefaults):
  """Composite processor that is prepared for export."""

  name: str
  asset_map: assets2npt.AssetMap
  _input_signature: jaxtyping.PyTree
  _output_signature: jaxtyping.PyTree
  config: transform_config_pb2.CompositeProcessorConfig

  @property
  def input_signature(self) -> jaxtyping.PyTree:
    return self._input_signature

  @property
  def output_signature(self) -> jaxtyping.PyTree:
    return self._output_signature

  def to_npt_saveable(self) -> saveable.Saveable:
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
    assets2npt.save_assets(self.asset_map, export_path)
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
  """Composite processor for Neptune."""

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
    self._processor_config = composite_processor.to_proto()
    self._asset_map = assets2npt.AssetMap(self._name)
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

    The processor may use its own assets. When it runs in the Neptune runtime,
    these assets will be loaded from the model export directory, so they need
    to be copied there during export. We use assets2npt to copy the assets and
    need to update the processor config to point to the new asset paths.
    """
    for resource in self._processor_config.resources:
      for asset in resource.assets:
        if asset.path in self._asset_map.assets:
          asset.path = os.path.join(
              self._asset_map.bundle_name,
              self._asset_map.assets[asset.path],
          )

  def prepare(
      self,
      input_signature: jaxtyping.PyTree,
      **kwargs: Any,
  ) -> PreparedCompositeProcessor:
    """Prepares the processor for export."""
    del kwargs  # unused

    jd_output_signature = self._get_output_signature(input_signature)

    npt_input_signature = jax.tree.map(
        lambda x: shlo_type.ShloTensorSpec(
            shape=x.shape, dtype=_convert_jd_type_to_shlo_type(x.dtype)
        ),
        input_signature,
    )

    npt_output_signature = jax.tree.map(
        lambda x: shlo_type.ShloTensorSpec(
            shape=x.shape, dtype=_convert_jd_type_to_shlo_type(x.dtype)
        ),
        jd_output_signature,
    )

    return PreparedCompositeProcessor(
        self._name,
        self._asset_map,
        npt_input_signature,
        npt_output_signature,
        self._processor_config,
    )
