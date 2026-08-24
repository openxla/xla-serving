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

"""Python ops and modules related to CpuEmbeddingLookupTransform."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import jax

from data.kernels.cpu_embedding import cpu_embedding_lookup_transform_pybind
from data.kernels.cpu_embedding import cpu_embedding_module
from data.kernels.cpu_embedding import dense_cpu_embedding_lookup_transform_pybind
from data.processor.python import composite_processor_pybind
from data.processor.python import transform

MANIFEST_ASSET_NAME = cpu_embedding_module.MANIFEST_ASSET_NAME
serialize = cpu_embedding_module.serialize

CpuEmbeddingResource = cpu_embedding_module.CpuEmbeddingResource
MappedEmbeddingTable = cpu_embedding_module.MappedEmbeddingTable
CpuEmbeddingModule = cpu_embedding_module.CpuEmbeddingModule


class DenseCpuEmbeddingLookupTransform(transform.ExportableTransform):
  """Transform for direct dense CPU embedding lookup (pure tensor IO).

  Supports both single-table and multi-table parallel lookup on the C++
  threadpool.
  """

  def __init__(
      self,
      resource: cpu_embedding_module.CpuEmbeddingResource,
      feature_names: Union[str, Sequence[str]],
      pad_value: int = 0,
      embedding_dim: Optional[Union[int, Sequence[int]]] = None,
      output_feature_names: Optional[Sequence[str]] = None,
  ):
    super().__init__()
    self._resource_wrapper = resource
    if isinstance(feature_names, str):
      self._feature_names = [feature_names]
      self._is_single_feature = True
    else:
      self._feature_names = list(feature_names)
      self._is_single_feature = False

    self._pad_value = pad_value
    self._output_feature_names = (
        list(output_feature_names) if output_feature_names is not None else []
    )

    if embedding_dim is not None:
      if isinstance(embedding_dim, int):
        self._embedding_dims = [embedding_dim] * len(self._feature_names)
      else:
        self._embedding_dims = list(embedding_dim)
    else:
      self._embedding_dims = [
          resource.get_embedding_dim(f)
          if hasattr(resource, "get_embedding_dim")
          else resource.embedding_dim
          for f in self._feature_names
      ]

    self._transform = dense_cpu_embedding_lookup_transform_pybind.DenseCpuEmbeddingLookupTransform(
        resource.resource,
        self._feature_names,
        self._pad_value,
        self._output_feature_names,
    )

  def initialize(self, input_signature: Any) -> None:
    pass

  def export_assets(self) -> dict[str, str]:
    return self._resource_wrapper.export_assets()

  def get_output_signature(self, input_signature: Any) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten(input_signature)
    if not leaves:
      return input_signature

    out_specs = []
    for i, leaf in enumerate(leaves):
      shape = list(leaf.shape)
      batch_size = shape[0] if shape else -1
      emb_dim = (
          self._embedding_dims[i]
          if i < len(self._embedding_dims)
          else self._embedding_dims[0]
      )
      out_specs.append(
          jax.ShapeDtypeStruct(
              shape=(batch_size, emb_dim), dtype=jax.numpy.float32
          )
      )
    return jax.tree_util.tree_unflatten(treedef, out_specs)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      flat_inputs = list(args[0])
    elif len(args) == 1:
      flat_inputs = [args[0]]
    else:
      flat_inputs = list(args)

    result = super().__call__(flat_inputs)
    if (
        self._is_single_feature
        and isinstance(result, (list, tuple))
        and len(result) == 1
    ):
      return result[0]
    return result


class CpuEmbeddingLookupTransformWrapper(transform.ExportableTransform):
  """Transform wrapping C++ CpuEmbeddingLookupTransform for multi-table vector IO."""

  def __init__(
      self,
      resource: cpu_embedding_module.CpuEmbeddingResource,
      feature_name: str,
      embedding_dim: int,
      output_feature_name: str,
      dense_lookup: bool = True,
  ):
    super().__init__()
    self._resource_wrapper = resource
    self._feature_name = feature_name
    self._output_feature_name = output_feature_name
    self._embedding_dim = embedding_dim
    self._dense_lookup = dense_lookup

    self._transform = (
        cpu_embedding_lookup_transform_pybind.CpuEmbeddingLookupTransform(
            resource.resource,
            [self._feature_name],
            self._dense_lookup,
            [self._output_feature_name],
            self._embedding_dim,
            [[-1, self._embedding_dim]],
        )
    )

  def initialize(self, input_signature: Any) -> None:
    pass

  def export_assets(self) -> dict[str, str]:
    return self._resource_wrapper.export_assets()

  def get_output_signature(self, input_signature: Any) -> Any:
    return composite_processor_pybind.TypeInfo.host_buffer_vector()


def cpu_embedding(
    input_tensors: Any,
    resource: cpu_embedding_module.CpuEmbeddingResource,
    feature_name: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    pad_value: int = 0,
    embedding_dim: Optional[Union[int, Sequence[int]]] = None,
) -> Any:
  """Functional op for direct dense CPU embedding lookup.

  Supports both single-table and multi-table parallel lookups.

  Args:
    input_tensors: A single HostBuffer/JAX array or sequence of HostBuffers.
    resource: CpuEmbeddingResource containing loaded embedding tables.
    feature_name: Single feature name (used when looking up 1 table).
    feature_names: Sequence of feature names (used when looking up N tables in
      parallel).
    pad_value: Padding value to ignore during reduction and use for empty rows
      (default 0).
    embedding_dim: Optional embedding dimension(s).

  Returns:
    A single embedding tensor (if 1 feature) or a list of embedding tensors (if
    multiple features).
  """
  if feature_names is None:
    if feature_name is None:
      raise ValueError("Either feature_name or feature_names must be provided.")
    names_arg: Union[str, Sequence[str]] = feature_name
    is_multi = False
  else:
    names_arg = list(feature_names)
    is_multi = True

  t = DenseCpuEmbeddingLookupTransform(
      resource=resource,
      feature_names=names_arg,
      pad_value=pad_value,
      embedding_dim=embedding_dim,
  )
  if is_multi:
    if isinstance(input_tensors, (list, tuple)):
      return t(list(input_tensors))
    return t([input_tensors])
  return t(input_tensors)


__all__ = [
    "CpuEmbeddingResource",
    "MappedEmbeddingTable",
    "CpuEmbeddingModule",
    "DenseCpuEmbeddingLookupTransform",
    "CpuEmbeddingLookupTransformWrapper",
    "cpu_embedding",
    "MANIFEST_ASSET_NAME",
    "serialize",
]
