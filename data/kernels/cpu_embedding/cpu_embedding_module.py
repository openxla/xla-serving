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

"""Python wrappers for CPU Embedding lookups."""

from collections.abc import Mapping
import dataclasses
import numbers
import os
import tempfile
from typing import Any, Sequence

from google.protobuf import text_format
import jax
import jaxtyping
import numpy as np

from data.kernels.cpu_embedding import cpu_embedding_lookup_transform_pybind
from data.kernels.cpu_embedding import cpu_embedding_pb2
from data.kernels.cpu_embedding import cpu_embedding_resource_pybind
from data.kernels.cpu_embedding import cpu_embedding_serializer_pybind
from data.processor.python import composite_processor_pybind
from data.resource_manager.python import resource_manager

MANIFEST_ASSET_NAME = cpu_embedding_serializer_pybind.MANIFEST_ASSET_NAME
serialize = cpu_embedding_serializer_pybind.serialize
_COMBINE_MODE_ENUM = (
    cpu_embedding_pb2.CpuEmbeddingManifest.EmbeddingTable.CombineMode
)
_COMBINE_MODES = {
    "SUM": _COMBINE_MODE_ENUM.SUM,
    "MEAN": _COMBINE_MODE_ENUM.MEAN,
    "SQRT_N": _COMBINE_MODE_ENUM.SQRT_N,
}


def _combine_mode(
    mode: str,
) -> cpu_embedding_pb2.CpuEmbeddingManifest.EmbeddingTable.CombineMode:
  try:
    return _COMBINE_MODES[mode]
  except KeyError as error:
    raise ValueError(f"Unknown combine mode: {mode}") from error


def _build_manifest(
    table_shapes: Mapping[str, Sequence[int]],
    combine_modes: Mapping[str, str],
    feature_specs: Sequence[dict[str, Any]],
    *,
    max_cpu_slots: int = 8,
    block_size: int = 64,
) -> tuple[cpu_embedding_pb2.CpuEmbeddingManifest, dict[str, int]]:
  manifest = cpu_embedding_pb2.CpuEmbeddingManifest()
  manifest.max_cpu_slots = max_cpu_slots
  manifest.block_size = block_size
  table_dims = {}
  for table_name, shape in table_shapes.items():
    if (
        not isinstance(table_name, str)
        or not table_name
        or "/" in table_name
        or "\\" in table_name
        or ".." in table_name
        or table_name == MANIFEST_ASSET_NAME
    ):
      raise ValueError(
          "Invalid table name (empty, reserved, or path traversal): "
          f"{table_name!r}"
      )
    table = manifest.tables.add()
    table.name = table_name
    table.vocab_size = shape[0]
    table.embedding_dim = shape[1]
    table.combine_mode = _combine_mode(combine_modes.get(table_name, "MEAN"))
    table_dims[table_name] = int(shape[1])

  embedding_dims = {}
  for spec_dict in feature_specs:
    feature = manifest.feature_specs.add()
    feature.name = spec_dict["name"]
    feature.table = spec_dict["table_name"]
    if "max_sequence_length" in spec_dict:
      feature.max_sequence_length = spec_dict["max_sequence_length"]
    if feature.table in table_dims:
      embedding_dims[feature.name] = table_dims[feature.table]
  return manifest, embedding_dims


@dataclasses.dataclass(frozen=True)
class MappedEmbeddingTable:
  """A raw row-major, little-endian FP32 table mapped read-only at load time.

  Replacing or unlinking the path does not retarget the mapping. Modifying or
  truncating the mapped file is unsupported.
  """

  path: str | os.PathLike[str]
  shape: tuple[int, int]


class CpuEmbeddingResource:
  """Python wrapper for CpuEmbeddingEngineResource.

  In-memory tables and their manifest are serialized in a temporary directory.
  Mapped tables retain their source paths; only their manifest is temporary.

  Args:
    table_data: A mapping from table names to their numpy arrays containing the
      embedding weights (usually float32).
    combine_modes: A mapping from table names to their aggregation mode string.
      Supported modes typically include "MEAN", "SUM", and "SQRT_N".
    feature_specs: A list of dictionaries defining the lookup features. Each
      dictionary must contain "name", "table_name", and optionally
      "max_sequence_length".
    max_cpu_slots: Number of threads in the CPU lookup worker pool.
    block_size: Only legacy single-feature in-place lookups use this setting.
      The grouped `CpuEmbeddingModule` path and the legacy
      `DenseCpuEmbeddingLookupTransform` API ignore it.

  Example usage:

  ```python
  table_data = {
      "table1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
  }
  combine_modes = {"table1": "MEAN"}
  feature_specs = [{
      "name": "my_feature",
      "table_name": "table1",
      "max_sequence_length": 5,  # Optional argument.
  }]

  resource = CpuEmbeddingResource(table_data, combine_modes, feature_specs)
  ```
  """

  def __init__(
      self,
      table_data: dict[str, np.ndarray],
      combine_modes: dict[str, str],
      feature_specs: Sequence[dict[str, Any]],
      max_cpu_slots: int = 8,
      block_size: int = 64,
  ) -> None:
    manifest, self._embedding_dims = _build_manifest(
        {name: data.shape for name, data in table_data.items()},
        combine_modes,
        feature_specs,
        max_cpu_slots=max_cpu_slots,
        block_size=block_size,
    )

    first_table_data = next(iter(table_data.values()))
    self._embedding_dim = first_table_data.shape[1]

    self._temp_dir = tempfile.TemporaryDirectory(prefix="jce_resource_")
    try:
      self._assets = cpu_embedding_serializer_pybind.serialize(
          self._temp_dir.name,
          table_data,
          text_format.MessageToString(manifest),
      )
      self._resource = cpu_embedding_resource_pybind.CpuEmbeddingEngineResource(
          self._assets
      )
      # Load eagerly to validate the engine.
      self._resource.load()
    except Exception:
      self._temp_dir.cleanup()
      raise

  @classmethod
  def from_mapped_tables(
      cls,
      table_files: Mapping[str, MappedEmbeddingTable],
      combine_modes: Mapping[str, str],
      feature_specs: Sequence[dict[str, Any]],
      max_cpu_slots: int = 8,
      block_size: int = 64,
  ) -> "CpuEmbeddingResource":
    """Creates a resource backed by read-only mapped FP32 tables.

    The C++ loader validates each file against its declared shape. Only the
    generated manifest is stored in the resource's temporary directory.
    """
    if not table_files:
      raise ValueError("At least one mapped embedding table is required.")

    normalized_tables: list[tuple[str, str, tuple[int, int]]] = []
    for table_name, descriptor in table_files.items():
      if not isinstance(descriptor, MappedEmbeddingTable):
        raise TypeError(
            "Mapped tables must use MappedEmbeddingTable descriptors."
        )

      try:
        raw_shape = tuple(descriptor.shape)
      except TypeError as error:
        raise ValueError(
            f"Table {table_name!r} must have an exact 2-D shape."
        ) from error
      if len(raw_shape) != 2:
        raise ValueError(f"Table {table_name!r} must have an exact 2-D shape.")
      if any(
          isinstance(dim, (bool, np.bool_))
          or not isinstance(dim, numbers.Integral)
          or dim <= 0
          for dim in raw_shape
      ):
        raise ValueError(
            f"Table {table_name!r} dimensions must be positive integers."
        )
      shape = (int(raw_shape[0]), int(raw_shape[1]))

      table_path = os.path.abspath(os.fspath(descriptor.path))
      if not isinstance(table_path, str):
        raise TypeError(f"Table {table_name!r} path must resolve to a string.")

      normalized_tables.append((table_name, table_path, shape))

    manifest, embedding_dims = _build_manifest(
        {name: shape for name, _, shape in normalized_tables},
        combine_modes,
        feature_specs,
        max_cpu_slots=max_cpu_slots,
        block_size=block_size,
    )

    instance = cls.__new__(cls)
    instance._temp_dir = tempfile.TemporaryDirectory(prefix="jce_resource_")
    try:
      instance._assets = list(
          serialize(
              instance._temp_dir.name,
              {},
              text_format.MessageToString(manifest),
          )
      )
      instance._assets.extend(
          resource_manager.Asset(path=path, name=name)
          for name, path, _ in normalized_tables
      )
      instance._embedding_dims = embedding_dims
      instance._embedding_dim = normalized_tables[0][2][1]
      instance._resource = (
          cpu_embedding_resource_pybind.CpuEmbeddingEngineResource(
              instance._assets
          )
      )
      instance._resource.load()
    except Exception:
      instance._temp_dir.cleanup()
      raise
    return instance

  def get_embedding_dim(self, feature_name: str) -> int:
    if feature_name not in self._embedding_dims:
      raise KeyError(f"Feature '{feature_name}' not found.")
    return self._embedding_dims[feature_name]

  @property
  def resource(
      self,
  ) -> cpu_embedding_resource_pybind.CpuEmbeddingEngineResource:
    return self._resource

  @property
  def assets(self) -> Sequence[resource_manager.Asset]:
    return self._assets

  @property
  def embedding_dim(self) -> int:
    return self._embedding_dim

  def export_assets(self) -> dict[str, str]:
    assets_dict = {}
    for asset in self._assets:
      assets_dict["assets/" + asset.name] = asset.path
    return assets_dict

  def close(self) -> None:
    self._temp_dir.cleanup()


class CpuEmbeddingModule:
  """Python transform module wrapper for CPU Embedding lookups.

  This class creates a processing module that performs embedding lookups
  using a configured `CpuEmbeddingResource`.

  Args:
    resource: The `CpuEmbeddingResource` containing the initialized CPU
      embedding engine, tables, and lookup configurations.
    feature_names: A string or list of strings indicating the input feature
      names (usually containing integer IDs) to perform the lookup on.
    output_feature_names: A string or list of strings indicating the names
      assigned to the resulting embedding features.
    output_shapes: A sequence of dimensions (for a single output) or a list of
      sequences of dimensions (for multiple outputs) defining the expected
      shapes of the resulting embeddings.
    dense_lookup: A boolean indicating whether to perform dense lookups.
      Defaults to False (sparse lookup).

  Note: `feature_name`, `output_feature_name`, and `output_shape` are accepted
  for backward compatibility with single-feature setups.

  Example usage:

  ```python
  # Setup the resource (with the loaded tables/manifest)
  resource = CpuEmbeddingResource(table_data, combine_modes, feature_specs)

  # Create the CPU lookup module
  module = CpuEmbeddingModule(
      resource=resource,
      feature_names=["user_ids", "item_ids"],
      output_feature_names=["user_emb", "item_emb"],
      output_shapes=[[32], [64]],
  )
  ```
  """

  def __init__(
      self,
      resource: CpuEmbeddingResource,
      feature_names: str | Sequence[str] | None = None,
      output_feature_names: str | Sequence[str] | None = None,
      output_shapes: Sequence[int] | Sequence[Sequence[int]] | None = None,
      dense_lookup: bool = False,
      embedding_dim: int | None = None,
      # For backwards compatibility with keyword arguments:
      feature_name: str | None = None,
      output_feature_name: str | None = None,
      output_shape: Sequence[int] | None = None,
  ):
    super().__init__()
    self._resource_wrapper = resource

    resolved_features = feature_names or feature_name
    resolved_outputs = output_feature_names or output_feature_name
    resolved_shapes = output_shapes or output_shape
    resolved_embedding_dim = (
        embedding_dim if embedding_dim is not None else resource.embedding_dim
    )

    if resolved_features is None:
      raise ValueError("Must provide feature_names or feature_name.")
    if resolved_outputs is None:
      raise ValueError(
          "Must provide output_feature_names or output_feature_name."
      )
    if resolved_shapes is None:
      raise ValueError("Must provide output_shapes or output_shape.")

    # Normalize features and outputs to lists of strings
    if isinstance(resolved_features, str):
      self._feature_names = [resolved_features]
    else:
      self._feature_names = list(resolved_features)

    if isinstance(resolved_outputs, str):
      self._output_feature_names = [resolved_outputs]
    else:
      self._output_feature_names = list(resolved_outputs)

    # Normalize shapes to a list of lists of integers
    if len(self._feature_names) == 1:
      if not resolved_shapes:
        self._output_shapes = [[]]
      elif isinstance(resolved_shapes[0], int):
        self._output_shapes = [list(resolved_shapes)]  # pytype: disable=wrong-arg-types
      else:
        self._output_shapes = [list(s) for s in resolved_shapes]  # pytype: disable=wrong-arg-types
    else:
      self._output_shapes = [list(s) for s in resolved_shapes]  # pytype: disable=wrong-arg-types

    # Assert sequence lengths are equal
    if len(self._feature_names) != len(self._output_feature_names) or len(
        self._feature_names
    ) != len(self._output_shapes):
      raise ValueError(
          "Sequence length mismatch: "
          f"feature_names ({len(self._feature_names)}), "
          f"output_feature_names ({len(self._output_feature_names)}), "
          f"output_shapes ({len(self._output_shapes)})."
      )

    self._transform = (
        cpu_embedding_lookup_transform_pybind.CpuEmbeddingLookupTransform(
            resource.resource,
            self._feature_names,
            dense_lookup,
            self._output_feature_names,
            resolved_embedding_dim,
            [  # pyrefly: ignore[bad-argument-type]
                [d if d is not None else -1 for d in shape]
                for shape in self._output_shapes
            ],
        )
    )

  @property
  def transform(self) -> composite_processor_pybind.TransformBase:
    return self._transform

  def export_assets(self) -> dict[str, str]:
    return self._resource_wrapper.export_assets()

  def get_output_signature(
      self, input_signature: jaxtyping.PyTree
  ) -> jaxtyping.PyTree:
    """Returns the output PyTree signature for the preprocessor transform."""
    del input_signature
    output_sig = {}
    for name, shape in zip(self._output_feature_names, self._output_shapes):
      output_sig[name] = jax.ShapeDtypeStruct(
          shape=tuple(shape), dtype=jax.numpy.float32
      )

    # Unflatten slash-delimited keys into a nested dictionary PyTree structure
    nested_sig = {}
    for key, value in output_sig.items():
      parts = key.split("/")
      d = nested_sig
      for part in parts[:-1]:
        if part not in d:
          d[part] = {}
        d = d[part]
      d[parts[-1]] = value

    return nested_sig
