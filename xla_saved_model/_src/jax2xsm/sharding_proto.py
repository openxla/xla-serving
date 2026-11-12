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

"""Conversion between `jax.sharding.NamedSharding` and its proto form.

The manifest stores XLA `OpSharding` protos for serving runtimes, which lose
mesh axis names, axis types, and memory kinds. Recording `NamedSharding` (over
deduplicated `AbstractMesh` instances) in the JAX supplemental mirrors JAX's own
export serialization and lets `xsm2jax` reconstruct `jax.export.Exported` with
full sharding fidelity.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import jax

from xla_saved_model.protos import jax_supplemental_pb2

AbstractMeshProto = jax_supplemental_pb2.AbstractMesh
NamedShardingsProto = jax_supplemental_pb2.NamedShardings
_AxisTypeProto = jax_supplemental_pb2.AbstractMesh.AxisType

_AXIS_TYPE_TO_PROTO = {
    jax.sharding.AxisType.Auto: _AxisTypeProto.AUTO,
    jax.sharding.AxisType.Explicit: _AxisTypeProto.EXPLICIT,
    jax.sharding.AxisType.Manual: _AxisTypeProto.MANUAL,
}
_AXIS_TYPE_FROM_PROTO = {
    proto: axis for axis, proto in _AXIS_TYPE_TO_PROTO.items()
}


def _abstract_device_to_proto(
    device: jax.sharding.AbstractDevice,
) -> jax_supplemental_pb2.AbstractDevice:
  """Serializes a `jax.sharding.AbstractDevice`."""
  proto = jax_supplemental_pb2.AbstractDevice(
      device_kind=device.device_kind,
      platform=getattr(device, "platform", ""),
  )
  if device.num_cores is not None:
    proto.num_cores = device.num_cores
  return proto


def _abstract_device_from_proto(
    proto: jax_supplemental_pb2.AbstractDevice,
    device_cls: type[Any] = jax.sharding.AbstractDevice,
) -> Any:
  """Deserializes an `AbstractDevice` proto."""
  num_cores = proto.num_cores if proto.HasField("num_cores") else None
  if any(f.name == "platform" for f in dataclasses.fields(device_cls)):
    return device_cls(
        device_kind=proto.device_kind,
        num_cores=num_cores,
        platform=proto.platform,
    )
  return device_cls(
      device_kind=proto.device_kind,
      num_cores=num_cores,
  )


def abstract_mesh_to_proto(
    mesh: jax.sharding.AbstractMesh,
) -> AbstractMeshProto:
  """Serializes a `jax.sharding.AbstractMesh`."""
  proto = AbstractMeshProto(
      axis_sizes=mesh.axis_sizes,
      axis_names=mesh.axis_names,
      axis_types=[_AXIS_TYPE_TO_PROTO[t] for t in mesh.axis_types],
  )
  if mesh.abstract_device is not None:
    proto.abstract_device.CopyFrom(
        _abstract_device_to_proto(mesh.abstract_device)
    )
  return proto


def abstract_mesh_from_proto(
    proto: AbstractMeshProto,
) -> jax.sharding.AbstractMesh:
  """Deserializes an `AbstractMesh` proto.

  Args:
    proto: The serialized mesh.

  Returns:
    The reconstructed `jax.sharding.AbstractMesh`.

  Raises:
    ValueError: If axis fields disagree in length or contain an unknown axis
      type.
  """
  if not (
      len(proto.axis_sizes) == len(proto.axis_names) == len(proto.axis_types)
  ):
    raise ValueError(
        "AbstractMesh axis_sizes, axis_names, and axis_types must have the same"
        f" length; got {len(proto.axis_sizes)}, {len(proto.axis_names)}, and"
        f" {len(proto.axis_types)}."
    )
  axis_types = []
  for axis_type in proto.axis_types:
    if axis_type not in _AXIS_TYPE_FROM_PROTO:
      raise ValueError(f"Unknown AbstractMesh axis type: {axis_type}.")
    axis_types.append(_AXIS_TYPE_FROM_PROTO[axis_type])
  abstract_device = (
      _abstract_device_from_proto(proto.abstract_device)
      if proto.HasField("abstract_device")
      else None
  )
  return jax.sharding.AbstractMesh(
      tuple(proto.axis_sizes),
      tuple(proto.axis_names),
      tuple(axis_types),
      abstract_device=abstract_device,
  )


def _partition_spec_axis_to_proto(
    axis: Any,
) -> jax_supplemental_pb2.PartitionSpecAxis:
  """Serializes a single dimension entry of a `PartitionSpec`."""
  if axis is None:
    return jax_supplemental_pb2.PartitionSpecAxis()
  if axis is jax.sharding.PartitionSpec.UNCONSTRAINED:
    return jax_supplemental_pb2.PartitionSpecAxis(unconstrained=True)
  if isinstance(axis, str):
    return jax_supplemental_pb2.PartitionSpecAxis(axes=[axis])
  return jax_supplemental_pb2.PartitionSpecAxis(axes=list(axis))


def _partition_spec_axis_from_proto(
    proto: jax_supplemental_pb2.PartitionSpecAxis,
) -> Any:
  """Deserializes a single dimension entry of a `PartitionSpec`."""
  if proto.unconstrained:
    if proto.axes:
      raise ValueError(
          "PartitionSpecAxis cannot set both `unconstrained` and `axes` (got"
          f" axes={list(proto.axes)!r})."
      )
    return jax.sharding.PartitionSpec.UNCONSTRAINED
  if not proto.axes:
    return None
  if len(proto.axes) == 1:
    return proto.axes[0]
  return tuple(proto.axes)


def _partition_spec_to_proto(
    spec: jax.sharding.PartitionSpec,
) -> jax_supplemental_pb2.PartitionSpec:
  """Serializes a `jax.sharding.PartitionSpec`."""
  # Indexing or iterating a `PartitionSpec` raises `ValueError` when `reduced`
  # or `unreduced` is non-empty, and the public `.partitions` property was added
  # after JAX 0.10.0, so read `_partitions` directly as JAX's serializer does.
  partitions = spec._partitions  # pylint: disable=protected-access
  return jax_supplemental_pb2.PartitionSpec(
      partitions=[_partition_spec_axis_to_proto(axis) for axis in partitions],
      reduced=sorted(spec.reduced),
      unreduced=sorted(spec.unreduced),
  )


def _partition_spec_from_proto(
    proto: jax_supplemental_pb2.PartitionSpec,
) -> jax.sharding.PartitionSpec:
  """Deserializes a `PartitionSpec` proto."""
  partitions = tuple(
      _partition_spec_axis_from_proto(axis) for axis in proto.partitions
  )
  return jax.sharding.PartitionSpec(
      *partitions,
      reduced=frozenset(proto.reduced),
      unreduced=frozenset(proto.unreduced),
  )


def to_protos(
    in_shardings: Sequence[jax.sharding.NamedSharding | None] | None,
    out_shardings: Sequence[jax.sharding.NamedSharding | None] | None,
) -> tuple[
    list[AbstractMeshProto],
    NamedShardingsProto | None,
    NamedShardingsProto | None,
]:
  """Serializes input and output `NamedSharding` sequences with shared meshes.

  When every entry in a sequence is `None` (or the sequence itself is `None` or
  empty), the corresponding `NamedShardings` proto is omitted (`None`) so
  unsharded artifacts remain unchanged.

  Args:
    in_shardings: Flattened input leaf shardings, or `None`.
    out_shardings: Flattened output leaf shardings, or `None`.

  Returns:
    `(abstract_meshes, input_named_shardings, output_named_shardings)`.
  """
  mesh_to_index: dict[jax.sharding.AbstractMesh, int] = {}
  abstract_mesh_protos: list[AbstractMeshProto] = []

  def _encode_sequence(
      shardings: Sequence[jax.sharding.NamedSharding | None] | None,
  ) -> NamedShardingsProto | None:
    if not shardings:
      return None
    idx_to_sharding: dict[int, jax_supplemental_pb2.NamedSharding] = {}
    for idx, sharding in enumerate(shardings):
      if sharding is None:
        continue
      abstract_mesh = sharding.mesh.abstract_mesh
      mesh_index = mesh_to_index.get(abstract_mesh)
      if mesh_index is None:
        mesh_index = len(abstract_mesh_protos)
        mesh_to_index[abstract_mesh] = mesh_index
        abstract_mesh_protos.append(abstract_mesh_to_proto(abstract_mesh))
      sharding_proto = jax_supplemental_pb2.NamedSharding(
          abstract_mesh_index=mesh_index,
          spec=_partition_spec_to_proto(sharding.spec),
      )
      if sharding.memory_kind is not None:
        sharding_proto.memory_kind = sharding.memory_kind
      idx_to_sharding[idx] = sharding_proto
    if not idx_to_sharding:
      return None
    return NamedShardingsProto(idx_to_sharding=idx_to_sharding)

  in_proto = _encode_sequence(in_shardings)
  out_proto = _encode_sequence(out_shardings)
  return abstract_mesh_protos, in_proto, out_proto


def from_proto(
    proto: NamedShardingsProto | None,
    abstract_meshes: Sequence[jax.sharding.AbstractMesh],
    num_leaves: int,
) -> tuple[jax.sharding.NamedSharding | None, ...]:
  """Deserializes a `NamedShardings` proto into a per-leaf tuple.

  Args:
    proto: The serialized leaf shardings, or `None` when none were recorded.
    abstract_meshes: The deserialized `Function.abstract_meshes` entries.
    num_leaves: Number of flattened leaves in the corresponding signature.

  Returns:
    A tuple of length `num_leaves` holding each leaf's `NamedSharding` or
    `None`.

  Raises:
    ValueError: If a leaf index or `abstract_mesh_index` is out of bounds, or a
      `PartitionSpec` entry is invalid.
  """
  if proto is None:
    return (None,) * num_leaves
  result: list[jax.sharding.NamedSharding | None] = [None] * num_leaves
  for idx, sharding_proto in sorted(proto.idx_to_sharding.items()):
    if idx >= num_leaves:
      raise ValueError(
          f"NamedSharding leaf index {idx} is out of range for signature with"
          f" {num_leaves} leaves."
      )
    mesh_index = sharding_proto.abstract_mesh_index
    if mesh_index >= len(abstract_meshes):
      raise ValueError(
          f"NamedSharding abstract_mesh_index {mesh_index} is out of range for"
          f" {len(abstract_meshes)} recorded abstract meshes."
      )
    spec = _partition_spec_from_proto(sharding_proto.spec)
    memory_kind = (
        sharding_proto.memory_kind
        if sharding_proto.HasField("memory_kind")
        else None
    )
    result[idx] = jax.sharding.NamedSharding(
        abstract_meshes[mesh_index], spec, memory_kind=memory_kind
    )
  return tuple(result)
