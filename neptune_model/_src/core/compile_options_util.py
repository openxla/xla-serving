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

"""Utility functions for XLA compile options."""

from collections.abc import Mapping, Sequence
import logging
from typing import Any

from absl import flags
from google.protobuf import descriptor
from google.protobuf import text_format
import jax

from neptune_model.protos import manifest_pb2
from neptune_model.protos.xla import xla_data_pb2
from neptune_model.protos.xla.pjrt.proto import compile_options_pb2

# A mapping between XLAflag names and protobuf field names.


def _generate_compilation_options(
    compile_environment,
    jax_mesh: jax.sharding.Mesh | None = None,
) -> compile_options_pb2.CompileOptionsProto:
  """Generates the compilation options for the given compilation environment."""
  compile_options = compile_options_pb2.CompileOptionsProto()
  executable_build_options = compile_options_pb2.ExecutableBuildOptionsProto()
  if compile_environment is not None:
    executable_build_options.comp_envs.CopyFrom(compile_environment)
  executable_build_options.num_replicas = 1
  executable_build_options.num_partitions = 1
  executable_build_options.device_ordinal = -1
  executable_build_options.process_count = 1
  executable_build_options.allow_spmd_sharding_propagation_to_parameters.append(
      False
  )
  executable_build_options.allow_spmd_sharding_propagation_to_output.append(
      False
  )
  # Set the device assignment to a single-replica. If the jax mesh is
  # provided, we will set the device assignment to the jax mesh.
  if jax_mesh is not None:
    partition = 0
    device_assignment = xla_data_pb2.DeviceAssignmentProto()
    device_assignment.replica_count = 1
    for d in jax_mesh.devices.flat:
      computation_device = (
          xla_data_pb2.DeviceAssignmentProto.ComputationDevice()
      )
      computation_device.replica_device_ids.append(d.id)
      device_assignment.computation_devices.append(computation_device)
      partition += 1
    # Reset num_partitions to the number of devices in the mesh.
    executable_build_options.num_partitions = partition
    device_assignment.computation_count = partition
    executable_build_options.device_assignment.CopyFrom(device_assignment)
  executable_build_options.use_shardy_partitioner = (
      jax.config.jax_use_shardy_partitioner
  )
  compile_options.executable_build_options.CopyFrom(executable_build_options)
  return compile_options


def generate_xla_compile_options(
    native_serialization_platforms: Sequence[str] | None,
    xla_flags_per_platform: Mapping[str, Sequence[str]] | None = None,
    jax_mesh: jax.sharding.Mesh | None = None,
    persist_xla_flags: bool = True,
) -> manifest_pb2.CompileOptionsProtoMap:
  """Sets the XLA compilation options.

  Args:
    native_serialization_platforms: A sequence of platform names that the
      compile options will be set for. If None, the compile options will be set
      for TPU only.
    xla_flags_per_platform: A mapping from platform name to a list of xla flags
      which will be used to override the default XLA compilation flags.
    jax_mesh: The JAX mesh used for sharding. If None, the compile options will
      be set for a default single-replica.
    persist_xla_flags: Whether to persist XLA flags in the compile options.

  Returns:
    A `CompileOptionsProtoMap` containing the XLA compilation options per
    platform.

  Raises:
    ValueError: If an unknown platform is provided as a native serialization
      platform.
    ValueError: If an unknown platform is provided as a platform for XLA flags.
    ValueError: If a platform is provided for XLA flags which is not provided
      in the native serialization platforms.
    ValueError: If the supplied XLA flag overrides cannot be parsed.
    ValueError: If `xla_flags` are provided but `persist_xla_flags` is False.
      This ensures that the XLA flags are persisted in the compile options,
      otherwise they would be lost, leading to unexpected behavior.
  """
  tpu_platform_name = manifest_pb2.Platform.Name(
      manifest_pb2.Platform.TPU
  ).lower()
  compile_options_map = manifest_pb2.CompileOptionsProtoMap()
  if native_serialization_platforms is None:
    # If no native serialization platforms are specified, we will set the
    # compilation environment for TPU only.
    platforms = [tpu_platform_name]
  else:
    platforms = native_serialization_platforms

  # Validate the platform provided are valid.
  valid_platforms = {
      p.lower() for p in manifest_pb2.Platform.DESCRIPTOR.values_by_name
  }
  for platform in platforms:
    if platform.lower() not in valid_platforms:
      raise ValueError(
          f'Platform "{platform}" is not a valid platform. Valid platforms are:'
          f' {sorted(list(valid_platforms))}'
      )

  if xla_flags_per_platform:
    logging.info('Setting XLA flags per platform: %s', xla_flags_per_platform)
    # validate XLA flags provided are for a provided platform.
    for xla_platform in xla_flags_per_platform.keys():
      if xla_platform.lower() not in platforms:
        raise ValueError(
            f'Platform "{xla_platform}" used for XLA flags is not provided in'
            ' the native_serialization_platforms. '
        )

  for platform in platforms:
    compile_environment = None
    compile_options_map.map[platform.lower()].CopyFrom(
        _generate_compilation_options(compile_environment, jax_mesh)
    )
  if not persist_xla_flags:
    for compile_options in compile_options_map.map.values():
      compile_options.executable_build_options.comp_envs.Clear()
  return compile_options_map


def _validate_xla_flags_setting(
    xla_flags_overrides: Sequence[str] | None, persist_xla_flags: bool
) -> None:
  """Validates the XLA flags setting.

  XLA flag overrides are allowed only when XLA flags are required to be
  persisted.

  Args:
    xla_flags_overrides: A sequence of XLA flags provided for overriding. Can be
      None.
    persist_xla_flags: A boolean indicating whether the XLA flags should be
      persisted in the compile options.

  Raises:
    ValueError: If `xla_flags_overrides` is not None but `persist_xla_flags` is
    False.
  """
  if xla_flags_overrides and not persist_xla_flags:
    raise ValueError(
        'persist_xla_flags must be True if xla_flags_overrides are provided.'
    )
