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
import re

from google.protobuf import descriptor
import jax

import jaxlib

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import
from xla_saved_model.protos.xla.pjrt.proto import compile_options_pb2  # pylint: disable=g-direct-tensorflow-import

# A mapping between XLAflag names and protobuf field names.


def _generate_compilation_options(
    compile_environment,
    jax_mesh: jax.sharding.Mesh | None = None,
) -> compile_options_pb2.CompileOptionsProto:
  """Generates `CompileOptionsProto` for a single lowered target platform.

  Populates `ExecutableBuildOptionsProto` unconditionally across all target
  platforms (`cpu`, `cuda`, `tpu`) so that every serialized platform entry
  carries consistent single-replica SPMD and Shardy build metadata:
    * `num_replicas = 1` and `num_partitions = jax_mesh.size` (or `1` when
      `jax_mesh` is `None`), matching the device count recorded in
      `Manifest.device_assignment_by_coords` when exported with a `jax_mesh`.
      At load time, the serving runtime checks that `len(devices)` in
      `device_assignment_by_coords` equals `num_replicas * num_partitions`.
    * `device_ordinal = -1` and `process_count = 1` for single-host SPMD
      execution, plus an explicit single-replica `DeviceAssignmentProto` over
      `jax_mesh.devices.flat` whenever `jax_mesh` is provided.
    * `use_shardy_partitioner = True` paired with `[False]` for both
      `allow_spmd_sharding_propagation_to_parameters` and
      `allow_spmd_sharding_propagation_to_output`, which Shardy's input/output
      sharding adjustment helpers read whenever Shardy partitioning is enabled.

  Args:
    compile_environment: Optional `CompilationEnvironmentsProto` packed with
      platform-specific compiler flags (for example on TPU), or `None`.
    jax_mesh: Optional JAX `Mesh` describing the single-replica SPMD device
      topology. When `None`, defaults to 1 replica and 1 partition without an
      explicit `device_assignment`.

  Returns:
    A populated `CompileOptionsProto` for the target platform.
  """
  compile_options = compile_options_pb2.CompileOptionsProto()
  executable_build_options = compile_options_pb2.ExecutableBuildOptionsProto()

  default_opts = jaxlib.xla_client.CompileOptions()
  default_opts_proto = compile_options_pb2.CompileOptionsProto.FromString(
      default_opts.SerializeAsString()
  )
  executable_build_options.debug_options.CopyFrom(
      default_opts_proto.executable_build_options.debug_options
  )
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
  # Populate a single-replica XLA device assignment from `jax_mesh` when
  # provided so `num_partitions` and `device_assignment` match any manifest
  # `device_assignment_by_coords` across CPU, GPU, and TPU platforms.
  if jax_mesh is not None:
    partition = 0
    device_assignment = xla_data_pb2.DeviceAssignmentProto()
    device_assignment.replica_count = 1
    for device in jax_mesh.devices.flat:
      computation_device = (
          xla_data_pb2.DeviceAssignmentProto.ComputationDevice()
      )
      computation_device.replica_device_ids.append(device.id)
      device_assignment.computation_devices.append(computation_device)
      partition += 1
    # Reset num_partitions to the number of devices in the mesh.
    executable_build_options.num_partitions = partition
    device_assignment.computation_count = partition
    executable_build_options.device_assignment.CopyFrom(device_assignment)
  executable_build_options.use_shardy_partitioner = True
  compile_options.executable_build_options.CopyFrom(executable_build_options)
  return compile_options


def generate_xla_compile_options(
    native_serialization_platforms: Sequence[str] | None,
    xla_flags_per_platform: Mapping[str, Sequence[str]] | None = None,
    jax_mesh: jax.sharding.Mesh | None = None,
    persist_xla_flags: bool = True,
) -> manifest_pb2.CompileOptionsProtoMap:
  """Generates per-platform XLA `CompileOptionsProto` entries for export.

  Every target platform in `native_serialization_platforms` (`cpu`, `cuda`,
  `tpu`) receives a complete `ExecutableBuildOptionsProto` with
  single-replica SPMD and Shardy settings derived from `jax_mesh`, while
  platform-specific compiler flag overrides (such as TPU `comp_envs` or GPU
  `env_option_overrides`) are layered onto the corresponding platform entry.

  Args:
    native_serialization_platforms: Sequence of target platform names to
      populate compile options for. If `None`, defaults to TPU only (`['tpu']`).
    xla_flags_per_platform: Optional mapping from platform name to `--key=value`
      XLA flag overrides for that platform. Currently only `'tpu'` and `'cuda'`
      flag overrides are applied; `'cpu'` flag overrides are ignored.
    jax_mesh: Optional JAX mesh used for sharding. Populates `num_replicas = 1`,
      `num_partitions = jax_mesh.size`, and `device_assignment` on every target
      platform; when `None`, defaults to `num_replicas = 1` and `num_partitions
      = 1`.
    persist_xla_flags: Whether to persist platform-specific XLA flag overrides
      in the returned compile options.

  Returns:
    A `CompileOptionsProtoMap` keyed by lowercase platform name.

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
          f" {sorted(list(valid_platforms))}"
      )

  if xla_flags_per_platform:
    logging.info("Setting XLA flags per platform: %s", xla_flags_per_platform)
    # validate XLA flags provided are for a provided platform.
    for xla_platform in xla_flags_per_platform.keys():
      if xla_platform.lower() not in platforms:
        raise ValueError(
            f'Platform "{xla_platform}" used for XLA flags is not provided in'
            " the native_serialization_platforms. "
        )

  for platform in platforms:
    if xla_flags_per_platform:
      xla_flags_overrides = xla_flags_per_platform.get(platform, None)
      if xla_flags_overrides:
        _validate_xla_flags_setting(xla_flags_overrides, persist_xla_flags)
    else:
      xla_flags_overrides = None

    platform_lower = platform.lower()
    compile_environment = None

    compile_options = _generate_compilation_options(
        compile_environment,
        jax_mesh,
    )

    compile_options_map.map[platform_lower].CopyFrom(compile_options)

  if not persist_xla_flags:
    for compile_options in compile_options_map.map.values():
      compile_options.executable_build_options.comp_envs.Clear()
      compile_options.env_option_overrides.clear()
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
        "persist_xla_flags must be True if xla_flags_overrides are provided."
    )
