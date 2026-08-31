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

"""Neptune Offline AOT Compiler Factory."""

from collections.abc import Callable, Iterator, Sequence
import copy
import dataclasses
from typing import Any

from absl import logging
import jax
from jax.experimental import topologies
import jax.extend as jex
from jax.interpreters import mlir

import jaxlib

from neptune_model._src.core import (
    manifest_utils,
)

try:
  from neptune.runtime.python import optimizer as neptune_optimizer  # pylint: disable=g-import-not-at-top
except ImportError:
  neptune_optimizer = None

if neptune_optimizer is not None:
  OptimizerType = (
      neptune_optimizer.StablehloOptimizer
      | Callable[[bytes, Sequence[int]], bytes]
  )
else:
  OptimizerType = Any

_SUPPORTED_PLATFORMS = frozenset({"cpu", "cuda", "tpu"})


@dataclasses.dataclass(frozen=True)
class TargetConfig:
  """Configuration for a single offline AOT compilation target topology.

  Attributes:
    target_topology_label: Identifier for the compiled artifact saved in the
      manifest (matches CompiledExecutableBody.target_topology_label in
      manifest.proto).
    platform: Name of the target platform ("cuda", "tpu", "cpu"). Note: 'gpu' is
      not supported; use 'cuda' directly.
    enable_deviceless_compilation: Whether to perform compilation without
      physical devices attached.
    serialized_topology: Optional serialized topology bytes required for
      deviceless compilation.
    autotune_database_path: Optional path to the CUDA autotune database.
    target_topology: Optional JAX topology mesh specification string (e.g.
      "2x2x2").
    compile_portable_executable: Optional override to compile as a portable
      executable.
    optimizer: Optional Neptune StableHLO optimizer instance or custom callable
      hook.
    compile_options: Optional pre-configured `xla_client.CompileOptions`
      override.
  """

  target_topology_label: str
  platform: str
  enable_deviceless_compilation: bool = False
  serialized_topology: bytes | None = None
  autotune_database_path: str | None = None
  target_topology: str | None = None
  compile_portable_executable: bool | None = None
  optimizer: OptimizerType | None = None
  compile_options: Any = None


def _validate_target_config(config: TargetConfig) -> TargetConfig:
  """Validates a single TargetConfig instance fail-fast upfront."""
  platform = config.platform.lower().strip()
  if platform not in _SUPPORTED_PLATFORMS:
    if platform == "gpu":
      raise ValueError(
          f"Unsupported platform '{config.platform}'. Did you mean 'cuda'?"
          " Neptune requires concrete platform names (e.g. 'cuda', 'tpu',"
          " 'cpu')."
      )
    raise ValueError(
        f"Unsupported platform '{config.platform}'. Supported platforms:"
        f" {sorted(_SUPPORTED_PLATFORMS)}"
    )

  if config.enable_deviceless_compilation:
    if config.serialized_topology is None and config.target_topology is None:
      raise ValueError(
          f"Deviceless compilation for platform '{platform}' requires either"
          " `serialized_topology` or `target_topology`."
      )
    if platform == "cuda" and not config.autotune_database_path:
      raise ValueError(
          "autotune_database_path must be provided for CUDA deviceless"
          " compilation to guarantee determinism."
      )

  return config


def _get_compile_fn(
    client: Any,
    enable_deviceless_compilation: bool,
) -> Callable[..., Any]:
  """Returns the compilation method for the given execution mode."""
  if enable_deviceless_compilation:
    return client.compile
  if hasattr(client, "compile_and_load"):
    return client.compile_and_load
  return client.compile


def _apply_optimizer(
    mlir_module_serialized: bytes,
    weight_indices: Sequence[int],
    optimizer: OptimizerType,
) -> bytes:
  """Dispatches optimization to C++ StablehloOptimizer or callable."""
  if isinstance(optimizer, type):
    raise TypeError(
        "Expected an optimizer instance or callable, but got class "
        f"'{optimizer.__name__}'. Did you forget to instantiate it?"
    )
  if neptune_optimizer is not None and isinstance(
      optimizer, neptune_optimizer.StablehloOptimizer
  ):
    result = optimizer.optimize(mlir_module_serialized, weight_indices)
    if not isinstance(result, bytes):
      raise TypeError(
          f"Optimizer.optimize must return bytes, got {type(result).__name__}"
      )
    return result
  if callable(optimizer):
    result = optimizer(mlir_module_serialized, weight_indices)
    if not isinstance(result, bytes):
      raise TypeError(
          f"Optimizer callable must return bytes, got {type(result).__name__}"
      )
    return result
  raise TypeError(
      f"Unsupported optimizer type: {type(optimizer)}. Expected an instance"
      " of StablehloOptimizer or a callable."
  )


def _optimize_mlir_payload(
    mlir_module_serialized: bytes,
    module_kept_var_idx: Sequence[int],
    num_weights: int,
    optimizer: OptimizerType,
) -> bytes:
  """Applies Neptune StableHLO optimization passes to the serialized MLIR."""
  if num_weights < 0:
    raise ValueError(f"num_weights must be non-negative, got {num_weights}")

  weight_indices = [
      idx for idx in module_kept_var_idx if 0 <= idx < num_weights
  ]
  return _apply_optimizer(mlir_module_serialized, weight_indices, optimizer)


def create_offline_compiler_hook(
    target_configs: Sequence[TargetConfig],
) -> manifest_utils.CompilerHook:
  """Creates a compiler hook for offline AOT compilation."""
  if not target_configs:
    raise ValueError("target_configs must not be empty.")

  validated_configs = [_validate_target_config(cfg) for cfg in target_configs]

  def compiler_hook(
      mlir_module_serialized: bytes,
      module_kept_var_idx: Sequence[int],
      num_weights: int,
  ) -> Iterator[manifest_utils.CompiledArtifact]:
    if not mlir_module_serialized:
      raise ValueError("mlir_module_serialized must not be empty.")

    logging.info(
        "Offline compiler hook triggered for %d configs", len(validated_configs)
    )

    for config in validated_configs:
      target_topology_label = config.target_topology_label
      enable_deviceless_compilation = config.enable_deviceless_compilation
      platform = config.platform.lower().strip()
      serialized_topology = config.serialized_topology
      target_topology_name = config.target_topology
      autotune_database_path = config.autotune_database_path

      logging.info(
          "Compiling for topology '%s' on platform '%s'",
          target_topology_label,
          platform,
      )

      # 1. Topology Resolution Phase
      if enable_deviceless_compilation:
        if serialized_topology is not None:
          topology = topologies.TopologyDescription.deserialize(
              serialized_topology
          )
        elif target_topology_name is not None:
          topology = topologies.get_topology_desc(
              target_topology_name, platform=platform
          )
        else:
          raise ValueError(
              f"Deviceless compilation for platform '{platform}' requires"
              " either `serialized_topology` or `target_topology`."
          )
      else:
        topology = topologies.get_attached_topology(platform=platform)

      if not topology.devices:
        raise RuntimeError(
            f"No devices found for target topology on platform '{platform}'"
        )

      # 2. MLIR Optimization Phase
      optimized_mlir = mlir_module_serialized
      if config.optimizer is not None:
        optimized_mlir = _optimize_mlir_payload(
            mlir_module_serialized,
            module_kept_var_idx,
            num_weights,
            optimizer=config.optimizer,
        )

      # 3. Scope MLIR Context around both parsing and compilation
      with mlir.make_ir_context():
        try:
          module = jex.mlir.deserialize_portable_artifact(optimized_mlir)
        except (ValueError, RuntimeError, mlir.ir.MLIRError):
          try:
            module = mlir.ir.Module.parse(optimized_mlir)
          except mlir.ir.MLIRError as e:
            raise ValueError(f"Failed to parse MLIR module: {e}") from e

        # 4. Compile Options Configuration
        if config.compile_options is not None:
          compile_options = copy.deepcopy(config.compile_options)
        else:
          compile_options = jaxlib.xla_client.CompileOptions()

        if config.compile_portable_executable is not None:
          compile_options.compile_portable_executable = (
              config.compile_portable_executable
          )
        elif platform != "cpu" and len(topology.devices) <= 1:
          compile_options.compile_portable_executable = True

        if autotune_database_path:
          debug_options = compile_options.executable_build_options.debug_options
          debug_options.xla_gpu_per_fusion_autotune_cache_dir = (
              autotune_database_path
          )

        # 5. Compilation Phase
        client = topology.devices[0].client
        device_list = jaxlib.xla_client.DeviceList(tuple(topology.devices))
        compile_fn = _get_compile_fn(client, enable_deviceless_compilation)
        executable = compile_fn(
            module, device_list, compile_options=compile_options
        )

      # 6. Serialization Phase
      pjrt_executable_bytes = executable.serialize()
      if not pjrt_executable_bytes:
        raise RuntimeError(
            "PJRT executable serialization returned empty bytes for target "
            f"topology '{target_topology_label}' on platform '{platform}'."
        )
      get_version_fn = getattr(executable, "get_executable_version", None)
      raw_version = get_version_fn() if get_version_fn is not None else b""
      executable_version = (
          raw_version if isinstance(raw_version, bytes) else b""
      )

      yield manifest_utils.CompiledArtifact(
          pjrt_executable_bytes=pjrt_executable_bytes,
          kept_var_idx=list(module_kept_var_idx),
          target_topology_label=target_topology_label,
          executable_version=executable_version,
      )

  return compiler_hook
