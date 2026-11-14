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

"""XSE Offline AOT Compiler Factory."""

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

from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import manifest_utils
from xla_saved_model.protos.xla.pjrt.proto import compile_options_pb2  # pylint: disable=g-direct-tensorflow-import

optimizer_ext = None
OptimizerType = Callable[[bytes, Sequence[int]], bytes]

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
    serialized_topology: Optional serialized topology bytes required for
      deviceless compilation.
    autotune_database_path: Optional path to the CUDA autotune database.
    target_topology: Optional JAX topology mesh specification string (e.g.
      "2x2x2").
    compile_portable_executable: Optional override to compile as a portable
      executable.
    optimizer: Optional XSE StableHLO optimizer instance or custom callable
      hook.
    compile_options: Optional pre-configured `xla_client.CompileOptions`
      override.
  """

  target_topology_label: str
  platform: str
  serialized_topology: bytes | None = None
  autotune_database_path: str | None = None
  target_topology: str | None = None
  compile_portable_executable: bool | None = None
  optimizer: OptimizerType | None = None
  compile_options: Any = None

  @property
  def is_deviceless(self) -> bool:
    """Returns whether compilation is performed without attached devices."""
    return (
        self.serialized_topology is not None or self.target_topology is not None
    )


def _validate_target_config(config: TargetConfig) -> None:
  """Validates a single TargetConfig instance fail-fast upfront."""
  platform = config.platform.lower().strip()

  # 1. Platform validation
  if platform not in _SUPPORTED_PLATFORMS:
    raise ValueError(
        f"Unsupported platform '{config.platform}'. Supported platforms:"
        f" {sorted(_SUPPORTED_PLATFORMS)}"
    )

  # 2. Autotune database validation
  if config.autotune_database_path:
    if platform != "cuda":
      raise ValueError(
          "autotune_database_path is only supported for CUDA platform, but"
          f" got platform '{platform}'."
      )
    if not file_utils.exists(config.autotune_database_path):
      raise FileNotFoundError(
          "Autotune database path does not exist:"
          f" {config.autotune_database_path}"
      )
  elif platform == "cuda" and config.is_deviceless:
    logging.warning(
        "autotune_database_path was not provided for CUDA deviceless"
        " compilation. XLA will fallback to internal heuristics/cost-models"
        " which may not be deterministic or optimal in production."
    )


def _resolve_topology(config: TargetConfig) -> topologies.TopologyDescription:
  """Resolves topology description from serialized bytes, name, or hardware."""
  platform = config.platform.lower().strip()
  if not config.is_deviceless:
    return topologies.get_attached_topology(platform=platform)
  if config.serialized_topology is not None:
    return topologies.TopologyDescription.deserialize(
        config.serialized_topology
    )
  if config.target_topology is not None:
    return topologies.get_topology_desc(
        config.target_topology, platform=platform
    )
  raise ValueError(
      f"Deviceless compilation for platform '{platform}' requires either"
      " `serialized_topology` or `target_topology`."
  )


def _build_compile_options(
    config: TargetConfig,
    device_count: int,
) -> jaxlib.xla_client.CompileOptions:
  """Constructs and configures XLA CompileOptions for target topology."""
  compile_options = (
      copy.deepcopy(config.compile_options)
      if config.compile_options is not None
      else jaxlib.xla_client.CompileOptions()
  )

  # Portable executable policy
  if config.compile_portable_executable is not None:
    compile_options.compile_portable_executable = (
        config.compile_portable_executable
    )
  elif device_count == 1:
    # Single-device targets default to portable executables (omitting device
    # assignment) so the runtime can dynamically schedule across cores via
    # serving_core_selector. Multi-device meshes require explicit SPMD
    # assignments and are rejected by runtime if portable.
    compile_options.compile_portable_executable = True

  # Autotune database routing
  if config.autotune_database_path:
    debug_options = compile_options.executable_build_options.debug_options
    if file_utils.isdir(config.autotune_database_path):
      debug_options.xla_autotune_cache_dir = config.autotune_database_path
    else:
      debug_options.xla_load_autotune_cache_from = config.autotune_database_path

  return compile_options


# The debug options and environment option overrides that only apply to the
# offline compilation, and aren't recorded with the compiled executables: the
# autotune database is a path on the compiling host. The deprecated
# xla_gpu_require_complete_aot_autotune_results flag alias is still accepted by
# XLA, so it's stripped as well.
_OFFLINE_ONLY_DEBUG_OPTIONS = (
    "xla_load_autotune_cache_from",
    "xla_autotune_cache_dir",
)
_OFFLINE_ONLY_ENV_OPTION_OVERRIDES = (
    "xla_fail_on_autotune_cache_miss",
    "xla_gpu_require_complete_aot_autotune_results",
)


def _to_recorded_compile_options(
    compile_options: jaxlib.xla_client.CompileOptions,
) -> compile_options_pb2.CompileOptionsProto:
  """Returns the compile options to record with the compiled executables.

  Args:
    compile_options: The compile options that the executables were compiled
      with.

  Returns:
    `compile_options`, without the options that only apply to the offline
    compilation.
  """
  proto = compile_options_pb2.CompileOptionsProto.FromString(
      compile_options.SerializeAsString()
  )
  debug_options = proto.executable_build_options.debug_options
  for field in _OFFLINE_ONLY_DEBUG_OPTIONS:
    # Skip fields that the XLA version in use doesn't define yet.
    if field in debug_options.DESCRIPTOR.fields_by_name:
      debug_options.ClearField(field)
  for key in _OFFLINE_ONLY_ENV_OPTION_OVERRIDES:
    proto.env_option_overrides.pop(key, None)
  return proto


def _get_compile_fn(
    client: Any,
    is_deviceless: bool,
) -> Callable[..., Any]:
  """Returns the compilation method for the given execution mode."""
  if is_deviceless:
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
  if optimizer_ext is not None and isinstance(
      optimizer, optimizer_ext.StablehloOptimizer
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
  """Applies XSE StableHLO optimization passes to the serialized MLIR."""
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

  for cfg in target_configs:
    _validate_target_config(cfg)

  def compiler_hook(
      mlir_module_serialized: bytes,
      module_kept_var_idx: Sequence[int],
      num_weights: int,
  ) -> Iterator[manifest_utils.CompiledArtifact]:
    if not mlir_module_serialized:
      raise ValueError("mlir_module_serialized must not be empty.")

    logging.info(
        "Offline compiler hook triggered for %d configs", len(target_configs)
    )

    for config in target_configs:
      target_topology_label = config.target_topology_label
      platform = config.platform.lower().strip()

      logging.info(
          "Compiling for topology '%s' on platform '%s'",
          target_topology_label,
          platform,
      )

      # 1. Topology Resolution Phase
      topology = _resolve_topology(config)
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
        compile_options = _build_compile_options(config, len(topology.devices))

        # 5. Compilation Phase
        client = topology.devices[0].client
        device_list = jaxlib.xla_client.DeviceList(tuple(topology.devices))
        compile_fn = _get_compile_fn(client, config.is_deviceless)
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
          platform=platform,
          # The compile options don't describe the devices of executables
          # compiled for multiple devices, which the runtime takes from them.
          compile_options=(
              _to_recorded_compile_options(compile_options)
              if compile_options.compile_portable_executable
              else None
          ),
      )

  return compiler_hook
