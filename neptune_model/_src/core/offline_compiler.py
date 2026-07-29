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

from collections.abc import Callable, Iterator, Mapping, Sequence
import copy
import dataclasses
import os
from typing import Any

from absl import logging
import jax
from jax.experimental import topologies
from mlir import ir

from neptune_model._src.core import (
    manifest_utils,
)
from .third_party.neptune.runtime.python import (
    optimizer as neptune_optimizer,
)

OptimizerType = (
    neptune_optimizer.StablehloOptimizer
    | Callable[[bytes, Sequence[int]], bytes]
    | None
)

_SUPPORTED_BACKENDS = frozenset({"cpu", "cuda", "tpu"})


@dataclasses.dataclass(frozen=True)
class TargetConfig:
  """Configuration for a single offline AOT compilation target topology.

  Attributes:
    target_topology_label: Identifier for the compiled artifact saved in the
      manifest (matches CompiledExecutableBody.target_topology_label in
      manifest.proto).
    enable_deviceless_compilation: Whether to perform compilation without
      physical devices attached.
    backend_name: Name of the target platform/backend ("cuda", "tpu", "cpu").
      Note: 'gpu' is not supported; use 'cuda' directly.
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
  backend_name: str
  enable_deviceless_compilation: bool = False
  serialized_topology: bytes | None = None
  autotune_database_path: str | None = None
  target_topology: str | None = None
  compile_portable_executable: bool | None = None
  optimizer: OptimizerType = None
  compile_options: Any = None

  @classmethod
  def from_dict(cls, config_dict: Mapping[str, Any]) -> "TargetConfig":
    """Creates a TargetConfig instance from a dictionary representation."""
    return cls(
        target_topology_label=config_dict["target_topology_label"],
        enable_deviceless_compilation=config_dict.get(
            "enable_deviceless_compilation", False
        ),
        backend_name=config_dict["backend_name"],
        serialized_topology=config_dict.get("serialized_topology"),
        autotune_database_path=config_dict.get("autotune_database_path"),
        target_topology=config_dict.get("target_topology"),
        compile_portable_executable=config_dict.get(
            "compile_portable_executable"
        ),
        optimizer=config_dict.get("optimizer"),
        compile_options=config_dict.get("compile_options"),
    )


def _optimize_mlir_payload(
    mlir_module_serialized: bytes,
    module_kept_var_idx: Sequence[int],
    num_weights: int,
    optimizer: OptimizerType = None,
) -> bytes:
  """Applies Neptune StableHLO optimization passes to the serialized MLIR."""
  if optimizer is None:
    return mlir_module_serialized

  if num_weights < 0:
    raise ValueError(f"num_weights must be non-negative, got {num_weights}")

  if isinstance(optimizer, type):
    raise TypeError(
        "Expected an optimizer instance or callable, but got class "
        f"'{optimizer.__name__}'. Did you forget to instantiate it?"
    )

  weight_indices = [
      idx for idx in module_kept_var_idx if 0 <= idx < num_weights
  ]

  if isinstance(optimizer, neptune_optimizer.StablehloOptimizer):
    return optimizer.optimize(mlir_module_serialized, weight_indices)
  if callable(optimizer):
    optimized_bytes = optimizer(mlir_module_serialized, weight_indices)
    if not isinstance(optimized_bytes, bytes):
      raise TypeError(
          "Optimizer callable must return bytes, got"
          f" {type(optimized_bytes).__name__}"
      )
    return optimized_bytes
  raise TypeError(f"Unsupported optimizer type: {type(optimizer)}")


def create_offline_compiler_hook(
    target_configs: (
        TargetConfig
        | Mapping[str, Any]
        | Sequence[TargetConfig | Mapping[str, Any]]
    ),
) -> Callable[
    [bytes, Sequence[int], int],
    Iterator[manifest_utils.CompiledArtifact],
]:
  """Creates a compiler hook for offline AOT compilation."""
  if isinstance(target_configs, (TargetConfig, Mapping)):
    raw_configs = [target_configs]
  elif isinstance(target_configs, Sequence) and not isinstance(
      target_configs, (str, bytes)
  ):
    raw_configs = list(target_configs)
  else:
    raise TypeError(
        f"Unsupported type for target_configs: {type(target_configs).__name__}"
    )

  if not raw_configs:
    raise ValueError("target_configs must not be empty.")

  validated_configs: list[TargetConfig] = []
  # Validate and normalize target configurations upfront.
  for item in raw_configs:
    cfg = (
        item if isinstance(item, TargetConfig) else TargetConfig.from_dict(item)
    )
    backend = cfg.backend_name.lower().strip()
    if backend not in _SUPPORTED_BACKENDS:
      if backend == "gpu":
        raise ValueError(
            f"Unsupported backend_name '{cfg.backend_name}'. Did you mean"
            " 'cuda'? Neptune requires concrete backend names (e.g."
            " 'cuda', 'tpu', 'cpu')."
        )
      raise ValueError(
          f"Unsupported backend_name '{cfg.backend_name}'. Supported"
          f" backends: {sorted(_SUPPORTED_BACKENDS)}"
      )
    if cfg.autotune_database_path:
      if not os.path.exists(cfg.autotune_database_path):
        raise FileNotFoundError(
            "Autotune database path does not exist:"
            f" {cfg.autotune_database_path}"
        )
      if not os.path.isdir(cfg.autotune_database_path):
        raise NotADirectoryError(
            "Autotune database path must be a directory:"
            f" {cfg.autotune_database_path}"
        )
    if cfg.enable_deviceless_compilation:
      if cfg.serialized_topology is None and cfg.target_topology is None:
        raise ValueError(
            f"Deviceless compilation for platform '{backend}' requires "
            "either `serialized_topology` or `target_topology`."
        )
    if (
        cfg.enable_deviceless_compilation
        and backend == "cuda"
        and not cfg.autotune_database_path
    ):
      raise ValueError(
          "autotune_database_path must be provided for CUDA deviceless"
          " compilation to guarantee determinism."
      )
    validated_configs.append(cfg)

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
      backend_name = config.backend_name.lower().strip()
      serialized_topology = config.serialized_topology
      target_topology_name = config.target_topology
      autotune_database_path = config.autotune_database_path

      logging.info(
          "Compiling for topology '%s' on platform '%s'",
          target_topology_label,
          backend_name,
      )

      # 1. Topology Resolution Phase
      if enable_deviceless_compilation:

        if serialized_topology is not None:
          topology = topologies.TopologyDescription.deserialize(
              serialized_topology
          )
        elif target_topology_name is not None:
          topology = topologies.get_topology_desc(
              platform=backend_name, topology_name=target_topology_name
          )
        else:
          raise AssertionError("Unreachable state in topology resolution.")
      else:
        topology = topologies.get_attached_topology(platform=backend_name)

      if not topology.devices:
        raise RuntimeError(
            f"No devices found for target topology on platform '{backend_name}'"
        )

      client = topology.devices[0].client

      # 2. MLIR Optimization Phase
      optimized_mlir = _optimize_mlir_payload(
          mlir_module_serialized,
          module_kept_var_idx,
          num_weights,
          optimizer=config.optimizer,
      )

      # 3. Scope MLIR Context around both parsing and compilation
      with jax.interpreters.mlir.make_ir_context():
        try:
          module = ir.Module.parse(optimized_mlir)
        except ir.MLIRError as e:
          raise ValueError(f"Failed to parse MLIR module: {e}") from e

        # 4. Compile Options Configuration
        if config.compile_options is not None:
          compile_options = copy.deepcopy(config.compile_options)
        else:
          compile_options = jax.jaxlib.xla_client.CompileOptions()

        if config.compile_portable_executable is not None:
          compile_options.compile_portable_executable = (
              config.compile_portable_executable
          )
        elif backend_name != "cpu" and len(topology.devices) <= 1:
          compile_options.compile_portable_executable = True

        if autotune_database_path:
          debug_options = compile_options.executable_build_options.debug_options
          debug_options.xla_gpu_per_fusion_autotune_cache_dir = (
              autotune_database_path
          )

        # 5. Compilation Phase
        device_list = jax.jaxlib.xla_client.DeviceList(tuple(topology.devices))
        compile_fn = getattr(client, "compile", None)
        compile_and_load_fn = getattr(client, "compile_and_load", None)

        if enable_deviceless_compilation:
          if not compile_fn:
            raise AttributeError(
                f"Target client '{type(client).__name__}' does not support"
                " `compile` required for deviceless compilation."
            )
          target_fn = compile_fn
        else:
          target_fn = compile_fn or compile_and_load_fn
          if not target_fn:
            raise AttributeError(
                f"Target client '{type(client).__name__}' supports neither"
                " `compile` nor `compile_and_load`."
            )

        try:
          executable = target_fn(
              module, device_list, compile_options=compile_options
          )
        except jax.errors.JaxRuntimeError as e:
          msg = str(e).upper()
          if any(
              k in msg
              for k in (
                  "UNIMPLEMENTED",
                  "NOT_FOUND",
                  "NOT FOUND",
                  "NO COMPILER FACTORY",
              )
          ):
            if enable_deviceless_compilation:
              raise RuntimeError(
                  "Deviceless compilation failed: `client.compile` returned"
                  f" UNIMPLEMENTED/NOT_FOUND for platform '{backend_name}'."
                  " Fallback to `compile_and_load` is disabled when"
                  " `enable_deviceless_compilation=True` because"
                  " `compile_and_load` requires physical hardware allocation."
              ) from e
            elif compile_fn and compile_and_load_fn:
              logging.warning(
                  "client.compile returned UNIMPLEMENTED/NOT_FOUND; falling"
                  " back to compile_and_load: %s",
                  e,
              )
              executable = compile_and_load_fn(
                  module, device_list, compile_options=compile_options
              )
            else:
              raise
          else:
            raise

      # 6. Serialization Phase
      pjrt_executable_bytes = executable.serialize()
      if not pjrt_executable_bytes:
        raise RuntimeError(
            "PJRT executable serialization returned empty bytes for target "
            f"topology '{target_topology_label}' on platform '{backend_name}'."
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
