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

"""Tests for Neptune offline AOT compiler across CPU, CUDA, and TPU backends."""

from collections.abc import Sequence
import typing
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import topologies
import jax.numpy as jnp

from neptune_model._src.core import offline_compiler
from .third_party.neptune.runtime.python import (
    optimizer as neptune_optimizer,
)


def _get_stablehlo_mlir() -> bytes:
  """Generates valid StableHLO MLIR bytecode dynamically using JAX."""
  lowered = jax.jit(lambda x: x + 1.0).lower(jnp.ones((2, 2), jnp.float32))
  module: Any = lowered.compiler_ir(dialect="stablehlo")
  assert module is not None
  return module.operation.get_asm(binary=True)


class OfflineCompilerTest(parameterized.TestCase):
  """Unified test suite for Neptune offline AOT compiler."""

  # ---------------------------------------------------------------------------
  # 1. Deviceless AOT Compilation Tests (Parameterized across backends)
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      (
          "cuda_h100",
          "cuda",
          "h100=8",
          True,
      ),
      (
          "tpu_df2x2",
          "tpu",
          "df:2x2",
          False,
      ),
  )
  def test_deviceless_compilation_named_topology(
      self, backend: str, topology_name: str, needs_autotune_db: bool
  ):
    """Verifies deviceless AOT compilation using named topology descriptors."""
    try:
      topologies.get_topology_desc(topology_name, platform=backend)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(
          f"{backend.upper()} topology factory not registered in this"
          f" binary: {e}"
      )

    stablehlo_mlir = _get_stablehlo_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{backend}_{topology_name}_named",
        enable_deviceless_compilation=True,
        backend_name=backend,
        target_topology=topology_name,
        autotune_database_path=autotune_db,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(
        artifacts[0].target_topology_label, f"{backend}_{topology_name}_named"
    )
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertEqual(artifacts[0].kept_var_idx, [0])
    self.assertIsInstance(artifacts[0].executable_version, bytes)

  @parameterized.named_parameters(
      (
          "cpu",
          "cpu",
          None,
          False,
      ),
      (
          "cuda_h100",
          "cuda",
          "h100=8",
          True,
      ),
      (
          "tpu_df2x2",
          "tpu",
          "df:2x2",
          False,
      ),
  )
  def test_deviceless_compilation_serialized_topology(
      self, backend: str, topology_name: str | None, needs_autotune_db: bool
  ):
    """Verifies deviceless AOT compilation from serialized topology bytes."""
    try:
      if backend == "cpu":
        topo = topologies.get_attached_topology(platform="cpu")
      else:
        assert topology_name is not None
        topo = topologies.get_topology_desc(topology_name, platform=backend)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(
          f"{backend.upper()} topology factory not registered in this"
          f" binary: {e}"
      )

    serialized_topo = topo.serialize()
    stablehlo_mlir = _get_stablehlo_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{backend}_serialized",
        enable_deviceless_compilation=True,
        backend_name=backend,
        serialized_topology=serialized_topo,
        autotune_database_path=autotune_db,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(
        artifacts[0].target_topology_label, f"{backend}_serialized"
    )
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertIsInstance(artifacts[0].executable_version, bytes)

  # ---------------------------------------------------------------------------
  # 2. Attached Hardware Execution Tests (Gated by hardware availability)
  # ---------------------------------------------------------------------------

  @parameterized.parameters("cpu", "cuda", "tpu")
  def test_compilation_with_attached_hardware(self, backend: str):
    """Verifies attached compilation when physical accelerator is present."""
    try:
      devices = jax.devices(backend)
      if not devices:
        self.skipTest(f"Requires attached {backend.upper()} hardware.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(f"Requires attached {backend.upper()} hardware: {e}")

    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label=f"{backend}_attached",
        enable_deviceless_compilation=False,
        backend_name=backend,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, f"{backend}_attached")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertEqual(artifacts[0].kept_var_idx, [0])
    self.assertIsInstance(artifacts[0].executable_version, bytes)

  # ---------------------------------------------------------------------------
  # 3. Optimizer Integration Tests
  # ---------------------------------------------------------------------------

  def test_offline_compilation_with_stablehlo_optimizer(self):
    """Verifies compilation with generic StablehloOptimizer instance."""
    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_opt",
        enable_deviceless_compilation=False,
        backend_name="cpu",
        optimizer=neptune_optimizer.StablehloOptimizer(),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  def test_offline_compilation_with_stablehlo_tpu_optimizer(self):
    """Verifies compilation with StablehloTpuOptimizer instance on TPU."""
    try:
      topologies.get_topology_desc("df:2x2", platform="tpu")
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(f"TPU topology factory not registered in this binary: {e}")

    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="tpu_opt",
        enable_deviceless_compilation=True,
        backend_name="tpu",
        target_topology="df:2x2",
        optimizer=neptune_optimizer.StablehloTpuOptimizer(),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  def test_offline_compilation_with_callable_optimizer(self):
    """Verifies compilation with a custom callable optimizer."""
    stablehlo_mlir = _get_stablehlo_mlir()
    called: list[tuple[bytes, list[int]]] = []

    def custom_optimizer(
        mlir_bytes: bytes, weight_indices: Sequence[int]
    ) -> bytes:
      called.append((mlir_bytes, list(weight_indices)))
      return mlir_bytes

    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_callable_opt",
        enable_deviceless_compilation=False,
        backend_name="cpu",
        optimizer=custom_optimizer,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 1))

    self.assertLen(called, 1)
    self.assertEqual(called[0][1], [0])
    self.assertLen(artifacts, 1)
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  # ---------------------------------------------------------------------------
  # 4. Target Configuration & Normalization Tests
  # ---------------------------------------------------------------------------

  def test_offline_compilation_multiple_target_configs(self):
    """Verifies compilation with multiple target configs in a single hook."""
    stablehlo_mlir = _get_stablehlo_mlir()
    target_configs = [
        offline_compiler.TargetConfig(
            target_topology_label="target_1",
            enable_deviceless_compilation=False,
            backend_name="cpu",
        ),
        offline_compiler.TargetConfig(
            target_topology_label="target_2",
            enable_deviceless_compilation=False,
            backend_name="cpu",
        ),
    ]
    hook = offline_compiler.create_offline_compiler_hook(target_configs)
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 2)
    self.assertEqual(artifacts[0].target_topology_label, "target_1")
    self.assertEqual(artifacts[1].target_topology_label, "target_2")

  def test_target_config_from_dict(self):
    """Verifies TargetConfig creation and hook execution from dictionary."""
    config_dict = {
        "target_topology_label": "dict_target",
        "enable_deviceless_compilation": False,
        "backend_name": "cpu",
        "optimizer": None,
    }
    stablehlo_mlir = _get_stablehlo_mlir()
    hook = offline_compiler.create_offline_compiler_hook([config_dict])
    artifacts = list(hook(stablehlo_mlir, [0], 0))
    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, "dict_target")

  def test_hook_factory_polymorphic_inputs(self):
    """Verifies hook creation with single TargetConfig and single dict."""
    stablehlo_mlir = _get_stablehlo_mlir()
    cfg = offline_compiler.TargetConfig(
        target_topology_label="single_cfg",
        backend_name="cpu",
    )
    hook1 = offline_compiler.create_offline_compiler_hook(cfg)
    self.assertLen(list(hook1(stablehlo_mlir, [0], 0)), 1)

    hook2 = offline_compiler.create_offline_compiler_hook({
        "target_topology_label": "single_dict",
        "backend_name": "cpu",
    })
    self.assertLen(list(hook2(stablehlo_mlir, [0], 0)), 1)

  # ---------------------------------------------------------------------------
  # 5. Error Handling & Validation Tests
  # ---------------------------------------------------------------------------

  def test_empty_mlir_module_bytes_raises_error(self):
    """Verifies ValueError when passing empty bytes as MLIR module."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        backend_name="cpu",
    )
    hook = offline_compiler.create_offline_compiler_hook(config)
    with self.assertRaisesRegex(
        ValueError, "mlir_module_serialized must not be empty"
    ):
      list(hook(b"", [0], 0))

  def test_invalid_mlir_parse_failure(self):
    """Verifies ValueError when passing invalid MLIR module bytes."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        enable_deviceless_compilation=False,
        backend_name="cpu",
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    with self.assertRaisesRegex(ValueError, "Failed to parse MLIR module"):
      list(hook(b"invalid_mlir_bytes", [0], 0))

  def test_empty_target_configs_raises_error(self):
    """Verifies ValueError when target_configs is empty."""
    with self.assertRaisesRegex(
        ValueError, "target_configs must not be empty."
    ):
      offline_compiler.create_offline_compiler_hook([])

  def test_invalid_target_configs_type_raises_error(self):
    """Verifies TypeError when target_configs is of unsupported type."""
    with self.assertRaisesRegex(
        TypeError, "Unsupported type for target_configs: int"
    ):
      offline_compiler.create_offline_compiler_hook(typing.cast(Any, 123))

  @parameterized.parameters("rocm", "unknown_backend", "directx")
  def test_unsupported_backend_name_raises_error(self, invalid_backend: str):
    """Verifies ValueError when backend_name is completely unrecognized."""
    with self.assertRaisesRegex(
        ValueError, f"Unsupported backend_name '{invalid_backend}'"
    ):
      offline_compiler.create_offline_compiler_hook({
          "target_topology_label": "invalid_target",
          "backend_name": invalid_backend,
      })

  def test_unsupported_backend_name_gpu_suggests_cuda(self):
    """Verifies ValueError when passing abstract 'gpu' instead of 'cuda'."""
    with self.assertRaisesRegex(
        ValueError, "Unsupported backend_name 'gpu'. Did you mean 'cuda'?"
    ):
      offline_compiler.create_offline_compiler_hook({
          "target_topology_label": "invalid_gpu",
          "backend_name": "gpu",
      })

  def test_deviceless_missing_topology_spec_raises_error(self):
    """Verifies error when no topology specification is provided."""
    config = offline_compiler.TargetConfig(
        target_topology_label="deviceless_tpu",
        enable_deviceless_compilation=True,
        backend_name="tpu",
    )
    with self.assertRaisesRegex(
        ValueError,
        "Deviceless compilation for platform 'tpu' requires either",
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_cuda_deviceless_missing_autotune_db_raises_error(self):
    """Verifies error when autotune_database_path is missing for CUDA deviceless."""
    config = offline_compiler.TargetConfig(
        target_topology_label="deviceless_cuda",
        enable_deviceless_compilation=True,
        serialized_topology=b"dummy_topo_bytes",
        backend_name="cuda",
    )
    with self.assertRaisesRegex(
        ValueError,
        "autotune_database_path must be provided for CUDA deviceless",
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_cuda_autotune_database_path_nonexistent_raises_error(self):
    """Verifies FileNotFoundError when autotune_database_path does not exist."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_target",
        enable_deviceless_compilation=False,
        backend_name="cuda",
        autotune_database_path="/fake/nonexistent/autotune_db_dir",
    )
    with self.assertRaisesRegex(
        FileNotFoundError, "Autotune database path does not exist"
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_cuda_autotune_database_path_is_file_raises_error(self):
    """Verifies NotADirectoryError when autotune_database_path is a file."""
    temp_file = self.create_tempfile().full_path
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_target",
        enable_deviceless_compilation=False,
        backend_name="cuda",
        autotune_database_path=temp_file,
    )
    with self.assertRaisesRegex(
        NotADirectoryError, "Autotune database path must be a directory"
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_uninstantiated_optimizer_class_raises_type_error(self):
    """Verifies TypeError when uninstantiated class is passed as optimizer."""
    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        backend_name="cpu",
        optimizer=typing.cast(Any, neptune_optimizer.StablehloOptimizer),
    )
    hook = offline_compiler.create_offline_compiler_hook(config)
    with self.assertRaisesRegex(
        TypeError, "Expected an optimizer instance or callable, but got class"
    ):
      list(hook(stablehlo_mlir, [0], 0))

  def test_callable_optimizer_returns_non_bytes_raises_type_error(self):
    """Verifies TypeError when custom optimizer returns non-bytes."""
    stablehlo_mlir = _get_stablehlo_mlir()
    bad_opt = typing.cast(Any, lambda mlir, idx: "non_bytes_return")
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        enable_deviceless_compilation=False,
        backend_name="cpu",
        optimizer=bad_opt,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    with self.assertRaisesRegex(
        TypeError, "Optimizer callable must return bytes, got str"
    ):
      list(hook(stablehlo_mlir, [0], 0))

  def test_unsupported_optimizer_type_raises_type_error(self):
    """Verifies TypeError when passing an unsupported optimizer type."""
    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        enable_deviceless_compilation=False,
        backend_name="cpu",
        optimizer=typing.cast(Any, "invalid_optimizer_type"),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    with self.assertRaisesRegex(
        TypeError, "Unsupported optimizer type: <class 'str'>"
    ):
      list(hook(stablehlo_mlir, [0], 0))

  def test_optimize_mlir_payload_negative_num_weights_raises_error(self):
    """Verifies ValueError when num_weights is negative."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        enable_deviceless_compilation=False,
        backend_name="cpu",
        optimizer=neptune_optimizer.StablehloOptimizer(),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    stablehlo_mlir = _get_stablehlo_mlir()
    with self.assertRaisesRegex(ValueError, "num_weights must be non-negative"):
      list(hook(stablehlo_mlir, [0], -1))


if __name__ == "__main__":
  absltest.main()
