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
import pathlib
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import export as jax_export
from jax.experimental import topologies
import jax.numpy as jnp

from neptune_model._src.core import file_utils
from neptune_model._src.core import offline_compiler

try:
  from neptune.runtime.python import optimizer as neptune_optimizer  # pylint: disable=g-import-not-at-top
except ImportError:
  neptune_optimizer = None


def _get_vhlo_portable_artifact() -> bytes:
  """Generates valid VHLO portable artifact using jax.export."""
  exported = jax_export.export(jax.jit(lambda x: x + 1.0))(
      jax.ShapeDtypeStruct((2, 2), jnp.float32)
  )
  return exported.mlir_module_serialized


def _get_raw_text_mlir() -> bytes:
  """Returns raw text MLIR module bytes."""
  return (
      b"module { func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {"
      b" return %arg0 : tensor<2x2xf32> } }"
  )


def _get_stablehlo_mlir() -> bytes:
  """Generates valid StableHLO MLIR bytecode dynamically using JAX."""
  lowered = jax.jit(lambda x: x + 1.0).lower(jnp.ones((2, 2), jnp.float32))
  module: Any = lowered.compiler_ir(dialect="stablehlo")
  assert module is not None
  return module.operation.get_asm(binary=True)


class OfflineCompilerTest(parameterized.TestCase):
  """Unified test suite for Neptune offline AOT compiler."""

  def _get_topology_desc(
      self, topology_name: str, platform: str
  ) -> topologies.TopologyDescription:
    """Safely retrieves topology descriptor or skips if plugin is missing."""
    try:
      return topologies.get_topology_desc(topology_name, platform=platform)
    except (NotImplementedError, RuntimeError) as e:
      self.skipTest(
          f"{platform.upper()} topology support not installed in this binary:"
          f" {e}"
      )

  # TODO: Remove skip once jaxlib >= 0.11.2 is deployed.
  def _get_serialized_topology(
      self, topology_name: str, platform: str
  ) -> bytes:
    """Retrieves serialized topology bytes or skips if unsupported."""
    topo = self._get_topology_desc(topology_name, platform=platform)
    try:
      return topo.serialize()
    except (AttributeError, RuntimeError) as e:
      self.skipTest(
          f"{platform.upper()} topology serialization not supported in this"
          f" binary: {e}"
      )

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
  def test_deviceless_compilation_vhlo_portable_artifact(
      self, platform: str, topology_name: str, needs_autotune_db: bool
  ):
    """Verifies deviceless AOT compilation of genuine VHLO portable artifact."""
    serialized_topo = self._get_serialized_topology(
        topology_name, platform=platform
    )
    vhlo_bytes = _get_vhlo_portable_artifact()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_vhlo_artifact",
        enable_deviceless_compilation=True,
        platform=platform,
        serialized_topology=serialized_topo,
        autotune_database_path=autotune_db,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(vhlo_bytes, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(
        artifacts[0].target_topology_label, f"{platform}_vhlo_artifact"
    )
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertEqual(artifacts[0].kept_var_idx, [0])
    self.assertIsInstance(artifacts[0].executable_version, bytes)

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
  def test_deviceless_compilation_raw_text_mlir(
      self, platform: str, topology_name: str, needs_autotune_db: bool
  ):
    """Verifies deviceless AOT compilation of raw text MLIR fallback payload."""
    serialized_topo = self._get_serialized_topology(
        topology_name, platform=platform
    )
    raw_mlir_bytes = _get_raw_text_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_raw_text_mlir",
        enable_deviceless_compilation=True,
        platform=platform,
        serialized_topology=serialized_topo,
        autotune_database_path=autotune_db,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(raw_mlir_bytes, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(
        artifacts[0].target_topology_label, f"{platform}_raw_text_mlir"
    )
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertEqual(artifacts[0].kept_var_idx, [0])
    self.assertIsInstance(artifacts[0].executable_version, bytes)

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
      self, platform: str, topology_name: str, needs_autotune_db: bool
  ):
    """Verifies deviceless AOT compilation using named topology descriptors."""
    self._get_topology_desc(topology_name, platform=platform)
    stablehlo_mlir = _get_stablehlo_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_{topology_name}_named",
        enable_deviceless_compilation=True,
        platform=platform,
        target_topology=topology_name,
        autotune_database_path=autotune_db,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(
        artifacts[0].target_topology_label, f"{platform}_{topology_name}_named"
    )
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertEqual(artifacts[0].kept_var_idx, [0])
    self.assertIsInstance(artifacts[0].executable_version, bytes)

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
  def test_deviceless_compilation_serialized_topology(
      self, platform: str, topology_name: str, needs_autotune_db: bool
  ):
    """Verifies deviceless AOT compilation from serialized topology bytes."""
    serialized_topo = self._get_serialized_topology(
        topology_name, platform=platform
    )
    stablehlo_mlir = _get_stablehlo_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_serialized",
        enable_deviceless_compilation=True,
        platform=platform,
        serialized_topology=serialized_topo,
        autotune_database_path=autotune_db,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(
        artifacts[0].target_topology_label, f"{platform}_serialized"
    )
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertIsInstance(artifacts[0].executable_version, bytes)

  # ---------------------------------------------------------------------------
  # 2. Attached Hardware Compilation Tests
  # ---------------------------------------------------------------------------

  def test_compilation_with_attached_cpu_hardware(self):
    """Verifies attached compilation on physical CPU hardware."""
    devices = jax.devices("cpu")
    self.assertNotEmpty(devices, "CPU hardware must be present.")

    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_attached",
        enable_deviceless_compilation=False,
        platform="cpu",
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, "cpu_attached")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertEqual(artifacts[0].kept_var_idx, [0])
    self.assertIsInstance(artifacts[0].executable_version, bytes)

  @parameterized.parameters("cuda", "tpu")
  def test_compilation_with_attached_hardware(self, platform: str):
    """Verifies attached compilation when physical accelerator is present."""
    try:
      jax.devices(platform)
    except (RuntimeError, ValueError) as e:
      self.skipTest(f"Requires attached {platform.upper()} hardware: {e}")

    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_attached",
        enable_deviceless_compilation=False,
        platform=platform,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, f"{platform}_attached")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertEqual(artifacts[0].kept_var_idx, [0])
    self.assertIsInstance(artifacts[0].executable_version, bytes)

  # ---------------------------------------------------------------------------
  # 3. Optimizer Integration Tests
  # ---------------------------------------------------------------------------

  def test_offline_compilation_with_stablehlo_optimizer(self):
    """Verifies compilation with generic StablehloOptimizer instance."""
    if neptune_optimizer is None:
      self.skipTest("neptune_optimizer C++ extension not available in OSS.")
    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_opt",
        enable_deviceless_compilation=False,
        platform="cpu",
        optimizer=neptune_optimizer.StablehloOptimizer(),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  def test_offline_compilation_with_stablehlo_tpu_optimizer(self):
    """Verifies compilation with StablehloTpuOptimizer instance on TPU."""
    if neptune_optimizer is None:
      self.skipTest("neptune_optimizer C++ extension not available in OSS.")
    self._get_topology_desc("df:2x2", platform="tpu")

    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="tpu_opt",
        enable_deviceless_compilation=True,
        platform="tpu",
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
        platform="cpu",
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
            platform="cpu",
        ),
        offline_compiler.TargetConfig(
            target_topology_label="target_2",
            enable_deviceless_compilation=False,
            platform="cpu",
        ),
    ]
    hook = offline_compiler.create_offline_compiler_hook(target_configs)
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 2)
    self.assertEqual(artifacts[0].target_topology_label, "target_1")
    self.assertEqual(artifacts[1].target_topology_label, "target_2")

  # ---------------------------------------------------------------------------
  # 5. Error Handling & Validation Tests
  # ---------------------------------------------------------------------------

  def test_empty_mlir_module_bytes_raises_error(self):
    """Verifies ValueError when passing empty bytes as MLIR module."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    with self.assertRaisesRegex(
        ValueError, "mlir_module_serialized must not be empty"
    ):
      list(hook(b"", [0], 0))

  def test_invalid_mlir_parse_failure(self):
    """Verifies ValueError when passing invalid MLIR module bytes."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        enable_deviceless_compilation=False,
        platform="cpu",
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

  @parameterized.parameters("rocm", "unknown_backend", "directx")
  def test_unsupported_platform_raises_error(self, invalid_platform: str):
    """Verifies ValueError when platform is completely unrecognized."""
    config = offline_compiler.TargetConfig(
        target_topology_label="invalid_target",
        platform=invalid_platform,
    )
    with self.assertRaisesRegex(
        ValueError, f"Unsupported platform '{invalid_platform}'"
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_unsupported_platform_raises_value_error(self):
    """Verifies ValueError when passing an unsupported platform."""
    config = offline_compiler.TargetConfig(
        target_topology_label="invalid_platform",
        platform="gpu",
    )
    with self.assertRaisesRegex(
        ValueError, "Unsupported platform 'gpu'. Supported platforms:"
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_deviceless_missing_topology_spec_raises_error(self):
    """Verifies error when no topology specification is provided."""
    config = offline_compiler.TargetConfig(
        target_topology_label="deviceless_tpu",
        enable_deviceless_compilation=True,
        platform="tpu",
    )
    with self.assertRaisesRegex(
        ValueError,
        "Deviceless compilation for platform 'tpu' requires either",
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_cuda_deviceless_without_autotune_db_compiles_with_warning(self):
    """Verifies CUDA deviceless compilation succeeds without autotune db."""
    serialized_topo = self._get_serialized_topology("h100=8", platform="cuda")

    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_no_db",
        enable_deviceless_compilation=True,
        platform="cuda",
        serialized_topology=serialized_topo,
        autotune_database_path=None,
    )
    with self.assertLogs(level="WARNING") as log_output:
      hook = offline_compiler.create_offline_compiler_hook([config])
      artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, "cuda_no_db")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)
    self.assertTrue(
        any(
            "autotune_database_path was not provided" in msg
            for msg in log_output.output
        )
    )

  def test_cuda_deviceless_with_directory_autotune_db(self):
    """Verifies CUDA deviceless compilation routes directory autotune db."""
    serialized_topo = self._get_serialized_topology("h100=8", platform="cuda")

    stablehlo_mlir = _get_stablehlo_mlir()
    temp_dir = self.create_tempdir()
    db_dir = pathlib.Path(temp_dir.full_path) / "autotune_dir"
    file_utils.mkdir_p(str(db_dir))

    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_dir_db",
        enable_deviceless_compilation=True,
        platform="cuda",
        serialized_topology=serialized_topo,
        autotune_database_path=str(db_dir),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, "cuda_dir_db")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  def test_cuda_deviceless_with_file_autotune_db(self):
    """Verifies CUDA deviceless compilation routes file autotune db."""
    serialized_topo = self._get_serialized_topology("h100=8", platform="cuda")

    stablehlo_mlir = _get_stablehlo_mlir()
    temp_dir = self.create_tempdir()
    db_file = pathlib.Path(temp_dir.full_path) / "autotune.textproto"
    with file_utils.open_file(str(db_file), "w") as f:
      f.write("# autotune database placeholder\n")

    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_file_db",
        enable_deviceless_compilation=True,
        platform="cuda",
        serialized_topology=serialized_topo,
        autotune_database_path=str(db_file),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, "cuda_file_db")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  def test_cuda_deviceless_nonexistent_autotune_db_raises_file_not_found(self):
    """Verifies FileNotFoundError when nonexistent autotune db is given."""
    serialized_topo = self._get_serialized_topology("h100=8", platform="cuda")

    nonexistent_path = str(
        pathlib.Path(self.create_tempdir().full_path) / "does_not_exist.pb"
    )
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_missing_db",
        enable_deviceless_compilation=True,
        platform="cuda",
        serialized_topology=serialized_topo,
        autotune_database_path=nonexistent_path,
    )
    with self.assertRaises(FileNotFoundError):
      offline_compiler.create_offline_compiler_hook([config])

  def test_non_cuda_autotune_db_raises_value_error(self):
    """Verifies ValueError when autotune db is given for non-CUDA platform."""
    config = offline_compiler.TargetConfig(
        target_topology_label="tpu_with_db",
        platform="tpu",
        target_topology="df:2x2",
        autotune_database_path="/tmp/fake_autotune.pb",
    )
    with self.assertRaisesRegex(
        ValueError,
        "autotune_database_path is only supported for CUDA platform",
    ):
      offline_compiler.create_offline_compiler_hook([config])

  def test_uninstantiated_optimizer_class_raises_type_error(self):
    """Verifies TypeError when uninstantiated class is passed as optimizer."""

    class _UninstantiatedClass:
      pass

    stablehlo_mlir = _get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
        optimizer=cast(Any, _UninstantiatedClass),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    with self.assertRaisesRegex(
        TypeError, "Expected an optimizer instance or callable, but got class"
    ):
      list(hook(stablehlo_mlir, [0], 0))

  def test_callable_optimizer_returns_non_bytes_raises_type_error(self):
    """Verifies TypeError when custom optimizer returns non-bytes."""
    stablehlo_mlir = _get_stablehlo_mlir()
    bad_opt = cast(Any, lambda mlir, idx: "non_bytes_return")
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        enable_deviceless_compilation=False,
        platform="cpu",
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
        platform="cpu",
        optimizer=cast(Any, "invalid_optimizer_type"),
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
        platform="cpu",
        optimizer=lambda mlir, idx: mlir,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    stablehlo_mlir = _get_stablehlo_mlir()
    with self.assertRaisesRegex(ValueError, "num_weights must be non-negative"):
      list(hook(stablehlo_mlir, [0], -1))


if __name__ == "__main__":
  absltest.main()
