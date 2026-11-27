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

"""Tests for XSE offline AOT compiler across CPU, CUDA, and TPU backends."""

from collections.abc import Sequence
import pathlib
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jaxlib
from xla_saved_model._src.core import offline_compiler
from xla_saved_model._src.core import test_utils


class OfflineCompilerTest(parameterized.TestCase):
  """Unified test suite for XSE offline AOT compiler."""

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
    serialized_topo = test_utils.get_serialized_topology(
        topology_name, platform=platform
    )
    vhlo_bytes = test_utils.get_vhlo_portable_artifact()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_vhlo_artifact",
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
    serialized_topo = test_utils.get_serialized_topology(
        topology_name, platform=platform
    )
    raw_mlir_bytes = test_utils.get_raw_text_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_raw_text_mlir",
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
    test_utils.get_topology_desc(topology_name, platform=platform)
    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_{topology_name}_named",
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
    serialized_topo = test_utils.get_serialized_topology(
        topology_name, platform=platform
    )
    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_serialized",
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

    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_attached",
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
    test_utils.require_attached_hardware(platform)

    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_attached",
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

  def test_offline_compilation_with_callable_optimizer(self):
    """Verifies compilation with a custom callable optimizer."""
    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    called: list[tuple[bytes, list[int]]] = []

    def custom_optimizer(
        mlir_bytes: bytes, weight_indices: Sequence[int]
    ) -> bytes:
      called.append((mlir_bytes, list(weight_indices)))
      return mlir_bytes

    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_callable_opt",
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
    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    target_configs = [
        offline_compiler.TargetConfig(
            target_topology_label="target_1",
            platform="cpu",
        ),
        offline_compiler.TargetConfig(
            target_topology_label="target_2",
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

  def test_target_config_is_deviceless_deduction(self):
    """Verifies is_deviceless is auto-deduced from topology presence."""
    cfg_attached = offline_compiler.TargetConfig(
        target_topology_label="cpu_attached",
        platform="cpu",
    )
    self.assertFalse(cfg_attached.is_deviceless)

    cfg_named = offline_compiler.TargetConfig(
        target_topology_label="tpu_named",
        platform="tpu",
        target_topology="df:2x2",
    )
    self.assertTrue(cfg_named.is_deviceless)

    cfg_serialized = offline_compiler.TargetConfig(
        target_topology_label="cuda_serialized",
        platform="cuda",
        serialized_topology=b"serialized_bytes",
    )
    self.assertTrue(cfg_serialized.is_deviceless)

  def test_cuda_deviceless_without_autotune_db_compiles_with_warning(self):
    """Verifies CUDA deviceless compilation succeeds without autotune db."""
    serialized_topo = test_utils.get_serialized_topology(
        "h100=8", platform="cuda"
    )

    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_no_db",
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
    serialized_topo = test_utils.get_serialized_topology(
        "h100=8", platform="cuda"
    )

    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    db_dir = test_utils.create_sample_autotune_dir(
        self.create_tempdir().full_path
    )

    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_dir_db",
        platform="cuda",
        serialized_topology=serialized_topo,
        autotune_database_path=db_dir,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, "cuda_dir_db")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  def test_cuda_deviceless_with_file_autotune_db(self):
    """Verifies CUDA deviceless compilation routes file autotune db."""
    serialized_topo = test_utils.get_serialized_topology(
        "h100=8", platform="cuda"
    )

    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    db_file = test_utils.create_sample_autotune_file(
        self.create_tempdir().full_path
    )

    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_file_db",
        platform="cuda",
        serialized_topology=serialized_topo,
        autotune_database_path=db_file,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(stablehlo_mlir, [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].target_topology_label, "cuda_file_db")
    self.assertNotEmpty(artifacts[0].pjrt_executable_bytes)

  def test_cuda_deviceless_nonexistent_autotune_db_raises_file_not_found(self):
    """Verifies FileNotFoundError when nonexistent autotune db is given."""
    serialized_topo = test_utils.get_serialized_topology(
        "h100=8", platform="cuda"
    )

    nonexistent_path = str(
        pathlib.Path(self.create_tempdir().full_path) / "does_not_exist.pb"
    )
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_missing_db",
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

    stablehlo_mlir = test_utils.get_stablehlo_mlir()
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
    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    bad_opt = cast(Any, lambda mlir, idx: "non_bytes_return")
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
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
    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
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
        platform="cpu",
        optimizer=lambda mlir, idx: mlir,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    stablehlo_mlir = test_utils.get_stablehlo_mlir()
    with self.assertRaisesRegex(ValueError, "num_weights must be non-negative"):
      list(hook(stablehlo_mlir, [0], -1))

  # ---------------------------------------------------------------------------
  # Recorded Compile Options Tests
  # ---------------------------------------------------------------------------

  def test_deviceless_compilation_returns_recorded_compile_options(self):
    """Verifies that artifacts carry the compile options to record."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_target",
        platform="CUDA",
        serialized_topology=test_utils.get_serialized_topology(
            "h100=1", platform="cuda"
        ),
        autotune_database_path=self.create_tempdir().full_path,
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(test_utils.get_vhlo_portable_artifact(), [0], 0))

    self.assertLen(artifacts, 1)
    self.assertEqual(artifacts[0].platform, "cuda")
    compile_options = artifacts[0].compile_options
    self.assertIsNotNone(compile_options)
    # The autotune database is a path on the compiling host.
    recorded_debug_options = (
        compile_options.executable_build_options.debug_options
    )
    self.assertEmpty(recorded_debug_options.xla_autotune_cache_dir)
    self.assertEmpty(
        recorded_debug_options.xla_gpu_per_fusion_autotune_cache_dir
    )

  def test_deviceless_multi_device_compilation_records_no_compile_options(
      self,
  ):
    """Verifies that executables for multiple devices record no options."""
    config = offline_compiler.TargetConfig(
        target_topology_label="cuda_target",
        platform="cuda",
        serialized_topology=test_utils.get_serialized_topology(
            "h100=8", platform="cuda"
        ),
    )
    hook = offline_compiler.create_offline_compiler_hook([config])
    artifacts = list(hook(test_utils.get_vhlo_portable_artifact(), [0], 0))

    self.assertLen(artifacts, 1)
    self.assertIsNone(artifacts[0].compile_options)

  def test_to_recorded_compile_options_strips_offline_only_options(self):
    """Verifies that offline-only options aren't recorded."""
    compile_options = jaxlib.xla_client.CompileOptions()
    compile_options.compile_portable_executable = True
    debug_options = compile_options.executable_build_options.debug_options
    debug_options.xla_gpu_load_autotune_results_from = "/tmp/autotune.textproto"
    debug_options.xla_gpu_per_fusion_autotune_cache_dir = "/tmp/autotune"
    debug_options.xla_load_autotune_cache_from = "/tmp/autotune.textproto"
    debug_options.xla_autotune_cache_dir = "/tmp/autotune"
    compile_options.env_option_overrides = [
        ("xla_gpu_require_complete_aot_autotune_results", True),
        ("xla_fail_on_autotune_cache_miss", True),
        ("xla_gpu_autotune_level", 4),
    ]

    recorded = offline_compiler._to_recorded_compile_options(compile_options)

    self.assertTrue(recorded.compile_portable_executable)
    recorded_debug_options = recorded.executable_build_options.debug_options
    self.assertEmpty(recorded_debug_options.xla_gpu_load_autotune_results_from)
    self.assertEmpty(
        recorded_debug_options.xla_gpu_per_fusion_autotune_cache_dir
    )
    self.assertEmpty(recorded_debug_options.xla_load_autotune_cache_from)
    self.assertEmpty(recorded_debug_options.xla_autotune_cache_dir)
    self.assertEqual(
        set(recorded.env_option_overrides), {"xla_gpu_autotune_level"}
    )
    self.assertEqual(
        recorded.env_option_overrides["xla_gpu_autotune_level"].int_field, 4
    )


if __name__ == "__main__":
  absltest.main()
