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

"""Tests for Neptune CLI `recompile` command."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
from typer import testing

from neptune_model._src.core import file_utils
from neptune_model._src.core import persistence_lib
from neptune_model._src.core import test_utils
from neptune_model.cli import npt_cli


class RecompileCliTest(test_utils.NptTestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    try:
      self.runner = testing.CliRunner(mix_stderr=False)
    except TypeError:
      self.runner = testing.CliRunner()
    self.app = npt_cli.cli

  # -------------------------------------------------------------------------
  # 1. Multi-Platform Parameterized Deviceless & Attached Compilation
  # -------------------------------------------------------------------------

  @parameterized.named_parameters(
      ("cuda_h100_deviceless", "cuda", "h100=8"),
      ("tpu_df2x2_deviceless", "tpu", "df:2x2"),
  )
  def test_cli_recompile_multi_platform_deviceless(
      self,
      platform: str,
      target_topology: str,
  ):
    """Verifies deviceless compilation across CUDA and TPU topologies."""
    test_utils.get_topology_desc(target_topology, platform=platform)
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / f"in_{platform}")
    )
    output_base = str(pathlib.Path(self.temp_dir) / f"out_{platform}")

    args = [
        "recompile",
        "-i",
        input_model,
        "-o",
        output_base,
        "-p",
        platform,
        "-l",
        f"{platform}_target",
        "--target-topology",
        target_topology,
        "--no-auto-timestamp",
    ]

    result = self.runner.invoke(self.app, args)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("Recompiled model saved to", result.output)
    self.assertTrue(file_utils.exists(output_base))

    manifest = persistence_lib.load(output_base)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)
    exec_body = fn.body.stable_hlo_body.compiled_executables[0]
    self.assertEqual(exec_body.target_topology_label, f"{platform}_target")
    self.assert_compiled_executable_valid(output_base, exec_body)

  def test_cli_recompile_attached_cpu_hardware(self):
    """Verifies recompilation on attached CPU hardware."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_cpu_attached")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_cpu_attached")
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_attached_target",
            "--no-auto-timestamp",
            "--no-enable-deviceless-compilation",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("Recompiled model saved to", result.output)
    self.assertTrue(file_utils.exists(output_base))

    manifest = persistence_lib.load(output_base)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)
    exec_body = fn.body.stable_hlo_body.compiled_executables[0]
    self.assertEqual(exec_body.target_topology_label, "cpu_attached_target")
    self.assert_compiled_executable_valid(output_base, exec_body)

  @parameterized.parameters("cuda", "tpu")
  def test_cli_recompile_attached_hardware(self, platform: str):
    """Verifies recompilation on attached CUDA and TPU accelerator devices."""
    test_utils.require_attached_hardware(platform)

    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / f"in_attached_{platform}")
    )
    output_base = str(pathlib.Path(self.temp_dir) / f"out_attached_{platform}")
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            platform,
            "-l",
            f"{platform}_attached_target",
            "--no-auto-timestamp",
            "--no-enable-deviceless-compilation",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("Recompiled model saved to", result.output)
    self.assertTrue(file_utils.exists(output_base))

    manifest = persistence_lib.load(output_base)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)
    exec_body = fn.body.stable_hlo_body.compiled_executables[0]
    self.assertEqual(
        exec_body.target_topology_label, f"{platform}_attached_target"
    )
    self.assert_compiled_executable_valid(output_base, exec_body)

  def test_cli_recompile_serialized_topology_file(self):
    """Verifies recompilation with an explicit serialized topology file."""
    topo_file = test_utils.create_serialized_topology_file(
        "df:2x2", self.temp_dir, platform="tpu", filename="tpu_topo.pb"
    )
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_serialized_tpu")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_serialized_tpu")
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "tpu",
            "-l",
            "tpu_serialized_target",
            "--serialized-topology",
            topo_file,
            "--no-auto-timestamp",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertTrue(file_utils.exists(output_base))

    manifest = persistence_lib.load(output_base)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)
    exec_body = fn.body.stable_hlo_body.compiled_executables[0]
    self.assertEqual(exec_body.target_topology_label, "tpu_serialized_target")
    self.assert_compiled_executable_valid(output_base, exec_body)

  def test_cli_long_form_options(self):
    """Verifies recompilation using all explicit long-form option flags."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_long_opts")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_long_opts")
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "--input-model-path",
            input_model,
            "--output-model-path",
            output_base,
            "--platform",
            "cpu",
            "--target-topology-label",
            "cpu_long_target",
            "--no-auto-timestamp",
            "--no-enable-deviceless-compilation",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertTrue(file_utils.exists(output_base))
    manifest = persistence_lib.load(output_base)
    fn = manifest.objects["serving_default"].function
    exec_body = fn.body.stable_hlo_body.compiled_executables[0]
    self.assertEqual(exec_body.target_topology_label, "cpu_long_target")
    self.assert_compiled_executable_valid(output_base, exec_body)

  # -------------------------------------------------------------------------
  # 2. Versioning and Directory Resolution Tests
  # -------------------------------------------------------------------------

  @parameterized.named_parameters(
      ("long_flag", "--version"),
      ("short_flag", "-v"),
  )
  def test_cli_explicit_version(self, flag: str):
    """Verifies explicit version subdirectory placement."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_version")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_version")

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            flag,
            "42",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    expected_dir = str(pathlib.Path(output_base) / "42")
    self.assertTrue(file_utils.exists(expected_dir))

  def test_cli_auto_timestamp_enabled(self):
    """Verifies automatic timestamped subdirectory when version is omitted."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_auto_ts")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_auto_ts")

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "--auto-timestamp",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    subdirs = [p.name for p in pathlib.Path(output_base).iterdir()]
    self.assertLen(subdirs, 1)
    self.assertTrue(subdirs[0].isdigit())

  def test_cli_version_overrides_auto_timestamp(self):
    """Verifies explicit version flag takes precedence over auto-timestamp."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_version_override")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_version_override")

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "--version",
            "42",
            "--auto-timestamp",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    expected_dir = str(pathlib.Path(output_base) / "42")
    self.assertTrue(file_utils.exists(expected_dir))

  # -------------------------------------------------------------------------
  # 3. StableHLO Stripping and Inclusion Tests
  # -------------------------------------------------------------------------

  def test_cli_strip_stablehlo(self):
    """Verifies --strip-stablehlo clears bytecode in manifest."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_strip")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_strip")

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "--no-auto-timestamp",
            "--strip-stablehlo",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    manifest = persistence_lib.load(output_base)
    fn = manifest.objects["serving_default"].function
    self.assertFalse(fn.body.stable_hlo_body.HasField("stable_hlo"))
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)
    exec_body = fn.body.stable_hlo_body.compiled_executables[0]
    self.assert_compiled_executable_valid(output_base, exec_body)

  def test_cli_include_stablehlo(self):
    """Verifies explicit --include-stablehlo preserves bytecode in manifest."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_include")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_include")
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "--no-auto-timestamp",
            "--include-stablehlo",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    manifest = persistence_lib.load(output_base)
    fn = manifest.objects["serving_default"].function
    self.assertTrue(fn.body.stable_hlo_body.HasField("stable_hlo"))

  # -------------------------------------------------------------------------
  # 4. Overwrite Policy and Collision Tests
  # -------------------------------------------------------------------------

  def test_cli_overwrite_success(self):
    """Verifies overwriting an existing explicit version directory."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_overwrite")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_overwrite")
    version_dir = str(pathlib.Path(output_base) / "1")
    file_utils.mkdir_p(version_dir)
    sentinel = str(pathlib.Path(version_dir) / "stale_sentinel.txt")
    with file_utils.open_file(sentinel, "w") as f:
      f.write("stale_payload")

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "-v",
            "1",
            "--overwrite",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertTrue(file_utils.exists(version_dir))
    self.assertFalse(file_utils.exists(sentinel))
    self.assertTrue(
        file_utils.exists(str(pathlib.Path(version_dir) / "manifest.pb"))
    )

  def test_cli_overwrite_collision_fails(self):
    """Verifies exit code 1 on version collision with --no-overwrite."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_collision")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_collision")
    file_utils.mkdir_p(str(pathlib.Path(output_base) / "1"))

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "-v",
            "1",
            "--no-overwrite",
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("already exists", result.output)

  # -------------------------------------------------------------------------
  # 5. CUDA Autotune Database Tests
  # -------------------------------------------------------------------------

  def test_cli_cuda_deviceless_with_file_autotune_db(self):
    """Verifies CUDA deviceless compilation with file autotune DB."""
    test_utils.get_topology_desc("h100=8", platform="cuda")
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_cuda_file_db")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_cuda_file_db")
    db_file = test_utils.create_sample_autotune_file(
        self.create_tempdir().full_path
    )

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cuda",
            "-l",
            "cuda_target",
            "--target-topology",
            "h100=8",
            "--autotune-database-path",
            db_file,
            "--no-auto-timestamp",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertTrue(file_utils.exists(output_base))

  def test_cli_cuda_deviceless_with_directory_autotune_db(self):
    """Verifies CUDA deviceless compilation with directory autotune DB."""
    test_utils.get_topology_desc("h100=8", platform="cuda")
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_cuda_dir_db")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_cuda_dir_db")
    db_dir = test_utils.create_sample_autotune_dir(
        self.create_tempdir().full_path
    )

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cuda",
            "-l",
            "cuda_target",
            "--target-topology",
            "h100=8",
            "--autotune-database-path",
            db_dir,
            "--no-auto-timestamp",
        ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertTrue(file_utils.exists(output_base))

  def test_cli_cuda_deviceless_nonexistent_autotune_db_fails(self):
    """Verifies exit code 1 when non-existent autotune DB path is provided."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_cuda_missing_db")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "out_cuda_missing_db")
    nonexistent_path = str(
        pathlib.Path(self.create_tempdir().full_path) / "missing_db.pb"
    )

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "cuda",
            "-l",
            "cuda_target",
            "--autotune-database-path",
            nonexistent_path,
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Autotune database path not found", result.stderr)

  # -------------------------------------------------------------------------
  # 6. Negative and Failure Scenario Tests
  # -------------------------------------------------------------------------

  def test_cli_recompile_invalid_platform_fails(self):
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "input")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "output")

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            output_base,
            "-p",
            "gpu",
            "-l",
            "gpu_target",
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Unsupported platform 'gpu'", result.stderr)

  def test_cli_missing_input_path_fails(self):
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            "/nonexistent/path",
            "-o",
            str(pathlib.Path(self.temp_dir) / "out"),
            "-p",
            "cpu",
            "-l",
            "cpu_target",
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("does not exist", result.stderr)

  def test_cli_missing_serialized_topology_file_fails(self):
    """Verifies exit code 1 when serialized topology file path is missing."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_missing_topo")
    )
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            str(pathlib.Path(self.temp_dir) / "out_missing_topo"),
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--serialized-topology",
            "/nonexistent/topology/file.pb",
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Serialized topology file not found", result.stderr)

  @parameterized.named_parameters(
      ("missing_input", ["-o", "out", "-p", "cpu", "-l", "label"]),
      ("missing_output", ["-i", "in", "-p", "cpu", "-l", "label"]),
      ("missing_platform", ["-i", "in", "-o", "out", "-l", "label"]),
      ("missing_label", ["-i", "in", "-o", "out", "-p", "cpu"]),
  )
  def test_cli_missing_required_options_fails(self, flags: list[str]):
    """Verifies Typer returns exit code 2 when required options are omitted."""
    result = self.runner.invoke(self.app, ["recompile"] + flags)
    self.assertEqual(result.exit_code, 2)
    self.assertIn("Missing option", result.output)

  def test_cli_identical_input_output_paths_fails(self):
    """Verifies exit code 1 when input and output paths are identical."""
    model_dir = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "same_path")
    )
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            model_dir,
            "-o",
            model_dir,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("cannot be identical", result.stderr)

  def test_cli_recompile_stripped_model_fails(self):
    """Verifies exit code 1 when attempting to recompile stripped AOT model."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_recompile_stripped")
    )
    stripped_output = str(pathlib.Path(self.temp_dir) / "out_stripped")
    res1 = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            stripped_output,
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "--no-auto-timestamp",
            "--strip-stablehlo",
        ],
    )
    self.assertEqual(res1.exit_code, 0, msg=res1.output)
    res2 = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            stripped_output,
            "-o",
            str(pathlib.Path(self.temp_dir) / "out_second_attempt"),
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
            "--no-auto-timestamp",
        ],
    )
    self.assertEqual(res2.exit_code, 1)
    self.assertIn("StableHLO payload was stripped", res2.stderr)

  def test_cli_conflicting_topology_options_fails(self):
    """Verifies exit code 1 when both target and serialized topologies are given."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_conflict")
    )
    topo_file = str(pathlib.Path(self.temp_dir) / "placeholder_topo.pb")
    with file_utils.open_file(topo_file, "wb") as f:
      f.write(b"placeholder_proto_bytes")

    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            str(pathlib.Path(self.temp_dir) / "out_conflict"),
            "-p",
            "tpu",
            "-l",
            "tpu_target",
            "--target-topology",
            "df:2x2",
            "--serialized-topology",
            topo_file,
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn(
        "Cannot specify both --target-topology and --serialized-topology",
        result.stderr,
    )

  def test_cli_missing_deviceless_topology_fails(self):
    """Verifies exit code 1 when deviceless accelerator compilation lacks topology."""
    input_model = test_utils.create_test_neptune_model(
        str(pathlib.Path(self.temp_dir) / "in_missing_topo_opts")
    )
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            input_model,
            "-o",
            str(pathlib.Path(self.temp_dir) / "out_missing_topo_opts"),
            "-p",
            "cuda",
            "-l",
            "cuda_target",
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("requires either", result.stderr)

  def test_cli_error_output_written_to_stderr(self):
    """Verifies CLI error messages are emitted to stderr and not stdout."""
    result = self.runner.invoke(
        self.app,
        [
            "recompile",
            "-i",
            "/nonexistent/path",
            "-o",
            str(pathlib.Path(self.temp_dir) / "out"),
            "-p",
            "cpu",
            "-l",
            "cpu_target",
            "--no-enable-deviceless-compilation",
        ],
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Error: Input model path", result.stderr)
    self.assertNotIn("Error:", result.stdout)
    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
  absltest.main()
