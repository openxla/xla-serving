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

"""Tests for Neptune Ahead-Of-Time (AOT) Recompiler."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import topologies
import jax.numpy as jnp

from neptune_model._src.core import aot_recompiler
from neptune_model._src.core import constants
from neptune_model._src.core import file_utils
from neptune_model._src.core import offline_compiler
from neptune_model._src.core import persistence_lib
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config


def _create_neptune_model(
    export_path: str,
    *,
    save_shlo_to_file: bool = False,
) -> str:
  """Exports a Neptune model with genuine JAX math (sin(x) + w)."""

  def forward(params, inputs):
    return {"out": jnp.sin(inputs["x"]) + params["w"]}

  params = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)}
  spec = jax.ShapeDtypeStruct((2, 2), jnp.float32)

  bundle = jax_model.ApplyFnBundle(apply_fn=forward)
  model = jax_model.JaxModel(
      apply_fn_bundle=bundle, params=params, name="model"
  )
  cfg = serving_config.ServingConfig(
      serving_signature="serving_default",
      input_signature={"x": spec},
  )

  main_lib.save(
      models=model,
      export_path=export_path,
      serving_configs=[cfg],
      save_shlo_to_file=save_shlo_to_file,
  )
  return export_path


class AotRecompilerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  # ---------------------------------------------------------------------------
  # 1. Parameterized Output Version Path Resolution Unit Tests
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      (
          "linear_model_explicit_version",
          "models/linear_model",
          42,
          True,
          ("models/linear_model", "models/linear_model/42"),
      ),
      (
          "vit_model_numeric_path_suffix",
          "models/vit_model/101",
          None,
          True,
          ("models/vit_model", "models/vit_model/101"),
      ),
      (
          "dlrm_model_custom_export_no_timestamp",
          "models/dlrm_model/latest_export",
          None,
          False,
          ("models/dlrm_model", "models/dlrm_model/latest_export"),
      ),
  )
  def test_resolve_output_version_path(
      self, path, version, auto_timestamp, expected
  ):
    base, target = aot_recompiler._resolve_output_version_path(
        path, explicit_version=version, auto_timestamp=auto_timestamp
    )
    self.assertEqual((base, target), expected)

  # TODO: Remove skip once jaxlib >= 0.11.2 is deployed.
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
  # 2. Parameterized Multi-Platform Deviceless Recompilation Tests
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      (
          "cuda_h100_with_db",
          "cuda",
          "h100=8",
          True,
      ),
      (
          "cuda_h100_no_db",
          "cuda",
          "h100=8",
          False,
      ),
      (
          "tpu_df2x2",
          "tpu",
          "df:2x2",
          False,
      ),
  )
  def test_deviceless_single_target_recompile(
      self, platform: str, topology_name: str, needs_autotune_db: bool
  ):
    """Verifies deviceless AOT recompilation across CUDA and TPU."""
    serialized_topo = self._get_serialized_topology(
        topology_name, platform=platform
    )

    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "linear_model_input")
    )
    output_base = str(
        pathlib.Path(self.temp_dir) / "models" / "linear_model_out"
    )
    autotune_db = self.create_tempdir().full_path if needs_autotune_db else None

    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_target",
        platform=platform,
        enable_deviceless_compilation=True,
        serialized_topology=serialized_topo,
        autotune_database_path=autotune_db,
    )

    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
    )
    self.assertTrue(file_utils.exists(out_dir))

    manifest = persistence_lib.load(out_dir)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)

    exec_body = fn.body.stable_hlo_body.compiled_executables[0]
    self.assertEqual(exec_body.target_topology_label, f"{platform}_target")
    self.assertEqual(
        exec_body.aot_executable.mime_type, constants.IFRT_EXECUTABLE_MIME_TYPE
    )
    self.assertIsInstance(exec_body.executable_version, bytes)
    self.assertEqual(list(exec_body.aot_kept_var_idx), [0, 1])

    bin_path = str(
        pathlib.Path(out_dir)
        / exec_body.aot_executable.file_system_location.string_path
    )
    self.assertTrue(file_utils.exists(bin_path))
    with file_utils.open_file(bin_path, "rb") as f:
      bin_bytes = f.read()
    self.assertNotEmpty(bin_bytes)

  def test_deviceless_multi_target_recompile(self):
    """Verifies multi-target recompilation across 3 heterogeneous targets."""
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "vit_model_input")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "models" / "vit_model_out")

    serialized_tpu = self._get_serialized_topology("df:2x2", platform="tpu")
    serialized_cuda = self._get_serialized_topology("h100=8", platform="cuda")

    autotune_dir = self.create_tempdir().full_path
    configs = [
        offline_compiler.TargetConfig(
            target_topology_label="tpu_slice_1",
            platform="tpu",
            enable_deviceless_compilation=True,
            serialized_topology=serialized_tpu,
        ),
        offline_compiler.TargetConfig(
            target_topology_label="tpu_slice_2",
            platform="tpu",
            enable_deviceless_compilation=True,
            target_topology="df:2x2",
        ),
        offline_compiler.TargetConfig(
            target_topology_label="cuda_target",
            platform="cuda",
            enable_deviceless_compilation=True,
            serialized_topology=serialized_cuda,
            autotune_database_path=autotune_dir,
        ),
    ]

    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=configs,
        version=1,
    )

    manifest = persistence_lib.load(out_dir)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 3)
    labels = {
        e.target_topology_label
        for e in fn.body.stable_hlo_body.compiled_executables
    }
    self.assertEqual(labels, {"tpu_slice_1", "tpu_slice_2", "cuda_target"})

    for exec_body in fn.body.stable_hlo_body.compiled_executables:
      bin_path = str(
          pathlib.Path(out_dir)
          / exec_body.aot_executable.file_system_location.string_path
      )
      self.assertTrue(file_utils.exists(bin_path))
      with file_utils.open_file(bin_path, "rb") as f:
        self.assertNotEmpty(f.read())

  def test_attached_cpu_hardware_recompile(self):
    """Verifies attached recompilation on physical CPU hardware."""
    devices = jax.devices("cpu")
    self.assertNotEmpty(devices, "CPU hardware must be present.")

    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models/cpu_attached_in")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "models/cpu_attached_out")
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_attached",
        platform="cpu",
        enable_deviceless_compilation=False,
    )
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
    )
    self.assertTrue(file_utils.exists(out_dir))
    manifest = persistence_lib.load(out_dir)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        fn.body.stable_hlo_body.compiled_executables[0].target_topology_label,
        "cpu_attached",
    )

  @parameterized.parameters("cuda", "tpu")
  def test_attached_hardware_recompile(self, platform: str):
    """Verifies attached recompilation when physical accelerator is present."""
    try:
      jax.devices(platform)
    except (RuntimeError, ValueError) as e:
      self.skipTest(f"Requires attached {platform.upper()} hardware: {e}")

    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / f"{platform}_attached_in")
    )
    output_base = str(
        pathlib.Path(self.temp_dir) / "models" / f"{platform}_attached_out"
    )
    config = offline_compiler.TargetConfig(
        target_topology_label=f"{platform}_attached",
        platform=platform,
        enable_deviceless_compilation=False,
    )
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
    )
    self.assertTrue(file_utils.exists(out_dir))
    manifest = persistence_lib.load(out_dir)
    fn = manifest.objects["serving_default"].function
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        fn.body.stable_hlo_body.compiled_executables[0].target_topology_label,
        f"{platform}_attached",
    )

  # ---------------------------------------------------------------------------
  # 3. Parameter and Feature Scenario Tests
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      ("inlined_payload", False),
      ("external_shlo_file", True),
  )
  def test_strip_stablehlo_payload(self, save_shlo_to_file: bool):
    """Verifies stripping of inlined and external StableHLO payloads."""
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "dlrm_model_input"),
        save_shlo_to_file=save_shlo_to_file,
    )
    output_base = str(pathlib.Path(self.temp_dir) / "models" / "dlrm_model_out")

    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )

    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        include_stablehlo=False,
        version=1,
    )

    manifest = persistence_lib.load(out_dir)
    fn = manifest.objects["serving_default"].function
    self.assertFalse(fn.body.stable_hlo_body.HasField("stable_hlo"))
    if save_shlo_to_file:
      shlo_file = str(pathlib.Path(out_dir) / "serving_default.shlo")
      self.assertFalse(file_utils.exists(shlo_file))
    self.assertLen(fn.body.stable_hlo_body.compiled_executables, 1)

  def test_auto_timestamp_and_overwrite_succeeds(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "linear_model_base")
    )
    output_base = str(
        pathlib.Path(self.temp_dir) / "models" / "linear_model_out"
    )
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    # 1. Test auto_timestamp=True with version=None
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=None,
        auto_timestamp=True,
    )
    self.assertTrue(pathlib.Path(out_dir).name.isdigit())
    self.assertTrue(file_utils.exists(out_dir))

    # 2. Test overwrite=True on existing explicit version directory
    out_dir_v1 = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
    )
    out_dir_v1_recompiled = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
        overwrite=True,
    )
    self.assertEqual(out_dir_v1, out_dir_v1_recompiled)
    self.assertTrue(file_utils.exists(out_dir_v1))

  def test_recompile_replaces_existing_target_label_and_cleans_up_old_binary(
      self,
  ):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "vit_model_input")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "models" / "vit_model_out")
    configs_initial = [
        offline_compiler.TargetConfig(
            target_topology_label="cpu_target",
            platform="cpu",
        ),
        offline_compiler.TargetConfig(
            target_topology_label="cpu_target_secondary",
            platform="cpu",
        ),
    ]
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=configs_initial,
        version=1,
    )
    old_cpu_bin = str(
        pathlib.Path(out_dir) / "serving_default_cpu_target_pjrt_executable.bin"
    )
    self.assertTrue(file_utils.exists(old_cpu_bin))

    # Recompile targeting cpu_target again on the exported model
    recompile_out = aot_recompiler.recompile(
        input_model_path=out_dir,
        output_model_path=str(
            pathlib.Path(self.temp_dir) / "models" / "vit_recompiled"
        ),
        target_configs=configs_initial[0],
        version=1,
    )
    manifest = persistence_lib.load(recompile_out)
    fn = manifest.objects["serving_default"].function
    labels = [
        e.target_topology_label
        for e in fn.body.stable_hlo_body.compiled_executables
    ]
    self.assertEqual(labels, ["cpu_target_secondary", "cpu_target"])

  def test_recompile_poly_fn_with_multiple_concrete_functions(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "poly_model_input")
    )
    manifest = persistence_lib.load(input_model)
    single_fn = manifest.objects["serving_default"].function
    manifest.objects["serving_default"].ClearField("function")
    manifest.objects["serving_default"].poly_fn.concrete_functions.extend(
        [single_fn, single_fn]
    )
    manifest_path = str(
        pathlib.Path(input_model) / constants.MANIFEST_FILE_PATH
    )
    with file_utils.open_file(manifest_path, "wb") as f:
      f.write(manifest.SerializeToString())

    output_base = str(pathlib.Path(self.temp_dir) / "models" / "poly_model_out")
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
    )
    manifest_out = persistence_lib.load(out_dir)
    poly = manifest_out.objects["serving_default"].poly_fn
    self.assertLen(poly.concrete_functions, 2)
    for concrete_fn in poly.concrete_functions:
      self.assertLen(concrete_fn.body.stable_hlo_body.compiled_executables, 1)
      self.assertEqual(
          concrete_fn.body.stable_hlo_body.compiled_executables[
              0
          ].target_topology_label,
          "cpu_target",
      )

  # ---------------------------------------------------------------------------
  # 4. Negative and Failure Scenario Tests
  # ---------------------------------------------------------------------------

  def test_overwrite_collision_raises_error(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "linear_collision_in")
    )
    output_base = str(
        pathlib.Path(self.temp_dir) / "models" / "linear_collision_out"
    )
    file_utils.mkdir_p(str(pathlib.Path(output_base) / "1"))

    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )

    with self.assertRaises(FileExistsError):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=output_base,
          target_configs=config,
          version=1,
          overwrite=False,
      )

  def test_unversioned_path_collision_raises_error(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "linear_unversioned_in")
    )
    output_path = str(pathlib.Path(self.temp_dir) / "models" / "custom_export")
    file_utils.mkdir_p(output_path)

    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    with self.assertRaises(FileExistsError):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=output_path,
          target_configs=config,
          auto_timestamp=False,
          overwrite=False,
      )

  def test_preexisting_tmp_directory_cleanup_succeeds(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "linear_tmp_in")
    )
    output_base = str(pathlib.Path(self.temp_dir) / "models" / "linear_tmp_out")
    stale_tmp = str(pathlib.Path(output_base) / "1.tmp")
    file_utils.mkdir_p(stale_tmp)
    with file_utils.open_file(
        str(pathlib.Path(stale_tmp) / "stale.bin"), "wb"
    ) as f:
      f.write(b"stale_bytes")

    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
    )
    self.assertTrue(file_utils.exists(out_dir))
    self.assertFalse(file_utils.exists(stale_tmp))

  def test_valid_frequent_path_index_resolution(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "valid_frequent_in"),
        save_shlo_to_file=True,
    )
    manifest = persistence_lib.load(input_model)
    fn = manifest.objects["serving_default"].function
    shlo = fn.body.stable_hlo_body.stable_hlo
    shlo_rel_path = shlo.file_system_location.string_path
    shlo.file_system_location.ClearField("string_path")
    shlo.file_system_location.frequent_path_idx = 0
    manifest.frequent_paths.append(shlo_rel_path)

    manifest_path = str(
        pathlib.Path(input_model) / constants.MANIFEST_FILE_PATH
    )
    with file_utils.open_file(manifest_path, "wb") as f:
      f.write(manifest.SerializeToString())

    output_base = str(
        pathlib.Path(self.temp_dir) / "models" / "valid_frequent_out"
    )
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model,
        output_model_path=output_base,
        target_configs=config,
        version=1,
    )
    self.assertTrue(file_utils.exists(out_dir))

  def test_stripped_input_model_raises_error(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "stripped_input")
    )
    # Manually strip StableHLO to simulate an AOT-only input model
    manifest = persistence_lib.load(input_model)
    for _, obj in manifest.objects.items():
      if obj.HasField("function") and obj.function.body.HasField(
          "stable_hlo_body"
      ):
        obj.function.body.stable_hlo_body.ClearField("stable_hlo")
    manifest_path = str(
        pathlib.Path(input_model) / constants.MANIFEST_FILE_PATH
    )
    with file_utils.open_file(manifest_path, "wb") as f:
      f.write(manifest.SerializeToString())

    output_base = str(pathlib.Path(self.temp_dir) / "models" / "stripped_out")
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )

    with self.assertRaisesRegex(ValueError, "exported as an AOT-only artifact"):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=output_base,
          target_configs=config,
          version=1,
      )

  def test_identical_input_output_paths_raises_error(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "identical_in")
    )
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    with self.assertRaisesRegex(ValueError, "cannot be identical"):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=input_model,
          target_configs=config,
      )

  def test_nonexistent_input_model_path_raises_file_not_found(self):
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    with self.assertRaises(FileNotFoundError):
      aot_recompiler.recompile(
          input_model_path=str(
              pathlib.Path(self.temp_dir) / "models" / "nonexistent_model"
          ),
          output_model_path=str(pathlib.Path(self.temp_dir) / "models" / "out"),
          target_configs=config,
      )

  def test_empty_target_configs_raises_value_error(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "empty_configs_in")
    )
    with self.assertRaisesRegex(ValueError, "target_configs must not be empty"):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=str(pathlib.Path(self.temp_dir) / "models" / "out"),
          target_configs=[],
      )

  def test_invalid_frequent_path_index_raises_value_error(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "invalid_idx_in"),
        save_shlo_to_file=True,
    )
    manifest = persistence_lib.load(input_model)
    fn = manifest.objects["serving_default"].function
    loc = fn.body.stable_hlo_body.stable_hlo.file_system_location
    loc.frequent_path_idx = 999
    manifest_path = str(
        pathlib.Path(input_model) / constants.MANIFEST_FILE_PATH
    )
    with file_utils.open_file(manifest_path, "wb") as f:
      f.write(manifest.SerializeToString())

    output_base = str(
        pathlib.Path(self.temp_dir) / "models" / "invalid_idx_out"
    )
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    with self.assertRaisesRegex(ValueError, "Invalid frequent_path_idx 999"):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=output_base,
          target_configs=config,
          version=1,
      )

    loc.frequent_path_idx = -1
    with file_utils.open_file(manifest_path, "wb") as f:
      f.write(manifest.SerializeToString())
    with self.assertRaisesRegex(ValueError, "Invalid frequent_path_idx -1"):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=output_base,
          target_configs=config,
          version=1,
      )

  def test_file_system_location_empty_raises_value_error(self):
    input_model = _create_neptune_model(
        str(pathlib.Path(self.temp_dir) / "models" / "empty_fsl_in"),
        save_shlo_to_file=True,
    )
    manifest = persistence_lib.load(input_model)
    fn = manifest.objects["serving_default"].function
    fn.body.stable_hlo_body.stable_hlo.file_system_location.Clear()
    manifest_path = str(
        pathlib.Path(input_model) / constants.MANIFEST_FILE_PATH
    )
    with file_utils.open_file(manifest_path, "wb") as f:
      f.write(manifest.SerializeToString())

    output_base = str(pathlib.Path(self.temp_dir) / "models" / "empty_fsl_out")
    config = offline_compiler.TargetConfig(
        target_topology_label="cpu_target",
        platform="cpu",
    )
    with self.assertRaisesRegex(
        ValueError, "neither string_path nor frequent_path_idx"
    ):
      aot_recompiler.recompile(
          input_model_path=input_model,
          output_model_path=output_base,
          target_configs=config,
          version=1,
      )


if __name__ == "__main__":
  absltest.main()
