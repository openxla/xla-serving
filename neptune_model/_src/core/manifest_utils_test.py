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

from collections.abc import Sequence
import os
import typing
from typing import Any
from absl.testing import absltest
from neptune_model._src.core import device_assignment
from neptune_model._src.core import manifest_utils
from neptune_model._src.core import serializable_function
from neptune_model._src.core import shlo_function
from neptune_model._src.core import unstructured_data
from neptune_model.protos import manifest_pb2


class ManifestUtilsTest(absltest.TestCase):

  def test_build_manifest_proto_with_subfolder(self):
    tmp_dir = self.create_tempdir().full_path
    subfolder = 'test_subfolder'

    fn = serializable_function.SerializableFunction(
        body=unstructured_data.UnstructuredDataWithExtName(
            proto=manifest_pb2.UnstructuredData(inlined_bytes=b'test'),
            subfolder=subfolder,
            ext_name='txt',
        ),
        input_signature={},
        output_signature={},
    )
    obm_module = {'test_fn': fn}

    manifest_proto = manifest_utils.build_manifest_proto(obm_module, tmp_dir)

    self.assertIn('test_fn', manifest_proto.objects)
    self.assertTrue(
        manifest_proto.objects['test_fn'].function.body.other.HasField(
            'file_system_location'
        )
    )
    file_path = manifest_proto.objects[
        'test_fn'
    ].function.body.other.file_system_location.string_path
    self.assertEqual(file_path, os.path.join(subfolder, 'test_fn.txt'))
    full_path = os.path.join(tmp_dir, file_path)
    with open(full_path, 'rb') as f:
      self.assertEqual(f.read(), b'test')

  def test_build_device_assignment_by_coords_proto(self):
    # Test with an empty list of device assignments.
    device_assignments = []
    proto = manifest_utils._build_device_assignment_by_coords_proto(
        device_assignments
    )
    self.assertEqual(proto, manifest_pb2.DeviceAssignmentByCoords())

    # Test with a list of fully populated device assignments.
    device_assignments = [
        device_assignment.DeviceAssignment(id=0, coords=[0, 1], core_on_chip=0),
        device_assignment.DeviceAssignment(id=1, coords=[1, 0], core_on_chip=1),
    ]
    proto = manifest_utils._build_device_assignment_by_coords_proto(
        device_assignments
    )

    self.assertLen(proto.devices, 2)
    # Check first device.
    self.assertEqual(proto.devices[0].id, 0)
    self.assertSequenceEqual(proto.devices[0].coords, [0, 1])
    self.assertEqual(proto.devices[0].core_on_chip, 0)
    # Check second device.
    self.assertEqual(proto.devices[1].id, 1)
    self.assertSequenceEqual(proto.devices[1].coords, [1, 0])
    self.assertEqual(proto.devices[1].core_on_chip, 1)

    # Test with a list of device assignments with optional fields None.
    device_assignments = [
        device_assignment.DeviceAssignment(id=0),
        device_assignment.DeviceAssignment(id=1),
        device_assignment.DeviceAssignment(id=2),
    ]
    proto = manifest_utils._build_device_assignment_by_coords_proto(
        device_assignments
    )
    self.assertLen(proto.devices, 3)
    # Check devices where optional fields were None.
    for i, device in enumerate(proto.devices):
      self.assertEqual(device.id, i)
      self.assertSequenceEqual(device.coords, [])
      self.assertEqual(device.core_on_chip, 0)  # Proto default

  def test_build_manifest_proto_empty_or_invalid_sequence_raises_value_error(
      self,
  ):
    """Verifies ValueError when passing empty or heterogeneous sequence."""
    tmp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(ValueError, r'Unsupported module value type'):
      manifest_utils.build_manifest_proto(
          typing.cast(Any, {'empty': []}), tmp_dir
      )
    with self.assertRaisesRegex(ValueError, r'Unsupported module value type'):
      manifest_utils.build_manifest_proto(
          typing.cast(Any, {'invalid': ['not_a_function']}), tmp_dir
      )

  def test_aot_build_manifest_proto_with_compiler_hook(self):
    tmp_dir = self.create_tempdir().full_path

    fn = shlo_function.ShloFunction(
        mlir_module_serialized=b'test_stablehlo',
        calling_convention_version=1,
        module_kept_var_idx=(0, 1),
        lowering_platforms=('cpu', 'cuda'),
        supplemental_info={},
        physical_in_dtypes=(),
        physical_out_dtypes=(),
        input_signature={},
        output_signature={},
    )
    obm_module = {'test_fn': fn}

    def custom_compiler_hook(
        mlir_bytes: bytes,
        module_kept_var_idx: Sequence[int],
        num_weights: int,
    ) -> list[manifest_utils.CompiledArtifact]:
      self.assertEqual(mlir_bytes, b'test_stablehlo')
      self.assertSequenceEqual(module_kept_var_idx, (0, 1))
      self.assertEqual(num_weights, 0)
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_bytes_cpu',
              kept_var_idx=[0],
              target_topology_label='cpu_topology_label',
              executable_version=b'version_cpu',
          ),
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_bytes_gpu',
              kept_var_idx=[1],
              target_topology_label='gpu_topology_label',
              executable_version=b'version_gpu',
          ),
      ]

    manifest_proto = manifest_utils.build_manifest_proto(
        obm_module, tmp_dir, compiler_hook=custom_compiler_hook
    )

    self.assertIn('test_fn', manifest_proto.objects)
    fn_obj = manifest_proto.objects['test_fn'].function

    self.assertLen(fn_obj.body.stable_hlo_body.compiled_executables, 2)

    # Verify CPU artifact
    expected_file_path_cpu = os.path.join(
        tmp_dir, 'test_fn_cpu_topology_label_pjrt_executable.bin'
    )
    self.assertTrue(os.path.exists(expected_file_path_cpu))
    with open(expected_file_path_cpu, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_bytes_cpu')

    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.file_system_location.string_path,
        'test_fn_cpu_topology_label_pjrt_executable.bin',
    )
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.mime_type,
        'application/x-ifrt-executable',
    )
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            0
        ].target_topology_label,
        'cpu_topology_label',
    )
    self.assertEqual(
        list(
            fn_obj.body.stable_hlo_body.compiled_executables[0].aot_kept_var_idx
        ),
        [0],
    )
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[0].executable_version,
        b'version_cpu',
    )

    # Verify GPU artifact
    expected_file_path_gpu = os.path.join(
        tmp_dir, 'test_fn_gpu_topology_label_pjrt_executable.bin'
    )
    self.assertTrue(os.path.exists(expected_file_path_gpu))
    with open(expected_file_path_gpu, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_bytes_gpu')

    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            1
        ].aot_executable.file_system_location.string_path,
        'test_fn_gpu_topology_label_pjrt_executable.bin',
    )
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            1
        ].aot_executable.mime_type,
        'application/x-ifrt-executable',
    )
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            1
        ].target_topology_label,
        'gpu_topology_label',
    )
    self.assertEqual(
        list(
            fn_obj.body.stable_hlo_body.compiled_executables[1].aot_kept_var_idx
        ),
        [1],
    )
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[1].executable_version,
        b'version_gpu',
    )

    # CRITICAL: Verify that stable_hlo_body.module_kept_var_idx was NOT mutated
    # and still holds the original JIT indices [0, 1].
    self.assertEqual(
        list(fn_obj.body.stable_hlo_body.module_kept_var_idx), [0, 1]
    )

  def test_aot_build_manifest_proto_poly_fn_with_compiler_hook(self):
    """Verifies AOT compilation for polymorphic functions (multi-bucket)."""
    tmp_dir = self.create_tempdir().full_path

    fn0 = shlo_function.ShloFunction(
        mlir_module_serialized=b'test_stablehlo_bucket0',
        calling_convention_version=1,
        module_kept_var_idx=(0,),
        lowering_platforms=('cpu',),
        supplemental_info={},
        physical_in_dtypes=(),
        physical_out_dtypes=(),
        input_signature={},
        output_signature={},
    )
    fn1 = shlo_function.ShloFunction(
        mlir_module_serialized=b'test_stablehlo_bucket1',
        calling_convention_version=1,
        module_kept_var_idx=(0, 1),
        lowering_platforms=('cpu',),
        supplemental_info={},
        physical_in_dtypes=(),
        physical_out_dtypes=(),
        input_signature={},
        output_signature={},
    )
    obm_module = {'poly_fn_obj': [fn0, fn1]}

    def custom_compiler_hook(
        mlir_bytes: bytes,
        module_kept_var_idx: Sequence[int],
        _num_weights: int,
    ) -> list[manifest_utils.CompiledArtifact]:
      label = (
          'bucket0_target'
          if mlir_bytes == b'test_stablehlo_bucket0'
          else 'bucket1_target'
      )
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_' + mlir_bytes,
              kept_var_idx=list(module_kept_var_idx),
              target_topology_label=label,
              executable_version=b'version_v1',
          )
      ]

    manifest_proto = manifest_utils.build_manifest_proto(
        obm_module, tmp_dir, compiler_hook=custom_compiler_hook
    )

    self.assertIn('poly_fn_obj', manifest_proto.objects)
    poly_obj = manifest_proto.objects['poly_fn_obj'].poly_fn
    self.assertLen(poly_obj.concrete_functions, 2)

    # Verify Bucket 0 concrete function
    cf0 = poly_obj.concrete_functions[0]
    self.assertLen(cf0.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        cf0.body.stable_hlo_body.compiled_executables[0].target_topology_label,
        'bucket0_target',
    )
    file0 = os.path.join(
        tmp_dir, '__poly_fn_obj_0_bucket0_target_pjrt_executable.bin'
    )
    self.assertTrue(os.path.exists(file0))
    with open(file0, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_test_stablehlo_bucket0')

    # Verify Bucket 1 concrete function
    cf1 = poly_obj.concrete_functions[1]
    self.assertLen(cf1.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        cf1.body.stable_hlo_body.compiled_executables[0].target_topology_label,
        'bucket1_target',
    )
    file1 = os.path.join(
        tmp_dir, '__poly_fn_obj_1_bucket1_target_pjrt_executable.bin'
    )
    self.assertTrue(os.path.exists(file1))
    with open(file1, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_test_stablehlo_bucket1')

  def test_aot_build_manifest_proto_special_character_label_sanitization(self):
    """Verifies that special characters in function names and labels are

    sanitized in filenames.
    """
    tmp_dir = self.create_tempdir().full_path

    fn = shlo_function.ShloFunction(
        mlir_module_serialized=b'test_stablehlo',
        calling_convention_version=1,
        module_kept_var_idx=(0,),
        lowering_platforms=('cpu',),
        supplemental_info={},
        physical_in_dtypes=(),
        physical_out_dtypes=(),
        input_signature={},
        output_signature={},
    )
    obm_module = {'fn:special/name': fn}

    def custom_compiler_hook(
        _mlir_bytes: bytes,
        _module_kept_var_idx: Sequence[int],
        _num_weights: int,
    ) -> list[manifest_utils.CompiledArtifact]:
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_bytes',
              kept_var_idx=[0],
              target_topology_label='tpu:df:2x2/custom.v1',
              executable_version=b'v1',
          )
      ]

    manifest_proto = manifest_utils.build_manifest_proto(
        obm_module, tmp_dir, compiler_hook=custom_compiler_hook
    )

    self.assertIn('fn:special/name', manifest_proto.objects)
    fn_obj = manifest_proto.objects['fn:special/name'].function
    self.assertLen(fn_obj.body.stable_hlo_body.compiled_executables, 1)

    compiled_exec = fn_obj.body.stable_hlo_body.compiled_executables[0]
    expected_filename = (
        'fn_special_name_tpu_df_2x2_custom_v1_pjrt_executable.bin'
    )
    self.assertEqual(
        compiled_exec.aot_executable.file_system_location.string_path,
        expected_filename,
    )
    full_path = os.path.join(tmp_dir, expected_filename)
    self.assertTrue(os.path.exists(full_path))
    with open(full_path, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_bytes')

    self.assertEqual(
        compiled_exec.target_topology_label, 'tpu:df:2x2/custom.v1'
    )

  def test_aot_build_manifest_proto_num_weights_forwarding(self):
    """Verifies that num_weights (derived from data_names) is forwarded to

    compiler hook.
    """
    tmp_dir = self.create_tempdir().full_path

    fn = shlo_function.ShloFunction(
        mlir_module_serialized=b'test_stablehlo',
        calling_convention_version=1,
        module_kept_var_idx=(0, 1, 2),
        lowering_platforms=('cpu',),
        supplemental_info={},
        physical_in_dtypes=(),
        physical_out_dtypes=(),
        input_signature={},
        output_signature={},
        data_names=['param_w0', 'param_w1', 'param_b0'],
    )
    obm_module = {'weighted_fn': fn}

    received_weights: list[int] = []

    def custom_compiler_hook(
        _mlir_bytes: bytes,
        _module_kept_var_idx: Sequence[int],
        num_weights: int,
    ) -> list[manifest_utils.CompiledArtifact]:
      received_weights.append(num_weights)
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_bytes',
              kept_var_idx=[0],
              target_topology_label='cpu_target',
              executable_version=b'v1',
          )
      ]

    manifest_utils.build_manifest_proto(
        obm_module, tmp_dir, compiler_hook=custom_compiler_hook
    )

    self.assertEqual(received_weights, [3])

  def test_include_stablehlo_false_without_artifacts_raises_error(self):
    """Verifies error when include_stablehlo=False without AOT artifacts."""
    tmp_dir = self.create_tempdir().full_path
    fn = shlo_function.ShloFunction(
        mlir_module_serialized=b'module {}',
        calling_convention_version=1,
        module_kept_var_idx=(0,),
        lowering_platforms=('cpu',),
        supplemental_info={},
        physical_in_dtypes=(),
        physical_out_dtypes=(),
        input_signature={},
        output_signature={},
    )
    empty_compiler_hook = lambda mlir, idx, num_weights: []

    with self.assertRaisesRegex(
        ValueError,
        r'contains no executable payload',
    ):
      manifest_utils.build_manifest_proto(
          {'test_fn': fn},
          tmp_dir,
          compiler_hook=empty_compiler_hook,
          include_stablehlo=False,
      )

  def test_build_manifest_proto_include_stablehlo_false_with_artifacts_succeeds(
      self,
  ):
    """Verifies that include_stablehlo=False succeeds with AOT artifacts."""
    tmp_dir = self.create_tempdir().full_path
    fn = shlo_function.ShloFunction(
        mlir_module_serialized=b'module {}',
        calling_convention_version=1,
        module_kept_var_idx=(0,),
        lowering_platforms=('cpu',),
        supplemental_info={},
        physical_in_dtypes=(),
        physical_out_dtypes=(),
        input_signature={},
        output_signature={},
    )

    def valid_compiler_hook(_mlir, idx, _num_weights):
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'real_pjrt_exec',
              kept_var_idx=list(idx),
              target_topology_label='cpu_target',
              executable_version=b'v1.0',
          )
      ]

    manifest = manifest_utils.build_manifest_proto(
        {'test_fn': fn},
        tmp_dir,
        compiler_hook=valid_compiler_hook,
        include_stablehlo=False,
    )
    fn_proto = manifest.objects['test_fn'].function.body.stable_hlo_body
    self.assertFalse(
        fn_proto.HasField('stable_hlo'), 'StableHLO payload should be omitted.'
    )
    self.assertLen(fn_proto.compiled_executables, 1)
    self.assertEqual(
        fn_proto.compiled_executables[0].target_topology_label, 'cpu_target'
    )


if __name__ == '__main__':
  absltest.main()
