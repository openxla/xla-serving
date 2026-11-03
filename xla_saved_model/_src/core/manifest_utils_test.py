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
import pathlib
import typing
from typing import Any

from absl.testing import absltest

from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src.core import device_assignment
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import manifest_utils
from xla_saved_model._src.core import serializable_function
from xla_saved_model._src.core import shlo_function
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import type_proto_utils
from xla_saved_model._src.core import unstructured_data


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
    self.assertEqual(file_path, str(pathlib.Path(subfolder) / 'test_fn.txt'))
    full_path = str(pathlib.Path(tmp_dir) / file_path)
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
    expected_file_path_cpu = str(
        pathlib.Path(tmp_dir) / 'test_fn_cpu_topology_label_pjrt_executable.bin'
    )
    self.assertTrue(file_utils.exists(expected_file_path_cpu))
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
    expected_file_path_gpu = str(
        pathlib.Path(tmp_dir) / 'test_fn_gpu_topology_label_pjrt_executable.bin'
    )
    self.assertTrue(file_utils.exists(expected_file_path_gpu))
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

    def custom_compiler_hook(mlir_bytes, module_kept_var_idx, *_):
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
    file0 = str(
        pathlib.Path(tmp_dir)
        / '__poly_fn_obj_0_bucket0_target_pjrt_executable.bin'
    )
    self.assertTrue(file_utils.exists(file0))
    with open(file0, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_test_stablehlo_bucket0')

    # Verify Bucket 1 concrete function
    cf1 = poly_obj.concrete_functions[1]
    self.assertLen(cf1.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        cf1.body.stable_hlo_body.compiled_executables[0].target_topology_label,
        'bucket1_target',
    )
    file1 = str(
        pathlib.Path(tmp_dir)
        / '__poly_fn_obj_1_bucket1_target_pjrt_executable.bin'
    )
    self.assertTrue(file_utils.exists(file1))
    with open(file1, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_test_stablehlo_bucket1')

  def test_aot_build_manifest_proto_special_character_label_sanitization(self):
    """Verifies that special characters in function names/labels are sanitized."""
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

    def custom_compiler_hook(*_):
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
    full_path = str(pathlib.Path(tmp_dir) / expected_filename)
    self.assertTrue(file_utils.exists(full_path))
    with open(full_path, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_bytes')

    self.assertEqual(
        compiled_exec.target_topology_label, 'tpu:df:2x2/custom.v1'
    )

  def test_aot_build_manifest_proto_num_weights_forwarding(self):
    """Verifies that num_weights is forwarded to compiler hook."""
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

    def custom_compiler_hook(mlir_bytes, module_kept_var_idx, num_weights):
      self.assertEqual(mlir_bytes, b'test_stablehlo')
      self.assertEqual(list(module_kept_var_idx), [0, 1, 2])
      self.assertEqual(num_weights, 3)
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
    empty_compiler_hook = lambda *_: []

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

    def valid_compiler_hook(mlir_bytes, idx, num_weights):
      self.assertEqual(mlir_bytes, b'module {}')
      self.assertEqual(num_weights, 0)
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

  def test_resolve_relative_path_string_path(self):
    fsl = manifest_pb2.FileSystemLocation(string_path='test/path/model.shlo')
    manifest = manifest_pb2.Manifest()
    self.assertEqual(
        manifest_utils.resolve_relative_path(fsl, manifest),
        'test/path/model.shlo',
    )

  def test_resolve_relative_path_frequent_path_idx(self):
    fsl = manifest_pb2.FileSystemLocation(frequent_path_idx=1)
    manifest = manifest_pb2.Manifest()
    manifest.frequent_paths.extend(['path0', 'path1', 'path2'])
    self.assertEqual(
        manifest_utils.resolve_relative_path(fsl, manifest), 'path1'
    )

  def test_resolve_relative_path_invalid_frequent_path_idx_raises_error(self):
    manifest = manifest_pb2.Manifest()
    manifest.frequent_paths.extend(['path0'])
    with self.assertRaisesRegex(ValueError, r'Invalid frequent_path_idx 5'):
      manifest_utils.resolve_relative_path(
          manifest_pb2.FileSystemLocation(frequent_path_idx=5), manifest
      )
    with self.assertRaisesRegex(ValueError, r'Invalid frequent_path_idx -1'):
      manifest_utils.resolve_relative_path(
          manifest_pb2.FileSystemLocation(frequent_path_idx=-1), manifest
      )

  def test_resolve_relative_path_empty_fsl_raises_error(self):
    manifest = manifest_pb2.Manifest()
    with self.assertRaisesRegex(
        ValueError, r'neither string_path nor frequent_path_idx'
    ):
      manifest_utils.resolve_relative_path(
          manifest_pb2.FileSystemLocation(), manifest
      )

  def test_get_functions_yields_function_and_poly_fn(self):
    manifest = manifest_pb2.Manifest()
    manifest.objects['single_fn'].function.visibility = manifest_pb2.PUBLIC
    poly_fn = manifest.objects['poly_fn'].poly_fn
    poly_fn.concrete_functions.add().visibility = manifest_pb2.PUBLIC
    poly_fn.concrete_functions.add().visibility = manifest_pb2.PUBLIC

    fns = list(manifest_utils.get_functions(manifest))
    fn_names = [name for name, _ in fns]
    self.assertCountEqual(fn_names, ['single_fn', 'poly_fn_0', 'poly_fn_1'])

  def test_extract_stablehlo_payload_inlined_and_external(self):
    tmp_dir = self.create_tempdir().full_path
    manifest = manifest_pb2.Manifest()

    # 1. Inlined payload
    inlined_proto = manifest_pb2.UnstructuredData(
        inlined_bytes=b'inlined_bytes'
    )
    payload = manifest_utils.extract_stablehlo_payload(
        inlined_proto, manifest, tmp_dir
    )
    self.assertEqual(payload, b'inlined_bytes')

    # 2. External file payload
    file_path = str(pathlib.Path(tmp_dir) / 'external.shlo')
    with open(file_path, 'wb') as f:
      f.write(b'external_bytes')

    external_proto = manifest_pb2.UnstructuredData(
        file_system_location=manifest_pb2.FileSystemLocation(
            string_path='external.shlo'
        )
    )
    recorded_paths = set()
    payload = manifest_utils.extract_stablehlo_payload(
        external_proto,
        manifest,
        tmp_dir,
        recorded_external_paths=recorded_paths,
    )
    self.assertEqual(payload, b'external_bytes')
    self.assertEqual(recorded_paths, {'external.shlo'})

  def test_extract_stablehlo_payload_stripped(self):
    with self.assertRaisesRegex(ValueError, 'stripped'):
      manifest_utils.extract_stablehlo_payload(
          manifest_pb2.UnstructuredData(), manifest_pb2.Manifest(), '/tmp'
      )

  def test_read_unstructured_data_bytes_inlined_string(self):
    data = manifest_pb2.UnstructuredData(inlined_string='text')
    self.assertEqual(
        manifest_utils.read_unstructured_data_bytes(data, '/unused'), b'text'
    )

  def test_read_unstructured_data_bytes_frequent_path_and_range(self):
    tmp_dir = self.create_tempdir()
    tmp_dir.create_file('blob', content='0123456789')
    manifest = manifest_pb2.Manifest(frequent_paths=['blob'])
    data = manifest_pb2.UnstructuredData(
        file_system_location=manifest_pb2.FileSystemLocation(
            frequent_path_idx=0, offset_in_byte=2, size_in_byte=3
        )
    )
    self.assertEqual(
        manifest_utils.read_unstructured_data_bytes(
            data, tmp_dir.full_path, manifest
        ),
        b'234',
    )
    with self.assertRaisesRegex(ValueError, 'manifest is needed'):
      manifest_utils.read_unstructured_data_bytes(data, tmp_dir.full_path)

  def test_read_unstructured_data_bytes_rejects_bad_locations(self):
    tmp_dir = self.create_tempdir().full_path
    for path, error, regex in (
        ('/abs/blob', ValueError, 'must be relative'),
        ('../blob', ValueError, "cannot contain '..'"),
        ('missing', FileNotFoundError, 'does not exist'),
    ):
      data = manifest_pb2.UnstructuredData(
          file_system_location=manifest_pb2.FileSystemLocation(
              string_path=path
          )
      )
      with self.subTest(path), self.assertRaisesRegex(error, regex):
        manifest_utils.read_unstructured_data_bytes(data, tmp_dir)

  def test_read_unstructured_data_bytes_unset(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported'):
      manifest_utils.read_unstructured_data_bytes(
          manifest_pb2.UnstructuredData(), '/unused'
      )

  def test_build_shlo_function_proto_with_original_signatures(self):
    orig_in = {
        'x': shlo_type.ShloTensorSpec(
            shape=(2, 8, 8, 3),
            dtype=shlo_type.ShloDType.f32,
            host_transpose=(0, 3, 1, 2),
        )
    }
    orig_out = {
        'y': shlo_type.ShloTensorSpec(
            shape=(2, 10),
            dtype=shlo_type.ShloDType.f32,
        )
    }
    fn = shlo_function.ShloFunction(
        mlir_module_serialized=b'test_stablehlo',
        calling_convention_version=1,
        module_kept_var_idx=(0,),
        lowering_platforms=('tpu',),
        supplemental_info={},
        physical_in_dtypes=(None,),
        physical_out_dtypes=(None,),
        input_signature={
            'x': shlo_type.ShloTensorSpec(
                shape=(384,), dtype=shlo_type.ShloDType.f32
            )
        },
        output_signature={
            'y': shlo_type.ShloTensorSpec(
                shape=(20,), dtype=shlo_type.ShloDType.f32
            )
        },
        original_input_signature=orig_in,
        original_output_signature=orig_out,
    )
    manifest_proto = manifest_utils.build_manifest_proto(
        {'test_fn': fn}, self.create_tempdir().full_path
    )
    body = manifest_proto.objects['test_fn'].function.body.stable_hlo_body
    self.assertTrue(body.HasField('original_input_signature'))
    self.assertTrue(body.HasField('original_output_signature'))
    self.assertEqual(
        type_proto_utils.from_type_proto(body.original_input_signature),
        orig_in,
    )
    self.assertEqual(
        type_proto_utils.from_type_proto(body.original_output_signature),
        orig_out,
    )


if __name__ == '__main__':
  absltest.main()
