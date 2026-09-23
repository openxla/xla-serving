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

"""Persistence tests (saving/loading the model)."""

from collections.abc import Sequence
import os

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format

from xla_saved_model.protos import history_pb2
from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import type_pb2
from xla_saved_model._src.core import constants
from xla_saved_model._src.core import device_assignment
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import manifest_utils
from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.core import serializable_function
from xla_saved_model._src.core import shlo_function
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import test_utils
from xla_saved_model._src.core import unstructured_data
from xla_saved_model._src.core import value


class PersistenceLibTest(test_utils.XsmTestCase, parameterized.TestCase):

  def test_save_succeeds_with_all_fields(self):
    target_dir: str = self.create_tempdir().full_path

    func1 = TestShloFunction(b'func1')
    func2 = TestShloFunction(b'func2')
    func3 = TestSerializableFunction(b'func3')

    persistence_lib.save(
        module={
            'my_func': func1,
            'my_seq_of_funcs': [func2, func3],
            'my_value': value.ExternalValue(
                data=manifest_pb2.UnstructuredData(inlined_string='my_value'),
            ),
        },
        target_dir=target_dir,
        options=persistence_lib.SaveOptions(
            version=2,
            supplementals={
                'supp_a': persistence_lib.GlobalSupplemental(
                    data=manifest_pb2.UnstructuredData(inlined_string='supp_a'),
                    save_as='supp_a.pb',
                ),
                'supp_b': persistence_lib.GlobalSupplemental(
                    data=manifest_pb2.UnstructuredData(inlined_string='supp_b')
                ),
            },
            visibility={
                'my_func': manifest_pb2.PRIVATE,
                'my_seq_of_funcs': manifest_pb2.PUBLIC,
            },
            device_assignment_by_coords=[
                device_assignment.DeviceAssignment(
                    id=0,
                    coords=[0, 1, 2],
                    core_on_chip=0,
                ),
                device_assignment.DeviceAssignment(
                    id=1,
                    coords=[1, 2, 3],
                    core_on_chip=1,
                ),
            ],
        ),
    )

    model_version_path = os.path.join(
        target_dir, constants.MODEL_VERSION_FILENAME
    )
    self.assertTrue(os.path.exists(model_version_path))

    history_path = os.path.join(target_dir, constants.HISTORY_FILENAME)
    self.assertTrue(os.path.exists(history_path))
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertEqual(history_proto.model_version, constants.MANIFEST_VERSION)
    self.assertLen(history_proto.records, 1)
    self.assertEqual(history_proto.records[0].action, 'EXPORT')

    actual_manifest = persistence_lib.load(target_dir)
    expected_manifest = manifest_pb2.Manifest(
        objects={
            'my_func': manifest_pb2.TopLevelObject(
                function=func1.manifest_repr('my_func', manifest_pb2.PRIVATE)
            ),
            'my_seq_of_funcs': manifest_pb2.TopLevelObject(
                poly_fn=manifest_pb2.PolymorphicFunction(
                    concrete_functions=[
                        func2.manifest_repr(
                            '__my_seq_of_funcs_0', manifest_pb2.PUBLIC
                        ),
                        func3.manifest_repr(
                            '__my_seq_of_funcs_1',
                            manifest_pb2.PUBLIC,
                        ),
                    ],
                )
            ),
            'my_value': manifest_pb2.TopLevelObject(
                value=manifest_pb2.Value(
                    external=manifest_pb2.ExternalValue(
                        data=manifest_pb2.UnstructuredData(
                            inlined_string='my_value'
                        )
                    )
                )
            ),
        },
        supplemental_info={
            'supp_a': manifest_pb2.UnstructuredData(
                file_system_location=manifest_pb2.FileSystemLocation(
                    string_path='supp_a.pb'
                ),
                mime_type='',
                version='',
            ),
            'supp_b': manifest_pb2.UnstructuredData(
                inlined_string='supp_b',
            ),
        },
        device_assignment_by_coords=manifest_pb2.DeviceAssignmentByCoords(
            devices=[
                manifest_pb2.DeviceAssignmentByCoords.Device(
                    id=0, coords=[0, 1, 2], core_on_chip=0
                ),
                manifest_pb2.DeviceAssignmentByCoords.Device(
                    id=1, coords=[1, 2, 3], core_on_chip=1
                ),
            ]
        ),
    )

    self.assertProtosEqual(
        expected_manifest,
        actual_manifest,
    )

    # Check that the supplemental files are written only for the functions
    # that don't have them inlined.
    self.assertTrue(os.path.exists(os.path.join(target_dir, 'supp_a.pb')))
    self.assertTrue(
        os.path.exists(
            os.path.join(target_dir, 'my_func_supp_a_supplemental.pb')
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(target_dir, 'my_func_supp_b_supplemental.pb')
        )
    )
    self.assertFalse(
        os.path.exists(os.path.join(target_dir, '__my_seq_of_funcs_0.pb'))
    )
    self.assertTrue(
        os.path.exists(os.path.join(target_dir, '__my_seq_of_funcs_1.pb'))
    )

  def test_save_without_stablehlo(self):
    target_dir: str = self.create_tempdir().full_path
    func1 = TestShloFunction(b'func1')

    def custom_compiler_hook(
        _mlir_bytes: bytes,
        module_kept_var_idx: Sequence[int],
        _num_weights: int,
    ):
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_bytes',
              kept_var_idx=list(module_kept_var_idx),
              target_topology_label='target_label',
              executable_version=b'v1',
          )
      ]

    persistence_lib.save(
        module={'my_func': func1},
        target_dir=target_dir,
        options=persistence_lib.SaveOptions(
            version=2,
            include_stablehlo=False,
            compiler_hook=custom_compiler_hook,
        ),
    )
    actual_manifest = persistence_lib.load(target_dir)
    self.assertTrue(
        actual_manifest.objects['my_func'].function.body.HasField(
            'stable_hlo_body'
        )
    )
    self.assertFalse(
        actual_manifest.objects[
            'my_func'
        ].function.body.stable_hlo_body.HasField('stable_hlo')
    )
    exec_protos = actual_manifest.objects[
        'my_func'
    ].function.body.stable_hlo_body.compiled_executables
    self.assertLen(exec_protos, 1)
    self.assertEqual(exec_protos[0].target_topology_label, 'target_label')
    self.assertEqual(list(exec_protos[0].aot_kept_var_idx), [0, 1])
    self.assertEqual(exec_protos[0].executable_version, b'v1')
    self.assertEqual(
        exec_protos[0].aot_executable.mime_type,
        constants.IFRT_EXECUTABLE_MIME_TYPE,
    )
    expected_bin_path = os.path.join(
        target_dir, 'my_func_target_label_pjrt_executable.bin'
    )
    self.assertTrue(os.path.exists(expected_bin_path))
    with open(expected_bin_path, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_bytes')

  def test_save_without_stablehlo_and_no_hook_fails(self):
    """Verifies ValueError when saving without StableHLO and no compiler

    hook.
    """
    target_dir: str = self.create_tempdir().full_path
    func1 = TestShloFunction(b'func1')
    with self.assertRaisesRegex(ValueError, 'contains no executable payload'):
      persistence_lib.save(
          module={'my_func': func1},
          target_dir=target_dir,
          options=persistence_lib.SaveOptions(
              version=2, include_stablehlo=False, compiler_hook=None
          ),
      )

  @parameterized.parameters(None, 1, 0, -1)
  def test_save_fails_with_invalid_version(self, invalid_version):
    """Verifies ValueError when saving with version < 2 or None."""
    target_dir: str = self.create_tempdir().full_path
    func1 = TestShloFunction(b'func1')
    with self.assertRaisesRegex(ValueError, 'Version must be >= 2'):
      persistence_lib.save(
          module={'my_func': func1},
          target_dir=target_dir,
          options=persistence_lib.SaveOptions(version=invalid_version),
      )

  def test_aot_save_and_load_with_compiler_hook(self):
    target_dir: str = self.create_tempdir().full_path
    func1 = TestShloFunction(b'test_stablehlo')

    def custom_compiler_hook(
        mlir_bytes: bytes, module_kept_var_idx: Sequence[int], num_weights: int
    ):
      self.assertEqual(mlir_bytes, b'test_stablehlo')
      self.assertSequenceEqual(module_kept_var_idx, (0, 1))
      self.assertEqual(num_weights, 0)
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_bytes_cpu',
              kept_var_idx=[0],
              target_topology_label='cpu_topology_label',
              executable_version=b'version_cpu',
          )
      ]

    persistence_lib.save(
        module={'my_func': func1},
        target_dir=target_dir,
        options=persistence_lib.SaveOptions(
            version=2,
            compiler_hook=custom_compiler_hook,
        ),
    )

    actual_manifest = persistence_lib.load(target_dir)
    self.assertIn('my_func', actual_manifest.objects)
    fn_obj = actual_manifest.objects['my_func'].function
    self.assertLen(fn_obj.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.file_system_location.string_path,
        'my_func_cpu_topology_label_pjrt_executable.bin',
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
    self.assertEqual(
        fn_obj.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.mime_type,
        constants.IFRT_EXECUTABLE_MIME_TYPE,
    )

  def test_aot_save_and_load_poly_fn_with_compiler_hook(self):
    """Verifies save and load roundtrip of polymorphic function with AOT

    compiler hook.
    """
    target_dir: str = self.create_tempdir().full_path
    func0 = TestShloFunction(b'test_stablehlo_bucket0')
    func1 = TestShloFunction(b'test_stablehlo_bucket1')

    def custom_compiler_hook(
        mlir_bytes: bytes, module_kept_var_idx: Sequence[int], _num_weights: int
    ):
      if mlir_bytes == b'test_stablehlo_bucket0':
        label = 'bucket0_target'
        kept_var_idx = [0]
      else:
        label = 'bucket1_target'
        kept_var_idx = list(module_kept_var_idx)
      return [
          manifest_utils.CompiledArtifact(
              pjrt_executable_bytes=b'compiled_' + mlir_bytes,
              kept_var_idx=kept_var_idx,
              target_topology_label=label,
              executable_version=b'version_v1',
          )
      ]

    persistence_lib.save(
        module={'my_poly_func': [func0, func1]},
        target_dir=target_dir,
        options=persistence_lib.SaveOptions(
            version=2,
            compiler_hook=custom_compiler_hook,
        ),
    )

    actual_manifest = persistence_lib.load(target_dir)
    self.assertIn('my_poly_func', actual_manifest.objects)
    poly_obj = actual_manifest.objects['my_poly_func'].poly_fn
    self.assertLen(poly_obj.concrete_functions, 2)

    cf0 = poly_obj.concrete_functions[0]
    self.assertLen(cf0.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        cf0.body.stable_hlo_body.compiled_executables[0].target_topology_label,
        'bucket0_target',
    )
    self.assertEqual(
        list(cf0.body.stable_hlo_body.compiled_executables[0].aot_kept_var_idx),
        [0],
    )
    self.assertEqual(
        cf0.body.stable_hlo_body.compiled_executables[0].executable_version,
        b'version_v1',
    )
    self.assertEqual(
        cf0.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.mime_type,
        constants.IFRT_EXECUTABLE_MIME_TYPE,
    )
    self.assertEqual(
        cf0.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.file_system_location.string_path,
        '__my_poly_func_0_bucket0_target_pjrt_executable.bin',
    )
    file0 = os.path.join(
        target_dir, '__my_poly_func_0_bucket0_target_pjrt_executable.bin'
    )
    self.assertTrue(os.path.exists(file0))
    with open(file0, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_test_stablehlo_bucket0')

    cf1 = poly_obj.concrete_functions[1]
    self.assertLen(cf1.body.stable_hlo_body.compiled_executables, 1)
    self.assertEqual(
        cf1.body.stable_hlo_body.compiled_executables[0].target_topology_label,
        'bucket1_target',
    )
    self.assertEqual(
        list(cf1.body.stable_hlo_body.compiled_executables[0].aot_kept_var_idx),
        [0, 1],
    )
    self.assertEqual(
        cf1.body.stable_hlo_body.compiled_executables[0].executable_version,
        b'version_v1',
    )
    self.assertEqual(
        cf1.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.mime_type,
        constants.IFRT_EXECUTABLE_MIME_TYPE,
    )
    self.assertEqual(
        cf1.body.stable_hlo_body.compiled_executables[
            0
        ].aot_executable.file_system_location.string_path,
        '__my_poly_func_1_bucket1_target_pjrt_executable.bin',
    )
    file1 = os.path.join(
        target_dir, '__my_poly_func_1_bucket1_target_pjrt_executable.bin'
    )
    self.assertTrue(os.path.exists(file1))
    with open(file1, 'rb') as f:
      self.assertEqual(f.read(), b'compiled_test_stablehlo_bucket1')

  def test_load_fails_version_file_missing(self):
    target_dir: str = self.create_tempdir().full_path

    with self.assertRaisesRegex(Exception, constants.MODEL_VERSION_FILENAME):
      persistence_lib.load(target_dir)

  def test_load_fails_with_unsupported_version(self):
    target_dir: str = self.create_tempdir().full_path

    path = os.path.join(target_dir, constants.MODEL_VERSION_FILENAME)
    file_content = """
    version: "42"
    manifest_file_path: "test/path"
    mime_type: "test_mime_type; application/foo"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(
        ValueError, 'Unsupported manifest version "42"'
    ):
      persistence_lib.load(target_dir)

  def test_load_fails_with_unsupported_mime_type(self):
    target_dir: str = self.create_tempdir().full_path

    path = os.path.join(target_dir, constants.MODEL_VERSION_FILENAME)
    file_content = """
    version: "0.0.1"
    mime_type: "text/plain"
    manifest_file_path: "test/path"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(
        ValueError, 'Unsupported manifest type "text/plain"'
    ):
      persistence_lib.load(target_dir)

  def test_load_fails_with_absolute_manifest_path(self):
    target_dir: str = self.create_tempdir().full_path

    path = os.path.join(target_dir, constants.MODEL_VERSION_FILENAME)
    file_content = f"""
    version: "{constants.MANIFEST_VERSION}"
    mime_type: "{constants.MANIFEST_MIME_TYPE}"
    manifest_file_path: "/absolute/path/manifest.pb"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(
        ValueError, 'Manifest file path must be relative'
    ):
      persistence_lib.load(target_dir)

  def _save_minimal_model(self, target_dir: str) -> manifest_pb2.Manifest:
    persistence_lib.save(
        module={
            'my_value': value.ExternalValue(
                data=manifest_pb2.UnstructuredData(inlined_string='my_value'),
            ),
        },
        target_dir=target_dir,
        options=persistence_lib.SaveOptions(
            version=2,
        ),
    )
    return manifest_pb2.Manifest(
        objects={
            'my_value': manifest_pb2.TopLevelObject(
                value=manifest_pb2.Value(
                    external=manifest_pb2.ExternalValue(
                        data=manifest_pb2.UnstructuredData(
                            inlined_string='my_value'
                        )
                    )
                )
            ),
        },
    )



class TestShloFunctionSupplementalInfo(
    shlo_function.ShloFunctionSupplementalInfo
):

  __test__ = False

  def __init__(self, b: bytes):
    self._bytes = b

  def serializable_to_proto(
      self,
  ) -> unstructured_data.UnstructuredDataWithExtName:
    return unstructured_data.UnstructuredDataWithExtName(
        proto=manifest_pb2.UnstructuredData(inlined_bytes=self._bytes),
        ext_name='pb',
    )


class TestShloFunction(shlo_function.ShloFunction):

  __test__ = False

  def __init__(self, b: bytes):
    self.bytes = b
    self.input_signature = shlo_type.ShloTensorSpec(
        (1,), shlo_type.ShloDType.f32
    )
    self.output_signature = shlo_type.ShloTensorSpec(
        (
            1,
            2,
        ),
        shlo_type.ShloDType.f16,
    )
    self.mlir_module_serialized = b
    self.calling_convention_version = 42
    self.lowering_platforms = ('a', 'b')
    self.module_kept_var_idx = (0, 1)
    self.supplemental_info = {
        'supp_a': TestShloFunctionSupplementalInfo(b'supp_a'),
        'supp_b': TestShloFunctionSupplementalInfo(b'supp_b'),
    }
    self.physical_in_dtypes = (None, shlo_type.ShloDType.f32)
    self.physical_out_dtypes = (shlo_type.ShloDType.f16, None)

  def manifest_repr(
      self, key: str, visibility: manifest_pb2.Visibility
  ) -> manifest_pb2.Function:
    return manifest_pb2.Function(
        signature=type_pb2.FunctionSignature(
            input=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[type_pb2.DimensionSize(size=1)]
                            )
                        ),
                        dtype=type_pb2.DType.f32,
                    )
                )
            ),
            output=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[
                                    type_pb2.DimensionSize(size=1),
                                    type_pb2.DimensionSize(size=2),
                                ]
                            )
                        ),
                        dtype=type_pb2.DType.f16,
                    )
                )
            ),
        ),
        body=manifest_pb2.FunctionBody(
            stable_hlo_body=manifest_pb2.StableHloFunctionBody(
                stable_hlo=manifest_pb2.UnstructuredData(
                    inlined_bytes=self.bytes,
                    mime_type='application/x.mlir-stablehlo',
                    version='1.0',
                ),
                calling_convention_version=42,
                lowering_platforms=['a', 'b'],
                module_kept_var_idx=[0, 1],
                supplemental_info={
                    'supp_a': manifest_pb2.UnstructuredData(
                        file_system_location=manifest_pb2.FileSystemLocation(
                            string_path=f'{key}_supp_a_supplemental.pb'
                        ),
                        mime_type='',
                        version='',
                    ),
                    'supp_b': manifest_pb2.UnstructuredData(
                        file_system_location=manifest_pb2.FileSystemLocation(
                            string_path=f'{key}_supp_b_supplemental.pb'
                        ),
                        mime_type='',
                        version='',
                    ),
                },
            )
        ),
        visibility=visibility,
    )


class TestSerializableFunction(serializable_function.SerializableFunction):

  __test__ = False

  def __init__(self, b: bytes):
    self.input_signature = shlo_type.ShloTensorSpec(
        (1,), shlo_type.ShloDType.f32
    )
    self.output_signature = shlo_type.ShloTensorSpec(
        (
            1,
            2,
        ),
        shlo_type.ShloDType.f16,
    )
    self.body = unstructured_data.UnstructuredDataWithExtName(
        proto=manifest_pb2.UnstructuredData(inlined_bytes=b),
        ext_name='pb',
    )

  def manifest_repr(
      self, key: str, visibility: manifest_pb2.Visibility
  ) -> manifest_pb2.Function:
    return manifest_pb2.Function(
        signature=type_pb2.FunctionSignature(
            input=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[type_pb2.DimensionSize(size=1)]
                            )
                        ),
                        dtype=type_pb2.DType.f32,
                    )
                )
            ),
            output=type_pb2.Type(
                leaf=type_pb2.LeafType(
                    tensor_type=type_pb2.TensorType(
                        shape=type_pb2.Shape(
                            shape_with_known_rank=type_pb2.ShapeWithKnownRank(
                                dimension_sizes=[
                                    type_pb2.DimensionSize(size=1),
                                    type_pb2.DimensionSize(size=2),
                                ]
                            )
                        ),
                        dtype=type_pb2.DType.f16,
                    )
                )
            ),
        ),
        body=manifest_pb2.FunctionBody(
            other=manifest_pb2.UnstructuredData(
                file_system_location=manifest_pb2.FileSystemLocation(
                    string_path=f'{key}.pb',
                ),
                mime_type='',
                version='',
            )
        ),
        visibility=visibility,
    )


if __name__ == '__main__':
  absltest.main()
