# Copyright 2026 Google LLC
# Copyright 2026 Arm Ltd. and affiliates
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

"""Tests for CpuEmbeddingModule Python wrapper."""

import os
from absl.testing import absltest
from google.protobuf import text_format
import jax
import numpy as np
from data.kernels.cpu_embedding import cpu_embedding_lookup_transform_pybind
from data.kernels.cpu_embedding import cpu_embedding_pb2
from data.kernels.cpu_embedding import cpu_embedding_serializer_pybind
from data.kernels.cpu_embedding import unpack_host_buffer_vector_module
from data.ops import cpu_embedding
from data.processor.python import composite_processor_pybind
from data.processor.python import shape_dtype_struct


class CpuEmbeddingModuleTest(absltest.TestCase):

  def _execute(self, module, ids, row_pointers=(), weights=()):
    def pack(arrays):
      return composite_processor_pybind.Value([
          composite_processor_pybind.HostBuffer.from_dlpack(array)
          for array in arrays
      ])

    context = composite_processor_pybind.ExecuteContext()
    context.inputs = [pack(ids), pack(row_pointers), pack(weights)]
    module.transform.execute(context)
    self.assertLen(context.outputs, 1)
    return [
        np.from_dlpack(buffer).copy()
        for buffer in context.outputs[0].as_encapsulated_type()
    ]

  def test_resource_and_module_lifecycle(self):
    # Setup mock matrix values
    table_data = {
        "table1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    }
    combine_modes = {"table1": "MEAN"}
    feature_specs = [{
        "name": "table1",
        "table_name": "table1",
        "max_sequence_length": 5,
    }]

    # Instantiate Python Resource (triggers serialization)
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )

    # Verify assets were serialized
    self.assertLen(resource.assets, 2)
    manifest_asset = None
    table_asset = None
    for asset in resource.assets:
      if asset.name == cpu_embedding.MANIFEST_ASSET_NAME:
        manifest_asset = asset
      elif asset.name == "table1":
        table_asset = asset

    self.assertIsNotNone(manifest_asset)
    self.assertIsNotNone(table_asset)
    self.assertTrue(os.path.exists(manifest_asset.path))
    self.assertTrue(os.path.exists(table_asset.path))

    # Instantiate Python Module
    module = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_name="table1",
        output_feature_name="table1_out",
        output_shape=[1, 3],
    )

    # Verify module interface
    self.assertIsNotNone(module.transform)

    # Verify assets mapping
    assets_map = module.export_assets()
    self.assertEqual(
        assets_map,
        {
            f"assets/{cpu_embedding.MANIFEST_ASSET_NAME}": manifest_asset.path,
            "assets/table1": table_asset.path,
        },
    )

    # Verify output signature
    out_sig = module.get_output_signature(None)
    self.assertIn("table1_out", out_sig)
    self.assertEqual(out_sig["table1_out"].shape, (1, 3))
    self.assertEqual(out_sig["table1_out"].dtype, jax.numpy.float32)

    # Cleanup resource files
    resource.close()

  # Verify the public mapped-table path through execution.
  def test_mapped_resource_uses_original_asset_and_executes(self):
    table_data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    table_path = os.path.join(self.create_tempdir().full_path, "table.bin")
    table_data.tofile(table_path)
    resource = cpu_embedding.CpuEmbeddingResource.from_mapped_tables(
        {
            "feature1": cpu_embedding.MappedEmbeddingTable(
                path=table_path, shape=table_data.shape
            )
        },
        {"feature1": "SUM"},
        [{"name": "feature1", "table_name": "feature1"}],
        max_cpu_slots=3,
        block_size=17,
    )
    self.addCleanup(resource.close)

    assets = {asset.name: asset for asset in resource.assets}
    table_asset = assets["feature1"]
    self.assertEqual(table_asset.path, table_path)
    self.assertEmpty(table_asset.content)
    self.assertEqual(resource.get_embedding_dim("feature1"), 3)

    with open(
        assets[cpu_embedding.MANIFEST_ASSET_NAME].path, encoding="utf-8"
    ) as manifest_file:
      manifest = text_format.Parse(
          manifest_file.read(), cpu_embedding_pb2.CpuEmbeddingManifest()
      )
    self.assertEqual(manifest.max_cpu_slots, 3)
    self.assertEqual(manifest.block_size, 17)

    module = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_name="feature1",
        output_feature_name="feature1_out",
        output_shape=[2, 3],
        dense_lookup=True,
    )
    ids = np.array([[0, 2], [1, 1]], dtype=np.int32)
    result = self._execute(module, [ids])[0]
    np.testing.assert_array_equal(
        result,
        np.array([[8.0, 10.0, 12.0], [8.0, 10.0, 12.0]], np.float32),
    )

    dense_transform = cpu_embedding.DenseCpuEmbeddingLookupTransform(
        resource, "feature1", pad_value=-1
    )
    dense_result = dense_transform(
        composite_processor_pybind.HostBuffer.from_dlpack(ids)
    )
    np.testing.assert_array_equal(np.from_dlpack(dense_result), result)

  # Verify invalid mapped-table descriptors are rejected.
  def test_mapped_resource_rejects_invalid_descriptors(self):
    temp_dir = self.create_tempdir().full_path
    table_path = os.path.join(temp_dir, "table.bin")
    np.arange(6, dtype=np.float32).tofile(table_path)
    valid = cpu_embedding.MappedEmbeddingTable(table_path, (2, 3))
    combine_modes = {"table": "SUM"}
    feature_specs = [{"name": "feature", "table_name": "table"}]

    invalid_cases = [
        ("../table", valid, "Invalid table name"),
        (
            "table",
            cpu_embedding.MappedEmbeddingTable(table_path, (6,)),
            "exact 2-D shape",
        ),
        (
            "table",
            cpu_embedding.MappedEmbeddingTable(table_path, (0, 3)),
            "positive integers",
        ),
    ]
    for table_name, descriptor, error_pattern in invalid_cases:
      with self.subTest(error_pattern=error_pattern):
        with self.assertRaisesRegex(ValueError, error_pattern):
          cpu_embedding.CpuEmbeddingResource.from_mapped_tables(
              {table_name: descriptor}, combine_modes, feature_specs
          )

    with self.assertRaisesRegex(TypeError, "MappedEmbeddingTable"):
      cpu_embedding.CpuEmbeddingResource.from_mapped_tables(
          {"table": table_path}, combine_modes, feature_specs
      )

  def test_processor_execution_and_orbax_export(self):
    # Setup resource and module
    table_data = {
        "feature1": np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
    }
    combine_modes = {"feature1": "MEAN"}
    feature_specs = [{
        "name": "feature1",
        "table_name": "feature1",
    }]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )
    module = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_name="feature1",
        output_feature_name="feature1_out",
        output_shape=[1, 3],
    )

    # Test Execution via execute()
    ids = np.array([0, 0, 1], dtype=np.int64)
    row_pointers = np.array([0, 1, 3], dtype=np.int32)
    weights = np.array([1.0, 0.5, 0.5], dtype=np.float32)

    out_arr = self._execute(module, [ids], [row_pointers], [weights])[0]
    expected_output = np.array(
        [[1.0, 2.0, 3.0], [2.5, 3.5, 4.5]], dtype=np.float32
    )
    np.testing.assert_allclose(out_arr, expected_output)
    resource.close()

  def test_dense_processor_execution(self):
    # Setup resource and module with dense_lookup=True
    table_data = {
        "feature1": np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
    }
    combine_modes = {"feature1": "SUM"}
    feature_specs = [{
        "name": "feature1",
        "table_name": "feature1",
    }]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )
    module = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_name="feature1",
        output_feature_name="feature1_out",
        output_shape=[2, 3],
        dense_lookup=True,
    )

    # Using ExecuteContext directly for internal module execution

    # Test Execution via process() using 2D array of IDs [2, 2]
    ids = np.array([[0, 1], [1, 2]], dtype=np.int32)

    out_arr = self._execute(module, [ids])[0]
    # Expected output with SUM combiner:
    # Row 0: table1[0] + table1[1] = [1.0, 2.0, 3.0] + [4.0, 5.0, 6.0]
    # = [5.0, 7.0, 9.0]
    # Row 1: table1[1] + table1[2] = [4.0, 5.0, 6.0] + [7.0, 8.0, 9.0]
    # = [11.0, 13.0, 15.0]
    expected_output = np.array(
        [[5.0, 7.0, 9.0], [11.0, 13.0, 15.0]], dtype=np.float32
    )
    np.testing.assert_allclose(out_arr, expected_output)

    resource.close()

  def test_multi_lookup_execution(self):
    # Setup resource with 2 tables
    table_data = {
        "tableA": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "tableB": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    }
    combine_modes = {"tableA": "MEAN", "tableB": "MEAN"}
    feature_specs = [
        {"name": "tableA", "table_name": "tableA"},
        {"name": "tableB", "table_name": "tableB"},
    ]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )

    # Instantiate multi-lookup module
    module = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_names=["tableA", "tableB"],
        output_feature_names=["outA", "outB"],
        output_shapes=[[1, 2], [1, 2]],
    )

    # Verify assets mapping (both should be exported)
    assets_map = module.export_assets()
    self.assertIn(
        f"assets/{cpu_embedding_serializer_pybind.MANIFEST_ASSET_NAME}",
        assets_map,
    )
    self.assertIn("assets/tableA", assets_map)
    self.assertIn("assets/tableB", assets_map)

    # Verify output signatures
    out_sig = module.get_output_signature(None)
    self.assertEqual(out_sig["outA"].shape, (1, 2))
    self.assertEqual(out_sig["outB"].shape, (1, 2))

    # Using ExecuteContext directly for internal module execution

    # Table A Inputs
    ids_a = np.array([0, 1], dtype=np.int64)
    rp_a = np.array([0, 2], dtype=np.int32)
    w_a = np.array([1.0, 1.0], dtype=np.float32)

    # Table B Inputs
    ids_b = np.array([1, 1, 0], dtype=np.int64)
    rp_b = np.array([0, 3], dtype=np.int32)
    w_b = np.array([0.5, 0.25, 0.25], dtype=np.float32)

    out_a, out_b = self._execute(
        module, [ids_a, ids_b], [rp_a, rp_b], [w_a, w_b]
    )

    # Expected out_a: tableA[0]*1.0 + tableA[1]*1.0 / sum_weights(2.0)
    # = [1.0, 2.0] + [3.0, 4.0] / 2.0 = [2.0, 3.0]
    np.testing.assert_allclose(out_a, np.array([[2.0, 3.0]], dtype=np.float32))
    # Expected out_b: tableB[1]*0.5 + tableB[1]*0.25 + tableB[0]*0.25
    # = [3.5, 4.0] + [1.75, 2.0] + [1.25, 1.5] = [6.5, 7.5]
    np.testing.assert_allclose(out_b, np.array([[6.5, 7.5]], dtype=np.float32))

    resource.close()

  def test_dense_multi_lookup_execution(self):
    # Setup resource with 2 tables
    table_data = {
        "tableA": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "tableB": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    }
    combine_modes = {"tableA": "MEAN", "tableB": "MEAN"}
    feature_specs = [
        {"name": "tableA", "table_name": "tableA"},
        {"name": "tableB", "table_name": "tableB"},
    ]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )

    # Instantiate dense multi-lookup module
    module = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_names=["tableA", "tableB"],
        output_feature_names=["outA", "outB"],
        output_shapes=[[1, 2], [1, 2]],
        dense_lookup=True,
    )

    # Verify output signatures
    out_sig = module.get_output_signature(None)
    self.assertEqual(out_sig["outA"].shape, (1, 2))
    self.assertEqual(out_sig["outB"].shape, (1, 2))

    # Using ExecuteContext directly for internal module execution

    # Pass 2D arrays directly (shape [1, 2], containing int32 indices)
    ids_a = np.array([[0, 1]], dtype=np.int32)
    ids_b = np.array([[1, 1]], dtype=np.int32)

    out_a, out_b = self._execute(module, [ids_a, ids_b])

    # Combined tableA[0, 1] MEAN = [1.0, 2.0] + [3.0, 4.0] / 2.0 = [2.0, 3.0]
    np.testing.assert_allclose(out_a, np.array([[2.0, 3.0]], dtype=np.float32))

    # Combined tableB[1, 1] MEAN = [7.0, 8.0]
    np.testing.assert_allclose(out_b, np.array([[7.0, 8.0]], dtype=np.float32))

    resource.close()

  def test_path_traversal_blocked(self):
    table_data = {"../bad_table": np.array([[1.0]], dtype=np.float32)}
    combine_modes = {"../bad_table": "MEAN"}
    feature_specs = [{
        "name": "../bad_table",
        "table_name": "../bad_table",
    }]

    with self.assertRaisesRegex(ValueError, "path traversal"):
      _ = cpu_embedding.CpuEmbeddingResource(
          table_data, combine_modes, feature_specs
      )

  def test_dense_cpu_embedding_lookup_transform_direct_call(self):
    table_data = {
        "feat1": np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
        )
    }
    combine_modes = {"feat1": "SUM"}
    feature_specs = [{"name": "feat1", "table_name": "feat1"}]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )

    transform = cpu_embedding.DenseCpuEmbeddingLookupTransform(
        resource=resource,
        feature_names="feat1",
        pad_value=0,
    )

    # Verify output signature
    in_spec = (jax.ShapeDtypeStruct(shape=(2, 3), dtype=np.int32),)
    out_sig = transform.get_output_signature(in_spec)
    self.assertEqual(out_sig[0].shape, (2, 2))
    self.assertEqual(out_sig[0].dtype, jax.numpy.float32)

    # Test direct call with HostBuffer
    input_ids = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int32)
    hb_input = composite_processor_pybind.HostBuffer.from_dlpack(input_ids)
    output_hb = transform(hb_input)
    output_arr = np.from_dlpack(output_hb)

    # Row 0: [1] + [2] = [3.0, 3.0]
    # Row 1: [3] = [3.0, 3.0]
    expected = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(output_arr, expected)
    resource.close()

  def test_cpu_embedding_functional_op(self):
    table_data = {
        "feat1": np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        "feat2": np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    }
    combine_modes = {"feat1": "SUM", "feat2": "SUM"}
    feature_specs = [
        {"name": "feat1", "table_name": "feat1"},
        {"name": "feat2", "table_name": "feat2"},
    ]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )

    # Single feature call
    ids1 = np.array([[1, 0]], dtype=np.int32)
    hb1 = composite_processor_pybind.HostBuffer.from_dlpack(ids1)
    out1 = cpu_embedding.cpu_embedding(
        hb1, resource=resource, feature_name="feat1", pad_value=0
    )
    np.testing.assert_allclose(
        np.from_dlpack(out1), np.array([[1.0, 2.0]], dtype=np.float32)
    )

    # Multi-feature call
    ids2 = np.array([[1, 0]], dtype=np.int32)
    hb1 = composite_processor_pybind.HostBuffer.from_dlpack(ids1)
    hb2 = composite_processor_pybind.HostBuffer.from_dlpack(ids2)
    out1_multi, out2_multi = cpu_embedding.cpu_embedding(
        [hb1, hb2],
        resource=resource,
        feature_names=["feat1", "feat2"],
        pad_value=0,
    )
    np.testing.assert_allclose(
        np.from_dlpack(out1_multi), np.array([[1.0, 2.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.from_dlpack(out2_multi),
        np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
    )
    resource.close()

  def test_unpack_host_buffer_vector_module_lifecycle(self):
    module_single = (
        unpack_host_buffer_vector_module.UnpackHostBufferVectorModule(
            output_count=1,
            output_shapes=[[2, 4]],
        )
    )
    module_single.initialize(None)
    self.assertIsNotNone(module_single.transform)
    sig_single = module_single.get_output_signature(None)
    self.assertEqual(sig_single.shape, (2, 4))
    self.assertEqual(sig_single.dtype, jax.numpy.float32)

    module_multi = (
        unpack_host_buffer_vector_module.UnpackHostBufferVectorModule(
            output_count=2,
            output_shapes=[[2, 4], [2, 2]],
            dtype=jax.numpy.int32,
        )
    )
    module_multi.initialize(None)
    sig_multi = module_multi.get_output_signature(None)
    self.assertLen(sig_multi, 2)
    self.assertEqual(sig_multi[0].shape, (2, 4))
    self.assertEqual(sig_multi[1].shape, (2, 2))
    self.assertEqual(sig_multi[0].dtype, jax.numpy.int32)
    self.assertEqual(sig_multi[1].dtype, jax.numpy.int32)

  def test_dense_cpu_embedding_lookup_transform_explicit_dimensions_and_edge_cases(
      self,
  ):
    table_data = {
        "feat1": np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        "feat2": np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    }
    combine_modes = {"feat1": "SUM", "feat2": "SUM"}
    feature_specs = [
        {"name": "feat1", "table_name": "feat1"},
        {"name": "feat2", "table_name": "feat2"},
    ]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )

    # Test explicit integer embedding_dim
    t_int_dim = cpu_embedding.DenseCpuEmbeddingLookupTransform(
        resource=resource,
        feature_names=["feat1", "feat2"],
        embedding_dim=4,
    )
    t_int_dim.initialize(None)
    self.assertLen(t_int_dim._embedding_dims, 2)
    self.assertEqual(t_int_dim._embedding_dims, [4, 4])
    self.assertIn(
        f"assets/{cpu_embedding.MANIFEST_ASSET_NAME}", t_int_dim.export_assets()
    )

    # Test explicit sequence embedding_dim
    t_seq_dim = cpu_embedding.DenseCpuEmbeddingLookupTransform(
        resource=resource,
        feature_names=["feat1", "feat2"],
        embedding_dim=[2, 3],
    )
    self.assertEqual(t_seq_dim._embedding_dims, [2, 3])

    # Test empty input signature edge case
    empty_sig = t_seq_dim.get_output_signature([])
    self.assertEqual(empty_sig, [])

    resource.close()

  def test_cpu_embedding_lookup_transform_wrapper(self):
    table_data = {
        "feat1": np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32),
    }
    combine_modes = {"feat1": "SUM"}
    feature_specs = [{"name": "feat1", "table_name": "feat1"}]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )

    wrapper = cpu_embedding.CpuEmbeddingLookupTransformWrapper(
        resource=resource,
        feature_name="feat1",
        embedding_dim=2,
        output_feature_name="feat1_out",
    )
    wrapper.initialize(None)
    self.assertIsNotNone(wrapper.transform)
    self.assertIn(
        f"assets/{cpu_embedding.MANIFEST_ASSET_NAME}", wrapper.export_assets()
    )
    out_sig = wrapper.get_output_signature(None)
    self.assertIsNotNone(out_sig)
    resource.close()

  def test_dynamic_output_signatures_and_negative_dim_rejection(self):
    table_data = {
        "feat1": np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        "feat2": np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    }
    combine_modes = {"feat1": "SUM", "feat2": "SUM"}
    feature_specs = [
        {"name": "feat1", "table_name": "feat1"},
        {"name": "feat2", "table_name": "feat2"},
    ]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )
    self.addCleanup(resource.close)

    # CpuEmbeddingModule single-feature with dynamic batch dimension.
    mod_single = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_name="feat1",
        output_feature_name="feat1_out",
        output_shape=[None, 2],
    )
    sig_single = mod_single.get_output_signature(None)
    self.assertEqual(sig_single["feat1_out"].shape, (None, 2))
    self.assertEqual(sig_single["feat1_out"].dtype, np.float32)

    # Single-feature output_shapes accepts both 1D and 2D, and rejects empty [].
    for single_shapes in ([None, 2], [[None, 2]]):
      mod_single_shapes = cpu_embedding.CpuEmbeddingModule(
          resource,
          feature_names=["feat1"],
          output_feature_names=["feat1_out"],
          output_shapes=single_shapes,
      )
      self.assertEqual(
          mod_single_shapes.get_output_signature(None)["feat1_out"].shape,
          (None, 2),
      )

    with self.assertRaisesRegex(ValueError, "Sequence length mismatch"):
      cpu_embedding.CpuEmbeddingModule(
          resource,
          feature_names=["feat1"],
          output_feature_names=["feat1_out"],
          output_shapes=[],
      )
    with self.assertRaisesRegex(
        ValueError, "Must provide output_shapes or output_shape"
    ):
      cpu_embedding.CpuEmbeddingModule(
          resource,
          feature_names=["feat1"],
          output_feature_names=["feat1_out"],
      )

    # CpuEmbeddingModule multi-feature with dynamic batch dimensions.
    mod_multi = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_names=["feat1", "feat2"],
        output_feature_names=["out1", "out2"],
        output_shapes=[[None, 2], [None, 3]],
    )
    sig_multi = mod_multi.get_output_signature(None)
    self.assertEqual(sig_multi["out1"].shape, (None, 2))
    self.assertEqual(sig_multi["out2"].shape, (None, 3))

    # Deprecated -1 in CpuEmbeddingModule warns and normalizes to None; < -1
    # raises ValueError.
    with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
      mod_legacy = cpu_embedding.CpuEmbeddingModule(
          resource,
          feature_name="feat1",
          output_feature_name="feat1_out",
          output_shape=[-1, 2],
      )
    self.assertEqual(
        mod_legacy.get_output_signature(None)["feat1_out"].shape, (None, 2)
    )
    with self.assertRaises(ValueError):
      cpu_embedding.CpuEmbeddingModule(
          resource,
          feature_name="feat1",
          output_feature_name="feat1_out",
          output_shape=[-2, 2],
      )

    # Direct pybind CpuEmbeddingLookupTransform rejects negative sizes (-1, -2).
    with self.assertRaisesRegex(RuntimeError, "Expected a non-negative size"):
      cpu_embedding_lookup_transform_pybind.CpuEmbeddingLookupTransform(
          resource.resource,
          ["feat1"],
          False,
          ["feat1_out"],
          2,
          [[-1, 2]],
      )
    with self.assertRaisesRegex(RuntimeError, "Expected a non-negative size"):
      cpu_embedding_lookup_transform_pybind.CpuEmbeddingLookupTransform(
          resource.resource,
          ["feat1"],
          False,
          ["feat1_out"],
          2,
          [[-2, 2]],
      )

    # DenseCpuEmbeddingLookupTransform with dynamic and rank-0 input specs.
    dense_transform = cpu_embedding.DenseCpuEmbeddingLookupTransform(
        resource=resource,
        feature_names="feat1",
    )
    dyn_out = dense_transform.get_output_signature(
        (shape_dtype_struct.ShapeDtypeStruct(shape=(None, 3), dtype=np.int32),)
    )
    self.assertEqual(dyn_out[0].shape, (None, 2))
    self.assertEqual(dyn_out[0].dtype, np.float32)

    scalar_out = dense_transform.get_output_signature(
        (shape_dtype_struct.ShapeDtypeStruct(shape=(), dtype=np.int32),)
    )
    self.assertEqual(scalar_out[0].shape, (None, 2))

    class _LegacyLeaf:
      shape = (-1, 3)

    with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
      legacy_dense_out = dense_transform.get_output_signature((_LegacyLeaf(),))
    self.assertEqual(legacy_dense_out[0].shape, (None, 2))

    class _InvalidLeaf:
      shape = (-2, 3)

    with self.assertRaises(ValueError):
      dense_transform.get_output_signature((_InvalidLeaf(),))

    # UnpackHostBufferVectorModule with dynamic dimensions and deprecated -1.
    unpack_dyn = unpack_host_buffer_vector_module.UnpackHostBufferVectorModule(
        output_count=1,
        output_shapes=[[None, 4]],
    )
    self.assertEqual(unpack_dyn.get_output_signature(None).shape, (None, 4))
    with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
      unpack_legacy = (
          unpack_host_buffer_vector_module.UnpackHostBufferVectorModule(
              output_count=1,
              output_shapes=[[-1, 4]],
          )
      )
    self.assertEqual(unpack_legacy.get_output_signature(None).shape, (None, 4))
    with self.assertRaises(ValueError):
      unpack_host_buffer_vector_module.UnpackHostBufferVectorModule(
          output_count=1,
          output_shapes=[[-2, 4]],
      )


if __name__ == "__main__":
  absltest.main()
