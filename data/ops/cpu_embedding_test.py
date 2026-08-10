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

"""Tests for CpuEmbeddingModule Python wrapper."""

import os
from absl.testing import absltest
import jax
import numpy as np
from data.kernels.cpu_embedding import cpu_embedding_serializer_pybind
from data.ops import cpu_embedding
from data.processor.python import composite_processor_pybind


class CpuEmbeddingModuleTest(absltest.TestCase):

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

    inputs = [
        composite_processor_pybind.Value(
            [composite_processor_pybind.HostBuffer.from_dlpack(ids)]
        ),
        composite_processor_pybind.Value(
            [composite_processor_pybind.HostBuffer.from_dlpack(row_pointers)]
        ),
        composite_processor_pybind.Value(
            [composite_processor_pybind.HostBuffer.from_dlpack(weights)]
        ),
    ]

    context = composite_processor_pybind.ExecuteContext()
    context.inputs = inputs
    module.transform.execute(context)
    outputs = context.outputs
    self.assertLen(outputs, 1)

    out_hb_list = outputs[0].as_encapsulated_type()
    self.assertLen(out_hb_list, 1)
    out_arr = np.from_dlpack(out_hb_list[0])
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

    inputs = [
        composite_processor_pybind.Value(
            [composite_processor_pybind.HostBuffer.from_dlpack(ids)]
        ),
        composite_processor_pybind.Value([]),
        composite_processor_pybind.Value([]),
    ]

    context = composite_processor_pybind.ExecuteContext()
    context.inputs = inputs
    module.transform.execute(context)
    outputs = context.outputs
    self.assertLen(outputs, 1)

    out_hb_list = outputs[0].as_encapsulated_type()
    self.assertLen(out_hb_list, 1)
    out_arr = np.from_dlpack(out_hb_list[0])
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

    inputs = [
        composite_processor_pybind.Value([
            composite_processor_pybind.HostBuffer.from_dlpack(ids_a),
            composite_processor_pybind.HostBuffer.from_dlpack(ids_b),
        ]),
        composite_processor_pybind.Value([
            composite_processor_pybind.HostBuffer.from_dlpack(rp_a),
            composite_processor_pybind.HostBuffer.from_dlpack(rp_b),
        ]),
        composite_processor_pybind.Value([
            composite_processor_pybind.HostBuffer.from_dlpack(w_a),
            composite_processor_pybind.HostBuffer.from_dlpack(w_b),
        ]),
    ]

    context = composite_processor_pybind.ExecuteContext()
    context.inputs = inputs
    module.transform.execute(context)
    outputs = context.outputs
    self.assertLen(outputs, 1)

    out_hb_list = outputs[0].as_encapsulated_type()
    self.assertLen(out_hb_list, 2)

    out_a = np.from_dlpack(out_hb_list[0])
    out_b = np.from_dlpack(out_hb_list[1])

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

    inputs = [
        composite_processor_pybind.Value([
            composite_processor_pybind.HostBuffer.from_dlpack(ids_a),
            composite_processor_pybind.HostBuffer.from_dlpack(ids_b),
        ]),
        composite_processor_pybind.Value([]),
        composite_processor_pybind.Value([]),
    ]

    context = composite_processor_pybind.ExecuteContext()
    context.inputs = inputs
    module.transform.execute(context)
    outputs = context.outputs
    self.assertLen(outputs, 1)

    out_hb_list = outputs[0].as_encapsulated_type()
    self.assertLen(out_hb_list, 2)

    out_a = np.from_dlpack(out_hb_list[0])
    out_b = np.from_dlpack(out_hb_list[1])

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


if __name__ == "__main__":
  absltest.main()
