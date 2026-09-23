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

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from xla_serving_data.ops import cpu_embedding
from xla_serving_data.processor.proto import transform_config_pb2
from xla_serving_data.processor.python import composite_processor_pybind
from xla_serving_data_test.processor.python.examples import add_from_resource_transform_pybind
from xla_serving_data_test.processor.python.examples import add_int_transform_pybind
from xla_serving_data_test.processor.python.examples import int_resource_pybind
from xla_serving_data.resource_manager.python import resource_manager
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.processors import composite_processor


class TestAddIntModule:
  """Test module for AddIntTransform."""

  def __init__(self):
    self._transform = add_int_transform_pybind.AddIntTransform()

  @property
  def transform(self):
    return self._transform

  def export_assets(self):
    return {}

  def get_output_signature(self, input_signature):
    return input_signature


class TestAddFromResourceModule:
  """Test Module that uses a resource with an asset."""

  def __init__(self, resource: int_resource_pybind.IntResource):
    self._transform = (
        add_from_resource_transform_pybind.AddFromResourceTransform(resource)
    )

  @property
  def transform(self):
    return self._transform

  def export_assets(self):
    return self.transform.assets()

  def get_output_signature(self, input_signature):
    return input_signature


class CompositeProcessorTest(parameterized.TestCase):

  def test_convert_jd_type_to_shlo_type(self):
    self.assertEqual(
        composite_processor._convert_jd_type_to_shlo_type(np.dtype("int32")),
        shlo_type.ShloDType.i32,
    )
    self.assertEqual(
        composite_processor._convert_jd_type_to_shlo_type(np.dtype("float32")),
        shlo_type.ShloDType.f32,
    )
    self.assertEqual(
        composite_processor._convert_jd_type_to_shlo_type(np.dtype("str")),
        shlo_type.ShloDType.str,
    )
    with self.assertRaisesRegex(ValueError, "Expected a numpy.dtype"):
      composite_processor._convert_jd_type_to_shlo_type("int32")

  def test_init_empty_transforms(self):
    with self.assertRaisesRegex(
        ValueError, "Expected at least one transform to be passed in"
    ):
      composite_processor.CompositeProcessor("my_proc", [])

  def test_init_empty_name(self):
    t1 = TestAddIntModule()
    with self.assertRaisesRegex(
        ValueError, "CompositeProcessor name must not be empty"
    ):
      composite_processor.CompositeProcessor("", [t1])

  def test_composite_processor_without_assets(self):
    t1 = TestAddIntModule()
    proc = composite_processor.CompositeProcessor("no_asset_proc", [t1])

    input_signature = {
        "x": jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.int32),
    }
    prepared = proc.prepare(input_signature)

    xsm_input_signature = {
        "x": shlo_type.ShloTensorSpec(
            shape=(2, 2), dtype=shlo_type.ShloDType.i32
        ),
    }

    self.assertEqual(prepared.name, "no_asset_proc")
    self.assertIsNone(prepared.weights_name)
    self.assertEmpty(prepared.asset_map.assets)

    # Check signatures
    self.assertEqual(prepared.input_signature, xsm_input_signature)
    self.assertIn("x", prepared.output_signature)
    out_spec = prepared.output_signature["x"]
    self.assertEqual(out_spec.shape, (2, 2))
    self.assertEqual(out_spec.dtype, shlo_type.ShloDType.i32)

    # Check to_xsm_saveable
    saveable_fn = prepared.to_xsm_saveable()
    self.assertEqual(saveable_fn.input_signature, xsm_input_signature)
    self.assertEqual(saveable_fn.output_signature, prepared.output_signature)

    # Check on_persist
    persist_fn = prepared.on_persist()
    self.assertIsNotNone(persist_fn)
    export_dir = self.create_tempdir("export").full_path
    supps = persist_fn(export_dir)
    self.assertEqual(supps, {})

  def test_composite_processor_with_assets(self):
    dir1 = self.create_tempdir("dir1")
    dir2 = self.create_tempdir("dir2")

    path1_1 = dir1.create_file("data.txt", "10").full_path
    path1_2 = dir1.create_file("info.txt", "20").full_path
    path2_1 = dir2.create_file("data.txt", "30").full_path
    path2_2 = dir2.create_file("info.txt", "40").full_path

    asset1_1 = resource_manager.Asset(path=path1_1, name="ast1")
    asset1_2 = resource_manager.Asset(path=path1_2, name="ast2")
    resource1 = int_resource_pybind.IntResource([asset1_1, asset1_2])

    asset2_1 = resource_manager.Asset(path=path2_1, name="ast3")
    asset2_2 = resource_manager.Asset(path=path2_2, name="ast4")
    resource2 = int_resource_pybind.IntResource([asset2_1, asset2_2])

    t1 = TestAddFromResourceModule(resource1)
    t2 = TestAddFromResourceModule(resource2)
    proc = composite_processor.CompositeProcessor("asset_proc", [t1, t2])

    input_signature = {
        "val": jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.int32),
    }
    prepared = proc.prepare(input_signature)

    self.assertEqual(prepared.name, "asset_proc")
    self.assertLen(prepared.asset_map.assets, 4)
    self.assertEqual(
        prepared.asset_map.assets[path1_1],
        str(pathlib.Path("assets", "data.txt")),
    )
    self.assertEqual(
        prepared.asset_map.assets[path1_2],
        str(pathlib.Path("assets", "info.txt")),
    )
    self.assertEqual(
        prepared.asset_map.assets[path2_1],
        str(pathlib.Path("assets", "data_1.txt")),
    )
    self.assertEqual(
        prepared.asset_map.assets[path2_2],
        str(pathlib.Path("assets", "info_1.txt")),
    )

    # Verify that the processor config has its asset paths fixed up
    config = transform_config_pb2.CompositeProcessorConfig()
    config.ParseFromString(prepared.config.SerializeToString())

    fixed_paths = {}
    for res in config.resources:
      for ast in res.assets:
        fixed_paths[ast.name] = ast.path

    expected_paths = {
        "ast1": str(pathlib.Path("asset_proc", "assets", "data.txt")),
        "ast2": str(pathlib.Path("asset_proc", "assets", "info.txt")),
        "ast3": str(pathlib.Path("asset_proc", "assets", "data_1.txt")),
        "ast4": str(pathlib.Path("asset_proc", "assets", "info_1.txt")),
    }
    self.assertEqual(fixed_paths, expected_paths)

    # Check on_persist saving the asset
    persist_fn = prepared.on_persist()
    export_dir = self.create_tempdir("export_with_assets").full_path
    supps = persist_fn(export_dir)

    # No GlobalSupplemental is returned
    self.assertEmpty(supps)

    # Verify asset files were copied correctly under assets/<filename>
    for expected_rel, expected_content in [
        (pathlib.Path("assets", "data.txt"), "10"),
        (pathlib.Path("assets", "info.txt"), "20"),
        (pathlib.Path("assets", "data_1.txt"), "30"),
        (pathlib.Path("assets", "info_1.txt"), "40"),
    ]:
      dest_file = pathlib.Path(export_dir, "asset_proc", expected_rel)
      self.assertTrue(dest_file.exists())
      self.assertEqual(dest_file.read_text(), expected_content)

  def test_composite_processor_with_cpu_embedding(self):
    table_data = {
        "table1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    }
    combine_modes = {"table1": "MEAN"}
    feature_specs = [{
        "name": "table1",
        "table_name": "table1",
    }]
    resource = cpu_embedding.CpuEmbeddingResource(
        table_data, combine_modes, feature_specs
    )
    module = cpu_embedding.CpuEmbeddingModule(
        resource,
        feature_name="table1",
        output_feature_name="table1_out",
        output_shape=[1, 3],
    )

    proc = composite_processor.CompositeProcessor("jce_proc", [module])

    input_signature = {
        "table1_out": jax.ShapeDtypeStruct(shape=(1, 3), dtype=jnp.float32),
    }
    prepared = proc.prepare(input_signature)

    self.assertEqual(prepared.name, "jce_proc")
    # Verify we have 2 assets (manifest and weights table)
    self.assertLen(prepared.asset_map.assets, 2)

    # Check on_persist saving the assets
    export_dir = self.create_tempdir("jce_export").full_path
    persist_fn = prepared.on_persist()
    supps = persist_fn(export_dir)
    self.assertEmpty(supps)

    # Verify files exist in target location
    dest_manifest = pathlib.Path(
        export_dir, "jce_proc", "assets", "manifest.textproto"
    )
    dest_table = pathlib.Path(export_dir, "jce_proc", "assets", "table1.bin")
    self.assertTrue(dest_manifest.exists())
    self.assertTrue(dest_table.exists())

    resource.close()

  def test_from_graph_dynamic_shape_propagation(self):
    graph = composite_processor_pybind.ExecutionGraph()
    t_add = add_int_transform_pybind.AddIntTransform()
    idx = graph.add_node(
        "my_add",
        t_add,
        [
            composite_processor_pybind.GraphEdge.graph_input(0),
            composite_processor_pybind.GraphEdge.graph_input(1),
        ],
    )
    graph.set_outputs(
        [composite_processor_pybind.GraphEdge.node_output(idx, 0)]
    )

    input_structure = {
        "a": jax.ShapeDtypeStruct((), jnp.int32),
        "b": jax.ShapeDtypeStruct((), jnp.int32),
    }
    output_structure = {"out": jax.ShapeDtypeStruct((), jnp.int32)}
    struct = transform_config_pb2.GraphIOStructureProto()
    struct.input_structure.CopyFrom(
        composite_processor._tree_to_proto(input_structure)
    )
    struct.output_structure.CopyFrom(
        composite_processor._tree_to_proto(output_structure)
    )
    graph.io_structure = struct

    proc_from_graph = composite_processor.CompositeProcessor.from_graph(
        "from_graph_proc", graph, assets={}
    )

    # Call prepare with dynamic runtime batch size of (64, 2)
    dynamic_input_sig = {
        "a": jax.ShapeDtypeStruct(shape=(64, 2), dtype=jnp.int32),
        "b": jax.ShapeDtypeStruct(shape=(64, 2), dtype=jnp.int32),
    }
    prepared = proc_from_graph.prepare(dynamic_input_sig)

    # Verify output signature dynamically calculated (64, 2) instead of static proto shape ()
    self.assertEqual(prepared.output_signature["out"].shape, (64, 2))
    self.assertEqual(
        prepared.output_signature["out"].dtype, shlo_type.ShloDType.i32
    )


if __name__ == "__main__":
  absltest.main()
