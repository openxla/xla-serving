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

"""Python integration test for BucketPad + CpuEmbedding pipeline."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.export import obm_export
from orbax.export import serving_config as osc

from .third_party.neptune.data.kernels.bucket_pad import bucket_pad_transform_pybind
from .third_party.neptune.data.kernels.cpu_embedding import cpu_embedding_lookup_transform_pybind
from .third_party.neptune.data.kernels.cpu_embedding import cpu_embedding_module
from .third_party.neptune.data.kernels.cpu_embedding import pack_host_buffer_vector_transform_pybind
from .third_party.neptune.data.kernels.cpu_embedding import unpack_host_buffer_vector_transform_pybind
from .third_party.neptune.data.ops.bucket_pad import bucket_pad
from .third_party.neptune.data.processor.cc.utils import default_value_pb2
from .third_party.neptune.data.processor.proto import transform_config_pb2
from .third_party.neptune.data.processor.python import composite_processor_pybind
from .third_party.neptune.data.processor.python import processor


class BucketPadCpuEmbeddingSandboxPyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # 1. Deterministic embedding table (vocab_size=5, embedding_dim=4, float32)
    self.table_data = {
        'feat1': np.array(
            [
                [0.0, 0.0, 0.0, 0.0],  # index 0
                [1.0, 1.0, 1.0, 1.0],  # index 1
                [2.0, 2.0, 2.0, 2.0],  # index 2
                [3.0, 3.0, 3.0, 3.0],  # index 3
                [4.0, 4.0, 4.0, 4.0],  # index 4
            ],
            dtype=np.float32,
        )
    }
    self.combine_modes = {'feat1': 'SUM'}
    self.feature_specs = [{
        'name': 'feat1',
        'table_name': 'feat1',
    }]
    self.resource = cpu_embedding_module.CpuEmbeddingResource(
        self.table_data, self.combine_modes, self.feature_specs
    )

    # 2. Build ExecutionGraph DAG
    graph = composite_processor_pybind.ExecutionGraph()

    pad_proto = default_value_pb2.DefaultValue(int_value=0)
    bucket_pad_transform = bucket_pad_transform_pybind.BucketPadTransform(
        [4, 8, 16], -1, pad_proto
    )
    node0_idx = graph.add_node(
        'bucket_pad',
        bucket_pad_transform,
        [composite_processor_pybind.GraphEdge.graph_input(0)],
    )

    pack_t1 = (
        pack_host_buffer_vector_transform_pybind.PackHostBufferVectorTransform(
            1
        )
    )
    node1_idx = graph.add_node(
        'pack_ids',
        pack_t1,
        [composite_processor_pybind.GraphEdge.node_output(node0_idx, 0)],
    )

    pack_t0_a = (
        pack_host_buffer_vector_transform_pybind.PackHostBufferVectorTransform(
            0
        )
    )
    node2_idx = graph.add_node('pack_row_pointers', pack_t0_a, [])

    pack_t0_b = (
        pack_host_buffer_vector_transform_pybind.PackHostBufferVectorTransform(
            0
        )
    )
    node3_idx = graph.add_node('pack_weights', pack_t0_b, [])

    cpu_lookup_transform = (
        cpu_embedding_lookup_transform_pybind.CpuEmbeddingLookupTransform(
            self.resource.resource,
            ['feat1'],
            True,
            ['feat1_out'],
            4,
            [[-1, 4]],
        )
    )
    node4_idx = graph.add_node(
        'cpu_embedding_lookup',
        cpu_lookup_transform,
        [
            composite_processor_pybind.GraphEdge.node_output(node1_idx, 0),
            composite_processor_pybind.GraphEdge.node_output(node2_idx, 0),
            composite_processor_pybind.GraphEdge.node_output(node3_idx, 0),
        ],
    )

    unpack_t = unpack_host_buffer_vector_transform_pybind.UnpackHostBufferVectorTransform(
        1
    )
    node5_idx = graph.add_node(
        'unpack_output',
        unpack_t,
        [composite_processor_pybind.GraphEdge.node_output(node4_idx, 0)],
    )

    graph.set_outputs(
        [composite_processor_pybind.GraphEdge.node_output(node5_idx, 0)]
    )

    input_structure = {
        'ids': jax.ShapeDtypeStruct(shape=(-1, -1), dtype=jnp.int32),
    }
    output_structure = {
        'feat1_out': jax.ShapeDtypeStruct(shape=(-1, 4), dtype=jnp.float32),
    }
    struct = transform_config_pb2.GraphIOStructureProto()
    struct.input_structure.CopyFrom(processor._tree_to_proto(input_structure))
    struct.output_structure.CopyFrom(processor._tree_to_proto(output_structure))
    graph.io_structure = struct

    self.nd_processor = processor.NDProcessor.from_graph(
        graph=graph,
        assets=self.resource.export_assets(),
    )
    self.nd_processor.prepare({
        'ids': jax.ShapeDtypeStruct(shape=(2, 4), dtype=jnp.int32),
    })

  def tearDown(self):
    self.resource.close()
    super().tearDown()

  def test_case_a_dynamic_shape_valency_3(self):
    input_arr = np.array([[1, 2, 3], [4, 1, 2]], dtype=np.int32)
    output = self.nd_processor(input_arr)
    expected = np.array(
        [[6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0]], dtype=np.float32
    )
    np.testing.assert_allclose(output, expected)

  def test_case_b_dynamic_shape_valency_5(self):
    input_arr = np.array([[1, 2, 3, 4, 1], [2, 2, 2, 2, 2]], dtype=np.int32)
    output = self.nd_processor(input_arr)
    expected = np.array(
        [[11.0, 11.0, 11.0, 11.0], [10.0, 10.0, 10.0, 10.0]], dtype=np.float32
    )
    np.testing.assert_allclose(output, expected)

  def test_case_c_exceeds_max_bucket_raises(self):
    input_arr = np.ones((2, 20), dtype=np.int32)
    with self.assertRaises(Exception):
      self.nd_processor(input_arr)

  def test_export_and_load_model(self):
    model_dir = self.create_tempdir('sandbox_model').full_path
    input_spec = {
        'ids': jax.ShapeDtypeStruct(shape=(2, 4), dtype=jnp.int32),
    }
    serving_config = osc.ServingConfig(
        signature_key='serving_default',
        input_signature=[input_spec],
        preprocessors=[self.nd_processor],
    )
    oe = obm_export.ObmExport(
        module=None,
        serving_configs=[serving_config],
    )
    oe.save(model_path=model_dir)

    loaded_proc = processor.load_model(model_dir)
    input_arr = np.array([[1, 2, 3], [4, 1, 2]], dtype=np.int32)
    output = loaded_proc(input_arr)
    expected = np.array(
        [[6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0]], dtype=np.float32
    )
    np.testing.assert_allclose(output, expected)


if __name__ == '__main__':
  absltest.main()
