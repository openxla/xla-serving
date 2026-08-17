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

"""Test the CPU embedding engine pybind extension."""

from absl.testing import absltest
import numpy as np
from data.cpu_embedding.python import cpu_embedding_engine_pybind


class CpuEmbeddingEngineTest(absltest.TestCase):

  def test_embedding_table_lookup(self):
    table = cpu_embedding_engine_pybind.create_embedding_table(
        name="table",
        data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        combine_mode="mean",
    )
    self.assertEqual(table.name, "table")
    self.assertEqual(table.vocab_size, 3)
    self.assertEqual(table.embedding_dim, 2)
    np.testing.assert_array_equal(
        table.lookup([0, 2]), np.array([[1.0, 2.0], [5.0, 6.0]])
    )

  def test_feature_batch_csr_input(self):
    feature_batch = cpu_embedding_engine_pybind.create_feature_batch_csr_input(
        name="feature_batch",
        embedding_ids=[0, 1, 2],
        row_pointers=[0, 2, 3],
        gains=[1.0, 2.0, 3.0],
    )
    self.assertEqual(feature_batch.name, "feature_batch")
    np.testing.assert_array_equal(feature_batch.embedding_ids, [0, 1, 2])
    np.testing.assert_array_equal(feature_batch.row_pointers, [0, 2, 3])
    np.testing.assert_array_equal(feature_batch.weights, [1.0, 2.0, 3.0])

  def test_feature_spec(self):
    feature_spec = cpu_embedding_engine_pybind.FeatureSpec(
        name="feature",
        table_name="table",
        max_sequence_length=10,
    )
    self.assertEqual(feature_spec.name, "feature")
    self.assertEqual(feature_spec.table_name, "table")
    self.assertEqual(feature_spec.max_sequence_length, 10)

  def test_cpu_embedding_engine(self):
    engine = cpu_embedding_engine_pybind.CpuEmbeddingEngine()
    engine.register_feature_spec(
        cpu_embedding_engine_pybind.FeatureSpec(
            name="feature",
            table_name="table",
            max_sequence_length=0,
        )
    )
    engine.register_feature_spec(
        cpu_embedding_engine_pybind.FeatureSpec(
            name="seq_feature",
            table_name="table",
            max_sequence_length=5,
        )
    )
    data = np.arange(100, dtype=np.float32).reshape(25, 4)
    engine.add_local_table(
        "table",
        cpu_embedding_engine_pybind.create_embedding_table(
            name="table",
            data=data,
            combine_mode="mean",
        ),
    )

    # Create some feature batches.
    feature_batches = [
        cpu_embedding_engine_pybind.create_feature_batch_csr_input(
            name="feature",
            embedding_ids=[0, 8, 12, 15, 21],
            row_pointers=[0, 3, 5],
            gains=[],
        ),
        cpu_embedding_engine_pybind.create_feature_batch_csr_input(
            name="seq_feature",
            embedding_ids=[4, 13, 18, 23],
            row_pointers=[0, 2, 4],
            gains=[],
        ),
    ]
    result = engine.execute(feature_batches)
    self.assertLen(result, 2)

    activations, seq_activations = result
    np.testing.assert_allclose(
        activations,
        np.vstack([
            np.mean(data[[0, 8, 12], :], axis=0),
            np.mean(data[[15, 21], :], axis=0),
        ]),
    )
    expected_seq_activations = np.zeros((2, 5, 4), dtype=np.float32)
    expected_seq_activations[0, 0:2, :] = data[[4, 13], :]
    expected_seq_activations[1, 0:2, :] = data[[18, 23], :]
    np.testing.assert_array_equal(
        seq_activations,
        expected_seq_activations,
    )


if __name__ == "__main__":
  absltest.main()
