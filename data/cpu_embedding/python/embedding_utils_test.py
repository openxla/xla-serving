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

"""Unit tests for generic_embedding_graph module."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from xla_serving_data.cpu_embedding.python import embedding_utils
from xla_serving_data.kernels.cpu_embedding import cpu_embedding_module


class EmbeddingUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.table_data = {
        "table_0": (
            np.random.uniform(-1.0, 1.0, size=(100, 64)).astype(np.float32)
        )
    }
    self.combine_modes = {"table_0": "MEAN"}
    self.feature_specs = [
        {"name": "feat_0", "table_name": "table_0", "max_sequence_length": 0}
    ]
    self.resource = cpu_embedding_module.CpuEmbeddingResource(
        self.table_data, self.combine_modes, self.feature_specs
    )

  def test_build_dense_2d_combined_features(self):
    feature_specs = [
        {"name": "feat_0", "table_name": "table_0", "max_sequence_length": 0},
    ]
    input_signature = {
        "feat_0": jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.int32),
        "dense_features": jax.ShapeDtypeStruct(
            shape=(2, 13), dtype=jnp.float32
        ),
    }

    processor = embedding_utils.build_embedding_lookup(
        name="test_processor",
        input_signature=input_signature,
        feature_specs=feature_specs,
        jce_resource=self.resource,
        embedding_dim=64,
        dense_lookup=True,
    )

    self.assertIsNotNone(processor)
    prepared = processor.prepare(input_signature)
    self.assertLen(prepared.output_signature, 2)
    self.assertEqual(prepared.output_signature[0].shape, (None, 64))
    self.assertEqual(prepared.output_signature[1].shape, (2, 13))

  def test_build_3d_sequence_features(self):
    feature_specs = [
        {
            "name": "seq_feat",
            "table_name": "table_0",
            "max_sequence_length": 10,
        },
    ]
    input_signature = {
        "seq_feat": jax.ShapeDtypeStruct(shape=(2, 10), dtype=jnp.int32),
    }

    processor = embedding_utils.build_embedding_lookup(
        name="test_seq_processor",
        input_signature=input_signature,
        feature_specs=feature_specs,
        jce_resource=self.resource,
        embedding_dim=64,
        dense_lookup=True,
    )

    self.assertIsNotNone(processor)
    prepared = processor.prepare(input_signature)
    self.assertLen(prepared.output_signature, 1)
    self.assertEqual(prepared.output_signature[0].shape, (None, 10, 64))

  def test_build_mixed_pipeline(self):
    feature_specs = [
        {"name": "feat_0", "table_name": "table_0", "max_sequence_length": 0},
        {"name": "seq_feat", "table_name": "table_0", "max_sequence_length": 5},
    ]
    input_signature = {
        "feat_0": jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.int32),
        "seq_feat": jax.ShapeDtypeStruct(shape=(2, 5), dtype=jnp.int32),
        "dense_passthrough": jax.ShapeDtypeStruct(
            shape=(2, 8), dtype=jnp.float32
        ),
    }

    processor = embedding_utils.build_embedding_lookup(
        name="test_mixed_processor",
        input_signature=input_signature,
        feature_specs=feature_specs,
        jce_resource=self.resource,
        embedding_dim=64,
        dense_lookup=True,
    )

    self.assertIsNotNone(processor)
    prepared = processor.prepare(input_signature)
    self.assertLen(prepared.output_signature, 3)
    # Output 0: 2D combined feat_0 -> (None, 64)
    self.assertEqual(prepared.output_signature[0].shape, (None, 64))
    # Output 1: 3D sequence seq_feat -> (None, 5, 64)
    self.assertEqual(prepared.output_signature[1].shape, (None, 5, 64))
    # Output 2: Passthrough dense_passthrough -> (2, 8)
    self.assertEqual(prepared.output_signature[2].shape, (2, 8))


if __name__ == "__main__":
  absltest.main()
