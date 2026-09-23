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

"""Tests for XLA SavedModel export of DLRM model."""

import os

from absl.testing import absltest

from examples.dlrm import export
from examples.dlrm import model as model_lib


class ExportTest(absltest.TestCase):

  def test_export_model(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path)
    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )

  def test_export_model_custom_embedding_size(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path, embedding_size=64)
    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )

  def test_build_model_with_custom_embedding_size(self):
    table_configs, _ = model_lib.create_feature_specs(
        model_lib.VOCAB_SIZES, embedding_size=64
    )
    for table_config in table_configs.values():
      self.assertEqual(table_config.dim, 64)
    _, model = model_lib.build_model(embedding_size=64)
    self.assertEqual(model.embedding_size, 64)
    self.assertEqual(model.bottom_mlp_dims, [512, 256, 64])


if __name__ == '__main__':
  absltest.main()
