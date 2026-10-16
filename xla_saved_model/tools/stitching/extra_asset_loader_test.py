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

import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized

from xla_saved_model.tools import stitching
from xla_saved_model.tools.stitching import extra_asset_loader

_ASSET_NAME = 'config.textproto'
_CONTENTS = b'name: "config"\n\x00\xff'


class ExtraAssetLoaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_dir = self.create_tempdir().full_path
    pathlib.Path(self.model_dir, 'assets.extra').mkdir()
    pathlib.Path(self.model_dir, 'assets.extra', _ASSET_NAME).write_bytes(
        _CONTENTS
    )

  def test_find_extra_asset_path(self):
    self.assertEqual(
        extra_asset_loader.find_extra_asset_path(self.model_dir, _ASSET_NAME),
        os.path.join(self.model_dir, 'assets.extra', _ASSET_NAME),
    )

  def test_load_extra_asset(self):
    self.assertEqual(
        extra_asset_loader.load_extra_asset(self.model_dir, _ASSET_NAME),
        _CONTENTS,
    )
    self.assertEqual(
        extra_asset_loader.load_extra_asset(
            pathlib.Path(self.model_dir), _ASSET_NAME
        ),
        _CONTENTS,
    )
    self.assertEqual(
        stitching.load_extra_asset(self.model_dir, _ASSET_NAME), _CONTENTS
    )

  def test_missing_asset_raises(self):
    with self.assertRaisesRegex(FileNotFoundError, 'no extra asset'):
      extra_asset_loader.load_extra_asset(self.model_dir, 'missing.textproto')

  @parameterized.parameters('', '.', '..', 'assets/config', '../config')
  def test_name_that_is_not_a_file_name_raises(self, name):
    with self.assertRaisesRegex(ValueError, 'must be a file name'):
      extra_asset_loader.find_extra_asset_path(self.model_dir, name)


if __name__ == '__main__':
  absltest.main()
