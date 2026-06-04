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

from absl.testing import absltest
from absl.testing import parameterized

from neptune_model._src.assets2npt import main_lib


class Assets2NptMainLibTest(parameterized.TestCase):

  def test_file_hexdigest(self):
    temp_file = self.create_tempdir().create_file('test.txt', 'hello world')
    file_hash = main_lib.file_hexdigest(temp_file.full_path)
    self.assertEqual(
        file_hash,
        # blake3.blake3(b'hello world').hexdigest()
        'd74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24',
    )

  def test_asset_map(self):
    dir_path = self.create_tempdir()
    file_a = dir_path.create_file('a.txt', 'content a').full_path
    file_b = dir_path.create_file('b.txt', 'content b').full_path

    hash_a = main_lib.file_hexdigest(file_a)
    hash_b = main_lib.file_hexdigest(file_b)

    asset_map = main_lib.AssetMap()
    dest_a = asset_map.add_asset(file_a)
    dest_b = asset_map.add_asset(file_b)

    self.assertLen(asset_map.assets, 2)
    self.assertEqual(dest_a, os.path.join('assets', hash_a))
    self.assertEqual(dest_b, os.path.join('assets', hash_b))
    self.assertEqual(asset_map.assets[file_a], dest_a)
    self.assertEqual(asset_map.assets[file_b], dest_b)

  def test_asset_map_deduplication(self):
    dir_path = self.create_tempdir()
    file_a = dir_path.create_file('a.txt', 'shared content').full_path
    file_b = dir_path.create_file('b.txt', 'shared content').full_path

    expected_hash = main_lib.file_hexdigest(file_a)
    expected_dest = os.path.join('assets', expected_hash)

    asset_map = main_lib.AssetMap()
    dest_a = asset_map.add_asset(file_a)
    dest_b = asset_map.add_asset(file_b)

    self.assertLen(asset_map.assets, 2)
    self.assertEqual(dest_a, expected_dest)
    self.assertEqual(dest_b, expected_dest)

  def test_save_assets_empty(self):
    asset_map = main_lib.AssetMap()
    export_dir = self.create_tempdir('export_path').full_path
    result = main_lib.save_assets(asset_map, export_dir)
    self.assertIsNone(result)

  def test_save_assets_success(self):
    src_dir = self.create_tempdir('src_path').full_path
    export_dir = self.create_tempdir('export_path').full_path

    src_file1 = os.path.join(src_dir, 'asset1.txt')
    src_file2 = os.path.join(src_dir, 'asset2.txt')

    with open(src_file1, 'w') as f:
      f.write('content1')
    with open(src_file2, 'w') as f:
      f.write('content2')

    hash1 = main_lib.file_hexdigest(src_file1)
    hash2 = main_lib.file_hexdigest(src_file2)

    asset_map = main_lib.AssetMap()
    asset_map.add_asset(src_file1)
    asset_map.add_asset(src_file2)

    result = main_lib.save_assets(asset_map, export_dir)
    self.assertIsNone(result)

    dest_file1 = os.path.join(export_dir, 'assets', hash1)
    dest_file2 = os.path.join(export_dir, 'assets', hash2)

    self.assertTrue(os.path.exists(dest_file1))
    self.assertTrue(os.path.exists(dest_file2))

    with open(dest_file1, 'r') as f:
      self.assertEqual(f.read(), 'content1')
    with open(dest_file2, 'r') as f:
      self.assertEqual(f.read(), 'content2')

  def test_save_assets_outside_export_path(self):
    export_dir = self.create_tempdir('export').full_path
    src_file = self.create_tempdir('src').create_file('asset.txt').full_path

    asset_map = main_lib.AssetMap()
    # Manually insert a malicious path into the internal dictionary
    # to test validation.
    asset_map._assets[src_file] = '../../outside.txt'

    with self.assertRaisesRegex(ValueError, 'outside of the model export path'):
      main_lib.save_assets(asset_map, export_dir)


if __name__ == '__main__':
  absltest.main()
