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

from neptune_model._src.assets2npt import main_lib


class Assets2NptMainLibTest(parameterized.TestCase):

  def test_asset_map(self):
    dir_path = self.create_tempdir()
    file_a = dir_path.create_file('a.txt', 'content a').full_path
    file_b = dir_path.create_file('b.txt', 'content b').full_path

    asset_map = main_lib.AssetMap('bundle1')
    dest_a = asset_map.add_asset(file_a)
    dest_b = asset_map.add_asset(file_b)

    self.assertLen(asset_map.assets, 2)
    self.assertEqual(dest_a, os.path.join('assets', 'a.txt'))
    self.assertEqual(dest_b, os.path.join('assets', 'b.txt'))
    self.assertEqual(asset_map.assets[file_a], dest_a)
    self.assertEqual(asset_map.assets[file_b], dest_b)

  def test_asset_map_same_source_path(self):
    dir_path = self.create_tempdir()
    file_a = dir_path.create_file('a.txt', 'content').full_path

    expected_dest = os.path.join('assets', 'a.txt')

    asset_map = main_lib.AssetMap('bundle1')
    dest_a1 = asset_map.add_asset(file_a)
    dest_a2 = asset_map.add_asset(file_a)

    self.assertLen(asset_map.assets, 1)
    self.assertEqual(dest_a1, expected_dest)
    self.assertEqual(dest_a2, expected_dest)

  def test_asset_map_filename_collision(self):
    dir1 = self.create_tempdir('dir1')
    dir2 = self.create_tempdir('dir2')
    dir3 = self.create_tempdir('dir3')
    file1 = dir1.create_file('a.txt', 'content 1').full_path
    file2 = dir2.create_file('a.txt', 'content 2').full_path
    file3 = dir3.create_file('a.txt', 'content 3').full_path

    asset_map = main_lib.AssetMap('bundle1')
    dest1 = asset_map.add_asset(file1)
    dest2 = asset_map.add_asset(file2)
    dest3 = asset_map.add_asset(file3)

    self.assertLen(asset_map.assets, 3)
    self.assertEqual(dest1, os.path.join('assets', 'a.txt'))
    self.assertEqual(dest2, os.path.join('assets', 'a_1.txt'))
    self.assertEqual(dest3, os.path.join('assets', 'a_2.txt'))

  def test_save_assets_empty(self):
    asset_map = main_lib.AssetMap('bundle1')
    export_dir = self.create_tempdir('export_path').full_path
    result = main_lib.save_assets(asset_map, export_dir)
    self.assertIsNone(result)

  def test_save_assets_success(self):
    src_dir = self.create_tempdir('src_path')
    export_dir = self.create_tempdir('export_path').full_path

    src_file1 = src_dir.create_file('asset1.txt', 'content1').full_path
    src_file2 = src_dir.create_file('asset2.txt', 'content2').full_path

    asset_map = main_lib.AssetMap('bundle1')
    asset_map.add_asset(src_file1)
    asset_map.add_asset(src_file2)

    result = main_lib.save_assets(asset_map, export_dir)
    self.assertIsNone(result)

    dest_file1 = pathlib.Path(export_dir, 'bundle1', 'assets', 'asset1.txt')
    dest_file2 = pathlib.Path(export_dir, 'bundle1', 'assets', 'asset2.txt')

    self.assertTrue(dest_file1.exists())
    self.assertTrue(dest_file2.exists())

    self.assertEqual(dest_file1.read_text(), 'content1')
    self.assertEqual(dest_file2.read_text(), 'content2')

  def test_asset_map_empty_bundle_name(self):
    with self.assertRaisesRegex(ValueError, 'Bundle name must not be empty'):
      main_lib.AssetMap('')

  def test_save_assets_outside_export_path(self):
    export_dir = self.create_tempdir('export').full_path
    src_file = self.create_tempdir('src').create_file('asset.txt').full_path

    asset_map = main_lib.AssetMap('bundle1')
    # Manually insert a malicious path into the internal dictionary
    # to test validation.
    asset_map._assets[src_file] = '../../outside.txt'

    with self.assertRaisesRegex(ValueError, 'outside of the model export path'):
      main_lib.save_assets(asset_map, export_dir)

  def test_save_assets_bundle_already_exists(self):
    export_dir = self.create_tempdir('export')
    export_dir.mkdir('bundle1')
    src_file = self.create_tempdir('src').create_file('asset.txt').full_path

    asset_map = main_lib.AssetMap('bundle1')
    asset_map.add_asset(src_file)

    with self.assertRaisesRegex(ValueError, 'Asset bundle .* already exists'):
      main_lib.save_assets(asset_map, export_dir.full_path)


if __name__ == '__main__':
  absltest.main()
