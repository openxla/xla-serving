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
from neptune_model._src.core import persistence_lib
from neptune_model.protos import asset_map_pb2


class Assets2NptMainLibTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_suffix', 'foo.txt', 'foo.txt'),
      ('single_suffix', 'foo_1.txt', 'foo.txt'),
      ('large_suffix', 'foo_123.txt', 'foo.txt'),
      ('multiple_suffixes', 'foo_1_2.txt', 'foo.txt'),
      ('non_numeric', 'foo_bar.txt', 'foo_bar.txt'),
      ('trailing_underscore_no_num', 'foo_.txt', 'foo_.txt'),
      ('no_extension', 'foo_1', 'foo'),
  )
  def test_normalize_file_name(self, input_name, expected_name):
    self.assertEqual(main_lib._normalize_file_name(input_name), expected_name)

  def test_asset_map_builder_add_asset_idempotent(self):
    builder = main_lib._AssetMapBuilder()
    builder.add_asset('/path/to/foo.txt', 'my_supp')
    builder.add_asset('/path/to/foo.txt', 'my_supp')  # Should be skipped
    self.assertLen(builder.asset_map.assets, 1)
    self.assertEqual(
        builder.asset_map.assets['/path/to/foo.txt'], 'my_supp/assets/foo.txt'
    )

  def test_asset_map_builder_name_collision(self):
    builder = main_lib._AssetMapBuilder()
    builder.add_asset('/path1/foo.txt', 'my_supp')
    builder.add_asset('/path2/foo.txt', 'my_supp')
    builder.add_asset('/path3/foo_1.txt', 'my_supp')

    self.assertLen(builder.asset_map.assets, 3)
    self.assertEqual(
        builder.asset_map.assets['/path1/foo.txt'], 'my_supp/assets/foo.txt'
    )
    self.assertEqual(
        builder.asset_map.assets['/path2/foo.txt'], 'my_supp/assets/foo_1.txt'
    )
    self.assertEqual(
        builder.asset_map.assets['/path3/foo_1.txt'], 'my_supp/assets/foo_2.txt'
    )

  def test_build_asset_map(self):
    asset_sources = ['/path/b/foo.txt', '/path/a/foo.txt']
    asset_map = main_lib.build_asset_map('my_supp', asset_sources)

    self.assertLen(asset_map.assets, 2)
    self.assertEqual(
        asset_map.assets['/path/a/foo.txt'], 'my_supp/assets/foo.txt'
    )
    self.assertEqual(
        asset_map.assets['/path/b/foo.txt'], 'my_supp/assets/foo_1.txt'
    )

  def test_save_assets_invalid_supplemental_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Supplemental name must contain only alphanumeric'
    ):
      main_lib.save_assets('invalid name!', ['a', 'b'], '/tmp/export')

  def test_save_assets_empty_assets(self):
    result = main_lib.save_assets('my_supp', [], '/tmp/export')
    self.assertEqual(result, {})

  @parameterized.named_parameters(
      ('list_argument', False),
      ('map_argument', True),
  )
  def test_save_assets_success(self, use_map_argument):
    src_dir = self.create_tempdir('src_path').full_path
    export_dir = self.create_tempdir('export_path').full_path

    src_file1 = os.path.join(src_dir, 'asset1.txt')
    src_file2 = os.path.join(src_dir, 'asset2.txt')

    with open(src_file1, 'w') as f:
      f.write('content1')
    with open(src_file2, 'w') as f:
      f.write('content2')

    assets = [src_file1, src_file2]
    if use_map_argument:
      assets = main_lib.build_asset_map('my_supp', assets)

    result = main_lib.save_assets('my_supp', assets, export_dir)

    self.assertIsNotNone(result)
    self.assertIn('my_supp', result)
    supp = result['my_supp']
    self.assertIsInstance(supp, persistence_lib.GlobalSupplemental)
    self.assertEqual(supp.save_as, 'my_supp_asset_map.pb')
    self.assertEqual(
        supp.data.mime_type,
        'application/protobuf; type=neptune_assets.AssetMap',
    )
    self.assertEqual(supp.data.version, '0.0.1')

    asset_map = asset_map_pb2.AssetMap()
    asset_map.ParseFromString(supp.data.inlined_bytes)

    self.assertLen(asset_map.assets, 2)
    self.assertIn(src_file1, asset_map.assets)
    self.assertIn(src_file2, asset_map.assets)

    dest_file1 = os.path.join(export_dir, asset_map.assets[src_file1])
    dest_file2 = os.path.join(export_dir, asset_map.assets[src_file2])

    self.assertTrue(os.path.exists(dest_file1))
    self.assertTrue(os.path.exists(dest_file2))

    with open(dest_file1, 'r') as f:
      self.assertEqual(f.read(), 'content1')
    with open(dest_file2, 'r') as f:
      self.assertEqual(f.read(), 'content2')

  def test_save_assets_outside_export_path(self):
    export_dir = self.create_tempdir('export').full_path
    src_file = self.create_tempdir('src').create_file('asset.txt').full_path

    asset_map = asset_map_pb2.AssetMap()
    # Simulate a destination path that traverses outside export_path.
    asset_map.assets[src_file] = '../../outside.txt'

    with self.assertRaisesRegex(
        ValueError, 'outside of the model export path'
    ):
      main_lib.save_assets('my_supp', asset_map, export_dir)


if __name__ == '__main__':
  absltest.main()
