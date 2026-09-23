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

"""Tests for NPT CLI `show` command."""

from unittest import mock

from absl.testing import absltest
import typer
from typer.testing import CliRunner

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import type_pb2
from xla_saved_model.cli import show


class ShowTest(absltest.TestCase):

  @mock.patch.object(typer, 'secho')
  def test_read_proto_from_data_fails_with_absolute_path(self, mock_secho):
    data = manifest_pb2.UnstructuredData(
        file_system_location=manifest_pb2.FileSystemLocation(
            string_path='/absolute/path/to/proto.pb'
        )
    )
    proto = type_pb2.Type()

    with self.assertRaisesRegex(typer.Exit, '1'):
      show._read_proto_from_data('/base/dir', data, proto)

    mock_secho.assert_called_once_with(
        'Error: File system location must be relative, got'
        ' "/absolute/path/to/proto.pb"',
        fg=typer.colors.RED,
        err=True,
    )

  def test_show_details_short_flag(self):
    app = typer.Typer()
    app.command()(show.show)
    runner = CliRunner()
    result = runner.invoke(app, ['/nonexistent/path', '-d', 'serving_default'])
    self.assertIn('Failed to load XLA SavedModel', result.output)
    self.assertEqual(result.exit_code, 1)

  def test_get_stable_hlo_body_polymorphic_function(self):
    manifest = manifest_pb2.Manifest()
    poly_fn = manifest_pb2.PolymorphicFunction()

    fn1 = poly_fn.concrete_functions.add()
    fn1.body.stable_hlo_body.calling_convention_version = 1
    fn2 = poly_fn.concrete_functions.add()
    fn2.body.stable_hlo_body.calling_convention_version = 2

    manifest.objects['serving_default'].poly_fn.CopyFrom(poly_fn)

    # Defaults to first concrete function
    body1 = show._get_stable_hlo_body(manifest, 'serving_default')
    self.assertEqual(body1.calling_convention_version, 1)

    # Explicit concrete function index with #1
    body1_explicit = show._get_stable_hlo_body(manifest, 'serving_default #1')
    self.assertEqual(body1_explicit.calling_convention_version, 1)

    # Explicit concrete function index with #2
    body2 = show._get_stable_hlo_body(manifest, 'serving_default #2')
    self.assertEqual(body2.calling_convention_version, 2)


if __name__ == '__main__':
  absltest.main()
