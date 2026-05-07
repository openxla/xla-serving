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

from neptune_model.cli import show
from neptune_model.protos import manifest_pb2
from neptune_model.protos import type_pb2


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


if __name__ == '__main__':
  absltest.main()
