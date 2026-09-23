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

import pathlib

from absl.testing import absltest

from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src.core import unstructured_data


class UnstructuredDataTest(absltest.TestCase):

  def test_build_relative_filepath_from_extension(self):
    self.assertEqual(
        unstructured_data.build_relative_filepath_from_extension("foo", "txt"),
        "foo.txt",
    )
    self.assertEqual(
        unstructured_data.build_relative_filepath_from_extension(
            "foo", "txt", subfolder="bar"
        ),
        "bar/foo.txt",
    )

  def test_write_inlined_string_to_file(self):
    proto = manifest_pb2.UnstructuredData()
    proto.inlined_string = "hello"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    new_proto = unstructured_data.write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "text/plain")
    self.assertEqual(new_proto.version, "1")
    with open(pathlib.Path(dirname) / filename, "r") as f:
      self.assertEqual(f.read(), "hello")

  def test_write_inlined_bytes_to_file(self):
    proto = manifest_pb2.UnstructuredData()
    proto.inlined_bytes = b"hello"
    proto.mime_type = "application/octet-stream"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.bin"
    new_proto = unstructured_data.write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "application/octet-stream")
    self.assertEqual(new_proto.version, "1")
    with open(pathlib.Path(dirname) / filename, "rb") as f:
      self.assertEqual(f.read(), b"hello")

  def test_write_location_pointer_to_file(self):
    proto = manifest_pb2.UnstructuredData()
    proto.file_system_location.string_path = "path/to/file"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    with self.assertRaisesRegex(
        ValueError, "Can only write inlined data .*to a file"
    ):
      unstructured_data.write_inlined_data_to_file(proto, dirname, filename)

  def test_write_inlined_data_to_file_traversal(self):
    proto = manifest_pb2.UnstructuredData()
    proto.inlined_string = "hello"
    dirname = self.create_tempdir().full_path
    filename = "../data.txt"
    with self.assertRaisesRegex(
        ValueError, "contains directory traversal component"
    ):
      unstructured_data.write_inlined_data_to_file(proto, dirname, filename)

  def test_maybe_write_inlined_string_to_file(self):
    proto = manifest_pb2.UnstructuredData()
    proto.inlined_string = "hello"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    new_proto = unstructured_data.maybe_write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "text/plain")
    self.assertEqual(new_proto.version, "1")
    with open(pathlib.Path(dirname) / filename, "r") as f:
      self.assertEqual(f.read(), "hello")

  def test_maybe_write_inlined_bytes_to_file(self):
    proto = manifest_pb2.UnstructuredData()
    proto.inlined_bytes = b"hello"
    proto.mime_type = "application/octet-stream"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.bin"
    new_proto = unstructured_data.maybe_write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, filename)
    self.assertEqual(new_proto.mime_type, "application/octet-stream")
    self.assertEqual(new_proto.version, "1")
    with open(pathlib.Path(dirname) / filename, "rb") as f:
      self.assertEqual(f.read(), b"hello")

  def test_maybe_write_location_pointer_to_file(self):
    proto = manifest_pb2.UnstructuredData()
    proto.file_system_location.string_path = "path/to/file"
    proto.mime_type = "text/plain"
    proto.version = "1"
    dirname = self.create_tempdir().full_path
    filename = "data.txt"
    new_proto = unstructured_data.maybe_write_inlined_data_to_file(
        proto, dirname, filename
    )
    self.assertEqual(new_proto.WhichOneof("data"), "file_system_location")
    self.assertEqual(new_proto.file_system_location.string_path, "path/to/file")
    with self.assertRaises(FileNotFoundError):
      with open(pathlib.Path(dirname) / filename, "r") as f:
        f.read()


if __name__ == "__main__":
  absltest.main()
