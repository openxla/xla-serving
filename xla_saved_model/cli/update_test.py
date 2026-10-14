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

"""Tests for XSM / OBM CLI `update`, `history`, and `rollback` commands."""

import os
from unittest import mock

from absl.testing import absltest
from google.protobuf import text_format
import typer
from typer.testing import CliRunner

from xla_saved_model.protos import history_pb2
from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.core import (
    constants as core_constants,
)
from xla_saved_model._src.core import metadata
from xla_saved_model.cli import constants
from xla_saved_model.cli import update


class UpdateTest(absltest.TestCase):

  def _create_test_model(
      self,
      inlined_pipelines: bool = False,
      with_history: bool = True,
      initial_batch_size: int = 16,
  ) -> str:
    """Creates a valid exported XSM model directory for testing."""
    model_dir = self.create_tempdir().full_path

    # Write model version file
    model_version = metadata.ModelVersion(
        version=core_constants.MANIFEST_VERSION,
        mime_type=core_constants.MANIFEST_MIME_TYPE,
        manifest_file_path=core_constants.MANIFEST_FILE_PATH,
    )
    model_version.save(
        os.path.join(model_dir, core_constants.MODEL_VERSION_FILENAME)
    )

    # Build Pipelines proto
    pipelines = orchestration_pb2.Pipelines()
    pipeline = pipelines.name_to_pipeline['serving_default']
    pipeline.batch_options.max_batch_size = initial_batch_size

    # Build Manifest proto
    manifest = manifest_pb2.Manifest()
    if inlined_pipelines:
      manifest.supplemental_info['orchestration'].CopyFrom(
          manifest_pb2.UnstructuredData(
              inlined_bytes=pipelines.SerializeToString(),
              mime_type=constants.MIME_TYPE_PIPELINES,
          )
      )
    else:
      pipelines_rel_path = 'orchestration.pb'
      pipelines_abs_path = os.path.join(model_dir, pipelines_rel_path)
      with open(pipelines_abs_path, 'wb') as f:
        f.write(pipelines.SerializeToString())

      manifest.supplemental_info['orchestration'].CopyFrom(
          manifest_pb2.UnstructuredData(
              file_system_location=manifest_pb2.FileSystemLocation(
                  string_path=pipelines_rel_path
              ),
              mime_type=constants.MIME_TYPE_PIPELINES,
          )
      )

    manifest_path = os.path.join(model_dir, core_constants.MANIFEST_FILE_PATH)
    with open(manifest_path, 'wb') as f:
      f.write(manifest.SerializeToString())

    if with_history:
      history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
      history_proto = history_pb2.ModelHistory(
          model_version=core_constants.MANIFEST_VERSION,
          records=[
              history_pb2.HistoryRecord(
                  timestamp='2026-08-28T00:00:00Z',
                  action='EXPORT',
                  user='testuser',
                  description='Initial model export.',
              )
          ],
      )
      with open(history_path, 'w') as f:
        f.write(text_format.MessageToString(history_proto))

    return model_dir

  def _get_batch_size(
      self,
      pipelines: orchestration_pb2.Pipelines,
      name: str = 'serving_default',
  ) -> int:
    return pipelines.name_to_pipeline[name].batch_options.max_batch_size

  def test_update_max_batch_size_in_place(self):
    model_dir = self._create_test_model(initial_batch_size=16)

    # Execute update
    update.update(model_base_dir=model_dir, max_batch_size=32)

    # Verify active proto has new batch size
    pipelines = orchestration_pb2.Pipelines()
    pipelines_path = os.path.join(model_dir, 'orchestration.pb')
    with open(pipelines_path, 'rb') as f:
      pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(pipelines), 32)

    # Verify .orig backup was created with original value
    orig_path = f'{pipelines_path}.orig'
    self.assertTrue(os.path.exists(orig_path))
    orig_pipelines = orchestration_pb2.Pipelines()
    with open(orig_path, 'rb') as f:
      orig_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(orig_pipelines), 16)

    # Verify .bak backup was created with previous value
    bak_path = f'{pipelines_path}.bak'
    self.assertTrue(os.path.exists(bak_path))
    bak_pipelines = orchestration_pb2.Pipelines()
    with open(bak_path, 'rb') as f:
      bak_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(bak_pipelines), 16)

    # Verify history.txtpb has both EXPORT and UPDATE entries
    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertLen(history_proto.records, 2)
    self.assertEqual(history_proto.records[0].action, 'EXPORT')
    self.assertEqual(history_proto.records[1].action, 'UPDATE')
    change = history_proto.records[1].changes['serving_default.max_batch_size']
    self.assertEqual(change.previous_value, '16')
    self.assertEqual(change.new_value, '32')

  def test_subsequent_updates_preserve_orig_and_rotate_bak(self):
    model_dir = self._create_test_model(initial_batch_size=16)

    # 1st update: 16 -> 32
    update.update(model_base_dir=model_dir, max_batch_size=32)

    # 2nd update: 32 -> 64
    update.update(model_base_dir=model_dir, max_batch_size=64)

    pipelines_path = os.path.join(model_dir, 'orchestration.pb')

    # .orig must be preserved as initial pristine 16
    orig_pipelines = orchestration_pb2.Pipelines()
    with open(f'{pipelines_path}.orig', 'rb') as f:
      orig_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(orig_pipelines), 16)

    # .bak must hold immediately prior value 32
    bak_pipelines = orchestration_pb2.Pipelines()
    with open(f'{pipelines_path}.bak', 'rb') as f:
      bak_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(bak_pipelines), 32)

    # Active proto holds 64
    current_pipelines = orchestration_pb2.Pipelines()
    with open(pipelines_path, 'rb') as f:
      current_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(current_pipelines), 64)

    # History contains 3 entries
    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertLen(history_proto.records, 3)

  def test_update_validation_rejects_non_positive_batch_size(self):
    model_dir = self._create_test_model(initial_batch_size=16)

    # Zero batch size
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, max_batch_size=0)

    # Negative batch size
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, max_batch_size=-10)

    # Ensure proto was untouched
    pipelines_path = os.path.join(model_dir, 'orchestration.pb')
    pipelines = orchestration_pb2.Pipelines()
    with open(pipelines_path, 'rb') as f:
      pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(pipelines), 16)

  def test_update_output_dir_fork_mode(self):
    model_dir = self._create_test_model(initial_batch_size=16)
    fork_dir = os.path.join(self.create_tempdir().full_path, 'forked_model')

    # Execute fork update
    update.update(
        model_base_dir=model_dir, max_batch_size=64, output_dir=fork_dir
    )

    # Source directory must remain untouched
    src_pipelines = orchestration_pb2.Pipelines()
    with open(os.path.join(model_dir, 'orchestration.pb'), 'rb') as f:
      src_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(src_pipelines), 16)
    orig_file = os.path.join(model_dir, 'orchestration.pb.orig')
    self.assertFalse(os.path.exists(orig_file))

    # Forked directory must have updated value
    fork_pipelines = orchestration_pb2.Pipelines()
    with open(os.path.join(fork_dir, 'orchestration.pb'), 'rb') as f:
      fork_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(fork_pipelines), 64)
    # Forked directory has update entry in its history
    fork_history = history_pb2.ModelHistory()
    with open(os.path.join(fork_dir, constants.HISTORY_FILENAME), 'r') as f:
      text_format.Parse(f.read(), fork_history)
    self.assertLen(fork_history.records, 2)

  def test_update_synthesizes_history_if_missing(self):
    model_dir = self._create_test_model(
        with_history=False, initial_batch_size=16
    )
    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    self.assertFalse(os.path.exists(history_path))

    update.update(model_base_dir=model_dir, max_batch_size=32)

    self.assertTrue(os.path.exists(history_path))
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertLen(history_proto.records, 2)
    self.assertEqual(history_proto.records[0].action, 'EXPORT')
    self.assertIn('synthesized', history_proto.records[0].description)
    self.assertEqual(history_proto.records[1].action, 'UPDATE')

  def test_update_inlined_pipelines(self):
    model_dir = self._create_test_model(
        inlined_pipelines=True, initial_batch_size=16
    )

    update.update(model_base_dir=model_dir, max_batch_size=32)

    manifest = manifest_pb2.Manifest()
    manifest_path = os.path.join(model_dir, core_constants.MANIFEST_FILE_PATH)
    with open(manifest_path, 'rb') as f:
      manifest.ParseFromString(f.read())

    pipelines = orchestration_pb2.Pipelines()
    pipelines.ParseFromString(
        manifest.supplemental_info['orchestration'].inlined_bytes
    )
    self.assertEqual(self._get_batch_size(pipelines), 32)
    # Manifest backup created
    self.assertTrue(
        os.path.exists(
            os.path.join(model_dir, f'{core_constants.MANIFEST_FILE_PATH}.orig')
        )
    )

  def test_cli_runner_invocation(self):
    model_dir = self._create_test_model(initial_batch_size=16)

    app = typer.Typer()
    app.command(name='update')(update.update)
    runner = CliRunner()

    result = runner.invoke(app, [model_dir, '--max_batch_size', '48'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('Successfully updated model', result.output)

    # Verify update took effect
    pipelines = orchestration_pb2.Pipelines()
    with open(os.path.join(model_dir, 'orchestration.pb'), 'rb') as f:
      pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(pipelines), 48)

  def test_atomic_write_cleans_up_tmp_file_on_failure(self):
    model_dir = self._create_test_model(initial_batch_size=16)
    pipelines_path = os.path.join(model_dir, 'orchestration.pb')
    tmp_path = f'{pipelines_path}.tmp'

    with mock.patch.object(
        update.file_utils, 'rename', side_effect=RuntimeError('Disk error')
    ):
      with self.assertRaises(RuntimeError):
        pipelines = orchestration_pb2.Pipelines()
        update._atomic_write_proto(pipelines_path, pipelines)

    # Verify .tmp was cleaned up and does not linger on disk
    self.assertFalse(os.path.exists(tmp_path))

  def test_rollback_to_bak_restores_previous_state_bit_for_bit(self):
    model_dir = self._create_test_model(initial_batch_size=16)
    pipelines_path = os.path.join(model_dir, 'orchestration.pb')

    # 1st update: 16 -> 32
    update.update(model_base_dir=model_dir, max_batch_size=32)
    with open(pipelines_path, 'rb') as f:
      expected_bak_bytes = f.read()

    # 2nd update: 32 -> 64
    update.update(model_base_dir=model_dir, max_batch_size=64)

    # Rollback to .bak (default)
    update.rollback(model_base_dir=model_dir)

    with open(pipelines_path, 'rb') as f:
      restored_bytes = f.read()
    self.assertEqual(restored_bytes, expected_bak_bytes)

    pipelines = orchestration_pb2.Pipelines()
    pipelines.ParseFromString(restored_bytes)
    self.assertEqual(self._get_batch_size(pipelines), 32)

    # Verify ROLLBACK entry in history.txtpb
    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertLen(history_proto.records, 4)
    self.assertEqual(history_proto.records[3].action, 'ROLLBACK')
    change = history_proto.records[3].changes['serving_default.max_batch_size']
    self.assertEqual(change.previous_value, '64')
    self.assertEqual(change.new_value, '32')

    # Calling rollback a second time is a no-op and does not append to history
    update.rollback(model_base_dir=model_dir)
    history_proto_after = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto_after)
    self.assertLen(history_proto_after.records, 4)

  def test_rollback_to_orig_restores_initial_export_state(self):
    model_dir = self._create_test_model(initial_batch_size=16)
    pipelines_path = os.path.join(model_dir, 'orchestration.pb')
    with open(pipelines_path, 'rb') as f:
      pristine_bytes = f.read()

    # Multiple updates: 16 -> 32 -> 64
    update.update(model_base_dir=model_dir, max_batch_size=32)
    update.update(model_base_dir=model_dir, max_batch_size=64)

    # Rollback to .orig
    update.rollback(model_base_dir=model_dir, to='orig')

    with open(pipelines_path, 'rb') as f:
      restored_bytes = f.read()
    self.assertEqual(restored_bytes, pristine_bytes)

    pipelines = orchestration_pb2.Pipelines()
    pipelines.ParseFromString(restored_bytes)
    self.assertEqual(self._get_batch_size(pipelines), 16)

    # Verify ROLLBACK entry in history.txtpb
    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertLen(history_proto.records, 4)
    self.assertEqual(history_proto.records[3].action, 'ROLLBACK')
    change = history_proto.records[3].changes['serving_default.max_batch_size']
    self.assertEqual(change.previous_value, '64')
    self.assertEqual(change.new_value, '16')

  def test_rollback_inlined_pipelines(self):
    model_dir = self._create_test_model(
        inlined_pipelines=True, initial_batch_size=16
    )
    update.update(model_base_dir=model_dir, max_batch_size=32)
    update.rollback(model_base_dir=model_dir, to='orig')

    manifest = manifest_pb2.Manifest()
    manifest_path = os.path.join(model_dir, core_constants.MANIFEST_FILE_PATH)
    with open(manifest_path, 'rb') as f:
      manifest.ParseFromString(f.read())

    pipelines = orchestration_pb2.Pipelines()
    pipelines.ParseFromString(
        manifest.supplemental_info['orchestration'].inlined_bytes
    )
    self.assertEqual(self._get_batch_size(pipelines), 16)

  def test_rollback_fails_gracefully_when_backup_missing_or_invalid(self):
    model_dir = self._create_test_model(initial_batch_size=16)

    # No backups exist before first in-place update
    with self.assertRaises(typer.Exit):
      update.rollback(model_base_dir=model_dir, to='bak')

    with self.assertRaises(typer.Exit):
      update.rollback(model_base_dir=model_dir, to='orig')

    # Invalid --to target
    with self.assertRaises(typer.Exit):
      update.rollback(model_base_dir=model_dir, to='invalid_target')

  def test_history_command_and_cli_runner(self):
    # Missing history.txtpb raises typer.Exit
    model_no_history = self._create_test_model(
        with_history=False, initial_batch_size=16
    )
    with self.assertRaises(typer.Exit):
      update.history(model_base_dir=model_no_history)

    # Full CLI workflow: update -> rollback -> history
    model_dir = self._create_test_model(initial_batch_size=16)
    app = typer.Typer()
    app.command(name='update')(update.update)
    app.command(name='history')(update.history)
    app.command(name='rollback')(update.rollback)
    runner = CliRunner()

    res_update = runner.invoke(
        app, ['update', model_dir, '--max_batch_size', '32']
    )
    self.assertEqual(res_update.exit_code, 0)

    res_rollback = runner.invoke(app, ['rollback', model_dir, '--to', 'orig'])
    self.assertEqual(res_rollback.exit_code, 0)
    self.assertIn('Successfully rolled back model', res_rollback.output)

    res_history = runner.invoke(app, ['history', model_dir])
    self.assertEqual(res_history.exit_code, 0)
    self.assertIn('EXPORT', res_history.output)
    self.assertIn('UPDATE', res_history.output)
    self.assertIn('ROLLBACK', res_history.output)
    self.assertIn('16 -> 32', res_history.output)
    self.assertIn('32 -> 16', res_history.output)


if __name__ == '__main__':
  absltest.main()
