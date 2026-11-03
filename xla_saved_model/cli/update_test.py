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
from xla_saved_model._src.core import compile_options_util
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
      with_xla_compile_options: bool = False,
      inlined_xla_compile_options: bool = True,
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

    if with_xla_compile_options:
      opts_map = compile_options_util.generate_xla_compile_options(
          native_serialization_platforms=['tpu', 'cuda', 'cpu'],
          persist_xla_flags=True,
      )
      fn = manifest.objects['serving_default'].function
      fn.body.stable_hlo_body.stable_hlo.inlined_bytes = b'dummy_hlo'
      if inlined_xla_compile_options:
        fn.body.stable_hlo_body.supplemental_info[
            'xla_compile_options'
        ].CopyFrom(
            manifest_pb2.UnstructuredData(
                inlined_bytes=opts_map.SerializeToString(),
                mime_type=constants.MIME_TYPE_XLA_COMPILE_OPTIONS,
            )
        )
      else:
        opts_rel_path = 'compile_options.pb'
        opts_abs_path = os.path.join(model_dir, opts_rel_path)
        with open(opts_abs_path, 'wb') as f:
          f.write(opts_map.SerializeToString())
        fn.body.stable_hlo_body.supplemental_info[
            'xla_compile_options'
        ].CopyFrom(
            manifest_pb2.UnstructuredData(
                file_system_location=manifest_pb2.FileSystemLocation(
                    string_path=opts_rel_path
                ),
                mime_type=constants.MIME_TYPE_XLA_COMPILE_OPTIONS,
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

  def test_update_all_batch_options_and_rollback(self):
    model_dir = self._create_test_model(initial_batch_size=16)

    update.update(
        model_base_dir=model_dir,
        max_batch_size=64,
        batch_timeout_micros=5000,
        allowed_batch_sizes='8, 16, 32, 64',
        num_batch_threads=4,
        max_enqueued_batches=12,
        disable_large_batch_splitting=True,
        pad_variable_length_inputs=True,
    )

    pipelines = orchestration_pb2.Pipelines()
    pipelines_path = os.path.join(model_dir, 'orchestration.pb')
    with open(pipelines_path, 'rb') as f:
      pipelines.ParseFromString(f.read())

    opts = pipelines.name_to_pipeline['serving_default'].batch_options
    self.assertEqual(opts.max_batch_size, 64)
    self.assertEqual(opts.batch_timeout_micros, 5000)
    self.assertEqual(list(opts.allowed_batch_sizes), [8, 16, 32, 64])
    self.assertEqual(opts.num_batch_threads, 4)
    self.assertEqual(opts.max_enqueued_batches, 12)
    self.assertTrue(opts.disable_large_batch_splitting)
    self.assertTrue(opts.pad_variable_length_inputs)

    # Verify history record captures all 7 fields
    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertLen(history_proto.records, 2)
    changes = history_proto.records[1].changes
    self.assertEqual(changes['serving_default.max_batch_size'].new_value, '64')
    self.assertEqual(
        changes['serving_default.batch_timeout_micros'].new_value, '5000'
    )
    self.assertEqual(
        changes['serving_default.allowed_batch_sizes'].new_value,
        '[8, 16, 32, 64]',
    )
    self.assertEqual(
        changes['serving_default.num_batch_threads'].new_value, '4'
    )
    self.assertEqual(
        changes['serving_default.max_enqueued_batches'].new_value, '12'
    )
    self.assertEqual(
        changes['serving_default.disable_large_batch_splitting'].new_value,
        'True',
    )
    self.assertEqual(
        changes['serving_default.pad_variable_length_inputs'].new_value,
        'True',
    )

    # Verify rollback restores all fields and logs diffs in history
    update.rollback(model_base_dir=model_dir, to='orig')
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertLen(history_proto.records, 3)
    rb_changes = history_proto.records[2].changes
    self.assertEqual(
        rb_changes['serving_default.max_batch_size'].new_value, '16'
    )
    self.assertEqual(
        rb_changes['serving_default.allowed_batch_sizes'].new_value, '[]'
    )
    self.assertEqual(
        rb_changes['serving_default.pad_variable_length_inputs'].new_value,
        'False',
    )

  def test_update_validation_rejects_invalid_batch_options(self):
    model_dir = self._create_test_model(initial_batch_size=16)

    # No update flags specified
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir)

    # Negative batch_timeout_micros
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, batch_timeout_micros=-1)

    # Non-positive num_batch_threads or max_enqueued_batches
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, num_batch_threads=0)
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, max_enqueued_batches=0)

    # Malformed or non-ascending allowed_batch_sizes
    for bad_sizes in (
        '',
        '8,,16',
        '8,16,',
        '4,abc',
        '0,4,8',
        '-4,8',
        '8,4,16',
        '4,8,8,16',
    ):
      with self.assertRaises(typer.Exit):
        update.update(model_base_dir=model_dir, allowed_batch_sizes=bad_sizes)

    # allowed_batch_sizes[-1] > existing max_batch_size (16)
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, allowed_batch_sizes='4,8,32')

    # allowed_batch_sizes[-1] > newly provided max_batch_size (32)
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          max_batch_size=32,
          allowed_batch_sizes='8,16,64',
      )

    # allowed_batch_sizes[-1] != max_batch_size when
    # disable_large_batch_splitting=True
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          max_batch_size=32,
          allowed_batch_sizes='8,16',
          disable_large_batch_splitting=True,
      )

    # pad_variable_length_inputs=True without disable_large_batch_splitting=True
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          pad_variable_length_inputs=True,
      )

    # Valid update setting allowed_batch_sizes and pad_variable_length_inputs
    update.update(
        model_base_dir=model_dir,
        max_batch_size=32,
        allowed_batch_sizes='8,16,32',
        disable_large_batch_splitting=True,
        pad_variable_length_inputs=True,
    )

    # Lowering max_batch_size below existing allowed_batch_sizes[-1] (32) fails
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, max_batch_size=16)

    # Re-enabling large batch splitting while pad_variable_length_inputs is True
    # fails
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          disable_large_batch_splitting=False,
      )

  def test_update_specific_pipeline_name(self):
    model_dir = self._create_test_model(initial_batch_size=16)
    pipelines_path = os.path.join(model_dir, 'orchestration.pb')

    # Add a second pipeline to orchestration.pb
    pipelines = orchestration_pb2.Pipelines()
    with open(pipelines_path, 'rb') as f:
      pipelines.ParseFromString(f.read())
    pipelines.name_to_pipeline['prefill'].batch_options.max_batch_size = 8
    with open(pipelines_path, 'wb') as f:
      f.write(pipelines.SerializeToString())

    # Update only 'prefill' pipeline
    update.update(
        model_base_dir=model_dir,
        pipeline_name='prefill',
        max_batch_size=32,
    )

    updated = orchestration_pb2.Pipelines()
    with open(pipelines_path, 'rb') as f:
      updated.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(updated, 'serving_default'), 16)
    self.assertEqual(self._get_batch_size(updated, 'prefill'), 32)

    # Non-existent pipeline name raises typer.Exit
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          pipeline_name='non_existent',
          max_batch_size=64,
      )

  def _load_compile_options_map(
      self, model_dir: str, fn_name: str = 'serving_default'
  ) -> manifest_pb2.CompileOptionsProtoMap:
    manifest = manifest_pb2.Manifest()
    manifest_path = os.path.join(model_dir, core_constants.MANIFEST_FILE_PATH)
    with open(manifest_path, 'rb') as f:
      manifest.ParseFromString(f.read())
    body = manifest.objects[fn_name].function.body.stable_hlo_body
    data = body.supplemental_info['xla_compile_options']
    opts_map = manifest_pb2.CompileOptionsProtoMap()
    if data.HasField('inlined_bytes'):
      opts_map.ParseFromString(data.inlined_bytes)
    else:
      opts_path = os.path.join(model_dir, data.file_system_location.string_path)
      with open(opts_path, 'rb') as f:
        opts_map.ParseFromString(f.read())
    return opts_map

  def test_update_xla_flags_tpu_merge(self):
    model_dir = self._create_test_model(with_xla_compile_options=True)

    # 1st update: set two TPU XLA flags
    update.update(
        model_base_dir=model_dir,
        platform='tpu',
        xla_flags=(
            '--xla_jf_rematerialization_percent_shared_memory_limit=95,'
            ' --xla_sc_poison_buffers=false'
        ),
    )

    opts_map = self._load_compile_options_map(model_dir)
    tpu_opts = opts_map.map['tpu']
    if hasattr(compile_options_util, 'tpu_comp_env_pb2'):
      env = compile_options_util.tpu_comp_env_pb2.TpuCompilationEnvironment()
      tpu_opts.executable_build_options.comp_envs.environments[0].Unpack(env)
      self.assertEqual(
          env.xla_jf_rematerialization_percent_shared_memory_limit, 95
      )
      self.assertFalse(env.xla_sc_poison_buffers)
    else:
      self.assertEqual(
          tpu_opts.env_option_overrides[
              'xla_jf_rematerialization_percent_shared_memory_limit'
          ].int_field,
          95,
      )
      self.assertFalse(
          tpu_opts.env_option_overrides['xla_sc_poison_buffers'].bool_field
      )

    # 2nd update: merge (overwrite one flag, preserve the other, add a third)
    update.update(
        model_base_dir=model_dir,
        platform='tpu',
        xla_flags=(
            '--xla_jf_rematerialization_percent_shared_memory_limit=80'
            ' --xla_tpu_wait_n_cycles_before_program_termination=500'
        ),
    )

    opts_map = self._load_compile_options_map(model_dir)
    tpu_opts = opts_map.map['tpu']
    if hasattr(compile_options_util, 'tpu_comp_env_pb2'):
      env = compile_options_util.tpu_comp_env_pb2.TpuCompilationEnvironment()
      tpu_opts.executable_build_options.comp_envs.environments[0].Unpack(env)
      self.assertEqual(
          env.xla_jf_rematerialization_percent_shared_memory_limit, 80
      )
      self.assertFalse(env.xla_sc_poison_buffers)
      self.assertEqual(
          env.xla_tpu_wait_n_cycles_before_program_termination, 500
      )
    else:
      self.assertEqual(
          tpu_opts.env_option_overrides[
              'xla_jf_rematerialization_percent_shared_memory_limit'
          ].int_field,
          80,
      )
      self.assertFalse(
          tpu_opts.env_option_overrides['xla_sc_poison_buffers'].bool_field
      )
      self.assertEqual(
          tpu_opts.env_option_overrides[
              'xla_tpu_wait_n_cycles_before_program_termination'
          ].int_field,
          500,
      )

  def test_update_xla_flags_tpu_clear_and_rollback(self):
    model_dir = self._create_test_model(with_xla_compile_options=True)
    update.update(
        model_base_dir=model_dir,
        platform='tpu',
        xla_flags=(
            '--xla_jf_rematerialization_percent_shared_memory_limit=95,'
            ' --xla_sc_poison_buffers=false'
        ),
    )

    # Clear existing flags and apply a single flag
    update.update(
        model_base_dir=model_dir,
        platform='tpu',
        clear_xla_flags=True,
        xla_flags='--xla_jf_rematerialization_percent_shared_memory_limit=90',
    )
    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    self.assertEqual(
        history_proto.records[-1]
        .changes['serving_default.xla_flags.tpu']
        .new_value,
        '--xla_jf_rematerialization_percent_shared_memory_limit=90',
    )

    # Clear all persisted XLA flags without adding new ones
    update.update(
        model_base_dir=model_dir,
        platform='tpu',
        clear_xla_flags=True,
    )
    opts_map = self._load_compile_options_map(model_dir)
    tpu_opts = opts_map.map['tpu']
    self.assertEmpty(tpu_opts.executable_build_options.comp_envs.environments)
    self.assertEmpty(tpu_opts.env_option_overrides)

    # Rollback to .bak restores the previous flag (=90) from '(none)'
    update.rollback(model_base_dir=model_dir, to='bak')
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    rb_bak_change = history_proto.records[-1].changes[
        'serving_default.xla_flags.tpu'
    ]
    self.assertEqual(rb_bak_change.previous_value, '(none)')
    self.assertEqual(
        rb_bak_change.new_value,
        '--xla_jf_rematerialization_percent_shared_memory_limit=90',
    )

    # Rollback to .orig restores initial baseline TPU compilation environment
    update.rollback(model_base_dir=model_dir, to='orig')
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    rb_orig_change = history_proto.records[-1].changes[
        'serving_default.xla_flags.tpu'
    ]
    self.assertEqual(
        rb_orig_change.previous_value,
        '--xla_jf_rematerialization_percent_shared_memory_limit=90',
    )
    expected_orig = (
        '(default)'
        if hasattr(compile_options_util, 'tpu_comp_env_pb2')
        else '(none)'
    )
    self.assertEqual(rb_orig_change.new_value, expected_orig)

  def test_update_xla_flags_gpu_and_cpu(self):
    model_dir = self._create_test_model(
        with_xla_compile_options=True,
        inlined_xla_compile_options=False,
    )

    # Update GPU flags via --platform gpu
    update.update(
        model_base_dir=model_dir,
        platform='gpu',
        xla_flags=(
            '--xla_gpu_autotune_level=2, --xla_gpu_enable_triton_gemm=true'
        ),
    )

    opts_map = self._load_compile_options_map(model_dir)
    cuda_opts = opts_map.map['cuda']
    self.assertEqual(
        cuda_opts.env_option_overrides['xla_gpu_autotune_level'].int_field, 2
    )
    self.assertTrue(
        cuda_opts.env_option_overrides['xla_gpu_enable_triton_gemm'].bool_field
    )

    # Update CPU flags via --platform cpu
    update.update(
        model_base_dir=model_dir,
        platform='cpu',
        xla_flags='--xla_cpu_enable_fast_math=true',
    )
    opts_map = self._load_compile_options_map(model_dir)
    cpu_opts = opts_map.map['cpu']
    self.assertTrue(
        cpu_opts.env_option_overrides['xla_cpu_enable_fast_math'].bool_field
    )

    # Rollback external compile_options.pb to .bak (undoes CPU, keeps GPU)
    update.rollback(model_base_dir=model_dir, to='bak')
    opts_map = self._load_compile_options_map(model_dir)
    self.assertEmpty(opts_map.map['cpu'].env_option_overrides)
    self.assertEqual(
        opts_map.map['cuda']
        .env_option_overrides['xla_gpu_autotune_level']
        .int_field,
        2,
    )

  def test_update_xla_flags_function_name_and_combined_updates(self):
    model_dir = self._create_test_model(
        inlined_pipelines=True,
        with_xla_compile_options=True,
        inlined_xla_compile_options=True,
    )

    # Add a second StableHLO function 'prefill' to manifest.pb
    manifest_path = os.path.join(model_dir, core_constants.MANIFEST_FILE_PATH)
    manifest = manifest_pb2.Manifest()
    with open(manifest_path, 'rb') as f:
      manifest.ParseFromString(f.read())
    manifest.objects['prefill'].CopyFrom(manifest.objects['serving_default'])
    with open(manifest_path, 'wb') as f:
      f.write(manifest.SerializeToString())

    # Target only 'prefill' while also updating max_batch_size in a single call
    update.update(
        model_base_dir=model_dir,
        max_batch_size=64,
        function_name='prefill',
        platform='cpu',
        xla_flags='--xla_cpu_enable_fast_math=true',
    )

    with open(manifest_path, 'rb') as f:
      manifest.ParseFromString(f.read())
    updated_pipelines = orchestration_pb2.Pipelines()
    updated_pipelines.ParseFromString(
        manifest.supplemental_info['orchestration'].inlined_bytes
    )
    self.assertEqual(self._get_batch_size(updated_pipelines), 64)

    opts_serving = self._load_compile_options_map(model_dir, 'serving_default')
    opts_prefill = self._load_compile_options_map(model_dir, 'prefill')
    self.assertEmpty(opts_serving.map['cpu'].env_option_overrides)
    self.assertTrue(
        opts_prefill.map['cpu']
        .env_option_overrides['xla_cpu_enable_fast_math']
        .bool_field
    )

    # Rollback restores both inlined batch_options and xla_compile_options
    update.rollback(model_base_dir=model_dir, to='orig')
    opts_prefill_restored = self._load_compile_options_map(model_dir, 'prefill')
    self.assertEmpty(opts_prefill_restored.map['cpu'].env_option_overrides)

    history_path = os.path.join(model_dir, constants.HISTORY_FILENAME)
    history_proto = history_pb2.ModelHistory()
    with open(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
    rb_changes = history_proto.records[-1].changes
    self.assertEqual(
        rb_changes['serving_default.max_batch_size'].new_value, '16'
    )
    self.assertEqual(rb_changes['prefill.xla_flags.cpu'].new_value, '(none)')

  def test_rollback_combined_external_files(self):
    model_dir = self._create_test_model(
        inlined_pipelines=False,
        with_xla_compile_options=True,
        inlined_xla_compile_options=False,
    )

    update.update(
        model_base_dir=model_dir,
        max_batch_size=64,
        platform='cpu',
        xla_flags='--xla_cpu_enable_fast_math=true',
    )

    pipelines_path = os.path.join(model_dir, 'orchestration.pb')
    updated_pipelines = orchestration_pb2.Pipelines()
    with open(pipelines_path, 'rb') as f:
      updated_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(updated_pipelines), 64)
    self.assertTrue(
        self._load_compile_options_map(model_dir)
        .map['cpu']
        .env_option_overrides['xla_cpu_enable_fast_math']
        .bool_field
    )

    # Rollback restores both external orchestration.pb and compile_options.pb
    update.rollback(model_base_dir=model_dir, to='orig')
    restored_pipelines = orchestration_pb2.Pipelines()
    with open(pipelines_path, 'rb') as f:
      restored_pipelines.ParseFromString(f.read())
    self.assertEqual(self._get_batch_size(restored_pipelines), 16)
    self.assertEmpty(
        self._load_compile_options_map(model_dir).map['cpu'].env_option_overrides
    )

  def test_update_xla_flags_polymorphic_function(self):
    model_dir = self._create_test_model(
        with_xla_compile_options=True,
        inlined_xla_compile_options=True,
    )
    manifest_path = os.path.join(model_dir, core_constants.MANIFEST_FILE_PATH)
    manifest = manifest_pb2.Manifest()
    with open(manifest_path, 'rb') as f:
      manifest.ParseFromString(f.read())

    base_fn = manifest.objects['serving_default'].function
    del manifest.objects['serving_default']
    poly_fn = manifest.objects['poly_serving'].poly_fn
    poly_fn.concrete_functions.append(base_fn)
    poly_fn.concrete_functions.append(base_fn)
    with open(manifest_path, 'wb') as f:
      f.write(manifest.SerializeToString())

    # Target only the 2nd concrete overload ('poly_serving #2')
    update.update(
        model_base_dir=model_dir,
        function_name='poly_serving #2',
        platform='cpu',
        xla_flags='--xla_cpu_enable_fast_math=true',
    )

    with open(manifest_path, 'rb') as f:
      manifest.ParseFromString(f.read())
    opts_1 = manifest_pb2.CompileOptionsProtoMap.FromString(
        manifest.objects['poly_serving']
        .poly_fn.concrete_functions[0]
        .body.stable_hlo_body.supplemental_info['xla_compile_options']
        .inlined_bytes
    )
    opts_2 = manifest_pb2.CompileOptionsProtoMap.FromString(
        manifest.objects['poly_serving']
        .poly_fn.concrete_functions[1]
        .body.stable_hlo_body.supplemental_info['xla_compile_options']
        .inlined_bytes
    )
    self.assertEmpty(opts_1.map['cpu'].env_option_overrides)
    self.assertTrue(
        opts_2.map['cpu']
        .env_option_overrides['xla_cpu_enable_fast_math']
        .bool_field
    )

    # Out-of-bounds concrete index raises typer.Exit
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          function_name='poly_serving #3',
          platform='cpu',
          xla_flags='--xla_cpu_enable_fast_math=true',
      )

  def test_update_xla_flags_validation_errors(self):
    model_dir = self._create_test_model(with_xla_compile_options=True)

    # Empty xla_flags
    with self.assertRaises(typer.Exit):
      update.update(model_base_dir=model_dir, xla_flags='   ')

    # Malformed flag formats
    for bad_flag in ('no_dashes=1', '--missing_equals', '--=val'):
      with self.subTest(bad_flag=bad_flag):
        with self.assertRaises(typer.Exit):
          update.update(
              model_base_dir=model_dir, platform='cpu', xla_flags=bad_flag
          )

    # Invalid platform
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          platform='invalid_platform',
          xla_flags='--xla_cpu_enable_fast_math=true',
      )

    # Non-existent function_name
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          function_name='non_existent_fn',
          platform='cpu',
          xla_flags='--xla_cpu_enable_fast_math=true',
      )

    # Unknown or deprecated TPU flag (when internal TPU env is present)
    if hasattr(compile_options_util, 'tpu_comp_env_pb2'):
      with self.assertRaises(typer.Exit):
        update.update(
            model_base_dir=model_dir,
            platform='tpu',
            xla_flags='--xla_tpu_non_existent_flag=true',
        )
      with self.assertRaises(typer.Exit):
        update.update(
            model_base_dir=model_dir,
            platform='tpu',
            xla_flags='--xla_tpu_enable_experimental_fusion_cost_model=true',
        )

    # Non-GPU flag passed to GPU platform
    with self.assertRaises(typer.Exit):
      update.update(
          model_base_dir=model_dir,
          platform='gpu',
          xla_flags='--xla_cpu_enable_fast_math=true',
      )

    # Unsupported GPU flag (when internal GPU flag allowlist is present)
    if hasattr(compile_options_util, '_parse_env_option_overrides_for_gpu'):
      with self.assertRaises(typer.Exit):
        update.update(
            model_base_dir=model_dir,
            platform='gpu',
            xla_flags='--xla_gpu_unsupported_flag=true',
        )


if __name__ == '__main__':
  absltest.main()
