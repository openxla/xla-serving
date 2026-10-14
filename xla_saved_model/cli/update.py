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

"""CLI `update`, `history`, and `rollback` commands for exported models."""

from collections.abc import Mapping
import datetime
import getpass
import os
from typing import Any, NoReturn

from google.protobuf import message
from google.protobuf import text_format
import rich
import rich.table
import typer
from typing_extensions import Annotated

from xla_saved_model.protos import history_pb2
from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.core import (
    constants as core_constants,
)
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import manifest_utils
from xla_saved_model._src.core import persistence_lib
from xla_saved_model.cli import constants


def _bail(msg: str) -> NoReturn:
  """Prints an error message and terminates the CLI with non-zero exit code."""
  typer.secho(f'Error: {msg}', fg=typer.colors.RED, err=True)
  raise typer.Exit(1)


def _get_current_user() -> str:
  """Returns the username of the current execution environment."""
  try:
    return getpass.getuser()
  except OSError:
    return 'unknown'


def _read_proto_from_data(
    model_base_dir: str,
    data: manifest_pb2.UnstructuredData,
    proto: message.Message,
) -> None:
  """Reads a proto from the given UnstructuredData."""
  try:
    proto.ParseFromString(
        manifest_utils.read_unstructured_data_bytes(data, model_base_dir)
    )
  except (ValueError, FileNotFoundError) as e:
    _bail(str(e))


def _backup_file(file_path: str) -> None:
  """Creates .orig (on first edit) and .bak (on subsequent edits) backups."""
  orig_path = f'{file_path}.orig'
  if not file_utils.exists(orig_path):
    file_utils.copy(file_path, orig_path)

  bak_path = f'{file_path}.bak'
  file_utils.copy(file_path, bak_path, overwrite=True)


def _atomic_write_proto(
    file_path: str, proto: message.Message, backup: bool = True
) -> None:
  """Safely and atomically writes a proto file to disk with backups."""
  if backup and file_utils.exists(file_path):
    _backup_file(file_path)

  # Validate serialization
  serialized_bytes = proto.SerializeToString()
  test_proto = type(proto)()
  test_proto.ParseFromString(serialized_bytes)

  tmp_path = f'{file_path}.tmp'
  try:
    with file_utils.open_file(tmp_path, 'wb') as f:
      f.write(serialized_bytes)
    file_utils.rename(tmp_path, file_path, overwrite=True)
  finally:
    if file_utils.exists(tmp_path):
      file_utils.remove(tmp_path)


def ensure_history(model_base_dir: str) -> history_pb2.ModelHistory:
  """Ensures history.txtpb exists, synthesizing a baseline entry if missing."""
  history_path = os.path.join(model_base_dir, constants.HISTORY_FILENAME)
  if file_utils.exists(history_path):
    try:
      history_proto = history_pb2.ModelHistory()
      with file_utils.open_file(history_path, 'r') as f:
        text_format.Parse(f.read(), history_proto)
      return history_proto
    except (IOError, text_format.ParseError) as e:
      _bail(f'Failed to parse {constants.HISTORY_FILENAME}: {e}')

  # Synthesize baseline entry for older exported models
  now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
  return history_pb2.ModelHistory(
      model_version=core_constants.MANIFEST_VERSION,
      records=[
          history_pb2.HistoryRecord(
              timestamp=now_iso,
              action='EXPORT',
              user='unknown',
              description='Baseline export (synthesized upon first update).',
          )
      ],
  )


def record_history_entry(
    model_base_dir: str,
    action: str,
    description: str,
    changes: Mapping[str, Any],
) -> None:
  """Appends an entry to history.txtpb and writes it atomically."""
  history_proto = ensure_history(model_base_dir)
  now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
  user = _get_current_user()

  change_protos = {
      k: history_pb2.Change(
          previous_value=str(v['previous']),
          new_value=str(v['new']),
      )
      for k, v in changes.items()
  }

  history_proto.records.append(
      history_pb2.HistoryRecord(
          timestamp=now_iso,
          action=action,
          user=user,
          description=description,
          changes=change_protos,
      )
  )

  history_path = os.path.join(model_base_dir, constants.HISTORY_FILENAME)
  tmp_path = f'{history_path}.tmp'
  try:
    with file_utils.open_file(tmp_path, 'w') as f:
      f.write(text_format.MessageToString(history_proto))
    file_utils.rename(tmp_path, history_path, overwrite=True)
  finally:
    if file_utils.exists(tmp_path):
      file_utils.remove(tmp_path)


def _find_pipelines_data(
    manifest: manifest_pb2.Manifest,
) -> tuple[str, manifest_pb2.UnstructuredData] | None:
  """Finds the pipelines unstructured data in the manifest."""
  for name, data in manifest.supplemental_info.items():
    if data.mime_type in (
        constants.MIME_TYPE_PIPELINES,
        constants.MIME_TYPE_ORBAX_PIPELINES,
        core_constants.ORCHESTRATION_MULTI_PIPELINE_MIME_TYPE,
    ):
      return name, data
  return None


def update(
    model_base_dir: Annotated[
        str, typer.Argument(help='Base directory of the exported model.')
    ],
    max_batch_size: Annotated[
        int | None,
        typer.Option(
            '--max_batch_size',
            help='Maximum batch size (must be a positive integer).',
        ),
    ] = None,
    output_dir: Annotated[
        str | None,
        typer.Option(
            '--output_dir',
            help=(
                'Output directory to clone the modified model to. If omitted,'
                ' modifies the model in-place.'
            ),
        ),
    ] = None,
) -> None:
  """Updates batching configurations and options in an exported model."""
  if max_batch_size is None:
    _bail('No update options specified. Provide at least --max_batch_size.')

  if max_batch_size <= 0:
    _bail(f'max_batch_size must be positive, got {max_batch_size}')

  try:
    manifest = persistence_lib.load(model_base_dir)
  except Exception as e:  # pylint: disable=broad-except
    _bail(f'Failed to load model at "{model_base_dir}": {e}')

  pipelines_entry = _find_pipelines_data(manifest)
  if not pipelines_entry:
    _bail('No orchestration pipelines found in the model manifest.')

  unused_name, pipelines_data = pipelines_entry
  pipelines = orchestration_pb2.Pipelines()
  _read_proto_from_data(model_base_dir, pipelines_data, pipelines)

  if not pipelines.name_to_pipeline:
    _bail('No named pipelines found inside pipelines proto.')

  target_dir = model_base_dir
  if output_dir:
    output_dir = os.path.abspath(output_dir)
    if os.path.abspath(model_base_dir) == output_dir:
      _bail('--output_dir cannot be the same as model_base_dir.')
    if file_utils.exists(output_dir):
      _bail(f'Output directory "{output_dir}" already exists.')
    file_utils.copytree(model_base_dir, output_dir)
    target_dir = output_dir

  changes: dict[str, Any] = {}
  for pipeline_name, pipeline in sorted(pipelines.name_to_pipeline.items()):
    prev_batch_size = pipeline.batch_options.max_batch_size
    pipeline.batch_options.max_batch_size = max_batch_size
    changes[f'{pipeline_name}.max_batch_size'] = {
        'previous': prev_batch_size,
        'new': max_batch_size,
    }

  # Write back mutated proto
  if pipelines_data.HasField('file_system_location'):
    pipelines_file_path = os.path.join(
        target_dir, pipelines_data.file_system_location.string_path
    )
    _atomic_write_proto(pipelines_file_path, pipelines, backup=(not output_dir))
  elif pipelines_data.HasField('inlined_bytes'):
    pipelines_data.inlined_bytes = pipelines.SerializeToString()
    manifest_file_path = os.path.join(
        target_dir, core_constants.MANIFEST_FILE_PATH
    )
    _atomic_write_proto(manifest_file_path, manifest, backup=(not output_dir))

  # Record change in history.txtpb
  record_history_entry(
      target_dir,
      action='UPDATE',
      description='Updated max_batch_size via CLI.',
      changes=changes,
  )

  # Output confirmation table
  table = rich.table.Table(title='Model Update Applied', padding=(0, 1))
  table.add_column('Pipeline Parameter', style='cyan')
  table.add_column('Previous Value', style='red')
  table.add_column('Updated Value', style='green')

  for key, change in changes.items():
    table.add_row(key, str(change['previous']), str(change['new']))

  rich.print(table)
  if output_dir:
    mode_str = f'Cloned to `{target_dir}`'
  else:
    mode_str = f'In-place at `{target_dir}`'
  rich.print(
      f'[bold green]Successfully updated model ({mode_str})[/bold green]'
  )


def _get_pipeline_batch_sizes(
    model_base_dir: str,
    manifest: manifest_pb2.Manifest | None = None,
) -> dict[str, int]:
  """Reads current max_batch_size for each pipeline in the model, if present."""
  try:
    if manifest is None:
      manifest = persistence_lib.load(model_base_dir)
    pipelines_entry = _find_pipelines_data(manifest)
    if not pipelines_entry:
      return {}
    unused_name, pipelines_data = pipelines_entry
    pipelines = orchestration_pb2.Pipelines()
    _read_proto_from_data(model_base_dir, pipelines_data, pipelines)
    return {
        name: pipeline.batch_options.max_batch_size
        for name, pipeline in sorted(pipelines.name_to_pipeline.items())
    }
  except Exception:  # pylint: disable=broad-except
    return {}


def _find_backup_target(
    model_base_dir: str, manifest: manifest_pb2.Manifest, suffix: str
) -> str | None:
  """Returns the target pipeline file path if a .<suffix> backup exists."""
  pipelines_entry = _find_pipelines_data(manifest)
  if not pipelines_entry:
    return None

  unused_name, pipelines_data = pipelines_entry
  if pipelines_data.HasField('file_system_location'):
    target_path = os.path.join(
        model_base_dir, pipelines_data.file_system_location.string_path
    )
  elif pipelines_data.HasField('inlined_bytes'):
    target_path = os.path.join(
        model_base_dir, core_constants.MANIFEST_FILE_PATH
    )
  else:
    return None

  backup_path = f'{target_path}.{suffix}'
  if file_utils.exists(backup_path):
    return target_path
  return None


def history(
    model_base_dir: Annotated[
        str, typer.Argument(help='Base directory of the exported model.')
    ],
) -> None:
  """Displays the modification history of an exported model."""
  if not file_utils.exists(model_base_dir):
    _bail(f'Model directory does not exist: "{model_base_dir}"')

  history_path = os.path.join(model_base_dir, constants.HISTORY_FILENAME)
  if not file_utils.exists(history_path):
    _bail(f'No modification history found at "{history_path}".')

  history_proto = history_pb2.ModelHistory()
  try:
    with file_utils.open_file(history_path, 'r') as f:
      text_format.Parse(f.read(), history_proto)
  except (IOError, text_format.ParseError) as e:
    _bail(f'Failed to parse {constants.HISTORY_FILENAME}: {e}')

  version_str = history_proto.model_version or 'unknown'
  table = rich.table.Table(
      title=f'Model Modification History (version: {version_str})',
      padding=(0, 1),
  )
  table.add_column('#', style='dim')
  table.add_column('Timestamp', style='cyan')
  table.add_column('User', style='magenta')
  table.add_column('Action', style='bold yellow')
  table.add_column('Description', style='white')
  table.add_column('Changes', style='green')

  for idx, record in enumerate(history_proto.records, start=1):
    if record.changes:
      changes_str = '\n'.join(
          f'{k}: {v.previous_value} -> {v.new_value}'
          for k, v in sorted(record.changes.items())
      )
    else:
      changes_str = '-'
    table.add_row(
        str(idx),
        record.timestamp,
        record.user,
        record.action,
        record.description,
        changes_str,
    )

  rich.print(table)


def rollback(
    model_base_dir: Annotated[
        str, typer.Argument(help='Base directory of the exported model.')
    ],
    to: Annotated[
        str,
        typer.Option(
            '--to',
            help=(
                'Backup target to restore from: "bak" (previous state, default)'
                ' or "orig" (original exported state).'
            ),
        ),
    ] = 'bak',
) -> None:
  """Restores modified model configuration files from backup (.bak or .orig).

  Only a single previous state (.bak) and the initial exported state (.orig)
  are maintained. If the active configuration already matches the target backup
  (e.g., when running rollback twice in a row), the operation is a no-op and
  does not append a duplicate ROLLBACK record to history.txtpb.
  """
  target_suffix = to.lower().strip()
  if target_suffix not in ('bak', 'orig'):
    _bail(f'Invalid rollback target "{to}". Expected "bak" or "orig".')

  try:
    manifest = persistence_lib.load(model_base_dir)
  except Exception as e:  # pylint: disable=broad-except
    _bail(f'Failed to load model at "{model_base_dir}": {e}')

  target_path = _find_backup_target(model_base_dir, manifest, target_suffix)
  if not target_path:
    _bail(
        f'No .{target_suffix} backup files found in "{model_base_dir}" to'
        ' rollback to.'
    )

  backup_path = f'{target_path}.{target_suffix}'
  with (
      file_utils.open_file(target_path, 'rb') as f_cur,
      file_utils.open_file(backup_path, 'rb') as f_bak,
  ):
    if f_cur.read() == f_bak.read():
      rich.print(
          '[yellow]Model configuration is already at the'
          f' .{target_suffix} backup state; nothing to rollback.[/yellow]'
      )
      return

  before_batch_sizes = _get_pipeline_batch_sizes(
      model_base_dir, manifest=manifest
  )

  tmp_path = f'{target_path}.tmp'
  try:
    file_utils.copy(backup_path, tmp_path, overwrite=True)
    file_utils.rename(tmp_path, target_path, overwrite=True)
  finally:
    if file_utils.exists(tmp_path):
      file_utils.remove(tmp_path)

  after_batch_sizes = _get_pipeline_batch_sizes(model_base_dir)

  changes: dict[str, Any] = {}
  for name, new_val in sorted(after_batch_sizes.items()):
    prev_val = before_batch_sizes.get(name)
    if prev_val is not None and prev_val != new_val:
      changes[f'{name}.max_batch_size'] = {
          'previous': prev_val,
          'new': new_val,
      }

  if not changes:
    rel_target = os.path.relpath(target_path, model_base_dir)
    changes[rel_target] = {
        'previous': 'modified',
        'new': f'restored (.{target_suffix})',
    }

  record_history_entry(
      model_base_dir,
      action='ROLLBACK',
      description=(
          f'Rolled back model configuration from .{target_suffix} backup.'
      ),
      changes=changes,
  )

  table = rich.table.Table(
      title=f'Model Rollback Applied (.{target_suffix})', padding=(0, 1)
  )
  table.add_column('Target / Parameter', style='cyan')
  table.add_column('Previous Value', style='red')
  table.add_column('Restored Value', style='green')

  for key, change in changes.items():
    table.add_row(key, str(change['previous']), str(change['new']))

  rich.print(table)
  rich.print(
      '[bold green]Successfully rolled back model from'
      f' .{target_suffix} backup at `{model_base_dir}`[/bold green]'
  )
