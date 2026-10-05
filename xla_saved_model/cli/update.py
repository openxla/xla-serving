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

"""CLI `update` command and core library for modifying exported models."""

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
