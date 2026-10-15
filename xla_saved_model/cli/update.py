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

from collections.abc import Iterator, Mapping, Sequence
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
from xla_saved_model._src.core import compile_options_util
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


def _pipelines_file_rel_path(manifest: manifest_pb2.Manifest) -> str | None:
  """Returns the relative path of the pipelines file, if not inlined."""
  pipelines_entry = _find_pipelines_data(manifest)
  if pipelines_entry and pipelines_entry[1].HasField('file_system_location'):
    return pipelines_entry[1].file_system_location.string_path
  return None


def _iter_compile_options_data(
    manifest: manifest_pb2.Manifest,
) -> Iterator[tuple[str, str, manifest_pb2.UnstructuredData]]:
  """Yields the XLA compile options supplementals of all StableHLO functions.

  Args:
    manifest: The model manifest.

  Yields:
    `(object_name, function_label, data)` tuples. `function_label` is the
    object name for a plain function, and `<object_name> #<i>` (1-based) for
    the i-th concrete function of a polymorphic function (same as `show`).
  """
  for obj_name, obj in sorted(manifest.objects.items()):
    if obj.HasField('function'):
      fns = [(obj_name, obj.function)]
    elif obj.HasField('poly_fn'):
      fns = [
          (f'{obj_name} #{i}', fn)
          for i, fn in enumerate(obj.poly_fn.concrete_functions, start=1)
      ]
    else:
      continue
    for fn_label, fn in fns:
      if not fn.body.HasField('stable_hlo_body'):
        continue
      supplemental_info = fn.body.stable_hlo_body.supplemental_info
      for _, data in sorted(supplemental_info.items()):
        if data.mime_type in (
            constants.MIME_TYPE_XLA_COMPILE_OPTIONS,
            constants.MIME_TYPE_ORBAX_XLA_COMPILE_OPTIONS,
        ):
          yield obj_name, fn_label, data


def _config_file_rel_paths(manifest: manifest_pb2.Manifest) -> list[str]:
  """Returns relative paths of config files `update` may modify."""
  rel_paths = [core_constants.MANIFEST_FILE_PATH]
  pipelines_rel_path = _pipelines_file_rel_path(manifest)
  if pipelines_rel_path:
    rel_paths.append(pipelines_rel_path)
  for _, _, data in _iter_compile_options_data(manifest):
    if data.HasField('file_system_location'):
      rel_paths.append(data.file_system_location.string_path)
  return list(dict.fromkeys(rel_paths))


def _backup_config_files(
    model_base_dir: str,
    manifest: manifest_pb2.Manifest,
    modified_rel_paths: Sequence[str],
) -> None:
  """Backs up config files before they are modified in-place.

  Files about to be modified get `.orig` (on first edit) and `.bak` backups.
  Files which already have a `.bak` from an earlier update get their `.bak`
  refreshed as well, so that all `.bak` files consistently capture the model
  state right before the latest update.

  Args:
    model_base_dir: Base directory of the exported model.
    manifest: The model manifest.
    modified_rel_paths: Relative paths of the files about to be modified.
  """
  rel_paths = dict.fromkeys(
      list(_config_file_rel_paths(manifest)) + list(modified_rel_paths)
  )
  for rel_path in rel_paths:
    file_path = os.path.join(model_base_dir, rel_path)
    if not file_utils.exists(file_path):
      continue
    if rel_path in modified_rel_paths or file_utils.exists(f'{file_path}.bak'):
      _backup_file(file_path)


def _update_max_batch_size(
    model_base_dir: str,
    manifest: manifest_pb2.Manifest,
    max_batch_size: int,
    changes: dict[str, Any],
    writes: dict[str, message.Message],
) -> bool:
  """Updates max_batch_size of all pipelines in memory.

  Args:
    model_base_dir: Base directory of the exported model.
    manifest: The model manifest. Mutated if pipelines are inlined.
    max_batch_size: The new max batch size.
    changes: Dict to record the applied changes into.
    writes: Dict (relative path -> proto) to record files to write back into.

  Returns:
    Whether the manifest was modified.
  """
  pipelines_entry = _find_pipelines_data(manifest)
  if not pipelines_entry:
    _bail('No orchestration pipelines found in the model manifest.')

  unused_name, pipelines_data = pipelines_entry
  pipelines = orchestration_pb2.Pipelines()
  _read_proto_from_data(model_base_dir, pipelines_data, pipelines)

  if not pipelines.name_to_pipeline:
    _bail('No named pipelines found inside pipelines proto.')

  for pipeline_name, pipeline in sorted(pipelines.name_to_pipeline.items()):
    prev_batch_size = pipeline.batch_options.max_batch_size
    pipeline.batch_options.max_batch_size = max_batch_size
    changes[f'{pipeline_name}.max_batch_size'] = {
        'previous': prev_batch_size,
        'new': max_batch_size,
    }

  if pipelines_data.HasField('file_system_location'):
    writes[pipelines_data.file_system_location.string_path] = pipelines
  elif pipelines_data.HasField('inlined_bytes'):
    pipelines_data.inlined_bytes = pipelines.SerializeToString()
    return True
  return False


def _update_xla_flags(
    model_base_dir: str,
    manifest: manifest_pb2.Manifest,
    xla_flags: Sequence[str],
    platform: str | None,
    functions: Sequence[str] | None,
    changes: dict[str, Any],
    writes: dict[str, message.Message],
) -> bool:
  """Applies XLA flag overrides to the XLA compile options in memory.

  Args:
    model_base_dir: Base directory of the exported model.
    manifest: The model manifest. Mutated if compile options are inlined.
    xla_flags: XLA flags in the format '--flag_name=flag_value'.
    platform: The platform whose compile options to modify. If None, the compile
      options must contain exactly one platform.
    functions: Names of the functions whose compile options to modify. If empty,
      the compile options of all functions are modified.
    changes: Dict to record the applied changes into.
    writes: Dict (relative path -> proto) to record files to write back into.

  Returns:
    Whether the manifest was modified.
  """
  entries = list(_iter_compile_options_data(manifest))
  if functions:
    requested = set(functions)
    available = {obj_name for obj_name, _, _ in entries} | {
        fn_label for _, fn_label, _ in entries
    }
    missing = sorted(requested - available)
    if missing:
      _bail(
          'No XLA compile options found for function(s):'
          f' {", ".join(missing)}. Functions with XLA compile options:'
          f' {", ".join(sorted(available)) or "<none>"}.'
      )
    entries = [
        entry
        for entry in entries
        if entry[0] in requested or entry[1] in requested
    ]
  if not entries:
    _bail('No XLA compile options found in the model manifest.')

  manifest_modified = False
  updated_files = set()
  for _, fn_label, data in entries:
    rel_path = None
    if data.HasField('file_system_location'):
      rel_path = data.file_system_location.string_path
      if rel_path in updated_files:
        # The same compile options file is shared by multiple functions.
        continue
      updated_files.add(rel_path)

    opts = manifest_pb2.CompileOptionsProtoMap()
    _read_proto_from_data(model_base_dir, data, opts)

    available_platforms = ', '.join(sorted(opts.map)) or '<none>'
    platform_keys = {key.lower(): key for key in opts.map}
    if platform:
      target_platform = platform_keys.get(platform.strip().lower())
      if target_platform is None:
        _bail(
            f'Function `{fn_label}` has no XLA compile options for platform'
            f' "{platform}" (available: {available_platforms}).'
        )
    elif len(platform_keys) == 1:
      target_platform = next(iter(opts.map))
    else:
      _bail(
          f'Function `{fn_label}` has XLA compile options for platforms:'
          f' {available_platforms}. Use --platform to select one.'
      )

    try:
      flag_changes = compile_options_util.apply_xla_flags_overrides(
          opts.map[target_platform], target_platform, xla_flags
      )
    except ValueError as e:
      _bail(
          f'Failed to apply XLA flags to `{fn_label}` ({target_platform}): {e}'
      )

    for flag_name, (previous, new) in flag_changes.items():
      changes[f'{fn_label}.{target_platform}.{flag_name}'] = {
          'previous': previous,
          'new': new,
      }

    if rel_path is not None:
      writes[rel_path] = opts
    elif data.HasField('inlined_bytes'):
      data.inlined_bytes = opts.SerializeToString()
      manifest_modified = True
  return manifest_modified


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
    xla_flags: Annotated[
        list[str] | None,
        typer.Option(
            '--xla_flags',
            help=(
                'XLA flags to set in the XLA compile options, in the format'
                ' "--flag_name=flag_value". Can be repeated, and each value'
                ' may contain multiple whitespace-separated flags (like'
                ' XLA_FLAGS). Flags that are not specified are left'
                ' untouched.'
            ),
        ),
    ] = None,
    platform: Annotated[
        str | None,
        typer.Option(
            '--platform',
            help=(
                'Platform (e.g. tpu, cuda) of the XLA compile options to'
                ' modify with --xla_flags. Required if the compile options'
                ' contain more than one platform.'
            ),
        ),
    ] = None,
    functions: Annotated[
        list[str] | None,
        typer.Option(
            '--function',
            help=(
                'Name of a function whose XLA compile options to modify with'
                ' --xla_flags. Can be repeated. If omitted, all functions'
                ' with XLA compile options are modified.'
            ),
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
  """Updates batching configurations and XLA flags in an exported model."""
  flags = [flag for value in xla_flags or [] for flag in value.split()]
  if max_batch_size is None and not flags:
    _bail(
        'No update options specified. Provide at least --max_batch_size or'
        ' --xla_flags.'
    )

  if max_batch_size is not None and max_batch_size <= 0:
    _bail(f'max_batch_size must be positive, got {max_batch_size}')

  if (platform or functions) and not flags:
    _bail('--platform and --function can only be used with --xla_flags.')

  try:
    manifest = persistence_lib.load(model_base_dir)
  except Exception as e:  # pylint: disable=broad-except
    _bail(f'Failed to load model at "{model_base_dir}": {e}')

  # Apply all modifications in memory first, so that invalid options never
  # leave the model partially modified.
  changes: dict[str, Any] = {}
  # Relative path -> mutated proto to write back.
  writes: dict[str, message.Message] = {}
  manifest_modified = False
  descriptions = []
  if max_batch_size is not None:
    manifest_modified |= _update_max_batch_size(
        model_base_dir, manifest, max_batch_size, changes, writes
    )
    descriptions.append('max_batch_size')
  if flags:
    manifest_modified |= _update_xla_flags(
        model_base_dir, manifest, flags, platform, functions, changes, writes
    )
    descriptions.append('XLA flags')
  if manifest_modified:
    writes[core_constants.MANIFEST_FILE_PATH] = manifest

  target_dir = model_base_dir
  if output_dir:
    output_dir = os.path.abspath(output_dir)
    if os.path.abspath(model_base_dir) == output_dir:
      _bail('--output_dir cannot be the same as model_base_dir.')
    if file_utils.exists(output_dir):
      _bail(f'Output directory "{output_dir}" already exists.')
    file_utils.copytree(model_base_dir, output_dir)
    target_dir = output_dir
  else:
    _backup_config_files(target_dir, manifest, list(writes))

  # Write back mutated protos. The manifest (if modified) is written last.
  for rel_path, proto in writes.items():
    _atomic_write_proto(os.path.join(target_dir, rel_path), proto, backup=False)

  # Record change in history.txtpb
  record_history_entry(
      target_dir,
      action='UPDATE',
      description=f'Updated {" and ".join(descriptions)} via CLI.',
      changes=changes,
  )

  # Output confirmation table
  table = rich.table.Table(title='Model Update Applied', padding=(0, 1))
  table.add_column('Parameter', style='cyan')
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


def _find_backup_targets(
    model_base_dir: str, manifest: manifest_pb2.Manifest, suffix: str
) -> list[str]:
  """Returns relative paths of config files with a .<suffix> backup."""
  return [
      rel_path
      for rel_path in _config_file_rel_paths(manifest)
      if file_utils.exists(os.path.join(model_base_dir, f'{rel_path}.{suffix}'))
  ]


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

  target_rel_paths = _find_backup_targets(
      model_base_dir, manifest, target_suffix
  )
  if not target_rel_paths:
    _bail(
        f'No .{target_suffix} backup files found in "{model_base_dir}" to'
        ' rollback to.'
    )

  to_restore = []
  for rel_path in target_rel_paths:
    target_path = os.path.join(model_base_dir, rel_path)
    with (
        file_utils.open_file(target_path, 'rb') as f_cur,
        file_utils.open_file(f'{target_path}.{target_suffix}', 'rb') as f_bak,
    ):
      if f_cur.read() != f_bak.read():
        to_restore.append(rel_path)
  if not to_restore:
    rich.print(
        '[yellow]Model configuration is already at the'
        f' .{target_suffix} backup state; nothing to rollback.[/yellow]'
    )
    return
  # Restore the manifest last, since it references the other files.
  to_restore.sort(key=lambda p: p == core_constants.MANIFEST_FILE_PATH)

  before_batch_sizes = _get_pipeline_batch_sizes(
      model_base_dir, manifest=manifest
  )
  pipelines_rel_path = _pipelines_file_rel_path(manifest)

  for rel_path in to_restore:
    target_path = os.path.join(model_base_dir, rel_path)
    backup_path = f'{target_path}.{target_suffix}'
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

  for rel_path in to_restore:
    if changes and rel_path == pipelines_rel_path:
      # Already described by the max_batch_size changes above.
      continue
    changes[rel_path] = {
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
