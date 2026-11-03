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

from collections.abc import Collection, Mapping, Sequence
import datetime
import getpass
import os
from typing import Any, NoReturn

from google.protobuf import descriptor
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
from xla_saved_model.protos.xla.pjrt.proto import compile_options_pb2  # pylint: disable=g-direct-tensorflow-import

_VALID_PLATFORMS = ('cpu', 'cuda', 'gpu', 'rocm', 'tpu')


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


def _parse_allowed_batch_sizes(raw: str) -> list[int]:
  """Parses and validates a comma-separated list of allowed batch sizes."""
  if not raw.strip():
    _bail('allowed_batch_sizes cannot be empty.')
  parts = [p.strip() for p in raw.split(',')]
  try:
    sizes = [int(p) for p in parts]
  except ValueError:
    _bail(
        'allowed_batch_sizes must be a comma-separated list of integers, got'
        f' "{raw}".'
    )
  if any(s <= 0 for s in sizes):
    _bail(
        'allowed_batch_sizes must contain strictly positive integers, got'
        f' {sizes}.'
    )
  if any(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1)):
    _bail(
        f'allowed_batch_sizes must be in strictly ascending order, got {sizes}.'
    )
  return sizes


def _normalize_platform(platform: str) -> str:
  """Normalizes and validates a compilation platform name."""
  normalized = platform.strip().lower()
  if normalized not in _VALID_PLATFORMS:
    _bail(
        f'Invalid platform "{platform}". Supported platforms: cpu, cuda (or'
        ' gpu), rocm, tpu.'
    )
  return 'cuda' if normalized == 'gpu' else normalized


def _parse_xla_flags_list(raw: str) -> list[str]:
  """Parses space- or comma-separated XLA flags in --name=value format."""
  tokens = raw.replace(',', ' ').split()
  if not tokens:
    _bail('xla_flags cannot be empty.')
  for flag in tokens:
    if not flag.startswith('--') or '=' not in flag[2:]:
      _bail(
          f'Invalid XLA flag "{flag}". All flags must be in the format'
          ' --flag_name=flag_value.'
      )
    flag_name, _ = flag[2:].split('=', 1)
    if not flag_name:
      _bail(
          f'Invalid XLA flag "{flag}". Flag name cannot be empty in'
          ' --flag_name=flag_value.'
      )
  return tokens


def _validate_xla_flags_for_platform(
    platform: str, parsed_flags: Sequence[str]
) -> None:
  """Validates XLA flags for the target platform before mutating model state."""
  if not parsed_flags:
    return
  try:
    if platform == 'tpu' and hasattr(compile_options_util, 'tpu_comp_env_pb2'):
      tpu_flags: dict[str, str] = {}
      for flag in parsed_flags:
        flag_name, flag_value = flag[2:].split('=', 1)
        field_descriptor = compile_options_util._get_field_for_flag(  # pylint: disable=protected-access
            flag_name
        )
        if field_descriptor.GetOptions().deprecated:
          raise ValueError(
              '[DEPRECATED_XLA_TPU_FLAG_USE] TpuCompilationEnvironment has'
              f' deprecated fields in use: {flag_name}'
          )
        tpu_flags[flag_name] = flag_value
      compile_options_util.parse_tpu_flags(tpu_flags)
    elif platform in ('cuda', 'gpu') and hasattr(
        compile_options_util, '_parse_env_option_overrides_for_gpu'
    ):
      compile_options_util._parse_env_option_overrides_for_gpu(  # pylint: disable=protected-access
          parsed_flags
      )
    else:
      for flag in parsed_flags:
        flag_name, _ = flag[2:].split('=', 1)
        if not flag_name.startswith('xla_'):
          raise ValueError(
              f'XLA flag "{flag}" is invalid: flag name must start with "xla_".'
          )
        if platform in ('cuda', 'gpu') and not flag_name.startswith('xla_gpu'):
          raise ValueError(f'{flag} is not a valid XLA GPU compilation flag.')
  except ValueError as e:
    _bail(str(e))


def _option_override_to_str(
    override: compile_options_pb2.OptionOverrideProto,
) -> str:
  """Formats an OptionOverrideProto value as a string."""
  which = override.WhichOneof('value')
  if which == 'bool_field':
    return 'true' if override.bool_field else 'false'
  if which == 'int_field':
    return str(override.int_field)
  if which == 'double_field':
    return str(override.double_field)
  if which == 'string_field':
    return override.string_field
  return ''


def _str_to_option_override(
    value: str,
) -> compile_options_pb2.OptionOverrideProto:
  """Parses a string flag value into an OptionOverrideProto."""
  override_proto = compile_options_pb2.OptionOverrideProto()
  if value.lower() == 'true':
    override_proto.bool_field = True
  elif value.lower() == 'false':
    override_proto.bool_field = False
  elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
    override_proto.int_field = int(value)
  else:
    try:
      override_proto.double_field = float(value)
    except ValueError:
      override_proto.string_field = value
  return override_proto


def _format_tpu_field_value(fd: descriptor.FieldDescriptor, val: Any) -> str:
  """Formats a TpuCompilationEnvironment field value as a CLI string."""
  if fd.type == descriptor.FieldDescriptor.TYPE_BOOL:
    return 'true' if val else 'false'
  if fd.type == descriptor.FieldDescriptor.TYPE_ENUM:
    if fd.label == descriptor.FieldDescriptor.LABEL_REPEATED:
      return ','.join(
          fd.enum_type.values_by_number[x].name.lower() for x in val
      )
    return fd.enum_type.values_by_number[val].name.lower()
  if fd.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
    return text_format.MessageToString(val, as_one_line=True).strip()
  if fd.label == descriptor.FieldDescriptor.LABEL_REPEATED:
    return ','.join(str(x) for x in val)
  return str(val)


def _format_xla_flags(
    compile_options: compile_options_pb2.CompileOptionsProto,
    explicit_keys: Collection[str] | None = None,
) -> str:
  """Formats customized XLA flags in CompileOptionsProto for display."""
  flags_dict: dict[str, str] = {}
  comp_envs = compile_options.executable_build_options.comp_envs
  if comp_envs.environments and hasattr(
      compile_options_util, 'tpu_comp_env_pb2'
  ):
    tpu_env_cls = (
        compile_options_util.tpu_comp_env_pb2.TpuCompilationEnvironment
    )
    actual_env = tpu_env_cls()
    if comp_envs.environments[0].Unpack(actual_env):
      default_env = tpu_env_cls.FromString(
          compile_options_util.tpu_comp_env.create_default_tpu_comp_env()
      )
      actual_fields = actual_env.ListFields()
      is_sparse = len(actual_fields) < len(default_env.ListFields())
      for fd, val in actual_fields:
        is_explicit = explicit_keys is not None and fd.name in explicit_keys
        is_diff = getattr(default_env, fd.name) != val
        if is_explicit or is_sparse or is_diff:
          flags_dict[fd.name] = (
              f'--{fd.name}={_format_tpu_field_value(fd, val)}'
          )

  for k, v in sorted(compile_options.env_option_overrides.items()):
    flags_dict[k] = f'--{k}={_option_override_to_str(v)}'

  if flags_dict:
    return ' '.join(flags_dict[k] for k in sorted(flags_dict.keys()))
  if comp_envs.environments:
    return '(default)'
  return '(none)'


def _apply_xla_flags_to_compile_options(
    compile_options: compile_options_pb2.CompileOptionsProto,
    platform: str,
    parsed_flags: Sequence[str],
    clear_xla_flags: bool,
) -> None:
  """Clears and/or merges XLA flags into a platform's CompileOptionsProto."""
  if clear_xla_flags:
    compile_options.executable_build_options.comp_envs.Clear()
    compile_options.env_option_overrides.clear()

  if not parsed_flags:
    return

  if platform == 'tpu' and hasattr(compile_options_util, 'tpu_comp_env_pb2'):
    tpu_flags = dict(flag[2:].split('=', 1) for flag in parsed_flags)
    env_override = compile_options_util.parse_tpu_flags(tpu_flags)
    comp_envs = compile_options.executable_build_options.comp_envs
    tpu_env_cls = (
        compile_options_util.tpu_comp_env_pb2.TpuCompilationEnvironment
    )
    if comp_envs.environments:
      env = tpu_env_cls()
      comp_envs.environments[0].Unpack(env)
    else:
      env = tpu_env_cls.FromString(
          compile_options_util.tpu_comp_env.create_default_tpu_comp_env()
      )
    env.MergeFrom(env_override)
    any_proto = compile_options_util.any_pb2.Any()
    any_proto.Pack(env)
    comp_envs.Clear()
    comp_envs.environments.append(any_proto)
  elif platform in ('cuda', 'gpu') and hasattr(
      compile_options_util, '_apply_gpu_compilation_env_options'
  ):
    compile_options_util._apply_gpu_compilation_env_options(  # pylint: disable=protected-access
        compile_options, parsed_flags
    )
  else:
    for flag in parsed_flags:
      key, value = flag[2:].split('=', 1)
      compile_options.env_option_overrides[key].CopyFrom(
          _str_to_option_override(value)
      )


def _find_xla_compile_options_data(
    body: manifest_pb2.StableHloFunctionBody,
) -> tuple[str, manifest_pb2.UnstructuredData] | None:
  """Finds the XLA compile options unstructured data in a StableHLO body."""
  for name, data in sorted(body.supplemental_info.items()):
    if data.mime_type in (
        constants.MIME_TYPE_XLA_COMPILE_OPTIONS,
        constants.MIME_TYPE_ORBAX_XLA_COMPILE_OPTIONS,
    ):
      return name, data
  return None


def _iter_all_stable_hlo_bodies(
    manifest: manifest_pb2.Manifest,
) -> list[tuple[str, manifest_pb2.StableHloFunctionBody]]:
  """Returns all (function_label, StableHloFunctionBody) pairs in manifest."""
  bodies: list[tuple[str, manifest_pb2.StableHloFunctionBody]] = []
  for name, obj in sorted(manifest.objects.items()):
    if obj.HasField('function') and obj.function.body.HasField(
        'stable_hlo_body'
    ):
      bodies.append((name, obj.function.body.stable_hlo_body))
    elif obj.HasField('poly_fn'):
      num_concrete = len(obj.poly_fn.concrete_functions)
      for idx, fn in enumerate(obj.poly_fn.concrete_functions):
        if fn.body.HasField('stable_hlo_body'):
          label = f'{name} #{idx + 1}' if num_concrete > 1 else name
          bodies.append((label, fn.body.stable_hlo_body))
  return bodies


def _get_target_stable_hlo_bodies(
    manifest: manifest_pb2.Manifest, function_name: str | None
) -> list[tuple[str, manifest_pb2.StableHloFunctionBody]]:
  """Resolves target StableHLO function bodies for XLA flag updates."""
  if function_name is None:
    all_bodies = [
        (label, body)
        for label, body in _iter_all_stable_hlo_bodies(manifest)
        if _find_xla_compile_options_data(body) is not None
    ]
    if not all_bodies:
      _bail(
          'No StableHLO functions with XLA compile options found in the model'
          ' manifest.'
      )
    return all_bodies

  base_name = function_name
  concrete_idx: int | None = None
  if ' #' in function_name:
    part_base, idx_str = function_name.rsplit(' #', 1)
    if idx_str.isdigit():
      base_name, concrete_idx = part_base, int(idx_str) - 1
  elif '#' in function_name:
    part_base, idx_str = function_name.rsplit('#', 1)
    if idx_str.isdigit():
      base_name, concrete_idx = part_base, int(idx_str) - 1

  if base_name not in manifest.objects:
    available = ', '.join(sorted(manifest.objects.keys()))
    _bail(
        f'Function "{function_name}" not found in the manifest. Available'
        f' objects: {available}'
    )

  obj = manifest.objects[base_name]
  if obj.HasField('function'):
    if concrete_idx is not None and concrete_idx != 0:
      _bail(
          f'Function "{base_name}" is not polymorphic (got concrete index'
          f' {concrete_idx + 1}).'
      )
    if not obj.function.body.HasField('stable_hlo_body'):
      _bail(f'Function "{function_name}" does not have a StableHLO body.')
    return [(base_name, obj.function.body.stable_hlo_body)]

  if obj.HasField('poly_fn'):
    poly_fn = obj.poly_fn
    if not poly_fn.concrete_functions:
      _bail(f'Polymorphic function "{base_name}" has no concrete functions.')
    num_concrete = len(poly_fn.concrete_functions)
    if concrete_idx is not None:
      if concrete_idx < 0 or concrete_idx >= num_concrete:
        _bail(
            f'Invalid concrete function index {concrete_idx + 1} for'
            f' "{base_name}" (available: 1..{num_concrete}).'
        )
      fn = poly_fn.concrete_functions[concrete_idx]
      if not fn.body.HasField('stable_hlo_body'):
        _bail(f'Function "{function_name}" does not have a StableHLO body.')
      label = (
          f'{base_name} #{concrete_idx + 1}' if num_concrete > 1 else base_name
      )
      return [(label, fn.body.stable_hlo_body)]

    bodies = [
        (
            f'{base_name} #{idx + 1}' if num_concrete > 1 else base_name,
            fn.body.stable_hlo_body,
        )
        for idx, fn in enumerate(poly_fn.concrete_functions)
        if fn.body.HasField('stable_hlo_body')
    ]
    if not bodies:
      _bail(
          f'Polymorphic function "{base_name}" has no StableHLO concrete'
          ' functions.'
      )
    return bodies

  _bail(f'Object "{base_name}" is not a function.')


def _resolve_platform_key(
    opts_map: manifest_pb2.CompileOptionsProtoMap, normalized_platform: str
) -> str:
  """Resolves the platform key in CompileOptionsProtoMap."""
  if normalized_platform in ('cuda', 'gpu'):
    if 'cuda' in opts_map.map:
      return 'cuda'
    if 'gpu' in opts_map.map:
      return 'gpu'
    return 'cuda'
  return normalized_platform


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
    batch_timeout_micros: Annotated[
        int | None,
        typer.Option(
            '--batch_timeout_micros',
            help=(
                'Maximum microseconds to wait before outputting an incomplete'
                ' batch (must be >= 0).'
            ),
        ),
    ] = None,
    allowed_batch_sizes: Annotated[
        str | None,
        typer.Option(
            '--allowed_batch_sizes',
            help=(
                'Comma-separated list of strictly ascending positive batch'
                ' sizes (e.g. "1,2,4,8,16").'
            ),
        ),
    ] = None,
    num_batch_threads: Annotated[
        int | None,
        typer.Option(
            '--num_batch_threads',
            help='Number of batch scheduling threads (must be >= 1).',
        ),
    ] = None,
    max_enqueued_batches: Annotated[
        int | None,
        typer.Option(
            '--max_enqueued_batches',
            help=(
                'Maximum number of batches enqueued before failing fast (must'
                ' be >= 1).'
            ),
        ),
    ] = None,
    disable_large_batch_splitting: Annotated[
        bool | None,
        typer.Option(
            '--disable_large_batch_splitting/--enable_large_batch_splitting',
            help='Whether to disable splitting large input batches.',
        ),
    ] = None,
    pad_variable_length_inputs: Annotated[
        bool | None,
        typer.Option(
            '--pad_variable_length_inputs/--no_pad_variable_length_inputs',
            help=(
                'Whether to dynamically pad variable-length non-batch'
                ' dimensions to the maximum size across requests within each'
                ' batch. Requires disable_large_batch_splitting=True.'
            ),
        ),
    ] = None,
    pipeline_name: Annotated[
        str | None,
        typer.Option(
            '--pipeline_name',
            help=(
                'Optional name of a specific pipeline to update. If omitted,'
                ' updates all pipelines.'
            ),
        ),
    ] = None,
    platform: Annotated[
        str,
        typer.Option(
            '--platform',
            help=(
                'Target compilation platform for XLA flags (e.g. "tpu",'
                ' "cuda" or "gpu", "cpu", "rocm").'
            ),
        ),
    ] = 'tpu',
    xla_flags: Annotated[
        str | None,
        typer.Option(
            '--xla_flags',
            help=(
                'Space- or comma-separated XLA compiler flags in'
                ' --flag_name=flag_value format.'
            ),
        ),
    ] = None,
    clear_xla_flags: Annotated[
        bool,
        typer.Option(
            '--clear_xla_flags',
            help=(
                'Clear existing persisted XLA flags for the target platform'
                ' before applying any new --xla_flags.'
            ),
        ),
    ] = False,
    function_name: Annotated[
        str | None,
        typer.Option(
            '--function_name',
            help=(
                'Optional name of a specific StableHLO function to update XLA'
                ' flags for (e.g. "serving_default"). If omitted, updates all'
                ' StableHLO functions.'
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
  """Updates batching configs and XLA compilation flags in an exported model."""
  if max_batch_size is not None and max_batch_size <= 0:
    _bail(f'max_batch_size must be positive, got {max_batch_size}')

  if batch_timeout_micros is not None and batch_timeout_micros < 0:
    _bail(
        f'batch_timeout_micros must be non-negative, got {batch_timeout_micros}'
    )

  if num_batch_threads is not None and num_batch_threads <= 0:
    _bail(f'num_batch_threads must be >= 1, got {num_batch_threads}')

  if max_enqueued_batches is not None and max_enqueued_batches <= 0:
    _bail(f'max_enqueued_batches must be >= 1, got {max_enqueued_batches}')

  parsed_allowed_sizes: list[int] | None = None
  if allowed_batch_sizes is not None:
    parsed_allowed_sizes = _parse_allowed_batch_sizes(allowed_batch_sizes)

  field_updates: list[tuple[str, Any]] = [
      (name, val)
      for name, val in (
          ('max_batch_size', max_batch_size),
          ('batch_timeout_micros', batch_timeout_micros),
          ('allowed_batch_sizes', parsed_allowed_sizes),
          ('num_batch_threads', num_batch_threads),
          ('max_enqueued_batches', max_enqueued_batches),
          ('disable_large_batch_splitting', disable_large_batch_splitting),
          ('pad_variable_length_inputs', pad_variable_length_inputs),
      )
      if val is not None
  ]
  has_xla_updates = (xla_flags is not None) or clear_xla_flags

  if not field_updates and not has_xla_updates:
    _bail(
        'No update options specified. Provide at least one batching'
        ' configuration flag (e.g. --max_batch_size, --batch_timeout_micros,'
        ' --allowed_batch_sizes, --num_batch_threads,'
        ' --max_enqueued_batches, --disable_large_batch_splitting,'
        ' --pad_variable_length_inputs) or XLA compilation flag'
        ' (--xla_flags, --clear_xla_flags).'
    )

  normalized_platform = 'tpu'
  parsed_xla_flags: list[str] = []
  if has_xla_updates:
    normalized_platform = _normalize_platform(platform)
    if xla_flags is not None:
      parsed_xla_flags = _parse_xla_flags_list(xla_flags)
    _validate_xla_flags_for_platform(normalized_platform, parsed_xla_flags)

  try:
    manifest = persistence_lib.load(model_base_dir)
  except Exception as e:  # pylint: disable=broad-except
    _bail(f'Failed to load model at "{model_base_dir}": {e}')

  changes: dict[str, Any] = {}
  updated_fields: list[str] = []
  external_writes: dict[str, message.Message] = {}
  manifest_dirty = False

  if field_updates:
    pipelines_entry = _find_pipelines_data(manifest)
    if not pipelines_entry:
      _bail('No orchestration pipelines found in the model manifest.')

    unused_name, pipelines_data = pipelines_entry
    pipelines = orchestration_pb2.Pipelines()
    _read_proto_from_data(model_base_dir, pipelines_data, pipelines)

    if not pipelines.name_to_pipeline:
      _bail('No named pipelines found inside pipelines proto.')

    if pipeline_name is not None:
      if pipeline_name not in pipelines.name_to_pipeline:
        available = ', '.join(sorted(pipelines.name_to_pipeline.keys()))
        _bail(
            f'Pipeline "{pipeline_name}" not found. Available pipelines:'
            f' {available}'
        )
      target_pipelines = [
          (pipeline_name, pipelines.name_to_pipeline[pipeline_name])
      ]
    else:
      target_pipelines = sorted(pipelines.name_to_pipeline.items())

    # Cross-field validation across effective pipeline batch_options
    for p_name, pipeline in target_pipelines:
      bo = pipeline.batch_options
      effective_max = (
          max_batch_size if max_batch_size is not None else bo.max_batch_size
      )
      effective_allowed = (
          parsed_allowed_sizes
          if parsed_allowed_sizes is not None
          else list(bo.allowed_batch_sizes)
      )
      effective_disable_splitting = (
          disable_large_batch_splitting
          if disable_large_batch_splitting is not None
          else bo.disable_large_batch_splitting
      )
      effective_pad_variable = (
          pad_variable_length_inputs
          if pad_variable_length_inputs is not None
          else bo.pad_variable_length_inputs
      )

      if effective_allowed:
        if (
            effective_disable_splitting
            and effective_allowed[-1] != effective_max
        ):
          _bail(
              f'Pipeline "{p_name}": max_batch_size ({effective_max}) must'
              ' equal the largest allowed_batch_size'
              f' ({effective_allowed[-1]}) when'
              ' disable_large_batch_splitting is True.'
          )
        if (
            not effective_disable_splitting
            and effective_allowed[-1] > effective_max
        ):
          _bail(
              f'Pipeline "{p_name}": largest allowed_batch_size'
              f' ({effective_allowed[-1]}) cannot exceed max_batch_size'
              f' ({effective_max}).'
          )

      if effective_pad_variable and not effective_disable_splitting:
        _bail(
            f'Pipeline "{p_name}": disable_large_batch_splitting must be True'
            ' when pad_variable_length_inputs is True.'
        )

    updated_fields.extend(field_name for field_name, _ in field_updates)
    for p_name, pipeline in target_pipelines:
      bo = pipeline.batch_options
      for field_name, new_val in field_updates:
        if field_name == 'allowed_batch_sizes':
          prev = list(bo.allowed_batch_sizes)
          bo.allowed_batch_sizes[:] = new_val
        else:
          prev = getattr(bo, field_name)
          setattr(bo, field_name, new_val)
        changes[f'{p_name}.{field_name}'] = {
            'previous': prev,
            'new': new_val,
        }

    if pipelines_data.HasField('file_system_location'):
      external_writes[pipelines_data.file_system_location.string_path] = (
          pipelines
      )
    elif pipelines_data.HasField('inlined_bytes'):
      pipelines_data.inlined_bytes = pipelines.SerializeToString()
      manifest_dirty = True

  if has_xla_updates:
    target_bodies = _get_target_stable_hlo_bodies(manifest, function_name)
    target_xla_entries: list[
        tuple[
            str,
            manifest_pb2.UnstructuredData,
            manifest_pb2.CompileOptionsProtoMap,
        ]
    ] = []
    for fn_label, body in target_bodies:
      opts_entry = _find_xla_compile_options_data(body)
      if not opts_entry:
        _bail(f'Function "{fn_label}" does not have XLA compile options.')
      unused_name, opts_data = opts_entry
      opts_map = manifest_pb2.CompileOptionsProtoMap()
      _read_proto_from_data(model_base_dir, opts_data, opts_map)
      target_xla_entries.append((fn_label, opts_data, opts_map))

    explicit_flag_keys = {
        flag[2:].split('=', 1)[0] for flag in parsed_xla_flags
    }
    for fn_label, opts_data, opts_map in target_xla_entries:
      platform_key = _resolve_platform_key(opts_map, normalized_platform)
      compile_options = opts_map.map[platform_key]
      prev_flags_str = _format_xla_flags(compile_options)
      _apply_xla_flags_to_compile_options(
          compile_options,
          normalized_platform,
          parsed_xla_flags,
          clear_xla_flags,
      )
      new_flags_str = _format_xla_flags(
          compile_options, explicit_keys=explicit_flag_keys
      )
      changes[f'{fn_label}.xla_flags.{platform_key}'] = {
          'previous': prev_flags_str,
          'new': new_flags_str,
      }

      if opts_data.HasField('file_system_location'):
        external_writes[opts_data.file_system_location.string_path] = opts_map
      elif opts_data.HasField('inlined_bytes'):
        opts_data.inlined_bytes = opts_map.SerializeToString()
        manifest_dirty = True

    updated_fields.append(f'xla_flags ({normalized_platform})')

  target_dir = model_base_dir
  if output_dir:
    output_dir = os.path.abspath(output_dir)
    if os.path.abspath(model_base_dir) == output_dir:
      _bail('--output_dir cannot be the same as model_base_dir.')
    if file_utils.exists(output_dir):
      _bail(f'Output directory "{output_dir}" already exists.')
    file_utils.copytree(model_base_dir, output_dir)
    target_dir = output_dir

  for rel_path, proto_msg in sorted(external_writes.items()):
    abs_path = os.path.join(target_dir, rel_path)
    _atomic_write_proto(abs_path, proto_msg, backup=(not output_dir))

  if manifest_dirty:
    manifest_file_path = os.path.join(
        target_dir, core_constants.MANIFEST_FILE_PATH
    )
    _atomic_write_proto(manifest_file_path, manifest, backup=(not output_dir))

  # Record change in history.txtpb
  fields_summary = ', '.join(updated_fields)
  record_history_entry(
      target_dir,
      action='UPDATE',
      description=f'Updated {fields_summary} via CLI.',
      changes=changes,
  )

  # Output confirmation table
  table = rich.table.Table(title='Model Update Applied', padding=(0, 1))
  table.add_column('Target / Parameter', style='cyan')
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


def _get_pipeline_batch_options(
    model_base_dir: str,
    manifest: manifest_pb2.Manifest | None = None,
) -> dict[str, Any]:
  """Reads current batch_options fields for each pipeline in the model."""
  try:
    if manifest is None:
      manifest = persistence_lib.load(model_base_dir)
    pipelines_entry = _find_pipelines_data(manifest)
    if not pipelines_entry:
      return {}
    unused_name, pipelines_data = pipelines_entry
    pipelines = orchestration_pb2.Pipelines()
    _read_proto_from_data(model_base_dir, pipelines_data, pipelines)
    options: dict[str, Any] = {}
    for name, pipeline in sorted(pipelines.name_to_pipeline.items()):
      bo = pipeline.batch_options
      options[f'{name}.max_batch_size'] = bo.max_batch_size
      options[f'{name}.batch_timeout_micros'] = bo.batch_timeout_micros
      options[f'{name}.allowed_batch_sizes'] = list(bo.allowed_batch_sizes)
      options[f'{name}.num_batch_threads'] = bo.num_batch_threads
      options[f'{name}.max_enqueued_batches'] = bo.max_enqueued_batches
      options[f'{name}.disable_large_batch_splitting'] = (
          bo.disable_large_batch_splitting
      )
      options[f'{name}.pad_variable_length_inputs'] = (
          bo.pad_variable_length_inputs
      )
    return options
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

  before_options = _get_pipeline_batch_options(
      model_base_dir, manifest=manifest
  )

  tmp_path = f'{target_path}.tmp'
  try:
    file_utils.copy(backup_path, tmp_path, overwrite=True)
    file_utils.rename(tmp_path, target_path, overwrite=True)
  finally:
    if file_utils.exists(tmp_path):
      file_utils.remove(tmp_path)

  after_options = _get_pipeline_batch_options(model_base_dir)

  changes: dict[str, Any] = {}
  for key, new_val in sorted(after_options.items()):
    prev_val = before_options.get(key)
    if prev_val is not None and prev_val != new_val:
      changes[key] = {
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
