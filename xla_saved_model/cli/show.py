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

"""NPT CLI `show` command."""

import itertools
import os
from typing import List, NoReturn

from google.protobuf import message
from google.protobuf import text_format
import jax.extend as jex
from jax.interpreters import mlir
import rich
import rich.table
import typer
from typing_extensions import Annotated

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model.protos import tf_concrete_function_handle_pb2
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import persistence_lib
from xla_saved_model.cli import constants
from xla_saved_model.cli import printer


def _bail(msg: str) -> NoReturn:
  typer.secho(f'Error: {msg}', fg=typer.colors.RED, err=True)
  raise typer.Exit(1)


def _get_stable_hlo_body(
    manifest: manifest_pb2.Manifest, fn_name: str
) -> manifest_pb2.StableHloFunctionBody:
  """Returns the StableHLO function body for the given function."""
  base_name = fn_name
  concrete_idx = 0
  if ' #' in fn_name:
    base_name, idx_str = fn_name.rsplit(' #', 1)
    if idx_str.isdigit():
      concrete_idx = int(idx_str) - 1
  elif '#' in fn_name:
    base_name, idx_str = fn_name.rsplit('#', 1)
    if idx_str.isdigit():
      concrete_idx = int(idx_str) - 1

  if base_name not in manifest.objects:
    _bail(f'Function `{fn_name}` not found in the manifest')

  obj = manifest.objects[base_name]
  if obj.HasField('function'):
    fn = obj.function
  elif obj.HasField('poly_fn'):
    poly_fn = obj.poly_fn
    if not poly_fn.concrete_functions:
      _bail(f'Polymorphic function `{base_name}` has no concrete functions')
    if concrete_idx < 0 or concrete_idx >= len(poly_fn.concrete_functions):
      _bail(
          f'Invalid concrete function index {concrete_idx + 1} for'
          f' `{base_name}` (available: 1..{len(poly_fn.concrete_functions)})'
      )
    fn = poly_fn.concrete_functions[concrete_idx]
  else:
    _bail(f'Object `{base_name}` is not a function')

  if not fn.body.HasField('stable_hlo_body'):
    _bail(f'Function `{fn_name}` does not have a StableHLO body')
  return fn.body.stable_hlo_body


def _show_xla_compile_options(
    model_base_dir: str,
    manifest: manifest_pb2.Manifest,
    fn_name: str,
) -> None:
  """Shows XLA compile options for a function."""
  body = _get_stable_hlo_body(manifest, fn_name)

  found_data = False
  for name, data in sorted(body.supplemental_info.items()):
    if data.mime_type in (
        constants.MIME_TYPE_XLA_COMPILE_OPTIONS,
        constants.MIME_TYPE_ORBAX_XLA_COMPILE_OPTIONS,
    ):
      found_data = True
      opts = manifest_pb2.CompileOptionsProtoMap()
      _read_proto_from_data(model_base_dir, data, opts)

      table = rich.table.Table(fn_name, name, padding=(0, 1, 1, 1))
      for key, value in sorted(opts.map.items()):
        table.add_row(key, printer.xla_compile_options(value))
      rich.print(table)

  if not found_data:
    _bail(f'Function `{fn_name}` does not have XLA compile options')


def _save_stablehlo(
    manifest: manifest_pb2.Manifest, fn_name: str, output_file: str
) -> None:
  """Saves the StableHLO function body to the given file."""
  body = _get_stable_hlo_body(manifest, fn_name)
  with mlir.make_ir_context():
    stablehlo_module = jex.mlir.deserialize_portable_artifact(
        body.stable_hlo.inlined_bytes
    )
    module_str = str(stablehlo_module)
  with file_utils.open_file(output_file, 'w') as f:
    f.write(module_str)
  rich.print(f'Saved StableHLO function to `{output_file}`')


def show(
    model_base_dir: Annotated[
        str, typer.Argument(help='Base directory of the model')
    ],
    details: Annotated[
        List[str],
        typer.Option(
            '--details',
            '-d',
            help=(
                'Show the detailed information about an object or a'
                ' supplemental'
            ),
        ),
    ] = [],
    verbose: Annotated[
        bool,
        typer.Option(
            '--verbose',
            '-v',
            help='Show more verbose information',
        ),
    ] = False,
    savehlo: Annotated[
        str,
        typer.Option(
            '--savehlo',
            help='Save the StableHLO function body to the given file',
        ),
    ] = '',
    xla_compile_options: Annotated[
        bool,
        typer.Option(
            f'--{constants.MIME_TYPE_FLAGS[constants.MIME_TYPE_XLA_COMPILE_OPTIONS]}',
            help='Show XLA compile options for a function',
        ),
    ] = False,
) -> None:
  """Shows the summary of an XLA SavedModel."""
  rich.print('Loading XLA SavedModel... ')
  manifest: manifest_pb2.Manifest
  try:
    manifest = persistence_lib.load(model_base_dir)
  except IOError as e:
    _bail(f'Failed to load XLA SavedModel: {e}')

  rich.print(
      f'Manifest: {manifest.ByteSize()//1024} KiB,'
      f' {len(manifest.objects)} objects,'
      f' {len(manifest.supplemental_info)} supplementals'
  )

  # Support both --details=a --details=b and --details=a,b.
  details = tuple(  # pyrefly: ignore[bad-assignment]
      set(itertools.chain.from_iterable([d.split(',') for d in details]))
  )

  if details and savehlo:
    if len(details) > 1:
      _bail('Please specify only one function to save')
    _save_stablehlo(manifest, details[0], savehlo)
    return

  if details and xla_compile_options:
    if len(details) > 1:
      _bail('Please specify only one function to show XLA compile options')
    _show_xla_compile_options(model_base_dir, manifest, details[0])
    return

  if details:
    rich.print(f"Showing details for {', '.join(details)}...")
    for obj in details:
      _show_details(model_base_dir, manifest, obj, verbose)
    return

  if manifest.objects:
    table = rich.table.Table('Object', 'Type', padding=(0, 1, 1, 1))
    for name, obj in sorted(manifest.objects.items()):
      table.add_row(name, printer.top_level_object(obj))
    rich.print(table)

  if manifest.supplemental_info:
    table = rich.table.Table('Supplemental', 'Type', padding=(0, 1, 1, 1))
    for name, supp in sorted(manifest.supplemental_info.items()):
      table.add_row(name, printer.unstructured_data(supp))
    rich.print(table)

  if manifest.device_assignment_by_coords.devices:
    table = rich.table.Table('Device assignment')
    table.add_row(
        text_format.MessageToString(
            manifest.device_assignment_by_coords,
            use_short_repeated_primitives=True,
        )
    )
    rich.print(table)


def _show_details(
    model_base_dir: str,
    manifest: manifest_pb2.Manifest,
    details: str,
    verbose: bool,
) -> None:
  """Shows details for an object or a supplemental."""
  if details in manifest.objects:
    _show_object_details(
        model_base_dir, details, manifest.objects[details], verbose
    )

  if details in manifest.supplemental_info:
    _show_supplemental_details(
        model_base_dir, details, manifest.supplemental_info[details], verbose
    )

  if (
      details not in manifest.objects
      and details not in manifest.supplemental_info
  ):
    rich.print(f'Unknown object or supplemental: {details}')


def _show_object_details(
    model_base_dir: str,
    name: str,
    obj: manifest_pb2.TopLevelObject,
    verbose: bool,
) -> None:
  """Shows details for an object."""
  if obj.HasField('function'):
    _show_function_details(model_base_dir, name, obj.function, verbose)
  elif obj.HasField('value'):
    _show_value_details(name, obj.value)
  elif obj.HasField('poly_fn'):
    for i, fn in enumerate(obj.poly_fn.concrete_functions):
      _show_function_details(model_base_dir, f'{name} #{i+1}', fn, verbose)
  else:
    rich.print('Unknown object type')


def _show_function_details(
    model_base_dir: str, name: str, fn: manifest_pb2.Function, verbose: bool
) -> None:
  """Shows the function details (StableHLO or TF ConcreteFunction handle)."""

  table = rich.table.Table('Function', name, padding=(0, 1, 1, 1))

  if fn.body.HasField('stable_hlo_body'):
    body = fn.body.stable_hlo_body
    table.add_row('Type', 'StableHLO')
    table.add_row(
        'Visibility',
        manifest_pb2.Visibility.Name(fn.visibility).lower(),
    )
    hint = (
        ' (`[green]--savehlo <file>[/green]` to save locally)'
        if fn.body.HasField('stable_hlo_body')
        else ''
    )
    table.add_row('Size', f'{len(body.stable_hlo.inlined_bytes)} bytes' + hint)

    table.add_row(
        'Input signature',
        printer.collapse(
            printer.xsm_type(fn.signature.input), verbose_mode=verbose
        ),
    )
    table.add_row(
        'Output signature',
        printer.collapse(
            printer.xsm_type(fn.signature.output), verbose_mode=verbose
        ),
    )
    table.add_row('Lowering platforms', ','.join(body.lowering_platforms))
    table.add_row(
        'Calling convention version', f'{body.calling_convention_version}'
    )
    if fn.data_names:
      if verbose:
        table.add_row(
            'Data names',
            f'{len(fn.data_names)} total: \n'
            + printer.pad('\n'.join(fn.data_names), ' - '),
        )
      else:
        table.add_row(
            'Data names',
            f'{len(fn.data_names)} total (`[green]--verbose[/green]` to'
            ' expand)',
        )
    if fn.gradient_function_name:
      table.add_row('Gradient function name', f'{fn.gradient_function_name}')
    if body.supplemental_info:
      for name, supp in sorted(body.supplemental_info.items()):
        table.add_row(name, printer.unstructured_data(supp))
  elif fn.body.HasField('other'):
    match fn.body.other.mime_type:
      case (
          constants.MIME_TYPE_TF_CONCRETE_FUNCTION
          | constants.MIME_TYPE_ORBAX_TF_CONCRETE_FUNCTION
      ):
        table.add_row('Type', 'TF ConcreteFunction handle')
        handle = tf_concrete_function_handle_pb2.TfConcreteFunctionHandle()
        _read_proto_from_data(model_base_dir, fn.body.other, handle)
        table.add_row('Name', handle.fn_name)
        table.add_row('Inputs', '\n'.join(handle.input_names))
        table.add_row('Outputs', '\n'.join(handle.output_names))
      case _:
        table.add_row('Type', printer.unstructured_data(fn.body.other))
  else:
    table.add_row('Type', 'Unknown function type')

  rich.print(table)


def _show_value_details(name: str, value: manifest_pb2.Value) -> None:
  """Shows details for a value."""
  if value.HasField('external'):
    table = rich.table.Table('External value', name, padding=(0, 1, 1, 1))
    table.add_row(
        'Loader',
        manifest_pb2.LoaderType.Name(value.external.loader_type),
    )
    table.add_row('Data', printer.unstructured_data(value.external.data))
    rich.print(table)
  elif value.HasField('tuple'):
    table = rich.table.Table('Tuple', name, padding=(0, 1, 1, 1))
    table.add_row('Elements', '\n'.join(value.tuple.elements))
    rich.print(table)
  elif value.HasField('named_tuple'):
    table = rich.table.Table('Named tuple', name, padding=(0, 1, 1, 1))
    for pair in value.named_tuple.elements:
      table.add_row(pair.name, pair.value)
    rich.print(table)
  else:
    rich.print('Unknown value type')


def _show_supplemental_details(
    model_base_dir: str,
    name: str,
    data: manifest_pb2.UnstructuredData,
    verbose: bool,
) -> None:
  match data.mime_type:
    case constants.MIME_TYPE_PIPELINES | constants.MIME_TYPE_ORBAX_PIPELINES:
      _show_pipelines(model_base_dir, data, verbose)
    case _:
      table = rich.table.Table('Suppemental', 'Type', padding=(0, 1, 1, 1))
      table.add_row(name, printer.unstructured_data(data))
      rich.print(table)


def _read_proto_from_data(
    model_base_dir: str,
    data: manifest_pb2.UnstructuredData,
    proto: message.Message,
) -> None:
  """Reads a proto from the given UnstructuredData."""
  if data.HasField('file_system_location'):
    if os.path.isabs(data.file_system_location.string_path):
      _bail(
          'File system location must be relative, got'
          f' "{data.file_system_location.string_path}"'
      )
    with file_utils.open_file(
        os.path.join(model_base_dir, data.file_system_location.string_path),
        'rb',
    ) as f:
      proto.ParseFromString(f.read())
  elif data.HasField('inlined_bytes'):
    proto.ParseFromString(data.inlined_bytes)
  elif data.HasField('inlined_string'):
    proto.ParseFromString(data.inlined_string.encode('utf-8'))
  else:
    _bail(f'Unsupported data type: {data.mime_type}')


def _show_pipelines(
    model_base_dir: str,
    data: manifest_pb2.UnstructuredData,
    verbose: bool,
) -> None:
  """Shows the summary of the pipelines referenced in the manifest."""
  pipelines = orchestration_pb2.Pipelines()
  _read_proto_from_data(model_base_dir, data, pipelines)

  for name, pipeline in sorted(pipelines.name_to_pipeline.items()):
    table = rich.table.Table('Pipeline', name, padding=(0, 1, 1, 1))
    table.add_row('Input', printer.xsm_type(pipeline.signature.input))
    table.add_row('Output', printer.xsm_type(pipeline.signature.output))

    if pipeline.named_signature.inputs:
      input_names = [
          f'{input_type.name}: {printer.tensor_type(input_type.tensor_type)}'
          for input_type in pipeline.named_signature.inputs.named_tensor_types
      ]
      table.add_row('Input names', '\n'.join(input_names))

    if pipeline.named_signature.outputs:
      output_names = [
          f'{output_type.name}: {printer.tensor_type(output_type.tensor_type)}'
          for output_type in pipeline.named_signature.outputs.named_tensor_types
      ]
      table.add_row('Output names', '\n'.join(output_names))

    if pipeline.components:
      pre_processor_names = [
          c.function_name
          for c in pipeline.components
          if c.role == orchestration_pb2.ROLE_PREPROCESSOR
      ]
      model_fns = [
          f'name: {c.function_name}\nweights: {c.weights_name}'
          for c in pipeline.components
          if c.role == orchestration_pb2.ROLE_MODEL
      ]
      post_processor_names = [
          c.function_name
          for c in pipeline.components
          if c.role == orchestration_pb2.ROLE_POSTPROCESSOR
      ]
    else:
      pre_processor_names = pipeline.pre_processor_names
      model_fns = [
          f'name: {fn.model_function_name}\nweights: {fn.weights_name}'
          for fn in pipeline.model_functions
      ]
      post_processor_names = pipeline.post_processor_names

    table.add_row('Pre-processors', '\n'.join(pre_processor_names))
    table.add_row('Model functions', '\n\n'.join(model_fns))
    table.add_row('Post-processors', '\n'.join(post_processor_names))

    table.add_row(
        'Batch options',
        printer.collapse(
            printer.batch_options(pipeline.batch_options), verbose_mode=verbose
        ),
    )

    rich.print(table)
