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

"""Tests for XSM CLI `show` command."""

import importlib
import io
import sys
from typing import Any
from unittest import mock

from absl.testing import absltest
import rich
import rich.console
import typer
from typer.testing import CliRunner

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model.protos import type_pb2
from xla_saved_model.cli import constants
from xla_saved_model.cli import show

try:
  from xla_serving_data.processor.proto import transform_config_pb2  # pylint: disable=g-import-not-at-top
  from xla_serving_data.resource_manager.proto import resource_manager_pb2  # pylint: disable=g-import-not-at-top
except ImportError:
  transform_config_pb2 = None
  resource_manager_pb2 = None

_PROTO_PKG = 'xla_serving_data.processor.proto'


def _make_composite_function(config: Any = None) -> manifest_pb2.Function:
  fn = manifest_pb2.Function()
  x_type = fn.signature.input.dict.string_to_type['x']
  x_type.leaf.tensor_type.dtype = type_pb2.DType.f32
  x_type.leaf.tensor_type.shape.shape_with_known_rank.dimension_sizes.add(
      size=4
  )
  fn.signature.output.leaf.tensor_type.dtype = type_pb2.DType.si32
  fn.body.other.mime_type = constants.MIME_TYPE_COMPOSITE_PROCESSOR
  if config is not None:
    fn.body.other.inlined_bytes = config.SerializeToString()
  return fn


def _sample_composite_config() -> Any:
  assert transform_config_pb2 is not None
  assert resource_manager_pb2 is not None
  config = transform_config_pb2.CompositeProcessorConfig(
      custom_asset_path='assets/custom',
  )
  transform_0 = config.transforms.add(
      name='AddTransform_0',
      op_name='Add',
      device=transform_config_pb2.DEVICE_CPU,
  )
  transform_0.input_edges.add(
      type=transform_config_pb2.EdgeProto.GRAPH_INPUT,
      source_index=0,
  )
  transform_0.input_edges.add(
      type=transform_config_pb2.EdgeProto.CONSTANT,
  ).constant_value.type_url = 'type.googleapis.com/example.ConstCfg'

  transform_1 = config.transforms.add(
      name='ScaleTransform_1',
      op_name='Scale',
      resource_ids=[7, 99],
  )
  transform_1.input_edges.add(
      type=transform_config_pb2.EdgeProto.NODE_OUTPUT,
      source_index=0,
      port_index=0,
  )
  transform_1.detail.type_url = 'type.googleapis.com/example.ScaleCfg'

  config.output_edges.add(
      type=transform_config_pb2.EdgeProto.NODE_OUTPUT,
      source_index=1,
      port_index=0,
  )

  named_resource = config.resources.add(
      name='scale_table',
      class_name='ScaleResource',
      module_name='neptune.data.scale',
      id=7,
  )
  named_resource.assets.append(
      resource_manager_pb2.AssetProto(name='factors', path='assets/factors.npy')
  )
  named_resource.assets.append(
      resource_manager_pb2.AssetProto(name='inline_cfg', content='abc')
  )

  unnamed_resource = config.resources.add(
      name='99',
      class_name='BiasResource',
      id=99,
  )
  unnamed_resource.assets.append(
      resource_manager_pb2.AssetProto(name='bias', path='assets/bias.txt')
  )
  return config


def _render_function_details(
    fn: manifest_pb2.Function, verbose: bool = False
) -> str:
  buf = io.StringIO()
  console = rich.console.Console(file=buf, width=120, force_terminal=False)
  with mock.patch.object(rich, 'print', side_effect=console.print):
    show._show_function_details('/base/dir', 'transform_inputs', fn, verbose)
  return buf.getvalue()


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

  @absltest.skipIf(
      transform_config_pb2 is None, 'xla-serving-data is not installed'
  )
  def test_show_composite_processor_details_default(self):
    fn = _make_composite_function(_sample_composite_config())
    output = _render_function_details(fn, verbose=False)

    self.assertIn('CompositeProcessor', output)
    self.assertIn('x: [4, f32]', output)
    self.assertIn('si32', output)
    self.assertIn('2 total:', output)
    self.assertIn('AddTransform_0 (Add)', output)
    self.assertIn('ScaleTransform_1 (Scale)', output)
    self.assertIn('2 total (`--verbose` to expand)', output)
    self.assertIn('assets/custom', output)
    # Verbose-only fields should not appear in default output.
    self.assertNotIn('inputs:', output)
    self.assertNotIn('Outputs', output)
    self.assertNotIn('scale_table', output)
    self.assertNotIn('ScaleResource', output)
    self.assertNotIn('assets/factors.npy', output)
    self.assertNotIn('neptune.data.scale', output)
    self.assertNotIn('ScaleCfg', output)

  @absltest.skipIf(
      transform_config_pb2 is None, 'xla-serving-data is not installed'
  )
  def test_show_composite_processor_details_verbose(self):
    fn = _make_composite_function(_sample_composite_config())
    output = _render_function_details(fn, verbose=True)

    self.assertIn(
        'inputs: graph.input[0]:x, const.ConstCfg',
        output,
    )
    self.assertIn('device: DEVICE_CPU', output)
    self.assertIn('inputs: AddTransform_0.output[0]', output)
    self.assertIn('resources: scale_table, resource[1]', output)
    self.assertIn('extra_args: ScaleCfg', output)
    self.assertIn('Outputs', output)
    self.assertIn('- ScaleTransform_1.output[0]', output)
    self.assertIn('- scale_table (ScaleResource)', output)
    self.assertIn('- resource[1] (BiasResource)', output)
    self.assertIn('module: neptune.data.scale', output)
    self.assertIn('factors: assets/factors.npy', output)
    self.assertIn('inline_cfg: <inline, 3 characters>', output)
    self.assertIn('bias: assets/bias.txt', output)

  @absltest.skipIf(
      transform_config_pb2 is None, 'xla-serving-data is not installed'
  )
  def test_show_composite_processor_details_empty(self):
    assert transform_config_pb2 is not None
    fn = _make_composite_function(
        transform_config_pb2.CompositeProcessorConfig()
    )
    output = _render_function_details(fn, verbose=True)

    self.assertIn('CompositeProcessor', output)
    self.assertNotIn('Transforms', output)
    self.assertNotIn('Outputs', output)
    self.assertNotIn('Resources', output)
    self.assertNotIn('Custom asset path', output)

  @absltest.skipIf(
      transform_config_pb2 is None, 'xla-serving-data is not installed'
  )
  def test_show_composite_processor_unnamed_transform_uses_node_index(self):
    assert transform_config_pb2 is not None
    config = transform_config_pb2.CompositeProcessorConfig()
    config.transforms.add(op_name='Add')
    transform_1 = config.transforms.add(name='Scale_1', op_name='Scale')
    transform_1.input_edges.add(
        type=transform_config_pb2.EdgeProto.NODE_OUTPUT, source_index=0
    )
    config.output_edges.add(
        type=transform_config_pb2.EdgeProto.NODE_OUTPUT, source_index=0
    )
    output = _render_function_details(
        _make_composite_function(config), verbose=True
    )

    self.assertIn('- node[0] (Add)', output)
    self.assertIn('inputs: node[0].output[0]', output)
    self.assertIn('- node[0].output[0]', output)
    self.assertNotIn('(unnamed)', output)

  @absltest.skipIf(
      transform_config_pb2 is None, 'xla-serving-data is not installed'
  )
  def test_show_composite_processor_escapes_markup_in_names(self):
    assert transform_config_pb2 is not None
    assert resource_manager_pb2 is not None
    config = transform_config_pb2.CompositeProcessorConfig(
        custom_asset_path='assets[/y]'
    )
    config.transforms.add(name='feat[bold]_v2', op_name='Add', resource_ids=[1])
    resource = config.resources.add(name='tbl[/x]', class_name='Res', id=1)
    resource.assets.append(
        resource_manager_pb2.AssetProto(name='a[red]', path='data[/x].bin')
    )
    output = _render_function_details(
        _make_composite_function(config), verbose=True
    )

    self.assertIn('feat[bold]_v2 (Add)', output)
    self.assertIn('resources: tbl[/x]', output)
    self.assertIn('- tbl[/x] (Res)', output)
    self.assertIn('a[red]: data[/x].bin', output)
    self.assertIn('assets[/y]', output)

  @absltest.skipIf(
      transform_config_pb2 is None, 'xla-serving-data is not installed'
  )
  def test_transform_config_pb2_is_available(self):
    self.assertIsNotNone(show.transform_config_pb2)

  def test_show_works_without_xla_serving_data(self):
    config = (
        _sample_composite_config() if transform_config_pb2 is not None else None
    )
    fn = _make_composite_function(config)
    try:
      with mock.patch.dict(sys.modules, {_PROTO_PKG: None}):
        importlib.reload(show)
        self.assertIsNone(show.transform_config_pb2)
        output = _render_function_details(fn, verbose=False)
        self.assertIn('CompositeProcessor', output)
        self.assertIn('[4, f32]', output)
        self.assertIn('si32', output)
        self.assertIn(show._COMPOSITE_UNAVAILABLE_HINT, output)
        self.assertNotIn('AddTransform_0', output)
    finally:
      importlib.reload(show)
      self.assertEqual(show.transform_config_pb2, transform_config_pb2)

  def test_show_stablehlo_function_renders_weights_name(self):
    fn = manifest_pb2.Function(weights_name='model_weights')
    fn.body.stable_hlo_body.calling_convention_version = 10
    fn.body.stable_hlo_body.lowering_platforms.append('cpu')

    output = _render_function_details(fn, verbose=False)

    self.assertIn('Weights name', output)
    self.assertIn('model_weights', output)

  def test_show_stablehlo_function_omits_unset_weights_name(self):
    fn = manifest_pb2.Function()
    fn.body.stable_hlo_body.calling_convention_version = 10
    fn.body.stable_hlo_body.lowering_platforms.append('cpu')

    output = _render_function_details(fn, verbose=False)

    self.assertNotIn('Weights name', output)

  def test_show_pipelines_resolves_weights_from_manifest_with_fallback(self):
    manifest = manifest_pb2.Manifest()
    manifest.objects['manifest_fn'].function.weights_name = 'manifest_weights'
    poly = manifest.objects['poly_fn'].poly_fn
    poly.concrete_functions.add(weights_name='poly_weights')
    poly.concrete_functions.add(weights_name='poly_weights')

    pipelines = orchestration_pb2.Pipelines()
    p1 = pipelines.name_to_pipeline['p_manifest']
    p1.components.add(
        function_name='manifest_fn',
        role=orchestration_pb2.ROLE_MODEL,
    )
    p_poly = pipelines.name_to_pipeline['p_poly']
    p_poly.components.add(
        function_name='poly_fn',
        role=orchestration_pb2.ROLE_MODEL,
    )
    p2 = pipelines.name_to_pipeline['p_legacy']
    p2.components.add(
        function_name='legacy_fn',
        role=orchestration_pb2.ROLE_MODEL,
        weights_name='legacy_weights',
    )
    p3 = pipelines.name_to_pipeline['p_legacy_model_fn']
    p3.model_functions.add(
        model_function_name='legacy_bound_fn',
        weights_name='legacy_bound_weights',
    )

    data = manifest_pb2.UnstructuredData(
        inlined_bytes=pipelines.SerializeToString(),
        mime_type=constants.MIME_TYPE_PIPELINES,
    )
    buf = io.StringIO()
    console = rich.console.Console(file=buf, width=120, force_terminal=False)
    with mock.patch.object(rich, 'print', side_effect=console.print):
      show._show_pipelines('/base/dir', data, verbose=False, manifest=manifest)
    output = buf.getvalue()

    self.assertIn('weights: manifest_weights', output)
    self.assertIn('weights: poly_weights', output)
    self.assertIn('weights: legacy_weights', output)
    self.assertIn('weights: legacy_bound_weights', output)


if __name__ == '__main__':
  absltest.main()
