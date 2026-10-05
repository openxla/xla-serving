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

"""Tests for weights_loader."""

from collections.abc import Callable
import dataclasses
import os
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src import core as xsm
from xla_saved_model._src.core import constants as core_constants
from xla_saved_model._src.export import constants as export_constants
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.jax2xsm import main_lib as jax2xsm_lib
from xla_saved_model.tools import stitching
from xla_saved_model.tools.stitching import weights_loader

_FUNCTION_NAME = 'serving_default'


# Stands in for a custom node such as `qwix.QArray`. Its fields are declared in
# the opposite of alphabetical order, so flattening it and flattening the dict
# a checkpoint restores it as yield the leaves in different orders.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Quantized:
  value: Any
  scale: Any
  name: str = dataclasses.field(metadata=dict(static=True), default='')


# A node registered without keys, so its key paths hold `FlattenedIndexKey`s.
@dataclasses.dataclass(frozen=True)
class _Keyless:
  first: Any
  second: Any


jax.tree_util.register_pytree_node(
    _Keyless,
    lambda node: ((node.first, node.second), None),
    lambda _, children: _Keyless(*children),
)


def _abstract(params):
  return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)


def _save_model(params, apply_fn, export_path):
  model = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=apply_fn,
          jax_export_kwargs={'platforms': ['cpu']},
      ),
      params=params,
  )
  main_lib.save(
      model,
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature=_FUNCTION_NAME,
              input_signature={
                  'x': jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32)
              },
          )
      ],
  )


def _dense(params, inputs):
  return {'out': inputs['x'] * params['w'] + params['b']}


def _quantized(params, inputs):
  kernel = params['kernel']
  return {'out': inputs['x'] * kernel.value * kernel.scale}


def _save_dense_model(export_path):
  _save_model({'w': jnp.ones((2,)), 'b': jnp.zeros((2,))}, _dense, export_path)


def _edit_manifest(
    export_path: str, edit: Callable[[manifest_pb2.Manifest], None]
) -> None:
  """Applies `edit` to the manifest saved under `export_path`."""
  manifest = xsm.load(export_path)
  edit(manifest)
  manifest_path = os.path.join(export_path, core_constants.MANIFEST_FILE_PATH)
  with open(manifest_path, 'wb') as f:
    f.write(manifest.SerializeToString())


def _weights_name(manifest: manifest_pb2.Manifest) -> str:
  """Returns the name of the only Orbax checkpoint in the manifest."""
  (name,) = [
      name
      for name, obj in manifest.objects.items()
      if obj.WhichOneof('case') == 'value'
      and obj.value.external.data.mime_type
      == export_constants.ORBAX_CHECKPOINT_MIME_TYPE
  ]
  return name


def _weights_data(
    manifest: manifest_pb2.Manifest,
) -> manifest_pb2.UnstructuredData:
  return manifest.objects[_weights_name(manifest)].value.external.data


def _replace_pipelines(
    manifest: manifest_pb2.Manifest,
    pipelines: orchestration_pb2.Pipelines,
    mime_type: str = core_constants.ORCHESTRATION_MULTI_PIPELINE_MIME_TYPE,
) -> None:
  """Replaces the manifest's pipelines with `pipelines`, inlined."""
  for data in manifest.supplemental_info.values():
    if data.mime_type == core_constants.ORCHESTRATION_MULTI_PIPELINE_MIME_TYPE:
      data.mime_type = mime_type
      data.inlined_bytes = pipelines.SerializeToString()
      return
  raise AssertionError('The manifest has no pipelines.')


class FindCheckpointPathTest(absltest.TestCase):

  def test_follows_the_pipeline_binding(self):
    export_path = self.create_tempdir().full_path
    _save_model(
        {'w': jnp.ones((2,)), 'b': jnp.zeros((2,))}, _dense, export_path
    )

    path = weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME)

    self.assertTrue(path.startswith(export_path))
    self.assertTrue(os.path.isdir(path))

  def test_rejects_a_model_without_pipelines(self):
    spec = jax.ShapeDtypeStruct((), jnp.float32)
    export_path = self.create_tempdir().full_path
    xsm.save(
        {
            'fn': jax2xsm_lib.convert(
                jax.jit(lambda p, x: p['w'] * x),
                ({'w': spec}, spec),
                {},
                platforms=[xsm.manifest_pb2.Platform.CPU],
            ),
            'my_weights': jax2xsm_lib.convert_path_to_value(
                'my_checkpoint/', mime_type='application/x.orbax-checkpoint'
            ),
        },
        export_path,
        xsm.SaveOptions(version=2),
    )

    with self.assertRaisesRegex(ValueError, 'no orchestration pipelines'):
      weights_loader.find_checkpoint_path(export_path, 'fn')
    with self.assertRaisesRegex(ValueError, 'no orchestration pipelines'):
      weights_loader.load_params(export_path, 'fn', {'w': spec})

  def test_rejects_a_function_without_bound_weights(self):
    export_path = self.create_tempdir().full_path
    _save_model(
        {'w': jnp.ones((2,)), 'b': jnp.zeros((2,))}, _dense, export_path
    )

    with self.assertRaisesRegex(ValueError, "No pipeline binds weights to 'f'"):
      weights_loader.find_checkpoint_path(export_path, 'f')

  def test_follows_a_legacy_binding_under_the_orbax_mime_type(self):
    export_path = self.create_tempdir().full_path
    _save_dense_model(export_path)
    expected = weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME)

    def edit(manifest):
      pipelines = orchestration_pb2.Pipelines()
      pipelines.name_to_pipeline['p'].model_functions.add(
          model_function_name='legacy_fn', weights_name=_weights_name(manifest)
      )
      _replace_pipelines(
          manifest,
          pipelines,
          mime_type=(
              'application/protobuf; type=orbax_export_orchestration.Pipelines'
          ),
      )

    _edit_manifest(export_path, edit)

    self.assertEqual(
        weights_loader.find_checkpoint_path(export_path, 'legacy_fn'), expected
    )

  def test_ignores_bindings_without_weights(self):
    export_path = self.create_tempdir().full_path
    _save_dense_model(export_path)

    def edit(manifest):
      pipelines = orchestration_pb2.Pipelines()
      pipeline = pipelines.name_to_pipeline['p']
      pipeline.components.add(function_name='fn')
      pipeline.model_functions.add(model_function_name='fn')
      _replace_pipelines(manifest, pipelines)

    _edit_manifest(export_path, edit)

    with self.assertRaisesRegex(
        ValueError, "No pipeline binds weights to 'fn'"
    ):
      weights_loader.find_checkpoint_path(export_path, 'fn')

  def test_rejects_weights_missing_from_the_manifest(self):
    export_path = self.create_tempdir().full_path
    _save_dense_model(export_path)

    def edit(manifest):
      del manifest.objects[_weights_name(manifest)]

    _edit_manifest(export_path, edit)

    with self.assertRaisesRegex(ValueError, 'are not in the manifest'):
      weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME)

  def test_rejects_weights_that_are_not_an_orbax_checkpoint(self):
    export_path = self.create_tempdir().full_path
    _save_dense_model(export_path)

    def edit(manifest):
      _weights_data(manifest).mime_type = 'application/octet-stream'

    # The weights can no longer be found by MIME type once edited, so resolve
    # the name up front.
    weights_name = _weights_name(xsm.load(export_path))
    _edit_manifest(export_path, edit)

    with self.assertRaisesRegex(
        ValueError,
        f"'{weights_name}' are not an Orbax checkpoint \\(MIME type"
        " 'application/octet-stream'\\)",
    ):
      weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME)

  def test_rejects_weights_without_a_file_system_location(self):
    export_path = self.create_tempdir().full_path
    _save_dense_model(export_path)

    def edit(manifest):
      _weights_data(manifest).inlined_bytes = b'weights'

    _edit_manifest(export_path, edit)

    with self.assertRaisesRegex(
        ValueError, 'do not name a file system location'
    ):
      weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME)

  def test_rejects_a_path_escaping_with_dot_dot(self):
    export_path = self.create_tempdir().full_path
    _save_dense_model(export_path)

    def edit(manifest):
      _weights_data(manifest).file_system_location.string_path = (
          'checkpoint/../../elsewhere'
      )

    _edit_manifest(export_path, edit)

    with self.assertRaisesRegex(ValueError, "cannot contain '..'"):
      weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME)

  def test_returns_an_absolute_path_unchanged(self):
    export_path = self.create_tempdir().full_path
    checkpoint_path = self.create_tempdir().full_path
    _save_dense_model(export_path)

    def edit(manifest):
      _weights_data(manifest).file_system_location.string_path = checkpoint_path

    _edit_manifest(export_path, edit)

    self.assertEqual(
        weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME),
        checkpoint_path,
    )

  def test_resolves_a_frequent_path(self):
    export_path = self.create_tempdir().full_path
    _save_dense_model(export_path)

    def edit(manifest):
      manifest.frequent_paths.append('shared/checkpoint')
      _weights_data(manifest).file_system_location.frequent_path_idx = (
          len(manifest.frequent_paths) - 1
      )

    _edit_manifest(export_path, edit)

    self.assertEqual(
        weights_loader.find_checkpoint_path(export_path, _FUNCTION_NAME),
        os.path.join(export_path, 'shared/checkpoint'),
    )


class LoadParamsTest(absltest.TestCase):

  def test_round_trips(self):
    params = {'w': jnp.array([2.0, 3.0]), 'b': jnp.array([1.0, -1.0])}
    export_path = self.create_tempdir().full_path
    _save_model(params, _dense, export_path)

    apply_fn = stitching.load_apply_fn(export_path, _FUNCTION_NAME)
    loaded = stitching.load_params(
        export_path, _FUNCTION_NAME, _abstract(params)
    )

    jax.tree.map(np.testing.assert_array_equal, loaded, params)
    inputs = {'x': jnp.array([1.0, 2.0])}
    np.testing.assert_array_equal(
        apply_fn(loaded, inputs)['out'], _dense(params, inputs)['out']
    )

  def test_rebuilds_custom_nodes(self):
    params = {
        'kernel': _Quantized(
            value=jnp.array([1.0, 2.0]), scale=jnp.array([3.0, 4.0]), name='k'
        )
    }
    export_path = self.create_tempdir().full_path
    _save_model(params, _quantized, export_path)

    apply_fn = stitching.load_apply_fn(export_path, _FUNCTION_NAME)
    loaded = stitching.load_params(
        export_path, _FUNCTION_NAME, _abstract(params)
    )

    self.assertIsInstance(loaded['kernel'], _Quantized)
    self.assertEqual(loaded['kernel'].name, 'k')
    np.testing.assert_array_equal(loaded['kernel'].value, [1.0, 2.0])
    np.testing.assert_array_equal(loaded['kernel'].scale, [3.0, 4.0])
    inputs = {'x': jnp.array([1.0, 1.0])}
    np.testing.assert_array_equal(
        apply_fn(loaded, inputs)['out'], _quantized(params, inputs)['out']
    )

  def test_uses_the_structure_recorded_by_apply_fn(self):
    params = {
        'kernel': _Quantized(
            value=jnp.array([1.0, 2.0]), scale=jnp.array([3.0, 4.0]), name='k'
        )
    }
    export_path = self.create_tempdir().full_path
    _save_model(params, _quantized, export_path)

    apply_fn = stitching.load_apply_fn(export_path, _FUNCTION_NAME)
    structure = stitching.get_params_structure(apply_fn)
    loaded = stitching.load_params(export_path, _FUNCTION_NAME, structure)

    self.assertEqual(jax.tree.structure(structure), jax.tree.structure(params))
    self.assertIsInstance(loaded['kernel'], _Quantized)
    self.assertEqual(loaded['kernel'].name, 'k')
    inputs = {'x': jnp.array([1.0, 1.0])}
    np.testing.assert_array_equal(
        apply_fn(loaded, inputs)['out'], _quantized(params, inputs)['out']
    )


class GetParamsStructureTest(absltest.TestCase):

  def test_returns_abstract_params(self):
    params = {'w': jnp.ones((2,)), 'b': jnp.zeros((2,), jnp.int32)}
    export_path = self.create_tempdir().full_path
    _save_model(params, _dense, export_path)

    apply_fn = stitching.load_apply_fn(export_path, _FUNCTION_NAME)

    self.assertEqual(
        stitching.get_params_structure(apply_fn), _abstract(params)
    )


class RestructureParamsTest(absltest.TestCase):

  def test_matches_leaves_by_name_not_position(self):
    expected = _Quantized(
        value=jax.ShapeDtypeStruct((2,), jnp.float32),
        scale=jax.ShapeDtypeStruct((3,), jnp.float32),
    )
    restored = {'scale': jnp.zeros((3,)), 'value': jnp.ones((2,))}

    result = weights_loader.restructure_params(restored, expected)

    np.testing.assert_array_equal(result.value, jnp.ones((2,)))
    np.testing.assert_array_equal(result.scale, jnp.zeros((3,)))

  def test_reports_missing_leaves(self):
    expected = {'a': jax.ShapeDtypeStruct((), jnp.float32)}
    with self.assertRaisesRegex(ValueError, r"missing \['a'\]"):
      weights_loader.restructure_params({}, expected)

  def test_reports_unexpected_leaves(self):
    expected = {'a': jax.ShapeDtypeStruct((), jnp.float32)}
    restored = {'a': jnp.zeros(()), 'b': jnp.zeros(())}
    with self.assertRaisesRegex(ValueError, r"unexpected \['b'\]"):
      weights_loader.restructure_params(restored, expected)

  def test_reports_mismatched_leaves(self):
    expected = {'a': jax.ShapeDtypeStruct((2,), jnp.float32)}
    with self.assertRaisesRegex(ValueError, 'mismatched'):
      weights_loader.restructure_params({'a': jnp.zeros((3,))}, expected)
    with self.assertRaisesRegex(ValueError, 'mismatched'):
      weights_loader.restructure_params(
          {'a': jnp.zeros((2,), jnp.int32)}, expected
      )

  def test_describes_mismatched_leaves(self):
    expected = {'a': {'b': jax.ShapeDtypeStruct((2,), jnp.float32)}}
    with self.assertRaisesRegex(
        ValueError, r"mismatched \['a/b: int32\[3\], expected float32\[2\]'\]"
    ):
      weights_loader.restructure_params(
          {'a': {'b': jnp.zeros((3,), jnp.int32)}}, expected
      )

  def test_reports_a_rank_mismatch(self):
    expected = {'a': jax.ShapeDtypeStruct((2,), jnp.float32)}
    with self.assertRaisesRegex(ValueError, 'mismatched'):
      weights_loader.restructure_params({'a': jnp.zeros((2, 1))}, expected)
    with self.assertRaisesRegex(ValueError, 'mismatched'):
      weights_loader.restructure_params({'a': jnp.zeros(())}, expected)

  def test_accepts_any_size_for_a_symbolic_dimension(self):
    expected = {
        'a': jax.ShapeDtypeStruct(
            jax.export.symbolic_shape('b, 2'), jnp.float32
        )
    }

    result = weights_loader.restructure_params(
        {'a': jnp.ones((5, 2))}, expected
    )

    np.testing.assert_array_equal(result['a'], jnp.ones((5, 2)))
    # Concrete dimensions next to symbolic ones are still checked.
    with self.assertRaisesRegex(ValueError, 'mismatched'):
      weights_loader.restructure_params({'a': jnp.ones((5, 3))}, expected)

  def test_accepts_concrete_arrays_as_the_expected_structure(self):
    expected = {'a': jnp.zeros((2,)), 'b': jnp.zeros((), jnp.int32)}
    restored = {'a': jnp.ones((2,)), 'b': jnp.ones((), jnp.int32)}

    result = weights_loader.restructure_params(restored, expected)

    jax.tree.map(np.testing.assert_array_equal, result, restored)

  def test_matches_sequence_indices_to_names(self):
    expected = {
        'layers': [
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.float32),
        ]
    }
    # Sequences may be restored either as sequences or as dicts keyed by index.
    for layers in (
        [jnp.zeros((1,)), jnp.ones((2,))],
        (jnp.zeros((1,)), jnp.ones((2,))),
        {'1': jnp.ones((2,)), '0': jnp.zeros((1,))},
    ):
      with self.subTest(type(layers).__name__):
        result = weights_loader.restructure_params({'layers': layers}, expected)

        self.assertIsInstance(result['layers'], list)
        np.testing.assert_array_equal(result['layers'][0], jnp.zeros((1,)))
        np.testing.assert_array_equal(result['layers'][1], jnp.ones((2,)))

  def test_matches_flattened_indices_of_keyless_nodes(self):
    expected = _Keyless(
        jax.ShapeDtypeStruct((1,), jnp.float32),
        jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    restored = {'1': jnp.ones((2,)), '0': jnp.zeros((1,))}

    result = weights_loader.restructure_params(restored, expected)

    self.assertIsInstance(result, _Keyless)
    np.testing.assert_array_equal(result.first, jnp.zeros((1,)))
    np.testing.assert_array_equal(result.second, jnp.ones((2,)))

  def test_limits_the_number_of_reported_leaves(self):
    max_reported = weights_loader._MAX_REPORTED  # pylint: disable=protected-access
    spec = jax.ShapeDtypeStruct((), jnp.float32)
    expected = {f'a{i}': spec for i in range(max_reported + 2)}

    with self.assertRaises(ValueError) as cm:
      weights_loader.restructure_params({}, expected)

    message = str(cm.exception)
    for i in range(max_reported):
      self.assertIn(f"'a{i}'", message)
    self.assertNotIn(f"'a{max_reported}'", message)

  def test_reports_all_kinds_of_problems_together(self):
    spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    expected = {'missing': spec, 'mismatched': spec}
    restored = {'mismatched': jnp.zeros((3,)), 'unexpected': jnp.zeros((2,))}

    with self.assertRaisesRegex(
        ValueError,
        r"^The checkpoint does not match the function's parameters: missing"
        r" \['missing'\]; mismatched \['mismatched: .*'\]; unexpected"
        r" \['unexpected'\]\.$",
    ):
      weights_loader.restructure_params(restored, expected)


if __name__ == '__main__':
  absltest.main()
