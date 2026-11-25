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

"""Tests for jax_function_loader tools."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from xla_saved_model._src import core as xsm
from xla_saved_model._src import xsm2jax
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib as export_main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.jax2xsm import main_lib as jax2xsm_lib
from xla_saved_model.tools import stitching
from xla_saved_model.tools.stitching import apply_function_loader


def _get_shape_dtype_struct(x):
  return jax.tree_util.tree_map(
      lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), x
  )


def _poly_linear(params, inputs):
  return {'out': inputs['x'] @ params['w'] + params['b']}


def _save_concretized_poly_model(export_path, batch_sizes=(1, 2, 4)):
  params = {
      'w': jnp.array([[2.0, -1.0], [0.5, 3.0]], dtype=jnp.float32),
      'b': jnp.array([1.0, -2.0], dtype=jnp.float32),
  }
  model = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=_poly_linear,
          input_polymorphic_shapes={'x': 'b, _'},
          input_polymorphic_shape_symbol_values={'b': list(batch_sizes)},
          jax_export_kwargs={'platforms': ['cpu']},
      ),
      params=params,
  )
  cfg = serving_config.ServingConfig(
      serving_signature='serving_default',
      input_signature={
          'x': jax.ShapeDtypeStruct(shape=('b', 2), dtype=jnp.float32)
      },
  )
  export_main_lib.save(
      model,
      export_path=export_path,
      serving_configs=[cfg],
  )
  return params


class ApplyFunctionLoaderTest(parameterized.TestCase):

  def test_load_apply_fn_e2e(self):
    def jax_model_fn(params, x):
      return 3.0 * params + x

    jax_params = jnp.array(2.0, dtype=jnp.float32)
    test_input = jnp.array(5.0, dtype=jnp.float32)
    params_spec = _get_shape_dtype_struct(jax_params)
    input_spec = _get_shape_dtype_struct(test_input)

    xsm_shlo_fn = jax2xsm_lib.convert(
        jax.jit(jax_model_fn),
        (params_spec, input_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )

    model_function_name = 'test_fn'
    save_dir_path = self.create_tempdir().full_path
    xsm.save(
        {model_function_name: xsm_shlo_fn},
        save_dir_path,
        xsm.SaveOptions(version=2),
    )

    apply_fn = apply_function_loader.load_apply_fn(
        save_dir_path, model_function_name
    )

    self.assertIsInstance(apply_fn, xsm2jax.ExportedApplyFn)
    self.assertIsInstance(apply_fn.exported, jax.export.Exported)

    # Test direct call
    direct_result = apply_fn(jax_params, test_input)
    expected_result = jax_model_fn(jax_params, test_input)
    self.assertEqual(direct_result, expected_result)
    self.assertEqual(direct_result, 11.0)

    # Test import via package stitching
    fn_via_package = stitching.load_apply_fn(save_dir_path, model_function_name)
    self.assertEqual(fn_via_package(jax_params, test_input), expected_result)

    # Test passing a pathlib.Path object
    fn_from_path = apply_function_loader.load_apply_fn(
        pathlib.Path(save_dir_path), model_function_name
    )
    self.assertEqual(fn_from_path(jax_params, test_input), expected_result)

  def test_function_not_found(self):
    save_dir_path = self.create_tempdir().full_path
    xsm.save({}, save_dir_path, xsm.SaveOptions(version=2))
    with self.assertRaises(KeyError):
      apply_function_loader.load_apply_fn(save_dir_path, 'non_existent_fn')

  def test_model_dir_not_found(self):
    with self.assertRaises(FileNotFoundError):
      apply_function_loader.load_apply_fn('/non/existent/dir', 'fn')

  def test_missing_standard_mime_type_raises_value_error(self):
    def jax_model_fn(params, x):
      return params + x + 1.0

    jax_params = jnp.array(2.0, dtype=jnp.float32)
    test_input = jnp.array(1.0, dtype=jnp.float32)
    params_spec = _get_shape_dtype_struct(jax_params)
    input_spec = _get_shape_dtype_struct(test_input)
    xsm_shlo_fn = jax2xsm_lib.convert(
        jax.jit(jax_model_fn),
        (params_spec, input_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir_path = self.create_tempdir().full_path
    xsm.save(
        {'tampered_fn': xsm_shlo_fn},
        save_dir_path,
        xsm.SaveOptions(version=2),
    )
    manifest = xsm.persistence_lib.load(save_dir_path)
    fn = manifest.objects['tampered_fn'].function
    for data in fn.body.stable_hlo_body.supplemental_info.values():
      data.mime_type = 'unsupported/mime-type'
    (pathlib.Path(save_dir_path) / 'manifest.pb').write_bytes(
        manifest.SerializeToString()
    )

    with self.assertRaisesRegex(ValueError, 'standard MIME type'):
      apply_function_loader.load_apply_fn(save_dir_path, 'tampered_fn')

  def test_file_system_location_path_traversal_raises(self):
    def jax_model_fn(params, x):
      return params + x

    jax_params = jnp.array(2.0, dtype=jnp.float32)
    test_input = jnp.array(1.0, dtype=jnp.float32)
    params_spec = _get_shape_dtype_struct(jax_params)
    input_spec = _get_shape_dtype_struct(test_input)
    xsm_shlo_fn = jax2xsm_lib.convert(
        jax.jit(jax_model_fn),
        (params_spec, input_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir_path = self.create_tempdir().full_path
    xsm.save(
        {'traversal_fn': xsm_shlo_fn},
        save_dir_path,
        xsm.SaveOptions(version=2),
    )
    manifest = xsm.persistence_lib.load(save_dir_path)
    fn = manifest.objects['traversal_fn'].function
    for data in fn.body.stable_hlo_body.supplemental_info.values():
      data.file_system_location.string_path = '../secret.pb'
    (pathlib.Path(save_dir_path) / 'manifest.pb').write_bytes(
        manifest.SerializeToString()
    )

    with self.assertRaisesRegex(ValueError, "cannot contain '\\.\\.'"):
      apply_function_loader.load_apply_fn(save_dir_path, 'traversal_fn')

  def test_file_system_location_absolute_path_raises(self):
    def jax_model_fn(params, x):
      return params + x

    jax_params = jnp.array(2.0, dtype=jnp.float32)
    test_input = jnp.array(1.0, dtype=jnp.float32)
    params_spec = _get_shape_dtype_struct(jax_params)
    input_spec = _get_shape_dtype_struct(test_input)
    xsm_shlo_fn = jax2xsm_lib.convert(
        jax.jit(jax_model_fn),
        (params_spec, input_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir_path = self.create_tempdir().full_path
    xsm.save(
        {'abs_path_fn': xsm_shlo_fn},
        save_dir_path,
        xsm.SaveOptions(version=2),
    )
    manifest = xsm.persistence_lib.load(save_dir_path)
    fn = manifest.objects['abs_path_fn'].function
    for data in fn.body.stable_hlo_body.supplemental_info.values():
      data.file_system_location.string_path = '/etc/passwd'
    (pathlib.Path(save_dir_path) / 'manifest.pb').write_bytes(
        manifest.SerializeToString()
    )

    with self.assertRaisesRegex(ValueError, 'must be relative'):
      apply_function_loader.load_apply_fn(save_dir_path, 'abs_path_fn')

  def test_load_poly_fn_e2e_and_restitching(self):
    export_path = self.create_tempdir().full_path
    params = _save_concretized_poly_model(export_path, batch_sizes=(1, 2, 4))

    apply_fn = stitching.load_apply_fn(export_path, 'serving_default')

    self.assertIsInstance(apply_fn, stitching.PolymorphicExportedApplyFn)
    self.assertLen(apply_fn.concrete_functions, 3)
    self.assertEqual(apply_fn.function_name, 'serving_default')

    params_struct = stitching.get_params_structure(apply_fn)
    self.assertEqual(params_struct, _get_shape_dtype_struct(params))
    loaded_params = stitching.load_params(
        export_path, 'serving_default', params_struct
    )

    with self.assertRaisesRegex(ValueError, '3 concrete functions'):
      _ = apply_fn.exported

    def stitched_downstream(variables, inputs):
      upstream_out = apply_fn(variables['upstream'], inputs)['out']
      return {'score': upstream_out * variables['scale'] + 1.0}

    stitched_vars = {
        'upstream': loaded_params,
        'scale': jnp.array(0.5, dtype=jnp.float32),
    }
    stitched_export_path = self.create_tempdir().full_path
    stitched_model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=stitched_downstream,
            input_polymorphic_shapes={'x': 'b, _'},
            input_polymorphic_shape_symbol_values={'b': [1, 2, 4]},
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params=stitched_vars,
    )
    export_main_lib.save(
        stitched_model,
        export_path=stitched_export_path,
        serving_configs=[
            serving_config.ServingConfig(
                serving_signature='serving_default',
                input_signature={
                    'x': jax.ShapeDtypeStruct(shape=('b', 2), dtype=jnp.float32)
                },
            )
        ],
    )
    reloaded_stitched = stitching.load_apply_fn(
        stitched_export_path, 'serving_default'
    )
    reloaded_vars = stitching.load_params(
        stitched_export_path,
        'serving_default',
        stitching.get_params_structure(reloaded_stitched),
    )

    for batch_size in (1, 2, 4):
      with self.subTest(batch_size=batch_size):
        inputs = {
            'x': (
                jnp.arange(batch_size * 2, dtype=jnp.float32).reshape(
                    batch_size, 2
                )
            )
        }
        expected = _poly_linear(params, inputs)['out']
        np.testing.assert_allclose(
            apply_fn(loaded_params, inputs)['out'], expected
        )
        np.testing.assert_allclose(
            jax.jit(apply_fn)(loaded_params, inputs)['out'], expected
        )
        eval_out = jax.eval_shape(apply_fn, loaded_params, inputs)
        self.assertEqual(eval_out['out'].shape, (batch_size, 2))
        self.assertEqual(eval_out['out'].dtype, jnp.float32)

        expected_stitched = expected * 0.5 + 1.0
        np.testing.assert_allclose(
            reloaded_stitched(reloaded_vars, inputs)['score'],
            expected_stitched,
        )

  def test_load_poly_fn_tpu_platform_reexport_on_cpu(self):
    params = {'w': jnp.array([2.0, 3.0], dtype=jnp.float32)}
    params_spec = _get_shape_dtype_struct(params)
    concrete_xsm_fns = [
        jax2xsm_lib.convert(
            jax.jit(lambda p, x: {'out': x['x'] * p['w']}),
            (
                params_spec,
                {'x': jax.ShapeDtypeStruct((b, 2), jnp.float32)},
            ),
            {},
            platforms=[xsm.manifest_pb2.Platform.TPU],
        )
        for b in (1, 2)
    ]
    save_dir = self.create_tempdir().full_path
    xsm.save(
        {'serving_default': concrete_xsm_fns},
        save_dir,
        xsm.SaveOptions(version=2),
    )

    apply_fn = stitching.load_apply_fn(save_dir, 'serving_default')
    self.assertIsInstance(apply_fn, stitching.PolymorphicExportedApplyFn)
    self.assertEqual(stitching.get_params_structure(apply_fn), params_spec)

    def downstream(p, x):
      return {'out': apply_fn(p, x)['out'] + 1.0}

    for b in (1, 2):
      with self.subTest(batch_size=b):
        exported = jax.export.export(jax.jit(downstream), platforms=('tpu',))(
            params_spec, {'x': jax.ShapeDtypeStruct((b, 2), jnp.float32)}
        )
        self.assertEqual(exported.out_avals[0].shape, (b, 2))

  def test_load_poly_fn_partial_concretization_and_exact_priority(self):
    def fn(params, inputs):
      return inputs['x'] + params['b']

    params = {'b': jnp.array(1.0, dtype=jnp.float32)}
    params_spec = _get_shape_dtype_struct(params)
    sym_b1 = jax.export.symbolic_shape('1, l')
    sym_b2 = jax.export.symbolic_shape('2, l')
    sym_any = jax.export.symbolic_shape('b, l')
    concrete_1 = jax2xsm_lib.convert(
        jax.jit(fn),
        (params_spec, {'x': jax.ShapeDtypeStruct(sym_b1, jnp.float32)}),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    concrete_2 = jax2xsm_lib.convert(
        jax.jit(fn),
        (params_spec, {'x': jax.ShapeDtypeStruct(sym_b2, jnp.float32)}),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir = self.create_tempdir().full_path
    xsm.save(
        {'partial': [concrete_1, concrete_2]},
        save_dir,
        xsm.SaveOptions(version=2),
    )

    poly_fn = apply_function_loader.load_apply_fn(save_dir, 'partial')
    np.testing.assert_allclose(
        poly_fn(params, {'x': jnp.ones((1, 3), dtype=jnp.float32)}),
        jnp.full((1, 3), 2.0, dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        poly_fn(params, {'x': jnp.ones((2, 5), dtype=jnp.float32)}),
        jnp.full((2, 5), 2.0, dtype=jnp.float32),
    )

    # Exact concrete match beats a symbolic fallback.
    exact_fn = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x['x'] + 10.0 * p['b']),
        (params_spec, {'x': jax.ShapeDtypeStruct((2, 3), jnp.float32)}),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    fallback_fn = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x['x'] + p['b']),
        (params_spec, {'x': jax.ShapeDtypeStruct(sym_any, jnp.float32)}),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir_priority = self.create_tempdir().full_path
    xsm.save(
        {'priority': [fallback_fn, exact_fn]},
        save_dir_priority,
        xsm.SaveOptions(version=2),
    )
    priority_poly = apply_function_loader.load_apply_fn(
        save_dir_priority, 'priority'
    )
    np.testing.assert_allclose(
        priority_poly(params, {'x': jnp.zeros((2, 3), dtype=jnp.float32)}),
        jnp.full((2, 3), 10.0, dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        priority_poly(params, {'x': jnp.zeros((3, 4), dtype=jnp.float32)}),
        jnp.full((3, 4), 1.0, dtype=jnp.float32),
    )

  def test_load_poly_fn_single_concrete_function_exposes_exported(self):
    export_path = self.create_tempdir().full_path
    params = _save_concretized_poly_model(export_path, batch_sizes=(2,))
    manifest = xsm.persistence_lib.load(export_path)
    single_fn = manifest.objects['serving_default'].function
    manifest.objects['single_poly'].poly_fn.concrete_functions.add().CopyFrom(
        single_fn
    )
    (pathlib.Path(export_path) / 'manifest.pb').write_bytes(
        manifest.SerializeToString()
    )

    apply_fn = apply_function_loader.load_apply_fn(export_path, 'single_poly')
    self.assertIsInstance(
        apply_fn, apply_function_loader.PolymorphicExportedApplyFn
    )
    self.assertIsInstance(apply_fn.exported, jax.export.Exported)
    inputs = {'x': jnp.ones((2, 2), dtype=jnp.float32)}
    np.testing.assert_allclose(
        apply_fn(params, inputs)['out'], _poly_linear(params, inputs)['out']
    )

  def test_load_poly_fn_empty_raises_value_error(self):
    save_dir = self.create_tempdir().full_path
    xsm.save({}, save_dir, xsm.SaveOptions(version=2))
    manifest = xsm.persistence_lib.load(save_dir)
    manifest.objects['empty_poly'].poly_fn.SetInParent()
    (pathlib.Path(save_dir) / 'manifest.pb').write_bytes(
        manifest.SerializeToString()
    )

    with self.assertRaisesRegex(ValueError, 'has no concrete functions'):
      apply_function_loader.load_apply_fn(save_dir, 'empty_poly')

  def test_load_poly_fn_inconsistent_params_structure_raises_value_error(self):
    x_spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    fn_a = jax2xsm_lib.convert(
        jax.jit(lambda p, x: p['w'] + x),
        ({'w': jax.ShapeDtypeStruct((2,), jnp.float32)}, x_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    fn_b = jax2xsm_lib.convert(
        jax.jit(lambda p, x: p['v'] + x),
        ({'v': jax.ShapeDtypeStruct((2,), jnp.float32)}, x_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir = self.create_tempdir().full_path
    xsm.save({'bad_poly': [fn_a, fn_b]}, save_dir, xsm.SaveOptions(version=2))

    with self.assertRaisesRegex(ValueError, 'different `params` structure'):
      apply_function_loader.load_apply_fn(save_dir, 'bad_poly')

  @parameterized.named_parameters(
      dict(
          testcase_name='unmatched_batch_size',
          inputs={'x': jax.ShapeDtypeStruct((3, 2), jnp.float32)},
          regex=r'No concrete function.*Available input signatures',
      ),
      dict(
          testcase_name='wrong_dtype',
          inputs={'x': jax.ShapeDtypeStruct((2, 2), jnp.int32)},
          regex=r'No concrete function.*Available input signatures',
      ),
      dict(
          testcase_name='pytree_structure_mismatch',
          inputs={'other': jax.ShapeDtypeStruct((2, 2), jnp.float32)},
          regex=r'pytree structure mismatch',
      ),
  )
  def test_poly_fn_mismatch_raises_value_error(self, inputs, regex):
    export_path = self.create_tempdir().full_path
    params = _save_concretized_poly_model(export_path, batch_sizes=(1, 2))
    apply_fn = stitching.load_apply_fn(export_path, 'serving_default')

    with self.assertRaisesRegex(ValueError, regex):
      apply_fn(params, inputs)

  def test_poly_fn_symbolic_batch_against_concrete_raises_value_error(self):
    export_path = self.create_tempdir().full_path
    params = _save_concretized_poly_model(export_path, batch_sizes=(1, 2))
    apply_fn = stitching.load_apply_fn(export_path, 'serving_default')

    sym_x = jax.ShapeDtypeStruct(jax.export.symbolic_shape('b, 2'), jnp.float32)
    with self.assertRaisesRegex(ValueError, 'No concrete function'):
      jax.export.export(jax.jit(apply_fn))(
          _get_shape_dtype_struct(params), {'x': sym_x}
      )

  def test_poly_fn_ambiguous_match_raises_value_error(self):
    spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    fn_a = jax2xsm_lib.convert(
        jax.jit(lambda p, x: p + x),
        (spec, spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    fn_b = jax2xsm_lib.convert(
        jax.jit(lambda p, x: 2.0 * p + x),
        (spec, spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir = self.create_tempdir().full_path
    xsm.save({'dup_poly': [fn_a, fn_b]}, save_dir, xsm.SaveOptions(version=2))
    apply_fn = apply_function_loader.load_apply_fn(save_dir, 'dup_poly')

    with self.assertRaisesRegex(ValueError, 'Ambiguous call'):
      apply_fn(jnp.ones((2,), jnp.float32), jnp.ones((2,), jnp.float32))

  def test_poly_fn_partial_concretization_specificity_tie_break(self):
    p_spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    sym_b_l = jax.export.symbolic_shape('b, l')
    sym_1_l = jax.export.symbolic_shape('1, l')
    fn_general = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x + p[0]),
        (p_spec, jax.ShapeDtypeStruct(sym_b_l, jnp.float32)),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    fn_specialized = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x + p[1]),
        (p_spec, jax.ShapeDtypeStruct(sym_1_l, jnp.float32)),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir = self.create_tempdir().full_path
    xsm.save(
        {'partial_poly': [fn_general, fn_specialized]},
        save_dir,
        xsm.SaveOptions(version=2),
    )
    apply_fn = apply_function_loader.load_apply_fn(save_dir, 'partial_poly')
    params = jnp.array([10.0, 100.0], dtype=jnp.float32)

    # Batch 1 matches both `f32[1, l]` and `f32[b, l]`; `f32[1, l]` is strictly
    # more specific and must win.
    np.testing.assert_allclose(
        apply_fn(params, jnp.ones((1, 5), dtype=jnp.float32)),
        np.full((1, 5), 101.0, dtype=np.float32),
    )
    # Batch 3 matches only `f32[b, l]`.
    np.testing.assert_allclose(
        apply_fn(params, jnp.ones((3, 5), dtype=jnp.float32)),
        np.full((3, 5), 11.0, dtype=np.float32),
    )

  def test_poly_fn_incomparable_partial_concretization_is_ambiguous(self):
    p_spec = jax.ShapeDtypeStruct((1,), jnp.float32)
    fn_first = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x + p),
        (
            p_spec,
            jax.ShapeDtypeStruct(
                jax.export.symbolic_shape('1, l'), jnp.float32
            ),
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    fn_second = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x + 2.0 * p),
        (
            p_spec,
            jax.ShapeDtypeStruct(
                jax.export.symbolic_shape('b, 5'), jnp.float32
            ),
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir = self.create_tempdir().full_path
    xsm.save(
        {'incomp_poly': [fn_first, fn_second]},
        save_dir,
        xsm.SaveOptions(version=2),
    )
    apply_fn = apply_function_loader.load_apply_fn(save_dir, 'incomp_poly')

    with self.assertRaisesRegex(ValueError, 'Ambiguous call'):
      apply_fn(jnp.ones((1,), jnp.float32), jnp.ones((1, 5), jnp.float32))

  def test_poly_fn_shared_symbolic_dim_consistency(self):
    p_spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    fn_square = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x + p[0]),
        (
            p_spec,
            jax.ShapeDtypeStruct(
                jax.export.symbolic_shape('b, b'), jnp.float32
            ),
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    fn_rect = jax2xsm_lib.convert(
        jax.jit(lambda p, x: x + p[1]),
        (
            p_spec,
            jax.ShapeDtypeStruct(
                jax.export.symbolic_shape('b, c'), jnp.float32
            ),
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir = self.create_tempdir().full_path
    xsm.save(
        {'sym_eq_poly': [fn_square, fn_rect]},
        save_dir,
        xsm.SaveOptions(version=2),
    )
    apply_fn = apply_function_loader.load_apply_fn(save_dir, 'sym_eq_poly')
    params = jnp.array([10.0, 100.0], dtype=jnp.float32)

    # Non-square (3, 5) violates `f32[b, b]` and uniquely routes to `f32[b, c]`.
    np.testing.assert_allclose(
        apply_fn(params, jnp.ones((3, 5), dtype=jnp.float32)),
        np.full((3, 5), 101.0, dtype=np.float32),
    )

    # Square-only `poly_fn` rejects non-square (3, 5) and accepts (4, 4).
    square_only_dir = self.create_tempdir().full_path
    xsm.save({'sq': [fn_square]}, square_only_dir, xsm.SaveOptions(version=2))
    sq_fn = apply_function_loader.load_apply_fn(square_only_dir, 'sq')
    np.testing.assert_allclose(
        sq_fn(params, jnp.ones((4, 4), dtype=jnp.float32)),
        np.full((4, 4), 11.0, dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, 'No concrete function'):
      sq_fn(params, jnp.ones((3, 5), dtype=jnp.float32))

  def test_poly_fn_bf16_bool_kwargs_and_params_mismatch_message(self):
    p_spec = jax.ShapeDtypeStruct((2,), jnp.bfloat16)
    fn_1 = jax2xsm_lib.convert(
        jax.jit(lambda p, x: jnp.where(x['flag'], x['v'] + p, x['v'])),
        (
            p_spec,
            {
                'v': jax.ShapeDtypeStruct((1, 2), jnp.bfloat16),
                'flag': jax.ShapeDtypeStruct((), jnp.bool_),
            },
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    fn_2 = jax2xsm_lib.convert(
        jax.jit(lambda p, x: jnp.where(x['flag'], x['v'] + p, x['v'])),
        (
            p_spec,
            {
                'v': jax.ShapeDtypeStruct((2, 2), jnp.bfloat16),
                'flag': jax.ShapeDtypeStruct((), jnp.bool_),
            },
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )
    save_dir = self.create_tempdir().full_path
    xsm.save({'bf16_poly': [fn_1, fn_2]}, save_dir, xsm.SaveOptions(version=2))
    apply_fn = apply_function_loader.load_apply_fn(save_dir, 'bf16_poly')
    params = jnp.array([1.0, 2.0], dtype=jnp.bfloat16)
    inputs = {
        'v': jnp.ones((2, 2), dtype=jnp.bfloat16),
        'flag': jnp.array(True),
    }

    out = apply_fn(params=params, inputs=inputs)
    np.testing.assert_allclose(
        out.astype(jnp.float32),
        np.array([[2.0, 3.0], [2.0, 3.0]], dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, 'params mismatch'):
      apply_fn(jnp.ones((2,), dtype=jnp.float32), inputs)

  def test_function_not_found_lists_poly_fn_in_available_functions(self):
    export_path = self.create_tempdir().full_path
    _save_concretized_poly_model(export_path, batch_sizes=(1, 2))

    with self.assertRaisesRegex(KeyError, 'serving_default'):
      apply_function_loader.load_apply_fn(export_path, 'missing_fn')


if __name__ == '__main__':
  absltest.main()
