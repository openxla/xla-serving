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
import jax
import jax.numpy as jnp

from xla_saved_model._src import core as xsm
from xla_saved_model._src import xsm2jax
from xla_saved_model._src.jax2xsm import main_lib as jax2xsm_lib
from xla_saved_model.tools import stitching
from xla_saved_model.tools.stitching import apply_function_loader


def _get_shape_dtype_struct(x):
  return jax.tree_util.tree_map(
      lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), x
  )


class ApplyFunctionLoaderTest(absltest.TestCase):

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



if __name__ == '__main__':
  absltest.main()
