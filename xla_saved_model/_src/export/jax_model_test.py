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

"""Tests for xla_saved_model._src.export.jax_model."""

import pathlib

from absl.testing import absltest
import jax.numpy as jnp

from xla_saved_model._src.export import jax_model


class JaxModelTest(absltest.TestCase):

  def test_create_jax_model_with_bundle_and_params(self):
    bundle = jax_model.ApplyFnBundle(apply_fn=lambda p, x: x * p['w'])
    model = jax_model.JaxModel(
        name='test_model',
        apply_fn_bundle=bundle,
        params={'w': jnp.array(2.0)},
    )
    self.assertEqual(
        model._apply_fn_bundle[jax_model.DEFAULT_METHOD_KEY], bundle
    )
    self.assertEqual(model.params, {'w': jnp.array(2.0)})
    self.assertIsNone(model.jax_mesh)

  def test_create_jax_model_with_bundle_map_and_params(self):
    bundle = jax_model.ApplyFnBundle(apply_fn=lambda p, x: x * p['w'])
    model = jax_model.JaxModel(
        name='test_model',
        apply_fn_bundle={'another_key': bundle},
        params={'w': jnp.array(2.0)},
    )
    self.assertEqual(model._apply_fn_bundle['another_key'], bundle)

  def test_save_checkpoint_with_checkpoint_path_does_not_save_params(self):
    bundle = jax_model.ApplyFnBundle(apply_fn=lambda p, x: x * p['w'])
    model = jax_model.JaxModel(
        name='test_model',
        apply_fn_bundle=bundle,
        params={'w': jnp.array(2.0)},
        checkpoint_path='external/path/to/checkpoint',
    )
    temp_dir = self.create_tempdir().full_path
    result = model.save_checkpoint(
        export_path=temp_dir,
        checkpoint_path='checkpoints/test_model',
        weights_name='test_model_weights',
    )
    self.assertIn('test_model_weights', result)
    ext_val = result['test_model_weights']
    self.assertEqual(
        ext_val.data.file_system_location.string_path,
        'external/path/to/checkpoint',
    )
    # Verify no checkpoints were written to export_path
    export_path_entries = list(pathlib.Path(temp_dir).iterdir())
    self.assertEmpty(export_path_entries)

  def test_save_checkpoint_without_checkpoint_path_saves_params(self):
    bundle = jax_model.ApplyFnBundle(apply_fn=lambda p, x: x * p['w'])
    model = jax_model.JaxModel(
        name='test_model',
        apply_fn_bundle=bundle,
        params={'w': jnp.array(2.0)},
    )
    temp_dir = self.create_tempdir().full_path
    result = model.save_checkpoint(
        export_path=temp_dir,
        checkpoint_path='checkpoints',
        weights_name='weights',
    )
    self.assertIn('weights', result)
    ext_val = result['weights']
    self.assertEqual(
        ext_val.data.file_system_location.string_path,
        'checkpoints',
    )
    # Verify checkpoint was written to export_path / checkpoints
    self.assertTrue((pathlib.Path(temp_dir) / 'checkpoints').exists())

  def test_save_checkpoint_without_params_or_checkpoint_path(self):
    bundle = jax_model.ApplyFnBundle(apply_fn=lambda p, x: x * 2.0)
    model = jax_model.JaxModel(
        name='test_model',
        apply_fn_bundle=bundle,
    )
    temp_dir = self.create_tempdir().full_path
    result = model.save_checkpoint(
        export_path=temp_dir,
        checkpoint_path='checkpoints',
        weights_name='weights',
    )
    self.assertEmpty(result)


if __name__ == '__main__':
  absltest.main()
