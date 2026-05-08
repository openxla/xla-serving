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

"""Tests for neptune.model._src.export.jax_model."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from neptune_model._src.export import jax_model


class JaxModelTest(absltest.TestCase):

  def test_create_jax_model_with_bundle_and_params(self):
    bundle = jax_model.ApplyFnBundle(apply_fn=lambda p, x: x * p['w'])
    model = jax_model.JaxModel(
        apply_fn_bundle=bundle,
        params={'w': jnp.array(2.0)},
    )
    self.assertEqual(
        model._apply_fn_bundle[jax_model.SERVING_DEFAULT_KEY], bundle
    )
    self.assertEqual(model.params, {'w': jnp.array(2.0)})
    self.assertIsNone(model.jax_mesh)

  def test_create_jax_model_with_bundle_map_and_params(self):
    bundle = jax_model.ApplyFnBundle(apply_fn=lambda p, x: x * p['w'])
    model = jax_model.JaxModel(
        apply_fn_bundle={'another_key': bundle},
        params={'w': jnp.array(2.0)},
    )
    self.assertEqual(model._apply_fn_bundle['another_key'], bundle)
    self.assertEqual(model.params, {'w': jnp.array(2.0)})
    self.assertIsNone(model.jax_mesh)

  def test_concretize_polymorphic_shapes(self):
    symbolic_specs = {
        'input': jax.ShapeDtypeStruct(shape=('b', 'l'), dtype=jnp.int32)
    }
    symbol_to_values = {'b': [1, 2], 'l': [4, 5]}
    concretized = jax_model._concretize_polymorphic_shapes(
        symbolic_specs, symbol_to_values
    )
    expected = [
        {'input': jax.ShapeDtypeStruct(shape=(1, 4), dtype=jnp.int32)},
        {'input': jax.ShapeDtypeStruct(shape=(1, 5), dtype=jnp.int32)},
        {'input': jax.ShapeDtypeStruct(shape=(2, 4), dtype=jnp.int32)},
        {'input': jax.ShapeDtypeStruct(shape=(2, 5), dtype=jnp.int32)},
    ]
    self.assertEqual(concretized, expected)

  def test_concretize_polymorphic_shapes_with_concrete_dim(self):
    symbolic_specs = {
        'input': jax.ShapeDtypeStruct(shape=(10, 'l'), dtype=jnp.int32)
    }
    symbol_to_values = {'l': [4, 5]}
    concretized = jax_model._concretize_polymorphic_shapes(
        symbolic_specs, symbol_to_values
    )
    expected = [
        {'input': jax.ShapeDtypeStruct(shape=(10, 4), dtype=jnp.int32)},
        {'input': jax.ShapeDtypeStruct(shape=(10, 5), dtype=jnp.int32)},
    ]
    self.assertEqual(concretized, expected)


if __name__ == '__main__':
  absltest.main()
