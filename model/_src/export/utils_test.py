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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from neptune_model._src.core import shlo_type
from neptune_model._src.export import utils


class UtilsTest(parameterized.TestCase):

  def test_to_jax_spec_with_valid_input(self):
    self.assertEqual(
        utils.to_jax_spec((
            jnp.ones((2, 3)),
            {
                'a': jnp.zeros((4,), dtype=jnp.int32),
                'b': jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
            },
        )),
        (
            jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32),
            {
                'a': jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.int32),
                'b': jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
            },
        ),
    )

  def test_to_jax_spec_invalid_input(self):
    tree = (jnp.ones((2, 3)), {'a': 1})  # Integer lacks shape/dtype
    with self.assertRaisesRegex(
        ValueError, 'Failed to convert 1 to jax.ShapeDtypeStruct'
    ):
      utils.to_jax_spec(tree)

  def test_to_shlo_tensor_spec_valid(self):
    self.assertEqual(
        utils.to_shlo_tensor_spec((
            jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32),
            {
                'a': jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.int32),
                'b': shlo_type.ShloTensorSpec(
                    shape=(2,),
                    dtype=shlo_type.ShloDType.f32,
                    name='foo',
                ),
            },
        )),
        (
            shlo_type.ShloTensorSpec(
                shape=(2, 3),
                dtype=shlo_type.ShloDType.f32,
                name='0',
            ),
            {
                'a': shlo_type.ShloTensorSpec(
                    shape=(4,),
                    dtype=shlo_type.ShloDType.i32,
                    name='1.a',
                ),
                'b': shlo_type.ShloTensorSpec(
                    shape=(2,),
                    dtype=shlo_type.ShloDType.f32,
                    name='foo',
                ),
            },
        ),
    )

  def test_to_shlo_tensor_spec_invalid_input(self):
    tree = (jnp.ones((2, 3)), {'a': 1})
    with self.assertRaisesRegex(
        ValueError, 'Failed to convert 1 to ShloTensorSpec'
    ):
      utils.to_shlo_tensor_spec(tree)


if __name__ == '__main__':
  absltest.main()
