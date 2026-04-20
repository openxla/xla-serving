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

from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from neptune_model._src.export import jax_model
from neptune_model._src.export import utils


class UtilsTest(parameterized.TestCase):

  def test_to_jax_spec_with_valid_input(self):
    self.assertEqual(
        utils.to_jax_spec(
            (jnp.ones((2, 3)), {'a': jnp.zeros((4,), dtype=jnp.int32)})
        ),
        (
            jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32),
            {'a': jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.int32)},
        ),
    )

  def test_to_jax_spec_invalid_input(self):
    tree = (jnp.ones((2, 3)), {'a': 1})  # Integer lacks shape/dtype
    with self.assertRaisesRegex(
        ValueError, 'All leaf nodes must have "shape" and "dtype"'
    ):
      utils.to_jax_spec(tree)

if __name__ == '__main__':
  absltest.main()
