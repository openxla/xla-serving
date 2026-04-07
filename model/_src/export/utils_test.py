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
