from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from .third_party.neptune.model._src.export import jax_model
from .third_party.neptune.model._src.export import utils


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

  @parameterized.named_parameters(
      dict(
          testcase_name='single_fn',
          apply_fn=lambda x: x,
          method_key=None,
          expected_fn_provider=lambda fn: fn,
          error=None,
      ),
      dict(
          testcase_name='mapping_valid_key_one',
          apply_fn={'one': lambda x: x, 'two': lambda x: x * 2},
          method_key='one',
          expected_fn_provider=lambda fn: fn['one'],
          error=None,
      ),
      dict(
          testcase_name='mapping_valid_key_two',
          apply_fn={'one': lambda x: x, 'two': lambda x: x * 2},
          method_key='two',
          expected_fn_provider=lambda fn: fn['two'],
          error=None,
      ),
      dict(
          testcase_name='mapping_none_key',
          apply_fn={'one': lambda x: x},
          method_key=None,
          expected_fn_provider=None,
          error=(
              ValueError,
              'method_key must be provided when apply_fn is a mapping.',
          ),
      ),
      dict(
          testcase_name='mapping_invalid_key',
          apply_fn={'one': lambda x: x},
          method_key='invalid',
          expected_fn_provider=None,
          error=(
              ValueError,
              'method_key "invalid" not found in apply_fn keys:',
          ),
      ),
  )
  def test_get_apply_fn(
      self,
      apply_fn: Callable[..., Any] | dict[str, Callable[..., Any]],
      method_key: str | None,
      expected_fn_provider: (
          Callable[[Callable[..., Any] | dict[str, Callable[..., Any]]], Any]
          | None
      ),
      error: tuple[type[Exception], str] | None,
  ):
    model = jax_model.JaxModel(apply_fn=apply_fn, params={})
    if error:
      with self.assertRaisesRegex(*error):
        utils.get_apply_fn(model, method_key=method_key)
    else:
      self.assertIs(
          utils.get_apply_fn(model, method_key=method_key),
          expected_fn_provider(apply_fn),
      )


if __name__ == '__main__':
  absltest.main()
