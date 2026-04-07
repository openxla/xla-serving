"""Tests for neptune.model._src.export.jax_model."""

from absl.testing import absltest
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


if __name__ == '__main__':
  absltest.main()
