"""Tests for neptune.model._src.export.jax_model."""

import pathlib
from absl.testing import absltest
from .third_party.neptune.model._src.export import jax_model


class JaxModelTest(absltest.TestCase):

  def test_no_params_or_checkpoint_path_raises_error(self):
    with self.assertRaises(ValueError):
      jax_model.JaxModel(apply_fn=lambda _, x: x)

  def test_both_params_and_checkpoint_path_raises_error(self):
    with self.assertRaises(ValueError):
      jax_model.JaxModel(
          apply_fn=lambda _, x: x,
          params={'w': 1},
          checkpoint_path=pathlib.Path('/tmp/checkpoint'),
      )


if __name__ == '__main__':
  absltest.main()
