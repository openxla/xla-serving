"""Tests for neptune.model._src.export.main_lib."""

from absl.testing import absltest
from google3.third_party.neptune.model._src.export import jax_model
from google3.third_party.neptune.model._src.export import main_lib


class SaveModelTest(absltest.TestCase):

  def test_save_raises_not_implemented_error(self):
    """Tests that save() raises NotImplementedError."""
    model = jax_model.JaxModel(apply_fn=lambda _, x: x, params={'w': 1})
    with self.assertRaises(NotImplementedError):
      main_lib.save(
          model=model,
          export_path='/tmp/export',
          serving_configs=[],
      )


if __name__ == '__main__':
  absltest.main()
