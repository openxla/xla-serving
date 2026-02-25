"""Library for exporting JAX models."""

from collections.abc import Sequence

from absl import logging

from google3.third_party.neptune.model._src.export import jax_model
from google3.third_party.neptune.model._src.export import serving_config


def save(
    model: jax_model.JaxModel,
    export_path: str,
    serving_configs: Sequence[serving_config.ServingConfig],
) -> None:
  """Saves a JAX model to neptune model format.

  Args:
    model: The JAX model to save.
    export_path: The directory to which to save the model.
    serving_configs: The serving configurations.
  """
  logging.info(
      'Saving model=%s to export_path=%s with serving_configs=%s',
      model,
      export_path,
      serving_configs,
  )
  raise NotImplementedError()
