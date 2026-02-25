"""Serving configuration for JAX models."""

import dataclasses

import jaxtyping


@dataclasses.dataclass
class BatchOptions:
  """Batch options for the Neptune model."""


@dataclasses.dataclass(kw_only=True)
class ServingConfig:
  """Serving config for a Jax model.

  Attributes:
    serving_signature: The key of the serving signature which can be used to
      invoke a specific exported model.
    input_signature: The input signature of the model.
    method_key: Used to select a specific apply_fn from the model.
    batch_options: Batch options for serving.
  """

  serving_signature: str
  # TODO: Evaluate if we need to make this a sequence of PyTrees.
  input_signature: jaxtyping.PyTree = None
  method_key: str | None = None
  batch_options: BatchOptions | None = dataclasses.field(
      repr=False, default=None
  )
