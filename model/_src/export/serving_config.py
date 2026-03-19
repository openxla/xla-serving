"""Serving configuration for JAX models."""

import dataclasses
from typing import Any, Sequence

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
    method_key: Used to select a specific apply_fn from the model.
    input_signature: The input signature of the model.
    input_polymophic_shapes: The polymorphic shapes of the input signature. See
      `jax.export.symbolic_args_specs` for more details.
    polymorphic_constraints: Constraints on polymorphic dimensions, e.g., "b >=
      1". See `jax.export.symbolic_args_specs` for more details.
    batch_options: Batch options for serving.
  """

  serving_signature: str = 'serving_default'
  method_key: str | None = None
  # TODO: Evaluate if we need to make this a sequence of PyTrees.
  input_signature: jaxtyping.PyTree = None
  input_polymophic_shapes: Any | None = None
  polymorphic_constraints: Sequence[str] = ()
  batch_options: BatchOptions | None = dataclasses.field(
      repr=False, default=None
  )
