"""A JAX model, consisting of an apply_fn and its parameters."""

from collections.abc import Mapping
import dataclasses
import pathlib
import jax
import jaxtyping
from .third_party.neptune.model._src.jax2npt import typing


@dataclasses.dataclass(frozen=True)
class JaxModel:
  """A JAX model, consisting of an apply_fn and its parameters.

  Exactly one of `params` or `checkpoint_path` must be provided.

  Attributes:
    apply_fn: The function for applying the model, or a mapping of such
      functions.
    params: The parameters of the model.
    checkpoint_path: The path to a checkpoint from which to load the parameters.
    jax_mesh: Mesh for the model.
  """

  apply_fn: typing.ApplyFn | Mapping[str, typing.ApplyFn]
  params: jaxtyping.PyTree | None = None
  checkpoint_path: pathlib.Path | None = None
  jax_mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    if (self.params is None) == (self.checkpoint_path is None):
      raise ValueError(
          'Exactly one of `params` or `checkpoint_path` must be provided.'
      )
