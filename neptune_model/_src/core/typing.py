"""Core types for Neptune model."""

import dataclasses

import numpy as np


@dataclasses.dataclass
class TensorSpec:
  """A specification for the shape, dtype, sharding, and layout of a StableHLO tensor.

  Attributes:
    shape: The shape of the tensor. None represents dynamic or unknown
      dimensions.
    dtype: The dtype of the tensor.
  """

  shape: tuple[int | None, ...]
  dtype: np.typing.DTypeLike

  def __post_init__(self) -> None:
    self.dtype = np.dtype(self.dtype)
