# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The shape and dtype of a tensor in a neptune signature.

Unlike `jax.ShapeDtypeStruct`, a dimension may be unknown, which is None. A
negative size is never read as unknown: it is an error.
"""

from collections.abc import Iterable
import dataclasses
from typing import Any

import jax
import numpy as np
import numpy.typing as npt


def _is_size(d: Any) -> bool:
  """Returns True iff `d` is a Python or NumPy int, and not a bool."""
  return isinstance(d, (int, np.integer)) and not isinstance(
      d, (bool, np.bool_)
  )


def to_partial_shape(shape: Iterable[Any]) -> tuple[int | None, ...]:
  """Returns the shape of an input signature leaf, with None for unknown dims.

  Accepts the shape of any signature leaf, such as a `jax.ShapeDtypeStruct`,
  whose dims may be symbolic.

  Args:
    shape: The dims. None, and any dim that isn't an int (e.g. a JAX symbolic
      dim), is unknown. A non-negative int is a static size.

  Returns:
    The dims, with None for each unknown one.

  Raises:
    ValueError: If a dim is a negative int. -1 never means unknown.
  """
  shape = tuple(shape)
  dims = []
  for i, d in enumerate(shape):
    if not _is_size(d):
      dims.append(None)
    elif d < 0:
      raise ValueError(
          f'Dimension {i} is {d} in shape {shape!r}. Use None for an'
          ' unknown dimension; -1 is not accepted.'
      )
    else:
      dims.append(int(d))
  return tuple(dims)


@dataclasses.dataclass(frozen=True, init=False)
class ShapeDtypeStruct:
  """The shape and dtype of a tensor, with None for an unknown dimension.

  Attributes:
    shape: A tuple with a non-negative int, or None for an unknown dimension,
      per dimension. Any sequence is accepted and stored as a tuple of Python
      ints and Nones.
    dtype: A `np.dtype`. Anything `np.dtype` accepts is converted, as
      `jax.ShapeDtypeStruct` does.
  """

  shape: tuple[int | None, ...]
  dtype: np.dtype

  def __init__(
      self,
      shape: Iterable[int | np.integer | None],
      dtype: npt.DTypeLike,
  ):
    if not isinstance(shape, Iterable) or isinstance(shape, (str, bytes)):
      raise TypeError(
          f'ShapeDtypeStruct shape must be a sequence, got {shape!r}'
      )
    original_shape = tuple(shape)
    dims = []
    for i, d in enumerate(original_shape):
      if d is None:
        dims.append(None)
      elif not _is_size(d):
        raise TypeError(
            f'ShapeDtypeStruct dimension {i} must be an int or None, got'
            f' {d!r} in shape {original_shape!r}'
        )
      elif d < 0:
        raise ValueError(
            f'ShapeDtypeStruct dimension {i} is {d} in shape'
            f' {original_shape!r}. Use None for an unknown dimension; -1 is not'
            ' accepted.'
        )
      else:
        dims.append(int(d))
    object.__setattr__(self, 'shape', tuple(dims))
    object.__setattr__(self, 'dtype', np.dtype(dtype))

  def to_jax(self) -> jax.ShapeDtypeStruct:
    """Returns the `jax.ShapeDtypeStruct`, which can't hold an unknown dim.

    Raises:
      ValueError: If a dimension is unknown.
    """
    if any(d is None for d in self.shape):
      raise ValueError(
          f'Cannot convert shape {self.shape} with an unknown dimension to a'
          ' jax.ShapeDtypeStruct.'
      )
    return jax.ShapeDtypeStruct(self.shape, self.dtype)
