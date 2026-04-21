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

"""Utility functions for Neptune model export."""

from typing import Any

import jax
import jaxtyping
import numpy as np

from neptune_model._src.core import shlo_type


def to_jax_spec(tree: jaxtyping.PyTree) -> jaxtyping.PyTree:
  """Converts a PyTree with tensor-like leaves to one with ShapeDtypeStruct leaves.

  Args:
    tree: A PyTree where leaves have `shape` and `dtype` attributes.

  Returns:
    A PyTree with the same structure as the input, but with leaves replaced by
    jax.ShapeDtypeStruct instances.

  Raises:
    ValueError: If any leaf node in the tree lacks `shape` or `dtype`
    attributes.
  """

  def _convert_leaf(x: Any) -> jax.ShapeDtypeStruct:
    try:
      if isinstance(x, shlo_type.ShloTensorSpec):
        return jax.ShapeDtypeStruct(
            x.shape, shlo_type.shlo_dtype_to_np_dtype(x.dtype)
        )
      return jax.ShapeDtypeStruct(x.shape, x.dtype)
    except Exception as err:
      raise ValueError(
          f'Failed to convert {x!r} to jax.ShapeDtypeStruct'
      ) from err

  return jax.tree_util.tree_map(_convert_leaf, tree)


def to_shlo_tensor_spec(
    tree: jaxtyping.PyTree,
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Converts leafs of a PyTree of shape/dtype objects to ShloTensorSpec.

  Args:
    tree: A PyTree where leaves have shape/dtype attributes.

  Returns:
    A PyTree with the same structure as the input, but with leaves converted to
    ShloTensorSpec.

  Raises:
    ValueError: If any leaf node in the tree cannot be converted to
      ShloTensorSpec.
  """

  def _convert_leaf(x: Any) -> shlo_type.ShloTensorSpec:
    if isinstance(x, shlo_type.ShloTensorSpec):
      return x
    try:
      return shlo_type.ShloTensorSpec(
          x.shape, shlo_type.np_dtype_to_shlo_dtype(np.dtype(x.dtype))
      )
    except Exception as err:
      raise ValueError(f'Failed to convert {x!r} to ShloTensorSpec') from err

  return jax.tree_util.tree_map(_convert_leaf, tree)
