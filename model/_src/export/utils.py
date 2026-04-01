"""Utility functions for Neptune model export."""

from collections.abc import Mapping

import jax
import jaxtyping

from neptune_model._src.export import jax_model
from neptune_model._src.jax2npt import typing


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
  try:
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree
    )
  except AttributeError as e:
    raise ValueError(
        'Failed to convert tree to JAX spec. All leaf nodes must have "shape"'
        ' and "dtype" attributes.'
    ) from e


def get_apply_fn(
    model: jax_model.JaxModel,
    method_key: str | None,
) -> typing.ApplyFn:
  """Retrieves the apply_fn from a JaxModel.

  If model.apply_fn is a single callable, it is returned directly. If it is
  a mapping, the function associated with `method_key` is returned.

  Args:
    model: The JaxModel instance.
    method_key: The key to use if model.apply_fn is a mapping.

  Returns:
    The selected apply_fn.

  Raises:
    ValueError: If model.apply_fn is a mapping but method_key is None or if
    the method_key is not found in the mapping.
  """
  if isinstance(model.apply_fn, Mapping):
    if method_key is None:
      raise ValueError(
          'method_key must be provided when apply_fn is a mapping.'
      )
    try:
      return model.apply_fn[method_key]
    except KeyError as e:
      raise ValueError(
          f'method_key "{method_key}" not found in apply_fn keys:'
          f' {list(model.apply_fn.keys())}'
      ) from e
  else:
    return model.apply_fn

