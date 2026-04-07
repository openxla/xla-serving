"""Utility functions for Neptune model export."""

import jax
import jaxtyping


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

