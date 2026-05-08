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

"""Utility functions for OBM test cases."""

from collections.abc import Callable
import operator
from typing import Any
from absl.testing import parameterized
import jax
import jaxtyping


def _shape_dtype_equal(tree1: Any, tree2: Any) -> bool:
  return tuple(tree1.shape) == tuple(tree2.shape) and tree1.dtype == tree2.dtype


class NptTestCase(parameterized.TestCase):  # pylint: disable=g-doc-args
  """Base class for OBM test cases.

  Provides utility methods for testing OBM code.
  """

  def assertTreesEqual(
      self,
      tree1: jaxtyping.PyTree,
      tree2: jaxtyping.PyTree,
      leaf_eq: Callable[[Any, Any], bool] = operator.eq,
  ):  # pylint: disable=invalid-name
    """Asserts that two pytrees are equal."""
    if jax.tree.structure(tree1) != jax.tree.structure(tree2):
      raise AssertionError(
          f'Trees have different structures\ntree1:\n{tree1},\ntree2:\n{tree2}'
      )

    eq_tree = jax.tree.map(leaf_eq, tree1, tree2)
    if not all(jax.tree.leaves(eq_tree)):
      raise AssertionError(
          f'Trees are not equal\ntree1:\n{tree1},\ntree2:\n{tree2}'
      )

  def assertShapeDtypeEqual(
      self,
      tree1: jaxtyping.PyTree,
      tree2: jaxtyping.PyTree,
  ):  # pylint: disable=invalid-name
    """Asserts that two pytrees have leaves with the same shape and dtype."""
    self.assertTreesEqual(tree1, tree2, leaf_eq=_shape_dtype_equal)
