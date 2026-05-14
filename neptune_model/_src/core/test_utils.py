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

import collections
from collections.abc import Callable, Iterable, Mapping
import copy
import operator
from typing import Any, cast

from absl.testing import parameterized
from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import text_format
import jax
import jaxtyping


def _shape_dtype_equal(tree1: Any, tree2: Any) -> bool:
  return tuple(tree1.shape) == tuple(tree2.shape) and tree1.dtype == tree2.dtype


def _clear_fields(msg: message.Message, fields: Iterable[str]) -> None:
  """Recursively clears fields matching dot-separated paths in a proto."""
  if not hasattr(msg, "ClearField"):
    return

  clear_map = collections.defaultdict(list)
  for field in fields:
    if not field:
      continue
    prefix, _, suffix = field.partition(".")
    if suffix:
      clear_map[prefix].append(suffix)
    else:
      msg.ClearField(prefix)

  for prefix, suffixes in clear_map.items():
    fd = msg.DESCRIPTOR.fields_by_name.get(prefix)
    if fd is None:
      continue

    # We only recurse if the field is a message/group descriptor
    if fd.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
      attr_val = getattr(msg, prefix)
      if isinstance(attr_val, Mapping):
        # Handle map fields
        for k in attr_val:
          _clear_fields(attr_val[k], suffixes)
      elif fd.is_repeated:
        # Handle repeated fields
        for entry in attr_val:
          _clear_fields(entry, suffixes)
      else:
        # Handle singular message fields
        if msg.HasField(prefix):
          _clear_fields(attr_val, suffixes)


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
          f"Trees have different structures\ntree1:\n{tree1},\ntree2:\n{tree2}"
      )

    eq_tree = jax.tree.map(leaf_eq, tree1, tree2)
    if not all(jax.tree.leaves(eq_tree)):
      raise AssertionError(
          f"Trees are not equal\ntree1:\n{tree1},\ntree2:\n{tree2}"
      )

  def assertShapeDtypeEqual(
      self,
      tree1: jaxtyping.PyTree,
      tree2: jaxtyping.PyTree,
  ):  # pylint: disable=invalid-name
    """Asserts that two pytrees have leaves with the same shape and dtype."""
    self.assertTreesEqual(tree1, tree2, leaf_eq=_shape_dtype_equal)

  def assertProtosEqual(
      self,
      expected: str | message.Message,
      actual: message.Message,
      *,
      ignored_fields: Iterable[str] | None = None,
  ):  # pylint: disable=invalid-name
    """Asserts that two protos are equal."""

    if isinstance(expected, str):
      expected_msg = actual.__class__()
      text_format.Parse(expected, expected_msg)
      parsed_expected = cast(message.Message, expected_msg)
    else:
      parsed_expected = copy.deepcopy(cast(message.Message, expected))

    parsed_actual = copy.deepcopy(actual)

    if ignored_fields:
      _clear_fields(parsed_expected, ignored_fields)
      _clear_fields(parsed_actual, ignored_fields)

    self.assertEqual(
        text_format.MessageToString(parsed_expected),
        text_format.MessageToString(parsed_actual),
    )

