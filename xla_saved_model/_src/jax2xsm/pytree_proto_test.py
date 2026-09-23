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

"""Tests for `pytree_proto`."""

import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax

from xla_saved_model.protos import jax_supplemental_pb2
from xla_saved_model._src.jax2xsm import pytree_proto


# The leaf fields are typed `Any` because these tests only ever care about the
# structure; they use plain scalars as stand-in leaves so that no accelerator is
# needed.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Registered:
  """A custom node registered for serialization, like `qwix.QArray`."""

  value: Any
  scale: Any
  label: str = dataclasses.field(metadata=dict(static=True), default="")


# The registry is process-global and rejects duplicates, so register once at
# import time rather than per test.
pytree_proto.register_pytree_node_serialization(
    _Registered,
    serialized_name="neptune.test._Registered",
    # The auxiliary data is the tuple of static fields, here `(label,)`.
    serialize_auxdata=lambda aux: aux[0].encode("utf-8"),
    deserialize_auxdata=lambda b: (b.decode("utf-8"),),
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Unregistered:
  """A custom node that was never registered for serialization."""

  value: Any


class _NotAPyTreeNode:
  """Registered for serialization but never registered as a PyTree node."""


pytree_proto.register_pytree_node_serialization(
    _NotAPyTreeNode,
    serialized_name="neptune.test._NotAPyTreeNode",
    serialize_auxdata=lambda aux: b"",
    deserialize_auxdata=lambda b: None,
)


def _treedef(tree):
  return jax.tree_util.tree_structure(tree)


class RegisterTest(parameterized.TestCase):

  def test_returns_the_type_so_it_can_be_used_as_a_decorator(self):
    class _Fresh:
      pass

    returned = pytree_proto.register_pytree_node_serialization(
        _Fresh,
        serialized_name="neptune.test._Fresh",
        serialize_auxdata=lambda aux: b"",
        deserialize_auxdata=lambda b: None,
    )

    self.assertIs(returned, _Fresh)

  def test_duplicate_type_raises(self):
    with self.assertRaisesRegex(ValueError, "Duplicate serialization"):
      pytree_proto.register_pytree_node_serialization(
          _Registered,
          serialized_name="neptune.test._RegisteredAgain",
          serialize_auxdata=lambda aux: b"",
          deserialize_auxdata=lambda b: None,
      )

  def test_duplicate_serialized_name_raises(self):
    class _Other:
      pass

    with self.assertRaisesRegex(ValueError, "Duplicate serialization"):
      pytree_proto.register_pytree_node_serialization(
          _Other,
          serialized_name="neptune.test._Registered",
          serialize_auxdata=lambda aux: b"",
          deserialize_auxdata=lambda b: None,
      )

  def test_builtin_container_raises(self):
    with self.assertRaisesRegex(ValueError, "represented by the proto"):
      pytree_proto.register_pytree_node_serialization(
          dict,
          serialized_name="neptune.test.dict",
          serialize_auxdata=lambda aux: b"",
          deserialize_auxdata=lambda b: None,
      )


class ContainsCustomNodeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("leaf", 1.0),
      ("none", None),
      ("tuple", (1.0, 2.0)),
      ("list", [1.0, 2.0]),
      ("dict", {"a": 1.0, "b": 2.0}),
      ("nested", {"a": (1.0, [2.0, None]), "b": {"c": 3.0}}),
      ("empty_containers", ((), [], {})),
  )
  def test_false_for_manifest_representable(self, tree):
    self.assertFalse(pytree_proto.contains_custom_node(_treedef(tree)))

  @parameterized.named_parameters(
      ("bare", _Registered(1.0, 2.0)),
      ("nested", {"a": [(_Registered(1.0, 2.0),)]}),
  )
  def test_true_for_custom_node(self, tree):
    self.assertTrue(pytree_proto.contains_custom_node(_treedef(tree)))


class RoundTripTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("leaf", 1.0),
      ("none", None),
      ("tuple", (1.0, 2.0)),
      ("list", [1.0, 2.0]),
      ("dict", {"a": 1.0, "b": 2.0}),
      ("nested", {"a": (1.0, [2.0, None]), "b": {"c": 3.0}}),
      ("empty_containers", ((), [], {})),
      ("custom", _Registered(1.0, 2.0)),
      ("custom_with_auxdata", _Registered(1.0, 2.0, "some_label")),
      ("custom_nested", {"a": [(_Registered(1.0, 2.0, "x"), 3.0)], "b": None}),
      ("custom_containing_none", _Registered(1.0, None)),
      ("args_and_kwargs", ((_Registered(1.0, 2.0), {"x": 3.0}), {})),
  )
  def test_round_trip_preserves_structure(self, tree):
    original = _treedef(tree)

    restored = pytree_proto.from_proto(pytree_proto.to_proto(original))

    self.assertEqual(restored, original)

  def test_restored_tree_holds_the_custom_node_type(self):
    # The point of the whole exercise: a caller unflattening with the restored
    # structure gets a `_Registered`, not the `list` the manifest implies.
    original = _treedef(_Registered(1.0, 2.0, "x"))

    restored = pytree_proto.from_proto(pytree_proto.to_proto(original))
    value = jax.tree_util.tree_unflatten(restored, ["p", "q"])

    self.assertIsInstance(value, _Registered)
    self.assertEqual(value.value, "p")
    self.assertEqual(value.scale, "q")
    self.assertEqual(value.label, "x")

  def test_dict_keys_round_trip_in_flattening_order(self):
    # JAX sorts dict keys when flattening; the proto must record that order so
    # keys stay attached to the right children.
    original = _treedef({"c": 1.0, "a": 2.0, "b": 3.0})

    proto = pytree_proto.to_proto(original)
    restored = pytree_proto.from_proto(proto)

    self.assertEqual(list(proto.children_names), ["a", "b", "c"])
    self.assertEqual(restored, original)


class ToProtoTest(parameterized.TestCase):

  def test_leaf_has_no_children(self):
    proto = pytree_proto.to_proto(_treedef(1.0))

    self.assertEqual(proto.kind, jax_supplemental_pb2.PyTreeDef.LEAF)
    self.assertEmpty(proto.children)

  def test_custom_node_records_registered_name_not_import_path(self):
    proto = pytree_proto.to_proto(_treedef(_Registered(1.0, 2.0, "x")))

    self.assertEqual(proto.kind, jax_supplemental_pb2.PyTreeDef.CUSTOM)
    self.assertEqual(proto.custom_name, "neptune.test._Registered")
    self.assertEqual(proto.custom_auxdata, b"x")

  def test_unregistered_custom_node_raises(self):
    with self.assertRaisesRegex(
        pytree_proto.UnsupportedPyTreeDefError, "not registered"
    ):
      pytree_proto.to_proto(_treedef(_Unregistered(1.0)))

  def test_non_string_dict_key_raises(self):
    with self.assertRaisesRegex(
        pytree_proto.UnsupportedPyTreeDefError, "string keys"
    ):
      pytree_proto.to_proto(_treedef({1: 2.0}))


class FromProtoTest(parameterized.TestCase):

  def test_unregistered_name_raises(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.CUSTOM,
        custom_name="neptune.test._NeverRegistered",
    )

    with self.assertRaisesRegex(ValueError, "not registered"):
      pytree_proto.from_proto(proto)

  def test_type_that_is_not_a_pytree_node_raises(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.CUSTOM,
        custom_name="neptune.test._NotAPyTreeNode",
    )

    with self.assertRaisesRegex(ValueError, "register_pytree_node"):
      pytree_proto.from_proto(proto)

  def test_leaf_with_children_raises(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.LEAF,
        children=[
            jax_supplemental_pb2.PyTreeDef(
                kind=jax_supplemental_pb2.PyTreeDef.LEAF
            )
        ],
    )

    with self.assertRaisesRegex(ValueError, "LEAF cannot have"):
      pytree_proto.from_proto(proto)

  def test_mismatched_dict_keys_raise(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.DICT,
        children=[
            jax_supplemental_pb2.PyTreeDef(
                kind=jax_supplemental_pb2.PyTreeDef.LEAF
            )
        ],
        children_names=["a", "b"],
    )

    with self.assertRaisesRegex(ValueError, "keys but"):
      pytree_proto.from_proto(proto)

  def test_unknown_kind_raises(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.KIND_UNSPECIFIED
    )

    with self.assertRaisesRegex(ValueError, "Unknown PyTreeDef kind"):
      pytree_proto.from_proto(proto)


if __name__ == "__main__":
  absltest.main()
