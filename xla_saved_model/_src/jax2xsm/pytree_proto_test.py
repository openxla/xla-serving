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
import sys
import types
from typing import Any, NamedTuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax

from xla_saved_model.protos import jax_supplemental_pb2
from xla_saved_model._src.jax2xsm import auxdata
from xla_saved_model._src.jax2xsm import pytree_proto


# The leaf fields are typed `Any` because these tests only ever care about the
# structure; they use plain scalars as stand-in leaves so that no accelerator is
# needed.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Custom:
  """A custom node with a static field, like `qwix.QArray`."""

  value: Any
  scale: Any
  label: str = dataclasses.field(metadata=dict(static=True), default="")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _NoStaticFields:
  """A custom node whose auxiliary data is the empty tuple."""

  value: Any


@chex.dataclass
class _ChexPair:
  """A `chex.dataclass` node, whose auxiliary data is a tuple of field names."""

  first: Any
  second: Any


class _Point(NamedTuple):
  """A namedtuple, which JAX treats as a node without registration."""

  x: Any
  y: Any


def _treedef(tree):
  return jax.tree_util.tree_structure(tree)


class DefaultSerializedNameTest(parameterized.TestCase):

  def test_is_the_module_and_qualified_name(self):
    self.assertEqual(
        pytree_proto.default_serialized_name(_Custom),
        f"{__name__}._Custom",
    )

  def test_distinguishes_nested_classes_of_the_same_name(self):
    class _Outer:

      class _Inner:
        pass

    # `__name__` alone would collide with any other `_Inner`; `__qualname__`
    # keeps the enclosing scope, which is what makes the derived name usable
    # as an identity.
    self.assertEndsWith(
        pytree_proto.default_serialized_name(_Outer._Inner),
        "._Outer._Inner",
    )

  def test_is_unaffected_by_a_shorter_module_that_re_exports_the_type(self):
    # A library that re-exports its public API still records the module that
    # defines the type, so the name does not depend on what this process
    # happens to have imported.
    package = types.ModuleType("xla_saved_model_test_package")

    class _ReExported:
      pass

    _ReExported.__qualname__ = "_ReExported"
    _ReExported.__module__ = "xla_saved_model_test_package._src.impl"
    setattr(package, "_ReExported", _ReExported)
    self.enter_context(
        mock.patch.dict(sys.modules, {"xla_saved_model_test_package": package})
    )

    self.assertEqual(
        pytree_proto.default_serialized_name(_ReExported),
        "xla_saved_model_test_package._src.impl._ReExported",
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
      ("bare", _Custom(1.0, 2.0)),
      ("nested", {"a": [(_Custom(1.0, 2.0),)]}),
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
      ("custom", _Custom(1.0, 2.0)),
      ("custom_with_auxdata", _Custom(1.0, 2.0, "some_label")),
      ("custom_nested", {"a": [(_Custom(1.0, 2.0, "x"), 3.0)], "b": None}),
      ("custom_containing_none", _Custom(1.0, None)),
      ("args_and_kwargs", ((_Custom(1.0, 2.0), {"x": 3.0}), {})),
      ("no_static_fields", _NoStaticFields(1.0)),
      ("no_static_fields_nested", {"a": [(_NoStaticFields(1.0), 2.0)]}),
  )
  def test_round_trip_preserves_structure(self, tree):
    original = _treedef(tree)

    restored = pytree_proto.from_proto(pytree_proto.to_proto(original))

    self.assertEqual(restored, original)

  def test_restored_tree_holds_the_custom_node_type(self):
    # The point of the whole exercise: a caller unflattening with the restored
    # structure gets a `_Custom`, not the `list` the manifest implies.
    original = _treedef(_Custom(1.0, 2.0, "x"))

    restored = pytree_proto.from_proto(pytree_proto.to_proto(original))
    value = jax.tree_util.tree_unflatten(restored, ["p", "q"])

    self.assertIsInstance(value, _Custom)
    self.assertEqual(value.value, "p")
    self.assertEqual(value.scale, "q")
    self.assertEqual(value.label, "x")

  def test_chex_dataclass_round_trips(self):
    original = _treedef(_ChexPair(first=1.0, second=2.0))

    restored = pytree_proto.from_proto(pytree_proto.to_proto(original))
    value = jax.tree_util.tree_unflatten(restored, ["p", "q"])

    self.assertEqual(restored, original)
    self.assertIsInstance(value, _ChexPair)
    self.assertEqual(value.first, "p")
    self.assertEqual(value.second, "q")

  def test_namedtuple_round_trips(self):
    original = _treedef(_Point(1.0, 2.0))

    restored = pytree_proto.from_proto(pytree_proto.to_proto(original))

    self.assertEqual(restored, original)

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

  def test_custom_node_records_the_derived_name(self):
    proto = pytree_proto.to_proto(_treedef(_Custom(1.0, 2.0, "x")))

    self.assertEqual(proto.kind, jax_supplemental_pb2.PyTreeDef.CUSTOM)
    self.assertEqual(proto.custom_name, f"{__name__}._Custom")
    # The auxiliary data of a `register_dataclass` node is the tuple of its
    # static fields, here `("x",)`.
    self.assertEqual(auxdata.deserialize(proto.custom_auxdata), ("x",))

  def test_unserializable_auxdata_raises(self):
    @jax.tree_util.register_pytree_node_class
    class _OpaqueAuxData:

      def tree_flatten(self):
        return (1.0,), object()

      @classmethod
      def tree_unflatten(cls, aux_data, children):
        del aux_data, children
        return cls()

    with self.assertRaisesRegex(
        pytree_proto.UnsupportedPyTreeDefError, "cannot be serialized"
    ):
      pytree_proto.to_proto(_treedef(_OpaqueAuxData()))

  def test_non_string_dict_key_raises(self):
    with self.assertRaisesRegex(
        pytree_proto.UnsupportedPyTreeDefError, "string keys"
    ):
      pytree_proto.to_proto(_treedef({1: 2.0}))


class FromProtoTest(parameterized.TestCase):

  def test_name_no_imported_module_defines_raises(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.CUSTOM,
        custom_name="xla_saved_model.test._NeverDefined",
    )

    with self.assertRaisesRegex(
        pytree_proto.UnresolvedPyTreeNodeError, "not defined by any module"
    ):
      pytree_proto.from_proto(proto)

  def test_unresolved_node_error_is_a_value_error(self):
    # Callers that predate the distinct type still catch it.
    self.assertTrue(
        issubclass(pytree_proto.UnresolvedPyTreeNodeError, ValueError)
    )

  def test_resolves_a_name_against_an_imported_module(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.CUSTOM,
        custom_name=pytree_proto.default_serialized_name(_NoStaticFields),
        custom_auxdata=auxdata.serialize(()),
        children=[
            jax_supplemental_pb2.PyTreeDef(
                kind=jax_supplemental_pb2.PyTreeDef.LEAF
            )
        ],
    )

    self.assertEqual(
        pytree_proto.from_proto(proto), _treedef(_NoStaticFields(1.0))
    )

  def test_a_name_that_is_not_a_pytree_node_does_not_resolve(self):
    # This module defines `_treedef`, but it is a function, not a node type.
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.CUSTOM,
        custom_name=f"{__name__}._treedef",
    )

    with self.assertRaises(pytree_proto.UnresolvedPyTreeNodeError):
      pytree_proto.from_proto(proto)

  def test_resolution_consults_only_loaded_modules(self):
    name = pytree_proto.default_serialized_name(_NoStaticFields)

    # Nothing is loaded, so a name could only resolve by importing what it
    # says, which is exactly what must never happen. The helper is called
    # directly, and caught by hand, because the assertion machinery would
    # itself need modules.
    raised_with_nothing_loaded = False
    with mock.patch.dict(sys.modules, clear=True):
      try:
        pytree_proto._check_pytree_node_imported(name)
      except pytree_proto.UnresolvedPyTreeNodeError:
        raised_with_nothing_loaded = True

    self.assertTrue(raised_with_nothing_loaded)
    self.assertIs(
        pytree_proto._check_pytree_node_imported(name), _NoStaticFields
    )

  def test_malformed_auxdata_raises(self):
    proto = jax_supplemental_pb2.PyTreeDef(
        kind=jax_supplemental_pb2.PyTreeDef.CUSTOM,
        custom_name=pytree_proto.default_serialized_name(_NoStaticFields),
        custom_auxdata=b"not json",
    )

    with self.assertRaisesRegex(ValueError, "Could not read the auxiliary"):
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
