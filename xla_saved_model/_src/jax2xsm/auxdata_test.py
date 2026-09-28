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

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from xla_saved_model._src.jax2xsm import auxdata


class RoundTripTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty", ()),
      ("none", (None,)),
      ("bool", (True,)),
      ("int", (7,)),
      ("float", (0.5,)),
      ("string", ("int8",)),
      ("numpy_dtype", (np.dtype("int8"),)),
      ("numpy_scalar_type", (np.int8,)),
      ("jax_numpy_scalar_type", (jnp.int8,)),
      ("several_values", ("int8", None, 3, np.dtype("float32"))),
      # The auxiliary data is whatever the flatten function returned, which
      # need not be a tuple at all, and a static field can itself hold a
      # sequence, such as a shape.
      ("not_a_tuple", None),
      ("bare_string", "int8"),
      ("a_list", ["int8", 3]),
      ("nested", (("a", ("b", 1)), [2, [3]], None)),
  )
  def test_round_trips(self, aux_data):
    serialized = auxdata.serialize(aux_data)
    self.assertIsInstance(serialized, bytes)

    restored = auxdata.deserialize(serialized)

    self.assertEqual(restored, aux_data)
    # A `PyTreeDef` holding `numpy.dtype("int8")` is not interchangeable with
    # one holding `"int8"` or `jax.numpy.int8`, even though all three compare
    # equal, so the spelling has to survive as well as the value.
    self.assertIs(type(restored), type(aux_data))

  @parameterized.named_parameters(
      ("string", "int8"),
      ("numpy_dtype", np.dtype("int8")),
      ("jax_numpy_scalar_type", jnp.int8),
  )
  def test_distinguishes_spellings_of_the_same_dtype(self, qtype):
    serialized = auxdata.serialize((qtype,))

    (restored,) = auxdata.deserialize(serialized)

    self.assertIs(type(restored), type(qtype))
    self.assertEqual(restored, qtype)

  def test_distinguishes_a_tuple_from_a_list(self):
    self.assertIsInstance(auxdata.deserialize(auxdata.serialize(("a",))), tuple)
    self.assertIsInstance(auxdata.deserialize(auxdata.serialize(["a"])), list)


class RejectionTest(parameterized.TestCase):

  def test_rejects_a_value_it_cannot_write_down(self):
    with self.assertRaisesRegex(ValueError, "Cannot serialize"):
      auxdata.serialize((object(),))

  def test_rejects_a_namedtuple(self):
    # A namedtuple is a tuple, but rebuilding it as one would change the
    # structure it belongs to, so it is refused rather than flattened.
    class _Point(NamedTuple):
      x: int

    with self.assertRaisesRegex(ValueError, "Cannot serialize"):
      auxdata.serialize(_Point(1))

  @parameterized.named_parameters(
      ("generic", np.generic),
      ("number", np.number),
      ("integer", np.integer),
      ("floating", np.floating),
      ("flexible", np.flexible),
  )
  def test_rejects_an_abstract_numpy_scalar_type(self, scalar_type):
    # These have no dtype of their own, so they cannot be read back.
    with self.assertRaisesRegex(ValueError, "Cannot serialize"):
      auxdata.serialize((scalar_type,))

  @parameterized.named_parameters(
      ("not_json", b"{"),
      ("not_utf8", b"\xff"),
      ("not_a_list", b'{"qtype": "int8"}'),
      ("not_a_pair", b'["str"]'),
      ("unknown_kind", b'["complex", "1+2j"]'),
      ("unknown_jax_scalar_type", b'["jax.numpy.scalar_type", "int9"]'),
      ("elements_are_not_a_list", b'["tuple", "int8"]'),
      ("kind_is_not_a_string", b'[["int"], 1]'),
      ("none_with_a_payload", b'["none", 1]'),
      ("int_holding_a_string", b'["int", "1"]'),
      ("int_holding_a_bool", b'["int", true]'),
      ("bool_holding_an_int", b'["bool", 1]'),
      ("str_holding_an_int", b'["str", 1]'),
      ("unknown_numpy_dtype", b'["numpy.dtype", "int9"]'),
      ("numpy_dtype_not_a_string", b'["numpy.dtype", 8]'),
      ("unknown_numpy_scalar_type", b'["numpy.scalar_type", "int9"]'),
      ("jax_scalar_type_not_a_string", b'["jax.numpy.scalar_type", 8]'),
      # Other attributes of `jax.numpy` are not scalar types.
      ("jax_numpy_function", b'["jax.numpy.scalar_type", "zeros"]'),
      ("jax_numpy_module", b'["jax.numpy.scalar_type", "linalg"]'),
  )
  def test_rejects_malformed_bytes(self, serialized):
    with self.assertRaises(ValueError):
      auxdata.deserialize(serialized)


if __name__ == "__main__":
  absltest.main()
