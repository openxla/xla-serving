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

from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from data.processor.proto import transform_config_pb2
from data.processor.python import shape_dtype_struct

ShapeDtypeStruct = shape_dtype_struct.ShapeDtypeStruct


class ShapeDtypeStructTest(absltest.TestCase):

  def test_normalizes_shape_and_dtype(self):
    s = ShapeDtypeStruct([np.int64(2), None, 3], 'int32')
    self.assertEqual(s.shape, (2, None, 3))
    self.assertIs(type(s.shape[0]), int)
    self.assertEqual(s.dtype, np.dtype(np.int32))
    self.assertIsInstance(s.dtype, np.dtype)
    self.assertEqual(ShapeDtypeStruct((), jnp.float32).dtype, np.float32)
    self.assertEqual(ShapeDtypeStruct((), 'object').dtype, np.dtype('O'))

  def test_equality_and_hash(self):
    a = ShapeDtypeStruct((None, 4), np.int32)
    b = ShapeDtypeStruct([None, np.int32(4)], 'int32')
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))
    self.assertNotEqual(a, ShapeDtypeStruct((1, 4), np.int32))
    self.assertNotEqual(a, ShapeDtypeStruct((None, 4), np.int64))

  def test_is_a_pytree_leaf(self):
    s = ShapeDtypeStruct((None,), np.int32)
    self.assertEqual(jax.tree_util.tree_leaves({'a': s}), [s])

  def test_rejects_negative_size(self):
    with self.assertRaisesRegex(
        ValueError, 'dimension 0 is -1.*Use None for an unknown dimension'
    ):
      ShapeDtypeStruct((-1, 4), np.int32)
    with self.assertRaisesRegex(ValueError, 'dimension 1 is -2'):
      ShapeDtypeStruct((1, np.int32(-2)), np.int32)

  def test_rejects_non_int_dimension(self):
    # Typed Any: these are the wrong types on purpose.
    bad_dims: list[Any] = [True, np.bool_(False), 2.0, 'batch']
    for d in bad_dims:
      with self.subTest(d=d):
        with self.assertRaisesRegex(TypeError, 'must be an int or None'):
          ShapeDtypeStruct((d,), np.int32)
    with self.assertRaisesRegex(TypeError, 'must be a sequence'):
      no_shape: Any = None
      ShapeDtypeStruct(no_shape, np.int32)

  def test_to_jax(self):
    self.assertEqual(
        ShapeDtypeStruct((2, 3), np.int32).to_jax(),
        jax.ShapeDtypeStruct((2, 3), jnp.int32),
    )
    with self.assertRaisesRegex(ValueError, 'unknown dimension'):
      ShapeDtypeStruct((None, 3), np.int32).to_jax()

  def test_to_partial_shape(self):
    self.assertEqual(
        shape_dtype_struct.to_partial_shape([None, 0, np.int64(3)]),
        (None, 0, 3),
    )
    self.assertEqual(shape_dtype_struct.to_partial_shape(()), ())
    # A dim that isn't an int, such as a JAX symbolic dim, is unknown.
    (batch,) = jax.export.symbolic_shape('b')
    self.assertEqual(
        shape_dtype_struct.to_partial_shape((batch, 'b', 4)), (None, None, 4)
    )
    with self.assertRaisesRegex(
        ValueError, 'Dimension 1 is -1.*Use None for an unknown dimension'
    ):
      shape_dtype_struct.to_partial_shape((2, -1))

  def test_tree_proto_roundtrip_and_leaf_iter(self):
    tree = {
        'a': ShapeDtypeStruct((None, 0, 4), np.int32),
        'b': [jax.ShapeDtypeStruct((2,), jnp.float32), None],
        'c': (ShapeDtypeStruct((), np.int64),),
    }
    proto = shape_dtype_struct.tree_to_proto(tree)
    self.assertEqual(
        shape_dtype_struct.proto_to_tree(proto),
        {
            'a': ShapeDtypeStruct((None, 0, 4), np.int32),
            'b': [ShapeDtypeStruct((2,), np.float32), None],
            'c': (ShapeDtypeStruct((), np.int64),),
        },
    )
    self.assertEqual(
        shape_dtype_struct.proto_to_tree(proto, iter(['x', 'y', 'z'])),
        {'a': 'x', 'b': ['y', None], 'c': ('z',)},
    )
    with self.assertRaisesRegex(ValueError, 'Dimension 0 is -1'):
      shape_dtype_struct.tree_to_proto(jax.ShapeDtypeStruct((-1, 4), jnp.int32))

    class StringDtypeLeaf:
      shape = (2,)
      dtype = 'int32'

    self.assertEqual(
        shape_dtype_struct.proto_to_tree(
            shape_dtype_struct.tree_to_proto(StringDtypeLeaf())
        ),
        ShapeDtypeStruct((2,), np.int32),
    )
    with self.assertRaisesRegex(ValueError, 'Unknown node type'):
      shape_dtype_struct.proto_to_tree(transform_config_pb2.PyTreeProto())

  def test_replace_deprecated_negative_one_shapes(self):
    with self.assertWarnsRegex(
        DeprecationWarning, 'TestCaller.*deprecated.*use None instead'
    ) as cm:
      replaced = shape_dtype_struct._replace_deprecated_negative_one_shapes(
          [[-1, 4], (np.int64(-1), None, 2)], 'TestCaller'
      )
    self.assertLen(cm.warnings, 1)
    self.assertEqual(replaced, [(None, 4), (None, None, 2)])
    # Other negative ints and booleans are passed through untouched.
    self.assertEqual(
        shape_dtype_struct._replace_deprecated_negative_one_shapes(
            [(-2, 4), (True, False)], 'TestCaller'
        ),
        [(-2, 4), (True, False)],
    )


if __name__ == '__main__':
  absltest.main()
