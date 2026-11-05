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

"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import layout as jax_layout
import jax.numpy as jnp
import numpy as np
from xla_saved_model.protos import type_pb2
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.jax2xsm import utils
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


def _get_spec():
  return shlo_type.ShloTensorSpec(shape=(1,), dtype=shlo_type.ShloDType.f32)


class _CustomNode:

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def tree_flatten(self):
    return ((self.x, self.y), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


jax.tree_util.register_pytree_node_class(_CustomNode)


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (jax.float0, shlo_type.ShloDType.bool),
      (jax.numpy.bfloat16, shlo_type.ShloDType.bf16),
      (jax.numpy.bool_, shlo_type.ShloDType.bool),
      (jax.numpy.int4, shlo_type.ShloDType.i4),
      (jax.numpy.int8, shlo_type.ShloDType.i8),
      (jax.numpy.int16, shlo_type.ShloDType.i16),
      (jax.numpy.int32, shlo_type.ShloDType.i32),
      (jax.numpy.int64, shlo_type.ShloDType.i64),
      (jax.numpy.uint4, shlo_type.ShloDType.ui4),
      (jax.numpy.uint8, shlo_type.ShloDType.ui8),
      (jax.numpy.uint16, shlo_type.ShloDType.ui16),
      (jax.numpy.uint32, shlo_type.ShloDType.ui32),
      (jax.numpy.uint64, shlo_type.ShloDType.ui64),
      (jax.numpy.float16, shlo_type.ShloDType.f16),
      (jax.numpy.float32, shlo_type.ShloDType.f32),
      (jax.numpy.float64, shlo_type.ShloDType.f64),
      (jax.numpy.complex64, shlo_type.ShloDType.c64),
      (jax.numpy.complex128, shlo_type.ShloDType.c128),
  )
  def test_get_physical_dtype_returns_expected_shlo_dtype(
      self, jax_dtype, expected_shlo_dtype
  ):
    self.assertEqual(utils.get_physical_dtype(jax_dtype), expected_shlo_dtype)

  def test_aval_dtype(self):
    aval = jax.core.ShapedArray((1, 2), jax.numpy.float32)
    self.assertEqual(utils.aval_dtype(aval), jax.numpy.float32)

  def test_hlo_sharding_to_op_sharding_with_none(self):
    self.assertIsNone(utils.hlo_sharding_to_op_sharding(None))

  def test_hlo_sharding_to_op_sharding(self):
    if jax.device_count() < 8:
      raise ValueError('Test requires 8 devices.')
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), ('x', 'y')
    )
    spec = jax.sharding.PartitionSpec('x', 'y')
    named_sharding = jax.sharding.NamedSharding(mesh, spec)
    hlo_sharding = named_sharding._to_xla_hlo_sharding(2)

    op_sharding = utils.hlo_sharding_to_op_sharding(hlo_sharding)

    self.assertIsInstance(op_sharding, xla_data_pb2.OpSharding)
    self.assertEqual(
        op_sharding.SerializeToString(),
        hlo_sharding.to_proto().SerializeToString(),
    )

  @parameterized.named_parameters(
      (f'_{idx}', jax_shape, expected_shlo_shape)
      for idx, (
          jax_shape,
          expected_shlo_shape,
          _,
      ) in enumerate([
          ((2, None, '3', 'a'), [2, None, 3, None], [None, None, None, 'a']),
          ((2, None, '3', '*'), [2, None, 3, None], [None, None, None, '*']),
          (
              (2, None, '3', '...'),
              [2, None, 3, None],
              [None, None, None, '...'],
          ),
          ([2, None, 3, '4'], [2, None, 3, 4], None),
          ((), [], None),
          ([], [], None),
      ])
  )
  def test_to_shlo_shape(self, jax_shape, expected_shlo_shape):
    self.assertEqual(
        utils._to_shlo_shape(jax_shape),
        expected_shlo_shape,
    )

  def test_to_shlo_dtype(self):
    jax_dtype = np.dtype(jnp.int32)
    self.assertEqual(
        utils._to_shlo_dtype(jax_dtype),
        shlo_type.ShloDType.i32,
    )

  def test_to_shlo_dtype_f0(self):
    jax_dtype = jax.dtypes.float0
    self.assertEqual(
        utils._to_shlo_dtype(jax_dtype),
        shlo_type.ShloDType.bool,
    )

  @parameterized.named_parameters(
      ('_str', 'int32'),
      # np.int32 is not allowed. It needs to be np.dtype(np.int32).
      ('_np_int32', np.int32),
      # Similarly for jnp.int32. It needs to be np.dtype(jnp.int32).
      ('_jnp_int32', jnp.int32),
  )
  def test_to_shlo_dtype_wrong_type(self, jax_dtype):
    with self.assertRaisesRegex(
        TypeError, r'jax_dtype must be an instance of np\.dtype, but'
    ):
      utils._to_shlo_dtype(jax_dtype)

  @parameterized.named_parameters(
      dict(
          testcase_name='dict_and_list',
          tree={'a': [_get_spec(), _get_spec()], 'b': {'c': _get_spec()}},
          expected_names=['a.0', 'a.1', 'b.c'],
      ),
      dict(
          testcase_name='tuple_and_nested_dict',
          tree=(_get_spec(), {'x': _get_spec()}),
          expected_names=['0', '1.x'],
      ),
      dict(
          testcase_name='list_of_tuples',
          tree=[(_get_spec(),), (_get_spec(), _get_spec())],
          expected_names=['0.0', '1.0', '1.1'],
      ),
      dict(
          testcase_name='custom_node',
          tree={'node': _CustomNode(_get_spec(), _get_spec())},
          expected_names=['node.0', 'node.1'],
      ),
      dict(
          testcase_name='single_spec',
          tree=_get_spec(),
          expected_names=[''],
      ),
      dict(
          testcase_name='list_with_one_spec',
          tree=[_get_spec()],
          expected_names=['0'],
      ),
  )
  def test_name_leaf(self, tree, expected_names):
    def _name_leaf_wrapper(tree):
      return jax.tree_util.tree_map_with_path(utils._name_leaf, tree)

    named_tree = _name_leaf_wrapper(tree)
    leaves, treedef = jax.tree_util.tree_flatten(named_tree)
    with self.subTest('check_treedef'):
      self.assertEqual(treedef, jax.tree_util.tree_structure(tree))
    with self.subTest('check_leaf_count'):
      self.assertLen(leaves, len(expected_names))
    with self.subTest('check_leaves_names'):
      self.assertEqual([leaf.name for leaf in leaves], expected_names)

  def test_to_shlo_spec_tree_with_name_leaves(self):
    avals_tree = {
        'a': jax.core.ShapedArray((1,), jnp.float32),
        'b': [jax.core.ShapedArray((2,), jnp.int32)],
    }
    avals, tree_def = jax.tree_util.tree_flatten(avals_tree)
    shardings = [
        xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_assignment_dimensions=[1],
            iota_reshape_dims=[0],
            iota_transpose_perm=[0],
        ),
        xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_assignment_dimensions=[1],
            iota_reshape_dims=[0],
            iota_transpose_perm=[0],
        ),
    ]
    layouts = [
        jax_layout.Layout(major_to_minor=(0,)),
        jax_layout.Layout(major_to_minor=(0,)),
    ]

    shlo_tensor_specs = utils.to_shlo_spec_tree(
        avals, shardings, layouts, tree_def, name_leaves=True
    )
    with self.subTest('check_jax_tree_leaves_names'):
      self.assertIsInstance(shlo_tensor_specs, dict)
      self.assertCountEqual(shlo_tensor_specs.keys(), ['a', 'b'])
      self.assertEqual(shlo_tensor_specs['a'].name, 'a')
      self.assertEqual(
          shlo_tensor_specs['a'].sharding,
          shardings[0],
      )
      self.assertEqual(
          shlo_tensor_specs['a'].layout.minor_to_major,
          [0],
      )
      self.assertIsInstance(shlo_tensor_specs['b'], list)
      self.assertEqual(shlo_tensor_specs['b'][0].name, 'b.0')
      self.assertEqual(
          shlo_tensor_specs['b'][0].sharding,
          shardings[1],
      )
      self.assertEqual(
          shlo_tensor_specs['b'][0].layout.minor_to_major,
          [0],
      )
      print(shlo_tensor_specs)

  def test_apply_upper_bound_dims_copies_annotation(self):
    user_signature = {
        'bounded': shlo_type.ShloTensorSpec(
            shape=(8, 4), dtype=shlo_type.ShloDType.f32, upper_bound_dims=(0,)
        ),
        'plain': shlo_type.ShloTensorSpec(
            shape=(2,), dtype=shlo_type.ShloDType.i32
        ),
        # Non-ShloTensorSpec leaves (e.g. jax.ShapeDtypeStruct) are ignored.
        'jax': jax.ShapeDtypeStruct((3,), jnp.float32),
    }
    converted_signature = {
        'bounded': shlo_type.ShloTensorSpec(
            shape=(8, 4), dtype=shlo_type.ShloDType.f32, name='bounded'
        ),
        'plain': shlo_type.ShloTensorSpec(
            shape=(2,), dtype=shlo_type.ShloDType.i32, name='plain'
        ),
        'jax': shlo_type.ShloTensorSpec(
            shape=(3,), dtype=shlo_type.ShloDType.f32, name='jax'
        ),
    }

    result = utils.apply_upper_bound_dims(user_signature, converted_signature)

    self.assertEqual(result['bounded'].upper_bound_dims, (0,))
    # Other attributes of the converted spec are preserved.
    self.assertEqual(result['bounded'].name, 'bounded')
    self.assertIsNone(result['plain'].upper_bound_dims)
    self.assertIsNone(result['jax'].upper_bound_dims)
    # The converted signature is left untouched.
    self.assertIsNone(converted_signature['bounded'].upper_bound_dims)

  def test_apply_upper_bound_dims_is_identity_without_annotation(self):
    user_signature = {
        'x': shlo_type.ShloTensorSpec(
            shape=(8, 4), dtype=shlo_type.ShloDType.f32
        )
    }
    converted_signature = {
        'x': shlo_type.ShloTensorSpec(
            shape=(8, 4), dtype=shlo_type.ShloDType.f32
        )
    }
    self.assertIs(
        utils.apply_upper_bound_dims(user_signature, converted_signature),
        converted_signature,
    )

  def test_apply_upper_bound_dims_rejects_polymorphic_bound(self):
    user_signature = {
        'x': shlo_type.ShloTensorSpec(
            shape=(8, 4), dtype=shlo_type.ShloDType.f32, upper_bound_dims=(0,)
        )
    }
    # The exporter made the bounded dimension polymorphic.
    converted_signature = {
        'x': shlo_type.ShloTensorSpec(
            shape=(None, 4), dtype=shlo_type.ShloDType.f32
        )
    }
    with self.assertRaisesRegex(ValueError, 'cannot be made polymorphic'):
      utils.apply_upper_bound_dims(user_signature, converted_signature)

  def test_apply_upper_bound_dims_rejects_structure_mismatch(self):
    user_signature = {
        'x': shlo_type.ShloTensorSpec(
            shape=(8, 4), dtype=shlo_type.ShloDType.f32, upper_bound_dims=(0,)
        )
    }
    converted_signature = {
        'y': shlo_type.ShloTensorSpec(
            shape=(8, 4), dtype=shlo_type.ShloDType.f32
        )
    }
    with self.assertRaisesRegex(ValueError, 'different structures'):
      utils.apply_upper_bound_dims(user_signature, converted_signature)

  def test_sharding_to_placement(self):
    self.assertEqual(
        utils.sharding_to_placement(None),
        type_pb2.PLACEMENT_UNSPECIFIED,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))
    device_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x')
    )
    self.assertEqual(
        utils.sharding_to_placement(device_sharding),
        type_pb2.PLACEMENT_UNSPECIFIED,
    )
    host_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x'), memory_kind='pinned_host'
    )
    self.assertEqual(
        utils.sharding_to_placement(host_sharding),
        type_pb2.HOST,
    )


if __name__ == '__main__':
  absltest.main()
