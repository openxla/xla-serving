"""Tests for reconstructing JAX function from XSE serialization protos."""

import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import dataclasses
from typing import Any, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
import orbax.checkpoint as ocp

from xla_saved_model.protos import jax_supplemental_pb2
from xla_saved_model._src import core as xsm
from xla_saved_model._src.jax2xsm import constants
from xla_saved_model._src.jax2xsm import main_lib
from xla_saved_model._src.test_utils import jax_mnist
from xla_saved_model._src.xsm2jax import main_lib as xsm_to_jax

F0 = jax_supplemental_pb2.DTypeRefinement.f0
DType = xsm.ShloDType


# A custom PyTree node with a static field, standing in for a type like
# `qwix.QArray` that appears in the parameters of a real exported model. The
# manifest cannot represent it, so it exercises the structure recorded in the
# JAX supplemental.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Params:
  weight: Any
  bias: Any
  name: str = dataclasses.field(metadata=dict(static=True), default='')


def _to_shape_dtype_struct(x):
  return jax.ShapeDtypeStruct(x.shape, x.dtype)


def _get_shape_dtype_struct(jax_pytree):
  return jax.tree_util.tree_map(_to_shape_dtype_struct, jax_pytree)


def _make_shape_refm(
    refinements: Sequence[str] | None,
) -> jax_supplemental_pb2.ShapeRefinement | None:
  if refinements is None:
    return None
  return jax_supplemental_pb2.ShapeRefinement(
      dimension_sizes=(
          jax_supplemental_pb2.DimensionSizeRefinement(size=r)
          for r in refinements
      )
  )


class XsmToJaxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (f'_{idx}', shlo_dim_size, refinement, expected_result)
      for idx, (shlo_dim_size, refinement, expected_result) in enumerate((
          (None, None, None),
          (None, '3', 3),
          (None, 'a', 'a'),
          (None, 'None', 'None'),
          (3, None, 3),
          (3, '3', 3),
      ))
  )
  def test_refine_dim_size_ok(self, shlo_dim_size, refinement, expected_result):
    self.assertEqual(
        xsm_to_jax._refine_dim_size(shlo_dim_size, refinement),
        expected_result,
    )

  @parameterized.named_parameters(
      (f'_{shlo_dim_size}_{refinement}', shlo_dim_size, refinement)
      for shlo_dim_size, refinement in (
          (3, 'a'),
          (3, 'None'),
          (3, '4'),
      )
  )
  def test_refine_dim_size_error(self, shlo_dim_size, refinement):
    with self.assertRaisesRegex(
        ValueError, r'is not a valid refinement of the ShloDimSize'
    ):
      xsm_to_jax._refine_dim_size(shlo_dim_size, refinement)

  @parameterized.named_parameters(
      (f'_{idx}', shlo_shape, refinement, err_msg)
      for idx, (shlo_shape, refinement, err_msg) in enumerate((
          (None, [], r'Can not convert an unknown-rank shape to JAX'),
          ([], [None], None),
          ([2, 3], [], None),
          ([2, 3], [None], None),
          ([2, 3], [None] * 3, None),
      ))
  )
  def test_refine_shape_error(self, shlo_shape, refinement, err_msg):
    refinement = jax_supplemental_pb2.ShapeRefinement(
        dimension_sizes=[
            jax_supplemental_pb2.DimensionSizeRefinement(size=size)
            for size in refinement
        ]
    )
    if err_msg is None:
      err_msg = r'rank.* is not equal to.* rank'
    with self.assertRaisesRegex(ValueError, err_msg):
      xsm_to_jax._refine_shape(shlo_shape, refinement)

  @parameterized.named_parameters(
      (
          f'_{shlo_dtype}_{refinement}_{expected_result}',
          shlo_dtype,
          refinement,
          expected_result,
      )
      for shlo_dtype, refinement, expected_result in (
          (DType.f32, None, np.float32),
          (DType.bool, None, np.bool_),
          (DType.bool, F0, jax.dtypes.float0),
      )
  )
  def test_refine_dtype_ok(self, shlo_dtype, refinement, expected_result):
    self.assertEqual(
        xsm_to_jax._refine_dtype(shlo_dtype, refinement),
        expected_result,
    )

  def test_refine_dtype_error(self):
    with self.assertRaisesRegex(
        ValueError, r'is not a valid refinement of the ShloDType'
    ):
      xsm_to_jax._refine_dtype(DType.f32, F0)

  @parameterized.named_parameters(
      (  # pylint: disable=g-complex-comprehension
          f'_{idx}_{use_map}',
          shlo_tensor_specs,
          refinements,
          expected_result,
          use_map,
      )
      for use_map in (False, True)
      for idx, (shlo_tensor_specs, refinements, expected_result) in enumerate((
          ([], [], []),
          (
              [((None, 3), DType.f32), ((), DType.f32), ((), DType.bool)],
              [(('b', '3'), None), None, ((), F0)],
              [
                  (('b', 3), np.float32),
                  ((), np.float32),
                  ((), jax.dtypes.float0),
              ],
          ),
      ))
  )
  def test_refine_tensor_specs_ok(
      self, shlo_tensor_specs, refinements, expected_result, use_map
  ):
    if use_map:
      refinements_proto = jax_supplemental_pb2.ShapeDTypeRefinements(
          map=jax_supplemental_pb2.ShapeDTypeRefinementMap(
              idx_to_refinement={
                  idx: jax_supplemental_pb2.ShapeDTypeRefinement(
                      shape=_make_shape_refm(pair[0]), dtype=pair[1]
                  )
                  for idx, pair in enumerate(refinements)
                  if pair is not None
              }
          )
      )
    else:
      refinements_proto = jax_supplemental_pb2.ShapeDTypeRefinements(
          list=jax_supplemental_pb2.ShapeDTypeRefinementList(
              refinements=(
                  jax_supplemental_pb2.ShapeDTypeRefinement(
                      shape=_make_shape_refm(shape), dtype=dtype
                  )
                  for shape, dtype in (
                      (None, None) if r is None else r for r in refinements
                  )
              )
          )
      )
    self.assertEqual(
        xsm_to_jax._refine_tensor_specs(
            [
                xsm.ShloTensorSpec(shape=shape, dtype=dtype)
                for shape, dtype in shlo_tensor_specs
            ],
            refinements_proto,
        ),
        expected_result,
    )

  @parameterized.named_parameters(
      (f'_{shlo_tensor_specs}_{refinements}', shlo_tensor_specs, refinements)
      for shlo_tensor_specs, refinements in (
          ([((), DType.f32)] * 2, [(None, None)]),
          ([((), DType.f32)] * 2, [(None, None)] * 3),
      )
  )
  def test_refine_tensor_specs_error(self, shlo_tensor_specs, refinements):
    with self.assertRaisesRegex(
        ValueError,
        r'The number of refinements .*is not equal to the number of tensor'
        r' specs',
    ):
      xsm_to_jax._refine_tensor_specs(
          shlo_tensor_specs,
          jax_supplemental_pb2.ShapeDTypeRefinements(
              list=jax_supplemental_pb2.ShapeDTypeRefinementList(
                  refinements=(
                      jax_supplemental_pb2.ShapeDTypeRefinement(
                          shape=shape, dtype=dtype
                      )
                      for shape, dtype in refinements
                  )
              )
          ),
      )

  @parameterized.named_parameters(
      (f'_{idx}', shape)
      for idx, shape in enumerate((
          (None,),
          (2, None, 3),
          ('a', None),
          (None, 'b'),
          ('a', None, 3),
      ))
  )
  def test_to_jax_shape_error(self, shape):
    with self.assertRaisesRegex(
        ValueError, r'jax.core.ShapedArray does not allow `None` dimensions'
    ):
      xsm_to_jax._to_jax_shape(shape, jax.export.SymbolicScope())

  @parameterized.named_parameters(
      (f'_{idx}', shape_dtype)
      for idx, shape_dtype in enumerate((
          ((), np.float32),
          ((2,), np.float32),
          ((2, 3), np.float32),
          (('a',), np.float32),
          (('a', 'b'), np.float32),
          (('a', 'a'), np.float32),
          ((2, 'a', 3, 'b', 'a', 4), np.float32),
      ))
  )
  def test_to_jax_shaped_array_ok(self, shape_dtype):
    shape, dtype = shape_dtype
    result = xsm_to_jax._to_jax_shaped_array(
        shape_dtype, jax.export.SymbolicScope()
    )
    self.assertIsInstance(result, jax.core.ShapedArray)
    self.assertEqual(result.dtype, dtype)
    self.assertLen(result.shape, len(shape))
    for i, (dim, expected_dim) in enumerate(zip(result.shape, shape)):
      self.assertEqual(
          str(dim), str(expected_dim), f'The {i}-th dimension size is wrong.'
      )

  def test_xsm_fn_to_apply_fn_simple_jax_e2e(self):
    def jax_model_fn(params, x):
      return 2 * params * x

    jax_params = jnp.array(2.5)
    params_args_spec = _get_shape_dtype_struct(jax_params)
    input_args_spec = params_args_spec

    # Export the jax function to XSE.
    xsm_shlo_fn = main_lib.convert(
        jax.jit(jax_model_fn),
        (params_args_spec, input_args_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )

    xsm_module = dict()
    model_function_name = 'my_model_fn'
    xsm_module[model_function_name] = xsm_shlo_fn

    weights_name = 'my_weights'
    checkpoint_path = 'my_checkpoint/'
    xsm_module[weights_name] = main_lib.convert_path_to_value(
        checkpoint_path,
        mime_type='orbax_checkpoint',
    )

    save_dir_path = os.path.join(self.create_tempdir())

    xsm.save(
        xsm_module,
        save_dir_path,
        xsm.SaveOptions(version=2),
    )

    # Loading now.
    manifest_proto = xsm.load(save_dir_path)

    # Loads the model function
    loaded_xsm_function = manifest_proto.objects[model_function_name].function

    # and its supplemental.
    jax_supplemental_filename = (
        loaded_xsm_function.body.stable_hlo_body.supplemental_info[
            xsm.JAX_SPECIFIC_INFO
        ].file_system_location.string_path
    )
    jax_supplemental_proto = jax_supplemental_pb2.Function()
    with open(
        os.path.join(save_dir_path, jax_supplemental_filename), 'rb'
    ) as f:
      jax_supplemental_proto.ParseFromString(f.read())

    # Deserialize the jax function and compare results from the 2 jax functions.
    deserialized_apply_function = xsm_to_jax.xsm_fn_to_apply_fn(
        loaded_xsm_function,
        jax_supplemental_proto,
    )

    test_input = jnp.array(7.0)

    # Test calling with (params, inputs)
    result_from_deserialized_apply_fn = deserialized_apply_function(
        jax_params, test_input
    )
    result_from_original_jax_call = jax_model_fn(jax_params, test_input)

    self.assertEqual(
        result_from_deserialized_apply_fn, result_from_original_jax_call
    )
    self.assertEqual(result_from_deserialized_apply_fn, 35.0)
    self.assertIsInstance(
        deserialized_apply_function.exported, jax.export.Exported
    )

  def test_xsm_fn_to_apply_fn_preserves_custom_pytree_node(self):
    # The manifest has no way to express a custom PyTree node, so it records
    # one as a plain list of its leaves. Unless the real structure is carried
    # in the JAX supplemental, the restored function expects that list and
    # rejects a call made with the node itself, even though every leaf matches.
    def jax_model_fn(params, x):
      return params.weight * x + params.bias

    jax_params = _Params(
        weight=jnp.array(2.5), bias=jnp.array(1.0), name='my_layer'
    )
    test_input = jnp.array(7.0)

    xsm_shlo_fn = main_lib.convert(
        jax.jit(jax_model_fn),
        (
            _get_shape_dtype_struct(jax_params),
            _to_shape_dtype_struct(test_input),
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )

    model_function_name = 'my_model_fn'
    xsm_module = {model_function_name: xsm_shlo_fn}
    save_dir_path = os.path.join(self.create_tempdir())
    xsm.save(xsm_module, save_dir_path, xsm.SaveOptions(version=2))

    manifest_proto = xsm.load(save_dir_path)
    loaded_xsm_function = manifest_proto.objects[model_function_name].function
    jax_supplemental_filename = (
        loaded_xsm_function.body.stable_hlo_body.supplemental_info[
            xsm.JAX_SPECIFIC_INFO
        ].file_system_location.string_path
    )
    jax_supplemental_proto = jax_supplemental_pb2.Function()
    with open(
        os.path.join(save_dir_path, jax_supplemental_filename), 'rb'
    ) as f:
      jax_supplemental_proto.ParseFromString(f.read())

    deserialized_apply_function = xsm_to_jax.xsm_fn_to_apply_fn(
        loaded_xsm_function,
        jax_supplemental_proto,
    )

    # The restored calling convention still expects a `_Params`, so the call
    # below goes through. Without the structure it would expect a two element
    # list and raise.
    self.assertEqual(
        deserialized_apply_function.exported.in_tree,
        jax.tree_util.tree_structure(((jax_params, test_input), {})),
    )
    self.assertEqual(
        deserialized_apply_function(jax_params, test_input),
        jax_model_fn(jax_params, test_input),
    )

    # And that could only have come from the supplemental: the manifest has no
    # way to express it.
    self.assertTrue(jax_supplemental_proto.HasField('input_tree'))

  def test_xsm_fn_to_apply_fn_mnist(self):
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

    model = jax_mnist.JaxMnist()

    input_args_spec = jax.ShapeDtypeStruct((4, 28, 28, 1), jnp.float64)
    params = model.init(
        jax.random.PRNGKey(666),
        jnp.ones(shape=input_args_spec.shape, dtype=input_args_spec.dtype),
    )

    def get_mesh():
      devices = mesh_utils.create_device_mesh((2, 2, 2))
      return jax.sharding.Mesh(devices, ('b', 'x', 'y'))

    mesh = get_mesh()

    params_sharding_spec = jax.tree_util.tree_map(
        lambda _: NamedSharding(mesh, jax.sharding.PartitionSpec('y')), params
    )
    input_sharding_spec = NamedSharding(
        mesh, PartitionSpec('b', 'x', None, None)
    )

    model_apply_fn = jax.jit(
        model.apply,
        in_shardings=(
            params_sharding_spec,
            input_sharding_spec,
        ),
        out_shardings=NamedSharding(mesh, PartitionSpec('b', 'y')),
    )

    params_args_spec = _get_shape_dtype_struct(params)

    # Export the jax function to XSE.
    xsm_shlo_fn = main_lib.convert(
        model_apply_fn,
        (params_args_spec, input_args_spec),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
    )

    xsm_module = dict()

    model_function_name = 'mnist_forward_fn'
    xsm_module[model_function_name] = xsm_shlo_fn
    save_dir_path = os.path.join(self.create_tempdir())

    # Save the params to orbax checkpoint, which will be loaded later.
    checkpoint_path = 'my_checkpoint/'
    checkpoint_abs_path = os.path.join(save_dir_path, checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(checkpoint_abs_path, params)
    weights_name = 'my_weights'
    xsm_module[weights_name] = main_lib.convert_path_to_value(
        checkpoint_path,
        mime_type='orbax_checkpoint',
    )

    xsm.save(
        xsm_module,
        save_dir_path,
        xsm.SaveOptions(version=2),
    )

    # Loading now.
    manifest_proto = xsm.load(save_dir_path)

    # Loads the model function
    loaded_xsm_function = manifest_proto.objects[model_function_name].function

    # and its supplemental.
    jax_supplemental_filename = (
        loaded_xsm_function.body.stable_hlo_body.supplemental_info[
            xsm.JAX_SPECIFIC_INFO
        ].file_system_location.string_path
    )
    jax_supplemental_proto = jax_supplemental_pb2.Function()
    with open(
        os.path.join(save_dir_path, jax_supplemental_filename), 'rb'
    ) as f:
      jax_supplemental_proto.ParseFromString(f.read())

    # Deserialize the jax function and compare results from the 2 jax functions.
    deserialized_apply_function = xsm_to_jax.xsm_fn_to_apply_fn(
        loaded_xsm_function,
        jax_supplemental_proto,
    )

    # Restore/load the params from the saved orbax checkpoint,
    # this will be fed into the deserialized jax function only.
    loaded_checkpoint_path = manifest_proto.objects[
        weights_name
    ].value.external.data.file_system_location.string_path
    restored_params = checkpointer.restore(
        os.path.join(save_dir_path, loaded_checkpoint_path)
    )

    test_input_data = jax.device_put(
        jax.random.uniform(
            jax.random.PRNGKey(999), (4, 28, 28, 1), dtype=jnp.float64
        ),
        input_sharding_spec,
    )

    result_from_original_jax_call = model_apply_fn(params, test_input_data)
    result_from_deserialized_apply_fn = deserialized_apply_function(
        jax.device_put(restored_params, params_sharding_spec),
        test_input_data,
    )

    self.assertTrue(
        jnp.array_equal(
            result_from_deserialized_apply_fn,
            result_from_original_jax_call,
        )
    )

  def test_exported_apply_fn(self):
    def simple_fn(params, inputs):
      return inputs * params['w']

    exported = jax.export.export(jax.jit(simple_fn))(
        {'w': jax.core.ShapedArray((), jnp.float32)},
        jax.core.ShapedArray((2,), jnp.float32),
    )

    # Test ExportedApplyFn wrapper on an Exported object
    apply_fn = xsm_to_jax.ExportedApplyFn(exported)
    self.assertIsInstance(apply_fn, xsm_to_jax.ExportedApplyFn)
    self.assertIs(apply_fn.exported, exported)
    self.assertEqual(apply_fn.fun_name, exported.fun_name)
    self.assertEqual(apply_fn.in_tree, exported.in_tree)
    self.assertEqual(apply_fn.out_tree, exported.out_tree)

    # Test calling with (params, inputs)
    params = {'w': jnp.float32(2.0)}
    arg = jnp.array([1.0, 2.0], dtype=jnp.float32)
    self.assertTrue(
        jnp.array_equal(apply_fn(params, arg), jnp.array([2.0, 4.0]))
    )

    # Test repr and str
    self.assertIn('ExportedApplyFn', repr(apply_fn))
    self.assertEqual(str(apply_fn), str(exported))

  def test_exported_apply_fn_invalid_arg_count(self):
    # 1 argument function
    exported_1_arg = jax.export.export(jax.jit(lambda x: x * 2.0))(
        jax.core.ShapedArray((), jnp.float32)
    )
    with self.assertRaisesRegex(
        ValueError, 'taking exactly 2 arguments.*got 1'
    ):
      xsm_to_jax.ExportedApplyFn(exported_1_arg)

    # 3 arguments function
    exported_3_arg = jax.export.export(jax.jit(lambda a, b, c: a + b + c))(
        jax.core.ShapedArray((), jnp.float32),
        jax.core.ShapedArray((), jnp.float32),
        jax.core.ShapedArray((), jnp.float32),
    )
    with self.assertRaisesRegex(
        ValueError, 'taking exactly 2 arguments.*got 3'
    ):
      xsm_to_jax.ExportedApplyFn(exported_3_arg)

  def test_exported_apply_fn_invalid_type(self):
    with self.assertRaisesRegex(
        TypeError, 'expects a jax.export.Exported object, got str'
    ):
      xsm_to_jax.ExportedApplyFn('not an Exported object')

  def test_xsm_fn_to_apply_fn_with_tpu_io_optimizations_e2e(self):
    def jax_model_fn(params, inputs):
      spatial = jnp.einsum('nhwc,hwo->nhwo', inputs['img'], params.weight)
      return {
          'feat': spatial + params.bias,
          'norm': jnp.sum(spatial, axis=(1, 2, 3)),
      }

    jax_params = _Params(
        weight=(
            jnp.arange(3 * 5 * 4, dtype=jnp.float32).reshape((3, 5, 4)) * 0.1
        ),
        bias=jnp.array([0.5, -1.0, 1.5, 2.0], dtype=jnp.float32),
        name='conv_head',
    )
    test_inputs = {
        'img': (
            jnp.arange(2 * 3 * 5 * 2, dtype=jnp.float32).reshape((2, 3, 5, 2))
            - 5.0
        ),
    }

    xsm_shlo_fn = main_lib.convert(
        jax.jit(jax_model_fn),
        (
            _get_shape_dtype_struct(jax_params),
            _get_shape_dtype_struct(test_inputs),
        ),
        {},
        platforms=[xsm.manifest_pb2.Platform.CPU],
        model_param_names=['weight', 'bias'],
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
        max_transpose_channels=3,
    )

    save_dir_path = self.create_tempdir().full_path
    xsm.save(
        {'my_opt_fn': xsm_shlo_fn},
        save_dir_path,
        xsm.SaveOptions(version=2),
    )

    manifest_proto = xsm.load(save_dir_path)
    loaded_fn = manifest_proto.objects['my_opt_fn'].function
    supp_filename = (
        loaded_fn.body.stable_hlo_body.supplemental_info[xsm.JAX_SPECIFIC_INFO]
        .file_system_location.string_path
    )
    jax_supp_proto = jax_supplemental_pb2.Function()
    with open(os.path.join(save_dir_path, supp_filename), 'rb') as f:
      jax_supp_proto.ParseFromString(f.read())

    restored_fn = xsm_to_jax.xsm_fn_to_apply_fn(loaded_fn, jax_supp_proto)
    actual = restored_fn(jax_params, test_inputs)
    expected = jax_model_fn(jax_params, test_inputs)
    np.testing.assert_allclose(actual['feat'], expected['feat'], rtol=1e-5)
    np.testing.assert_allclose(actual['norm'], expected['norm'], rtol=1e-5)

    # Logical shape validation rejects mismatched caller shapes with identical
    # element counts (e.g. (6, 10) vs (2, 3, 5, 2)).
    wrong_shape_inputs = {
        'img': jnp.ones((6, 10), dtype=jnp.float32),
    }
    with self.assertRaisesRegex(ValueError, 'Input leaf shape mismatch'):
      restored_fn(jax_params, wrong_shape_inputs)

    # Mismatched PyTree structure raises ValueError rather than bypassing
    # `_in_adapters`.
    with self.assertRaisesRegex(ValueError, 'Input PyTree structure mismatch'):
      restored_fn(
          jax_params,
          {'wrong_key': jnp.ones((2, 3, 5, 2), dtype=jnp.float32)},
      )

    # Host `np.ndarray` inputs are transposed and flattened on host without
    # premature device placement prior to `Exported.call`.
    np_params = _Params(
        weight=np.asarray(jax_params.weight),
        bias=np.asarray(jax_params.bias),
        name='conv_head',
    )
    np_inputs = {'img': np.asarray(test_inputs['img'])}
    with mock.patch.object(
        jax.export.Exported,
        'call',
        autospec=True,
        side_effect=jax.export.Exported.call,
    ) as mock_call:
      np_actual = restored_fn(np_params, np_inputs)
    self.assertIsInstance(mock_call.call_args.args[2]['img'], np.ndarray)
    self.assertEqual(mock_call.call_args.args[2]['img'].shape, (60,))
    np.testing.assert_allclose(np_actual['feat'], expected['feat'], rtol=1e-5)
    np.testing.assert_allclose(np_actual['norm'], expected['norm'], rtol=1e-5)

  def test_exported_apply_fn_rejects_mismatched_adapter_lengths(self):
    exported = jax.export.export(jax.jit(lambda p, x: x + p))(
        jax.core.ShapedArray((2,), jnp.float32),
        jax.core.ShapedArray((2,), jnp.float32),
    )
    with self.assertRaisesRegex(ValueError, '`in_adapters` has 1 entries'):
      xsm_to_jax.ExportedApplyFn(exported, in_adapters=[((2,), None, (2,))])
    with self.assertRaisesRegex(
        ValueError, '`out_logical_shapes` has 2 entries'
    ):
      xsm_to_jax.ExportedApplyFn(exported, out_logical_shapes=[(2,), (2,)])


if __name__ == '__main__':
  absltest.main()
