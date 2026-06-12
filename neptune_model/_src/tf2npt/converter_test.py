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

import dataclasses
import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import tensorflow as tf

from neptune_model._src.core import test_utils
from neptune_model._src.tf2npt import converter
from tensorflow.core.protobuf import saved_model_pb2  # pylint: disable=g-direct-tensorflow-import

_T1 = tf.TensorSpec((2, 3), tf.float32)
_T2 = tf.TensorSpec((4, 5), tf.float64)
_T3 = tf.TensorSpec((6,), tf.float32)


@dataclasses.dataclass
class _Dataclass:
  a: tf.TensorSpec
  b: tf.TensorSpec


@chex.dataclass
class _ChexDataclass:
  a: tf.TensorSpec
  b: tf.TensorSpec

  def tree_flatten(self):
    return (self.a, self.b), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


_INPUT_SIGNATURES = (
    (
        'positional_only',
        (
            (_T1, _T2, _T3),
            {},
        ),
        (
            (),
            {
                'args_0': _T1,
                'args_1': _T2,
                'args_2': _T3,
            },
        ),
    ),
    (
        'kwargs_only',
        ((), {'a': _T1, 'b': _T2}),
        ((), {'a': _T1, 'b': _T2}),
    ),
    (
        'positional_and_kwargs',
        ((_T1, _T2, _T3), {'a': _T1, 'b': _T2, 'c': _T3}),
        (
            (),
            {
                'args_0': _T1,
                'args_1': _T2,
                'args_2': _T3,
                'a': _T1,
                'b': _T2,
                'c': _T3,
            },
        ),
    ),
    (
        'dict_positional_and_kwargs',
        (({'a': _T1, 'b': _T2},), {'c': _T3}),
        ((), {'args_0': _T1, 'args_0_1': _T2, 'c': _T3}),
    ),
    (
        'registered_dataclass',
        ((_ChexDataclass(a=_T1, b=_T2),), {}),
        (
            (),
            {'args_0': _T1, 'args_0_1': _T2},
        ),
    ),
)
_OUTPUT_SIGNATURES = (
    (
        'tuple',
        (_T1, _T2, _T3),
        {'output_0': _T1, 'output_1': _T2, 'output_2': _T3},
    ),
    (
        'scalar',
        _T1,
        {'output_0': _T1},
    ),
    (
        'tuple_one_element',
        (_T1,),
        {'output_0': _T1},
    ),
    (
        'list',
        [_T2, _T1],
        {'output_0': _T2, 'output_1': _T1},
    ),
    (
        'dict',
        {'a': _T1, 'b': _T2},
        {'a': _T1, 'b': _T2},
    ),
    (
        'registered_dataclass',
        (_ChexDataclass(a=_T1, b=_T2)),
        {'a': _T1, 'b': _T2},
    ),
    (
        'nested_dict',
        {'a': {'b': _T1, 'c': _T2}, 'd': {'e': _T3}},
        {'a.b': _T1, 'a.c': _T2, 'd.e': _T3},
    ),
    (
        'mixed_types',
        (_T1, _T2, {'a': _T2}, {'b': [_T1, {'c': _T3}]}),
        {
            'output_0': _T1,
            'output_1': _T2,
            'output_2.a': _T2,
            'output_3.b.output_0': _T1,
            'output_3.b.output_1.c': _T3,
        },
    ),
)


class Tf2NptConverterTest(
    tf.test.TestCase,
    test_utils.NptTestCase,
):

  @parameterized.named_parameters(
      list(
          dict(
              testcase_name=f'_in_{input_case}_out_{output_case}',
              old_input_sig=old_input_sig,
              new_input_sig=new_input_sig,
              old_output_sig=old_output_sig,
              new_output_sig=new_output_sig,
          )
          for (
              (input_case, old_input_sig, new_input_sig),
              (output_case, old_output_sig, new_output_sig),
          ) in itertools.product(_INPUT_SIGNATURES, _OUTPUT_SIGNATURES)
      )
  )
  def test_to_keyword_only_fn(
      self,
      old_input_sig,
      new_input_sig,
      old_output_sig,
      new_output_sig,
  ):

    @tf.function(autograph=False)
    def f(*args, **kwargs):
      del args, kwargs
      return jax.tree.map(
          lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype),
          old_output_sig,
      )

    args, kwargs = old_input_sig
    cf = f.get_concrete_function(*args, **kwargs)
    new_cf = converter._to_keyword_only_fn(cf)

    self.assertShapeDtypeEqual(
        new_cf.structured_input_signature,
        new_input_sig,
    )
    self.assertShapeDtypeEqual(
        converter.get_output_signature(new_cf),
        new_output_sig,
    )

  def test_to_keyword_only_fn_fails_with_unregistered_dataclass_input(self):
    @tf.function(autograph=False)
    def f(*args, **kwargs):
      del args, kwargs
      return ()

    args, kwargs = ((_Dataclass(a=_T1, b=_T2),), {})
    with self.assertRaises(TypeError):
      f.get_concrete_function(*args, **kwargs)

  def test_to_keyword_only_fn_fails_with_unregistered_dataclass_output(self):
    output_sig = _Dataclass(a=_T1, b=_T2)

    @tf.function(autograph=False)
    def f():
      return jax.tree.map(
          lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype),
          output_sig,
      )

    with self.assertRaises(AttributeError):
      f.get_concrete_function()

  @parameterized.named_parameters(
      dict(
          testcase_name='_allows_mixed_types',
          get_return_value=lambda x, y: (x, {'a': y}),
      ),
      dict(
          testcase_name='_allows_nested_dicts',
          get_return_value=lambda x, y: {
              'sc_features': {
                  'client_sequential': x,
                  'nested': {'client_platform': y},
              }
          },
      ),
  )
  def test_return_value_discrepancies_with_tfsm(
      self,
      get_return_value,
  ):
    # There is a subset of return values that are supported by NPT but not
    # by TF SavedModel, e.g. nested dictionaries.
    @tf.function(autograph=False)
    def f(x, y):
      return get_return_value(x, y)

    cf = f.get_concrete_function(_T1, _T2)

    converter.save_tf_functions(self.create_tempdir().full_path, {'f': cf})

    with self.assertRaisesRegex(ValueError, 'Got a non-Tensor value'):
      tf.saved_model.save(
          f,
          self.create_tempdir().full_path,
          signatures={'f': cf},
      )

  def test_save_polymorphic_function_with_default_args(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec((2, 3, 4), tf.float32),
            tf.TensorSpec((2, 3, 4), tf.float32),
            (tf.TensorSpec((), tf.float32), tf.TensorSpec((), tf.float32)),
        ]
    )
    def tf_fn(
        x,
        y=tf.ones(shape=(2, 3, 4), dtype=tf.float32),
        z=(tf.constant(67.0), tf.constant(41.0)),
    ):
      return {'result': 2 * x + y + z[0] + z[1]}

    save_dir_path = self.create_tempdir().full_path
    prepared_fn = converter.prepare_function(
        'tf_fn',
        tf_fn,
        input_signature=[
            tf.TensorSpec((2, 3, 4), tf.float32),
            tf.TensorSpec((2, 3, 4), tf.float32),
            (tf.TensorSpec((), tf.float32), tf.TensorSpec((), tf.float32)),
        ],
    )

    converter.save_tf_functions(
        save_dir_path,
        {'tf_fn': prepared_fn},
    )

    with open(
        os.path.join(
            save_dir_path,
            converter.NPT_TF_SAVED_MODEL_SUB_DIR,
            'saved_model.pb',
        ),
        'rb',
    ) as f:
      saved_model_proto = saved_model_pb2.SavedModel()
      saved_model_proto.ParseFromString(f.read())

    saved_defaults = (
        saved_model_proto.meta_graphs[0].signature_def['tf_fn'].defaults
    )
    self.assertAllEqual(
        tf.make_ndarray(saved_defaults['y']),
        tf.ones(shape=(2, 3, 4), dtype=tf.float32),
    )
    self.assertAllEqual(tf.make_ndarray(saved_defaults['z']), tf.constant(67.0))
    self.assertAllEqual(
        tf.make_ndarray(saved_defaults['z_1']), tf.constant(41.0)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_suppress_x64_outputs',
          suppress_x64_outputs=True,
      ),
      dict(
          testcase_name='_do_not_suppress_x64_outputs',
          suppress_x64_outputs=False,
      ),
  )
  def test_suppress_x64_outputs(self, suppress_x64_outputs):
    @tf.function
    def tf_fn(x, y):
      return tf.cast(x, tf.float64), tf.cast(y, tf.int64)

    save_dir_path = self.create_tempdir().full_path
    prepared_fn = converter.prepare_function(
        'tf_fn',
        tf_fn,
        input_signature=[
            tf.TensorSpec((2, 3), tf.float32),
            tf.TensorSpec((2, 3), tf.int32),
        ],
        suppress_x64_outputs=suppress_x64_outputs,
    )

    converter.save_tf_functions(
        save_dir_path,
        {'tf_fn': prepared_fn},
    )

    loaded_tf_module = tf.saved_model.load(
        os.path.join(save_dir_path, converter.NPT_TF_SAVED_MODEL_SUB_DIR)
    )
    loaded_tf_fn = loaded_tf_module.signatures['tf_fn']
    output = loaded_tf_fn(
        x=tf.ones((2, 3), tf.float32),
        y=tf.ones((2, 3), tf.int32),
    )
    self.assertEqual(
        output['output_0'].dtype,
        tf.float32 if suppress_x64_outputs else tf.float64,
    )
    self.assertEqual(
        output['output_1'].dtype,
        tf.int32 if suppress_x64_outputs else tf.int64,
    )

  def test_has_x64_outputs_handles_none(self):
    @tf.function
    def return_none():
      pass

    concrete_fn = return_none.get_concrete_function()
    self.assertFalse(converter._has_x64_outputs(concrete_fn))


if __name__ == '__main__':
  absltest.main()
