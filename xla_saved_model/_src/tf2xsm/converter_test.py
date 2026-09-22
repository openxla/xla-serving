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

import collections
import dataclasses
import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import tensorflow as tf

from xla_saved_model._src.core import test_utils
from xla_saved_model._src.tf2xsm import converter
from tensorflow.core.protobuf import saved_model_pb2  # pylint: disable=g-direct-tensorflow-import

_T1 = tf.TensorSpec((2, 3), tf.float32)
_T2 = tf.TensorSpec((4, 5), tf.float64)
_T3 = tf.TensorSpec((6,), tf.float32)


_NamedOutputs = collections.namedtuple('_NamedOutputs', ['alpha', 'beta'])


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
        # Named tuple children are keyed by field name, not by index, so the
        # outputs are named after the fields rather than 'output_N'.
        'named_tuple',
        _NamedOutputs(alpha=_T1, beta=_T2),
        {'alpha': _T1, 'beta': _T2},
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


class Tf2XsmConverterTest(
    tf.test.TestCase,
    test_utils.XsmTestCase,
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

  def test_output_signature_preserves_tracked_dict_structure(self):
    class _Scaler(tf.Module):

      def __init__(self):
        super().__init__()
        # TF replaces dicts assigned to a trackable with `_DictWrapper`, and
        # functions built from them return `_DictWrapper` too.
        self.scales = {'a': 2.0, 'b': 3.0}

      @tf.function(autograph=False)
      def __call__(self, x):
        return tf.nest.map_structure(lambda scale: x * scale, self.scales)

    # `tf.function` holds only a weak reference to the bound instance, so the
    # module has to outlive tracing.
    scaler = _Scaler()
    cf = scaler.__call__.get_concrete_function(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    output_signature = converter.get_output_signature(cf)

    self.assertEqual(
        jax.tree_util.tree_structure(output_signature),
        jax.tree_util.tree_structure({'a': 0, 'b': 0}),
    )

  def test_output_signature_preserves_tracked_list_structure(self):
    class _Splitter(tf.Module):

      def __init__(self):
        super().__init__()
        # TF replaces lists assigned to a trackable with `ListWrapper`, the
        # sequence counterpart of `_DictWrapper`.
        self.scales = [2.0, 3.0]

      @tf.function(autograph=False)
      def __call__(self, x):
        return tf.nest.map_structure(lambda scale: x * scale, self.scales)

    splitter = _Splitter()
    cf = splitter.__call__.get_concrete_function(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    output_signature = converter.get_output_signature(cf)

    self.assertEqual(
        jax.tree_util.tree_structure(output_signature),
        jax.tree_util.tree_structure([0, 0]),
    )
    self.assertEqual(
        [spec.name for spec in output_signature],
        ['output_0', 'output_1'],
    )

  def test_output_signature_names_named_tuple_fields_like_tf(self):
    # TF names the SavedModel outputs of a named-tuple-returning function after
    # the fields ('alpha', 'beta'), not 'output_0'/'output_1' as it does for a
    # plain tuple or list.
    @tf.function(autograph=False)
    def f(x):
      return _NamedOutputs(alpha=x + 1.0, beta=x + 2.0)

    cf = f.get_concrete_function(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    output_signature = converter.get_output_signature(cf)

    self.assertIsInstance(output_signature, _NamedOutputs)
    self.assertEqual(output_signature.alpha.name, 'alpha')
    self.assertEqual(output_signature.beta.name, 'beta')

  @parameterized.named_parameters(
      dict(testcase_name='_single', get_return_value=lambda x: x + 1.0),
      dict(testcase_name='_tuple', get_return_value=lambda x: (x + 1.0, x)),
      dict(testcase_name='_list', get_return_value=lambda x: [x + 1.0, x]),
      dict(
          testcase_name='_dict',
          get_return_value=lambda x: {'p': x + 1.0, 'q': x},
      ),
      dict(
          testcase_name='_named_tuple',
          get_return_value=lambda x: _NamedOutputs(alpha=x + 1.0, beta=x),
      ),
  )
  def test_output_names_match_tf_saved_model(self, get_return_value):
    # `get_output_signature` promises names that match the ones TF writes into
    # the SavedModel, so anchor them against a real TF SavedModel rather than
    # against hardcoded strings. Composite tensors are excluded: TF exposes one
    # signature key per composite while XSM has no composite type and needs a
    # distinct name per component (see `test_composite_tensor_outputs`).
    @tf.function(autograph=False)
    def f(x):
      return get_return_value(x)

    cf = f.get_concrete_function(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    # The function has to be owned by a module: saving a bare `tf.function`
    # with `signatures` makes TF attach the signature map to the function
    # itself and then fail to serialize it.
    module = tf.Module()
    module.serve = f
    path = self.create_tempdir().full_path
    tf.saved_model.save(module, path, signatures={'serving_default': cf})
    tf_output_names = tf.saved_model.load(path).signatures[
        'serving_default'
    ].structured_outputs.keys()

    output_signature = converter.get_output_signature(cf)

    self.assertCountEqual(
        [spec.name for spec in jax.tree.leaves(output_signature)],
        tf_output_names,
    )

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
            converter.XSM_TF_SAVED_MODEL_SUB_DIR,
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
        os.path.join(save_dir_path, converter.XSM_TF_SAVED_MODEL_SUB_DIR)
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

  def test_composite_tensor_inputs(self):
    @tf.function
    def tf_fn(ragged: tf.RaggedTensor, sparse: tf.SparseTensor):
      return tf.reduce_sum(ragged) + tf.reduce_sum(sparse.values)

    save_dir_path = self.create_tempdir().full_path
    ragged_spec = tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32)
    sparse_spec = tf.SparseTensorSpec(shape=(None,), dtype=tf.int32)

    prepared_fn = converter.prepare_function(
        'tf_fn',
        tf_fn,
        input_signature=[ragged_spec, sparse_spec],
    )

    converter.save_tf_functions(
        save_dir_path,
        {'tf_fn': prepared_fn},
    )

    loaded_tf_module = tf.saved_model.load(
        os.path.join(save_dir_path, converter.XSM_TF_SAVED_MODEL_SUB_DIR)
    )
    loaded_tf_fn = loaded_tf_module.signatures['tf_fn']

    ragged = tf.ragged.constant([[1, 2], [3]], dtype=tf.int32)
    sparse = tf.sparse.from_dense([4, 0, 5])

    # A loaded TF function cannot be called with composite tensors,
    # we need to pass their flattened components.
    outputs = loaded_tf_fn(
        ragged=ragged.values,
        ragged_1=ragged.row_splits,
        sparse=sparse.indices,
        sparse_1=sparse.values,
        sparse_2=sparse.dense_shape,
    )
    self.assertEqual(outputs['output_0'].numpy(), 15)

  def test_composite_tensor_outputs(self):
    @tf.function
    def tf_fn(x, y):
      ragged = tf.RaggedTensor.from_row_splits(values=x, row_splits=y)
      sparse = tf.sparse.from_dense([4, 0, 5])
      return ragged, sparse

    save_dir_path = self.create_tempdir().full_path
    prepared_fn = converter.prepare_function(
        'tf_fn',
        tf_fn,
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ],
    )

    converter.save_tf_functions(
        save_dir_path,
        {'tf_fn': prepared_fn},
    )

    loaded_tf_module = tf.saved_model.load(
        os.path.join(save_dir_path, converter.XSM_TF_SAVED_MODEL_SUB_DIR)
    )
    loaded_tf_fn = loaded_tf_module.signatures['tf_fn']
    outputs = loaded_tf_fn(
        x=tf.constant([1, 2, 3], tf.int32),
        y=tf.constant([0, 2, 3], tf.int32),
    )

    self.assertAllEqual(outputs['output_0'].numpy(), [1, 2, 3])
    self.assertAllEqual(outputs['output_0_0'].numpy(), [0, 2, 3])
    self.assertAllEqual(outputs['output_1'].numpy(), [[0], [2]])
    self.assertAllEqual(outputs['output_1_0'].numpy(), [4, 5])
    self.assertAllEqual(outputs['output_1_1'].numpy(), [3])

  def test_ragged_tensor_producer_consumer(self):
    @tf.function
    def producer(x, y):
      return tf.RaggedTensor.from_row_splits(values=x, row_splits=y)

    @tf.function
    def consumer(values, row_splits):
      # Rebuild the ragged tensor from its components
      ragged = tf.RaggedTensor.from_row_splits(
          values=values, row_splits=row_splits
      )
      return tf.reduce_sum(ragged)

    save_dir_path = self.create_tempdir().full_path
    prepared_producer = converter.prepare_function(
        'producer',
        producer,
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ],
    )
    prepared_consumer = converter.prepare_function(
        'consumer',
        consumer,
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ],
    )

    converter.save_tf_functions(
        save_dir_path,
        {'producer': prepared_producer, 'consumer': prepared_consumer},
    )

    loaded_tf_module = tf.saved_model.load(
        os.path.join(save_dir_path, converter.XSM_TF_SAVED_MODEL_SUB_DIR)
    )

    prod_out = loaded_tf_module.signatures['producer'](
        x=tf.constant([1, 2, 3], tf.int32),
        y=tf.constant([0, 2, 3], tf.int32),
    )

    cons_out = loaded_tf_module.signatures['consumer'](
        values=prod_out['output_0'],
        row_splits=prod_out['output_0_0'],
    )

    self.assertAllEqual(cons_out['output_0'].numpy(), 6)

  @parameterized.named_parameters(
      dict(
          testcase_name='_simple_with_none',
          spec_tree=(
              _T1,
              None,
              (_T2, None, [_T3]),
          ),
      ),
      dict(
          testcase_name='_composite_with_none',
          spec_tree=(
              _T1,
              None,
              (
                  tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
                  None,
                  [_T3],
              ),
          ),
      ),
  )
  def test_tree_dict_helpers(self, spec_tree):
    def _spec_to_tensor(spec):
      if isinstance(spec, tf.RaggedTensorSpec):
        return tf.ragged.constant([[1, 2], [3]], dtype=spec.dtype)
      return tf.zeros(spec.shape, spec.dtype)

    tensor_tree = jax.tree.map(_spec_to_tensor, spec_tree)

    leaves = jax.tree.leaves(tensor_tree)
    flat_len = sum(
        len(tf.nest.flatten(l, expand_composites=True)) for l in leaves
    )
    names = [f'n_{i}' for i in range(flat_len)]

    d = converter._tree_to_dict(tensor_tree, names)
    self.assertLen(d, flat_len)

    reconstructed = converter._dict_to_tree(d, names, spec_tree)

    tf.nest.assert_same_structure(reconstructed, tensor_tree)
    for r, t in zip(
        jax.tree.leaves(reconstructed), jax.tree.leaves(tensor_tree)
    ):
      self.assertAllEqual(r, t)

  def test_flat_input_signature_handles_none_and_composite(self):
    @tf.function
    def f(ragged: tf.RaggedTensor, x, y=None):
      del y
      return tf.reduce_sum(ragged) + x

    ragged_spec = tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32)
    cf = f.get_concrete_function(
        ragged_spec, tf.constant(1, dtype=tf.int32), None
    )

    names, leaves = converter._flat_input_signature(cf)

    # ragged has 2 flat components (values, row_splits).
    # x has 1 component.
    # y is None and skipped.
    # Total = 3.
    self.assertLen(leaves, 3)
    self.assertLen(names, 3)


if __name__ == '__main__':
  absltest.main()
