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

import os

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from .net.proto2.contrib.pyutil import compare
from google.protobuf import text_format
from neptune_model._src.core import persistence_lib
from neptune_model._src.tf2npt import converter
from neptune_model.protos import manifest_pb2
from neptune_model.protos import tf_concrete_function_handle_pb2


class Tf2NptE2eTest(
    tf.test.TestCase,
    parameterized.TestCase,
):

  def setUp(self):
    super().setUp()
    base_path = os.path.dirname(os.path.abspath(__file__))
    self._testdata_dir = os.path.join(base_path, 'testdata')

  def _get_testdata_path(self, filename: str) -> str:
    """Returns the full path to a file in the testdata directory."""
    return os.path.join(self._testdata_dir, filename)

  def test_e2e(self):
    var1 = tf.Variable(100.0)
    var2 = tf.Variable(200.0)
    vocab = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=['v', 'w'],
            values=[300.0, 400.0],
        ),
        default_value=10000.0,
    )

    @tf.function(
        input_signature=[tf.TensorSpec((2, 3, 4), tf.float32, name='a')]
    )
    def pre_processor_fn(a):
      return (
          a**2 + var1 + vocab.lookup(tf.constant('v')),
          tf.constant(67.0),
      )

    @tf.function(
        input_signature=[{
            'b': tf.TensorSpec((5, 6), tf.float32, name='b'),
            'c': tf.TensorSpec((7, 8), tf.float32, name='c'),
        }]
    )
    def post_processor_fn(inputs):
      return {
          'd': inputs['b'] ** 3 + var2,
          'e': inputs['c'] + vocab.lookup(tf.constant('w')),
      }

    pre_processor = converter.prepare_function('pre', pre_processor_fn)
    post_processor = converter.prepare_function('post', post_processor_fn)

    save_dir_path = self.create_tempdir().full_path

    npt_functions = {
        'pre': pre_processor.serializable_fn,
        'post': post_processor.serializable_fn,
    }

    persistence_lib.save(
        npt_functions,
        save_dir_path,
        persistence_lib.SaveOptions(
            version=2,
            supplementals=converter.save_tf_functions(
                save_dir_path,
                {
                    'tf_name_pre': pre_processor,
                    'tf_name_post': post_processor,
                },
                trackable_resources=(var1, var2, {'vocab': vocab}),
            ),
        ),
    )

    manifest_proto = persistence_lib.load(save_dir_path)
    expected_manifest_proto_path = self._get_testdata_path(
        'manifest_with_concrete_function.textproto'
    )

    with open(expected_manifest_proto_path, 'r') as f:
      expected_manifest_proto_text = f.read()

    expected_manifest_proto = text_format.Parse(
        expected_manifest_proto_text, manifest_pb2.Manifest()
    )
    compare.assertProto2Equal(
        self,
        expected_manifest_proto,
        manifest_proto,
    )

    pre_processor_proto = (
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle()
    )
    with open(os.path.join(save_dir_path, 'pre.pb'), 'rb') as f:
      pre_processor_proto.ParseFromString(f.read())

    compare.assertProto2Equal(
        self,
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(
            fn_name='pre',
            input_names=['a'],
            output_names=['output_0', 'output_1'],
        ),
        pre_processor_proto,
    )

    post_processor_proto = (
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle()
    )
    with open(os.path.join(save_dir_path, 'post.pb'), 'rb') as f:
      post_processor_proto.ParseFromString(f.read())

    compare.assertProto2Equal(
        self,
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(
            fn_name='post',
            input_names=['b', 'c'],
            output_names=['d', 'e'],
        ),
        post_processor_proto,
    )

    loaded_tf_module = tf.saved_model.load(
        os.path.join(save_dir_path, converter.NPT_TF_SAVED_MODEL_SUB_DIR)
    )

    pre_input = tf.ones(shape=(2, 3, 4), dtype=tf.float32) * 2
    pre_result = loaded_tf_module.signatures['tf_name_pre'](pre_input)

    tf.nest.assert_same_structure(
        pre_result, {'output_0': None, 'output_1': None}
    )
    self.assertAllClose(
        tf.nest.flatten(pre_result),
        pre_processor_fn(pre_input),
    )

    post_input = {
        'b': tf.ones(shape=(5, 6), dtype=tf.float32) * 2,
        'c': tf.ones(shape=(7, 8), dtype=tf.float32) * 3,
    }
    post_result = loaded_tf_module.signatures['tf_name_post'](
        b=post_input['b'], c=post_input['c']
    )

    tf.nest.assert_same_structure(post_result, {'d': None, 'e': None})
    self.assertAllClose(
        post_result,
        post_processor_fn(post_input),
    )


if __name__ == '__main__':
  absltest.main()
