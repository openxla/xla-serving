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

from absl.testing import absltest
import tensorflow as tf

from neptune_model._src.core import shlo_type
from neptune_model._src.core import test_utils
from neptune_model._src.processors import tf_processor
from neptune_model._src.tf2npt import converter
from neptune_model.protos import tf_concrete_function_handle_pb2


class TfProcessorTest(test_utils.NptTestCase):

  def test_prepare_tf_processor(self):
    @tf.function
    def simple_tf_fn(x):
      return x * 2.0

    processor = tf_processor.TfProcessor(
        name='double',
        fn=simple_tf_fn,
    )

    input_signature = tf.TensorSpec(shape=[None, 4], dtype=tf.float32)
    prepared_processor = processor.prepare(input_signature)

    self.assertIsInstance(prepared_processor, tf_processor.PreparedTfProcessor)
    self.assertEqual(prepared_processor.name, 'double')
    self.assertIsNone(prepared_processor.weights_name)
    self.assertIsNotNone(prepared_processor.prepared_function)

    saveable_fn = prepared_processor.to_npt_saveable()
    self.assertEqual(
        saveable_fn.body.proto.mime_type,
        converter.TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE,
    )
    fn_handle = tf_concrete_function_handle_pb2.TfConcreteFunctionHandle()
    fn_handle.ParseFromString(saveable_fn.body.proto.inlined_bytes)
    self.assertEqual(fn_handle.fn_name, 'double')
    self.assertEqual(fn_handle.input_names, ['x'])
    self.assertEqual(fn_handle.output_names, ['output_0'])

    self.assertShapeDtypeEqual(
        prepared_processor.input_signature,
        ((shlo_type.ShloTensorSpec([None, 4], shlo_type.ShloDType.f32),), {}),
    )
    self.assertShapeDtypeEqual(
        prepared_processor.output_signature,
        shlo_type.ShloTensorSpec([None, 4], shlo_type.ShloDType.f32),
    )

  def test_prepare_tf_processor_with_complex_signature(self):
    @tf.function
    def simple_tf_fn(x, *, y):
      return {'z': x * 2.0 + y, 'z1': [x + y]}

    processor = tf_processor.TfProcessor(
        name='tree',
        fn=simple_tf_fn,
    )

    input_signature = (
        (tf.TensorSpec(shape=[1], dtype=tf.float32),),
        {'y': tf.TensorSpec(shape=[1], dtype=tf.float32)},
    )
    prepared_processor = processor.prepare(input_signature)

    self.assertEqual(prepared_processor.name, 'tree')

    saveable_fn = prepared_processor.to_npt_saveable()
    self.assertEqual(
        saveable_fn.body.proto.mime_type,
        converter.TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE,
    )
    fn_handle = tf_concrete_function_handle_pb2.TfConcreteFunctionHandle()
    fn_handle.ParseFromString(saveable_fn.body.proto.inlined_bytes)
    self.assertEqual(fn_handle.fn_name, 'tree')
    self.assertEqual(fn_handle.input_names, ['x', 'y'])
    self.assertEqual(fn_handle.output_names, ['z', 'z1.output_0'])

    self.assertShapeDtypeEqual(
        prepared_processor.input_signature,
        (
            (shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32),),
            {'y': shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32)},
        ),
    )
    self.assertShapeDtypeEqual(
        prepared_processor.output_signature,
        {
            'z': shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32),
            'z1': [shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32)],
        },
    )

  def test_prepare_fails_with_empty_name(self):
    processor = tf_processor.TfProcessor(
        name='',
        fn=lambda x: x,
    )
    with self.assertRaisesRegex(
        ValueError, 'TfProcessor name must not be empty'
    ):
      processor.prepare(tf.TensorSpec(shape=[], dtype=tf.float32))


if __name__ == '__main__':
  absltest.main()
