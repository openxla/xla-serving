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

import pathlib

from absl.testing import absltest
import jax
import jax.numpy as jnp
import tensorflow as tf

from .net.proto2.contrib.pyutil import compare
from google.protobuf import text_format
from neptune_model._src.core import persistence_lib
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.processors import processor as processor_lib
from neptune_model._src.processors import tf_processor
from neptune_model.protos import orchestration_pb2
from neptune_model.protos import tf_concrete_function_handle_pb2


class MainLibOrchestrationTest(compare.Proto2Assertions, absltest.TestCase):

  def setUp(self):
    super().setUp()
    base_path = pathlib.Path(__file__).absolute().parent
    self._testdata_dir = base_path / 'testdata'

  def _get_testdata_path(self, filename: str) -> pathlib.Path:
    """Returns the full path to a file in the testdata directory."""
    return self._testdata_dir / filename

  def _assert_proto_equal_to_testdata(
      self,
      actual_proto,
      testdata_filename: str,
      ignored_fields: list[str] | None = None,
  ):
    """Asserts that a parsed proto equals a textproto from testdata."""
    expected_proto_path = self._get_testdata_path(testdata_filename)
    with open(expected_proto_path, 'r') as f:
      expected_proto_text = f.read()

    expected_proto = text_format.Parse(
        expected_proto_text, type(actual_proto)()
    )

    compare.assertProto2Equal(
        self,
        expected_proto,
        actual_proto,
        ignored_fields=ignored_fields,
    )

  def test_save_model_with_sequential_processing_and_tf_processors(self):
    var1 = tf.Variable(1.0)

    @tf.function
    def pre_fn(x):
      return x + var1

    var2 = tf.Variable(2.0)

    @tf.function
    def post_fn(y):
      return y * var2

    def model_fn(params, inputs):
      return inputs * params['w']

    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=model_fn,
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array(3.0, dtype=jnp.float32)},
    )

    cfg = serving_config.ServingConfig(
        serving_signature='serving_default',
        input_signature=jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32),
        processing=processor_lib.SequentialProcessing(
            preprocessors=[
                tf_processor.TfProcessor('pre', pre_fn),
            ],
            postprocessors=[
                tf_processor.TfProcessor('post', post_fn),
            ],
            options=processor_lib.ProcessingOptions(
                tf_trackable_resources=[var1, var2],
            ),
        ),
    )

    export_path = pathlib.Path(self.create_tempdir().full_path)
    main_lib.save(
        model=model,
        export_path=str(export_path),
        serving_configs=[cfg],
    )

    manifest_proto = persistence_lib.load(str(export_path))
    with open(export_path / 'orchestration.pb', 'rb') as f:
      pipelines_proto = orchestration_pb2.Pipelines.FromString(f.read())
    with open(export_path / 'pre.pb', 'rb') as f:
      tf_pre_processor_proto = (
          tf_concrete_function_handle_pb2.TfConcreteFunctionHandle.FromString(
              f.read()
          )
      )
    with open(export_path / 'post.pb', 'rb') as f:
      tf_post_processor_proto = (
          tf_concrete_function_handle_pb2.TfConcreteFunctionHandle.FromString(
              f.read()
          )
      )

    loaded_tf_model = tf.saved_model.load(export_path / 'tf_saved_model')

    with self.subTest('check_manifest_proto'):
      self._assert_proto_equal_to_testdata(
          manifest_proto,
          'model_with_tf_processors/manifest.textproto',
          ignored_fields=[
              'objects.function.body.stable_hlo_body.stable_hlo.inlined_bytes',
          ],
      )

    with self.subTest('check_pipelines_proto'):
      self._assert_proto_equal_to_testdata(
          pipelines_proto, 'model_with_tf_processors/pipelines.textproto'
      )

    with self.subTest('check_tf_pre_processor_proto'):
      self._assert_proto_equal_to_testdata(
          tf_pre_processor_proto,
          'model_with_tf_processors/preprocessor.textproto',
      )

    with self.subTest('check_tf_post_processor_proto'):
      self._assert_proto_equal_to_testdata(
          tf_post_processor_proto,
          'model_with_tf_processors/postprocessor.textproto',
      )

    with self.subTest('check_tf_saved_model_written'):
      self.assertTrue(
          (export_path / 'tf_saved_model' / 'saved_model.pb').exists()
      )

    with self.subTest('check_tf_functions_loaded'):
      self.assertIn('pre', loaded_tf_model.signatures)
      self.assertIn('post', loaded_tf_model.signatures)

    with self.subTest('check_tf_variables_loaded'):
      var1_loaded, var2_loaded = loaded_tf_model.resources[0]
      # Check that variables var1 and var2 are present and have correct values
      self.assertEqual(var1_loaded.numpy(), 1.0)
      self.assertEqual(var2_loaded.numpy(), 2.0)


if __name__ == '__main__':
  absltest.main()
