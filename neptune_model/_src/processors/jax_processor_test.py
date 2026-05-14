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
import jax
import jax.numpy as jnp

from neptune_model._src.core import shlo_type
from neptune_model._src.core import test_utils
from neptune_model._src.export import jax_model
from neptune_model._src.processors import jax_processor


class JaxProcessorTest(test_utils.NptTestCase):

  def test_calculate_jax_model_input_sig(self):
    params_spec = shlo_type.ShloTensorSpec(
        shape=[1], dtype=shlo_type.ShloDType.f32
    )
    input_spec = shlo_type.ShloTensorSpec(
        shape=[4], dtype=shlo_type.ShloDType.f32
    )
    kw_spec = shlo_type.ShloTensorSpec(shape=[2], dtype=shlo_type.ShloDType.f32)

    with self.subTest(name='with_kwargs'):
      apply_fn_input_sig = ((params_spec, input_spec), {'kw': kw_spec})
      input_sig = jax_processor._calculate_jax_model_input_sig(
          apply_fn_input_sig
      )
      self.assertEqual(input_sig, ((input_spec,), {'kw': kw_spec}))

    with self.subTest(name='without_kwargs'):
      apply_fn_input_sig = (params_spec, input_spec)
      input_sig = jax_processor._calculate_jax_model_input_sig(
          apply_fn_input_sig
      )
      self.assertEqual(input_sig, ((input_spec,), {}))

    with self.subTest(name='not_sequence_fails'):
      with self.assertRaisesRegex(ValueError, 'must be a sequence'):
        jax_processor._calculate_jax_model_input_sig(params_spec)

    with self.subTest(name='wrong_num_pos_args_fails'):
      with self.assertRaisesRegex(ValueError, 'must have exactly two elements'):
        jax_processor._calculate_jax_model_input_sig((params_spec,))

  def test_input_signature_with_polymorphic_shape(self):
    """Tests that JaxProcessor.prepare works with polymorphic shapes."""
    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=lambda params, x: x * params['w'],
            input_polymorphic_shapes='b,_',
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    input_signature = jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.float32)
    serving_signature = 'serving_default'

    processor = jax_processor.JaxProcessor(
        name=serving_signature,
        model=model,
        method_key=None,
    )

    prepared_processor = processor.prepare(input_signature)

    with self.subTest('check_input_signature'):
      # input_sig = ((input_tree,), kwargs_tree)
      # input is ShapeDtypeStruct(shape=('b', 1), dtype=float32)
      input_sig = prepared_processor.input_signature
      self.assertEqual(input_sig[0][0].shape, [None, 1])
      self.assertEqual(input_sig[0][0].dtype, shlo_type.ShloDType.f32)

    with self.subTest('check_output_signature'):
      output_sig = prepared_processor.output_signature
      self.assertEqual(output_sig.shape, [None, 1])
      self.assertEqual(output_sig.dtype, shlo_type.ShloDType.f32)

  def test_shlo_function_data_names(self):
    def _apply_fn(params, x):
      return x * params['layer1']['w'] + params['layer1']['b']

    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=_apply_fn,
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={
            'layer1': {'w': jnp.array([2.0]), 'b': jnp.array([1.0])},
            'layer2': {'w': jnp.array([3.0]), 'b': jnp.array([4.0])},
        },
    )

    processor = jax_processor.JaxProcessor(
        name='serving_default',
        model=model,
    )
    input_signature = jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.float32)
    prepared = processor.prepare(input_signature)

    shlo_fns = prepared.shlo_fns
    self.assertLen(shlo_fns, 1)
    self.assertEqual(
        shlo_fns[0].data_names,
        ['layer1.b', 'layer1.w', 'layer2.b', 'layer2.w'],
    )


if __name__ == '__main__':
  absltest.main()
