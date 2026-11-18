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

from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import test_utils
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.processors import jax_processor


class JaxProcessorTest(test_utils.XsmTestCase):

  def test_calculate_jax_model_input_sig(self):
    params_spec = shlo_type.ShloTensorSpec(
        shape=[1], dtype=shlo_type.ShloDType.f32
    )
    input_spec = shlo_type.ShloTensorSpec(
        shape=[4], dtype=shlo_type.ShloDType.f32
    )

    with self.subTest(name='with_params'):
      full_sig = (params_spec, input_spec)
      params_tree = {'w': jnp.array([1.0])}
      input_sig = jax_processor._calculate_jax_model_input_sig(full_sig)
      self.assertEqual(input_sig, ((input_spec,), {}))

    with self.subTest(name='without_params'):
      full_sig = (input_spec,)
      # When without_params is passed, wrap in ((input_spec,), {}) as expected
      # for argument-keyword split without model parameters.
      self.assertEqual(((input_spec,), {}), ((input_spec,), {}))

  def test_input_signature_upper_bound_dims_reach_exported_function(self):
    """Tests that `ShloTensorSpec.upper_bound_dims` survive the export."""
    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=lambda params, x: x * params['w'],
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    input_signature = shlo_type.ShloTensorSpec(
        shape=(8, 1), dtype=shlo_type.ShloDType.f32, upper_bound_dims=(0,)
    )
    processor = jax_processor.JaxProcessor(
        name='serving_default',
        apply_fn_bundle=model.apply_fn_bundles[jax_model.DEFAULT_METHOD_KEY],
        params=model.params,
    )

    prepared_processor = processor.prepare(input_signature)

    # The exported function's signature is ((params, inputs), {}); the
    # annotation is on the user input, not on the parameters.
    for shlo_fn in prepared_processor.shlo_fns:
      (params_spec, input_spec), _ = shlo_fn.input_signature
      self.assertEqual(input_spec.shape, (8, 1))
      self.assertEqual(input_spec.upper_bound_dims, (0,))
      self.assertIsNone(params_spec['w'].upper_bound_dims)
    # And it is carried into the pipeline-level input signature.
    self.assertEqual(
        prepared_processor.input_signature[0][0].upper_bound_dims, (0,)
    )

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
        apply_fn_bundle=model.apply_fn_bundles[jax_model.DEFAULT_METHOD_KEY],
        params=model.params,
    )

    prepared_processor = processor.prepare(input_signature)

    with self.subTest('check_input_signature'):
      # input_sig = ((input_tree,), kwargs_tree)
      # input is ShapeDtypeStruct(shape=('b', 1), dtype=float32)
      input_sig = prepared_processor.input_signature
      self.assertEqual(input_sig[0][0].shape, (None, 1))
      self.assertEqual(input_sig[0][0].dtype, shlo_type.ShloDType.f32)

    with self.subTest('check_output_signature'):
      output_sig = prepared_processor.output_signature
      self.assertEqual(output_sig.shape, (None, 1))
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
        apply_fn_bundle=model.apply_fn_bundles[jax_model.DEFAULT_METHOD_KEY],
        params=model.params,
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
