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
import jax.numpy as jnp
from neptune_model._src.llm.experimental import caching
from neptune_model._src.llm.experimental import config as llm_config
from neptune_model._src.llm.experimental import generation


def _get_shape_safe(inputs):
  if isinstance(inputs, dict):
    return inputs["tokens"].shape
  return inputs.shape


class GenerationTest(absltest.TestCase):

  def test_build_neptune_apply_fn_returns_callable(self):
    config = llm_config.GenerationConfig(strategy="beam_search")

    def mock_forward(params, inputs, cache, is_prefill):
      shape = _get_shape_safe(inputs)
      return jnp.zeros((shape[0], shape[1], 10)), cache

    initial_cache_factory = lambda b: jnp.zeros((b, 1))

    apply_fn = generation.build_neptune_apply_fn(
        model_forward_step_fn=mock_forward,
        generation_config=config,
        initial_cache_factory=initial_cache_factory,
        cache_processor=caching.PagedCacheProcessor(),
    )

    self.assertTrue(callable(apply_fn))

  def test_beam_search_success(self):
    config = llm_config.GenerationConfig(
        strategy="beam_search", max_output_tokens=32
    )

    def mock_forward(params, inputs, cache, is_prefill):
      shape = _get_shape_safe(inputs)
      return jnp.zeros((shape[0], shape[1], 10)), cache

    initial_cache_factory = lambda b: jnp.zeros((b, 1))

    apply_fn = generation.build_neptune_apply_fn(
        model_forward_step_fn=mock_forward,
        generation_config=config,
        initial_cache_factory=initial_cache_factory,
        cache_processor=caching.PagedCacheProcessor(),
    )

    params = {}
    inputs = jnp.ones((2, 5), dtype=jnp.int32)  # Batch 2, SeqLen 5

    result = apply_fn(params, inputs)

    # Expected shape: [batch_size, beam_size, max_new_tokens]
    self.assertEqual(result.shape, (2, 1, 32))

  def test_unsupported_strategy_raises_value_error(self):
    # 'scoring' is in Literal but not handled in match
    config = llm_config.GenerationConfig(strategy="scoring")

    def mock_forward(params, inputs, cache, is_prefill):
      shape = _get_shape_safe(inputs)
      return jnp.zeros((shape[0], shape[1], 10)), cache

    initial_cache_factory = lambda b: jnp.zeros((b, 1))

    apply_fn = generation.build_neptune_apply_fn(
        model_forward_step_fn=mock_forward,
        generation_config=config,
        initial_cache_factory=initial_cache_factory,
        cache_processor=caching.PagedCacheProcessor(),
    )

    params = {}
    inputs = jnp.ones((2, 5))

    with self.assertRaisesRegex(ValueError, "Unsupported strategy: scoring"):
      apply_fn(params, inputs)


if __name__ == "__main__":
  absltest.main()