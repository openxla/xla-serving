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

"""Unit tests for LLMModelWrapper and compiled standard generation loop."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jaxtyping

from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import model_wrapper


class MockModel(model_wrapper.LLMModelWrapper):
  """Simple mock model implementing the generation loop duck typing contract."""

  def initial_cache_factory(self, batch_size: int) -> jaxtyping.PyTree:
    return jnp.zeros((batch_size, 1))

  def prefill(
      self,
      params: jaxtyping.PyTree,
      input_ids: jax.Array,
      cache: jaxtyping.PyTree,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, jaxtyping.PyTree]:
    del params, prompt_len
    last_tokens = input_ids[:, -1]
    next_expected = (last_tokens + 1) % 10

    one_hot = jax.nn.one_hot(next_expected, num_classes=10)
    return one_hot * 10.0, cache

  def decode_step(
      self,
      params: jaxtyping.PyTree,
      token_ids: jax.Array,
      cache: jaxtyping.PyTree,
      step: int,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, jaxtyping.PyTree]:
    del params, step, prompt_len
    last_tokens = token_ids[:, -1]
    next_expected = (last_tokens + 1) % 10

    one_hot = jax.nn.one_hot(next_expected, num_classes=10)
    return one_hot * 10.0, cache


class ModelWrapperTest(absltest.TestCase):

  def test_greedy_generation_correctness(self):
    config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.GREEDY,
        max_output_tokens=4,
        pad_token_id=0,
    )
    model = MockModel(generation_config=config)
    apply_fn = model.compile_generation_loop()

    params = {}
    inputs = {
        "input_ids": jnp.array([[1, 2, 3]], dtype=jnp.int32),
    }
    outputs = apply_fn(params, inputs)

    expected_tokens = jnp.array([[4, 5, 6, 7]], dtype=jnp.int32)
    self.assertEqual(outputs["token_ids"].tolist(), expected_tokens.tolist())

  def test_top_k_generation_correctness(self):
    config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.TOP_K,
        max_output_tokens=4,
        top_k=2,
        temperature=0.01,
        pad_token_id=0,
    )
    model = MockModel(generation_config=config)
    apply_fn = model.compile_generation_loop()

    params = {}
    inputs = {
        "input_ids": jnp.array([[1, 2, 3]], dtype=jnp.int32),
    }
    outputs = apply_fn(params, inputs)

    expected_tokens = jnp.array([[4, 5, 6, 7]], dtype=jnp.int32)
    self.assertEqual(outputs["token_ids"].tolist(), expected_tokens.tolist())

  def test_beam_search_raises_value_error(self):
    config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.BEAM_SEARCH
    )
    model = MockModel(generation_config=config)
    with self.assertRaises(ValueError):
      model.compile_generation_loop()

  def test_scoring_mode_correctness(self):
    config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.SCORING,
        pad_token_id=0,
    )
    model = MockModel(generation_config=config)
    score_fn = model.compile_generation_loop()

    params = {}
    inputs = {
        "input_ids": jnp.array([[1, 2, 3]], dtype=jnp.int32),
    }
    outputs = score_fn(params, inputs)

    self.assertIn("logits", outputs)
    self.assertIn("log_probs", outputs)
    self.assertEqual(outputs["logits"].shape, (1, 10))
    self.assertEqual(outputs["log_probs"].shape, (1, 10))

  def test_scoring_mode_with_target_ids(self):
    config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.SCORING,
        pad_token_id=0,
    )
    model = MockModel(generation_config=config)
    score_fn = model.compile_generation_loop()

    params = {}
    inputs = {
        "input_ids": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "target_ids": jnp.array([4], dtype=jnp.int32),
    }
    outputs = score_fn(params, inputs)

    self.assertIn("target_log_probs", outputs)
    self.assertEqual(outputs["target_log_probs"].shape, (1,))


if __name__ == "__main__":
  absltest.main()

