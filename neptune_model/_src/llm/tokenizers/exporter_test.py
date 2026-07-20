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

"""Unit tests for exporting LLMs with tokenization plugins."""

# pylint: disable=g-importing-member

import os
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from neptune_model._src.llm import config as core_config
from neptune_model._src.llm import model_wrapper
from neptune_model._src.llm.tokenizers import config as tokenizer_config
from neptune_model._src.llm.tokenizers import exporter


# TODO(armandouv): Consider adding an integration test that exports a real LLM
# model architecture (e.g. from _src/models/ or examples/) with checked-in
# tokenizer assets, rather than relying solely on MockModel for pipeline
# verification.
class MockModel(model_wrapper.LLMModelWrapper):
  """Dummy model wrapper for testing tokenized export."""

  def initial_cache_factory(self, batch_size: int) -> Any:
    return jnp.zeros((batch_size, 1))

  def prefill(
      self,
      params: Any,
      input_ids: jax.Array,
      cache: Any,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, Any]:
    batch_size = input_ids.shape[0]
    # Reference params["w"] so JAX export retains it in module_kept_var_idx
    w_sum = jnp.sum(params["w"])
    logits = jnp.zeros((batch_size, 100)) + w_sum
    logits = logits.at[:, 4].set(100.0)
    return logits, cache

  def decode_step(
      self,
      params: Any,
      token_ids: jax.Array,
      cache: Any,
      step: int,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, Any]:
    batch_size = token_ids.shape[0]
    w_sum = jnp.sum(params["w"])
    logits = jnp.zeros((batch_size, 100)) + w_sum
    logits = logits.at[:, 4].set(100.0)
    return logits, cache


class ExporterTest(absltest.TestCase):

  def test_export_llm_with_keras_hub_tokenizer_success(self):
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../testdata/qwen_dummy",
    )

    tokenizer_config_instance = tokenizer_config.TokenizerConfig(
        tokenizer_type=tokenizer_config.TokenizerType.KERAS_HUB,
        model_name_or_path=model_path,
    )
    generation_config_instance = core_config.GenerationConfig(
        sampling_method=core_config.SamplingMethod.GREEDY,
        max_output_tokens=10,
        max_prompt_length=22,
    )
    export_dir = self.create_tempdir().full_path
    export_config = exporter.NeptuneLLMExportWithTokenizerConfig(
        export_dir=export_dir,
        allowed_batch_sizes=(1,),
        generation_config=generation_config_instance,
        tokenizer_config=tokenizer_config_instance,
    )
    model = MockModel(generation_config=generation_config_instance)
    params = {"w": jnp.zeros((10, 10))}

    exporter.export_llm_with_tokenizer(
        config=export_config,
        model=model,
        params=params,
    )
    self.assertTrue(os.path.exists(export_dir))

if __name__ == "__main__":
  absltest.main()
