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

from .pyglib import resources
from neptune_model._src.llm import config as core_config
from neptune_model._src.llm import model_wrapper
from neptune_model.llm.tokenizers import config as tokenizer_config
from neptune_model.llm.tokenizers import exporter
from neptune_model_runner import model_runner

# Ensure offline mode for Hugging Face during tests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


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
        resources.GetRunfilesDir(),
        "/third_party/neptune/neptune_model/llm/tokenizers/testdata/qwen_dummy",
    )
    tokenizer_config_instance = tokenizer_config.TokenizerConfig(
        tokenizer_type="keras_hub",
        model_name_or_path=model_path,
    )
    generation_config_instance = core_config.GenerationConfig(
        sampling_method="greedy",
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
    model = MockModel()
    params = {"w": jnp.zeros((10, 10))}

    exporter.export_llm_with_tokenizer(
        config=export_config,
        model=model,
        params=params,
    )
    self.assertTrue(os.path.exists(export_dir))

    runner = model_runner.ModelRunner(model_path=export_dir)
    inputs = {
        "text_input": np.array(["ab"], dtype=object),
    }
    outputs = runner.signature("serving_default").run_with_dict_return(**inputs)
    self.assertIn("text_output", outputs)
    self.assertEqual(outputs["text_output"].shape, (1,))
    self.assertEqual(outputs["text_output"][0], b"ab" * 10)

  def test_export_llm_with_unsupported_tokenizer_type_raises(self):
    # pytype: disable=wrong-arg-types
    tokenizer_config_instance = tokenizer_config.TokenizerConfig(
        tokenizer_type="unsupported",
    )
    # pytype: enable=wrong-arg-types
    generation_config_instance = core_config.GenerationConfig(
        sampling_method="greedy",
        max_output_tokens=10,
        max_prompt_length=22,
    )
    export_config = exporter.NeptuneLLMExportWithTokenizerConfig(
        export_dir="/tmp/test_invalid_export",
        allowed_batch_sizes=(1,),
        generation_config=generation_config_instance,
        tokenizer_config=tokenizer_config_instance,
    )
    model = MockModel()
    params = {"w": jnp.zeros((10, 10))}

    with self.assertRaises(ValueError):
      exporter.export_llm_with_tokenizer(
          config=export_config,
          model=model,
          params=params,
      )


if __name__ == "__main__":
  absltest.main()
