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

"""Exporter wrapper to enable tokenization plugins on top of Neptune LLM Core."""

from collections.abc import Sequence
import dataclasses

import jax
import jaxtyping

from neptune_model._src.llm import config as core_config
from neptune_model._src.llm import export as core_exporter
from neptune_model._src.llm import model_wrapper
from neptune_model._src.llm.tokenizers import config as tokenizers_config
from neptune_model._src.llm.tokenizers import processors
from neptune_model._src.processors import tf_processor


@dataclasses.dataclass(frozen=True)
class NeptuneLLMExportWithTokenizerConfig:
  """Configuration for Neptune LLM export with tokenization."""

  export_dir: str
  allowed_batch_sizes: Sequence[int]
  generation_config: core_config.GenerationConfig
  tokenizer_config: tokenizers_config.TokenizerConfig


def export_llm_with_tokenizer(
    config: NeptuneLLMExportWithTokenizerConfig,
    model: model_wrapper.LLMModelWrapper,
    params: jaxtyping.PyTree,
    jax_mesh: jax.sharding.Mesh | None = None,
) -> None:
  """Wraps core export_llm with KerasHub tokenization stages."""
  tokenizer_config_instance = config.tokenizer_config
  generation_config_instance = config.generation_config

  if (
      tokenizer_config_instance.tokenizer_type
      != tokenizers_config.TokenizerType.KERAS_HUB
  ):
    raise ValueError(
        "Unsupported tokenizer type:"
        f" {tokenizer_config_instance.tokenizer_type}. Only 'keras_hub' is"
        " supported."
    )

  target_path = (
      tokenizer_config_instance.model_name_or_path
      or tokenizer_config_instance.vocab_path
  )
  if not target_path:
    raise ValueError(
        "model_name_or_path (or vocab_path) must be specified for keras_hub"
        " tokenizer."
    )

  max_prompt_length = generation_config_instance.max_prompt_length
  tokenizer = processors.KerasHubTokenizerTF(
      model_name_or_path=target_path,
      eos_token_id=tokenizer_config_instance.eos_token_id,
      pad_token_id=tokenizer_config_instance.pad_token_id,
      max_seq_len=max_prompt_length,
  )

  preprocessor = tf_processor.TfProcessor(
      name="llm_pre_process",
      fn=tokenizer.pre_process_fn,
  )
  postprocessor = tf_processor.TfProcessor(
      name="llm_post_process",
      fn=tokenizer.post_process_fn,
  )
  trackable_resources = tokenizer.get_trackable_resources()

  # Map back to Core Config
  core_export_config = core_config.NeptuneLLMExportConfig(
      export_dir=config.export_dir,
      allowed_batch_sizes=config.allowed_batch_sizes,
      generation_config=generation_config_instance,
  )

  # Invoke Core Exporter
  core_exporter.export_llm(
      config=core_export_config,
      model=model,
      params=params,
      jax_mesh=jax_mesh,
      pre_processor=preprocessor,
      post_processor=postprocessor,
      tf_trackable_resources=trackable_resources,
  )
