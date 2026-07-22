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

"""Configuration classes for exporting LLMs with Neptune."""

from collections.abc import Sequence
import dataclasses
import enum


class SamplingMethod(enum.Enum):
  GREEDY = "greedy"
  TOP_K = "top_k"
  BEAM_SEARCH = "beam_search"
  SCORING = "scoring"


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
  """Configuration for on-device JAX autoregressive token generation.

  Attributes:
    sampling_method: Sampling method to use for token generation.
    max_output_tokens: Maximum number of tokens to generate.
    temperature: Temperature for sampling (ignored for "greedy").
    top_k: Top-k tokens to consider during sampling.
    beam_size: Beam size for beam_search strategy. If None, beam search is
      disabled.
    max_prompt_length: Maximum number of input prompt tokens allowed.
    pad_token_id: Padding token ID used for prompt padding and length
      calculation.
  """

  sampling_method: SamplingMethod = SamplingMethod.GREEDY
  max_output_tokens: int = 128
  temperature: float = 1.0
  top_k: int = 50
  beam_size: int | None = None
  max_prompt_length: int = 1024
  pad_token_id: int = 0


@dataclasses.dataclass(frozen=True)
class NeptuneLLMExportConfig:
  """Consolidated configuration for the E2E Neptune LLM export pipeline.

  Attributes:
    export_dir: Target directory to save the Neptune model bundle.
    allowed_batch_sizes: A list of static batch sizes allowed during serving.
    generation_config: Autoregressive generation configuration.
  """

  export_dir: str
  allowed_batch_sizes: Sequence[int]
  generation_config: GenerationConfig
