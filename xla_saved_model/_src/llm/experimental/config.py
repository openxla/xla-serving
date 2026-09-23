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

"""Configuration classes for LLM generation."""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
  """Configuration for on-device JAX autoregressive token generation.

  Attributes:
    strategy: Generation strategy. Currently only "beam_search" is supported.
    max_output_tokens: Maximum number of tokens to generate dynamically.
    beam_size: Beam size for beam_search strategy. If None, beam search is
      disabled.
  """

  strategy: Literal["beam_search", "scoring"] = "beam_search"
  max_output_tokens: int = 128
  beam_size: int = 1
