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

"""Configuration for tokenization plugins in Neptune LLM."""

import dataclasses
import enum


class TokenizerType(enum.Enum):
  KERAS_HUB = "keras_hub"


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
  """Configuration for the pre/post-processing tokenizer.

  Attributes:
    tokenizer_type: Type of tokenizer, defaults to "keras_hub".
    vocab_path: Path to the vocabulary file.
    model_name_or_path: Hugging Face model identifier or path.
    bos_token_id: Beginning-of-sequence token ID.
    eos_token_id: End-of-sequence token ID.
    pad_token_id: Padding token ID.
  """

  tokenizer_type: TokenizerType = TokenizerType.KERAS_HUB
  vocab_path: str = ""
  model_name_or_path: str = ""
  bos_token_id: int | None = None
  eos_token_id: int | None = None
  pad_token_id: int | None = None
