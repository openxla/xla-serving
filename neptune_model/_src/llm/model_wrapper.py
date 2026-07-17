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

"""Base contract wrapper for LLMs exported to Neptune."""

from typing import Any
import jax


class LLMModelWrapper:
  """Base class for wrapping an LLM model to be exported using the Neptune LLM framework."""

  def initial_cache_factory(self, batch_size: int) -> Any:
    """Returns the initial KV Cache state for a given batch size."""
    raise NotImplementedError()

  def prefill(
      self,
      params: Any,
      input_ids: jax.Array,
      cache: Any,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, Any]:
    """Runs the prefill pass and returns (last_token_logits, next_cache).

    Args:
      params: Model parameters.
      input_ids: Input token IDs, shape (batch_size, max_seq_len).
      cache: Initial KV cache.
      prompt_len: Actual length of each prompt, shape (batch_size,).

    Returns:
      last_token_logits: Logits for the last token of the prompt, shape
        (batch_size, vocab_size).
      next_cache: Updated KV cache.
    """
    raise NotImplementedError()

  def decode_step(
      self,
      params: Any,
      token_ids: jax.Array,
      cache: Any,
      step: int,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, Any]:
    """Runs a single decode step and returns (logits, next_cache).

    Args:
      params: Model parameters.
      token_ids: Last generated tokens, shape (batch_size * beam_size, 1) or
        (batch_size, 1).
      cache: Current KV cache.
      step: Current step index (1-indexed, starts at 1, goes up to
        max_new_tokens-1).
      prompt_len: Actual prompt length, shape (batch_size,).

    Returns:
      logits: Logits for the step, shape (batch_size * beam_size, vocab_size) or
        (batch_size, vocab_size).
      next_cache: Updated KV cache.
    """
    raise NotImplementedError()
