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
import jax.numpy as jnp
import jaxtyping

from neptune_model._src.jax2npt import typing as neptune_typing
from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import sampling


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
        max_output_tokens-1).
      prompt_len: Actual prompt length, shape (batch_size,).

    Returns:
      logits: Logits for the step, shape (batch_size * beam_size, vocab_size) or
        (batch_size, vocab_size).
      next_cache: Updated KV cache.
    """
    raise NotImplementedError()

  def compile_generation_loop(
      self,
      generation_config: llm_config.GenerationConfig,
  ) -> neptune_typing.ApplyFn:
    """Compiles standard greedy / top_k autoregressive decoding loop.

    Subclasses can override this method to provide custom decoding algorithms
    (such as TPU PagedAttention beam search) while delegating other sampling
    strategies back to super().

    Args:
      generation_config: Generation configuration.

    Returns:
      A compiled JAX function to generate tokens.
    """
    max_output_tokens = generation_config.max_output_tokens
    sampling_method = generation_config.sampling_method

    if sampling_method not in ("greedy", "top_k"):
      raise ValueError(
          f"Unsupported standard sampling method: {sampling_method}. Beam"
          " search must be handled by custom model wrapper implementations."
      )

    def generate_fn(
        params: jaxtyping.PyTree,
        inputs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
      input_ids = inputs["input_ids"]
      if "prompt_len" in inputs:
        prompt_len = inputs["prompt_len"]
      else:
        prompt_len = jnp.sum(
            input_ids != generation_config.pad_token_id,
            axis=-1,
            dtype=jnp.int32,
        )
      batch_size = input_ids.shape[0]

      # 1. Allocate KV Cache and execute Prefill Pass
      empty_cache = self.initial_cache_factory(batch_size)
      prefill_logits, cache = self.prefill(
          params, input_ids, empty_cache, prompt_len
      )

      # 2. Select First Generated Token
      if sampling_method == "greedy":
        next_token = sampling.greedy_sample(prefill_logits)
        keys = None
      elif sampling_method == "top_k":
        hash_seeds = jnp.sum(input_ids, axis=-1, dtype=jnp.int32)
        master_key = jax.random.PRNGKey(1337)
        keys = jax.vmap(lambda seed: jax.random.fold_in(master_key, seed))(
            hash_seeds
        )
        split_keys = jax.vmap(jax.random.split)(keys)
        keys = split_keys[:, 0, :]
        step_key = split_keys[:, 1, :]
        next_token = jax.vmap(
            lambda k, l: sampling.top_k_sample(
                k, l, generation_config.top_k, generation_config.temperature
            )
        )(step_key, prefill_logits)
      else:
        raise ValueError(f"Unexpected sampling method: {sampling_method}")

      # Pre-allocate output token buffer of shape
      # [batch_size, max_output_tokens].
      output_tokens = jnp.zeros(
          (batch_size, max_output_tokens), dtype=jnp.int32
      )
      output_tokens = output_tokens.at[:, 0].set(next_token)

      # 3. Python unrolled decode loop
      for step in range(1, max_output_tokens):
        # Retrieve the token generated in the previous step -> shape [B, 1]
        last_generated_token = jax.lax.dynamic_slice_in_dim(
            output_tokens, step - 1, 1, axis=1
        )

        # Execute single-step decoder forward pass
        step_logits, cache = self.decode_step(
            params, last_generated_token, cache, step, prompt_len
        )

        # Select next token
        if sampling_method == "greedy":
          next_token_step = sampling.greedy_sample(step_logits)
        elif sampling_method == "top_k":
          split_keys = jax.vmap(jax.random.split)(keys)
          keys = split_keys[:, 0, :]
          step_key = split_keys[:, 1, :]
          next_token_step = jax.vmap(
              lambda k, l: sampling.top_k_sample(
                  k, l, generation_config.top_k, generation_config.temperature
              )
          )(step_key, step_logits)
        else:
          raise ValueError(f"Unexpected sampling method: {sampling_method}")

        # Write selected token into output buffer
        output_tokens = output_tokens.at[:, step].set(next_token_step)

      return {"token_ids": output_tokens}

    return generate_fn
