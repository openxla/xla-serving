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

import jax
import jax.numpy as jnp
import jaxtyping

from neptune_model._src.jax2npt import typing as neptune_typing
from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import sampling


class LLMModelWrapper:
  """Base contract wrapper for exported LLMs in the Neptune LLM framework.

  Subclasses implement model-specific execution primitives (cache allocation,
  prompt prefill, and single-step token decoding). The wrapper compiles these
  primitives into JAX-native autoregressive generation loops compatible with
  Neptune Model export and serving pipelines.
  """

  def __init__(
      self,
      generation_config: llm_config.GenerationConfig,
  ):
    """Initializes the wrapper with generation configuration.

    Args:
      generation_config: Generation configuration specifying decoding
        parameters.
    """
    self.generation_config = generation_config

  def initial_cache_factory(self, batch_size: int) -> jaxtyping.PyTree:
    """Allocates and returns the initial KV Cache state for a given batch size.

    Args:
      batch_size: Number of concurrent input sequences to allocate cache for.

    Returns:
      Initial KV cache PyTree state.
    """
    raise NotImplementedError()

  def prefill(
      self,
      params: jaxtyping.PyTree,
      input_ids: jax.Array,
      cache: jaxtyping.PyTree,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, jaxtyping.PyTree]:
    """Runs prefill for the given prompt input.

    Args:
      params: Model parameters PyTree.
      input_ids: 2D array of prompt token IDs with shape (batch_size,
        max_seq_len).
      cache: Initial KV cache PyTree allocated by initial_cache_factory.
      prompt_len: 1D array of shape (batch_size,) containing actual unpadded
        prompt lengths.

    Returns:
      A tuple of (last_token_logits, next_cache) where last_token_logits has
      shape (batch_size, vocab_size) and next_cache is the updated KV cache.
    """
    raise NotImplementedError()

  def decode_step(
      self,
      params: jaxtyping.PyTree,
      token_ids: jax.Array,
      cache: jaxtyping.PyTree,
      step: int,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, jaxtyping.PyTree]:
    """Runs a single autoregressive decode step for generated tokens.

    Args:
      params: Model parameters PyTree.
      token_ids: 2D array of the latest generated tokens with shape
        (num_active_seqs, 1).
      cache: Current KV cache PyTree.
      step: Current 1-indexed generation step (1 <= step < max_output_tokens).
      prompt_len: 1D array of shape (batch_size,) containing prompt lengths.

    Returns:
      A tuple of (logits, next_cache) where logits has shape
      (num_active_seqs, vocab_size) and next_cache is the updated KV cache.
    """
    raise NotImplementedError()

  def get_generation_config(self) -> llm_config.GenerationConfig:
    """Returns the generation configuration."""
    return self.generation_config

  def compile_generation_loop(
      self,
  ) -> neptune_typing.ApplyFn:
    """Compiles an autoregressive generation loop or scoring function into a callable JAX function.

    Subclasses can override this method to provide custom decoding algorithms
    (such as TPU PagedAttention beam search) while delegating standard sampling
    strategies back to super().

    Returns:
      A JAX apply function accepting (params, inputs) and returning a dictionary
      of output tensors (e.g. {"token_ids": output_tokens}, or {"logits":
      logits, "log_probs": log_probs, "target_log_probs": target_log_probs} when
      in SCORING mode).
    """
    if self.generation_config.pad_token_id is None:
      raise ValueError(
          "pad_token_id must be already set in the generation config."
      )

    sampling_method = self.generation_config.sampling_method

    if sampling_method not in (
        llm_config.SamplingMethod.GREEDY,
        llm_config.SamplingMethod.TOP_K,
        llm_config.SamplingMethod.SCORING,
    ):
      raise ValueError(
          f"Unsupported standard sampling method: {sampling_method}. Only"
          " GREEDY, TOP_K, and SCORING are supported. Other sampling methods"
          " must be handled by custom model wrapper implementations."
      )

    max_output_tokens = self.generation_config.max_output_tokens

    def generate_fn(
        params: jaxtyping.PyTree,
        inputs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
      input_ids = inputs["input_ids"]
      prompt_len = jnp.sum(
          input_ids != self.generation_config.pad_token_id,
          axis=-1,
          dtype=jnp.int32,
      )
      batch_size = input_ids.shape[0]

      # 1. Allocate KV Cache and execute Prefill Pass
      empty_cache = self.initial_cache_factory(batch_size)
      prefill_logits, cache = self.prefill(
          params, input_ids, empty_cache, prompt_len
      )

      if sampling_method == llm_config.SamplingMethod.SCORING:
        return sampling.compute_scores(prefill_logits, inputs.get("target_ids"))

      # 2. Select First Generated Token
      if sampling_method == llm_config.SamplingMethod.GREEDY:
        next_token = sampling.greedy_sample(prefill_logits)
        keys = None
      elif sampling_method == llm_config.SamplingMethod.TOP_K:
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
                k,
                l,
                self.generation_config.top_k,
                self.generation_config.temperature,
            )
        )(step_key, prefill_logits)
      else:
        raise ValueError(f"Unsupported sampling method: {sampling_method}")

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
        if sampling_method == llm_config.SamplingMethod.GREEDY:
          next_token_step = sampling.greedy_sample(step_logits)
        elif sampling_method == llm_config.SamplingMethod.TOP_K:
          split_keys = jax.vmap(jax.random.split)(keys)  # pyrefly: ignore[bad-argument-type]
          keys = split_keys[:, 0, :]
          step_key = split_keys[:, 1, :]
          next_token_step = jax.vmap(
              lambda k, l: sampling.top_k_sample(
                  k,
                  l,
                  self.generation_config.top_k,
                  self.generation_config.temperature,
              )
          )(step_key, step_logits)
        else:
          raise ValueError(f"Unexpected sampling method: {sampling_method}")

        # Write selected token into output buffer
        output_tokens = output_tokens.at[:, step].set(next_token_step)

      return {"token_ids": output_tokens}

    return generate_fn
