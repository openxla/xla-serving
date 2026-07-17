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

"""Standard autoregressive decoding loops and graph compilation for Neptune."""


import jax
import jax.numpy as jnp
import jaxtyping
from neptune_model._src.jax2npt import typing as neptune_typing
from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import model_wrapper


def greedy_sample(logits: jax.Array) -> jax.Array:
  """Samples the token with the highest logit value (greedy search)."""
  return jnp.argmax(logits, axis=-1)


def top_k_sample(
    key: jax.Array, logits: jax.Array, k: int, temperature: float
) -> jax.Array:
  """Samples a token from the top-k logits."""
  # Get top-k values and indices
  top_values, top_indices = jax.lax.top_k(logits, k)

  # Softmax on scaled values
  probs = jax.nn.softmax(top_values / temperature, axis=-1)

  # Sample from categorical distribution
  sample_idx = jax.random.categorical(key, jnp.log(probs + 1e-9))

  return top_indices[sample_idx]


def make_standard_loop(
    model: model_wrapper.LLMModelWrapper,
    generation_config: llm_config.GenerationConfig,
) -> neptune_typing.ApplyFn:
  """Compiles standard greedy / top_k autoregressive decoding loop.

  Args:
    model: LLM model wrapper instance.
    generation_config: Generation configuration.

  Returns:
    A JAX function to generate tokens.
  """
  max_new_tokens = generation_config.max_output_tokens
  sampling_method = generation_config.sampling_method

  if sampling_method not in ("greedy", "top_k"):
    raise ValueError(
        f"Unsupported standard sampling method: {sampling_method}. Beam search"
        " must be handled by custom model wrapper implementations."
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
          input_ids != generation_config.pad_token_id, axis=-1, dtype=jnp.int32
      )
    batch_size = input_ids.shape[0]

    # 1. Allocate KV Cache and execute Prefill Pass
    empty_cache = model.initial_cache_factory(batch_size)
    prefill_logits, cache = model.prefill(
        params, input_ids, empty_cache, prompt_len
    )

    # 2. Select First Generated Token
    if sampling_method == "greedy":
      next_token = greedy_sample(prefill_logits)
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
          lambda k, l: top_k_sample(
              k, l, generation_config.top_k, generation_config.temperature
          )
      )(step_key, prefill_logits)
    else:
      raise ValueError(f"Unexpected sampling method: {sampling_method}")

    # Pre-allocate output token buffer of shape [batch_size, max_new_tokens]
    output_tokens = jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
    output_tokens = output_tokens.at[:, 0].set(next_token)

    # 3. Python unrolled decode loop
    for step in range(1, max_new_tokens):
      # Retrieve the token generated in the previous step -> shape [B, 1]
      last_generated_token = jax.lax.dynamic_slice_in_dim(
          output_tokens, step - 1, 1, axis=1
      )

      # Execute single-step decoder forward pass
      step_logits, cache = model.decode_step(
          params, last_generated_token, cache, step, prompt_len
      )

      # Select next token
      if sampling_method == "greedy":
        next_token_step = greedy_sample(step_logits)
      elif sampling_method == "top_k":
        split_keys = jax.vmap(jax.random.split)(keys)
        keys = split_keys[:, 0, :]
        step_key = split_keys[:, 1, :]
        next_token_step = jax.vmap(
            lambda k, l: top_k_sample(
                k, l, generation_config.top_k, generation_config.temperature
            )
        )(step_key, step_logits)
      else:
        raise ValueError(f"Unexpected sampling method: {sampling_method}")

      # Write selected token into output buffer
      output_tokens = output_tokens.at[:, step].set(next_token_step)

    return {"token_ids": output_tokens}

  return generate_fn
