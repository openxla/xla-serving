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

from collections.abc import Callable
from typing import NamedTuple
import jax
import jax.numpy as jnp
import jaxtyping
from neptune_model._src.jax2npt import typing as neptune_typing
from neptune_model._src.llm.experimental import caching
from neptune_model._src.llm.experimental import config as llm_config
class BeamSearchState(NamedTuple):
  """State tracked during the autoregressive beam search decoding loop.
  Attributes:
    step: The current generation step index.
    current_tokens: Tensor recording the generated token history, shaped
      [batch_size, beam_size, max_new_tokens].
    loop_cache: The flat KV cache for all active beams.
    virtual_to_physical_map: Boolean or Integer map tracking beam routing pointers.
    loop_log_probs: Accumulated sequence log probabilities for each beam, shaped
      [batch_size, beam_size].
    loop_done: Boolean matrix tracking which beams have hit the EOS token,
      shaped [batch_size, beam_size].
  """
  step: int
  current_tokens: jax.Array
  loop_cache: jaxtyping.PyTree
  virtual_to_physical_map: jax.Array
  loop_log_probs: jax.Array
  loop_done: jax.Array
# TODO: Revisit type signatures for forward step function and cache
# factory.
# Type for the model's forward step function.
# Inputs: (params, inputs/token_ids, current_cache, is_prefill)
# Returns: (logits, updated_cache)
ModelForwardStepFn = Callable[
    [jaxtyping.PyTree, jaxtyping.PyTree, jaxtyping.PyTree, bool],
    tuple[jaxtyping.PyTree, jaxtyping.PyTree],
]
# Type for the function that initializes an empty KV cache.
# Inputs: (batch_size)
# Returns: Initial empty cache PyTree
InitialCacheFactory = Callable[[int], jaxtyping.PyTree]


def build_neptune_apply_fn(
    model_forward_step_fn: ModelForwardStepFn,
    generation_config: llm_config.GenerationConfig,
    initial_cache_factory: InitialCacheFactory,
    cache_processor: caching.CacheProcessor,
    eos_token_id: int = 2,
) -> neptune_typing.ApplyFn:
  """Wraps a step-level model forward pass into an autoregressive JAX loop.

  The wrapped function compiles down to a single StableHLO execution graph
  running natively on the TPU device.

  Args:
    model_forward_step_fn: A callable representing the model forward pass. It
      must accept: - params: The model weights PyTree. - inputs: A PyTree or
      Dictionary containing token IDs and optional zero-copy pointers/metadata.
      - cache: The current KV Cache PyTree. - is_prefill: A boolean indicating
      whether it is the prefill pass. It must return a tuple (logits,
      updated_cache) where logits has the shape [batch_size, sequence_length,
      vocab_size].
    generation_config: Autoregressive token generation configuration.
    initial_cache_factory: A callable that returns an empty KV Cache PyTree when
      passed the batch size (an integer).
    cache_processor: An object implementing CacheProcessor to handle memory layout.
    eos_token_id: End-of-sequence token ID.

  Returns:
    A wrapped generation function matching the signature:
      def generate(params, inputs) -> generated_tokens
    where inputs is an int32 tensor of shape [batch_size, prompt_len] holding
    input token IDs, and generated_tokens is an int32 tensor of shape
    [batch_size, max_new_tokens] (or [batch_size, beam_size, max_new_tokens]
    if using beam search).
  """

  strategy = generation_config.strategy
  max_new_tokens = generation_config.max_output_tokens
  def neptune_apply_fn(
      params: jaxtyping.PyTree,
      inputs: jaxtyping.PyTree,
  ) -> jaxtyping.PyTree:
    """The E2E compiled generation loop execution logic."""
    if isinstance(inputs, dict):
      inputs = inputs.get("tokens", inputs.get("input_ids", next(iter(inputs.values()))))
    batch_size = inputs.shape[0]
    match strategy:
      case "beam_search":
        beam_size = generation_config.beam_size
        # Prefill phase: Process the input prompt (is_prefill=True) to
        # initialize the KV cache and get logits for the first token generation.
        initial_cache = initial_cache_factory(batch_size)
        packed_prefill_inputs = cache_processor.pack_inputs(inputs, None)
        prefill_logits, cache = model_forward_step_fn(
            params, packed_prefill_inputs, initial_cache, True
        )
        # Grab logits for the last token position. This assumes inputs are
        # LEFT-PADDED so that the last token is always the last predictor.
        # If using right-padding, this would evaluate pad tokens instead!
        last_token_logits = prefill_logits[:, -1, :]
        log_probs = jax.nn.log_softmax(last_token_logits, axis=-1)
        topk_log_probs, topk_tokens = jax.lax.top_k(log_probs, beam_size)
        replicated_cache = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, beam_size, axis=0), cache
        )
        output_tokens = jnp.zeros(
            (batch_size, beam_size, max_new_tokens), dtype=jnp.int32
        )
        output_tokens = output_tokens.at[:, :, 0].set(topk_tokens)
        beam_log_probs = topk_log_probs
        is_done = topk_tokens == eos_token_id
        vocab_size = prefill_logits.shape[-1]
        eos_mask = jnp.arange(vocab_size) == eos_token_id
        completed_step_log_probs = jnp.where(eos_mask, 0.0, -1e9)
        batch_idx = jnp.arange(batch_size)[:, None]
        batch_offsets = batch_idx * beam_size
        # Autoregressive Decode Loop (JAX While Loop).
        def cond_fn(state: BeamSearchState) -> jax.Array:
          # Continue decoding only if we haven't exceeded the max token budget
          beneath_ceiling = state.step < max_new_tokens
          # Check if all beams across all batch items have emitted the EOS token
          all_beams_done = jnp.all(state.loop_done)
          # Continue if beneath the ceiling AND not all beams are done
          return beneath_ceiling & ~all_beams_done
        def body_fn(state: BeamSearchState) -> BeamSearchState:
          """Performs one step of autoregressive beam search decoding.
          This function executes the forward pass for the next token, updates
          the
          KV cache, scores the candidate beams, and selects the top-K beams to
          continue the search.
          Args:
            state: The BeamSearchState holding tensors from the previous step.
          Returns:
            The updated BeamSearchState containing the incremented step, new
            generated
            tokens, and realigned KV cache for the next step.
          """
          # Get last generated token for all B * K beams
          last_tokens = state.current_tokens[:, :, state.step - 1]
          # Flatten to shape [B * K, 1] for model forward pass
          flat_last_tokens = last_tokens.reshape((-1, 1))
          # Run single-step decoder pass; get flat_logits shape [B * K, 1, vocab]
          packed_inputs = cache_processor.pack_inputs(
              flat_last_tokens, state.virtual_to_physical_map
          )
          flat_logits, next_cache = model_forward_step_fn(
              params, packed_inputs, state.loop_cache, False
          )
          # Reshape flat_logits to [B, K, vocab]
          logits = flat_logits.reshape((batch_size, beam_size, -1))
          # Compute step log probabilities -> [B, K, vocab]
          step_log_probs = jax.nn.log_softmax(logits, axis=-1)
          step_log_probs = jnp.where(
              state.loop_done[:, :, None],
              completed_step_log_probs[None, None, :],
              step_log_probs,
          )
          # Prune locally first: get top-K tokens per beam standalone
          topk_step_log_probs, topk_step_indices = jax.lax.top_k(
              step_log_probs, beam_size
          )
          # Compute cumulative scores ONLY on the survivors
          candidate_log_probs = (
              state.loop_log_probs[:, :, None] + topk_step_log_probs
          )
          # Flatten and get global survivors
          flat_candidates = candidate_log_probs.reshape((batch_size, -1))
          topk_log_probs, local_topk_indices = jax.lax.top_k(
              flat_candidates, beam_size
          )
          # Reconstruct the actual parent/word pointers
          parent_beam_indices = local_topk_indices // beam_size
          candidate_slot_indices = local_topk_indices % beam_size
          # Pull actual vocab word Id back from our Pruning pool
          next_tokens = topk_step_indices[
              batch_idx, parent_beam_indices, candidate_slot_indices
          ]
          # Update token history: Advanced indexing to gather rows
          shuffled_tokens = state.current_tokens[batch_idx, parent_beam_indices]
          next_tokens_buffer = shuffled_tokens.at[:, :, state.step].set(
              next_tokens
          )
          # Zero-copy logical indirection OR physical copying transparently
          next_loop_cache = cache_processor.shuffle_cache(
              next_cache, (batch_offsets + parent_beam_indices).reshape(-1)
          )
          # Update pointers for next step
          next_virtual_map = (batch_offsets + parent_beam_indices).reshape(-1)
          # Update is_done
          parent_done = state.loop_done[batch_idx, parent_beam_indices]
          next_done = parent_done | (next_tokens == eos_token_id)
          return BeamSearchState(
              step=state.step + 1,
              current_tokens=next_tokens_buffer,
              loop_cache=next_loop_cache,
              virtual_to_physical_map=next_virtual_map,
              loop_log_probs=topk_log_probs,
              loop_done=next_done,
          )
        initial_virtual_map = jnp.arange(batch_size * beam_size)
        initial_state = BeamSearchState(
            step=1,
            current_tokens=output_tokens,
            loop_cache=replicated_cache,
            virtual_to_physical_map=initial_virtual_map,
            loop_log_probs=beam_log_probs,
            loop_done=is_done,
        )
        final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
        return final_state.current_tokens
      case _:
        raise ValueError(f"Unsupported strategy: {strategy}")

  return neptune_apply_fn
