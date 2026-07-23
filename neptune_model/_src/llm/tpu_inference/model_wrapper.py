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

"""TPU-Inference wrapper for Neptune LLM Serving."""

# pylint: disable=g-import-not-at-top

import math

from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping

try:
  # pytype: disable=import-error
  from tpu_inference import utils as tpu_inference_utils
  from tpu_inference.layers.common import attention_metadata
  from tpu_inference.runner import kv_cache as tpu_inference_kv_cache
  # pytype: enable=import-error
  _TPU_INFERENCE_AVAILABLE = True
  _TPU_INFERENCE_IMPORT_ERROR = None
except ImportError as e:
  tpu_inference_utils = None
  attention_metadata = None
  tpu_inference_kv_cache = None
  _TPU_INFERENCE_AVAILABLE = False
  _TPU_INFERENCE_IMPORT_ERROR = e

from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import model_wrapper


def _require_tpu_inference():
  if not _TPU_INFERENCE_AVAILABLE:
    raise ImportError(
        "Failed to import tpu_inference. Please make sure that tpu_inference"
        " support is installed (e.g. via `pip install"
        " neptune-model[tpu_inference]`)."
    ) from _TPU_INFERENCE_IMPORT_ERROR


class TPUInferenceWrapper(model_wrapper.LLMModelWrapper):
  """Adapts a TPUModelRunner from tpu-inference to the Neptune LLM framework."""

  def __init__(
      self,
      tpu_runner: Any,
      generation_config: llm_config.GenerationConfig,
  ):
    """Initializes the TPU inference wrapper using GenerationConfig.

    Args:
      tpu_runner: Concrete TPUModelRunner instance loaded with model weights,
        device mesh, and KV cache specifications.
      generation_config: Generation configuration.

    Raises:
      ImportError: If the 'tpu_inference' package is not available.
    """
    _require_tpu_inference()
    super().__init__(generation_config=generation_config)

    self.tpu_runner = tpu_runner
    self.max_seq_len = (
        generation_config.max_prompt_length
        + generation_config.max_output_tokens
    )
    self.max_output_tokens = generation_config.max_output_tokens
    if (
        generation_config.sampling_method
        == llm_config.SamplingMethod.BEAM_SEARCH
    ):
      if generation_config.beam_size is None:
        raise ValueError(
            "generation_config.beam_size must be specified for beam search."
        )
      self.beam_width = generation_config.beam_size
    else:
      self.beam_width = 1

    self.block_size = tpu_runner.block_size
    # TODO(armandouv): We can be more conservative in the number of blocks
    # reserved for each beam, since the prefix blocks are shared across them.
    self.max_blocks_per_beam = math.ceil(self.max_seq_len / self.block_size)

    # Reconstruct layer name cache indices configuration
    kv_cache_specs = tpu_runner.kv_cache_manager.get_kv_cache_spec()
    sorted_layers = sorted(
        kv_cache_specs.keys(),
        key=lambda x: int(x.split(".")[-1])
        if x.split(".")[-1].isdigit()
        else 0,
    )
    layer_name_to_kvcache_index = {}
    for idx, layer_name in enumerate(sorted_layers):
      layer_name_to_kvcache_index[layer_name] = idx

    # Apply shared layers mapping
    for (
        layer_name,
        target_layer_name,
    ) in tpu_runner.kv_cache_manager.shared_kv_cache_layers.items():
      layer_name_to_kvcache_index[layer_name] = layer_name_to_kvcache_index[
          target_layer_name
      ]

    self.kv_cache_indices = tuple(layer_name_to_kvcache_index.items())
    self.max_idx = max(layer_name_to_kvcache_index.values())
    self.kv_cache_specs = kv_cache_specs
    self.layer_name_to_kvcache_index = layer_name_to_kvcache_index

  def initial_cache_factory(self, batch_size: int = 1) -> jaxtyping.PyTree:
    """Allocates a physical KV Cache on TPU HBM.

    Args:
      batch_size: Batch size of the input sequences (defaults to 1).

    Returns:
      An allocated KV cache PyTree.
    """
    if self.beam_width > 1:
      num_blocks = batch_size * self.beam_width * self.max_blocks_per_beam
    else:
      num_blocks = batch_size * self.max_blocks_per_beam

    kv_caches = [None] * (self.max_idx + 1)
    for layer_name, idx in self.layer_name_to_kvcache_index.items():
      if kv_caches[idx] is not None:
        continue
      spec = self.kv_cache_specs[layer_name]
      assert tpu_inference_kv_cache is not None
      assert tpu_inference_utils is not None
      caches = tpu_inference_kv_cache.create_kv_caches(
          num_blocks=num_blocks,
          block_size=spec.block_size,
          num_kv_heads=spec.num_kv_heads,
          head_size=spec.head_size,
          mesh=self.tpu_runner.mesh,
          layer_names=[layer_name],
          cache_dtype=tpu_inference_utils.to_jax_dtype(spec.dtype),
          use_mla=self.tpu_runner.model_config.use_mla,
      )
      kv_caches[idx] = caches[0]
    return kv_caches

  def prefill(
      self,
      params: jaxtyping.PyTree,
      input_ids: jax.Array,
      cache: jaxtyping.PyTree,
      prompt_len: jax.Array | None = None,
  ) -> tuple[jax.Array, jaxtyping.PyTree]:
    """Runs the prompt prefill pass and initializes KV cache for prompt tokens.

    Constructs PagedAttention metadata for the prefill request distribution,
    executes the model forward pass on TPU, and slices logits at the last valid
    prompt token position.

    Args:
      params: Model parameters PyTree.
      input_ids: 2D array of prompt token IDs with shape (batch_size,
        prompt_input_len).
      cache: Physical KV cache PyTree allocated by initial_cache_factory.
      prompt_len: Optional 1D array of actual unpadded prompt token lengths.

    Returns:
      A tuple of (logits, next_cache) where logits has shape (batch_size,
      vocab_size)
      and next_cache is a tuple of (kv_caches, cur_block_tables).
    """
    kv_caches = cache
    batch_size, prompt_input_len = input_ids.shape
    if prompt_len is None:
      prompt_len = jnp.sum(
          input_ids != self.get_generation_config().pad_token_id,
          axis=-1,
          dtype=jnp.int32,
      )

    max_allocated_tokens = self.max_blocks_per_beam * self.block_size
    if prompt_input_len + self.max_output_tokens > max_allocated_tokens:
      raise ValueError(
          f"Generation length ({prompt_input_len} prompt tokens + "
          f"{self.max_output_tokens} output tokens) exceeds allocated block "
          f"capacity ({max_allocated_tokens} tokens). Surpassing allocated "
          "block capacity is not supported yet."
      )

    # Reconstruct AttentionMetadata for Prefill
    input_positions = jnp.tile(
        jnp.arange(prompt_input_len, dtype=jnp.int32), (batch_size, 1)
    ).ravel()

    block_tables = jnp.arange(
        batch_size * self.max_blocks_per_beam, dtype=jnp.int32
    )

    assert attention_metadata is not None
    prefill_attn_metadata = attention_metadata.AttentionMetadata(
        input_positions=input_positions,
        block_tables=block_tables,
        seq_lens=jnp.full((batch_size,), prompt_input_len, dtype=jnp.int32),
        query_start_loc=(
            jnp.arange(batch_size + 1, dtype=jnp.int32) * prompt_input_len
        ),
        request_distribution=jnp.array([0, 0, batch_size], dtype=jnp.int32),
    )

    state_leaves = jax.tree_util.tree_leaves(params)
    model_outputs = self.tpu_runner.model_fn(
        state_leaves,
        kv_caches,
        input_ids.ravel(),
        prefill_attn_metadata,
        None,
        input_positions,
        self.kv_cache_indices,
        None,
        None,
        True,
        True,
    )
    kv_caches, hidden_states = model_outputs[0], model_outputs[1]

    # Slice logits at the last token index of the prompt
    last_token_indices = jnp.arange(
        batch_size, dtype=jnp.int32
    ) * prompt_input_len + (prompt_len - 1)
    last_hidden_state = jnp.take(hidden_states, last_token_indices, axis=0)
    logits = self.tpu_runner.compute_logits_fn(
        state_leaves, last_hidden_state, None
    )
    logits = logits.astype(jnp.float32)

    # Initialize decode block tables structure
    if self.beam_width > 1:
      next_cache = (kv_caches, None)
    else:
      block_indices = jnp.arange(self.max_blocks_per_beam)[None, :]
      batch_ids = jnp.arange(batch_size)[:, None]
      cur_block_tables = block_indices + batch_ids * self.max_blocks_per_beam
      next_cache = (kv_caches, cur_block_tables)

    return logits, next_cache

  def decode_step(
      self,
      params: jaxtyping.PyTree,
      token_ids: jax.Array,
      cache: jaxtyping.PyTree,
      step: int,
      prompt_len: jax.Array,
  ) -> tuple[jax.Array, jaxtyping.PyTree]:
    """Runs a single autoregressive decoding step.

    Constructs step attention metadata for active decode sequences, executes
    the TPU model forward pass, and returns step logits.

    Args:
      params: Model parameters PyTree.
      token_ids: 2D array of the latest generated tokens with shape
        (num_active_seqs, 1).
      cache: Current cache tuple (kv_caches, cur_block_tables).
      step: Current 1-indexed decode step (1 <= step < max_output_tokens).
      prompt_len: 1D array of shape (batch_size,) containing prompt token
        lengths.

    Returns:
      A tuple of (logits_step, next_cache) where logits_step has shape
      (num_active_seqs, vocab_size) and next_cache contains updated KV cache
      state.
    """
    kv_caches, cur_block_tables = cache
    num_active_seqs = token_ids.shape[0]

    # Calculate token positions and sequence lengths
    cur_positions = jnp.broadcast_to(prompt_len, (num_active_seqs,)) - 1 + step
    cur_seq_lens = jnp.broadcast_to(prompt_len, (num_active_seqs,)) + step
    cur_block_tables_1d = cur_block_tables.reshape(-1)

    query_start_loc = jnp.arange(num_active_seqs + 1, dtype=jnp.int32)
    request_distribution = jnp.array([num_active_seqs, 0, 0], dtype=jnp.int32)

    assert attention_metadata is not None
    step_attn_metadata = attention_metadata.AttentionMetadata(
        input_positions=cur_positions,
        block_tables=cur_block_tables_1d,
        seq_lens=cur_seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )

    state_leaves = jax.tree_util.tree_leaves(params)
    model_step_outputs = self.tpu_runner.model_fn(
        state_leaves,
        kv_caches,
        token_ids.ravel(),
        step_attn_metadata,
        None,
        cur_positions,
        self.kv_cache_indices,
        None,
        None,
        True,
        True,
    )
    kv_caches, hidden_states = model_step_outputs[0], model_step_outputs[1]

    logits_step = self.tpu_runner.compute_logits_fn(
        state_leaves, hidden_states, None
    )
    logits_step = logits_step.astype(jnp.float32)

    return logits_step, (kv_caches, cur_block_tables)
