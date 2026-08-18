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

import functools
import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping

from neptune_model._src.jax2npt import typing as neptune_typing
from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import model_wrapper


tpu_inference_utils: Any = None
attention_metadata: Any = None
tpu_inference_kv_cache: Any = None

try:
  from tpu_inference import utils as _tpu_inference_utils
  from tpu_inference.layers.common import attention_metadata as _attention_metadata
  from tpu_inference.runner import kv_cache as _tpu_inference_kv_cache

  tpu_inference_utils = _tpu_inference_utils
  attention_metadata = _attention_metadata
  tpu_inference_kv_cache = _tpu_inference_kv_cache
  _TPU_INFERENCE_AVAILABLE = True
  _TPU_INFERENCE_IMPORT_ERROR = None
except ImportError as e:
  _TPU_INFERENCE_AVAILABLE = False
  _TPU_INFERENCE_IMPORT_ERROR = e


@functools.partial(
    jax.jit, static_argnames=("beam_width", "batch_size"), donate_argnums=(0,)
)
def _shuffle_kv_caches(
    kv_caches: list[jax.Array],
    parent_beam_ids: jax.Array,
    block_tables: jax.Array,
    beam_width: int,
    batch_size: int,
    current_block_idx: jax.Array,
) -> list[jax.Array]:
  """Shuffles KV cache physical pages on TPU when beams diverge across blocks.

  Args:
    kv_caches: List of layer-wise KV cache device buffers.
    parent_beam_ids: Selected parent beam indices of shape (batch_size,
      beam_width).
    block_tables: Current 3D block tables of shape (batch_size, beam_width,
      max_blocks).
    beam_width: Static number of beams.
    batch_size: Static batch size.
    current_block_idx: Physical block indices being decoded of shape
      (batch_size,).

  Returns:
    Updated list of KV cache buffers with synchronized page assignments.
  """
  kv_caches = jax.lax.optimization_barrier(kv_caches)
  batch_arange = jnp.arange(batch_size)[:, None]
  parent_blocks = block_tables[
      batch_arange, parent_beam_ids, current_block_idx[:, None]
  ].reshape(-1)
  child_blocks = block_tables[
      batch_arange, jnp.arange(beam_width)[None, :], current_block_idx[:, None]
  ].reshape(-1)

  total_beams = batch_size * beam_width

  def shuffle_layer(kv_cache: jax.Array) -> jax.Array:
    parent_data = kv_cache[parent_blocks]
    for i in range(total_beams):
      child_block = child_blocks[i]
      start_indices = (child_block, 0, 0, 0, 0)
      kv_cache = jax.lax.dynamic_update_slice(
          kv_cache, jnp.expand_dims(parent_data[i], 0), start_indices
      )
    return kv_cache

  new_caches = [shuffle_layer(c) for c in kv_caches]
  return jax.lax.optimization_barrier(new_caches)


@functools.partial(
    jax.jit, static_argnames=("beam_width", "batch_size"), donate_argnums=(0,)
)
def _cow_block_copy_jit(
    kv_caches: list[jax.Array],
    parent_blocks: jax.Array,
    child_blocks: jax.Array,
    beam_width: int,
    batch_size: int,
) -> list[jax.Array]:
  """Copies parent KV blocks to target child blocks (Copy-On-Write).

  Args:
    kv_caches: List of layer-wise KV cache device buffers.
    parent_blocks: Source physical block indices to copy from of shape
      (batch_size,).
    child_blocks: Target physical block indices for child beams of shape
      (batch_size, beam_width).
    beam_width: Static number of beams.
    batch_size: Static batch size.

  Returns:
    Updated list of KV cache buffers after performing the physical block copy.
  """
  total_beams = batch_size * beam_width
  flat_child_blocks = child_blocks.reshape(-1)
  flat_parent_blocks = jnp.repeat(parent_blocks, beam_width)

  new_kv_caches = []
  for layer_cache in kv_caches:
    parent_data = layer_cache[flat_parent_blocks]
    for i in range(total_beams):
      child_block = flat_child_blocks[i]
      start_indices = (child_block, 0, 0, 0, 0)
      layer_cache = jax.lax.dynamic_update_slice(
          layer_cache, jnp.expand_dims(parent_data[i], axis=0), start_indices
      )
    new_kv_caches.append(layer_cache)
  return new_kv_caches


@jax.jit(static_argnames=("beam_width",))
def _select_next_beams(
    logits_step: jax.Array, cum_logprobs: jax.Array, beam_width: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Selects top-k candidates across beams for the beam search step.

  Args:
    logits_step: Decoder output logits for current step, shape (beam_width,
      vocab_size).
    cum_logprobs: Cumulative log probabilities so far, shape (beam_width,).
    beam_width: Static number of active beams.

  Returns:
    parent_beam_ids: Selected parent beam index for each new beam hypothesis.
    token_ids: Newly selected token ID for each beam hypothesis.
    top_scores: Updated cumulative log probabilities for the top hypotheses.
  """
  logprobs_step = jax.nn.log_softmax(logits_step, axis=-1)
  total_logprobs = logprobs_step + cum_logprobs[:, None]
  top_scores_per_beam, top_indices_per_beam = jax.lax.top_k(
      total_logprobs, k=2 * beam_width
  )
  flat_scores = top_scores_per_beam.ravel()
  flat_indices = top_indices_per_beam.ravel()
  top_scores, global_top_indices = jax.lax.top_k(flat_scores, k=beam_width)
  parent_beam_ids = global_top_indices // (2 * beam_width)
  token_ids = flat_indices[global_top_indices]
  return parent_beam_ids, token_ids, top_scores


def _require_tpu_inference():
  if not _TPU_INFERENCE_AVAILABLE:
    raise ImportError(
        "Failed to import tpu_inference. Please make sure that tpu_inference"
        " support is installed (e.g. via `pip install"
        " neptune-model[tpu_inference]`)."
    ) from _TPU_INFERENCE_IMPORT_ERROR


def _sanitize_model_fn(tpu_runner: Any) -> Any:
  """Extracts the model step function safe for outer JIT compilation.

  Background:
    JAX forbids `compiler_options` on inner `@jax.jit` functions when invoked
    inside an outer `@jax.jit` or `jax.export.export` context (e.g. Neptune's
    `generate_fn`), raising:
      `ValueError: compiler_options can only be passed to top-level jax.jit.`

  Resolution Strategy:
    1. VLLM / TorchAX models: `tpu_inference` provides `step_fn_no_options` or
       `step_fun_impl` on the underlying `VllmModelWrapper` (see
       `tpu_inference/models/vllm/vllm_model_wrapper.py`).
    2. Flax NNX models: In
       `tpu_inference/models/common/model_loader.py:run_model`, the forward
       step is defined as:
         `state = jax.tree_util.tree_unflatten(treedef, state_leaves)`
         `model = nnx.merge(graphdef, state)`
         `return model(*args)`
       decorated with `@jax.jit(compiler_options=...)`. We execute this exact
       same forward pass directly without the nested `@jax.jit` decorator.
    3. Fallback: Return `tpu_runner.model_fn`.

  Args:
    tpu_runner: Concrete TPUModelRunner instance.

  Returns:
    Step function safe for invocation within outer JIT contexts.
  """
  # 1. Check for un-optioned step function exposed by VLLM / TorchAX wrappers.
  # Reference:
  # tpu_inference/models/vllm/vllm_model_wrapper.py (step_fn_no_options)
  inner_wrapper = getattr(tpu_runner, "model", None)
  if inner_wrapper is not None:
    unoptioned_step_fn = getattr(
        inner_wrapper, "step_fun_impl", None
    ) or getattr(inner_wrapper, "step_fn_no_options", None)
    if unoptioned_step_fn is not None:
      return unoptioned_step_fn

  # 2. For Flax NNX models, execute the model forward pass directly.
  # Reference: tpu_inference/models/common/model_loader.py (run_model)
  # `run_model` in model_loader.py executes `nnx.merge(graphdef, state)(*args)`
  # inside a `@jax.jit(compiler_options=...)`. To avoid the nested JIT error in
  # Neptune, we perform the exact same unflatten + merge without the decorator.
  if hasattr(tpu_runner, "model") and hasattr(tpu_runner, "state"):
    try:
      graphdef, _ = nnx.split(tpu_runner.model)
      state_treedef = jax.tree_util.tree_structure(tpu_runner.state)

      def direct_flax_forward(state_leaves, *args, **kwargs):
        kwargs.pop("shared_attention_metadata", None)
        kwargs.pop("spec_step_idx", None)
        state = jax.tree_util.tree_unflatten(state_treedef, state_leaves)
        model = nnx.merge(graphdef, state)
        return model(*args, **kwargs)

      return direct_flax_forward
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  # 3. Default fallback.
  return tpu_runner.model_fn


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
    self._model_fn = _sanitize_model_fn(tpu_runner)
    self.max_prompt_length = generation_config.max_prompt_length
    if (
        generation_config.sampling_method
        == llm_config.SamplingMethod.BEAM_SEARCH
    ):
      if generation_config.beam_size is None:
        raise ValueError(
            "generation_config.beam_size must be specified for beam search."
        )
      self.beam_width = generation_config.beam_size
      self.max_decode_tokens = generation_config.max_output_tokens
    elif generation_config.sampling_method == llm_config.SamplingMethod.SCORING:
      if generation_config.num_candidates is None:
        raise ValueError(
            "generation_config.num_candidates must be specified for scoring."
        )
      if generation_config.max_candidate_tokens is None:
        raise ValueError(
            "generation_config.max_candidate_tokens must be specified for"
            " scoring."
        )
      self.beam_width = generation_config.num_candidates
      self.max_decode_tokens = generation_config.max_candidate_tokens
    else:
      self.beam_width = 1
      self.max_decode_tokens = generation_config.max_output_tokens

    self.max_seq_len = self.max_prompt_length + self.max_decode_tokens

    self.block_size = tpu_runner.block_size
    self.max_blocks_per_seq = math.ceil(self.max_seq_len / self.block_size)
    self.max_prompt_blocks = math.ceil(self.max_prompt_length / self.block_size)
    self.max_decode_blocks = (
        math.ceil(self.max_decode_tokens / self.block_size) + 1
    )

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
      num_blocks = (
          batch_size * self.max_prompt_blocks
          + batch_size * self.beam_width * self.max_decode_blocks
      )
    else:
      num_blocks = batch_size * self.max_blocks_per_seq

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

    max_allocated_tokens = (
        self.max_prompt_blocks * self.block_size
        if self.beam_width > 1
        else self.max_blocks_per_seq * self.block_size
    )
    if prompt_input_len > max_allocated_tokens:
      raise ValueError(
          f"Prompt length ({prompt_input_len} tokens) exceeds allocated prompt"
          f" block capacity ({max_allocated_tokens} tokens)."
      )

    # Reconstruct AttentionMetadata for Prefill
    input_positions = jnp.tile(
        jnp.arange(prompt_input_len, dtype=jnp.int32), (batch_size, 1)
    ).ravel()

    prefill_block_tables = jnp.zeros(
        (batch_size, self.max_blocks_per_seq), dtype=jnp.int32
    )
    prompt_stride = (
        self.max_prompt_blocks
        if self.beam_width > 1
        else self.max_blocks_per_seq
    )
    prompt_blocks = (
        jnp.arange(self.max_prompt_blocks, dtype=jnp.int32)[None, :]
        + (jnp.arange(batch_size, dtype=jnp.int32) * prompt_stride)[:, None]
    )
    prefill_block_tables = prefill_block_tables.at[
        :, : self.max_prompt_blocks
    ].set(prompt_blocks)
    block_tables = prefill_block_tables.reshape(-1)

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
    model_outputs = self._model_fn(
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
      block_indices = jnp.arange(self.max_blocks_per_seq)[None, :]
      batch_ids = jnp.arange(batch_size)[:, None]
      cur_block_tables = block_indices + batch_ids * self.max_blocks_per_seq
      next_cache = (kv_caches, cur_block_tables)

    return logits, next_cache

  def _replicate_cache(
      self,
      cache: jaxtyping.PyTree,
      beam_width: int,
      prompt_len: jax.Array,
  ) -> jaxtyping.PyTree:
    """Prepares and replicates the cache block tables for decoding multiple beams.

    Allocates private physical blocks for child beams and applies Copy-on-Write
    (CoW) physical block copying on TPU for the boundary prompt block.

    Args:
      cache: Tuple of (kv_caches, _) from the prefill stage.
      beam_width: Number of active beam search hypotheses.
      prompt_len: 1D array of shape (batch_size,) containing prompt token
        lengths.

    Returns:
      Updated cache tuple (kv_caches, cur_block_tables) configured for
      multi-beam decode.
    """
    kv_caches, _ = cache
    batch_size = prompt_len.shape[0]
    last_prompt_block_idx = (prompt_len - 1) // self.block_size  # (batch_size,)

    # Construct 3D block table mapping:
    # - Columns [0, last_prompt_col) point to shared prompt physical blocks.
    # - Columns [last_prompt_col, max_blocks_per_seq) point to private decode
    #   blocks allocated per (batch_idx, beam_idx).
    col_indices = jnp.arange(self.max_blocks_per_seq, dtype=jnp.int32)[
        None, None, :
    ]
    batch_prompt_offset = (
        jnp.arange(batch_size, dtype=jnp.int32) * self.max_prompt_blocks
    )[:, None, None]
    shared_blocks = batch_prompt_offset + jnp.minimum(
        col_indices, self.max_prompt_blocks - 1
    )

    beam_indices = jnp.arange(beam_width, dtype=jnp.int32)[None, :, None]
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None, None]
    seq_idx = batch_indices * beam_width + beam_indices

    private_base = (
        batch_size * self.max_prompt_blocks + seq_idx * self.max_decode_blocks
    )

    last_prompt_col = last_prompt_block_idx[:, None, None]
    rel_decode_col = jnp.maximum(0, col_indices - last_prompt_col)
    private_blocks = private_base + rel_decode_col

    is_shared = col_indices < last_prompt_col
    cur_block_tables = jnp.where(is_shared, shared_blocks, private_blocks)

    # Perform Copy-on-Write (CoW) on TPU: copy the boundary prompt block into
    # each child beam's initial private decode block so they can write
    # subsequent tokens without mutating sibling beams.
    batch_arange = jnp.arange(batch_size)
    parent_blocks = (
        batch_arange * self.max_prompt_blocks + last_prompt_block_idx
    )
    child_blocks = cur_block_tables[
        batch_arange[:, None],
        jnp.arange(beam_width)[None, :],
        last_prompt_block_idx[:, None],
    ]

    kv_caches = _cow_block_copy_jit(
        kv_caches, parent_blocks, child_blocks, beam_width, batch_size
    )

    return (kv_caches, cur_block_tables)

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
    batch_size = prompt_len.shape[0]
    beam_width = num_active_seqs // batch_size

    # Calculate token positions and sequence lengths
    prompt_len_rep = jnp.repeat(prompt_len, beam_width)
    cur_positions = prompt_len_rep - 1 + step
    cur_seq_lens = prompt_len_rep + step
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
    model_step_outputs = self._model_fn(
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

  def compile_generation_loop(self) -> neptune_typing.ApplyFn:
    """Compiles an optimized autoregressive generation loop for TPU serving.

    For 'beam_search' sampling, compiles a JIT-optimized beam search loop
    tracking
    log probabilities, beam history backtracking, and physical KV cache page
    shuffling.
    For standard sampling (greedy / top-k), delegates to
    super().compile_generation_loop.

    Returns:
      A JAX apply function accepting (params, inputs) and returning generated
      token IDs and cumulative log probabilities.
    """
    config = self.get_generation_config()
    if config.pad_token_id is None:
      raise ValueError(
          "pad_token_id must be already set in the generation config."
      )
    sampling_method = config.sampling_method

    if sampling_method == llm_config.SamplingMethod.SCORING:
      num_candidates = self.beam_width
      max_candidate_tokens = self.max_decode_tokens

      def score_candidates(
          params: jaxtyping.PyTree,
          inputs: dict[str, jax.Array],
      ) -> dict[str, jax.Array]:
        """Evaluates candidate completion log-likelihood in a single parallel extend pass.

        Instead of looping over sequential decode steps, candidate scoring runs:
        1. Prefill on the prompt to populate shared KV cache and score step 0.
        2. A single causal extend pass over all candidate continuation tokens in
        parallel.
        3. Chunked LM head projection to compute exact cumulative
        log-probabilities.

        Args:
          params: Model parameters PyTree.
          inputs: Dictionary of input tensors, including: - "input_ids": Prompt
            token IDs, shape (batch_size, prompt_len). - "candidate_ids":
            Candidate token IDs, shape (batch_size, num_candidates,
            max_candidate_tokens).

        Returns:
          A dictionary of candidate scores and intermediate log-probabilities.
        """
        input_ids = inputs["input_ids"]
        candidate_ids = inputs["candidate_ids"]
        batch_size = input_ids.shape[0]

        prompt_len = jnp.sum(
            input_ids != config.pad_token_id, axis=-1, dtype=jnp.int32
        )

        # 1. Prefill prompt once to populate KV cache and obtain initial step 0
        # logits.
        empty_cache = self.initial_cache_factory(batch_size)
        prefill_logits, cache = self.prefill(
            params, input_ids, empty_cache, prompt_len=prompt_len
        )

        # 2. Extract step 0 log-probabilities: log P(y_1 | prompt).
        prefill_logprobs = jax.nn.log_softmax(prefill_logits, axis=-1)
        first_tokens = candidate_ids[:, :, 0]
        step0_logprobs = jnp.take_along_axis(
            prefill_logprobs[:, None, :], first_tokens[:, :, None], axis=-1
        ).squeeze(-1)
        is_first_token_valid = (first_tokens != config.pad_token_id).astype(
            jnp.float32
        )
        step0_logprobs = step0_logprobs * is_first_token_valid

        if max_candidate_tokens <= 1:
          return {
              "scores": step0_logprobs,
              "step0_logprobs": step0_logprobs,
              "continuation_scores": jnp.zeros_like(step0_logprobs),
          }

        # 3. Replicate KV cache table across all candidates
        cache = self._replicate_cache(
            cache, num_candidates, prompt_len=prompt_len
        )
        kv_caches, cur_block_tables = cache

        # 4. Single-pass parallel extend forward pass over all candidate
        # suffixes.
        extend_tokens_len = max_candidate_tokens - 1
        extend_inputs = candidate_ids[:, :, :-1]
        target_tokens = candidate_ids[:, :, 1:]

        total_seqs = batch_size * num_candidates
        flat_extend_inputs = extend_inputs.reshape(
            total_seqs * extend_tokens_len
        )

        prompt_len_rep = jnp.repeat(prompt_len, num_candidates)
        seq_offsets = (
            prompt_len_rep[:, None]
            + jnp.arange(extend_tokens_len, dtype=jnp.int32)[None, :]
        )
        extend_positions = seq_offsets.reshape(-1)

        seq_lens = prompt_len_rep + extend_tokens_len
        query_start_loc = (
            jnp.arange(total_seqs + 1, dtype=jnp.int32) * extend_tokens_len
        )
        request_distribution = jnp.array([0, 0, total_seqs], dtype=jnp.int32)

        cur_block_tables_1d = cur_block_tables.reshape(-1)

        assert attention_metadata is not None
        extend_attn_metadata = attention_metadata.AttentionMetadata(
            input_positions=extend_positions,
            block_tables=cur_block_tables_1d,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )

        state_leaves = jax.tree_util.tree_leaves(params)
        model_extend_outputs = self._model_fn(
            state_leaves,
            kv_caches,
            flat_extend_inputs,
            extend_attn_metadata,
            None,
            extend_positions,
            self.kv_cache_indices,
            None,
            None,
            True,
            True,
        )
        extend_hidden_states = model_extend_outputs[1]

        # 5. Target token logprobs computation in chunks to avoid HBM exhaustion
        flat_target_tokens = target_tokens.reshape(-1)
        num_flat_tokens = flat_target_tokens.shape[0]
        chunk_size = 2048
        num_chunks = (num_flat_tokens + chunk_size - 1) // chunk_size
        padded_num_tokens = num_chunks * chunk_size

        pad_len = padded_num_tokens - num_flat_tokens
        padded_hidden_states = jnp.pad(
            extend_hidden_states, ((0, pad_len), (0, 0))
        )
        padded_targets = jnp.pad(
            flat_target_tokens,
            (0, pad_len),
            constant_values=config.pad_token_id,
        )

        chunked_hidden = padded_hidden_states.reshape(
            num_chunks, chunk_size, -1
        )
        chunked_targets = padded_targets.reshape(num_chunks, chunk_size)

        def _compute_chunk_logprobs(carry, chunk_data):
          h_chunk, t_chunk = chunk_data
          c_logits = self.tpu_runner.compute_logits_fn(
              state_leaves, h_chunk, None
          )
          c_logits = c_logits.astype(jnp.float32)
          c_logprobs = jax.nn.log_softmax(c_logits, axis=-1)
          c_target_logprobs = jnp.take_along_axis(
              c_logprobs, t_chunk[:, None], axis=-1
          ).squeeze(-1)
          return carry, c_target_logprobs

        _, chunked_results = jax.lax.scan(
            _compute_chunk_logprobs, None, (chunked_hidden, chunked_targets)
        )
        flat_step_logprobs = chunked_results.reshape(-1)[:num_flat_tokens]
        step_logprobs = flat_step_logprobs.reshape(
            batch_size, num_candidates, extend_tokens_len
        )

        is_valid_token = (target_tokens != config.pad_token_id).astype(
            jnp.float32
        )
        masked_step_logprobs = step_logprobs * is_valid_token
        continuation_scores = jnp.sum(masked_step_logprobs, axis=-1)

        total_scores = step0_logprobs + continuation_scores

        return {
            "scores": total_scores,
            "step0_logprobs": step0_logprobs,
            "continuation_scores": continuation_scores,
        }

      return score_candidates

    if sampling_method != llm_config.SamplingMethod.BEAM_SEARCH:
      return super().compile_generation_loop()

    max_output_tokens = config.max_output_tokens
    beam_width = self.beam_width

    def generate(
        params: jaxtyping.PyTree,
        inputs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
      input_ids = inputs["input_ids"]
      prompt_len = jnp.sum(
          input_ids != config.pad_token_id, axis=-1, dtype=jnp.int32
      )
      batch_size = input_ids.shape[0]

      # 1. Allocate initial cache and run prefill pass
      empty_cache = self.initial_cache_factory(batch_size)
      logits, cache = self.prefill(
          params, input_ids, empty_cache, prompt_len=prompt_len
      )

      # 2. Initialize beams from prefill logits: logits shape
      # (batch_size, vocab_size)
      logprobs_step = jax.nn.log_softmax(logits, axis=-1)
      top_logprobs, top_token_ids = jax.lax.top_k(logprobs_step, k=beam_width)

      next_tokens = top_token_ids.reshape(batch_size * beam_width, 1)
      cum_logprobs = top_logprobs  # shape (batch_size, beam_width)

      all_tokens = jnp.zeros(
          (max_output_tokens, batch_size, beam_width, 1), dtype=jnp.int32
      )
      all_tokens = all_tokens.at[0].set(
          top_token_ids.reshape(batch_size, beam_width, 1)
      )

      # 3. Replicate prompt KV cache table across beams
      cache = self._replicate_cache(cache, beam_width, prompt_len=prompt_len)

      # 4. Run decode loop using self.decode_step
      vmapped_select = jax.vmap(_select_next_beams, in_axes=(0, 0, None))

      for step in range(1, max_output_tokens):
        logits_step, cache = self.decode_step(
            params=params,
            token_ids=next_tokens,
            cache=cache,
            step=step,
            prompt_len=prompt_len,
        )
        kv_caches, cur_block_tables = cache

        # 4a. Select top-k candidate beams per batch item independently.
        logits_step_3d = logits_step.reshape(batch_size, beam_width, -1)
        parent_beam_ids, token_ids, cum_logprobs = vmapped_select(
            logits_step_3d, cum_logprobs, beam_width
        )

        # 4b. For completed block table columns, update child beams to reference
        # their selected parent beam's completed physical blocks.
        cur_seq_len_val = prompt_len + step
        current_block_idx = (cur_seq_len_val - 1) // self.block_size
        col_indices = jnp.arange(self.max_blocks_per_seq, dtype=jnp.int32)[
            None, None, :
        ]
        is_past_col = col_indices < current_block_idx[:, None, None]
        parent_block_tables = jnp.take_along_axis(
            cur_block_tables, parent_beam_ids[:, :, None], axis=1
        )
        cur_block_tables = jnp.where(
            is_past_col, parent_block_tables, cur_block_tables
        )

        # 4c. Synchronize active decode KV cache pages on TPU when beam
        # parentage diverges.
        if step < max_output_tokens - 1:
          kv_caches = _shuffle_kv_caches(
              kv_caches,
              parent_beam_ids,
              cur_block_tables,
              beam_width,
              batch_size,
              current_block_idx,
          )

        cache = (kv_caches, cur_block_tables)
        next_tokens = token_ids.reshape(batch_size * beam_width, 1)
        all_tokens = all_tokens.at[step].set(
            token_ids.reshape(batch_size, beam_width, 1)
        )

        # 4d. Reorder token generation history according to selected parent
        # beams.
        all_tokens_prev = all_tokens[:step]
        all_tokens_prev_reordered = jnp.take_along_axis(
            all_tokens_prev, parent_beam_ids[None, :, :, None], axis=2
        )
        all_tokens = all_tokens.at[:step].set(all_tokens_prev_reordered)

      decoded_tokens = jnp.squeeze(all_tokens, axis=-1).transpose(1, 2, 0)

      return {
          "token_ids": decoded_tokens,
          "logprobs": cum_logprobs,
      }

    return generate
