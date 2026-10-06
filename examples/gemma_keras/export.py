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

"""XLA SavedModel export for Gemma3 with prefill + greedy decode in the graph.

Decode runs in a ``jax.lax.while_loop``, so the KV cache never leaves the graph:
one PredictRequest yields one full generation.

``prompt_len`` and ``eos_token_id`` are [batch]-shaped because the serving
batcher needs a leading batch dim, but every row carries the same value (row 0
is read).

Import via ``examples.gemma_keras`` so ``KERAS_BACKEND=jax`` is set first.
"""

import os
import re
import tempfile

import jax
import jax.numpy as jnp
import keras
import keras_hub
import numpy as np
from orbax import checkpoint as ocp

from examples.gemma_keras import DEFAULT_MODEL_PRESET
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config


DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_MAX_NEW_TOKENS = 64


def _empty_kv_cache(batch_size, num_layers, total_len, num_kv_heads, head_dim):
  """Returns a zero KV cache in the layout `call_with_cache` expects.

  keras_hub 0.32 replaced the single stacked
  [batch, layers, 2, len, heads, head_dim] tensor with a tuple of per-layer
  (key, value) pairs, each [batch, len, heads, head_dim].
  """
  # Tolerates suffixes like '0.32.0rc0'.
  match = re.match(r'(\d+)\.(\d+)', keras_hub.__version__)
  if not match:
    raise ValueError(
        f'Cannot parse keras_hub version: {keras_hub.__version__!r}'
    )
  if (int(match.group(1)), int(match.group(2))) < (0, 32):
    return jnp.zeros(
        (batch_size, num_layers, 2, total_len, num_kv_heads, head_dim),
        dtype=jnp.float32,
    )
  layer_shape = (batch_size, total_len, num_kv_heads, head_dim)
  return tuple(
      (
          jnp.zeros(layer_shape, dtype=jnp.float32),
          jnp.zeros(layer_shape, dtype=jnp.float32),
      )
      for _ in range(num_layers)
  )


def export_model(
    export_path: str,
    dtype: str = 'fp32',
    preset: str = DEFAULT_MODEL_PRESET,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
  """Exports Gemma3 with an in-graph generation loop.

  Args:
    export_path: Directory to write the exported model to.
    dtype: 'fp32' or 'bf16' (Keras mixed_bfloat16 policy).
    preset: keras_hub preset name or local path.
    max_seq_len: Padded prompt length; longer prompts need a re-export.
    max_new_tokens: Max tokens to generate; decode stops early at EOS.

  Limitation: `prompt_len` and `eos_token_id` come from row 0 (the cache index
  is a scalar), so batched requests must share them; else use batch size 1.
  """
  if dtype not in ('fp32', 'bf16'):
    raise ValueError(f"Unsupported dtype: {dtype}. Expected 'fp32' or 'bf16'.")

  total_len = max_seq_len + max_new_tokens

  # The mixed-precision policy is process-global; restored in `finally`.
  original_policy = keras.mixed_precision.global_policy()
  if dtype == 'bf16':
    keras.mixed_precision.set_global_policy('mixed_bfloat16')

  try:
    gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(preset)
    backbone = gemma_lm.backbone
    num_layers = backbone.num_layers
    num_kv_heads = backbone.num_key_value_heads
    head_dim = backbone.head_dim

    all_variables = (
        gemma_lm.trainable_variables + gemma_lm.non_trainable_variables
    )
    params = [v.value for v in all_variables]

    def apply_fn(params, inputs):
      token_ids = inputs['token_ids']            # [batch, max_seq_len] int32
      # Row 0 applies to the whole batch (see docstring).
      # Clamp so bad values can't read padding or unwritten cache slots.
      prompt_len = jnp.clip(inputs['prompt_len'][0], 1, max_seq_len)
      eos_token_id = inputs['eos_token_id'][0]
      batch_size = token_ids.shape[0]

      # call_with_cache reads Keras variables; bind them to `params` for the
      # whole pipeline, including the traced while_loop body.
      state_mapping = list(zip(all_variables, params))
      with keras.StatelessScope(state_mapping=state_mapping):
        # fp32 KV cache regardless of policy; the model casts as needed.
        cache = _empty_kv_cache(
            batch_size, num_layers, total_len, num_kv_heads, head_dim
        )
        # 1 for real prompt positions; extended as tokens are generated.
        positions = jnp.arange(total_len, dtype=jnp.int32)
        prompt_mask_1d = (positions < prompt_len).astype(jnp.int32)
        cache_padding_mask = jnp.broadcast_to(
            prompt_mask_1d, (batch_size, total_len)
        )

        # Prefill: seed the KV cache with the whole prompt.
        _, prefill_hidden, cache = gemma_lm.call_with_cache(
            token_ids=token_ids,
            cache=cache,
            cache_update_index=jnp.int32(0),
            padding_mask=cache_padding_mask,
        )
        # First token: unembed only the last real prompt position.
        last_hidden = jax.lax.dynamic_slice_in_dim(
            prefill_hidden, prompt_len - 1, 1, axis=1
        )
        last_hidden = jnp.squeeze(last_hidden, axis=1)
        last_prompt_logits = backbone.token_embedding(last_hidden, reverse=True)
        first_tokens = jnp.argmax(last_prompt_logits, axis=-1).astype(jnp.int32)

        # Output buffer; positions past a row's EOS stay zero.
        generated_init = jnp.zeros(
            (batch_size, max_new_tokens), dtype=jnp.int32
        )
        generated_init = generated_init.at[:, 0].set(first_tokens)
        done_init = first_tokens == eos_token_id

        # Decode: greedy argmax until max_new_tokens or every row hits EOS.
        def cond_fn(state):
          step, _cache, _generated, _mask, _current, done_mask = state
          return jnp.logical_and(
              step < max_new_tokens,
              jnp.logical_not(jnp.all(done_mask)),
          )

        def body_fn(state):
          step, cache, generated, mask, current_token, done_mask = state
          cache_idx = prompt_len + step - 1
          new_mask = mask.at[:, cache_idx].set(1)

          logits, _, new_cache = gemma_lm.call_with_cache(
              token_ids=current_token[:, None],
              cache=cache,
              cache_update_index=cache_idx,
              padding_mask=new_mask,
          )
          next_tokens = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)
          # Rows already past EOS emit zeros.
          next_tokens = jnp.where(done_mask, jnp.int32(0), next_tokens)
          new_generated = generated.at[:, step].set(next_tokens)
          new_done = jnp.logical_or(done_mask, next_tokens == eos_token_id)
          return (
              step + 1, new_cache, new_generated, new_mask, next_tokens,
              new_done,
          )

        init_state = (
            jnp.int32(1),         # step (position 0 already holds first_tokens)
            cache,
            generated_init,
            cache_padding_mask,
            first_tokens,
            done_init,
        )
        _, _, final_generated, _, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, init_state
        )

      return {'generated_ids': final_generated}

    input_signature = {
        'token_ids': jax.ShapeDtypeStruct(
            shape=('b', max_seq_len), dtype=jnp.int32
        ),
        'prompt_len': jax.ShapeDtypeStruct(shape=('b',), dtype=jnp.int32),
        'eos_token_id': jax.ShapeDtypeStruct(shape=('b',), dtype=jnp.int32),
    }

    # Replayed with --warmup_server to pay JIT compile at start-up. The
    # out-of-vocab EOS keeps the loop from exiting early.
    warmup_examples = {
        'serving_default': [
            {
                'token_ids': np.zeros((1, max_seq_len), dtype=np.int32),
                'prompt_len': np.full((1,), 1, dtype=np.int32),
                'eos_token_id': np.full((1,), 2_000_000_000, dtype=np.int32),
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = os.path.join(tmpdir, 'ckpt')
      checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
      checkpointer.save(ckpt_path, params)
      ckpt_params = checkpointer.restore(ckpt_path)

    main_lib.save(
        export_path=export_path,
        models=jax_model.JaxModel(
            apply_fn_bundle=jax_model.ApplyFnBundle(
                apply_fn=apply_fn,
                input_polymorphic_shapes={
                    'token_ids': 'b, _',
                    'prompt_len': 'b',
                    'eos_token_id': 'b',
                },
                jax_export_kwargs={'platforms': ['cpu']},
            ),
            params=ckpt_params,
        ),
        serving_configs=[
            serving_config.ServingConfig(
                serving_signature='serving_default',
                input_signature=input_signature,
            ),
        ],
        warmup_examples=warmup_examples,
    )
  finally:
    keras.mixed_precision.set_global_policy(original_policy)
