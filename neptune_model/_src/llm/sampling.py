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

"""Sampling utilities for LLM generation in Neptune."""

import jax
import jax.numpy as jnp


def greedy_sample(logits: jax.Array) -> jax.Array:
  """Samples the token with the highest logit value (greedy search)."""
  return jnp.argmax(logits, axis=-1)


def top_k_sample(
    key: jax.Array, logits: jax.Array, k: int, temperature: float
) -> jax.Array:
  """Samples a token from the top-k logits."""
  top_values, top_indices = jax.lax.top_k(logits, k)
  probs = jax.nn.softmax(top_values / temperature, axis=-1)
  sample_idx = jax.random.categorical(key, jnp.log(probs + 1e-9))
  return top_indices[sample_idx]


def compute_scores(
    logits: jax.Array, target_ids: jax.Array | None = None
) -> dict[str, jax.Array]:
  """Computes sequence scores including logits and log probabilities."""
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  outputs = {
      "logits": logits,
      "log_probs": log_probs,
  }
  if target_ids is not None:
    target_log_probs = jnp.take_along_axis(
        log_probs, target_ids[:, None], axis=-1
    ).squeeze(axis=-1)
    outputs["target_log_probs"] = target_log_probs
  return outputs
