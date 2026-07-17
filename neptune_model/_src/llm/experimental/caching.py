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

"""Cache processors for handling different KV cache strategies."""

from typing import Protocol
import jax
import jaxtyping


class CacheProcessor(Protocol):
  """Interface for handling different cache strategies."""

  def pack_inputs(
      self, tokens: jax.Array, pointers: jax.Array | None
  ) -> jaxtyping.PyTree:
    """Packs tokens and pointers into the format expected by the model."""
    ...

  def shuffle_cache(
      self, cache: jaxtyping.PyTree, flat_indices: jax.Array
  ) -> jaxtyping.PyTree:
    """Handles the caching shuffling/routing."""
    ...


class PagedCacheProcessor:
  """Zero-copy cache processor using logical pointers."""

  def pack_inputs(
      self, tokens: jax.Array, pointers: jax.Array | None
  ) -> jaxtyping.PyTree:
    # Pack both token IDs and logical routing pointers into a dictionary:
    # - "tokens": The input token IDs for the current step (used for
    #   embedding lookup and model execution).
    # - "parent_indices": The logical routing pointers
    #   (`virtual_to_physical_map`) tracking each candidate beam's parent KV
    #   cache slot. Because `PagedCacheProcessor` avoids physically copying or
    #   shuffling the KV cache in `shuffle_cache` (zero-copy), paged attention
    #   mechanisms use these indices to dynamically attend to the correct
    #   historical cache slots without physical copy overhead.
    return {"tokens": tokens, "parent_indices": pointers}

  def shuffle_cache(
      self, cache: jaxtyping.PyTree, flat_indices: jax.Array
  ) -> jaxtyping.PyTree:
    return cache  # Zero-copy passthrough


class ContiguousCacheProcessor:
  """Classic cache processor physically moving tensors."""

  def pack_inputs(
      self, tokens: jax.Array, pointers: jax.Array | None
  ) -> jaxtyping.PyTree:
    return tokens  # Disgard pointers; classic dense tensor inputs

  def shuffle_cache(
      self, cache: jaxtyping.PyTree, flat_indices: jax.Array
  ) -> jaxtyping.PyTree:
    return jax.tree_util.tree_map(lambda x: x[flat_indices, ...], cache)
