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

"""Required __init__ file for package."""

import os

# This example is JAX-native: `export.py`'s `apply_fn` is traced by
# `jax_model.JaxModel` and calls `jnp` / `jax.ShapeDtypeStruct` directly, and
# `keras_hub.models.ViTImageClassifier.stateless_call` must dispatch through
# Keras' JAX backend to produce jax-traceable ops. Set the backend before any
# transitive keras/keras_hub import, and fail early if the caller has forced
# a different backend like tensorflow.
_backend = os.environ.setdefault('KERAS_BACKEND', 'jax')
if _backend != 'jax':
  raise RuntimeError(
      "examples.vit requires KERAS_BACKEND=jax; got "
      f"KERAS_BACKEND={_backend!r}."
  )
