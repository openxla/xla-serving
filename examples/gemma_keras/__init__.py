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

# export.py traces Keras ops with JAX, so the JAX backend must be set before
# keras is imported. Fail early if another backend was forced.
_backend = os.environ.setdefault('KERAS_BACKEND', 'jax')
if _backend != 'jax':
  raise RuntimeError(
      "examples.gemma_keras requires KERAS_BACKEND=jax; got "
      f"KERAS_BACKEND={_backend!r}."
  )

# Here, not in export.py, so tests can read it without importing keras.
DEFAULT_MODEL_PRESET = 'gemma3_instruct_4b'
