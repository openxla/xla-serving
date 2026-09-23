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

"""XLA SavedModel export for a ViT Base Patch16 224 image classifier.

Importing this module (via ``examples.vit``) sets ``KERAS_BACKEND=jax``
before keras is loaded; see ``examples/vit/__init__.py``.
"""

import jax
import jax.numpy as jnp
import keras
import keras_hub
import numpy as np

from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config


DEFAULT_MODEL_PRESET = 'vit_base_patch16_224_imagenet'


def export_model(
    export_path: str,
    dtype: str = 'fp32',
    preset: str = DEFAULT_MODEL_PRESET,
):
  """Exports the ViT Base image classifier using XLA SavedModel.

  Args:
    export_path: Directory to write the exported model to.
    dtype: Model compute dtype. One of 'fp32' or 'bf16'. 'bf16' uses the Keras
      mixed_bfloat16 policy; inputs remain fp32.
    preset: keras_hub preset name or local path to load into
      ViTImageClassifier.from_preset. Defaults to the public
      vit_base_patch16_224_imagenet preset.
  """
  if dtype not in ('fp32', 'bf16'):
    raise ValueError(f"Unsupported dtype: {dtype}. Expected 'fp32' or 'bf16'.")

  # Keras mixed_precision is a process-global policy. Snapshot the caller's
  # policy and restore it in `finally` so repeated calls with different dtypes
  # don't leak state across invocations.
  original_policy = keras.mixed_precision.global_policy()
  if dtype == 'bf16':
    keras.mixed_precision.set_global_policy('mixed_bfloat16')

  try:
    # Pass preprocessor=None to avoid loading KerasHub text/image preprocessing
    # pipelines that require tensorflow-text. The XLA SavedModel export expects
    # raw float32 image tensors directly.
    vit_model = keras_hub.models.ViTImageClassifier.from_preset(
        preset, preprocessor=None
    )

    # Use Keras 3's public `stateless_call` to run the model as a pure function
    # of its trainable / non-trainable variables — no private API required.
    params = {
        'trainable': [v.value for v in vit_model.trainable_variables],
        'non_trainable': [v.value for v in vit_model.non_trainable_variables],
    }

    def apply_fn(params, inputs):
      # `stateless_call` also returns updated non_trainable_variables (for e.g.
      # BN running stats); at inference with training=False they don't change,
      # so we discard them.
      logits, _ = vit_model.stateless_call(
          params['trainable'],
          params['non_trainable'],
          inputs['images'],
          training=False,
      )
      return {'logits': logits.astype(jnp.float32)}

    input_signature = {
        'images': jax.ShapeDtypeStruct(
            shape=('b', 224, 224, 3), dtype=jnp.float32
        ),
    }

    # Match the preprocessing range in preprocess_image.py ([-1, 1]) so warmup
    # traces the graph against in-distribution activations.
    warmup_image = np.random.uniform(
        -1.0, 1.0, size=(1, 224, 224, 3)
    ).astype(np.float32)

    main_lib.save(
        export_path=export_path,
        models=jax_model.JaxModel(
            apply_fn_bundle=jax_model.ApplyFnBundle(
                apply_fn=apply_fn,
                input_polymorphic_shapes={
                    'images': 'b, _, _, _',
                },
                jax_export_kwargs={'platforms': ['cpu', 'tpu']},
            ),
            params=params,
        ),
        serving_configs=[
            serving_config.ServingConfig(
                serving_signature='serving_default',
                input_signature=input_signature,
            )
        ],
        warmup_examples={
            'serving_default': [
                {'images': warmup_image},
            ]
        },
    )
  finally:
    keras.mixed_precision.set_global_policy(original_policy)
