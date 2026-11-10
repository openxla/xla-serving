# Copyright 2026 Arm Ltd. and affiliates
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

"""Export an E5 encoder with masked mean pooling and L2 normalization."""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from transformers import AutoModel

from xla_saved_model._src.export import jax_model, serving_config
from xla_saved_model._src.export import main_lib as xsm_export
from xla_saved_model._src.torch2xsm import main_lib as torch2xsm

DEFAULT_MODEL = "intfloat/e5-base-v2"
DEFAULT_REVISION = "1c644c92ad3ba1efdad3f1451a637716616a20e8"
_L2_NORMALIZATION_EPSILON = 1e-12


class E5EncoderWrapper(torch.nn.Module):
  """Expose positional tensors and only last_hidden_state to TorchAX."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self.model = model

  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      token_type_ids: torch.Tensor,
  ) -> torch.Tensor:
    return self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    ).last_hidden_state


def export_model(export_path):
  """Save FP32 embeddings for batch 1 and sequence lengths 128, 256 and 512.

  Args:
    export_path: Directory to write the XLA SavedModel bundle to.
  """
  model = (
      AutoModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
      .float()
      .eval()
  )
  weights, jfunc = torch2xsm.convert_to_jax(E5EncoderWrapper(model))

  def apply_fn(
      params: Any, inputs: dict[str, jax.Array]
  ) -> dict[str, jax.Array]:
    hidden_states = jfunc(
        params,
        (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        ),
    )
    mask = inputs["attention_mask"][..., None].astype(hidden_states.dtype)
    # Avoid division by zero when the binary mask is empty.
    pooled = jnp.sum(hidden_states * mask, axis=1) / jnp.maximum(
        jnp.sum(mask, axis=1), 1
    )
    norm = jnp.sqrt(jnp.sum(jnp.square(pooled), axis=1, keepdims=True))
    embeddings = pooled / jnp.maximum(norm, _L2_NORMALIZATION_EPSILON)
    return {"embeddings": embeddings}

  serving_configs = [
      serving_config.ServingConfig(
          serving_signature=f"embed_b1_s{sequence_length}",
          input_signature={
              name: jax.ShapeDtypeStruct((1, sequence_length), jnp.int32)
              for name in ("input_ids", "attention_mask", "token_type_ids")
          },
      )
      for sequence_length in (128, 256, 512)
  ]
  xsm_export.save(
      models=jax_model.JaxModel(
          apply_fn_bundle=jax_model.ApplyFnBundle(apply_fn=apply_fn),
          params=weights,
      ),
      export_path=export_path,
      serving_configs=serving_configs,
  )
