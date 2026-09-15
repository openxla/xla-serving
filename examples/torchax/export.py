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
"""Export utility for PyTorch model to XLA SavedModel."""

import jax
import jax.numpy as jnp
import torch
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib as xsm_export
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.torch2xsm import main_lib as torch2xsm


def export_model(pt_model: torch.nn.Module, export_path: str):
  """Exports the PyTorch model to XLA SavedModel format via JAX conversion."""
  weights, jfunc = torch2xsm.convert_to_jax(pt_model)

  def apply_fn(params, inputs):
    output = jfunc(params, (inputs['x'],))
    return {'y': output}

  input_signature = {
      'x': jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.float32),
  }

  xsm_export.save(
      jax_model.JaxModel(
          apply_fn_bundle=jax_model.ApplyFnBundle(
              apply_fn=apply_fn,
          ),
          params=weights,
      ),
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
          )
      ],
  )
