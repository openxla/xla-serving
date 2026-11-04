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
"""Functions to Convert Pytorch Models to Jax and SHLO."""

from typing import Any

import jax
import torch
import torchax as tx

from xla_saved_model._src.core import shlo_function
from xla_saved_model._src.jax2xsm import main_lib as jax2xsm_lib


def convert_to_jax(pytorch_model: torch.nn.Module) -> tuple[Any, Any]:
  """Converts a PyTorch model to a JAX model."""
  env = tx.default_env()
  env.config.debug_print_each_op_operands = False
  weights, jfunc = tx.extract_jax(pytorch_model, env=env)
  return weights, jfunc


def convert_to_shlo_fn(
    pytorch_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    weights_name: str | None = None,
) -> shlo_function.ShloFunction:
  """Converts a PyTorch model and sample input to an XSE ShloFunction."""
  weights, jfunc = convert_to_jax(pytorch_model)
  input_tensor_shape = input_tensor.shape
  input_tensor_dtype = input_tensor.detach().cpu().numpy().dtype
  input_spec = (jax.ShapeDtypeStruct(input_tensor_shape, input_tensor_dtype),)

  exported = jax.export.export(jax.jit(jfunc))(weights, input_spec)
  return jax2xsm_lib.jax_exported_to_shlo_fn(
      exported, weights_name=weights_name
  )
