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
"""Functions to convert PyTorch models to StableHLO."""

from typing import Any

import torch

from neptune_model._src.core import shlo_function

try:
  from neptune_model._src.torch2npt import torchtpu_lib  # pylint: disable=g-import-not-at-top

  _TORCH_TPU_AVAILABLE = True
except ImportError:
  torchtpu_lib = None
  _TORCH_TPU_AVAILABLE = False


__all__ = [
    "convert_to_jax",
    "convert_to_shlo_fn",
]


def convert_to_jax(pytorch_model: torch.nn.Module) -> tuple[Any, Any]:
  """Converts a PyTorch model to a JAX model using TorchAX."""
  import torchax as tx  # pylint: disable=g-import-not-at-top

  env = tx.default_env()
  env.config.debug_print_each_op_operands = False
  weights, jfunc = tx.extract_jax(pytorch_model, env=env)
  return weights, jfunc


def _convert_via_torchax(
    pytorch_model: torch.nn.Module,
    sample_inputs: Any = None,
) -> shlo_function.ShloFunction:
  """Converts a PyTorch model to a Neptune ShloFunction via TorchAX."""
  import jax  # pylint: disable=g-import-not-at-top
  from neptune_model._src.jax2npt import main_lib as jax2npt_lib  # pylint: disable=g-import-not-at-top

  weights, jfunc = convert_to_jax(pytorch_model)
  if sample_inputs is None:
    raise ValueError("TorchAX lowering requires non-empty sample_inputs.")
  if isinstance(sample_inputs, (tuple, list)):
    input_tensor = sample_inputs[0]
  else:
    input_tensor = sample_inputs
  input_tensor_shape = input_tensor.shape
  input_tensor_dtype = input_tensor.detach().cpu().numpy().dtype
  input_spec = (jax.ShapeDtypeStruct(input_tensor_shape, input_tensor_dtype),)

  exported = jax.export.export(jax.jit(jfunc))(weights, input_spec)
  return jax2npt_lib.jax_exported_to_shlo_fn(exported)


def convert_to_shlo_fn(
    pytorch_model: torch.nn.Module,
    sample_inputs: Any = None,
    dynamic_shapes: Any | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    prefer_torchtpu: bool = True,
) -> shlo_function.ShloFunction:
  """Converts a PyTorch model to a Neptune ShloFunction.

  Prioritizes TorchTPU direct lowering when available (Google-internal).
  Falls back to TorchAX (PyTorch -> JAX -> StableHLO) in OSS or if requested.

  Args:
    pytorch_model: The PyTorch module to export and convert.
    sample_inputs: Sample input tensors or arguments.
    dynamic_shapes: Optional dynamic shape constraints.
    sample_kwargs: Optional keyword arguments for forward pass.
    prefer_torchtpu: If True, uses TorchTPU direct lowering; otherwise falls
      back to TorchAX.

  Returns:
    A Neptune ShloFunction representing the compiled StableHLO module.
  """
  if prefer_torchtpu and _TORCH_TPU_AVAILABLE and torchtpu_lib is not None:
    return torchtpu_lib.convert_to_shlo_fn(
        pytorch_model,
        sample_inputs=sample_inputs,
        dynamic_shapes=dynamic_shapes,
        sample_kwargs=sample_kwargs,
    )
  else:
    return _convert_via_torchax(pytorch_model, sample_inputs=sample_inputs)
