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
"""Tests for torch2npt main_lib."""

import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from neptune_model._src.core import test_utils
from neptune_model._src.jax2npt import main_lib as jax2npt_lib
from neptune_model._src.torch2npt import main_lib

# pylint: disable=g-import-not-at-top
try:
  # pytype: disable=import-error
  import torch
  from torch import nn
  # pytype: enable=import-error
except ImportError:
  torch = None
  nn = None
# pylint: enable=g-import-not-at-top

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


# pytype: disable=attribute-error
class MainLibTest(test_utils.NptTestCase):

  def test_pt_model_against_jax(self):
    pt_model = nn.Linear(10, 10)
    input_tensor = torch.randn(1, 10)
    shlo_fn_pt = main_lib.convert_to_shlo_fn(pt_model, input_tensor)

    weights, jfunc = main_lib.convert_to_jax(pt_model)

    input_np = input_tensor.detach().cpu().numpy()
    input_jax = jax.device_put(input_np)
    jax_output = jfunc(weights, (input_jax,))
    pt_output = pt_model(input_tensor).detach().cpu().numpy()
    np.testing.assert_allclose(jax_output, pt_output, rtol=1e-5, atol=1e-5)

    def native_jax_linear(w, inputs):
      x = inputs[0]
      return jnp.matmul(x, jnp.transpose(w["weight"])) + w["bias"]

    shlo_fn_jax = jax2npt_lib.convert(
        native_jax_linear,
        args_spec=(weights, (jax.ShapeDtypeStruct((1, 10), jnp.float32),)),
        kwargs_spec={},
    )
    self.assertEqual(shlo_fn_pt.input_signature, shlo_fn_jax.input_signature)
    self.assertEqual(shlo_fn_pt.output_signature, shlo_fn_jax.output_signature)

  def test_conversion_to_shlo(self):
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 16),
        nn.Unflatten(1, (1, 16)),
        nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )
    input_tensor = torch.randn(1, 10)
    shlo_fn = main_lib.convert_to_shlo_fn(model, input_tensor)

    self.assertNotEmpty(shlo_fn.mlir_module_serialized)


if __name__ == "__main__":
  absltest.main()
