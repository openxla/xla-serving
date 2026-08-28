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
"""Hermetic unit tests for torch2npt main_lib."""

from absl.testing import absltest
import torch
from torch import nn

from neptune_model._src.core import test_utils
from neptune_model._src.torch2npt import main_lib


class MainLibTest(test_utils.NptTestCase):

  def test_convert_to_shlo_fn_linear(self):
    pt_model = nn.Linear(10, 10)
    input_tensor = torch.randn(1, 10)
    shlo_fn = main_lib.convert_to_shlo_fn(pt_model, input_tensor)
    self.assertNotEmpty(shlo_fn.mlir_module_serialized)
    self.assertGreater(shlo_fn.calling_convention_version, 0)

  def test_convert_to_shlo_fn_sequential(self):
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
