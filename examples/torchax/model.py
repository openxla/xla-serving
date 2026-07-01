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
"""PyTorch model for Neptune export example."""

import torch
from torch import nn


def get_model() -> nn.Module:
  """Returns a simple PyTorch model with a few operations."""
  return nn.Sequential(
      nn.Linear(10, 20),
      nn.ReLU(),
      nn.Linear(20, 16),
      nn.Unflatten(1, (1, 16)),
      nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
      nn.Sigmoid(),
  )
