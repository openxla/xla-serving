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

"""Abstract base class for Neptune ExportableTransform modules."""

import abc
from typing import Any


class ExportableTransform(abc.ABC):
  """Abstract base class for exportable transforms written in C++."""

  def __init__(self, *args: Any, **kwargs: Any):
    self._args = args
    self._kwargs = kwargs
    self._transform = None

  def initialize(self, input_signature: Any) -> None:
    pass

  def get_output_signature(self, input_signature: Any) -> Any:
    if self._transform is not None and hasattr(
        self._transform, "get_output_signature"
    ):
      return self._transform.get_output_signature(input_signature)
    return input_signature

  @property
  def transform(self) -> Any:
    return self._transform

  def export_assets(self) -> dict[str, str]:
    return {}

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if self._transform is not None:
      return self._transform(*args, **kwargs)
    raise RuntimeError("Transform is not initialized.")
