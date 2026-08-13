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
    pass

  @abc.abstractmethod
  def initialize(self, input_signature: Any) -> None:
    pass

  @abc.abstractmethod
  def get_output_signature(self, input_signature: Any) -> Any:
    pass

  @property
  def transform(self) -> Any:
    return getattr(self, "_transform", None)

  def export_assets(self) -> dict[str, str]:
    return {}
