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

"""Values (i.e. non-function objects)."""

import abc
import dataclasses

import jaxtyping

from neptune_model._src.core import shlo_type
from neptune_model.protos import manifest_pb2


class Value(abc.ABC):
  pass


@dataclasses.dataclass
class ExternalValue(Value):
  data: manifest_pb2.UnstructuredData
  type: jaxtyping.PyTree[shlo_type.ShloTensorSpec] | None = None
  loader_type: int | None = None
