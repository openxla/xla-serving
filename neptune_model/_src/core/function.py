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

"""The `Function` base class."""

from collections.abc import Sequence
import dataclasses

import jaxtyping

from neptune_model._src.core import shlo_type


@dataclasses.dataclass(kw_only=True)
class Function:
  """An abstract base class for functions whose signatures are StableHLO types.

  Attributes:
    input_signature: the input signature of the function.
    output_signature: the output signature of the function.
    data_names: checkpoint data names used by the function.
    signature: the pair `(input_signature, output_signature)`.
  """

  input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec]
  output_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec]
  data_names: Sequence[str] | None = None

  @property
  def signature(
      self,
  ) -> tuple[
      jaxtyping.PyTree[shlo_type.ShloTensorSpec],
      jaxtyping.PyTree[shlo_type.ShloTensorSpec],
  ]:
    return self.input_signature, self.output_signature
