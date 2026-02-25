"""The `Function` base class."""

from collections.abc import Sequence
import dataclasses

import jaxtyping

from google3.third_party.neptune.model._src.core.shlo_type import ShloTensorSpec


@dataclasses.dataclass(kw_only=True)
class Function:
  """An abstract base class for functions whose signatures are StableHLO types.

  Attributes:
    input_signature: the input signature of the function.
    output_signature: the output signature of the function.
    data_names: checkpoint data names used by the function.
    signature: the pair `(input_signature, output_signature)`.
  """

  input_signature: jaxtyping.PyTree[ShloTensorSpec]
  output_signature: jaxtyping.PyTree[ShloTensorSpec]
  data_names: Sequence[str] | None = None

  @property
  def signature(
      self,
  ) -> tuple[
      jaxtyping.PyTree[ShloTensorSpec], jaxtyping.PyTree[ShloTensorSpec]
  ]:
    return self.input_signature, self.output_signature
