"""Functions backed by StableHLO."""

import abc
from collections.abc import Mapping
from dataclasses import dataclass  # pylint: disable=g-importing-member

from google3.third_party.neptune.model._src.core import function
from google3.third_party.neptune.model._src.core import shlo_type
from google3.third_party.neptune.model._src.core import unstructured_data


class ShloFunctionSupplementalInfo(abc.ABC):
  """Interface for ShloFunction's supplemental_info."""

  @abc.abstractmethod
  def serializable_to_proto(
      self,
  ) -> unstructured_data.UnstructuredDataWithExtName:
    """Serializes it to an `UnstructuredDataWithExtName`."""
    pass


@dataclass
class ShloFunction(function.Function):
  """A StableHLO function.

  Attributes:
    mlir_module_serialized: the serialized StableHLO.
    calling_convention_version: the calling convention version of the serialized
      StableHLO. See more versioning details at
      [this](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions)
      and
      [this](https://jax.readthedocs.io/en/latest/export/export.html#calling-convention-versions)
      doc.
    module_kept_var_idx: the indices of the arguments that will actually be
      passed to the StableHLO. See this
      [doc](https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention)
      for its use in the calling convention.
    lowering_platforms: a tuple containing at least one of 'tpu', 'cpu', 'cuda',
      'rocm'. See this
      [doc](https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention)
      for the calling convention for when there are multiple lowering platforms.
    supplemental_info: an opaque supplemental to the function. It can be used to
      carry e.g. JAX-specific information.
    physical_in_dtypes: physical input dtypes, which (if given) will override
      `input_signature`.
    physical_out_dtypes: physical output dtypes, which (if given) will override
      `output_signature`.
  """

  mlir_module_serialized: bytes
  calling_convention_version: int
  module_kept_var_idx: tuple[int, ...]
  lowering_platforms: tuple[str, ...]

  supplemental_info: Mapping[str, ShloFunctionSupplementalInfo]

  physical_in_dtypes: tuple[shlo_type.ShloDType | None, ...]
  physical_out_dtypes: tuple[shlo_type.ShloDType | None, ...]
