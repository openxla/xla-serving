"""Values (i.e. non-function objects)."""

import abc
import dataclasses

import jaxtyping

from neptune_model._src.core import shlo_type
from neptune_model.protos import manifest_pb2


class Value(abc.ABC):
  pass


@dataclasses.dataclass
class ExternalValue:
  data: manifest_pb2.UnstructuredData
  type: jaxtyping.PyTree[shlo_type.ShloTensorSpec] | None = None
