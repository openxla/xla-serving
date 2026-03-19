"""Functions that can be serialized to `UnstructuredData`."""

from dataclasses import dataclass  # pylint: disable=g-importing-member

from .third_party.neptune.model._src.core import function
from .third_party.neptune.model._src.core import unstructured_data


@dataclass
class SerializableFunction(function.Function):

  body: unstructured_data.UnstructuredDataWithExtName
