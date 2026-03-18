"""Base class for all saveable types."""

from collections.abc import Sequence
from .third_party.neptune.model._src.core import function
from .third_party.neptune.model._src.core import value

Saveable = function.Function | Sequence[function.Function] | value.Value
