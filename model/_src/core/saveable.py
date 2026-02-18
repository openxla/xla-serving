"""Base class for all saveable types."""

from collections.abc import Sequence
from google3.third_party.neptune.model._src.core import function
from google3.third_party.neptune.model._src.core import value

Saveable = function.Function | Sequence[function.Function] | value.Value
