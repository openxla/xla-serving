"""Base class for all saveable types."""

from collections.abc import Sequence
from neptune_model._src.core import function
from neptune_model._src.core import value

Saveable = function.Function | Sequence[function.Function] | value.Value
