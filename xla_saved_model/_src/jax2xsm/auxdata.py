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

"""Serialization for the auxiliary data of a custom PyTree node.

The auxiliary data is the second element of the pair returned by the node's
flatten function.

Values are written as `[kind, payload]` pairs rather than bare JSON because
JSON cannot express a dtype at all, because spellings of a dtype that compare
equal are not interchangeable in a `PyTreeDef`, and because a tuple and a list
are likewise not interchangeable.

The kinds handled here cover the auxiliary data seen in practice, not every
hashable object JAX allows. Others make `serialize` raise, and support for them
will be added as real usages come up.
"""

import json
from typing import Any

import jax.numpy as jnp
import numpy as np


def serialize(aux_data: Any) -> bytes:
  """Serializes the auxiliary data of a PyTree node.

  Args:
    aux_data: The auxiliary data to serialize.

  Returns:
    The serialized auxiliary data.

  Raises:
    ValueError: If `aux_data` holds a value of a kind this function cannot
      write down.
  """
  return json.dumps(_encode_value(aux_data)).encode("utf-8")


def deserialize(serialized_aux_data: bytes) -> Any:
  """Inverse of `serialize`.

  Args:
    serialized_aux_data: The bytes produced by `serialize`.

  Returns:
    The auxiliary data.

  Raises:
    ValueError: If the bytes are not what `serialize` produces.
  """
  try:
    encoded = json.loads(serialized_aux_data.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as e:
    raise ValueError(f"Malformed serialized auxiliary data: {e}") from e
  return _decode_value(encoded)


# How a value is written down, as a `[kind, payload]` pair. The kind decides
# how the payload is read back, and so has to distinguish spellings of a dtype
# that are equal but not interchangeable.
_NONE = "none"
_BOOL = "bool"
_INT = "int"
_FLOAT = "float"
_STR = "str"
_TUPLE = "tuple"
_LIST = "list"
_NUMPY_DTYPE = "numpy.dtype"
_NUMPY_SCALAR_TYPE = "numpy.scalar_type"
_JAX_NUMPY_SCALAR_TYPE = "jax.numpy.scalar_type"

# The kinds whose payload is the value itself, and the type it must have.
_PRIMITIVE_TYPES = {_BOOL: bool, _INT: int, _FLOAT: float, _STR: str}


def _encode_value(value: Any) -> list[Any]:
  """Returns `value` as a `[kind, payload]` pair of JSON-representable data."""
  if value is None:
    return [_NONE, None]
  # A bool is an int, so it has to be recognized first.
  if isinstance(value, bool):
    return [_BOOL, value]
  if isinstance(value, int):
    return [_INT, value]
  if isinstance(value, float):
    return [_FLOAT, value]
  if isinstance(value, str):
    return [_STR, value]
  # The exact type rather than `isinstance`: a namedtuple is a tuple, and
  # rebuilding it as a plain tuple would change the structure it belongs to.
  if value.__class__ is tuple:
    return [_TUPLE, [_encode_value(element) for element in value]]
  if value.__class__ is list:
    return [_LIST, [_encode_value(element) for element in value]]
  if isinstance(value, np.dtype):
    return [_NUMPY_DTYPE, str(value)]
  if isinstance(value, type) and issubclass(value, np.generic):
    # An abstract class such as `numpy.floating` either has no dtype or maps to
    # a concrete one that would not read back as the same class.
    try:
      dtype = np.dtype(value)
    except (TypeError, DeprecationWarning):
      dtype = None
    if dtype is not None and dtype.type is value:
      return [_NUMPY_SCALAR_TYPE, dtype.name]

  # `jax.numpy.int8` and friends are neither dtypes nor types; they are
  # singletons that carry a `dtype`. They are recognized by finding the
  # singleton again under the name of that dtype.
  name = getattr(getattr(value, "dtype", None), "name", None)
  if name is not None and getattr(jnp, name, None) is value:
    return [_JAX_NUMPY_SCALAR_TYPE, name]

  raise ValueError(
      f"Cannot serialize the auxiliary data value {value!r} of type"
      f" {type(value)}. Only None, bools, ints, floats, strings, dtypes, and"
      " tuples and lists of those are supported."
  )


def _decode_value(encoded: Any) -> Any:
  """Inverse of `_encode_value`."""
  if (
      not isinstance(encoded, list)
      or len(encoded) != 2
      or not isinstance(encoded[0], str)
  ):
    raise ValueError(f"Expected a [kind, payload] pair, got {encoded!r}.")
  kind, payload = encoded

  if kind == _NONE:
    if payload is not None:
      raise ValueError(f"Expected no payload for `{kind}`, got {payload!r}.")
    return None
  if kind in _PRIMITIVE_TYPES:
    # `type(...) is` rather than `isinstance`: a bool is an int.
    if type(payload) is not _PRIMITIVE_TYPES[kind]:
      raise ValueError(f"Expected a `{kind}` payload, got {payload!r}.")
    return payload
  if kind in (_TUPLE, _LIST):
    if not isinstance(payload, list):
      raise ValueError(f"Expected a list of elements, got {payload!r}.")
    elements = (_decode_value(element) for element in payload)
    return tuple(elements) if kind == _TUPLE else list(elements)

  if kind not in (_NUMPY_DTYPE, _NUMPY_SCALAR_TYPE, _JAX_NUMPY_SCALAR_TYPE):
    raise ValueError(f"Unknown auxiliary data value kind `{kind}`.")
  if not isinstance(payload, str):
    raise ValueError(f"Expected a dtype name for `{kind}`, got {payload!r}.")

  if kind == _JAX_NUMPY_SCALAR_TYPE:
    # Only accept what `_encode_value` recognizes, so that no other attribute
    # of `jax.numpy` can be named.
    scalar_type = getattr(jnp, payload, None)
    name = getattr(getattr(scalar_type, "dtype", None), "name", None)
    if name != payload:
      raise ValueError(f"`jax.numpy` has no scalar type named `{payload}`.")
    return scalar_type

  try:
    dtype = np.dtype(payload)
  except TypeError as e:
    raise ValueError(f"`{payload}` is not a NumPy dtype: {e}") from e
  return dtype if kind == _NUMPY_DTYPE else dtype.type
