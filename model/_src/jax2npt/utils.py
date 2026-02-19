"""Common utils shared between jax_to_polymorphic_function.py and main_lib.py."""

from collections.abc import Sequence
from typing import Any

import jax
from jax import export as jax_export
from jax.extend import sharding as jex_sharding
import numpy as np

from google3.third_party.neptune.model._src.core import shlo_type
from google3.third_party.neptune.model._src.jax2npt import typing
from google3.third_party.neptune.protos import manifest_pb2
from google3.third_party.tensorflow.compiler.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


def get_physical_dtype(dtype) -> shlo_type.ShloDType | None:
  try:
    if dtype == jax.numpy.int4:
      return shlo_type.ShloDType.i4
    if dtype == jax.numpy.uint4:
      return shlo_type.ShloDType.ui4
    if dtype == jax.numpy.bfloat16:
      return shlo_type.ShloDType.bf16
    if dtype == jax.float0:
      return shlo_type.ShloDType.bool
    return shlo_type.np_dtype_to_shlo_dtype(_jax_physical_dtype(dtype))
  except TypeError:
    return None


def assert_not_in_jax_transformation():
  """Raises an error if executed within a JAX transformation."""
  arr = jax.lax.iota("int32", 0)
  if isinstance(arr, jax.core.Tracer):
    raise ValueError("convert must be used outside all JAX transformations.")


def make_jax_exported_creator(
    fun_jax: typing.ApplyFn,
    platforms: Sequence[manifest_pb2.Platform] | None = None,
    native_serialization_disabled_checks: Sequence[
        jax_export.DisabledSafetyCheck
    ] = (),
):
  """Calls jax_export.export and returns a callable that returns Exported."""
  if platforms:
    native_serialization_platforms = tuple(
        manifest_pb2.Platform.Name(p).lower() for p in platforms
    )
  else:
    native_serialization_platforms = None

  return jax_export.export(
      fun_jax,
      platforms=native_serialization_platforms,
      disabled_checks=native_serialization_disabled_checks,
  )


JaxDType = np.dtype[Any]


def aval_dtype(a) -> JaxDType:
  assert isinstance(a, jax.core.ShapedArray)
  return a.dtype


def _jax_physical_aval(aval: jax.core.ShapedArray) -> jax.core.ShapedArray:
  """Converts JAX avals from logical to physical, if relevant.

  JAX might have avals whose logical vs physical shape/dtype may
  differ, and only the physical view is expected to possibly
  relate to EM. EM impl rules should operate on the physical form.

  A JAX logical aval might even correspond, in principle, to several
  physical avals, but we don't support those here. Instead we assert
  there is only one and return it.

  Args:
    aval: The JAX abstract value to be converted.

  Returns:
    The physical representation of the input aval. If the logical and physical
    representations are the same, the original `aval` is returned.
  """
  # TODO: jax.core doesn't publish physical_aval. Figure out how to
  # access it (or alike). physical_aval = jax.core.physical_aval(aval)
  physical_aval = aval
  assert (
      len(physical_aval.shape) >= len(aval.shape)
      and physical_aval.shape[: len(aval.shape)] == aval.shape
  ), (physical_aval, aval)
  return physical_aval


def _jax_physical_dtype(dtype) -> JaxDType:
  # assuming () is a fine stand-in shape
  return _jax_physical_aval(jax.core.ShapedArray((), dtype)).dtype


def hlo_sharding_to_op_sharding(
    hlo_sharding: Any | None,
) -> xla_data_pb2.OpSharding | None:
  """Converts `HloSharding` to proto `OpSharding`.

  Args:
    hlo_sharding: an `HloSharding`.

  Returns:
    An xla `OpSharding` proto.
  """
  if hlo_sharding is None:
    return None
  output = xla_data_pb2.OpSharding()
  # Note: `to_proto(self) -> xla_extension.OpSharding` so we need to serialize
  # and then parse.
  # TODO: See if go/proto-generated-python#sharing-messages can
  # help us share messages between Python and C++.
  output.ParseFromString(
      jex_sharding.get_serialized_proto_from_hlo_sharding(hlo_sharding)
  )
  return output
