"""Cumulative scan operations for jax_data."""

from typing import Any

import numpy as np

from data.kernels.numerical_transforms_module import CumprodTransform
from data.kernels.numerical_transforms_module import CumsumTransform
from data.kernels.numerical_transforms_module import CumulativeLogsumexpTransform
from data.processor.python import composite_processor_pybind as cp


def _apply_scan_op(
    transform_cls: Any,
    op_name: str,
    x: Any,
    axis: int | Any,
    exclusive: bool,
    reverse: bool,
) -> Any:
  """Helper to apply a cumulative scan operation."""
  del op_name
  if isinstance(axis, cp.HostBuffer):
    t = transform_cls(exclusive, reverse, None)
    inputs = [x, axis]
  else:
    axis_arr = np.asarray(axis)
    if axis_arr.ndim != 0:
      raise ValueError(
          f"ScanOp: axis must be a scalar, not shape {axis_arr.shape}"
      )
    t = transform_cls(exclusive, reverse, int(axis_arr))
    inputs = [x]
  ctx = cp.ExecuteContext()
  ctx.inputs = [cp.Value(getattr(v, "_hb", v)) for v in inputs]
  t.transform.execute(ctx)
  out = ctx.outputs[0].as_encapsulated_type()
  return type(x)(out) if hasattr(x, "_hb") else out


def cumsum(
    x: Any,
    axis: int | Any = 0,
    exclusive: bool = False,
    reverse: bool = False,
    name: str | None = None,
) -> Any:
  """Computes the cumulative sum of the tensor `x` along `axis`.

  Args:
    x: A `Tracer` or array input.
    axis: A scalar `int` or 0-D tensor specifying the axis along which to scan.
      Defaults to 0.
    exclusive: If `True`, perform exclusive cumsum. Defaults to False.
    reverse: If `True`, perform cumsum in the opposite direction. Defaults to
      False.
    name: A name for the operation (optional).

  Returns:
    The cumulative sum of `x`.

  Raises:
    RuntimeError: If called outside of tracing (not inside a jd.function).
  """
  del name  # Unused.
  return _apply_scan_op(CumsumTransform, "cumsum", x, axis, exclusive, reverse)


def cumprod(
    x: Any,
    axis: int | Any = 0,
    exclusive: bool = False,
    reverse: bool = False,
    name: str | None = None,
) -> Any:
  """Computes the cumulative product of the tensor `x` along `axis`.

  Args:
    x: A `Tracer` or array input.
    axis: A scalar `int` or 0-D tensor specifying the axis along which to scan.
      Defaults to 0.
    exclusive: If `True`, perform exclusive cumprod. Defaults to False.
    reverse: If `True`, perform cumprod in the opposite direction. Defaults to
      False.
    name: A name for the operation (optional).

  Returns:
    The cumulative product of `x`.

  Raises:
    RuntimeError: If called outside of tracing (not inside a jd.function).
  """
  del name  # Unused.
  return _apply_scan_op(
      CumprodTransform, "cumprod", x, axis, exclusive, reverse
  )


def cumulative_logsumexp(
    x: Any,
    axis: int | Any = 0,
    exclusive: bool = False,
    reverse: bool = False,
    name: str | None = None,
) -> Any:
  """Computes the cumulative logsumexp of the tensor `x` along `axis`.

  Args:
    x: A `Tracer` or array input.
    axis: A scalar `int` or 0-D tensor specifying the axis along which to scan.
      Defaults to 0.
    exclusive: If `True`, perform exclusive cumulative logsumexp. Defaults to
      False.
    reverse: If `True`, perform cumulative logsumexp in the opposite direction.
      Defaults to False.
    name: A name for the operation (optional).

  Returns:
    The cumulative logsumexp of `x`.

  Raises:
    RuntimeError: If called outside of tracing (not inside a jd.function).
  """
  del name  # Unused.
  return _apply_scan_op(
      CumulativeLogsumexpTransform,
      "cumulative_logsumexp",
      x,
      axis,
      exclusive,
      reverse,
  )
