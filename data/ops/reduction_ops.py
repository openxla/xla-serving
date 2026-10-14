# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright 2026 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reduction operations for jax_data.

Migrated from /third_party/tensorflow/python/ops/math_ops.py.
"""

from typing import Any

import numpy as np

from data.kernels.numerical_transforms_module import complexabs
from data.kernels.numerical_transforms_module import ReduceAllTransform
from data.kernels.numerical_transforms_module import ReduceAnyTransform
from data.kernels.numerical_transforms_module import ReduceEuclideanNormTransform
from data.kernels.numerical_transforms_module import ReduceMaxTransform
from data.kernels.numerical_transforms_module import ReduceMeanTransform
from data.kernels.numerical_transforms_module import ReduceMinTransform
from data.kernels.numerical_transforms_module import ReduceProdTransform
from data.kernels.numerical_transforms_module import ReduceSumTransform
from data.kernels.numerical_transforms_module import sqrt
from data.kernels.numerical_transforms_module import square
from data.kernels.numerical_transforms_module import subtract


def _get_input_shape_and_dtype(
    input_tensor: Any,
) -> tuple[tuple[int, ...], Any]:
  """Returns the shape and NumPy dtype of `input_tensor`."""
  if hasattr(input_tensor, "shape") and hasattr(input_tensor, "dtype"):
    return tuple(input_tensor.shape), np.dtype(input_tensor.dtype)
  arr = np.asarray(input_tensor)
  return arr.shape, arr.dtype


# Reduction operations
def _ReductionDims(x, axis):  # pylint: disable=invalid-name
  """Returns range(0, rank(x)) if axis is None."""
  if axis is not None:
    if isinstance(axis, (int, np.integer)):
      return [int(axis)]
    if isinstance(axis, np.ndarray) and axis.ndim > 1:
      raise ValueError(f"axis must be at most rank 1, got shape {axis.shape}")
    return [int(a) for a in axis]
  else:
    in_shape, _ = _get_input_shape_and_dtype(x)
    return list(range(0, len(in_shape)))


def _apply_reduction_op(
    transform_cls: Any,
    op_name: str,
    input_tensor: Any,
    axis: Any,
    keepdims: bool,
) -> Any:
  """Helper to apply a reduction operation."""
  del op_name
  t = transform_cls(
      _ReductionDims(input_tensor, axis),
      keepdims,
  )
  return t(input_tensor)


def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the sum of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `jd.math.add` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> # x has a shape of (2, 3) (two rows and three columns):
    >>> x = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32)
    >>> x
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
    >>> # sum all the elements
    >>> # 1 + 1 + 1 + 1 + 1+ 1 = 6
    >>> jd.function(jd.reduce_sum)(x).item()
    6
    >>> # reduce along the first dimension
    >>> # the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> jd.function(lambda a: jd.reduce_sum(a, 0))(x)
    array([2, 2, 2], dtype=int32)
    >>> # reduce along the second dimension
    >>> # the result is [1, 1] + [1, 1] + [1, 1] = [3, 3]
    >>> jd.function(lambda a: jd.reduce_sum(a, 1))(x)
    array([3, 3], dtype=int32)
    >>> # keep the original dimensions
    >>> jd.function(lambda a: jd.reduce_sum(a, 1, keepdims=True))(x)
    array([[3],
           [3]], dtype=int32)
    >>> # reduce along both dimensions
    >>> # the result is 1 + 1 + 1 + 1 + 1 + 1 = 6
    >>> # or, equivalently, reduce along rows, then reduce the resultant array
    >>> # [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> # 2 + 2 + 2 = 6
    >>> jd.function(lambda a: jd.reduce_sum(a, [0, 1]))(x).item()
    6

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor)]`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while jax_data returns the same dtype as the input.
  @end_compatibility
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceSumTransform, "reduce_sum", input_tensor, axis, keepdims
  )


def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the Euclidean norm of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.int32)
  jd.math.reduce_euclidean_norm(x)  # returns 4 as dtype is int32
  y = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32)
  jd.math.reduce_euclidean_norm(y)  # returns 4.1231055 which is sqrt(17)
  jd.math.reduce_euclidean_norm(y, 0)  # [sqrt(2), sqrt(5), sqrt(10)]
  jd.math.reduce_euclidean_norm(y, 1)  # [sqrt(14), sqrt(3)]
  jd.math.reduce_euclidean_norm(y, 1, keepdims=True)  # [[sqrt(14)], [sqrt(3)]]
  jd.math.reduce_euclidean_norm(y, [0, 1])  # sqrt(17)
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceEuclideanNormTransform,
      "reduce_euclidean_norm",
      input_tensor,
      axis,
      keepdims,
  )


def reduce_mean(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis` by computing the
  mean of elements across the dimensions in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a tensor with a single
  element is returned.

  For example:

  >>> x = np.array([[1., 1.], [2., 2.]], dtype=np.float32)
  >>> jd.function(jd.reduce_mean)(x)
  array(1.5, dtype=float32)
  >>> jd.function(lambda a: jd.reduce_mean(a, 0))(x)
  array([1.5, 1.5], dtype=float32)
  >>> jd.function(lambda a: jd.reduce_mean(a, 1))(x)
  array([1., 2.], dtype=float32)

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.mean

  Please note that `np.mean` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `jd.reduce_mean` has an aggressive type inference from `input_tensor`,
  for example:

  >>> x = np.array([1, 0, 1, 0], dtype=np.int32)
  >>> jd.function(jd.reduce_mean)(x)
  array(0, dtype=int32)
  >>> y = np.array([1., 0., 1., 0.], dtype=np.float32)
  >>> jd.function(jd.reduce_mean)(y)
  array(0.5, dtype=float32)

  @end_compatibility
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceMeanTransform, "reduce_mean", input_tensor, axis, keepdims
  )


def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the variance of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  >>> x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
  >>> jd.function(jd.math.reduce_variance)(x)
  array(1.25, dtype=float32)
  >>> jd.function(lambda a: jd.math.reduce_variance(a, 0))(x)
  array([1., 1.], dtype=float32)
  >>> jd.function(lambda a: jd.math.reduce_variance(a, 1))(x)
  array([0.25, 0.25], dtype=float32)

  Args:
    input_tensor: The tensor to reduce. Should have real or complex type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name scope for the associated operations (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor. Note,  for
    `complex64` or `complex128` input, the returned `Tensor` will be of type
    `float32` or `float64`, respectively.

  @compatibility(numpy)
  Equivalent to np.var

  Please note `np.var` has a `dtype` parameter that could be used to specify the
  output type. By default this is `dtype=float64`. On the other hand,
  `jd.math.reduce_variance` has aggressive type inference from `input_tensor`.
  @end_compatibility
  """
  del name  # Unused.
  _, in_dtype = _get_input_shape_and_dtype(input_tensor)
  if np.issubdtype(in_dtype, np.integer) or in_dtype == np.bool_:
    raise TypeError(
        "Input must be either real or complex. "
        f"Received integer type {in_dtype}."
    )
  means = reduce_mean(input_tensor, axis=axis, keepdims=True)
  diff = subtract(input_tensor, means)
  if np.issubdtype(in_dtype, np.complexfloating):
    # For complex values we need to take the absolute value before squaring.
    squared_deviations = square(complexabs(diff))
  else:
    squared_deviations = square(diff)
  return reduce_mean(squared_deviations, axis=axis, keepdims=keepdims)


def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the standard deviation of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  >>> x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
  >>> jd.function(jd.math.reduce_std)(x)
  array(1.118034, dtype=float32)
  >>> jd.function(lambda a: jd.math.reduce_std(a, 0))(x)
  array([1., 1.], dtype=float32)
  >>> jd.function(lambda a: jd.math.reduce_std(a, 1))(x)
  array([0.5, 0.5], dtype=float32)

  Args:
    input_tensor: The tensor to reduce. Should have real or complex type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name scope for the associated operations (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor. Note,  for
    `complex64` or `complex128` input, the returned `Tensor` will be of type
    `float32` or `float64`, respectively.

  @compatibility(numpy)
  Equivalent to np.std

  Please note `np.std` has a `dtype` parameter that could be used to specify the
  output type. By default this is `dtype=float64`. On the other hand,
  `jd.math.reduce_std` has aggressive type inference from `input_tensor`.
  @end_compatibility
  """
  del name  # Unused.
  variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
  return sqrt(variance)


def reduce_prod(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `jd.math.multiply` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `jd.math.multiply` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    >>> jd.function(jd.math.reduce_prod)(x)
    array(24., dtype=float32)
    >>> jd.function(lambda a: jd.math.reduce_prod(a, 0))(x)
    array([3., 8.], dtype=float32)
    >>> jd.function(lambda a: jd.math.reduce_prod(a, 1))(x)
    array([2., 12.],
    dtype=float32)

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.prod
  @end_compatibility
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceProdTransform, "reduce_prod", input_tensor, axis, keepdims
  )


def reduce_min(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the `jd.math.minimum` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `jd.math.minimum` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  >>> a = np.array([
  ...   [[1, 2], [3, 4]],
  ...   [[1, 2], [3, 4]]
  ... ], dtype=np.int32)
  >>> jd.function(jd.reduce_min)(a)
  array(1, dtype=int32)

  Choosing a specific axis returns minimum element in the given axis:

  >>> b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
  >>> jd.function(lambda x: jd.reduce_min(x, axis=0))(b)
  array([1, 2, 3], dtype=int32)
  >>> jd.function(lambda x: jd.reduce_min(x, axis=1))(b)
  array([1, 4], dtype=int32)

  Setting `keepdims` to `True` retains the dimension of `input_tensor`:

  >>> jd.function(lambda x: jd.reduce_min(x, keepdims=True))(a)
  array([[[1]]], dtype=int32)
  >>> jd.function(lambda x: jd.math.reduce_min(x, axis=0, keepdims=True))(a)
  array([[[1, 2],
          [3, 4]]], dtype=int32)

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.min
  @end_compatibility
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceMinTransform, "reduce_min", input_tensor, axis, keepdims
  )


def reduce_max(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `jd.math.maximum` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `jd.math.maximum` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Usage example:

    >>> x = np.array([5, 1, 2, 4], dtype=np.int32)
    >>> jd.function(jd.reduce_max)(x)
    array(5, dtype=int32)
    >>> x = np.array([-5, -1, -2, -4], dtype=np.int32)
    >>> jd.function(jd.reduce_max)(x)
    array(-1, dtype=int32)
    >>> x = np.array([4, float('nan')], dtype=np.float32)
    >>> jd.function(jd.reduce_max)(x)
    array(nan, dtype=float32)
    >>> x = np.array([float('nan'), float('nan')], dtype=np.float32)
    >>> jd.function(jd.reduce_max)(x)
    array(nan, dtype=float32)
    >>> x = np.array([float('-inf'), float('inf')], dtype=np.float32)
    >>> jd.function(jd.reduce_max)(x)
    array(inf, dtype=float32)

  See the numpy docs for `np.amax` and `np.nanmax` behavior.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceMaxTransform, "reduce_max", input_tensor, axis, keepdims
  )


def reduce_all(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `jd.math.logical_and` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `jd.math.logical_and` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = np.array([[True,  True], [False, False]])
    >>> jd.function(jd.math.reduce_all)(x)
    array(False)
    >>> jd.function(lambda a: jd.math.reduce_all(a, 0))(x)
    array([False, False])
    >>> jd.function(lambda a: jd.math.reduce_all(a, 1))(x)
    array([ True, False])

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.all
  @end_compatibility
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceAllTransform, "reduce_all", input_tensor, axis, keepdims
  )


def reduce_any(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `jd.math.logical_or` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `jd.math.logical_or` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = np.array([[True,  True], [False, False]])
    >>> jd.function(jd.reduce_any)(x)
    array(True)
    >>> jd.function(lambda a: jd.reduce_any(a, 0))(x)
    array([ True,  True])
    >>> jd.function(lambda a: jd.reduce_any(a, 1))(x)
    array([ True, False])

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.any
  @end_compatibility
  """
  del name  # Unused.
  return _apply_reduction_op(
      ReduceAnyTransform, "reduce_any", input_tensor, axis, keepdims
  )
