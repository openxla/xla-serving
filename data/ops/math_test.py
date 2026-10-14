import types
from typing import Any
import unittest

import jax
import ml_dtypes
import numpy as np
from scipy import special

from data.kernels.math import reduction_transform_config_pb2
from data.ops import bitwise
from data.ops import math
from data.processor.python import composite_processor_pybind as cp
from data.processor.python import transform as _transform_mod

_active_transforms: list[Any] | None = None


class _TensorProxy:
  """Wraps a HostBuffer to support Python operators in @jd.function tests."""

  def __init__(self, hb: cp.HostBuffer):
    self._hb = hb

  @property
  def shape(self) -> tuple[int, ...]:
    return tuple(self._hb.shape)

  @property
  def dtype(self) -> Any:
    return np.dtype(self._hb.dtype)

  def __add__(self, other: Any) -> Any:
    return math.add(self, other)

  def __sub__(self, other: Any) -> Any:
    return math.subtract(self, other)

  def __mul__(self, other: Any) -> Any:
    return math.multiply(self, other)

  def __truediv__(self, other: Any) -> Any:
    return getattr(math, "truediv", math.divide)(self, other)

  def __floordiv__(self, other: Any) -> Any:
    return getattr(math, "floordiv")(self, other)

  def __mod__(self, other: Any) -> Any:
    return getattr(math, "floormod")(self, other)

  def __pow__(self, other: Any) -> Any:
    return getattr(math, "pow")(self, other)

  def __neg__(self) -> Any:
    return math.negative(self)

  def __abs__(self) -> Any:
    return math.abs(self)


def _to_hb(x: Any) -> Any:
  if isinstance(x, _TensorProxy):
    return x._hb
  if isinstance(x, np.ndarray):
    try:
      return cp.HostBuffer.from_dlpack(np.ascontiguousarray(x))
    except (BufferError, RuntimeError):
      return cp.HostBuffer.from_dlpack(jax.numpy.asarray(x))
  return x


def _from_hb(x: Any) -> Any:
  if isinstance(x, (tuple, list)):
    return type(x)(_from_hb(v) for v in x)
  hb = x._hb if isinstance(x, _TensorProxy) else x
  if not isinstance(hb, cp.HostBuffer):
    return x
  try:
    return np.from_dlpack(hb)
  except (BufferError, RuntimeError):
    return np.asarray(jax.dlpack.from_dlpack(hb))


def _call_transform(self: Any, *args: Any, **kwargs: Any) -> Any:
  if _active_transforms is not None:
    _active_transforms.append(self._transform)
  unwrapped = [_to_hb(a) for a in args]
  if all(isinstance(u, cp.HostBuffer) for u in unwrapped):
    shape_ctx = cp.ShapeInferenceContext()
    shape_ctx.inputs = [
        cp.ShapeInferenceValue(cp.HostBufferSpec(u.dtype, list(u.shape)))
        for u in unwrapped
    ]
    self._transform.propagate_signature(shape_ctx)
  out = self._transform(*unwrapped, **kwargs)
  return _TensorProxy(out) if _active_transforms is not None else out


_transform_mod.ExportableTransform.__call__ = _call_transform  # type: ignore[method-assign]


class _ProcessorShim:

  def __init__(self, transforms: list[Any]):
    self._transforms = transforms
    self._composite_processor = self

  def to_proto(self) -> Any:
    return types.SimpleNamespace(
        transforms=[t.to_proto() for t in self._transforms]
    )


def _jd_function(fn: Any) -> Any:
  recorded: list[Any] = []

  def wrapped(*args: Any, **kwargs: Any) -> Any:
    global _active_transforms
    prev = _active_transforms
    recorded.clear()
    _active_transforms = recorded
    try:
      proxied = [
          _TensorProxy(_to_hb(a)) if isinstance(a, np.ndarray) else a
          for a in args
      ]
      return _from_hb(fn(*proxied, **kwargs))
    finally:
      _active_transforms = prev

  wrapped.get_processor = lambda: _ProcessorShim(list(recorded))
  return wrapped


class _JdNamespace:
  math = math
  bitwise = bitwise
  function = staticmethod(_jd_function)

  def __getattr__(self, name: str) -> Any:
    return getattr(math, name)


jd = _JdNamespace()


class RoundTest(unittest.TestCase):

  def testRounding(self):
    x = np.arange(-5.0, 5.0, 0.25)
    for dtype in [np.float32, np.float64, np.int32]:
      x_np = np.array(x, dtype=dtype)

      @jd.function
      def round_fn(a):
        return jd.math.round(a)

      y_jd = round_fn(x_np)
      y_np = np.round(x_np)
      np.testing.assert_allclose(y_jd, y_np, atol=1e-2)


class BinaryOpsTest(unittest.TestCase):

  def testAdd(self):
    @jd.function
    def my_add(a, b):
      return a + b

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    z = my_add(x, y)
    np.testing.assert_array_equal(z, np.array([5, 7, 9], dtype=np.int32))

  def testSubtract(self):
    @jd.function
    def my_sub(a, b):
      return a - b

    x = np.array([5, 7, 9], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    z = my_sub(x, y)
    np.testing.assert_array_equal(z, np.array([1, 2, 3], dtype=np.int32))

  def testErrorReceivedIfDtypeMismatchFromOp(self):
    @jd.function
    def my_add(a, b):
      return a + b

    a = np.ones([1], dtype=np.int32)
    b = np.ones([1], dtype=np.float32)
    with self.assertRaises(Exception):
      my_add(a, b)


class SignTest(unittest.TestCase):

  def testComplexSignTinyNoUnderflow(self):
    tiny = 1e-20
    for dtype, expected_dtype in [
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
    ]:
      x = np.array([tiny + tiny * 1j], dtype=dtype)

      @jd.function
      def sign_fn(a):
        return jd.math.sign(a)

      y_jd = sign_fn(x)
      expected = np.array(
          [complex(tiny, tiny) / abs(complex(tiny, tiny))], dtype=expected_dtype
      )
      np.testing.assert_allclose(y_jd, expected, atol=1e-6)
      self.assertEqual(y_jd.dtype, expected_dtype)

    z = np.array([0 + 0j], dtype=np.complex64)

    @jd.function
    def sign_zero(a):
      return jd.math.sign(a)

    self.assertEqual(sign_zero(z)[0], np.complex64(0))


class OperatorTest(unittest.TestCase):

  def testNoOpTracing(self):
    self.skipTest("Currently not working.")

    @jd.function
    def no_op(a, b):
      return a, b

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    a, b = no_op(x, y)
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, y)

  def testEmptyGraphEdgeSelectionTracing(self):
    self.skipTest("Currently not working.")

    @jd.function
    def no_op(a, b):
      del a
      return b

    # Verify edge a is pruned in empty graph.

  def testNonEmptyGraphEdgeSelectionTracing(self):
    @jd.function
    def select_second(a, b):
      _ = a + a
      return b

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    z = select_second(x, y)
    np.testing.assert_array_equal(z, y)

  def testConstantOpTracing(self):
    self.skipTest("Currently not working.")

    @jd.function
    def constant():
      return 3, [1, 2, 3]

    x, y = constant()
    self.assertEqual(x, 3)
    np.testing.assert_array_equal(y, np.array([1, 2, 3], dtype=np.int32))

  def testSideEffectTracing(self):
    self.skipTest("Currently not working.")

    class CustomClass:

      def __init__(self, value):
        self.value = value

      def increment(self):
        self.value += 1

    @jd.function
    def side_effect(c: CustomClass):
      c.increment()
      return True

    # Verify c can be incremented if value was a Tracer?
    # if value is a regular value it is unsupported.

  def testAbsMagicOperatorTracing(self):
    @jd.function
    def my_abs(a):
      return abs(a)

    x = np.array([-5, 7, -9], dtype=np.float32)
    z = my_abs(x)
    np.testing.assert_array_equal(z, np.array([5, 7, 9], dtype=np.float32))

  def testNegMagicOperatorTracing(self):
    @jd.function
    def my_neg(a):
      return -a

    x = np.array([5, -7, 0], dtype=np.float32)
    z = my_neg(x)
    np.testing.assert_array_equal(z, np.array([-5, 7, 0], dtype=np.float32))


class MathOpsTest(unittest.TestCase):

  def testMathAddTracing(self):
    @jd.function
    def my_add(a, b):
      return jd.math.add(a, b)

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    z = my_add(x, y)
    np.testing.assert_array_equal(z, np.array([5, 7, 9], dtype=np.int32))

  def testMathSubtractTracing(self):
    @jd.function
    def my_sub(a, b):
      return jd.math.subtract(a, b)

    x = np.array([5, 7, 9], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    z = my_sub(x, y)
    np.testing.assert_array_equal(z, np.array([1, 2, 3], dtype=np.int32))

  def testSqrtFloat32Tracing(self):
    @jd.function
    def my_sqrt(a):
      return jd.math.sqrt(a)

    x = np.array([[4.0], [16.0]], dtype=np.float32)
    z = my_sqrt(x)
    np.testing.assert_allclose(z, np.array([[2.0], [4.0]], dtype=np.float32))

  def testSqrtFloat64Tracing(self):
    @jd.function
    def my_sqrt(a):
      return jd.math.sqrt(a)

    x = np.array([9.0, 25.0, 100.0], dtype=np.float64)
    z = my_sqrt(x)
    np.testing.assert_allclose(z, np.array([3.0, 5.0, 10.0], dtype=np.float64))

  def testSqrtNegativeFloatNan(self):
    @jd.function
    def my_sqrt(a):
      return jd.math.sqrt(a)

    x = np.array([-4.0, 16.0], dtype=np.float32)
    z = my_sqrt(x)
    np.testing.assert_array_equal(np.isnan(z), [True, False])
    self.assertEqual(z[1], 4.0)

  def testSqrtTopLevelExport(self):
    @jd.function
    def my_sqrt(a):
      return jd.sqrt(a)

    x = np.array([4.0, 9.0], dtype=np.float32)
    z = my_sqrt(x)
    np.testing.assert_allclose(z, np.array([2.0, 3.0], dtype=np.float32))

  def testNegativeTracing(self):
    @jd.function
    def my_neg(a):
      return jd.math.negative(a)

    x = np.array([5.0, -7.0, 0.0], dtype=np.float32)
    z = my_neg(x)
    np.testing.assert_allclose(z, np.array([-5.0, 7.0, 0.0], dtype=np.float32))

  def testNegativeTopLevelExport(self):
    @jd.function
    def my_neg(a):
      return jd.negative(a)

    x = np.array([5.0, -7.0, 0.0], dtype=np.float32)
    z = my_neg(x)
    np.testing.assert_allclose(z, np.array([-5.0, 7.0, 0.0], dtype=np.float32))

  def testNegOperatorTracing(self):
    @jd.function
    def neg_operator(a):
      return -a

    x = np.array([5.0, -7.0, 0.0], dtype=np.float32)
    z = neg_operator(x)
    np.testing.assert_allclose(z, np.array([-5.0, 7.0, 0.0], dtype=np.float32))

    processor = neg_operator.get_processor()
    config = processor._composite_processor._composite_processor.to_proto()
    self.assertEqual(len(config.transforms), 1)
    self.assertEqual(config.transforms[0].op_name, "NegativeTransform")

  def testSigmoidTracing(self):
    @jd.function
    def my_sigmoid(a):
      return jd.math.sigmoid(a)

    x = np.array([0.0, 2.0], dtype=np.float32)
    z = my_sigmoid(x)
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_allclose(z, expected, rtol=1e-4)

  def testTrigOpsTracing(self):
    @jd.function
    def trig_fn(a, b, c):
      return (
          jd.math.sin(a),
          jd.math.cos(a),
          jd.math.tan(a),
          jd.math.asin(b),
          jd.math.acos(b),
          jd.math.atan(a),
          jd.math.sinh(a),
          jd.math.cosh(a),
          jd.math.tanh(a),
          jd.math.asinh(a),
          jd.math.acosh(c),
          jd.math.atanh(b),
      )

    x = np.array([0.0, 0.5], dtype=np.float32)
    b = np.array([0.0, 0.5], dtype=np.float32)
    c = np.array([1.5, 2.0], dtype=np.float32)
    res = trig_fn(x, b, c)
    np.testing.assert_allclose(res[0], np.sin(x), rtol=1e-4)
    np.testing.assert_allclose(res[1], np.cos(x), rtol=1e-4)
    np.testing.assert_allclose(res[2], np.tan(x), rtol=1e-4)
    np.testing.assert_allclose(res[3], np.arcsin(b), rtol=1e-4)
    np.testing.assert_allclose(res[4], np.arccos(b), rtol=1e-4)
    np.testing.assert_allclose(res[5], np.arctan(x), rtol=1e-4)
    np.testing.assert_allclose(res[6], np.sinh(x), rtol=1e-4)
    np.testing.assert_allclose(res[7], np.cosh(x), rtol=1e-4)
    np.testing.assert_allclose(res[8], np.tanh(x), rtol=1e-4)
    np.testing.assert_allclose(res[9], np.arcsinh(x), rtol=1e-4)
    np.testing.assert_allclose(res[10], np.arccosh(c), rtol=1e-4)
    np.testing.assert_allclose(res[11], np.arctanh(b), rtol=1e-4)

  def testTrigTopLevelExport(self):
    @jd.function
    def trig_fn(a):
      return jd.sin(a), jd.cos(a), jd.tan(a)

    x = np.array([0.0, 0.5], dtype=np.float32)
    s, c, t = trig_fn(x)
    np.testing.assert_allclose(s, np.sin(x), rtol=1e-4)
    np.testing.assert_allclose(c, np.cos(x), rtol=1e-4)
    np.testing.assert_allclose(t, np.tan(x), rtol=1e-4)

  def testConjAndComplexAbsTracing(self):
    @jd.function
    def my_fn(a):
      return (
          jd.math.conj(a),
          jd.math.complex_abs(a),
          jd.conj(a),
          jd.complex_abs(a),
      )

    x = np.array([3 + 4j, 0 - 5j], dtype=np.complex64)
    cj1, abs1, cj2, abs2 = my_fn(x)
    np.testing.assert_array_equal(cj1, np.conj(x))
    np.testing.assert_allclose(abs1, np.abs(x), rtol=1e-4)
    np.testing.assert_array_equal(cj2, np.conj(x))
    np.testing.assert_allclose(abs2, np.abs(x), rtol=1e-4)

  def testCumsumTracing(self):
    @jd.function
    def my_cumsum(a):
      return (
          jd.math.cumsum(a, axis=0, name="cumsum_0"),
          jd.math.cumsum(a, axis=1, exclusive=True, name="cumsum_1_excl"),
          jd.math.cumsum(a, axis=-1, reverse=True, name="cumsum_rev"),
          jd.math.cumsum(a, axis=1, exclusive=True, reverse=True),
          jd.cumsum(a, axis=0),
          jd.cumsum(a),
      )

    y = np.array([[2, 4, 6, 8], [1, 3, 5, 7]], dtype=np.int32)
    c0, c1_excl, c_rev, c_excl_rev, top_level_0, top_level = my_cumsum(y)

    # axis=0: [[2, 4, 6, 8], [3, 7, 11, 15]]
    np.testing.assert_array_equal(
        c0, np.array([[2, 4, 6, 8], [3, 7, 11, 15]], dtype=np.int32)
    )
    # axis=1 exclusive: [[0, 2, 6, 12], [0, 1, 4, 9]]
    np.testing.assert_array_equal(
        c1_excl, np.array([[0, 2, 6, 12], [0, 1, 4, 9]], dtype=np.int32)
    )
    # axis=-1 reverse: [[20, 18, 14, 8], [16, 15, 12, 7]]
    np.testing.assert_array_equal(
        c_rev, np.array([[20, 18, 14, 8], [16, 15, 12, 7]], dtype=np.int32)
    )
    # axis=1 exclusive and reverse: [[18, 14, 8, 0], [15, 12, 7, 0]]
    np.testing.assert_array_equal(
        c_excl_rev,
        np.array([[18, 14, 8, 0], [15, 12, 7, 0]], dtype=np.int32),
    )
    # top-level export
    np.testing.assert_array_equal(top_level_0, c0)
    np.testing.assert_array_equal(top_level, c0)

    # Test with float64 and 1D
    @jd.function
    def cumsum_1d(a):
      return jd.math.cumsum(a)

    x_1d = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    np.testing.assert_allclose(
        cumsum_1d(x_1d),
        np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float64),
    )

  def testCumprodTracing(self):
    @jd.function
    def my_cumprod(a):
      return (
          jd.math.cumprod(a, axis=0, name="cumprod_0"),
          jd.math.cumprod(a, axis=1, exclusive=True, name="cumprod_1_excl"),
          jd.math.cumprod(a, axis=-1, reverse=True, name="cumprod_rev"),
          jd.math.cumprod(a, axis=1, exclusive=True, reverse=True),
          jd.cumprod(a, axis=0),
          jd.cumprod(a),
      )

    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    c0, c1_excl, c_rev, c_excl_rev, top_level_0, top_level = my_cumprod(y)

    # axis=0: [[1, 2, 3], [4, 10, 18]]
    np.testing.assert_array_equal(
        c0, np.array([[1, 2, 3], [4, 10, 18]], dtype=np.int32)
    )
    # axis=1 exclusive: [[1, 1, 2], [1, 4, 20]]
    np.testing.assert_array_equal(
        c1_excl, np.array([[1, 1, 2], [1, 4, 20]], dtype=np.int32)
    )
    # axis=-1 reverse: [[6, 6, 3], [120, 30, 6]]
    np.testing.assert_array_equal(
        c_rev, np.array([[6, 6, 3], [120, 30, 6]], dtype=np.int32)
    )
    # axis=1 exclusive and reverse: [[6, 3, 1], [30, 6, 1]]
    np.testing.assert_array_equal(
        c_excl_rev, np.array([[6, 3, 1], [30, 6, 1]], dtype=np.int32)
    )
    # top-level export
    np.testing.assert_array_equal(top_level_0, c0)
    np.testing.assert_array_equal(top_level, c0)

    # Test with float32 and 1D
    @jd.function
    def cumprod_1d(a):
      return jd.math.cumprod(a)

    x_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(
        cumprod_1d(x_1d),
        np.array([1.0, 2.0, 6.0, 24.0], dtype=np.float32),
    )

  def testCumulativeLogsumexpTracing(self):
    @jd.function
    def my_logsumexp(a):
      return (
          jd.math.cumulative_logsumexp(a, axis=0, name="logsumexp_0"),
          jd.math.cumulative_logsumexp(
              a, axis=1, exclusive=True, name="logsumexp_1_excl"
          ),
          jd.math.cumulative_logsumexp(
              a, axis=-1, reverse=True, name="logsumexp_rev"
          ),
          jd.cumulative_logsumexp(a, axis=0),
          jd.cumulative_logsumexp(a),
      )

    y = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    c0, c1_excl, c_rev, top_level_0, top_level = my_logsumexp(y)

    expected_c0 = np.logaddexp.accumulate(y, axis=0)
    np.testing.assert_allclose(c0, expected_c0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(top_level_0, expected_c0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(top_level, expected_c0, rtol=1e-5, atol=1e-5)

    expected_c1_excl = np.array(
        [
            [-np.inf, 0.0, np.logaddexp(0.0, 1.0)],
            [-np.inf, 3.0, np.logaddexp(3.0, 4.0)],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(c1_excl, expected_c1_excl, rtol=1e-5, atol=1e-5)

    expected_c_rev = np.logaddexp.accumulate(y[:, ::-1], axis=-1)[:, ::-1]
    np.testing.assert_allclose(c_rev, expected_c_rev, rtol=1e-5, atol=1e-5)

    # Numerical stability test: large numbers where naive exp would overflow
    @jd.function
    def stable_fn(a):
      return jd.math.cumulative_logsumexp(a)

    large_x = np.array([1000.0, 1000.0], dtype=np.float32)
    # log(exp(1000) + exp(1000)) = 1000 + log(2) ~ 1000.693147
    out_stable = stable_fn(large_x)
    np.testing.assert_allclose(
        out_stable,
        np.array([1000.0, 1000.0 + np.log(2.0)], dtype=np.float32),
        rtol=1e-5,
    )


class LogicalOpsTest(unittest.TestCase):

  def testTruthTable(self):
    truth_values = [True, False]
    for x_val in truth_values:
      x = np.array([x_val], dtype=np.bool_)

      @jd.function
      def not_fn(a):
        return jd.math.logical_not(a)

      np.testing.assert_array_equal(not_fn(x), np.logical_not(x))

      for y_val in truth_values:
        y = np.array([y_val], dtype=np.bool_)

        @jd.function
        def binary_fn(a, b):
          return (
              jd.math.logical_and(a, b),
              jd.math.logical_or(a, b),
              jd.math.logical_xor(a, b),
          )

        and_res, or_res, xor_res = binary_fn(x, y)
        np.testing.assert_array_equal(and_res, np.logical_and(x, y))
        np.testing.assert_array_equal(or_res, np.logical_or(x, y))
        np.testing.assert_array_equal(xor_res, np.logical_xor(x, y))

  def testTensors(self):
    np.random.seed(42)
    x = np.random.randint(0, 2, size=(2, 3, 4)).astype(np.bool_)
    y = np.random.randint(0, 2, size=(2, 3, 4)).astype(np.bool_)

    @jd.function
    def logical_ops(a, b):
      return (
          jd.math.logical_not(a),
          jd.math.logical_and(a, b),
          jd.math.logical_or(a, b),
          jd.math.logical_xor(a, b),
      )

    not_res, and_res, or_res, xor_res = logical_ops(x, y)

    self.assertEqual(not_res.dtype, np.bool_)
    self.assertEqual(and_res.dtype, np.bool_)
    self.assertEqual(or_res.dtype, np.bool_)
    self.assertEqual(xor_res.dtype, np.bool_)

    np.testing.assert_array_equal(not_res, np.logical_not(x))
    np.testing.assert_array_equal(and_res, np.logical_and(x, y))
    np.testing.assert_array_equal(or_res, np.logical_or(x, y))
    np.testing.assert_array_equal(xor_res, np.logical_xor(x, y))

  def testTopLevelExports(self):
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)

    @jd.function
    def top_level_fn(a, b):
      return (
          jd.logical_not(a),
          jd.logical_and(a, b),
          jd.logical_or(a, b),
          jd.logical_xor(a, b),
      )

    not_res, and_res, or_res, xor_res = top_level_fn(x, y)
    np.testing.assert_array_equal(not_res, np.logical_not(x))
    np.testing.assert_array_equal(and_res, np.logical_and(x, y))
    np.testing.assert_array_equal(or_res, np.logical_or(x, y))
    np.testing.assert_array_equal(xor_res, np.logical_xor(x, y))

  def testShapeMismatch(self):
    x = np.array([True, False], dtype=np.bool_)
    y = np.array([True, False, True], dtype=np.bool_)

    for op in [jd.math.logical_and, jd.math.logical_or, jd.math.logical_xor]:
      with self.subTest(op=op):

        @jd.function
        def fn(a, b, op=op):
          return op(a, b)

        with self.assertRaises(Exception):
          fn(x, y)

  def testDtypeMismatch(self):
    x = np.array([1, 0], dtype=np.int32)
    y = np.array([0, 1], dtype=np.int32)

    for op in [jd.math.logical_and, jd.math.logical_or, jd.math.logical_xor]:
      with self.subTest(op=op):

        @jd.function
        def binary_fn(a, b, op=op):
          return op(a, b)

        with self.assertRaises(Exception):
          binary_fn(x, y)

    @jd.function
    def not_fn(a):
      return jd.math.logical_not(a)

    with self.assertRaises(Exception):
      not_fn(x)


# Reduction operation tests below (ReduceTest, VarianceTest, StdTest,
# EuclideanNormTest) are migrated from
# /third_party/tensorflow/python/kernel_tests/math_ops/reduction_ops_test.py
# and
# /third_party/tensorflow/python/kernel_tests/math_ops/reduction_ops_test_big.py.
class ReduceTest(unittest.TestCase):

  def testReduceAllDims(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    @jd.function
    def fn(a):
      return jd.math.reduce_sum(a)

    y_jd = fn(x)
    self.assertEqual(y_jd, 21)

  def testReduceExtendType(self):
    np.random.seed(0)
    in_f32 = np.random.randn(100, 100).astype(np.float32)
    in_bf16 = in_f32.astype(ml_dtypes.bfloat16)

    @jd.function
    def sum_fn(a):
      return jd.math.reduce_sum(a)

    out_f32 = sum_fn(in_f32)
    out_bf16 = sum_fn(in_bf16)
    expected = out_f32.astype(ml_dtypes.bfloat16)

    np.testing.assert_allclose(
        out_bf16.astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

  def testReduceExplicitAxes(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    for axis in (0, -2):

      @jd.function
      def fn0(a, axis=axis):
        return jd.math.reduce_sum(a, axis=axis)

      np.testing.assert_array_equal(fn0(x), [5, 7, 9])
    for axis in (1, -1):

      @jd.function
      def fn1(a, axis=axis):
        return jd.math.reduce_sum(a, axis=axis)

      np.testing.assert_array_equal(fn1(x), [6, 15])
    for axis in (
        None,
        (0, 1),
        (1, 0),
        (-1, 0),
        (0, -1),
        (-2, 1),
        (1, -2),
        (-1, -2),
        (-2, -1),
    ):

      @jd.function
      def fn_all(a, axis=axis):
        return jd.math.reduce_sum(a, axis=axis)

      self.assertEqual(fn_all(x), 21)

  def testReduceInvalidAxis(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([[0], [1]])

    @jd.function
    def fn(a):
      return jd.math.reduce_sum(a, axis)

    with self.assertRaisesRegex(ValueError, "must be at most rank 1"):
      fn(x)

  def testInvalidRepeatedReductionIndices(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    @jd.function
    def fn(a):
      return jd.math.reduce_sum(a, axis=[0, 0])

    with self.assertRaisesRegex(Exception, "Axes contains duplicate dimension"):
      fn(x)

  def testReduceVar(self):
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")

    @jd.function
    def var_all(a):
      return jd.math.reduce_variance(a)

    @jd.function
    def var_axis0(a):
      return jd.math.reduce_variance(a, axis=0)

    np.testing.assert_allclose(var_all(x), 0)
    np.testing.assert_allclose(var_axis0(x), [0, 0, 0])

    x_int = np.array([[1, 2, 1, 1], [1, 1, 0, 1]], dtype=np.int32)
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      var_all(x_int)

    x_np = np.array([[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    self.assertEqual(np.var(x_np), 0.25)
    self.assertEqual(var_all(x_np), 0.25)

  def testReduceVarComplex(self):
    complex_ys = [
        (np.array([0 - 1j, 0 + 1j], "complex64"), np.float32),
        (np.array([0 - 1j, 0 + 1j], "complex128"), np.float64),
    ]
    for y, dtype in complex_ys:

      @jd.function
      def fn(a):
        return jd.math.reduce_variance(a)

      y_result = fn(y)
      self.assertEqual(np.var(y), 1.0)
      self.assertEqual(y_result, 1.0)
      self.assertEqual(y_result.dtype, dtype)

  def testReduceStd(self):
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")

    @jd.function
    def std_all(a):
      return jd.math.reduce_std(a)

    @jd.function
    def std_axis0(a):
      return jd.math.reduce_std(a, axis=0)

    np.testing.assert_allclose(std_all(x), 0)
    np.testing.assert_allclose(std_axis0(x), [0, 0, 0])

    x_int = np.array([[1, 2, 1, 1], [1, 1, 0, 1]], dtype=np.int32)
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      std_all(x_int)

    x_np = np.array([[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    self.assertEqual(np.std(x_np), 0.5)
    self.assertEqual(std_all(x_np), 0.5)

  def testReduceStdComplex(self):
    complex_ys = [
        (np.array([0 - 1j, 0 + 1j], "complex64"), np.float32),
        (np.array([0 - 1j, 0 + 1j], "complex128"), np.float64),
    ]
    for y, dtype in complex_ys:

      @jd.function
      def fn(a):
        return jd.math.reduce_std(a)

      y_result = fn(y)
      self.assertEqual(np.std(y), 1.0)
      self.assertEqual(y_result, 1.0)
      self.assertEqual(y_result.dtype, dtype)

  def testAllReductionOpsAndTopLevelExports(self):
    x_bool = np.array([[True, False], [True, True]], dtype=np.bool_)

    @jd.function
    def bool_reductions(a):
      return (
          jd.math.reduce_all(a),
          jd.math.reduce_all(a, axis=0),
          jd.math.reduce_all(a, axis=1, keepdims=True),
          jd.math.reduce_any(a),
          jd.math.reduce_any(a, axis=0),
          jd.math.reduce_any(a, axis=1, keepdims=True),
          jd.reduce_all(a),
          jd.reduce_any(a),
      )

    all_res = bool_reductions(x_bool)
    np.testing.assert_array_equal(all_res[0], np.all(x_bool))
    np.testing.assert_array_equal(all_res[1], np.all(x_bool, axis=0))
    np.testing.assert_array_equal(
        all_res[2], np.all(x_bool, axis=1, keepdims=True)
    )
    np.testing.assert_array_equal(all_res[3], np.any(x_bool))
    np.testing.assert_array_equal(all_res[4], np.any(x_bool, axis=0))
    np.testing.assert_array_equal(
        all_res[5], np.any(x_bool, axis=1, keepdims=True)
    )
    np.testing.assert_array_equal(all_res[6], np.all(x_bool))
    np.testing.assert_array_equal(all_res[7], np.any(x_bool))

    x_num = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    @jd.function
    def num_reductions(a):
      return (
          jd.math.reduce_sum(a, axis=1),
          jd.math.reduce_prod(a, axis=1),
          jd.math.reduce_min(a, axis=0),
          jd.math.reduce_max(a, axis=0),
          jd.math.reduce_mean(a, axis=1, keepdims=True),
          jd.math.reduce_euclidean_norm(a, axis=1),
          jd.reduce_sum(a),
          jd.reduce_prod(a),
          jd.reduce_min(a),
          jd.reduce_max(a),
          jd.reduce_mean(a),
          jd.reduce_euclidean_norm(a),
          jd.reduce_std(a),
          jd.reduce_variance(a),
      )

    res = num_reductions(x_num)
    np.testing.assert_allclose(res[0], np.sum(x_num, axis=1))
    np.testing.assert_allclose(res[1], np.prod(x_num, axis=1))
    np.testing.assert_allclose(res[2], np.min(x_num, axis=0))
    np.testing.assert_allclose(res[3], np.max(x_num, axis=0))
    np.testing.assert_allclose(res[4], np.mean(x_num, axis=1, keepdims=True))
    np.testing.assert_allclose(res[5], np.linalg.norm(x_num, axis=1), rtol=1e-5)
    np.testing.assert_allclose(res[6], np.sum(x_num))
    np.testing.assert_allclose(res[7], np.prod(x_num))
    np.testing.assert_allclose(res[8], np.min(x_num))
    np.testing.assert_allclose(res[9], np.max(x_num))
    np.testing.assert_allclose(res[10], np.mean(x_num))
    np.testing.assert_allclose(res[11], np.linalg.norm(x_num), rtol=1e-5)
    np.testing.assert_allclose(res[12], np.std(x_num), rtol=1e-5)
    np.testing.assert_allclose(res[13], np.var(x_num), rtol=1e-5)

  def testSerialization(self):
    @jd.function
    def my_reduce_sum(x):
      return (
          jd.math.reduce_sum(x, axis=1, keepdims=True),
          jd.math.reduce_sum(x),
      )

    x = np.array([[2, 4, 6, 8], [1, 3, 5, 7]], dtype=np.int32)
    expected = np.array([[20], [16]], dtype=np.int32)
    expected_default = np.array(36, dtype=np.int32)
    out, out_default = my_reduce_sum(x)
    np.testing.assert_array_equal(expected, out)
    np.testing.assert_array_equal(expected_default, out_default)

    processor_proto = my_reduce_sum.get_processor().to_proto()
    self.assertEqual(len(processor_proto.transforms), 2)
    self.assertEqual(
        processor_proto.transforms[0].op_name, "ReduceSumTransform"
    )
    self.assertEqual(
        processor_proto.transforms[1].op_name, "ReduceSumTransform"
    )

    config = reduction_transform_config_pb2.ReductionTransformConfig()
    self.assertTrue(processor_proto.transforms[0].detail.Unpack(config))
    self.assertTrue(config.keep_dims)
    self.assertEqual(list(config.axis), [1])

    config_default = reduction_transform_config_pb2.ReductionTransformConfig()
    self.assertTrue(processor_proto.transforms[1].detail.Unpack(config_default))
    self.assertFalse(config_default.keep_dims)
    self.assertEqual(list(config_default.axis), [0, 1])

    transform = cp.TransformBase.from_proto(processor_proto.transforms[0])
    ctx = cp.ExecuteContext()
    ctx.inputs = [cp.Value(cp.HostBuffer.from_dlpack(x))]
    transform.execute(ctx)
    self.assertEqual(len(ctx.outputs), 1)
    out_hb = ctx.outputs[0].as_encapsulated_type()
    assert isinstance(out_hb, cp.HostBuffer)
    # pyrefly: ignore[bad-argument-type]
    np.testing.assert_array_equal(np.from_dlpack(out_hb), expected)


class XlogyTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.supported_dtypes: list[Any] = [
        np.float16,
        ml_dtypes.bfloat16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]

  def testXlogyNoZero(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.array([[0.1, 0.2, 3.5], [-2.0, -5.0, 30.0]], dtype=dtype)
      y = np.array([[0.1, 0.2, 3.5], [3.1, 4.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.math.xlogy(a, b)

      res = fn(x, y)
      expected = (
          x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
          * np.log(
              y.astype(np.complex128 if np.iscomplexobj(y) else np.float64)
          )
      ).astype(dtype)
      rtol = 1e-2 if dtype in (np.float16, ml_dtypes.bfloat16) else 1e-5
      np.testing.assert_allclose(res, expected, rtol=rtol, atol=rtol)

  def testXlogyWithZero(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.zeros((2, 3), dtype=dtype)
      y = np.array([[0.1, 0.2, 3.5], [0.0, 1.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.xlogy(a, b)

      res = fn(x, y)
      zeros_np = np.zeros_like(y)
      np.testing.assert_array_equal(res, zeros_np)

  def testXlogyWithZeroBroadcast(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.array([[0.0], [1.0]], dtype=dtype)
      y = np.array([[0.0, 0.1, 0.2], [0.1, 1.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.math.xlogy(a, b)

      res = fn(x, y)
      expected_row1 = (
          y[1].astype(np.complex128 if np.iscomplexobj(y) else np.float64) * 0
          + np.log(
              y[1].astype(np.complex128 if np.iscomplexobj(y) else np.float64)
          )
      ).astype(dtype)
      rtol = 1e-2 if dtype in (np.float16, ml_dtypes.bfloat16) else 1e-5
      np.testing.assert_array_equal(res[0], np.zeros(3, dtype=dtype))
      np.testing.assert_allclose(res[1], expected_row1, rtol=rtol, atol=rtol)


class Xlog1pyTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.supported_dtypes: list[Any] = [
        np.float16,
        ml_dtypes.bfloat16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]

  def testXlog1pyNoNeg1(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.array([[0.1, 0.2, 3.5], [-2.0, -5.0, 30.0]], dtype=dtype)
      y = np.array([[-0.1, -0.2, 3.5], [3.1, -0.9, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.math.xlog1py(a, b)

      res = fn(x, y)
      expected = (
          x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
          * np.log1p(
              y.astype(np.complex128 if np.iscomplexobj(y) else np.float64)
          )
      ).astype(dtype)
      rtol = 1e-2 if dtype in (np.float16, ml_dtypes.bfloat16) else 1e-5
      np.testing.assert_allclose(res, expected, rtol=rtol, atol=rtol)

  def testXlog1pyWithNegOne(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.zeros((2, 3), dtype=dtype)
      y = np.array([[0.1, 0.2, 3.5], [-1.0, 1.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.xlog1py(a, b)

      res = fn(x, y)
      zeros_np = np.zeros_like(y)
      np.testing.assert_array_equal(res, zeros_np)

  def testXlog1pyWithZeroBroadcast(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.array([[0.0], [1.0]], dtype=dtype)
      y = np.array([[-1.0, -0.1, 0.2], [0.1, 1.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.math.xlog1py(a, b)

      res = fn(x, y)
      expected_row1 = np.log1p(
          y[1].astype(np.complex128 if np.iscomplexobj(y) else np.float64)
      ).astype(dtype)
      rtol = 1e-2 if dtype in (np.float16, ml_dtypes.bfloat16) else 1e-5
      np.testing.assert_array_equal(res[0], np.zeros(3, dtype=dtype))
      np.testing.assert_allclose(res[1], expected_row1, rtol=rtol, atol=rtol)


class XdivyTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.supported_dtypes: list[Any] = [
        np.float16,
        ml_dtypes.bfloat16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]

  def testXdivyNoZero(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.array([[0.1, 0.2, 3.5], [-2.0, -5.0, 30.0]], dtype=dtype)
      y = np.array([[0.1, 0.2, 3.5], [3.1, 4.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.math.xdivy(a, b)

      res = fn(x, y)
      expected = (x / y).astype(dtype)
      rtol = 1e-2 if dtype in (np.float16, ml_dtypes.bfloat16) else 1e-5
      np.testing.assert_allclose(res, expected, rtol=rtol, atol=rtol)

  def testXdivyWithZero(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.zeros((2, 3), dtype=dtype)
      y = np.array([[0.1, 0.2, 3.5], [0.0, 1.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.xdivy(a, b)

      res = fn(x, y)
      zeros_np = np.zeros_like(y)
      np.testing.assert_array_equal(res, zeros_np)

  def testXdivyWithZeroBroadcast(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.array([[0.0], [1.0]], dtype=dtype)
      y = np.array([[0.0, 0.1, 0.2], [0.1, 1.0, 2.0]], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.math.xdivy(a, b)

      res = fn(x, y)
      expected_row1 = (1.0 / y[1]).astype(dtype)
      rtol = 1e-2 if dtype in (np.float16, ml_dtypes.bfloat16) else 1e-5
      np.testing.assert_array_equal(res[0], np.zeros(3, dtype=dtype))
      np.testing.assert_allclose(res[1], expected_row1, rtol=rtol, atol=rtol)


class IgammaTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.supported_dtypes: list[Any] = [np.float32, np.float64]

  def testIgammaAndIgammac(self) -> None:
    for dtype in self.supported_dtypes:
      a = np.linspace(0.1, 2.0, 15).reshape(1, 3, 5).astype(dtype)
      x = np.linspace(0.1, 10.0, 15).reshape(1, 3, 5).astype(dtype)

      @jd.function
      def fn(a_in: Any, x_in: Any) -> Any:
        return (
            jd.math.igamma(a_in, x_in),
            jd.math.igammac(a_in, x_in),
            jd.igamma(a_in, x_in),
            jd.igammac(a_in, x_in),
        )

      igamma_res, igammac_res, igamma_top, igammac_top = fn(a, x)
      expected_igamma = special.gammainc(a, x).astype(dtype)
      expected_igammac = special.gammaincc(a, x).astype(dtype)

      np.testing.assert_allclose(
          igamma_res, expected_igamma, rtol=1e-5, atol=1e-5
      )
      np.testing.assert_allclose(
          igammac_res, expected_igammac, rtol=1e-5, atol=1e-5
      )
      np.testing.assert_allclose(
          igamma_top, expected_igamma, rtol=1e-5, atol=1e-5
      )
      np.testing.assert_allclose(
          igammac_top, expected_igammac, rtol=1e-5, atol=1e-5
      )
      np.testing.assert_allclose(
          igamma_res + igammac_res, np.ones_like(a), rtol=1e-5, atol=1e-5
      )


class ZetaAndPolygammaTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.supported_dtypes: list[Any] = [np.float32, np.float64]

  def testZeta(self) -> None:
    for dtype in self.supported_dtypes:
      # Need x > 1
      x = np.linspace(1.1, 11.0, 15).reshape(1, 3, 5).astype(dtype)
      q = np.linspace(0.1, 2.0, 15).reshape(1, 3, 5).astype(dtype)

      @jd.function
      def fn(x_in: Any, q_in: Any) -> Any:
        return jd.math.zeta(x_in, q_in), jd.zeta(x_in, q_in)

      res, res_top = fn(x, q)
      expected = special.zeta(x, q).astype(dtype)
      np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(res_top, expected, rtol=1e-5, atol=1e-5)

  def testPolygamma(self) -> None:
    for dtype in self.supported_dtypes:
      n = np.arange(0, 15).reshape(1, 3, 5).astype(dtype)
      x = np.linspace(0.5, 10.0, 15).reshape(1, 3, 5).astype(dtype)

      @jd.function
      def fn(a_in: Any, x_in: Any) -> Any:
        return (
            jd.math.polygamma(a_in, x_in),
            jd.polygamma(a_in, x_in),
        )

      res, res_top = fn(n, x)
      expected = special.polygamma(n.astype(int), x).astype(dtype)
      np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(res_top, expected, rtol=1e-5, atol=1e-5)


class MaximumMinimumTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.supported_dtypes: list[Any] = [
        np.float16,
        ml_dtypes.bfloat16,
        np.float32,
        np.float64,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ]
    self.float_dtypes: list[Any] = [
        np.float16,
        ml_dtypes.bfloat16,
        np.float32,
        np.float64,
    ]

  def testMaximumMinimumBasic(self) -> None:
    for dtype in self.supported_dtypes:
      x = np.array([1, 10, 5, 20], dtype=dtype)
      y = np.array([2, 5, 15, 20], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return (
            jd.math.maximum(a, b),
            jd.math.minimum(a, b),
            jd.maximum(a, b),
            jd.minimum(a, b),
        )

      max_res, min_res, max_top, min_top = fn(x, y)
      expected_max = np.maximum(x, y)
      expected_min = np.minimum(x, y)
      np.testing.assert_array_equal(max_res, expected_max)
      np.testing.assert_array_equal(min_res, expected_min)
      np.testing.assert_array_equal(max_top, expected_max)
      np.testing.assert_array_equal(min_top, expected_min)

  def testMaximumMinimumNaNPropagation(self) -> None:
    for dtype in self.float_dtypes:
      nan = dtype(np.nan)
      one = dtype(1.0)
      x = np.array([nan, one, nan], dtype=dtype)
      y = np.array([one, nan, nan], dtype=dtype)

      @jd.function
      def fn(a: Any, b: Any) -> Any:
        return jd.math.maximum(a, b), jd.math.minimum(a, b)

      max_res, min_res = fn(x, y)
      self.assertTrue(np.all(np.isnan(max_res.astype(np.float32))))
      self.assertTrue(np.all(np.isnan(min_res.astype(np.float32))))


if __name__ == "__main__":
  unittest.main()
