"""Math operations namespace."""

from data.kernels.numerical_transforms_module import abs  # pylint: disable=redefined-builtin
from data.kernels.numerical_transforms_module import acos
from data.kernels.numerical_transforms_module import acosh
from data.kernels.numerical_transforms_module import add
from data.kernels.numerical_transforms_module import asin
from data.kernels.numerical_transforms_module import asinh
from data.kernels.numerical_transforms_module import atan
from data.kernels.numerical_transforms_module import atan2
from data.kernels.numerical_transforms_module import atanh
from data.kernels.numerical_transforms_module import besseli0 as bessel_i0
from data.kernels.numerical_transforms_module import besseli0e as bessel_i0e
from data.kernels.numerical_transforms_module import besseli1 as bessel_i1
from data.kernels.numerical_transforms_module import besseli1e as bessel_i1e
from data.kernels.numerical_transforms_module import besselj0 as bessel_j0
from data.kernels.numerical_transforms_module import besselj1 as bessel_j1
from data.kernels.numerical_transforms_module import besselk0 as bessel_k0
from data.kernels.numerical_transforms_module import besselk0e as bessel_k0e
from data.kernels.numerical_transforms_module import besselk1 as bessel_k1
from data.kernels.numerical_transforms_module import besselk1e as bessel_k1e
from data.kernels.numerical_transforms_module import bessely0 as bessel_y0
from data.kernels.numerical_transforms_module import bessely1 as bessel_y1
from data.kernels.numerical_transforms_module import ceil
from data.kernels.numerical_transforms_module import complexabs as complex_abs
from data.kernels.numerical_transforms_module import conj
from data.kernels.numerical_transforms_module import cos
from data.kernels.numerical_transforms_module import cosh
from data.kernels.numerical_transforms_module import dawsn
from data.kernels.numerical_transforms_module import digamma
from data.kernels.numerical_transforms_module import dividenonan as divide_no_nan
from data.kernels.numerical_transforms_module import erf
from data.kernels.numerical_transforms_module import erfc
from data.kernels.numerical_transforms_module import erfinv
from data.kernels.numerical_transforms_module import exp
from data.kernels.numerical_transforms_module import expint
from data.kernels.numerical_transforms_module import expm1
from data.kernels.numerical_transforms_module import floor
from data.kernels.numerical_transforms_module import floordiv
from data.kernels.numerical_transforms_module import floormod
from data.kernels.numerical_transforms_module import fresnelcos as fresnel_cos
from data.kernels.numerical_transforms_module import fresnelsin as fresnel_sin
from data.kernels.numerical_transforms_module import igamma
from data.kernels.numerical_transforms_module import igammac
from data.kernels.numerical_transforms_module import isfinite
from data.kernels.numerical_transforms_module import isinf
from data.kernels.numerical_transforms_module import isnan
from data.kernels.numerical_transforms_module import lgamma
from data.kernels.numerical_transforms_module import log
from data.kernels.numerical_transforms_module import log1p
from data.kernels.numerical_transforms_module import logicaland as logical_and
from data.kernels.numerical_transforms_module import logicalnot as logical_not
from data.kernels.numerical_transforms_module import logicalor as logical_or
from data.kernels.numerical_transforms_module import logicalxor as logical_xor
from data.kernels.numerical_transforms_module import maximum
from data.kernels.numerical_transforms_module import minimum
from data.kernels.numerical_transforms_module import multiply
from data.kernels.numerical_transforms_module import multiplynonan as multiply_no_nan
from data.kernels.numerical_transforms_module import ndtri
from data.kernels.numerical_transforms_module import negative
from data.kernels.numerical_transforms_module import polygamma
from data.kernels.numerical_transforms_module import pow  # pylint: disable=redefined-builtin
from data.kernels.numerical_transforms_module import reciprocal
from data.kernels.numerical_transforms_module import rint
from data.kernels.numerical_transforms_module import round  # pylint: disable=redefined-builtin
from data.kernels.numerical_transforms_module import rsqrt
from data.kernels.numerical_transforms_module import sigmoid
from data.kernels.numerical_transforms_module import sign
from data.kernels.numerical_transforms_module import sin
from data.kernels.numerical_transforms_module import sinh
from data.kernels.numerical_transforms_module import spence
from data.kernels.numerical_transforms_module import sqrt
from data.kernels.numerical_transforms_module import square
from data.kernels.numerical_transforms_module import subtract
from data.kernels.numerical_transforms_module import tan
from data.kernels.numerical_transforms_module import tanh
from data.kernels.numerical_transforms_module import truediv
from data.kernels.numerical_transforms_module import xdivy
from data.kernels.numerical_transforms_module import xlog1py
from data.kernels.numerical_transforms_module import xlogy
from data.kernels.numerical_transforms_module import zeta
from data.ops.reduction_ops import reduce_all
from data.ops.reduction_ops import reduce_any
from data.ops.reduction_ops import reduce_euclidean_norm
from data.ops.reduction_ops import reduce_max
from data.ops.reduction_ops import reduce_mean
from data.ops.reduction_ops import reduce_min
from data.ops.reduction_ops import reduce_prod
from data.ops.reduction_ops import reduce_std
from data.ops.reduction_ops import reduce_sum
from data.ops.reduction_ops import reduce_variance
from data.ops.scan_ops import cumprod
from data.ops.scan_ops import cumsum
from data.ops.scan_ops import cumulative_logsumexp

# Alias for backwards compatibility
neg = negative
# TensorFlow Python 3 division parity.
divide = truediv


# Export the following ops under the jd.math namespace.
__all__ = [
    # keep-sorted start
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "bessel_j0",
    "bessel_j1",
    "bessel_k0",
    "bessel_k0e",
    "bessel_k1",
    "bessel_k1e",
    "bessel_y0",
    "bessel_y1",
    "ceil",
    "complex_abs",
    "conj",
    "cos",
    "cosh",
    "cumprod",
    "cumsum",
    "cumulative_logsumexp",
    "dawsn",
    "digamma",
    "divide",
    "divide_no_nan",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "expint",
    "expm1",
    "floor",
    "floordiv",
    "floormod",
    "fresnel_cos",
    "fresnel_sin",
    "igamma",
    "igammac",
    "isfinite",
    "isinf",
    "isnan",
    "lgamma",
    "log",
    "log1p",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "multiply_no_nan",
    "ndtri",
    "neg",
    "negative",
    "polygamma",
    "pow",
    "reciprocal",
    "reduce_all",
    "reduce_any",
    "reduce_euclidean_norm",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_std",
    "reduce_sum",
    "reduce_variance",
    "rint",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "spence",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "truediv",
    "xdivy",
    "xlog1py",
    "xlogy",
    "zeta",
    # keep-sorted end
]
