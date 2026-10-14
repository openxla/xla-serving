"""Bitwise operations namespace."""

from typing import Any

from data.kernels import numerical_transforms_module as _nt
from data.kernels.numerical_transforms_module import bitwiseand as bitwise_and
from data.kernels.numerical_transforms_module import bitwiseor as bitwise_or
from data.kernels.numerical_transforms_module import bitwisexor as bitwise_xor
from data.kernels.numerical_transforms_module import invert
from data.kernels.numerical_transforms_module import leftshift as left_shift
from data.kernels.numerical_transforms_module import rightshift as right_shift


def __getattr__(name: str) -> Any:
  return getattr(_nt, name)


# Export the following ops under the bitwise namespace.
__all__ = [
    # keep-sorted start
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
    # keep-sorted end
]
