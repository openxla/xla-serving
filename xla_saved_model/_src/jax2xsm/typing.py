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

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any
import jaxtyping


# ApplyFn take two arguments, the first one is the model_params, the second one
# is the model_inputs.
ApplyFn = Callable[[jaxtyping.PyTree, jaxtyping.PyTree], jaxtyping.PyTree]


@dataclasses.dataclass(frozen=True)
class ApplyFnBundle:
  """Bundles an apply_fn with arguments for jax2xsm.convert.

  Attributes:
    apply_fn: The function to apply.
    input_polymorphic_shapes: Polymorphic shapes for function inputs, for
      `shapes_specs` in `jax.export.symbolic_args_specs`.
    polymorphic_constraints: Constraints on polymorphic dimensions, for
      `constraints` in `jax.export.symbolic_args_specs`.
    input_polymorphic_shape_symbol_values: A mapping of symbol names to values
      for the symbols in `input_polymorphic_shapes`.
    jax_export_kwargs: Keyword arguments for `jax.export.export`. Current
      available kwargs are platforms, _override_lowering_rules, disabled_checks.
      See
      https://docs.jax.dev/en/latest/_autosummary/jax.export.export.html#jax.export.export.
    use_input_aliasing: If True, the exported function takes one extra input
      argument per output, aliased to that output (`tf.aliasing_output`), so
      that an execution writes its results into the memory passed for those
      inputs instead of into buffers it allocates itself. This lets a runtime
      pre-allocate (and reuse) output memory. The extra arguments are appended
      after `apply_fn`'s own arguments and are derived automatically from the
      function's outputs; they are not part of the model's caller-facing
      signature. Only meaningful for AOT serving, so input shapes must be
      concrete: with `input_polymorphic_shapes`,
        `input_polymorphic_shape_symbol_values` is required.
    enable_input_r1_optimization: If True, flattens eligible static rank >= 2
      user inputs to 1-D at the StableHLO entry point and reshapes them back
      inside the function.
    enable_output_r1_optimization: If True, flattens eligible static rank >= 2
      outputs to 1-D before returning from the StableHLO function.
    max_transpose_channels: If > 0, transposes eligible 4-D NHWC user inputs
      with `C <= max_transpose_channels` and `C < W` to NCHW at the StableHLO
      entry point (recording `host_transpose = (0, 3, 1, 2)`) and transposes
      them back to NHWC on device.
  """

  apply_fn: ApplyFn
  input_polymorphic_shapes: Any | None = None
  polymorphic_constraints: Sequence[str] = ()
  input_polymorphic_shape_symbol_values: Mapping[str, Sequence[int]] | None = (
      None
  )
  jax_export_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  use_input_aliasing: bool = False
  enable_input_r1_optimization: bool = False
  enable_output_r1_optimization: bool = False
  max_transpose_channels: int = 0
