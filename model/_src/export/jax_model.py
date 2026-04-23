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

"""A JAX model, consisting of an apply_fn and its parameters."""

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
from typing import Any
from absl import logging
import jax
import jax.export as jax_export
import jaxtyping
from neptune_model._src import jax2npt
from neptune_model._src.core import compile_options_util
from neptune_model._src.core import shlo_function
from neptune_model._src.core import shlo_type
from neptune_model._src.export import constants
from neptune_model._src.export import utils
from neptune_model._src.jax2npt.typing import ApplyFn


def _concretize_polymorphic_shapes(
    symbolic_specs: (
        Mapping[Any, jaxtyping.PyTree[jax.ShapeDtypeStruct]]
        | Sequence[jaxtyping.PyTree[jax.ShapeDtypeStruct]]
    ),
    symbol_to_values: Mapping[str, Sequence[int]],
) -> Sequence[jaxtyping.PyTree[jax.ShapeDtypeStruct]]:
  """Concretizes a symbolic polymorphic shape to a list of concrete shapes.

  This function is used to concretize a symbolic spec with polymorphic shapes to
  a list of specs with concrete shapes.

  For example,
  Given a symbolic spec with polymorphic shapes
  {'prompt': jax.ShapeDtypeStruct(shape=(b, l), dtype=int32)}

  and the discrete values for the symbols
  {'b': [1, 2], 'l': [3, 4]},

  The result of concretization will be a list of specs with concrete shapes,
  [
      {'prompt': jax.ShapeDtypeStruct(shape=(1, 3), dtype=int32) },
      {'prompt': jax.ShapeDtypeStruct(shape=(1, 4), dtype=int32)},
      {'prompt': jax.ShapeDtypeStruct(shape=(2, 3), dtype=int32)},
      {'prompt': jax.ShapeDtypeStruct(shape=(2, 4), dtype=int32)},
  ]

  Args:
    symbolic_specs: The symbolic specs to concretize. For example, {'prompt':
      jax.ShapeDtypeStruct(shape=(b, l), dtype=int32)}.
    symbol_to_values: A mapping of symbol names to discrete values. For example,
      {'b': [1, 2], 'l': [3, 4]}.

  Returns:
    A sequence of concretized specs.
  """
  symbol_names = list(symbol_to_values.keys())
  # A list of tuples, where each tuple is a combo of symbol values.
  # For example, if the symbol names are ['b', 'l'] and the values for each
  # symbol are [[1, 2], [3, 4]], then the symbol_value_combos will be
  # [(1, 3), (1, 4), (2, 3), (2, 4)].
  symbol_value_combos = list(
      itertools.product(*(symbol_to_values[name] for name in symbol_names))
  )

  def _concretize_shapes(symbol_to_value: Mapping[str, int]):
    return jax.tree_util.tree_map(
        lambda spec: _concretize_spec(spec, symbol_to_value),
        symbolic_specs,
    )

  logging.info(
      '%s Concretizing polymorphic shapes for symbols %s with values %s.',
      constants.LOG_PREFIX,
      symbol_names,
      symbol_value_combos,
  )
  return [
      _concretize_shapes(dict(zip(symbol_names, symbol_value_combo)))
      for symbol_value_combo in symbol_value_combos
  ]


def _concretize_spec(
    symbolic_spec: jax.ShapeDtypeStruct,
    symbol_to_value: Mapping[str, int],
) -> jax.ShapeDtypeStruct:
  """Concretizes a symbolic spec with polymorphic shapes to a concrete one."""
  concrete_shape = []
  for d in symbolic_spec.shape:
    if isinstance(d, int):
      concrete_shape.append(d)
    else:
      # symbolic dim is an instance of shape_poly._DimExpr which
      # is private/internal to JAX, but it can be converted to a string.
      d_str = str(d)
      if d_str not in symbol_to_value:
        raise ValueError(
            f'Symbol {d_str!r} not found in symbol_to_value map:'
            f' {symbol_to_value}'
        )
      concrete_shape.append(symbol_to_value[d_str])
  return jax.ShapeDtypeStruct(tuple(concrete_shape), symbolic_spec.dtype)


@dataclasses.dataclass(frozen=True)
class ApplyFnBundle:
  """Bundles an apply_fn with arguments for jax2npt.convert.

  Attributes:
    apply_fn: The function to apply.
    input_polymophic_shapes: Polymorphic shapes for function inputs, for
      `shapes_specs` in `jax.export.symbolic_args_specs`.
    polymorphic_constraints: Constraints on polymorphic dimensions, for
      `constraints` in `jax.export.symbolic_args_specs`.
    input_polymorphic_shape_symbol_values: A mapping of symbol names to values
      for the symbols in `input_polymophic_shapes`.
    jax_export_kwargs: Keyword arguments for `jax.export.export`. Current
      available kwargs are platforms, _override_lowering_rules, disabled_checks.
      See
      https://docs.jax.dev/en/latest/_autosummary/jax.export.export.html#jax.export.export.
  """

  apply_fn: ApplyFn
  input_polymophic_shapes: Any | None = None
  polymorphic_constraints: Sequence[str] = ()
  input_polymorphic_shape_symbol_values: Mapping[str, Sequence[int]] | None = (
      None
  )
  jax_export_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


SERVING_DEFAULT_KEY = 'serving_default'


class JaxModel:
  """A JAX model, consisting of an apply_fn and its parameters.

  Exactly one of `params` or `checkpoint_path` must be provided.

  Attributes:
    apply_fn_bundle: The function bundle for applying the model, or a mapping of
      such bundles.
    params: The parameters of the model.
    jax_mesh: Mesh for the model.
  """

  def __init__(
      self,
      apply_fn_bundle: ApplyFnBundle | Mapping[str, ApplyFnBundle],
      params: jaxtyping.PyTree | None = None,
      jax_mesh: jax.sharding.Mesh | None = None,
  ):
    if isinstance(apply_fn_bundle, Mapping):
      self._apply_fn_bundle = apply_fn_bundle
    else:
      self._apply_fn_bundle = {SERVING_DEFAULT_KEY: apply_fn_bundle}
    self._params = params
    self._jax_mesh = jax_mesh

  @property
  def params(self) -> jaxtyping.PyTree | None:
    return self._params

  def method_keys(self) -> Sequence[str]:
    return list(self._apply_fn_bundle.keys())

  @property
  def jax_mesh(self) -> jax.sharding.Mesh | None:
    return self._jax_mesh

  def get_shlo_functions(
      self,
      input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
      method_key: str | None = None,
  ) -> Sequence[shlo_function.ShloFunction]:
    """Returns the SHLO function for the given method and input signature."""
    if method_key is None:
      if len(self._apply_fn_bundle) != 1:
        raise ValueError(
            'Method key is not specified but model has more than one apply_fn'
        )
      (method_key,) = self._apply_fn_bundle.keys()
    if method_key not in self._apply_fn_bundle:
      raise ValueError(f'Method key {method_key!r} not found in ApplyFnBundle')
    apply_fn_bundle = self._apply_fn_bundle[method_key]

    args_specs = jax_export.symbolic_args_specs(
        args=utils.to_jax_spec((self.params, input_signature)),
        shapes_specs=(
            '...',
            apply_fn_bundle.input_polymophic_shapes,
        ),
        constraints=apply_fn_bundle.polymorphic_constraints,
    )
    if apply_fn_bundle.input_polymorphic_shape_symbol_values is not None:
      args_specs_list = _concretize_polymorphic_shapes(
          args_specs,
          apply_fn_bundle.input_polymorphic_shape_symbol_values,
      )
    else:
      args_specs_list = [args_specs]

    compile_options_per_platform = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=apply_fn_bundle.jax_export_kwargs.get(
            'platforms', None
        ),
        jax_mesh=self._jax_mesh,
    )
    return tuple(
        jax2npt.convert(
            fun_jax=apply_fn_bundle.apply_fn,
            args_spec=args_spec,
            kwargs_spec={},
            jax_export_kwargs=apply_fn_bundle.jax_export_kwargs,
            xla_compile_options_per_platform=compile_options_per_platform,
        )
        for args_spec in args_specs_list
    )
