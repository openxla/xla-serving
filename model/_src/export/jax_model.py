"""A JAX model, consisting of an apply_fn and its parameters."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any
import jax
import jax.export as jax_export
import jaxtyping
from neptune_model._src import jax2npt
from neptune_model._src.core import shlo_function
from neptune_model._src.core import shlo_type
from neptune_model._src.export import utils
from neptune_model._src.jax2npt.typing import ApplyFn


@dataclasses.dataclass(frozen=True)
class ApplyFnBundle:
  """Bundles an apply_fn with arguments for jax2npt.convert.

  Attributes:
    apply_fn: The function to apply.
    input_polymophic_shapes: Polymorphic shapes for function inputs, for
      `shapes_specs` in `jax.export.symbolic_args_specs`.
    polymorphic_constraints: Constraints on polymorphic dimensions, for
      `constraints` in `jax.export.symbolic_args_specs`.
    jax_export_kwargs: Keyword arguments for `jax.export.export`. Current
      available kwargs are platforms, _override_lowering_rules, disabled_checks.
      See
      https://docs.jax.dev/en/latest/_autosummary/jax.export.export.html#jax.export.export.
  """

  apply_fn: ApplyFn
  input_polymophic_shapes: Any | None = None
  polymorphic_constraints: Sequence[str] = ()
  jax_export_kwargs: Mapping[str, Any] | None = None


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

  @property
  def jax_mesh(self) -> jax.sharding.Mesh | None:
    return self._jax_mesh

  def get_shlo_function(
      self,
      input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
      method_key: str | None = None,
  ) -> shlo_function.ShloFunction:
    """Returns the SHLO function for the given method and input signature."""
    if method_key is None:
      method_key = next(iter(self._apply_fn_bundle))
    apply_fn_bundle = self._apply_fn_bundle[method_key]

    args_specs = jax_export.symbolic_args_specs(
        args=utils.to_jax_spec((self.params, input_signature)),
        shapes_specs=(
            '...',
            apply_fn_bundle.input_polymophic_shapes,
        ),
        constraints=apply_fn_bundle.polymorphic_constraints,
    )
    return jax2npt.convert(
        fun_jax=apply_fn_bundle.apply_fn,
        args_spec=args_specs,
        kwargs_spec={},
        jax_export_kwargs=apply_fn_bundle.jax_export_kwargs,
    )
