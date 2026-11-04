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

"""JAX processor for XSE."""

from collections.abc import Sequence
import dataclasses
from typing import Any

import jax
import jaxtyping

from xla_saved_model._src import jax2xsm
from xla_saved_model._src.core import saveable
from xla_saved_model._src.core import shlo_function
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.processors import processor


def _infer_polymorphic_signature(
    concrete_signatures: Sequence[jaxtyping.PyTree[shlo_type.ShloTensorSpec]],
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Infers a polymorphic signature from a sequence of concrete signatures.

  E.g.
  Given a sequence of concrete signatures:
  (
      {'inputs': (ShloTensorSpec(shape=(1, 3), dtype=jnp.float32}),}
      {'inputs': (ShloTensorSpec(shape=(2, 3), dtype=jnp.float32}),}
  )

  Infer a polymorphic signature for them:
    {'inputs': (ShloTensorSpec(shape=(None, 3), dtype=jnp.float32}),)}
  since the first dimension is different across provided signatures.


  Args:
    concrete_signatures: A sequence of concrete signatures where each signature
      contains shape dtype spec with concrete shapes.

  Returns:
    A polymorphic signature where shape may contain None to indicate
    polymorphic shapes. If shapes of all input signatures are concrete
    and the same, the output signature will be the same as the input
    signature and have concrete shapes.
  """
  if not concrete_signatures:
    return {}

  def _infer_spec(*specs: shlo_type.ShloTensorSpec) -> shlo_type.ShloTensorSpec:
    if not specs:
      raise ValueError("`specs` must not be empty.")
    shapes = zip(*(s.shape for s in specs))
    poly_shape = [dims[0] if len(set(dims)) == 1 else None for dims in shapes]
    return dataclasses.replace(specs[0], shape=poly_shape)

  return jax.tree_util.tree_map(
      _infer_spec,
      *concrete_signatures,
      is_leaf=lambda x: isinstance(x, shlo_type.ShloTensorSpec),
  )


def _calculate_jax_model_input_sig(
    apply_fn_input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    has_donated_output_args: bool = False,
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Calculates the input signature for a JAX model function by excluding the model parameters.

  Args:
    apply_fn_input_signature: The input signature of the `apply_fn`, which is
      expected to have exactly two positional arguments: the model parameters
      and the actual inputs. When the function was exported for output
      donation, a third positional argument holds the donated output buffers.
    has_donated_output_args: Whether a third positional argument holding the
      donated output buffers is present.

  Returns:
    The input signature excluding the model parameters and, if present, the
    donated output buffers, structured as `((args,), kwargs)`.
  """
  if (
      # Already arg-kwarg split if the input signature is a sequence of two
      # elements, where the first element is a sequence of positional arguments
      # and the second element is a dictionary of keyword arguments.
      isinstance(apply_fn_input_signature, Sequence)
      and len(apply_fn_input_signature) == 2
      and isinstance(apply_fn_input_signature[0], Sequence)
      and isinstance(apply_fn_input_signature[1], dict)
  ):
    pos_args, kwargs = apply_fn_input_signature
  else:
    pos_args = apply_fn_input_signature
    kwargs = {}
  if not isinstance(pos_args, Sequence):
    raise ValueError(
        "The positional-arguments part of apply_fn_input_signature must be a"
        f" sequence. Got: {pos_args}"
    )
  expected_num_pos_args = 3 if has_donated_output_args else 2
  if len(pos_args) != expected_num_pos_args:
    raise ValueError(
        "The positional part of apply_fn_input_signature must have exactly"
        f" {expected_num_pos_args} elements. Got: {pos_args}"
    )
  # The donated output buffers, like the model parameters, are not supplied by
  # whoever calls the model: they carry no data and the runtime fills them in.
  # Dropping them here keeps them out of every caller-facing signature derived
  # from this one, while leaving them in the function's own signature, which is
  # what `module_kept_var_idx` is indexed against.
  return ((pos_args[1],), kwargs)


@dataclasses.dataclass(frozen=True)
class PreparedJaxProcessor(processor.ProcessorDefaults):
  """JAX processor that is prepared for export."""

  name: str
  input_signature: jaxtyping.PyTree
  output_signature: jaxtyping.PyTree
  shlo_fns: Sequence[shlo_function.ShloFunction]
  weights_name: str | None = None

  @property
  def has_donated_output_args(self) -> bool:
    """Whether the exported functions end with one donated buffer per output.

    Read off the exported functions rather than off the requested export
    settings, so that it describes what the artifact actually contains.

    Returns:
      True if output donation was applied to this processor's functions.
    """
    return any(fn.num_input_aliased_output for fn in self.shlo_fns)

  def to_xsm_saveable(self) -> saveable.Saveable:
    return self.shlo_fns[0] if len(self.shlo_fns) == 1 else tuple(self.shlo_fns)


@dataclasses.dataclass(frozen=True)
class JaxProcessor:
  """JAX processor for XSE.

  Attributes:
    name: The name of the JAX processor.
    apply_fn_bundle: The ApplyFnBundle containing the function and export
      settings.
    params: Parameters associated with the function, if any.
    weights_name: Optional manifest object key for the weights bound to this
      processor. If None or empty, no weights are bound.
    jax_mesh: The JAX mesh for compilation.
  """

  name: str
  apply_fn_bundle: jax2xsm.ApplyFnBundle
  params: jaxtyping.PyTree | None = None
  weights_name: str | None = None
  jax_mesh: jax.sharding.Mesh | None = None

  def prepare(
      self,
      input_signature: jaxtyping.PyTree,
      **kwargs: Any,
  ) -> PreparedJaxProcessor:
    """Prepares the JAX processor for export.

    Args:
      input_signature: The input signature of the JAX model's method.
      **kwargs: Additional keyword arguments (currently unused).

    Returns:
      A `PreparedJaxProcessor` instance containing the necessary information
      for export.

    Raises:
      ValueError: If the processor name is empty, or if no SHLO functions
        are found for the model with the given input signature and method key.
    """

    del kwargs

    if not self.name:
      raise ValueError("JaxProcessor name must not be empty")

    if self.params is not None:
      param_names_tree = jax.tree.map_with_path(
          lambda path, _: jax.tree_util.keystr(
              path, simple=True, separator="."
          ),
          self.params,
      )
      model_param_names = jax.tree.leaves(param_names_tree)
    else:
      model_param_names = None

    shlo_functions = jax2xsm.convert_apply_fn_bundle_to_shlo(
        apply_fn_bundle=self.apply_fn_bundle,
        params=self.params,
        input_signature=jax2xsm.utils.to_jax_spec(input_signature),
        model_param_names=model_param_names,
        jax_mesh=self.jax_mesh,
        weights_name=self.weights_name or None,
    )

    if not shlo_functions:
      raise ValueError("No SHLO functions found for model")

    # The internal SHLO function signature includes the model parameters as
    # the first argument, and the donated output buffers (if the model was
    # exported for output donation) as a trailing argument. We exclude them
    # here to create the external-facing input signature that only contains the
    # actual user inputs.
    jax_input_signature = _calculate_jax_model_input_sig(
        _infer_polymorphic_signature(
            [func.input_signature for func in shlo_functions]
        ),
        has_donated_output_args=self.apply_fn_bundle.use_input_aliasing,
    )
    jax_output_signature = _infer_polymorphic_signature(
        [func.output_signature for func in shlo_functions]
    )

    return PreparedJaxProcessor(
        self.name,
        jax_input_signature,
        jax_output_signature,
        shlo_functions,
        self.weights_name,
    )
