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

"""JAX processor for Neptune."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import pathlib
from typing import Any

import jax
import jaxtyping
import orbax.checkpoint as ocp

from neptune_model._src import jax2npt
from neptune_model._src.core import saveable
from neptune_model._src.core import shlo_function
from neptune_model._src.core import shlo_type
from neptune_model._src.core import value
from neptune_model._src.export import constants
from neptune_model._src.export import jax_model
from neptune_model._src.export import utils


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
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Calculates the input signature for a JAX model function by excluding the model parameters.

  Args:
    apply_fn_input_signature: The input signature of the `apply_fn`, which is
      expected to have exactly two positional arguments: the model parameters
      and the actual inputs.

  Returns:
    The input signature excluding the model parameters, structured as
    `((args,), kwargs)`.
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
  if len(pos_args) != 2:
    raise ValueError(
        "The positional part of apply_fn_input_signature must have exactly"
        f" two elements. Got: {pos_args}"
    )
  return ((pos_args[1],), kwargs)


@dataclasses.dataclass(frozen=True)
class PreparedJaxProcessor:
  """JAX processor that is prepared for export."""

  name: str
  input_signature: jaxtyping.PyTree
  output_signature: jaxtyping.PyTree
  shlo_fns: Sequence[shlo_function.ShloFunction]
  checkpoint_save_fn: (
      Callable[[str], Mapping[str, value.ExternalValue]] | None
  ) = None
  weights_name: str | None = None

  @property
  def checkpoint_path(self) -> str | None:
    return self.checkpoint_path or constants.DEFAULT_CHECKPOINT_SUBDIR

  def to_npt_saveable(self) -> saveable.Saveable:
    return self.shlo_fns[0] if len(self.shlo_fns) == 1 else tuple(self.shlo_fns)

  def on_persist(
      self,
  ) -> Callable[[str], Mapping[str, value.ExternalValue]] | None:
    return self.checkpoint_save_fn


@dataclasses.dataclass(frozen=True)
class JaxProcessor:
  """JAX processor for Neptune.

  Attributes:
    name: The name of the JAX processor.
    model: The JAX model to process.
    method_key: The method key of the JAX model to process.
    checkpoint_path: The path to the checkpoint directory. If None, a default
      path will be used.
    weights_name: The name of the weights file. If None, a default name will be
      used.
  """

  name: str
  model: jax_model.JaxModel
  method_key: str | None = None
  checkpoint_path: str | None = None
  weights_name: str | None = None

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

    # Instructs the runtime to only load the model parameters from the
    # checkpoint, not all keys present in the checkpoint.
    param_names_tree = jax.tree.map_with_path(
        lambda path, _: jax.tree_util.keystr(path, simple=True, separator="."),
        self.model.params,
    )

    shlo_functions = self.model.get_shlo_functions(
        utils.to_jax_spec(input_signature),
        self.method_key,
        model_param_names=jax.tree.leaves(param_names_tree),
    )

    if not shlo_functions:
      raise ValueError("No SHLO functions found for model")

    # The internal SHLO function signature includes the model parameters as
    # the first argument. We exclude them here to create the external-facing
    # input signature that only contains the actual user inputs.
    jax_input_signature = _calculate_jax_model_input_sig(
        _infer_polymorphic_signature(
            [func.input_signature for func in shlo_functions]
        )
    )
    jax_output_signature = _infer_polymorphic_signature(
        [func.output_signature for func in shlo_functions]
    )

    if self.model.params is not None:
      if self.checkpoint_path == constants.DEFAULT_CHECKPOINT_SUBDIR:
        # Backward compatibility: if the checkpoint path is the default
        # checkpoint subdirectory, this is the main exported JAX model
        # and we place it under the default checkpoint subdirectory.
        checkpoint_path = self.checkpoint_path
      else:
        checkpoint_path = (
            self.checkpoint_path or f"processor_checkpoint/{self.name}"
        )

      def _save_checkpoint(
          model_path: str,
      ) -> Mapping[str, value.ExternalValue]:
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        checkpointer.save(
            pathlib.Path(model_path) / checkpoint_path,
            self.model.params,
            force=True,
        )

        weights_name = self.weights_name or f"params_{self.name}"
        return {
            weights_name: jax2npt.convert_path_to_value(
                checkpoint_path,
                constants.ORBAX_CHECKPOINT_MIME_TYPE,
            )
        }

      checkpoint_save_fn = _save_checkpoint
    else:
      checkpoint_save_fn = None

    return PreparedJaxProcessor(
        self.name,
        jax_input_signature,
        jax_output_signature,
        shlo_functions,
        checkpoint_save_fn=checkpoint_save_fn,
        weights_name=self.weights_name,
    )
