"""Library for exporting JAX models to Neptune model format.

This library provides the `save` function to convert and save JAX models
according to the Neptune Model Specification. It uses `jax2npt` to convert
JAX functions to SHLO functions, and `persistence_lib` to save them
in Neptune's format.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

from absl import logging
import jax.export as jax_export
import jaxtyping

from neptune_model._src import jax2npt
from neptune_model._src.core import persistence_lib
from neptune_model._src.core import saveable
from neptune_model._src.core import shlo_type
from neptune_model._src.export import constants
from neptune_model._src.export import jax_model
from neptune_model._src.export import serving_config
from neptune_model._src.export import utils
from neptune_model.protos import orchestration_pb2


@dataclasses.dataclass
class NeptuneModelSaveBundle:
  """A bundle of objects to be saved in the Neptune model format.

  Attributes:
    manifest_objects: A dictionary of saveable objects to be saved in the
      manifest.
    pipelines: The pipeline definitions for model orchestration.
  """

  manifest_objects: dict[str, saveable.Saveable] = dataclasses.field(
      default_factory=dict
  )
  pipelines: orchestration_pb2.Pipelines = dataclasses.field(
      default_factory=orchestration_pb2.Pipelines
  )


@dataclasses.dataclass
class PipelineNode:
  """A node in the pipeline graph.

  Attributes:
    node: The node function.
    role: The role of this node, e.g., preprocessor, model, or postprocessor.
  """

  node: jax_model.JaxModel
  role: orchestration_pb2.Role


Signature = tuple[
    jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    jaxtyping.PyTree[shlo_type.ShloTensorSpec],
]  # (input_signature, output_signature)


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


def _update_save_bundle_for_jax_model(
    model: jax_model.JaxModel,
    method_key: str | None,
    input_signature: jaxtyping.PyTree,
    input_polymophic_shapes: Any | None,
    polymorphic_constraints: Sequence[str],
    serving_signature: str,
    save_bundle: NeptuneModelSaveBundle,
) -> Signature:
  """Converts a JAX model to a Saveable SHLO function and updates the save bundle.

  This function takes a JaxModel and input signatures, potentially selects a
  specific method from the model's apply_fn if it's a mapping, and then converts
  it into a Saveable object using jax2npt.convert., and adds it to the
  save_bundleShloFunction

  Args:
    model: The JaxModel instance to convert.
    method_key: Optional key to select a specific function if model.apply_fn is
      a Mapping. Required if model.apply_fn is a Mapping.
    input_signature: A PyTree representing the structure, shapes, and dtypes of
      the model inputs.
    input_polymophic_shapes: The polymorphic shapes of the input signature. See
      `jax.export.symbolic_args_specs` for more details.
    polymorphic_constraints: Constraints on polymorphic dimensions, e.g., "b >=
      1". See `jax.export.symbolic_args_specs` for more details.
    serving_signature: The key to use for the converted function in the
      save_bundle.
    save_bundle: The NeptuneModelSaveBundle to update with the converted
      function.

  Returns:
    A tuple containing the input and output signatures of the converted
    function.
  Raises:
    ValueError: If model.apply_fn is a Mapping but method_key is not provided
      or if the provided method_key is not found in the mapping.
  """
  apply_fn = utils.get_apply_fn(model, method_key)

  args_specs = jax_export.symbolic_args_specs(
      args=utils.to_jax_spec((model.params, input_signature)),
      shapes_specs=(
          "...",  #  "..." is same as `None`,
          input_polymophic_shapes,
      ),
      constraints=polymorphic_constraints,
  )

  model_shlo_function = jax2npt.convert(
      fun_jax=apply_fn,
      args_spec=args_specs,
      kwargs_spec={},  # kwargs left empty for now
  )
  save_bundle.manifest_objects[serving_signature] = model_shlo_function

  return (
      _calculate_jax_model_input_sig(model_shlo_function.input_signature),
      model_shlo_function.output_signature,
  )


def _create_pipeline(model: jax_model.JaxModel) -> Sequence[PipelineNode]:
  return [PipelineNode(node=model, role=orchestration_pb2.ROLE_MODEL)]


def _update_save_bundle(
    pipeline: Sequence[PipelineNode],
    cfg: serving_config.ServingConfig,
    save_bundle: NeptuneModelSaveBundle,
) -> None:
  """Processes a pipeline of nodes and updates the save bundle.

  Currently, only nodes with role MODEL are supported.

  Args:
    pipeline: A sequence of PipelineNodes to process.
    cfg: The serving configuration to use for conversion.
    save_bundle: The NeptuneModelSaveBundle to update.

  Raises:
    ValueError: If a node with an unsupported role is encountered.
  """
  components = []
  for node in pipeline:
    if node.role == orchestration_pb2.ROLE_MODEL:
      _, _ = _update_save_bundle_for_jax_model(
          model=node.node,
          method_key=cfg.method_key,
          input_signature=cfg.input_signature,
          input_polymophic_shapes=cfg.input_polymophic_shapes,
          polymorphic_constraints=cfg.polymorphic_constraints,
          serving_signature=cfg.serving_signature,
          save_bundle=save_bundle,
      )
      components.append(
          orchestration_pb2.Component(
              role=node.role,
              function_name=cfg.serving_signature,
          )
      )
    else:
      raise ValueError(f"Unsupported role: {node.role}")

  save_bundle.pipelines.name_to_pipeline[
      cfg.serving_signature
  ].components.extend(components)


def save(
    model: jax_model.JaxModel,
    export_path: str,
    serving_configs: Sequence[serving_config.ServingConfig],
) -> None:
  """Saves a JAX model to neptune model format.

  Args:
    model: The JAX model to save.
    export_path: The directory to which to save the model.
    serving_configs: The serving configurations.
  """
  logging.info(
      "Saving model=%s to export_path=%s with serving_configs=%s",
      model,
      export_path,
      serving_configs,
  )
  save_bundle = NeptuneModelSaveBundle()
  for cfg in serving_configs:
    pipeline = _create_pipeline(model)
    _update_save_bundle(
        pipeline=pipeline,
        cfg=cfg,
        save_bundle=save_bundle,
    )

  save_options = persistence_lib.SaveOptions(
      version=2,
      supplementals={
          constants.ORCHESTRATION_SUPPLEMENTAL_NAME: (
              utils.build_orchestration_supplemental(save_bundle.pipelines)
          )
      },
  )

  persistence_lib.save(
      module=save_bundle.manifest_objects,
      target_dir=export_path,
      options=save_options,
  )
