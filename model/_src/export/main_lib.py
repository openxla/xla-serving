"""Library for exporting JAX models."""

from collections.abc import Sequence
import dataclasses

from absl import logging
import jaxtyping

from google3.third_party.neptune.model._src import jax2npt
from google3.third_party.neptune.model._src.core import persistence_lib
from google3.third_party.neptune.model._src.core import saveable
from google3.third_party.neptune.model._src.core import shlo_type
from google3.third_party.neptune.model._src.export import jax_model
from google3.third_party.neptune.model._src.export import serving_config
from google3.third_party.neptune.model._src.export import utils
from google3.third_party.neptune.protos import orchestration_pb2


@dataclasses.dataclass
class NeptuneModelSaveBundle:
  """A bundle of objects to be saved in the Neptune model format.

  Attributes:
    manifest_objects: A dictionary of saveable objects to be saved in the
      manifest.
  """

  manifest_objects: dict[str, saveable.Saveable] = dataclasses.field(
      default_factory=dict
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


def _update_save_bundle_for_jax_model(
    input_signature: jaxtyping.PyTree,
    model: jax_model.JaxModel,
    method_key: str | None,
    serving_signature: str,
    save_bundle: NeptuneModelSaveBundle,
) -> Signature:
  """Converts a JAX model to a Saveable SHLO function and updates the save bundle.

  This function takes a JaxModel and input signatures, potentially selects a
  specific method from the model's apply_fn if it's a mapping, and then converts
  it into a Saveable object using jax2npt.convert., and adds it to the
  save_bundleShloFunction

  Args:
    input_signature: A PyTree representing the structure, shapes, and dtypes of
      the model inputs.
    model: The JaxModel instance to convert.
    method_key: Optional key to select a specific function if model.apply_fn is
      a Mapping. Required if model.apply_fn is a Mapping.
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

  args_specs = utils.to_jax_spec((model.params, input_signature))

  model_shlo_function = jax2npt.convert(
      fun_jax=apply_fn,
      args_spec=args_specs,
      kwargs_spec={},  # kwargs left empty for now
  )
  save_bundle.manifest_objects[serving_signature] = model_shlo_function

  # TODO: Update the input signature to exclude the weights.
  return (
      model_shlo_function.input_signature,
      model_shlo_function.output_signature,
  )


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

  for node in pipeline:
    if node.role == orchestration_pb2.ROLE_MODEL:
      _, _ = _update_save_bundle_for_jax_model(
          input_signature=cfg.input_signature,
          model=node.node,
          method_key=cfg.method_key,
          serving_signature=cfg.serving_signature,
          save_bundle=save_bundle,
      )
    else:
      raise ValueError(f'Unsupported role: {node.role}')


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
      'Saving model=%s to export_path=%s with serving_configs=%s',
      model,
      export_path,
      serving_configs,
  )

  save_bundle = NeptuneModelSaveBundle()
  for cfg in serving_configs:
    _update_save_bundle(
        pipeline=[PipelineNode(node=model, role=orchestration_pb2.ROLE_MODEL)],
        cfg=cfg,
        save_bundle=save_bundle,
    )
  persistence_lib.save(
      module=save_bundle.manifest_objects,
      target_dir=export_path,
      options=persistence_lib.SaveOptions(version=2),
  )
