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

"""Library for exporting JAX models to Neptune model format.

This library provides the `save` function to convert and save JAX models
according to the Neptune Model Specification. It uses `jax2npt` to convert
JAX functions to SHLO functions, and `persistence_lib` to save them
in Neptune's format.
"""

from collections.abc import Sequence
import dataclasses
import pathlib

from absl import logging
import jax
import jaxtyping
import orbax.checkpoint as ocp

from neptune_model._src import jax2npt
from neptune_model._src import tf2npt
from neptune_model._src.core import orchestration_proto_utils
from neptune_model._src.core import persistence_lib
from neptune_model._src.core import saveable
from neptune_model._src.core import shlo_type
from neptune_model._src.core import type_proto_utils
from neptune_model._src.export import constants
from neptune_model._src.export import jax_model
from neptune_model._src.export import serving_config
from neptune_model._src.export import utils
from neptune_model._src.processors import jax_processor
from neptune_model._src.processors import processor
from neptune_model._src.processors import tf_processor
from neptune_model.protos import orchestration_pb2


@dataclasses.dataclass
class NeptuneModelSaveBundle:
  """A bundle of objects to be saved in the Neptune model format.

  Attributes:
    manifest_objects: A dictionary of saveable objects to be saved in the
      manifest.
    model_params: The model parameters to be saved in the checkpoint.
    pipelines: The pipeline definitions for model orchestration.
    tf_prepared_functions: Tracks all TF processor prepared functions. During
      export, TF functions will be saved as TF SavedModel format in a
      subdirectory of the export path and added to the exported model's
      supplementals.
  """

  manifest_objects: dict[str, saveable.Saveable] = dataclasses.field(
      default_factory=dict
  )
  model_params: jaxtyping.PyTree[shlo_type.ShloTensorSpec] | None = None
  pipelines: orchestration_pb2.Pipelines = dataclasses.field(
      default_factory=orchestration_pb2.Pipelines
  )
  tf_prepared_functions: dict[str, tf2npt.PreparedFunction] = dataclasses.field(
      default_factory=dict
  )


@dataclasses.dataclass
class PipelineNode:
  """A node in the model execution pipeline.

  Attributes:
    processor: The processor to execute.
    role: The role of the component in the pipeline.
  """

  processor: processor.Processor
  role: orchestration_pb2.Role


def _build_pipeline_nodes(
    model: jax_model.JaxModel,
    cfg: serving_config.ServingConfig,
) -> Sequence[PipelineNode]:
  """Builds a sequence of PipelineNodes for the given model and serving config.

  The pipeline nodes are created based on the processing configuration in the
  serving config.

  Args:
    model: The JAX model being exported.
    cfg: The serving configuration for this model.

  Returns:
    A sequence of PipelineNodes. The ordering of the nodes is going to be
    preserved in the exported model.

  Raises:
    ValueError: If processing configuration is inconsistent with serving config
    batch options.
  """

  jax_processor_inst = jax_processor.JaxProcessor(
      model=model,
      name=cfg.serving_signature,
      method_key=cfg.method_key,
  )

  model_node = PipelineNode(
      processor=jax_processor_inst,
      role=orchestration_pb2.ROLE_MODEL,
  )

  match cfg.processing:
    case processor.SequentialProcessing() as sequential_processing:
      pre_nodes = [
          PipelineNode(p, orchestration_pb2.ROLE_PREPROCESSOR)
          for p in sequential_processing.preprocessors
      ]
      post_nodes = [
          PipelineNode(p, orchestration_pb2.ROLE_POSTPROCESSOR)
          for p in sequential_processing.postprocessors
      ]
      if cfg.batch_options is not None:
        if (
            cfg.batch_options.batch_component
            == orchestration_pb2.BatchOptions.PRE_PROCESSOR_AND_MODEL_FUNCTION
            and not pre_nodes
        ):
          raise ValueError(
              'Batch component is set to PRE_PROCESSOR_AND_MODEL_FUNCTION, but'
              ' there are no preprocessors.'
          )
        if (
            cfg.batch_options.batch_component
            == orchestration_pb2.BatchOptions.MODEL_FUNCTION_AND_POST_PROCESSOR
            and not post_nodes
        ):
          raise ValueError(
              'Batch component is set to POST_PROCESSOR_AND_MODEL_FUNCTION, but'
              ' there are no postprocessors.'
          )
      return [*pre_nodes, model_node, *post_nodes]
    case None:
      return [model_node]
    case _:
      raise ValueError(f'Unknown processing type: {type(cfg.processing)}')


def _update_save_bundle_with_pipeline_node(
    serving_signature: str,
    pipeline_node: PipelineNode,
    input_signature: jaxtyping.PyTree,
    save_bundle: NeptuneModelSaveBundle,
) -> tuple[jaxtyping.PyTree, jaxtyping.PyTree]:
  """Processes the pipeline node.

  Args:
    serving_signature: The serving signature of the pipeline.
    pipeline_node: The pipeline node to process.
    input_signature: The input signature of the pipeline node.
    save_bundle: The NeptuneModelSaveBundle to update.

  Returns:
    A tuple (input_signature, output_signature), where
    input_signature is the input signature of the pipeline node, and
    output_signature is the output signature of the pipeline node.
  """

  suppress_x64_outputs = (
      pipeline_node.role == orchestration_pb2.ROLE_PREPROCESSOR
      and not jax.config.jax_enable_x64
  )
  prepared_processor = pipeline_node.processor.prepare(
      input_signature,
      suppress_x64_outputs=suppress_x64_outputs,
  )
  save_bundle.manifest_objects[prepared_processor.name] = (
      prepared_processor.to_npt_saveable()
  )

  if isinstance(prepared_processor, tf_processor.PreparedTfProcessor):
    save_bundle.tf_prepared_functions[prepared_processor.name] = (
        prepared_processor.prepared_function
    )

  save_bundle.pipelines.name_to_pipeline[serving_signature].components.append(
      orchestration_pb2.Component(
          function_name=prepared_processor.name,
          role=pipeline_node.role,
          weights_name=prepared_processor.weights_name,
      )
  )

  return (
      prepared_processor.input_signature,
      prepared_processor.output_signature,
  )


def _update_save_bundle(
    pipeline_nodes: Sequence[PipelineNode],
    cfg: serving_config.ServingConfig,
    save_bundle: NeptuneModelSaveBundle,
) -> None:
  """Processes a pipeline of nodes and updates the save bundle.

  Args:
    pipeline_nodes: A sequence of PipelineNodes to process.
    cfg: The serving configuration to use for conversion.
    save_bundle: The NeptuneModelSaveBundle to update.
  """

  pipeline_input_signature: jaxtyping.PyTree | None = None
  pipeline_output_signature: jaxtyping.PyTree | None = None
  input_signature: jaxtyping.PyTree = utils.to_shlo_tensor_spec(
      cfg.input_signature
  )

  for node in pipeline_nodes:
    node_input_signature, node_output_signature = (
        _update_save_bundle_with_pipeline_node(
            cfg.serving_signature,
            node,
            input_signature,
            save_bundle,
        )
    )
    pipeline_input_signature = pipeline_input_signature or node_input_signature
    pipeline_output_signature = node_output_signature
    input_signature = node_output_signature

  pipeline_proto = save_bundle.pipelines.name_to_pipeline[cfg.serving_signature]
  pipeline_proto.signature.CopyFrom(
      type_proto_utils.to_function_signature_proto(
          pipeline_input_signature,
          pipeline_output_signature,
      )
  )
  pipeline_proto.named_signature.CopyFrom(
      orchestration_proto_utils.to_named_signature_proto(
          pipeline_input_signature,
          pipeline_output_signature,
      ),
  )
  if cfg.batch_options is not None:
    pipeline_proto.batch_options.CopyFrom(cfg.batch_options)


def _validate_save_params(
    model: jax_model.JaxModel,
    serving_configs: Sequence[serving_config.ServingConfig],
) -> None:
  """Validates the save parameters for the given model and serving configs.

  Should be used for validations that cross multiple serving configs and the
  JAX model.

  Args:
    model: The JAX model being exported.
    serving_configs: The serving configurations to validate.

  Raises:
    ValueError: If save params are not valid.
  """
  serving_signatures = {cfg.serving_signature for cfg in serving_configs}
  if len(serving_signatures) != len(serving_configs):
    raise ValueError(
        'Serving signatures must be unique across all serving configs'
    )

  for cfg in serving_configs:
    if not cfg.input_signature:
      raise ValueError(
          f'Serving config {cfg.serving_signature!r} must have input signature'
      )
    if cfg.method_key is None and len(model.method_keys()) != 1:
      raise ValueError(
          f'Serving config {cfg.serving_signature!r} does not specify a'
          ' method key, but the JAX model has more than one apply_fn:'
          f' {list(model.method_keys())}. Please specify a method key'
          ' explicitly.'
      )
    if cfg.method_key is not None and cfg.method_key not in model.method_keys():
      raise ValueError(
          f'Serving config {cfg.serving_signature!r} method key'
          f' `{cfg.method_key!r}` does not exist in the JAX model'
      )


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
  _validate_save_params(model, serving_configs)
  save_bundle = NeptuneModelSaveBundle()
  for cfg in serving_configs:
    _update_save_bundle(
        pipeline_nodes=_build_pipeline_nodes(model, cfg),
        cfg=cfg,
        save_bundle=save_bundle,
    )

  supplementals: dict[str, persistence_lib.GlobalSupplemental] = {
      constants.ORCHESTRATION_SUPPLEMENTAL_NAME: (
          persistence_lib.GlobalSupplemental(
              orchestration_proto_utils.serialize_pipelines_proto(
                  save_bundle.pipelines
              ),
              constants.DEFAULT_ORCHESTRATION_SUPPLEMENTAL_FILENAME,
          )
      ),
  }
  if save_bundle.tf_prepared_functions:
    trackable_resources = [
        cfg.processing.options.tf_trackable_resources
        for cfg in serving_configs
        if cfg.processing is not None
        and cfg.processing.options.tf_trackable_resources is not None
    ]

    tf_supplemental = tf2npt.save_tf_functions(
        export_path,
        save_bundle.tf_prepared_functions,
        trackable_resources=trackable_resources,
    )
    supplementals.update(tf_supplemental)

  checkpoint_dir = (
      pathlib.Path(export_path) / constants.DEFAULT_CHECKPOINT_SUBDIR
  )
  checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
  checkpointer.save(
      checkpoint_dir,
      model.params,
      force=True,
  )
  save_module = {
      constants.WEIGHTS_NAME: jax2npt.convert_path_to_value(
          constants.DEFAULT_CHECKPOINT_SUBDIR,
          constants.ORBAX_CHECKPOINT_MIME_TYPE,
      ),
      **save_bundle.manifest_objects,
  }

  save_options = persistence_lib.SaveOptions(
      version=2,
      supplementals=supplementals,
  )

  persistence_lib.save(
      module=save_module,
      target_dir=export_path,
      options=save_options,
  )
