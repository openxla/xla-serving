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

from collections.abc import Callable, Mapping, MutableSequence, Sequence
import dataclasses
from typing import Any

from absl import logging
import jax
import jaxtyping

from neptune_model._src.core import orchestration_proto_utils
from neptune_model._src.core import persistence_lib
from neptune_model._src.core import saveable
from neptune_model._src.core import shlo_type
from neptune_model._src.core import type_proto_utils
from neptune_model._src.core import value
from neptune_model._src.export import constants
from neptune_model._src.export import jax_model
from neptune_model._src.export import serving_config
from neptune_model._src.export import tensorflow_model
from neptune_model._src.export import warmup
from neptune_model._src.jax2npt import utils as jax2npt_utils
from neptune_model._src.processors import jax_processor
from neptune_model._src.processors import processor
from neptune_model._src.processors import tf_processor
from neptune_model._src.processors.tf_processor import save_tf_functions
from neptune_model.protos import orchestration_pb2

try:
  from neptune_model._src.processors import composite_processor
  CompositeModel = composite_processor.CompositeProcessor
except ImportError:
  composite_processor = None
  CompositeModel = None
Model = Any


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
    on_persist_fns: A list of callbacks to run prior to persisting the model.
  """

  manifest_objects: dict[str, saveable.Saveable] = dataclasses.field(
      default_factory=dict
  )
  model_params: jaxtyping.PyTree[shlo_type.ShloTensorSpec] | None = None
  pipelines: orchestration_pb2.Pipelines = dataclasses.field(
      default_factory=orchestration_pb2.Pipelines
  )
  tf_prepared_functions: dict[str, Any] = dataclasses.field(
      default_factory=dict
  )
  on_persist_fns: MutableSequence[
      Callable[
          [str],
          Mapping[
              str, value.ExternalValue | persistence_lib.GlobalSupplemental
          ],
      ]
  ] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PipelineNode:
  """A node in the model execution pipeline.

  Attributes:
    processor: The processor to execute.
    role: The role of the component in the pipeline.
    input_keys: The keys of the input dict that the function expects.
    output_keys: The keys of the output dict that the function produces.
  """

  processor: processor.Processor
  role: orchestration_pb2.Role
  input_keys: Sequence[str] = dataclasses.field(default_factory=list)
  output_keys: Sequence[str] = dataclasses.field(default_factory=list)


def _build_orchestration_pipeline_nodes(
    models: Sequence[Model],
    serving_signature: str,
    orchestration: serving_config.Orchestration,
) -> Sequence[PipelineNode]:
  """Builds pipeline nodes based on orchestration."""
  ordered_model_calls = orchestration.ordered_model_calls

  pipeline_nodes = []
  for model_call in ordered_model_calls:
    for m in models:
      if hasattr(m, 'name') and m.name == model_call.model_name:
        model = m
        break
    else:
      raise ValueError(
          f'Model {model_call.model_name!r} not found in provided models'
      )
    if model_call.method_key not in model.method_keys():
      raise ValueError(
          f'Method {model_call.method_key!r} not found in model {model.name!r}'
      )

    # Construct a unique processor name by combining the serving signature,
    # model name, and method key (e.g., "serving_default_tf_model_preprocessing").
    # This prevents identifier collisions when multiple serving configs call
    # the same model methods, or when methods with identical names are used
    # across different models.
    processor_name = f'{serving_signature}_{model.name}_{model_call.method_key}'

    match model:
      case jax_model.JaxModel():
        processor_inst = jax_processor.JaxProcessor(
            name=processor_name,
            apply_fn_bundle=model.apply_fn_bundles[model_call.method_key],
            params=model.params,
            weights_name=model.weights_name,
            jax_mesh=model.jax_mesh,
        )
      case tensorflow_model.TensorflowModel():
        fn = model.tf_functions[model_call.method_key]
        processor_inst = tf_processor.TfProcessor(
            name=processor_name,
            fn=fn,
        )
      case _ if CompositeModel is not None and isinstance(
          model, CompositeModel
      ):
        processor_inst = model
      case _:
        raise ValueError(f'Unknown model type: {type(model)}')

    pipeline_nodes.append(
        PipelineNode(
            processor=processor_inst,
            role=model_call.role,
            input_keys=model_call.input_keys,
            output_keys=model_call.output_keys,
        )
    )

  return pipeline_nodes


def _build_pipeline_nodes(
    models: Sequence[Model],
    cfg: serving_config.ServingConfig,
) -> Sequence[PipelineNode]:
  """Builds a sequence of PipelineNodes for the given models and serving config.

  The pipeline nodes are created based on the orchestration and processing
  configurations in the serving config.

  Args:
    models: A sequence of models available for the pipeline.
    cfg: The serving configuration for this pipeline.

  Returns:
    A sequence of PipelineNodes. The ordering of the nodes is going to be
    preserved in the exported model.

  Raises:
    ValueError: If a model/method is not found or batch options are invalid.
  """
  if cfg.orchestration:
    return _build_orchestration_pipeline_nodes(
        models, cfg.serving_signature, cfg.orchestration
    )

  # Without an orchestration workflow, export defaults to the single
  # provided JAX model.
  model = models[0]
  if not isinstance(model, jax_model.JaxModel):
    raise ValueError(f'Expected JaxModel, got {type(model)}')
  if len(model.apply_fn_bundles) > 1:
    raise ValueError(
        f'Model {model.name!r} has multiple apply functions. You must specify'
        ' an orchestration workflow in ServingConfig to select which function'
        ' to execute.'
    )

  return [
      PipelineNode(
          processor=jax_processor.JaxProcessor(
              name=cfg.serving_signature,
              apply_fn_bundle=next(iter(model.apply_fn_bundles.values())),
              params=model.params,
              weights_name=constants.WEIGHTS_NAME,
              jax_mesh=model.jax_mesh,
          ),
          role=orchestration_pb2.ROLE_MODEL,
      )
  ]


def _update_save_bundle_with_pipeline_node(
    serving_signature: str,
    pipeline_node: PipelineNode,
    input_signature: jaxtyping.PyTree,
    save_bundle: NeptuneModelSaveBundle,
    is_graph_based: bool = False,
) -> tuple[jaxtyping.PyTree, jaxtyping.PyTree]:
  """Processes the pipeline node.

  Args:
    serving_signature: The serving signature of the pipeline.
    pipeline_node: The pipeline node to process.
    input_signature: The input signature of the pipeline node.
    save_bundle: The NeptuneModelSaveBundle to update.
    is_graph_based: Whether the pipeline is graph based.

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

  on_persist_fn = prepared_processor.on_persist()
  if on_persist_fn is not None:
    save_bundle.on_persist_fns.append(on_persist_fn)

  if isinstance(prepared_processor, tf_processor.PreparedTfProcessor):
    save_bundle.tf_prepared_functions[prepared_processor.name] = (
        prepared_processor.prepared_function
    )

  component = orchestration_pb2.Component(
      function_name=prepared_processor.name,
      role=pipeline_node.role,
      weights_name=prepared_processor.weights_name,
  )
  if is_graph_based:
    shlo_input_sig = jax2npt_utils.to_shlo_tensor_spec(
        prepared_processor.input_signature
    )
    shlo_output_sig = jax2npt_utils.to_shlo_tensor_spec(
        prepared_processor.output_signature
    )
    component.named_signature.CopyFrom(
        orchestration_proto_utils.to_named_signature_proto(
            shlo_input_sig,
            shlo_output_sig,
        )
    )

  save_bundle.pipelines.name_to_pipeline[serving_signature].components.append(
      component
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

  is_graph_based = bool(cfg.orchestration and cfg.orchestration.graph_based)
  pipeline_input_signature: jaxtyping.PyTree | None = None
  pipeline_output_signature: jaxtyping.PyTree | None = None

  if is_graph_based:
    assert isinstance(cfg.input_signature, Mapping)
    available_tensor_specs = dict(cfg.input_signature)
    pipeline_input_signature = cfg.input_signature

    for node in pipeline_nodes:
      for k in node.input_keys:
        if k not in available_tensor_specs:
          node_name = getattr(node.processor, 'name', str(node.processor))
          raise ValueError(
              f'Input key {k!r} for node {node_name!r}'
              ' not found in available tensor specs:'
              f' {list(available_tensor_specs.keys())}'
          )
      node_input_signature = {
          k: available_tensor_specs[k] for k in node.input_keys
      }

      _, node_output_signature = _update_save_bundle_with_pipeline_node(
          cfg.serving_signature,
          node,
          node_input_signature,
          save_bundle,
          is_graph_based=True,
      )
      if isinstance(node_output_signature, Mapping):
        available_tensor_specs.update(node_output_signature)
      elif node.output_keys and len(node.output_keys) == 1:
        available_tensor_specs[node.output_keys[0]] = node_output_signature
      else:
        node_name = getattr(node.processor, 'name', str(node.processor))
        raise ValueError(
            f'Output signature of node {node_name!r} must be a Mapping in'
            f' graph-based orchestration. Got: {type(node_output_signature)}'
        )
      pipeline_output_signature = node_output_signature
  else:
    input_signature: jaxtyping.PyTree = cfg.input_signature
    for node in pipeline_nodes:
      node_input_signature, node_output_signature = (
          _update_save_bundle_with_pipeline_node(
              cfg.serving_signature,
              node,
              input_signature,
              save_bundle,
              is_graph_based=False,
          )
      )
      pipeline_input_signature = (
          pipeline_input_signature or node_input_signature
      )
      pipeline_output_signature = node_output_signature
      input_signature = node_output_signature

  shlo_input_signature = jax2npt_utils.to_shlo_tensor_spec(
      pipeline_input_signature
  )
  shlo_output_signature = jax2npt_utils.to_shlo_tensor_spec(
      pipeline_output_signature
  )

  pipeline_proto = save_bundle.pipelines.name_to_pipeline[cfg.serving_signature]
  pipeline_proto.keyword_arguments_based = is_graph_based

  pipeline_proto.signature.CopyFrom(
      type_proto_utils.to_function_signature_proto(
          shlo_input_signature,
          shlo_output_signature,
      )
  )
  pipeline_proto.named_signature.CopyFrom(
      orchestration_proto_utils.to_named_signature_proto(
          shlo_input_signature,
          shlo_output_signature,
      ),
  )
  if cfg.batch_options is not None:
    pipeline_proto.batch_options.CopyFrom(cfg.batch_options)
    ragged_indices = tf_processor.find_ragged_inputs_indices_if_any(
        cfg.input_signature
    )
    if ragged_indices:
      for val_idx, split_idx in ragged_indices:
        pipeline_proto.metadata_for_batching.ragged_inputs.append(
            orchestration_pb2.RaggedInputSpec(
                flat_values_index=val_idx, row_splits_index=split_idx
            )
        )


def _validate_save_params(
    models: Sequence[Model],
    serving_configs: Sequence[serving_config.ServingConfig],
) -> None:
  """Validates the save parameters for the given models and serving configs.

  Should be used for validations that cross multiple models/serving configs.

  Args:
    models: The models to validate.
    serving_configs: The serving configurations to validate.

  Raises:
    ValueError: If save params are not valid.
  """
  if not models:
    raise ValueError('Must specify at least one model.')

  tf_model_count = sum(
      isinstance(m, tensorflow_model.TensorflowModel) for m in models
  )
  if tf_model_count > 1:
    raise ValueError(
        'Multiple TensorflowModels are not supported. Got:'
        f' {tf_model_count} TensorflowModels.'
    )

  model_names = [m.name for m in models if hasattr(m, 'name')]
  if len(set(model_names)) != len(model_names):
    raise ValueError(
        'Model names must be unique across all provided models. Got:'
        f' {model_names}'
    )

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
    if len(models) > 1 and cfg.orchestration is None:
      raise ValueError(
          'When multiple models are provided, orchestration must be specified'
          f' in ServingConfig {cfg.serving_signature!r}.'
      )


def save(
    models: Sequence[Model] | Model,
    export_path: str = '',
    serving_configs: Sequence[serving_config.ServingConfig] = (),
    *,
    warmup_examples: Mapping[str, Sequence[Any]] | None = None,
    save_shlo_to_file: bool = False,
) -> None:
  """Saves model(s) to neptune model format.

  Args:
    models: The model or sequence of models to save.
    export_path: The directory to which to save the model.
    serving_configs: The serving configurations.
    warmup_examples: Optional mapping of serving signatures to a sequence of
      warmup examples. Each example is a dictionary mapping input names to
      tensors, NumPy arrays, or any object convertible to a NumPy array.
    save_shlo_to_file: Optional. If true, save the SHLO in a separate file
      instead of inlining it in the manifest.pb.
  """
  if not isinstance(models, Sequence):
    models = [models]

  logging.info(
      'Saving model=%s to export_path=%s with serving_configs=%s',
      models,
      export_path,
      serving_configs,
  )
  _validate_save_params(models, serving_configs)

  save_bundle = NeptuneModelSaveBundle()
  for cfg in serving_configs:
    _update_save_bundle(
        pipeline_nodes=_build_pipeline_nodes(models, cfg),
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
    trackable_resources = []
    for model_inst in models:
      if (
          isinstance(model_inst, tensorflow_model.TensorflowModel)
          and model_inst.trackable_resources is not None
      ):
        if isinstance(model_inst.trackable_resources, (list, tuple)):
          trackable_resources.extend(model_inst.trackable_resources)
        else:
          trackable_resources.append(model_inst.trackable_resources)

    tf_supplemental = save_tf_functions(
        export_path,
        save_bundle.tf_prepared_functions,
        trackable_resources=trackable_resources,
    )
    supplementals.update(tf_supplemental)

  save_module = save_bundle.manifest_objects.copy()

  is_single_model = len(models) == 1
  for model_inst in models:
    if isinstance(model_inst, jax_model.JaxModel):
      if is_single_model:
        ckpt_path = constants.DEFAULT_CHECKPOINT_SUBDIR
        w_name = constants.WEIGHTS_NAME
      else:
        ckpt_path = f'{constants.DEFAULT_CHECKPOINT_SUBDIR}/{model_inst.name}'
        w_name = model_inst.weights_name
      save_module.update(
          model_inst.save_checkpoint(
              export_path,
              checkpoint_path=ckpt_path,
              weights_name=w_name,
          )
      )

  for on_persist_fn in save_bundle.on_persist_fns:
    on_persist_result = on_persist_fn(export_path)
    for k, v in on_persist_result.items():
      match v:
        case persistence_lib.GlobalSupplemental():
          if k in supplementals:
            raise ValueError(f'Duplicate global supplemental {k!r} detected')
          supplementals[k] = v
        case value.ExternalValue():
          if k in save_module:
            raise ValueError(f'Duplicate saveable object {k!r} detected')
          save_module[k] = v

  save_options = persistence_lib.SaveOptions(
      version=2,
      supplementals=supplementals,
      save_shlo_to_file=save_shlo_to_file,
  )

  persistence_lib.save(
      module=save_module,
      target_dir=export_path,
      options=save_options,
  )

  if warmup_examples is not None:
    serving_signatures = {cfg.serving_signature for cfg in serving_configs}
    for sig in warmup_examples:
      if sig not in serving_signatures:
        raise ValueError(
            f'Warmup example signature {sig!r} does not match any serving'
            f' config signature: {list(serving_signatures)}'
        )
    warmup.write_warmup_file(warmup_examples, export_path)
