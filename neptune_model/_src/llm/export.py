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

"""E2E Orchestrator for exporting LLMs with Neptune."""

from collections.abc import Sequence
from typing import Any

import jax
import jaxtyping

from neptune_model._src.export import jax_model

from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import model_wrapper
from neptune_model._src.processors import processor
from neptune_model._src.processors import tf_processor
from neptune_model.protos import orchestration_pb2


def _require_tf():
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        "Failed to import tensorflow. Please make sure that TF support is"
        " enabled."
    ) from e
  return tf


def export_llm(
    config: llm_config.NeptuneLLMExportConfig,
    model: model_wrapper.LLMModelWrapper,
    params: jaxtyping.PyTree,
    jax_mesh: jax.sharding.Mesh | None = None,
    pre_processor: tf_processor.TfProcessor | None = None,
    post_processor: tf_processor.TfProcessor | None = None,
    tf_trackable_resources: Sequence[Any] | None = None,
) -> None:
  """Exports a JAX LLM to the Neptune Model format using a 3-stage pipeline.

  This function wraps the user's forward pass into a JAX-native generation loop,
  appends TF-based preprocessing/postprocessing tokenizer steps, compiles the
  pipeline for the specified batch size buckets, and saves the final SavedModel.

  Args:
    config: Consolidated configuration for LLM export.
    model: An implementation of LLMModelWrapper.
    params: Model parameters PyTree.
    jax_mesh: Optional JAX mesh configuration for TPU sharding.
    pre_processor: Optional custom TF pre-processor.
    post_processor: Optional custom TF post-processor.
    tf_trackable_resources: Optional TF trackable resources for the processors.
  """
  tf = _require_tf()
  preprocessors = []

  if pre_processor is not None:
    preprocessors.append(pre_processor)

  postprocessors = []
  if post_processor is not None:
    postprocessors.append(post_processor)

  tf_resources = list(tf_trackable_resources or [])

  # 2. Wrap the model implementation inside the JAX generation loop
  jax_gen_fn = model.compile_generation_loop()

  # 3. Setup Serving default signature and processing pipelines
  if preprocessors or postprocessors:
    processing = processor.SequentialProcessing(
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        options=processor.ProcessingOptions(
            tf_trackable_resources=tf_resources
        ),
    )
    input_signature = {
        "text_input": tf.TensorSpec(shape=(None,), dtype=tf.string)
    }
  else:
    processing = None
    prompt_dim = config.generation_config.max_prompt_length
    input_signature = {
        "input_ids": tf.TensorSpec(shape=(None, prompt_dim), dtype=tf.int32),
    }

  batch_options = orchestration_pb2.BatchOptions(
      allowed_batch_sizes=list(config.allowed_batch_sizes),
      batch_component=orchestration_pb2.BatchOptions.WHOLE_PIPELINE,
      max_enqueued_batches=10,
  )

  serving_cfg = serving_config.ServingConfig(
      serving_signature="serving_default",
      input_signature=input_signature,
      processing=processing,
      batch_options=batch_options,
  )

  # 5. Compile JAX graph with polymorphic shapes mapped to batch size buckets
  symbol_values = {"b": list(config.allowed_batch_sizes)}
  if (
      config.generation_config.max_prompt_length is not None
      or pre_processor is not None
  ):
    input_ids_shape = "b, _"
  else:
    input_ids_shape = "b, s"
  model_input_polymorphic_shapes = {
      "input_ids": input_ids_shape,
  }

  jax_export_kwargs = {"platforms": ["tpu"]}

  apply_fn_bundle = jax_model.ApplyFnBundle(
      apply_fn=jax_gen_fn,
      input_polymorphic_shapes=model_input_polymorphic_shapes,
      input_polymorphic_shape_symbol_values=symbol_values,
      jax_export_kwargs=jax_export_kwargs,
  )

  npt_jax_model = jax_model.JaxModel(
      apply_fn_bundle=apply_fn_bundle,
      params=params,
      jax_mesh=jax_mesh,
  )

  # 6. Save model bundle
  main_lib.save(
      models=npt_jax_model,
      export_path=config.export_dir,
      serving_configs=[serving_cfg],
  )
