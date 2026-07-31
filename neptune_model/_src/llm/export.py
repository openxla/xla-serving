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

import pathlib

import jax
import jaxtyping

from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.export import tensorflow_model
from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm import model_wrapper
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
    tf_model: tensorflow_model.TensorflowModel | None = None,
) -> None:
  """Exports a JAX LLM to the Neptune Model format using a 3-stage pipeline."""
  tf = _require_tf()

  # 1. Setup export directory
  pathlib.Path(config.export_dir).mkdir(parents=True, exist_ok=True)

  # 2. Wrap the model implementation inside the JAX generation loop
  jax_gen_fn = model.compile_generation_loop()

  # 3. Setup Serving default signature and orchestration pipelines
  models = []
  model_calls = []

  if tf_model is not None:
    models.append(tf_model)
    if "preprocessing" in tf_model.tf_functions:
      model_calls.append(
          serving_config.ModelCall(
              model_name=tf_model.name,
              method_key="preprocessing",
              role=orchestration_pb2.ROLE_PREPROCESSOR,
          )
      )

  model_calls.append(
      serving_config.ModelCall(
          model_name="jax_model",
          method_key="serving_default",
          role=orchestration_pb2.ROLE_MODEL,
      )
  )

  if tf_model is not None and "postprocessing" in tf_model.tf_functions:
    model_calls.append(
        serving_config.ModelCall(
            model_name=tf_model.name,
            method_key="postprocessing",
            role=orchestration_pb2.ROLE_POSTPROCESSOR,
        )
    )

  if tf_model is not None:
    orchestration = serving_config.Orchestration(model_calls=model_calls)
    input_signature = {
        "text_input": tf.TensorSpec(shape=(None,), dtype=tf.string)
    }
  else:
    orchestration = None
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
      orchestration=orchestration,
      batch_options=batch_options,
  )

  # 5. Compile JAX graph with polymorphic shapes mapped to batch size buckets
  symbol_values = {"b": list(config.allowed_batch_sizes)}
  if (
      config.generation_config.max_prompt_length is not None
      or tf_model is not None
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
      name="jax_model",
  )
  models.append(npt_jax_model)

  # 6. Save model bundle
  main_lib.save(
      models,
      export_path=config.export_dir,
      serving_configs=[serving_cfg],
  )
