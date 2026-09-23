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

"""Export orchestration for tpu-inference models to XSE."""

from typing import Any

import jax
import jaxtyping

from xla_saved_model._src.llm.tokenizers import exporter
from xla_saved_model._src.llm.tpu_inference import model_wrapper as tpu_inference_wrapper


def export_tpu_inference_model(
    tpu_runner: Any,
    config: exporter.XLASavedModelLLMExportWithTokenizerConfig,
    model_params: jaxtyping.PyTree | None = None,
    jax_mesh: jax.sharding.Mesh | None = None,
) -> None:
  """Exports a tpu-inference model to XSE serving using consolidated export config.

  Wraps a TPUModelRunner in TPUInferenceWrapper, binds TF tokenization
  preprocessing and postprocessing stages, and builds a complete SavedModel
  bundle tailored for on-device TPU serving.

  Args:
    tpu_runner: TPUModelRunner instance loaded with model weights and mesh
      config.
    config: Consolidated configuration containing export directory, batch sizes,
      generation parameters (sampling method, max tokens), and tokenizer setup.
    model_params: Optional explicit JAX model parameters. Defaults to
      runner.state.
    jax_mesh: Optional explicit JAX sharding mesh. Defaults to runner.mesh.

  Raises:
    ImportError: If the 'tpu_inference' package is not available.
  """
  model_wrapper = tpu_inference_wrapper.TPUInferenceWrapper(
      tpu_runner=tpu_runner,
      generation_config=config.generation_config,
  )

  exporter.export_llm_with_tokenizer(
      config=config,
      model=model_wrapper,
      params=model_params or tpu_runner.state,
      jax_mesh=jax_mesh or tpu_runner.mesh,
  )
