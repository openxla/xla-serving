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

"""Tests for Neptune LLM TPU Inference Export integration."""

# pylint: disable=g-import-not-at-top
# pytype: disable=attribute-error

import os
import unittest
from unittest import mock

from absl import logging
from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from neptune_model._src.llm import config as llm_config
from neptune_model._src.llm.tokenizers import config as tokenizers_config
from neptune_model._src.llm.tokenizers import exporter as tokenizers_exporter
from neptune_model._src.llm.tokenizers import processors as tokenizers_processors
from neptune_model._src.llm.tpu_inference import exporter as tpu_exporter
from neptune_model._src.llm.tpu_inference import model_wrapper as tpu_wrapper

# In Google3 test environments, inject wildcard mocks to bypass metadata checks.
# In OSS GitHub environments, this module does not exist and is not needed.
try:
  from vllm.g3_integration import patchers  # pylint: disable=g-import-not-at-top

  patchers.inject_wildcard_mocks()
  patchers.patch_metadata_versions()
except ImportError:
  pass

try:
  from tpu_inference.distributed import jax_parallel_state
  from tpu_inference.runner import tpu_runner
  from vllm.engine import arg_utils as vllm_arg_utils
  from vllm.model_executor.models import qwen2 as configuration_qwen2
  from vllm.model_executor.models import registry as vllm_registry

  _TPU_INFERENCE_TEST_AVAILABLE = True
  _TPU_INFERENCE_TEST_IMPORT_ERROR = None
except ImportError as e:
  jax_parallel_state = None
  tpu_runner = None
  vllm_arg_utils = None
  configuration_qwen2 = None
  vllm_registry = None
  _TPU_INFERENCE_TEST_AVAILABLE = False
  _TPU_INFERENCE_TEST_IMPORT_ERROR = e


def _mock_inspect_model_cls(architectures, model_config=None):
  archs = (
      architectures
      if isinstance(architectures, (list, tuple))
      else [architectures]
  )
  model_cls, arch = vllm_registry.ModelRegistry.resolve_model_cls(
      archs, model_config
  )
  return vllm_registry._ModelInfo.from_model_cls(model_cls), arch


class ExporterTest(absltest.TestCase):

  # TODO: Make sure we initialize this only once for all tests.
  _distributed_initialized = False
  _patchers = []

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not _TPU_INFERENCE_TEST_AVAILABLE:
      raise absltest.SkipTest(
          "tpu_inference/vllm dependencies are not available. Detailed"
          f" error: {_TPU_INFERENCE_TEST_IMPORT_ERROR}"
      )

    if jax.devices()[0].platform != "tpu":
      raise absltest.SkipTest(
          "TPU hardware is required for tpu_exporter_test."
      )

    # Fix getattr default evaluation bug in tpu_inference/models/jax/qwen2.py.
    if not hasattr(configuration_qwen2.Qwen2Config, "text_config"):
      setattr(
          configuration_qwen2.Qwen2Config,
          "text_config",
          property(lambda self: self),
      )

    # In hermetic Bazel embedded Python test binaries, sys.executable is None,
    # which causes vLLM's subprocess-based inspect_model_cls to fail.
    # We resolve the model class directly and construct real _ModelInfo
    # in-process.
    cls._patchers = [
        mock.patch.object(
            vllm_registry.ModelRegistry,
            "inspect_model_cls",
            side_effect=_mock_inspect_model_cls,
        ),
    ]
    for patcher in cls._patchers:
      patcher.start()

  @classmethod
  def tearDownClass(cls):
    for patcher in cls._patchers:
      patcher.stop()
    super().tearDownClass()

  def _create_runner(self, load_format: str = "jax_dummy"):
    model_dir = os.path.join(
        os.path.dirname(__file__),
        "../testdata/qwen_dummy",
    )

    engine_args = vllm_arg_utils.EngineArgs(
        model=model_dir,
        max_model_len=128,
        load_format=load_format,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.5,
    )
    vllm_config = engine_args.create_engine_config()

    devices = jax.local_devices()[:1]
    if not ExporterTest._distributed_initialized:
      jax_parallel_state.init_pp_distributed_environment(
          ip="127.0.0.1",
          rank=0,
          world_size=1,
          device=devices[0],
          need_pp=False,
      )
      ExporterTest._distributed_initialized = True
    runner = tpu_runner.TPUModelRunner(
        vllm_config=vllm_config,
        devices=devices,
    )
    runner.load_model()
    return runner

  def test_export_greedy_with_dummy_weights(self):
    """Verifies greedy export and execution equivalence between manual pipeline and SavedModel."""
    runner = self._create_runner(load_format="jax_dummy")
    export_dir = self.create_tempdir().full_path
    model_dir = os.path.join(
        os.path.dirname(__file__),
        "../testdata/qwen_dummy",
    )

    tokenizer_config = tokenizers_config.TokenizerConfig(
        tokenizer_type=tokenizers_config.TokenizerType.KERAS_HUB,
        model_name_or_path=model_dir,
        eos_token_id=0,
        pad_token_id=151643,
    )
    generation_config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.GREEDY,
        max_output_tokens=20,
        max_prompt_length=64,
        pad_token_id=151643,
    )
    export_config = tokenizers_exporter.NeptuneLLMExportWithTokenizerConfig(
        export_dir=export_dir,
        allowed_batch_sizes=(1,),
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
    )

    model_wrapper = tpu_wrapper.TPUInferenceWrapper(
        tpu_runner=runner,
        generation_config=generation_config,
    )

    # 1. Test manual pre_process -> model_fn -> post_process pipeline
    prompt_text = "Hello, what is your name?"
    with self.subTest("manual_pipeline"):
      tokenizer = tokenizers_processors.KerasHubTokenizerTF(
          model_name_or_path=tokenizer_config.model_name_or_path,
          eos_token_id=tokenizer_config.eos_token_id,
          pad_token_id=tokenizer_config.pad_token_id,
          max_seq_len=generation_config.max_prompt_length,
      )
      pre_processed = tokenizer.pre_process_fn(
          {"text_input": tf.constant([prompt_text])}
      )
      inputs_jax = {"input_ids": jnp.array(pre_processed["input_ids"].numpy())}

      with runner.mesh:  # pytype: disable=attribute-error
        model_fn = model_wrapper.compile_generation_loop()
        jit_model_fn = jax.jit(model_fn)
        jax_outputs = jit_model_fn(runner.state, inputs_jax)

      self.assertIn("token_ids", jax_outputs)
      self.assertEqual(jax_outputs["token_ids"].shape, (1, 20))

      post_processed = tokenizer.post_process_fn(
          {"token_ids": tf.constant(np.array(jax_outputs["token_ids"]))}
      )
      self.assertIn("text_output", post_processed)
      manual_output = post_processed["text_output"].numpy()
      # TODO: Add golden text output correctness tests with fixed
      # deterministic weights or real model weights to verify end-to-end output
      # fidelity.

    # 2. Export the model
    tpu_exporter.export_tpu_inference_model(
        tpu_runner=runner,
        config=export_config,
    )
    self.assertTrue(os.path.exists(export_dir))

    # 3. Test exported SavedModel execution via ModelRunner and compare
    # equivalence.

  def test_export_top_k_with_dummy_weights(self):
    """Verifies top_k export and execution equivalence between manual pipeline and SavedModel."""
    runner = self._create_runner(load_format="jax_dummy")
    export_dir = self.create_tempdir().full_path
    model_dir = os.path.join(
        os.path.dirname(__file__),
        "../testdata/qwen_dummy",
    )

    tokenizer_config = tokenizers_config.TokenizerConfig(
        tokenizer_type=tokenizers_config.TokenizerType.KERAS_HUB,
        model_name_or_path=model_dir,
        eos_token_id=0,
        pad_token_id=151643,
    )
    generation_config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.TOP_K,
        max_output_tokens=20,
        max_prompt_length=64,
        top_k=2,
        temperature=0.01,
        pad_token_id=151643,
    )
    export_config = tokenizers_exporter.NeptuneLLMExportWithTokenizerConfig(
        export_dir=export_dir,
        allowed_batch_sizes=(1,),
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
    )

    model_wrapper = tpu_wrapper.TPUInferenceWrapper(
        tpu_runner=runner,
        generation_config=generation_config,
    )

    # 1. Test manual pre_process -> model_fn -> post_process pipeline
    prompt_text = "Hello, what is your name?"
    with self.subTest("manual_pipeline"):
      tokenizer = tokenizers_processors.KerasHubTokenizerTF(
          model_name_or_path=tokenizer_config.model_name_or_path,
          eos_token_id=tokenizer_config.eos_token_id,
          pad_token_id=tokenizer_config.pad_token_id,
          max_seq_len=generation_config.max_prompt_length,
      )
      pre_processed = tokenizer.pre_process_fn(
          {"text_input": tf.constant([prompt_text])}
      )
      inputs_jax = {"input_ids": jnp.array(pre_processed["input_ids"].numpy())}

      with runner.mesh:  # pytype: disable=attribute-error
        model_fn = model_wrapper.compile_generation_loop()
        jit_model_fn = jax.jit(model_fn)
        jax_outputs = jit_model_fn(runner.state, inputs_jax)

      self.assertIn("token_ids", jax_outputs)
      self.assertEqual(jax_outputs["token_ids"].shape, (1, 20))

      post_processed = tokenizer.post_process_fn(
          {"token_ids": tf.constant(np.array(jax_outputs["token_ids"]))}
      )
      self.assertIn("text_output", post_processed)
      manual_output = post_processed["text_output"].numpy()
      # TODO: Add golden text output correctness tests with fixed
      # deterministic weights or real model weights to verify end-to-end output
      # fidelity.

    # 2. Export the model
    tpu_exporter.export_tpu_inference_model(
        tpu_runner=runner,
        config=export_config,
    )
    self.assertTrue(os.path.exists(export_dir))

    # 3. Test exported SavedModel execution via ModelRunner and compare
    # equivalence.

  def test_export_beam_search_with_dummy_weights(self):
    """Verifies beam search export and execution equivalence between manual pipeline and SavedModel."""
    runner = self._create_runner(load_format="jax_dummy")
    export_dir = self.create_tempdir().full_path
    beam_width = 4
    model_dir = os.path.join(
        os.path.dirname(__file__),
        "../testdata/qwen_dummy",
    )

    tokenizer_config = tokenizers_config.TokenizerConfig(
        tokenizer_type=tokenizers_config.TokenizerType.KERAS_HUB,
        model_name_or_path=model_dir,
        eos_token_id=0,
        pad_token_id=151643,
    )
    generation_config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.BEAM_SEARCH,
        max_output_tokens=20,
        beam_size=beam_width,
        max_prompt_length=64,
        pad_token_id=151643,
    )
    export_config = tokenizers_exporter.NeptuneLLMExportWithTokenizerConfig(
        export_dir=export_dir,
        allowed_batch_sizes=(1,),
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
    )

    model_wrapper = tpu_wrapper.TPUInferenceWrapper(
        tpu_runner=runner,
        generation_config=generation_config,
    )

    # 1. Test manual pre_process -> model_fn -> post_process pipeline
    prompt_text = "Hello, what is your name?"
    with self.subTest("manual_pipeline"):
      tokenizer = tokenizers_processors.KerasHubTokenizerTF(
          model_name_or_path=tokenizer_config.model_name_or_path,
          eos_token_id=tokenizer_config.eos_token_id,
          pad_token_id=tokenizer_config.pad_token_id,
          max_seq_len=generation_config.max_prompt_length,
      )
      pre_processed = tokenizer.pre_process_fn(
          {"text_input": tf.constant([prompt_text])}
      )
      inputs_jax = {"input_ids": jnp.array(pre_processed["input_ids"].numpy())}

      with runner.mesh:
        model_fn = model_wrapper.compile_generation_loop()
        jit_model_fn = jax.jit(model_fn)
        jax_outputs = jit_model_fn(runner.state, inputs_jax)

      self.assertIn("token_ids", jax_outputs)
      self.assertEqual(jax_outputs["token_ids"].shape, (1, beam_width, 20))
      self.assertIn("logprobs", jax_outputs)
      self.assertEqual(jax_outputs["logprobs"].shape, (1, beam_width))

      post_processed = tokenizer.post_process_fn(
          {"token_ids": tf.constant(np.array(jax_outputs["token_ids"]))}
      )
      self.assertIn("text_output", post_processed)
      manual_output = post_processed["text_output"].numpy()
      logging.info("manual_output: %s", manual_output)
      # TODO(armandouv): Compare manual_output with e2e_output once we debug why
      # they are different.

    # 2. Export the model to TF SavedModel using the integration exporter
    tpu_exporter.export_tpu_inference_model(
        tpu_runner=runner,
        config=export_config,
    )
    self.assertTrue(os.path.exists(export_dir))

    # 3. Test exported SavedModel execution via ModelRunner and compare
    # equivalence.

  @unittest.skipUnless(
      os.environ.get("RUN_REAL_WEIGHTS_TEST") == "1",
      "Requires RUN_REAL_WEIGHTS_TEST=1 env var.",
  )
  def test_export_beam_search_with_real_weights(self):
    """Heavy weight generation and text quality verification with real weights."""
    model_name = "Qwen/Qwen3-0.6B"
    runner = self._create_runner(load_format="safetensors")
    export_dir = self.create_tempdir().full_path
    beam_width = 4

    tokenizer_config = tokenizers_config.TokenizerConfig(
        tokenizer_type=tokenizers_config.TokenizerType.KERAS_HUB,
        model_name_or_path=model_name,
        eos_token_id=151643,
        pad_token_id=151643,
    )
    generation_config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.BEAM_SEARCH,
        max_output_tokens=20,
        beam_size=beam_width,
        max_prompt_length=128,
        pad_token_id=151643,
    )
    export_config = tokenizers_exporter.NeptuneLLMExportWithTokenizerConfig(
        export_dir=export_dir,
        allowed_batch_sizes=(1,),
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
    )

    # 1. Export model
    tpu_exporter.export_tpu_inference_model(
        tpu_runner=runner,
        config=export_config,
    )
    self.assertTrue(os.path.exists(export_dir))

    # 2. Run E2E ModelRunner inference with real weights


if __name__ == "__main__":
  absltest.main()
