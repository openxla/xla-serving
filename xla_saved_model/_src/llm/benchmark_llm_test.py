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

"""Tests for the XSE LLM benchmark CLI tool."""

from unittest import mock

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model._src.llm import benchmark_llm
from xla_saved_model._src.llm import config as llm_config
from xla_saved_model.protos.xla.python.ifrt import dtype_pb2  # pylint: disable=g-direct-tensorflow-import


class BenchmarkLlmTest(absltest.TestCase):

  def test_parse_args_defaults(self):
    args = benchmark_llm.parse_args(
        ["--mode", "model_fn", "--prompt", "Default test prompt"]
    )
    self.assertEqual(args.mode, "model_fn")
    self.assertEqual(args.model, "Qwen/Qwen3-0.6B")
    self.assertEqual(args.load_format, "auto")
    self.assertEqual(args.prompt, "Default test prompt")
    self.assertIsNone(args.max_seq_len)
    self.assertEqual(args.prompt_len, 128)
    self.assertEqual(args.new_tokens, 20)
    self.assertEqual(args.sampling_type, "beam_search")
    self.assertEqual(args.beam_width, 4)
    self.assertIsNone(args.tp_size)
    self.assertEqual(args.server_address, "localhost:8500")
    self.assertEqual(args.model_name, "qwen3")
    self.assertEqual(args.signature_name, "serving_default")
    self.assertEqual(args.batch_size, 1)
    self.assertEqual(args.num_batches, 3)

  def test_parse_args_overrides(self):
    args = benchmark_llm.parse_args([
        "--mode",
        "grpc_client",
        "--model",
        "Qwen/Qwen3-1.5B",
        "--load-format",
        "jax_dummy",
        "--max-seq-len",
        "1024",
        "--prompt-len",
        "256",
        "--new-tokens",
        "64",
        "--sampling-type",
        "greedy",
        "--beam-width",
        "8",
        "--tp-size",
        "2",
        "--server-address",
        "10.0.0.1:9000",
        "--model-name",
        "custom_model",
        "--signature-name",
        "custom_signature",
        "--prompt",
        "Hello world",
        "--batch-size",
        "4",
        "--num-batches",
        "10",
    ])
    self.assertEqual(args.mode, "grpc_client")
    self.assertEqual(args.model, "Qwen/Qwen3-1.5B")
    self.assertEqual(args.load_format, "jax_dummy")
    self.assertEqual(args.max_seq_len, 1024)
    self.assertEqual(args.prompt_len, 256)
    self.assertEqual(args.new_tokens, 64)
    self.assertEqual(args.sampling_type, "greedy")
    self.assertEqual(args.beam_width, 8)
    self.assertEqual(args.tp_size, 2)
    self.assertEqual(args.server_address, "10.0.0.1:9000")
    self.assertEqual(args.model_name, "custom_model")
    self.assertEqual(args.signature_name, "custom_signature")
    self.assertEqual(args.prompt, "Hello world")
    self.assertEqual(args.batch_size, 4)
    self.assertEqual(args.num_batches, 10)

  def test_main_insufficient_max_seq_len(self):
    with self.assertRaisesRegex(
        ValueError,
        r"max_seq_len \(1024\) must be at least prompt_len \(1024\) \+"
        r" new_tokens \(512\)",
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--prompt-len",
          "1024",
          "--new-tokens",
          "512",
          "--max-seq-len",
          "1024",
      ])

  def test_main_invalid_mode(self):
    with self.assertRaises(SystemExit):
      benchmark_llm.main(["--mode", "invalid_mode", "--prompt", "Test prompt"])

  def test_main_missing_prompt(self):
    with self.assertRaises(SystemExit):
      benchmark_llm.main(["--mode", "model_fn"])

  def test_main_model_runner_missing_model_path(self):
    with self.assertRaisesRegex(
        ValueError,
        r"--model-path must be specified for model_runner mode\.",
    ):
      benchmark_llm.main(["--mode", "model_runner", "--prompt", "Test prompt"])

  @mock.patch.object(benchmark_llm, "model_runner", None)
  def test_main_model_runner_unavailable(self):
    with self.assertRaisesRegex(
        RuntimeError,
        r"model_runner Python bindings could not be imported\. Ensure the C\+\+"
        r" runtime library is built and available\.",
    ):
      benchmark_llm.main([
          "--mode",
          "model_runner",
          "--model-path",
          "/tmp/fake_model",
          "--prompt",
          "Test prompt",
      ])

  def test_main_model_runner_benchmark(self):
    mock_runner_instance = mock.MagicMock()
    mock_sig = mock.MagicMock()
    mock_sig.run_with_dict_return.return_value = {
        "text_output": np.array(["ModelRunner generated text"], dtype=object)
    }
    mock_runner_instance.signature.return_value = mock_sig

    mock_model_runner_module = mock.MagicMock()
    mock_model_runner_module.ModelRunner.return_value = mock_runner_instance

    with mock.patch.object(
        benchmark_llm, "model_runner", mock_model_runner_module
    ):
      benchmark_llm.main([
          "--mode",
          "model_runner",
          "--model-path",
          "/tmp/test_export_dir",
          "--model-name",
          "qwen3",
          "--signature-name",
          "serving_default",
          "--prompt",
          "Test prompt text",
          "--batch-size",
          "2",
          "--num-batches",
          "3",
      ])

    mock_model_runner_module.ModelRunner.assert_called_once_with(
        model_name="qwen3",
        model_path="/tmp/test_export_dir",
    )
    mock_runner_instance.signature.assert_called_once_with("serving_default")
    # 1 warmup + 3 timed batches = 4 calls
    self.assertEqual(mock_sig.run_with_dict_return.call_count, 4)

  @mock.patch.object(benchmark_llm.jax, "jit", side_effect=lambda f: f)
  @mock.patch.object(benchmark_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      benchmark_llm.tokenizers_processors,
      "KerasHubTokenizerTF",
      autospec=True,
  )
  @mock.patch.object(
      benchmark_llm.tpu_wrapper,
      "TPUInferenceWrapper",
      autospec=True,
  )
  def test_main_model_fn_benchmark(
      self,
      mock_wrapper_cls,
      mock_tokenizer_cls,
      mock_init_runner,
      unused_mock_jit,
  ):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pre_process_fn.return_value = {
        "input_ids": tf.constant([[1, 2, 3]])
    }
    mock_tokenizer.post_process_fn.return_value = {
        "text_output": tf.constant([["Generated sample response"]])
    }
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_wrapper = mock.MagicMock()
    mock_wrapper.generation_config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.BEAM_SEARCH,
        max_output_tokens=20,
        beam_size=4,
        max_prompt_length=128,
    )
    mock_model_fn = mock.MagicMock(
        return_value={"token_ids": np.array([[1, 2]])}
    )
    mock_wrapper.compile_generation_loop.return_value = mock_model_fn
    mock_wrapper_cls.return_value = mock_wrapper

    benchmark_llm.main([
        "--mode",
        "model_fn",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--prompt",
        "Test prompt text",
        "--prompt-len",
        "128",
        "--new-tokens",
        "20",
        "--sampling-type",
        "beam_search",
        "--beam-width",
        "4",
        "--batch-size",
        "1",
        "--num-batches",
        "2",
    ])

    mock_init_runner.assert_called_once()
    mock_tokenizer_cls.assert_called_once_with(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_seq_len=128,
        max_candidate_tokens=None,
    )
    mock_wrapper.compile_generation_loop.assert_called_once()

  @mock.patch.object(
      benchmark_llm.grpc, "local_channel_credentials", autospec=True
  )
  @mock.patch.object(benchmark_llm.grpc, "secure_channel", autospec=True)
  def test_main_grpc_client_benchmark(
      self, mock_secure_channel_fn, mock_local_creds_fn
  ):
    mock_channel = mock.MagicMock()
    mock_predict_method = mock.MagicMock()
    sentinel_creds = mock.MagicMock()

    response = predict_pb2.PredictResponse()
    hb = host_buffer_pb2.HostBufferProto()
    hb.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
    hb.shape.dims.append(1)
    hb.string_data.elements.append(b"Generated answer from server")
    response.outputs["text_output"].CopyFrom(hb)

    mock_predict_method.return_value = response
    mock_channel.unary_unary.return_value = mock_predict_method
    mock_local_creds_fn.return_value = sentinel_creds
    mock_secure_channel_fn.return_value = mock_channel

    benchmark_llm.main([
        "--mode",
        "grpc_client",
        "--server-address",
        "localhost:8500",
        "--model-name",
        "qwen3",
        "--signature-name",
        "serving_default",
        "--prompt",
        "Test prompt text",
        "--batch-size",
        "1",
        "--num-batches",
        "2",
    ])

    mock_local_creds_fn.assert_called_once_with()
    mock_secure_channel_fn.assert_called_once_with(
        "localhost:8500", sentinel_creds
    )
    mock_channel.unary_unary.assert_called_once_with(
        f"/{predict_pb2.DESCRIPTOR.package}.PredictionService/Predict",
        request_serializer=predict_pb2.PredictRequest.SerializeToString,
        response_deserializer=predict_pb2.PredictResponse.FromString,
    )
    # 1 warmup + 2 timed iterations = 3 calls
    self.assertEqual(mock_predict_method.call_count, 3)

  @mock.patch.object(
      benchmark_llm.grpc, "alts_channel_credentials", autospec=True
  )
  @mock.patch.object(benchmark_llm.grpc, "secure_channel", autospec=True)
  def test_main_grpc_client_benchmark_alts(
      self, mock_secure_channel_fn, mock_alts_creds_fn
  ):
    mock_channel = mock.MagicMock()
    mock_predict_method = mock.MagicMock()
    sentinel_creds = mock.MagicMock()

    response = predict_pb2.PredictResponse()
    hb = host_buffer_pb2.HostBufferProto()
    hb.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
    hb.shape.dims.append(1)
    hb.string_data.elements.append(b"Generated answer from server")
    response.outputs["text_output"].CopyFrom(hb)

    mock_predict_method.return_value = response
    mock_channel.unary_unary.return_value = mock_predict_method
    mock_alts_creds_fn.return_value = sentinel_creds
    mock_secure_channel_fn.return_value = mock_channel

    benchmark_llm.main([
        "--mode",
        "grpc_client",
        "--server-address",
        "10.0.0.1:8500",
        "--use-alts-credentials",
        "--model-name",
        "qwen3",
        "--prompt",
        "Test prompt text",
        "--batch-size",
        "1",
        "--num-batches",
        "1",
    ])

    mock_alts_creds_fn.assert_called_once_with()
    mock_secure_channel_fn.assert_called_once_with(
        "10.0.0.1:8500", sentinel_creds
    )
    self.assertEqual(mock_predict_method.call_count, 2)

  @mock.patch.object(benchmark_llm.grpc, "insecure_channel", autospec=True)
  def test_main_grpc_client_benchmark_remote_insecure(self, mock_channel_fn):
    mock_channel = mock.MagicMock()
    mock_predict_method = mock.MagicMock()

    response = predict_pb2.PredictResponse()
    hb = host_buffer_pb2.HostBufferProto()
    hb.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
    hb.shape.dims.append(1)
    hb.string_data.elements.append(b"Generated answer from server")
    response.outputs["text_output"].CopyFrom(hb)

    mock_predict_method.return_value = response
    mock_channel.unary_unary.return_value = mock_predict_method
    mock_channel_fn.return_value = mock_channel

    benchmark_llm.main([
        "--mode",
        "grpc_client",
        "--server-address",
        "10.0.0.1:8500",
        "--model-name",
        "qwen3",
        "--prompt",
        "Test prompt text",
        "--batch-size",
        "1",
        "--num-batches",
        "1",
    ])

    mock_channel_fn.assert_called_once_with("10.0.0.1:8500")
    self.assertEqual(mock_predict_method.call_count, 2)

  def test_main_invalid_batch_size(self):
    with self.assertRaisesRegex(
        ValueError, r"--batch-size must be positive, got 0"
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--batch-size",
          "0",
      ])

  def test_main_invalid_num_batches(self):
    with self.assertRaisesRegex(
        ValueError, r"--num-batches must be positive, got -1"
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--num-batches",
          "-1",
      ])

  def test_main_invalid_num_candidates(self):
    with self.assertRaisesRegex(
        ValueError, r"--num-candidates must be positive, got 0"
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--num-candidates",
          "0",
      ])

  def test_main_invalid_candidate_tokens(self):
    with self.assertRaisesRegex(
        ValueError, r"--candidate-tokens must be positive, got 0"
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--candidate-tokens",
          "0",
      ])

  def test_main_invalid_beam_width(self):
    with self.assertRaisesRegex(
        ValueError, r"--beam-width must be positive, got 0"
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--beam-width",
          "0",
      ])

  def test_main_invalid_prompt_len(self):
    with self.assertRaisesRegex(
        ValueError, r"--prompt-len must be positive, got 0"
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--prompt-len",
          "0",
      ])

  def test_main_invalid_new_tokens(self):
    with self.assertRaisesRegex(
        ValueError, r"--new-tokens must be positive, got 0"
    ):
      benchmark_llm.main([
          "--mode",
          "model_fn",
          "--prompt",
          "Test prompt",
          "--new-tokens",
          "0",
      ])

  def test_format_text(self):
    self.assertEqual(benchmark_llm._format_text(b"hello"), "hello")
    self.assertEqual(benchmark_llm._format_text(bytearray(b"world")), "world")
    self.assertEqual(benchmark_llm._format_text("test"), "test")
    self.assertEqual(benchmark_llm._format_text(123), "123")

  def test_print_decoded_outputs_beam_search(self):
    # Single batch, multiple beams
    beams_1d = np.array([b"Beam candidate 0", b"Beam candidate 1"])
    with mock.patch("builtins.print") as mock_print:
      benchmark_llm._print_decoded_outputs(beams_1d, is_beam_search=True)
      mock_print.assert_any_call("  Beam 0: Beam candidate 0")
      mock_print.assert_any_call("  Beam 1: Beam candidate 1")

    # Multi-batch, multiple beams
    beams_2d = np.array([
        [b"Prompt 0 Beam 0", b"Prompt 0 Beam 1"],
        [b"Prompt 1 Beam 0", b"Prompt 1 Beam 1"],
    ])
    with mock.patch("builtins.print") as mock_print:
      benchmark_llm._print_decoded_outputs(beams_2d, is_beam_search=True)
      mock_print.assert_any_call("  [Batch 0]:")
      mock_print.assert_any_call("    Beam 0: Prompt 0 Beam 0")
      mock_print.assert_any_call("    Beam 1: Prompt 0 Beam 1")
      mock_print.assert_any_call("  [Batch 1]:")
      mock_print.assert_any_call("    Beam 0: Prompt 1 Beam 0")
      mock_print.assert_any_call("    Beam 1: Prompt 1 Beam 1")

  def test_print_decoded_outputs_standard_sampling(self):
    # Single batch standard output
    single_output = np.array([b"Single generated output"])
    with mock.patch("builtins.print") as mock_print:
      benchmark_llm._print_decoded_outputs(single_output, is_beam_search=False)
      mock_print.assert_any_call("  Single generated output")

    # Multi-batch standard output
    multi_output = np.array([b"Output batch 0", b"Output batch 1"])
    with mock.patch("builtins.print") as mock_print:
      benchmark_llm._print_decoded_outputs(multi_output, is_beam_search=False)
      mock_print.assert_any_call("  [Batch 0]: Output batch 0")
      mock_print.assert_any_call("  [Batch 1]: Output batch 1")

  def test_print_benchmark_metrics(self):
    with mock.patch("builtins.print") as mock_print:
      benchmark_llm._print_benchmark_metrics(
          latencies_ms=[10.0, 20.0, 30.0],
          batch_size=2,
          title="Test Results",
          new_tokens=10,
      )
      mock_print.assert_any_call("\n=== Test Results ===")
      mock_print.assert_any_call("Mean Batch Latency      : 20.00 ms")
      mock_print.assert_any_call("Min Batch Latency       : 10.00 ms")
      mock_print.assert_any_call("Max Batch Latency       : 30.00 ms")
      mock_print.assert_any_call("Queries Per Second (QPS): 100.00 qps")
      mock_print.assert_any_call("Generated Token Rate    : 1000.00 tokens/s")

    with mock.patch("builtins.print") as mock_print:
      benchmark_llm._print_benchmark_metrics(
          latencies_ms=[10.0, 20.0, 30.0],
          batch_size=2,
          title="Scoring Test Results",
          num_candidates=4,
          candidate_tokens=5,
      )
      mock_print.assert_any_call("\n=== Scoring Test Results ===")
      mock_print.assert_any_call("Queries Per Second (QPS): 100.00 qps")
      mock_print.assert_any_call(
          "Candidate Scoring Rate  : 400.00 candidates/s"
      )
      mock_print.assert_any_call("Scored Token Rate       : 2000.00 tokens/s")

  @mock.patch.object(benchmark_llm.jax, "jit", side_effect=lambda f: f)
  @mock.patch.object(benchmark_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      benchmark_llm.tokenizers_processors,
      "KerasHubTokenizerTF",
      autospec=True,
  )
  @mock.patch.object(
      benchmark_llm.tpu_wrapper,
      "TPUInferenceWrapper",
      autospec=True,
  )
  def test_main_model_fn_benchmark_greedy(
      self,
      mock_wrapper_cls,
      mock_tokenizer_cls,
      mock_init_runner,
      unused_mock_jit,
  ):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pre_process_fn.return_value = {
        "input_ids": tf.constant([[1, 2, 3]])
    }
    mock_tokenizer.post_process_fn.return_value = {
        "text_output": tf.constant(["Greedy sample response"])
    }
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_wrapper = mock.MagicMock()
    mock_wrapper.generation_config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.GREEDY,
        max_output_tokens=16,
        max_prompt_length=128,
    )
    mock_model_fn = mock.MagicMock(
        return_value={"token_ids": np.array([[1, 2]])}
    )
    mock_wrapper.compile_generation_loop.return_value = mock_model_fn
    mock_wrapper_cls.return_value = mock_wrapper

    benchmark_llm.main([
        "--mode",
        "model_fn",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--prompt",
        "Test greedy prompt",
        "--prompt-len",
        "128",
        "--new-tokens",
        "16",
        "--sampling-type",
        "greedy",
        "--batch-size",
        "1",
        "--num-batches",
        "2",
    ])

    mock_init_runner.assert_called_once()
    mock_tokenizer_cls.assert_called_once_with(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_seq_len=128,
        max_candidate_tokens=None,
    )
    mock_wrapper.compile_generation_loop.assert_called_once()

  @mock.patch.object(benchmark_llm.jax, "jit", side_effect=lambda f: f)
  @mock.patch.object(benchmark_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      benchmark_llm.tokenizers_processors,
      "KerasHubTokenizerTF",
      autospec=True,
  )
  @mock.patch.object(
      benchmark_llm.tpu_wrapper,
      "TPUInferenceWrapper",
      autospec=True,
  )
  def test_main_model_fn_benchmark_scoring(
      self,
      mock_wrapper_cls,
      mock_tokenizer_cls,
      mock_init_runner,
      unused_mock_jit,
  ):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pre_process_fn.return_value = {
        "input_ids": tf.constant([[1, 2, 3]]),
        "candidate_ids": tf.constant([[[4, 5], [6, 7]]]),
    }
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_wrapper = mock.MagicMock()
    mock_wrapper.generation_config = llm_config.GenerationConfig(
        sampling_method=llm_config.SamplingMethod.SCORING,
        num_candidates=2,
        max_candidate_tokens=2,
        max_prompt_length=128,
    )
    mock_model_fn = mock.MagicMock(
        return_value={"scores": np.array([[-1.5, -2.5]])}
    )
    mock_wrapper.compile_generation_loop.return_value = mock_model_fn
    mock_wrapper_cls.return_value = mock_wrapper

    benchmark_llm.main([
        "--mode",
        "model_fn",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--prompt",
        "Test scoring prompt",
        "--prompt-len",
        "128",
        "--sampling-type",
        "scoring",
        "--num-candidates",
        "2",
        "--candidate-tokens",
        "2",
        "--batch-size",
        "1",
        "--num-batches",
        "2",
    ])

    mock_init_runner.assert_called_once()
    mock_tokenizer_cls.assert_called_once_with(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_seq_len=128,
        max_candidate_tokens=2,
    )
    mock_wrapper.compile_generation_loop.assert_called_once()


if __name__ == "__main__":
  absltest.main()
