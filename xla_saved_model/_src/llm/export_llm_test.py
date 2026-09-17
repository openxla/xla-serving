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

"""Tests for the XSE LLM export CLI tool."""

from unittest import mock

from absl.testing import absltest

from xla_saved_model._src.llm import config as export_config
from xla_saved_model._src.llm import export_llm
from xla_saved_model._src.llm.tokenizers import exporter as tokenizers_exporter


class ExportLlmTest(absltest.TestCase):

  def test_parse_args_defaults(self):
    args = export_llm.parse_args([
        "--model",
        "test-model",
        "--export-dir",
        "/tmp/export",
    ])
    self.assertEqual(args.model, "test-model")
    self.assertEqual(args.export_dir, "/tmp/export")
    self.assertEqual(args.allowed_batch_sizes, "1")
    self.assertIsNone(args.max_seq_len)
    self.assertEqual(args.max_prompt_len, 1024)
    self.assertEqual(args.max_new_tokens, 8)
    self.assertIsNone(args.tp_size)
    self.assertEqual(args.sampling_type, "greedy")
    self.assertEqual(args.backend, "tpu-inference")

  def test_parse_args_overrides(self):
    args = export_llm.parse_args([
        "--model",
        "test-model",
        "--export-dir",
        "/tmp/export",
        "--allowed-batch-sizes",
        "1,2,4",
        "--max-seq-len",
        "2048",
        "--max-new-tokens",
        "256",
        "--sampling-type",
        "beam_search",
        "--beam-width",
        "8",
    ])
    self.assertEqual(args.allowed_batch_sizes, "1,2,4")
    self.assertEqual(args.max_seq_len, 2048)
    self.assertEqual(args.max_new_tokens, 256)
    self.assertEqual(args.sampling_type, "beam_search")
    self.assertEqual(args.beam_width, 8)

  @mock.patch.object(export_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      export_llm.tpu_exporter, "export_tpu_inference_model", autospec=True
  )
  def test_main_tpu_inference_backend(self, mock_export, mock_init_runner):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    export_llm.main([
        "--model",
        "Qwen/Qwen3-1.5B",
        "--export-dir",
        "/tmp/mocked_export",
        "--sampling-type",
        "top_k",
        "--max-prompt-len",
        "1024",
        "--max-new-tokens",
        "512",
    ])

    mock_init_runner.assert_called_once()
    called_args = mock_init_runner.call_args[0][0]
    self.assertEqual(called_args.model, "Qwen/Qwen3-1.5B")
    self.assertEqual(called_args.max_seq_len, 1536)

    mock_export.assert_called_once()
    call_kwargs = mock_export.call_args[1]
    self.assertIs(call_kwargs["tpu_runner"], mock_runner)

    # Verify export config passed down correctly
    config = call_kwargs["config"]
    self.assertIsInstance(
        config, tokenizers_exporter.XLASavedModelLLMExportWithTokenizerConfig
    )
    self.assertEqual(config.export_dir, "/tmp/mocked_export")
    self.assertEqual(config.allowed_batch_sizes, (1,))
    self.assertEqual(
        config.generation_config.sampling_method,
        export_config.SamplingMethod.TOP_K,
    )
    self.assertEqual(config.generation_config.max_output_tokens, 512)
    self.assertEqual(config.generation_config.max_prompt_length, 1024)
    self.assertIsNone(config.generation_config.beam_size)

  @mock.patch.object(export_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      export_llm.tpu_exporter, "export_tpu_inference_model", autospec=True
  )
  def test_main_tpu_inference_backend_multi_batch_sizes(
      self, mock_export, mock_init_runner
  ):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    export_llm.main([
        "--model",
        "Qwen/Qwen3-1.5B",
        "--export-dir",
        "/tmp/mocked_export",
        "--allowed-batch-sizes",
        "1,2,4",
    ])

    mock_export.assert_called_once()
    config = mock_export.call_args[1]["config"]
    self.assertEqual(config.allowed_batch_sizes, (1, 2, 4))

  @mock.patch.object(export_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      export_llm.tpu_exporter, "export_tpu_inference_model", autospec=True
  )
  def test_main_tpu_inference_backend_beam_search(
      self, mock_export, mock_init_runner
  ):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    export_llm.main([
        "--model",
        "Qwen/Qwen3-1.5B",
        "--export-dir",
        "/tmp/mocked_export",
        "--sampling-type",
        "beam_search",
        "--beam-width",
        "8",
    ])

    mock_init_runner.assert_called_once()
    called_args = mock_init_runner.call_args[0][0]
    self.assertEqual(called_args.max_seq_len, 1032)

    mock_export.assert_called_once()
    call_kwargs = mock_export.call_args[1]
    self.assertIs(call_kwargs["tpu_runner"], mock_runner)

    config = call_kwargs["config"]
    self.assertEqual(
        config.generation_config.sampling_method,
        export_config.SamplingMethod.BEAM_SEARCH,
    )
    self.assertEqual(config.generation_config.beam_size, 8)
    self.assertIsNone(config.generation_config.max_candidate_tokens)

  @mock.patch.object(export_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      export_llm.tpu_exporter, "export_tpu_inference_model", autospec=True
  )
  def test_main_tpu_inference_backend_scoring(
      self, mock_export, mock_init_runner
  ):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    export_llm.main([
        "--model",
        "Qwen/Qwen3-1.5B",
        "--export-dir",
        "/tmp/mocked_export",
        "--sampling-type",
        "scoring",
        "--num-candidates",
        "16",
        "--max-candidate-tokens",
        "24",
    ])

    mock_init_runner.assert_called_once()
    mock_export.assert_called_once()
    call_kwargs = mock_export.call_args[1]
    self.assertIs(call_kwargs["tpu_runner"], mock_runner)

    config = call_kwargs["config"]
    self.assertEqual(
        config.generation_config.sampling_method,
        export_config.SamplingMethod.SCORING,
    )
    self.assertIsNone(config.generation_config.beam_size)
    self.assertEqual(config.generation_config.num_candidates, 16)
    self.assertEqual(config.generation_config.max_candidate_tokens, 24)

  @mock.patch.object(export_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      export_llm.tpu_exporter, "export_tpu_inference_model", autospec=True
  )
  def test_main_tpu_inference_backend_custom_seq_len(
      self, unused_mock_export, mock_init_runner
  ):
    mock_runner = mock.MagicMock()
    mock_init_runner.return_value = mock_runner

    export_llm.main([
        "--model",
        "Qwen/Qwen3-1.5B",
        "--export-dir",
        "/tmp/mocked_export",
        "--max-prompt-len",
        "1024",
        "--max-new-tokens",
        "512",
        "--max-seq-len",
        "2048",
    ])

    called_args = mock_init_runner.call_args[0][0]
    self.assertEqual(called_args.max_seq_len, 2048)

  @mock.patch.object(export_llm, "_get_tpu_runner", autospec=True)
  @mock.patch.object(
      export_llm.tpu_exporter, "export_tpu_inference_model", autospec=True
  )
  def test_main_tpu_inference_backend_insufficient_seq_len(
      self, mock_export, mock_init_runner
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"max_seq_len \(1024\) must be at least max_prompt_len \(1024\) \+"
        r" max_new_tokens \(512\)",
    ):
      export_llm.main([
          "--model",
          "Qwen/Qwen3-1.5B",
          "--export-dir",
          "/tmp/mocked_export",
          "--max-prompt-len",
          "1024",
          "--max-new-tokens",
          "512",
          "--max-seq-len",
          "1024",
      ])

    mock_init_runner.assert_not_called()
    mock_export.assert_not_called()

  def test_main_invalid_prompt_len(self):
    with self.assertRaisesRegex(
        ValueError, r"--max-prompt-len must be positive, got 0"
    ):
      export_llm.main([
          "--model",
          "Qwen/Qwen3-1.5B",
          "--export-dir",
          "/tmp/mocked_export",
          "--max-prompt-len",
          "0",
      ])

  def test_main_invalid_new_tokens(self):
    with self.assertRaisesRegex(
        ValueError, r"--max-new-tokens must be positive, got 0"
    ):
      export_llm.main([
          "--model",
          "Qwen/Qwen3-1.5B",
          "--export-dir",
          "/tmp/mocked_export",
          "--max-new-tokens",
          "0",
      ])

  def test_main_invalid_beam_width(self):
    with self.assertRaisesRegex(
        ValueError, r"--beam-width must be positive, got 0"
    ):
      export_llm.main([
          "--model",
          "Qwen/Qwen3-1.5B",
          "--export-dir",
          "/tmp/mocked_export",
          "--beam-width",
          "0",
      ])

  def test_main_invalid_allowed_batch_sizes(self):
    with self.assertRaisesRegex(
        ValueError, r"Each batch size must be positive, got -1\."
    ):
      export_llm.main([
          "--model",
          "Qwen/Qwen3-1.5B",
          "--export-dir",
          "/tmp/mocked_export",
          "--allowed-batch-sizes",
          "-1",
      ])

  def test_main_invalid_sampling_type(self):
    with self.assertRaises(SystemExit):
      # argparse will reject invalid choices immediately and call sys.exit
      export_llm.main([
          "--model",
          "Qwen/Qwen3-1.5B",
          "--export-dir",
          "/tmp/mocked_export",
          "--sampling-type",
          "invalid_sampling_type",
      ])

  def test_main_unknown_backend(self):
    with self.assertRaises(SystemExit):
      # argparse will reject invalid choices immediately and call sys.exit
      export_llm.main([
          "--model",
          "test",
          "--export-dir",
          "/tmp",
          "--backend",
          "unknown-backend",
      ])


if __name__ == "__main__":
  absltest.main()
