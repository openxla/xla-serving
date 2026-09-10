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

r"""CLI tool for exporting OSS LLMs to XLA SavedModel across multiple backends.

This command-line utility exports open-source Hugging Face Large Language Models
(e.g., Qwen, Llama, Gemma) from JAX/vLLM into Orbax Model bundles ready for
serving on Google Cloud TPU via XLA Serving Engine (XSE).

General Usage:
  python3 -m xla_saved_model._src.llm.export_llm [options]
  # OR using the console script entry point:
  xsm-llm-export [options]

Commands & Usage Examples:
  1. Display full usage help:
     xsm-llm-export --help
     # OR:
     xsm-llm-export help

  2. Export a model with Greedy sampling:
     xsm-llm-export \
       --model Qwen/Qwen3-0.6B \
       --export-dir /tmp/qwen3_greedy_export \
       --sampling-type greedy \
       --allowed-batch-sizes 1 \
       --max-prompt-len 1024 \
       --max-new-tokens 32

  3. Export a model with Beam Search sampling:
     xsm-llm-export \
       --model Qwen/Qwen3-0.6B \
       --export-dir /tmp/qwen3_beam_export \
       --sampling-type beam_search \
       --beam-width 4 \
       --allowed-batch-sizes 1 \
       --max-prompt-len 512 \
       --max-new-tokens 16

Required Arguments:
  --model:      Hugging Face model ID or local path (e.g., Qwen/Qwen3-0.6B).
  --export-dir: Destination directory for the Orbax Model bundle.

Optional Arguments:
  --sampling-type:       greedy, top_k, beam_search, scoring.
  --beam-width:          Number of candidate beams when using beam search (default: 4).
  --allowed-batch-sizes: Comma-separated list of static batch sizes allowed during serving (default: '1').
  --tp-size:             Tensor parallel size (defaults to all available local TPU devices).
"""

import argparse
import sys
from typing import Any

import jax

from xla_saved_model._src.llm import config as export_config
from xla_saved_model._src.llm.tokenizers import config as tokenizers_config
from xla_saved_model._src.llm.tokenizers import exporter as tokenizers_exporter
from xla_saved_model._src.llm.tpu_inference import exporter as tpu_exporter

# pylint: disable=g-import-not-at-top
try:
  from vllm.engine import arg_utils as vllm_arg_utils
  from tpu_inference.distributed import jax_parallel_state
  from tpu_inference.runner import tpu_runner

  _TPU_INFERENCE_AVAILABLE = True
  _TPU_INFERENCE_IMPORT_ERROR = None
except ImportError as import_error:
  vllm_arg_utils = None
  jax_parallel_state = None
  tpu_runner = None
  _TPU_INFERENCE_AVAILABLE = False
  _TPU_INFERENCE_IMPORT_ERROR = import_error
# pylint: enable=g-import-not-at-top


def _require_tpu_inference():
  if not _TPU_INFERENCE_AVAILABLE:
    raise ImportError(
        "tpu-inference or vllm is not installed. Please install them to use"
        " this backend."
    ) from _TPU_INFERENCE_IMPORT_ERROR


def _get_tpu_runner(
    args: argparse.Namespace,
) -> Any:
  """Initializes the TPU Inference backend and returns a TPUModelRunner instance.

  Args:
    args: Parsed command-line arguments.

  Returns:
    Loaded TPUModelRunner instance.

  Raises:
    ImportError: If tpu_inference or vllm libraries are not installed.
    RuntimeError: If there are fewer devices available than requested by
      tp_size.
  """
  _require_tpu_inference()

  if args.tp_size is None:
    args.tp_size = jax.device_count()

  assert vllm_arg_utils is not None
  # We use vLLM's EngineArgs config loader to easily parse models from HF.
  engine_args = vllm_arg_utils.EngineArgs(
      model=args.model,
      max_model_len=args.max_seq_len,
      load_format=getattr(args, "load_format", "auto"),
      enforce_eager=True,
      tensor_parallel_size=args.tp_size,
      gpu_memory_utilization=0.8,
  )
  vllm_config = engine_args.create_engine_config()

  devices = jax.local_devices()[: args.tp_size]
  if len(devices) < args.tp_size:
    raise RuntimeError(
        f"Requested tp_size={args.tp_size} but only {len(devices)} devices"
        " available."
    )

  assert jax_parallel_state is not None
  jax_parallel_state.init_pp_distributed_environment(
      ip="127.0.0.1",
      rank=0,
      world_size=1,
      device=devices[0],
      need_pp=False,
  )

  assert tpu_runner is not None
  print(f"Initializing TPUModelRunner with {args.tp_size} devices...")
  runner = tpu_runner.TPUModelRunner(
      vllm_config=vllm_config,
      devices=devices,
      rank=0,
      is_first_rank=True,
      is_last_rank=True,
  )
  runner.load_model()
  return runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments and displays detailed usage help."""
  epilog_examples = """
Examples:
  1. Export Qwen3-0.6B with Greedy sampling:
     python -m xla_saved_model._src.llm.export_llm \\
       --model Qwen/Qwen3-0.6B \\
       --export-dir /tmp/qwen3_greedy_export \\
       --sampling-type greedy \\
       --batch-size 1 \\
       --max-prompt-len 1024 \\
       --max-new-tokens 32

  2. Export Qwen3-0.6B with Beam Search (beam_width=4):
     python -m xla_saved_model._src.llm.export_llm \\
       --model Qwen/Qwen3-0.6B \\
       --export-dir /tmp/qwen3_beam_export \\
       --sampling-type beam_search \\
       --beam-width 4 \\
       --batch-size 1 \\
       --max-prompt-len 512 \\
       --max-new-tokens 16

  3. Export across multiple TPU chips with Tensor Parallelism:
     python -m xla_saved_model._src.llm.export_llm \\
       --model Qwen/Qwen3-7B \\
       --export-dir /tmp/qwen3_7b_tp4 \\
       --tp-size 4 \\
       --sampling-type greedy
"""
  parser = argparse.ArgumentParser(
      description=(
          "XSE LLM Export CLI Tool - Export OSS LLM models to Orbax format "
          "for high-performance serving on XSE."
      ),
      epilog=epilog_examples,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--model",
      type=str,
      required=True,
      help="Hugging Face model ID or path (e.g., Qwen/Qwen3-0.6B).",
  )
  parser.add_argument(
      "--export-dir",
      type=str,
      required=True,
      help="Output directory for the exported XLA SavedModel.",
  )
  parser.add_argument(
      "--load-format",
      type=str,
      default="auto",
      help=(
          "Model weight load format (e.g. 'auto', 'safetensors', 'jax_dummy')."
      ),
  )
  parser.add_argument(
      "--allowed-batch-sizes",
      type=str,
      default="1",
      help=(
          "Comma-separated list of allowed batch sizes for serving (default:"
          " '1', e.g. '1,2,4')."
      ),
  )
  parser.add_argument(
      "--max-seq-len",
      type=int,
      default=None,
      help=(
          "Maximum total sequence length (prompt + completion). If omitted, "
          "defaults to max-prompt-len + max-new-tokens."
      ),
  )
  parser.add_argument(
      "--max-prompt-len",
      type=int,
      default=1024,
      help="Maximum allowed input prompt tokens (default: 1024).",
  )
  parser.add_argument(
      "--max-new-tokens",
      type=int,
      default=8,
      help="Maximum new tokens to generate per request (default: 8).",
  )
  parser.add_argument(
      "--tp-size",
      type=int,
      default=None,
      help=(
          "Tensor parallel size (number of TPU chips to shard across). "
          "Defaults to total available JAX devices."
      ),
  )
  parser.add_argument(
      "--sampling-type",
      type=str,
      choices=[
          sampling_method.value
          for sampling_method in export_config.SamplingMethod
      ],
      default="greedy",
      help=(
          "Sampling method to compile into the generation loop (greedy, "
          "top_k, beam_search, scoring)."
      ),
  )
  parser.add_argument(
      "--beam-width",
      type=int,
      default=4,
      help=(
          "Beam search width (number of candidate beams per prompt; only "
          "used when --sampling-type is beam_search)."
      ),
  )
  parser.add_argument(
      "--num-candidates",
      type=int,
      default=500,
      help="Number of candidate suffixes when using candidate scoring.",
  )
  parser.add_argument(
      "--max-candidate-tokens",
      type=int,
      default=32,
      help="Maximum token length of candidate completions for scoring.",
  )
  parser.add_argument(
      "--backend",
      type=str,
      choices=["tpu-inference"],
      default="tpu-inference",
      help="Backend implementation to use for model loading.",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None):
  args = parse_args(argv)
  if args.max_prompt_len <= 0:
    raise ValueError(
        f"--max-prompt-len must be positive, got {args.max_prompt_len}"
    )
  if args.max_new_tokens <= 0:
    raise ValueError(
        f"--max-new-tokens must be positive, got {args.max_new_tokens}"
    )
  if args.beam_width <= 0:
    raise ValueError(f"--beam-width must be positive, got {args.beam_width}")
  if args.num_candidates <= 0:
    raise ValueError(
        f"--num-candidates must be positive, got {args.num_candidates}"
    )

  parsed_batch_sizes = [
      int(b.strip()) for b in args.allowed_batch_sizes.split(",") if b.strip()
  ]
  if not parsed_batch_sizes:
    raise ValueError("--allowed-batch-sizes cannot be empty.")
  for b in parsed_batch_sizes:
    if b <= 0:
      raise ValueError(f"Each batch size must be positive, got {b}.")
  allowed_batch_sizes = tuple(parsed_batch_sizes)

  if args.max_seq_len is None:
    args.max_seq_len = args.max_prompt_len + args.max_new_tokens
  elif args.max_seq_len < args.max_prompt_len + args.max_new_tokens:
    raise ValueError(
        f"max_seq_len ({args.max_seq_len}) must be at least max_prompt_len"
        f" ({args.max_prompt_len}) + max_new_tokens ({args.max_new_tokens})"
    )

  print(
      f"Starting export pipeline for {args.model} using backend {args.backend}"
  )

  sampling_method = export_config.SamplingMethod(args.sampling_type)

  if sampling_method == export_config.SamplingMethod.BEAM_SEARCH:
    beam_size = args.beam_width
    num_candidates = None
    max_candidate_tokens = None
  elif sampling_method == export_config.SamplingMethod.SCORING:
    beam_size = None
    num_candidates = args.num_candidates
    max_candidate_tokens = args.max_candidate_tokens
  else:
    beam_size = None
    num_candidates = None
    max_candidate_tokens = None

  gen_config = export_config.GenerationConfig(
      max_prompt_length=args.max_prompt_len,
      max_output_tokens=args.max_new_tokens,
      sampling_method=sampling_method,
      beam_size=beam_size,
      num_candidates=num_candidates,
      max_candidate_tokens=max_candidate_tokens,
  )

  full_config = tokenizers_exporter.XLASavedModelLLMExportWithTokenizerConfig(
      export_dir=args.export_dir,
      allowed_batch_sizes=allowed_batch_sizes,
      generation_config=gen_config,
      tokenizer_config=tokenizers_config.TokenizerConfig(
          model_name_or_path=args.model,
      ),
  )

  if args.backend == "tpu-inference":
    runner = _get_tpu_runner(args)
    print(f"Exporting model to {args.export_dir}...")
    tpu_exporter.export_tpu_inference_model(
        tpu_runner=runner,
        config=full_config,
    )
  else:
    raise NotImplementedError(f"Backend {args.backend} is not implemented yet.")

  print("Export complete. You can now run it using model_runner.")


if __name__ == "__main__":
  main(sys.argv[1:])
