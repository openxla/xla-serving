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

r"""LLM Benchmarking CLI for XSE.

Supports three benchmarking modes:
  1. model_fn:     Benchmarks the JIT-compiled generation loop directly in JAX.
  2. grpc_client:  Sends gRPC requests to an existing C++ XLA SavedModel Server.
  3. model_runner: Benchmarks exported models via C++ ModelRunner Python
     bindings.

Usage examples:
  # 1. Benchmark JAX model_fn directly:
  python3 -m xla_saved_model._src.llm.benchmark_llm \
      --mode model_fn --model Qwen/Qwen3-0.6B

  # 2. Benchmark against a running gRPC C++ Model Server:
  python3 -m xla_saved_model._src.llm.benchmark_llm \
      --mode grpc_client --server-address localhost:8500

  # 3. Benchmark exported model via ModelRunner:
  python3 -m xla_saved_model._src.llm.benchmark_llm \
      --mode model_runner --model-path /path/to/exported/model
"""

import argparse
import dataclasses
import sys
import time
from typing import Any

import grpc
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model._src.llm import config as llm_config
from xla_saved_model._src.llm.tokenizers import processors as tokenizers_processors
from xla_saved_model._src.llm.tpu_inference import model_wrapper as tpu_wrapper

from xla_saved_model.protos.xla.python.ifrt import dtype_pb2  # pylint: disable=g-direct-tensorflow-import

# pylint: disable=g-import-not-at-top
try:
  from xla_saved_model_runner import model_runner
except ImportError:
  model_runner = None
# pylint: enable=g-import-not-at-top

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

# PredictionService is declared in prediction_service.proto, which shares its
# package with predict.proto. Deriving the method path from the descriptor keeps
# it correct if that package is ever renamed.
_PREDICT_METHOD = f"/{predict_pb2.DESCRIPTOR.package}.PredictionService/Predict"


def _require_tpu_inference() -> None:
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
  engine_args = vllm_arg_utils.EngineArgs(
      model=args.model,
      max_model_len=args.max_seq_len,
      load_format=args.load_format,
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


def _format_text(val: Any) -> str:
  """Formats raw output bytes or objects to a decoded string."""
  if isinstance(val, (bytes, bytearray)):
    return val.decode("utf-8", errors="replace")
  return str(val)


def _print_decoded_outputs(
    decoded_text: Any,
    is_beam_search: bool = False,
) -> None:
  """Prints generated text outputs cleanly formatted for beam search or standard sampling.

  Args:
    decoded_text: String or array of decoded strings/bytes.
    is_beam_search: Whether the generation used beam search.
  """
  arr = np.asarray(decoded_text)
  print("\n--- Sample Decoded Output ---")
  if is_beam_search or (arr.ndim == 2 and arr.shape[1] > 1):
    if arr.ndim == 1:
      arr = np.expand_dims(arr, 0)
    batch_size, num_beams = arr.shape[0], arr.shape[1]
    if batch_size == 1:
      for b_idx in range(num_beams):
        print(f"  Beam {b_idx}: {_format_text(arr[0, b_idx])}")
    else:
      for batch_idx in range(batch_size):
        print(f"  [Batch {batch_idx}]:")
        for b_idx in range(num_beams):
          print(f"    Beam {b_idx}: {_format_text(arr[batch_idx, b_idx])}")
  else:
    flat = arr.flatten()
    if len(flat) == 1:
      print(f"  {_format_text(flat[0])}")
    else:
      for idx, val in enumerate(flat):
        print(f"  [Batch {idx}]: {_format_text(val)}")
  print("-----------------------------\n")


def _print_benchmark_metrics(
    latencies_ms: list[float],
    batch_size: int,
    title: str,
    new_tokens: int | None = None,
    num_candidates: int | None = None,
    candidate_tokens: int | None = None,
) -> None:
  """Prints latency and throughput metrics for a benchmark run."""
  if not latencies_ms:
    return
  mean_ms = sum(latencies_ms) / len(latencies_ms)
  min_ms = min(latencies_ms)
  max_ms = max(latencies_ms)
  p50_ms = float(np.percentile(latencies_ms, 50))
  p90_ms = float(np.percentile(latencies_ms, 90))
  p99_ms = float(np.percentile(latencies_ms, 99))
  qps = (1000.0 / mean_ms) * batch_size
  header = f"=== {title} ==="
  print(f"\n{header}")
  print(f"Mean Batch Latency      : {mean_ms:.2f} ms")
  print(f"p50 Batch Latency       : {p50_ms:.2f} ms")
  print(f"p90 Batch Latency       : {p90_ms:.2f} ms")
  print(f"p99 Batch Latency       : {p99_ms:.2f} ms")
  print(f"Min Batch Latency       : {min_ms:.2f} ms")
  print(f"Max Batch Latency       : {max_ms:.2f} ms")
  print(f"Queries Per Second (QPS): {qps:.2f} qps")
  if num_candidates is not None and num_candidates > 0:
    cand_qps = qps * num_candidates
    print(f"Candidate Scoring Rate  : {cand_qps:.2f} candidates/s")
    if candidate_tokens is not None and candidate_tokens > 0:
      token_rate = cand_qps * candidate_tokens
      print(f"Scored Token Rate       : {token_rate:.2f} tokens/s")
  elif new_tokens is not None and new_tokens > 0:
    token_qps = qps * new_tokens
    print(f"Generated Token Rate    : {token_qps:.2f} tokens/s")
  print(f"{'=' * len(header)}\n")


def _generate_candidate_texts(
    batch_size: int,
    num_candidates: int,
    num_tokens_per_candidate: int,
) -> list[list[str]]:
  """Generates synthetic candidate completions with enough words to reach target length."""
  candidates_per_batch = []
  for cand_idx in range(num_candidates):
    words = [f"candidate_{cand_idx}"] + [
        f"token_{i}" for i in range(1, num_tokens_per_candidate + 2)
    ]
    candidates_per_batch.append(" " + " ".join(words))
  return [candidates_per_batch for _ in range(batch_size)]


def _block_until_ready(tree: Any) -> Any:
  """Blocks until all JAX array leaves in a PyTree are ready."""

  def _leaf_block(x: Any) -> Any:
    if hasattr(x, "block_until_ready"):
      return x.block_until_ready()
    return x

  return jax.tree_util.tree_map(_leaf_block, tree)


def _run_model_fn_benchmark(args: argparse.Namespace) -> None:
  """Benchmarks the JIT-compiled model generation loop directly in JAX."""
  print(f"=== Running Mode: model_fn (Model={args.model}) ===")
  runner = _get_tpu_runner(args)

  sampling_method = llm_config.SamplingMethod(args.sampling_type)
  is_beam_search = sampling_method == llm_config.SamplingMethod.BEAM_SEARCH
  is_scoring = sampling_method == llm_config.SamplingMethod.SCORING
  beam_size = args.beam_width if is_beam_search else None
  num_candidates = args.num_candidates if is_scoring else None
  candidate_tokens = args.candidate_tokens if is_scoring else None

  generation_config = llm_config.GenerationConfig(
      sampling_method=sampling_method,
      max_output_tokens=args.new_tokens,
      beam_size=beam_size,
      num_candidates=num_candidates,
      max_candidate_tokens=candidate_tokens,
      max_prompt_length=args.prompt_len,
  )

  tokenizer = tokenizers_processors.KerasHubTokenizerTF(
      model_name_or_path=args.model,
      max_seq_len=args.prompt_len,
      max_candidate_tokens=candidate_tokens,
  )
  wrapper = tpu_wrapper.TPUInferenceWrapper(
      tpu_runner=runner,
      generation_config=generation_config,
  )

  prompt_text = args.prompt
  print(f"Tokenizing prompt: {prompt_text!r} (batch_size={args.batch_size})")
  prompts = [prompt_text] * args.batch_size

  if is_scoring:
    assert num_candidates is not None
    assert candidate_tokens is not None
    print(
        f"Constructing and tokenizing {num_candidates} candidate completions"
        f" ({candidate_tokens} tokens each)..."
    )
    cands = _generate_candidate_texts(
        batch_size=args.batch_size,
        num_candidates=num_candidates,
        num_tokens_per_candidate=candidate_tokens,
    )
    pre_processed = tokenizer.pre_process_fn({
        "text_input": tf.constant(prompts),
        "candidate_texts": tf.constant(cands),
    })
    inputs_jax = {
        "input_ids": jnp.array(pre_processed["input_ids"].numpy()),
        "candidate_ids": jnp.array(pre_processed["candidate_ids"].numpy()),
    }
  else:
    pre_processed = tokenizer.pre_process_fn(
        {"text_input": tf.constant(prompts)}
    )
    inputs_jax = {
        "input_ids": jnp.array(pre_processed["input_ids"].numpy()),
    }

  print("Compiling JAX generation loop...")
  with runner.mesh:
    wrapper.generation_config = dataclasses.replace(
        wrapper.generation_config,
        pad_token_id=tokenizer.pad_token_id,
    )
    model_fn = wrapper.compile_generation_loop()
    jit_model_fn = jax.jit(model_fn)

    print("Running Warmup...")
    start_warmup = time.perf_counter()
    outputs = jit_model_fn(runner.state, inputs_jax)
    _block_until_ready(outputs)
    print(f"Warmup completed in {time.perf_counter() - start_warmup:.2f} s.")

    if is_scoring:
      scores = np.array(outputs["scores"])
      print(f"Computed candidate scores shape: {scores.shape}")
      print(f"Sample candidate scores (first 5): {scores[0, :5]}")
    else:
      post_processed = tokenizer.post_process_fn(
          {"token_ids": tf.constant(np.array(outputs["token_ids"]))}
      )
      decoded_text = post_processed["text_output"].numpy()
      _print_decoded_outputs(decoded_text, is_beam_search=is_beam_search)

    print(
        f"Running benchmark loop ({args.num_batches} iterations,"
        f" batch_size={args.batch_size})..."
    )
    latencies_ms = []
    for i in range(args.num_batches):
      t0 = time.perf_counter()
      res = jit_model_fn(runner.state, inputs_jax)
      _block_until_ready(res)
      dt = (time.perf_counter() - t0) * 1000.0
      latencies_ms.append(dt)
      print(f"  Iteration {i+1}/{args.num_batches}: {dt:.2f} ms")

    if is_scoring:
      _print_benchmark_metrics(
          latencies_ms=latencies_ms,
          batch_size=args.batch_size,
          title="Candidate Scoring model_fn Benchmark Results",
          num_candidates=num_candidates,
          candidate_tokens=candidate_tokens,
      )
    else:
      _print_benchmark_metrics(
          latencies_ms=latencies_ms,
          batch_size=args.batch_size,
          title="model_fn Benchmark Results",
          new_tokens=args.new_tokens,
      )


def _run_grpc_client_benchmark(args: argparse.Namespace) -> None:
  """Benchmarks an existing C++ XLA SavedModel Server via gRPC."""
  print(f"=== Running Mode: grpc_client (Server={args.server_address}) ===")
  channel = grpc.insecure_channel(args.server_address)
  predict_method = channel.unary_unary(
      _PREDICT_METHOD,
      request_serializer=predict_pb2.PredictRequest.SerializeToString,
      response_deserializer=predict_pb2.PredictResponse.FromString,
  )

  request = predict_pb2.PredictRequest()
  request.model_spec.name = args.model_name
  request.model_spec.signature_name = args.signature_name

  hb_proto = host_buffer_pb2.HostBufferProto()
  hb_proto.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
  hb_proto.shape.dims.append(args.batch_size)

  single_prompt = args.prompt
  for _ in range(args.batch_size):
    hb_proto.string_data.elements.append(single_prompt.encode("utf-8"))

  request.inputs["text_input"].CopyFrom(hb_proto)

  is_scoring = args.sampling_type == "scoring"
  if is_scoring:
    cands = _generate_candidate_texts(
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        num_tokens_per_candidate=args.candidate_tokens,
    )
    cand_proto = host_buffer_pb2.HostBufferProto()
    cand_proto.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
    cand_proto.shape.dims.extend([args.batch_size, args.num_candidates])
    for batch_cands in cands:
      for cand_str in batch_cands:
        cand_proto.string_data.elements.append(cand_str.encode("utf-8"))
    request.inputs["candidate_texts"].CopyFrom(cand_proto)

  print("Running Warmup gRPC request...")
  start_warmup = time.perf_counter()
  response = predict_method(request)
  print(f"Warmup completed in {time.perf_counter() - start_warmup:.2f} s.")

  for name, output_buffer in response.outputs.items():
    dims = list(output_buffer.shape.dims)
    print(f"Output Name: {name}, Shape: {dims}")
    if output_buffer.dtype.kind == dtype_pb2.DTypeProto.Kind.Value(
        "KIND_STRING"
    ):
      elements = [
          raw_bytes.decode("utf-8", errors="replace")
          for raw_bytes in output_buffer.string_data.elements
      ]
      arr = np.array(elements, dtype=object)
      if dims:
        try:
          arr = arr.reshape(dims)
        except ValueError:
          pass
      is_beam = len(dims) == 2 and dims[1] > 1
      _print_decoded_outputs(arr, is_beam_search=is_beam)
    elif output_buffer.dtype.kind in (
        dtype_pb2.DTypeProto.Kind.Value("KIND_F32"),
        dtype_pb2.DTypeProto.Kind.Value("KIND_F64"),
    ):
      if output_buffer.dtype.kind == dtype_pb2.DTypeProto.Kind.Value(
          "KIND_F32"
      ):
        dtype = np.float32
      else:
        dtype = np.float64
      arr = np.frombuffer(output_buffer.raw_data, dtype=dtype)
      if dims:
        try:
          arr = arr.reshape(dims)
        except ValueError:
          pass
      print(f"Sample candidate scores (first 5): {arr[0, :5]}")

  print(
      f"Running gRPC benchmark loop ({args.num_batches} batches,"
      f" batch_size={args.batch_size})..."
  )
  latencies_ms = []
  for i in range(args.num_batches):
    t0 = time.perf_counter()
    _ = predict_method(request)
    dt = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(dt)
    print(f"  Batch {i+1}/{args.num_batches}: {dt:.2f} ms")

  if is_scoring:
    _print_benchmark_metrics(
        latencies_ms=latencies_ms,
        batch_size=args.batch_size,
        title="Candidate Scoring grpc_client Benchmark Results",
        num_candidates=args.num_candidates,
        candidate_tokens=args.candidate_tokens,
    )
  else:
    _print_benchmark_metrics(
        latencies_ms=latencies_ms,
        batch_size=args.batch_size,
        title="grpc_client Benchmark Results",
        new_tokens=args.new_tokens,
    )


def _run_model_runner_benchmark(args: argparse.Namespace) -> None:
  """Benchmarks the exported model using XSE's C++ ModelRunner Python bindings."""
  print(f"=== Running Mode: model_runner (Path={args.model_path}) ===")
  if not args.model_path:
    raise ValueError("--model-path must be specified for model_runner mode.")
  if model_runner is None:
    raise RuntimeError(
        "model_runner Python bindings could not be imported. Ensure the C++"
        " runtime library is built and available."
    )

  print(f"Loading ModelRunner for model path: {args.model_path}...")
  runner = model_runner.ModelRunner(
      model_name=args.model_name,
      model_path=args.model_path,
  )
  sig = runner.signature(args.signature_name)

  prompt_text = args.prompt
  prompts = [prompt_text] * args.batch_size
  inputs = {
      "text_input": np.array(prompts, dtype=object),
  }

  is_scoring = args.sampling_type == "scoring"
  if is_scoring:
    cands = _generate_candidate_texts(
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        num_tokens_per_candidate=args.candidate_tokens,
    )
    inputs["candidate_texts"] = np.array(cands, dtype=object)

  print("Running Warmup ModelRunner request...")
  start_warmup = time.perf_counter()
  outputs = sig.run_with_dict_return(**inputs)
  print(f"Warmup completed in {time.perf_counter() - start_warmup:.2f} s.")

  if "scores" in outputs:
    scores = np.array(outputs["scores"])
    print(f"Output Name: scores, Shape: {scores.shape}")
    print(f"Sample candidate scores (first 5): {scores[0, :5]}")
  elif "text_output" in outputs:
    out_arr = outputs["text_output"]
    print(f"Output Name: text_output, Shape: {out_arr.shape}")
    _print_decoded_outputs(out_arr)

  print(
      f"Running ModelRunner benchmark loop ({args.num_batches} batches,"
      f" batch_size={args.batch_size})..."
  )
  latencies_ms = []
  for b in range(args.num_batches):
    t0 = time.perf_counter()
    _ = sig.run_with_dict_return(**inputs)
    batch_lat_ms = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(batch_lat_ms)
    print(f"  Batch {b + 1}/{args.num_batches}: {batch_lat_ms:.2f} ms")

  if is_scoring:
    _print_benchmark_metrics(
        latencies_ms=latencies_ms,
        batch_size=args.batch_size,
        title="Candidate Scoring model_runner Benchmark Results",
        num_candidates=args.num_candidates,
        candidate_tokens=args.candidate_tokens,
    )
  else:
    _print_benchmark_metrics(
        latencies_ms=latencies_ms,
        batch_size=args.batch_size,
        title="model_runner Benchmark Results",
        new_tokens=args.new_tokens,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments for LLM benchmarking."""
  parser = argparse.ArgumentParser(
      description="Consolidated LLM Benchmarking CLI for XSE."
  )
  parser.add_argument(
      "--mode",
      type=str,
      required=True,
      choices=["model_fn", "grpc_client", "model_runner"],
      help="Benchmarking mode to execute.",
  )
  # Model_fn specific args
  parser.add_argument(
      "--model",
      type=str,
      default="Qwen/Qwen3-0.6B",
      help="Hugging Face model ID or path (for model_fn mode).",
  )
  parser.add_argument(
      "--load-format",
      type=str,
      default="auto",
      help=(
          "Model weight loading format (default: 'auto'). Can be set to"
          " 'jax_dummy' for testing with dummy weights."
      ),
  )
  parser.add_argument(
      "--max-seq-len",
      type=int,
      default=None,
      help=(
          "Maximum total sequence length (prompt + completion). If omitted, "
          "defaults to prompt-len + new-tokens."
      ),
  )
  parser.add_argument(
      "--prompt-len",
      type=int,
      default=128,
      help="Maximum prompt token length (default: 128).",
  )
  parser.add_argument(
      "--new-tokens",
      type=int,
      default=20,
      help="Number of tokens to generate (default: 20).",
  )
  parser.add_argument(
      "--sampling-type",
      type=str,
      default="beam_search",
      choices=[m.value for m in llm_config.SamplingMethod],
      help="Sampling method to compile.",
  )
  parser.add_argument(
      "--beam-width",
      type=int,
      default=4,
      help="Beam search width (default: 4).",
  )
  parser.add_argument(
      "--num-candidates",
      type=int,
      default=500,
      help="Number of candidate suffixes for scoring (default: 500).",
  )
  parser.add_argument(
      "--candidate-tokens",
      type=int,
      default=10,
      help="Token length of each candidate suffix for scoring (default: 10).",
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
  # Model_runner specific args
  parser.add_argument(
      "--model-path",
      type=str,
      default=None,
      help="Path to exported SavedModel directory for model_runner mode.",
  )
  # Common & grpc_client args
  parser.add_argument(
      "--server-address",
      type=str,
      default="localhost:8500",
      help="Target server address for grpc_client mode.",
  )
  parser.add_argument(
      "--model-name",
      type=str,
      default="qwen3",
      help="Model spec name in the gRPC server.",
  )
  parser.add_argument(
      "--signature-name",
      type=str,
      default="serving_default",
      help="Signature name to query in the gRPC server.",
  )
  parser.add_argument(
      "--prompt",
      type=str,
      required=True,
      help="Prompt string to send for benchmarking.",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=1,
      help="Number of prompts per batch.",
  )
  parser.add_argument(
      "--num-batches",
      type=int,
      default=3,
      help="Number of timed test iterations to run.",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.batch_size <= 0:
    raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
  if args.num_batches <= 0:
    raise ValueError(f"--num-batches must be positive, got {args.num_batches}")
  if args.beam_width <= 0:
    raise ValueError(f"--beam-width must be positive, got {args.beam_width}")
  if args.num_candidates <= 0:
    raise ValueError(
        f"--num-candidates must be positive, got {args.num_candidates}"
    )
  if args.candidate_tokens <= 0:
    raise ValueError(
        f"--candidate-tokens must be positive, got {args.candidate_tokens}"
    )
  if args.prompt_len <= 0:
    raise ValueError(f"--prompt-len must be positive, got {args.prompt_len}")
  if args.new_tokens <= 0:
    raise ValueError(f"--new-tokens must be positive, got {args.new_tokens}")

  if args.max_seq_len is None:
    args.max_seq_len = args.prompt_len + args.new_tokens
  elif args.max_seq_len < args.prompt_len + args.new_tokens:
    raise ValueError(
        f"max_seq_len ({args.max_seq_len}) must be at least prompt_len"
        f" ({args.prompt_len}) + new_tokens ({args.new_tokens})"
    )

  if args.mode == "model_fn":
    _run_model_fn_benchmark(args)
  elif args.mode == "grpc_client":
    _run_grpc_client_benchmark(args)
  elif args.mode == "model_runner":
    _run_model_runner_benchmark(args)


if __name__ == "__main__":
  main(sys.argv[1:])
