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

"""Pre-processing and Post-processing tokenizers for LLM serving."""

import json
from typing import Any
import keras_hub
import tensorflow as tf
import transformers

AutoTokenizer = transformers.AutoTokenizer


def truncate_at_eos(token_ids: tf.Tensor, eos_token_id: int) -> tf.Tensor:
  """Truncates token IDs at the first EOS token index.

  Args:
    token_ids: Input tensor of token IDs.
    eos_token_id: Integer ID representing end of sequence.

  Returns:
    RaggedTensor of truncated token IDs.
  """
  is_eos = tf.equal(token_ids, eos_token_id)
  eos_cum = tf.cumsum(tf.cast(is_eos, tf.int32), axis=-1)
  keep_mask = tf.equal(eos_cum, 0)
  return tf.ragged.boolean_mask(token_ids, keep_mask)


def _resolve_tokenizer_class(hf_class_name: str, model_name_or_path: str):
  """Resolves the Keras Hub tokenizer class from the HF class or model path.

  Args:
    hf_class_name: Name of the Hugging Face tokenizer class.
    model_name_or_path: Hugging Face model identifier or file path.

  Returns:
    The corresponding Keras Hub tokenizer class.
  """
  # Normalize inputs
  norm_class_name = hf_class_name.replace("Fast", "").lower()
  model_name_lower = model_name_or_path.lower()

  # Map normalized Hugging Face tokenizer classes to Keras Hub implementations
  tokenizer_mapping = {
      "qwen3tokenizer": keras_hub.tokenizers.Qwen3Tokenizer,
      "qwen2tokenizer": getattr(
          keras_hub.tokenizers,
          "Qwen2Tokenizer",
          keras_hub.tokenizers.Qwen3Tokenizer,
      ),
      "llama3tokenizer": keras_hub.tokenizers.Llama3Tokenizer,
      "llamatokenizer": keras_hub.tokenizers.LlamaTokenizer,
      "gpt2tokenizer": keras_hub.tokenizers.GPT2Tokenizer,
  }

  if norm_class_name in tokenizer_mapping:
    return tokenizer_mapping[norm_class_name]

  if "qwen3" in norm_class_name or "qwen3" in model_name_lower:
    return keras_hub.tokenizers.Qwen3Tokenizer
  elif (
      "qwen2" in norm_class_name
      or "qwen" in norm_class_name
      or "qwen" in model_name_lower
  ):
    return getattr(
        keras_hub.tokenizers,
        "Qwen2Tokenizer",
        keras_hub.tokenizers.Qwen3Tokenizer,
    )
  elif (
      "llama3" in norm_class_name
      or "llama-3" in model_name_lower
      or "llama3" in model_name_lower
  ):
    return keras_hub.tokenizers.Llama3Tokenizer
  elif "llama" in norm_class_name or "llama" in model_name_lower:
    return keras_hub.tokenizers.LlamaTokenizer
  elif "gpt2" in norm_class_name or "gpt2" in model_name_lower:
    return keras_hub.tokenizers.GPT2Tokenizer

  print(
      f"Fallback to generic BytePairTokenizer for class {hf_class_name} / model"
      f" {model_name_or_path}"
  )
  return keras_hub.tokenizers.BytePairTokenizer


class KerasHubTokenizerTF:
  """A graph-compatible pre-processor and post-processor wrapping Keras Hub."""

  def __init__(
      self,
      model_name_or_path: str,
      eos_token_id: int = 2,
      pad_token_id: int = 0,
      max_seq_len: int = 64,
  ):
    """Initializes the Keras Hub TF tokenizer.

    Args:
      model_name_or_path: Hugging Face model identifier or directory path.
      eos_token_id: End of sequence token ID.
      pad_token_id: Padding token ID.
      max_seq_len: Maximum sequence length to pad or slice to.
    """
    self.model_name_or_path = model_name_or_path
    self.eos_token_id = eos_token_id
    self.pad_token_id = pad_token_id
    self.max_seq_len = max_seq_len

    # Load Hugging Face tokenizer to extract vocab and merges
    print(f"Loading Hugging Face tokenizer for {model_name_or_path}...")
    real_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    vocab = real_tokenizer.get_vocab()
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

    tokenizer_data = json.loads(real_tokenizer._tokenizer.to_str())
    merges = tokenizer_data.get("model", {}).get("merges", [])
    formatted_merges = [" ".join(pair) for pair in merges]

    hf_class_name = real_tokenizer.__class__.__name__
    print(f"Detected Hugging Face tokenizer class: {hf_class_name}")
    tokenizer_cls = _resolve_tokenizer_class(hf_class_name, model_name_or_path)
    print(f"Selected Keras Hub tokenizer class: {tokenizer_cls.__name__}")

    self.tokenizer = tokenizer_cls(
        vocabulary=vocab,
        merges=formatted_merges,
    )

  def pre_process_fn(
      self, inputs: dict[str, tf.Tensor]
  ) -> dict[str, tf.Tensor]:
    """Tokenizes text input inside TF graph.

    Args:
      inputs: Dictionary containing "text_input" string tensor.

    Returns:
      Dictionary containing "input_ids" integer tensor.
    """
    text_input = inputs["text_input"]

    ragged_tokens = self.tokenizer(text_input)
    if not isinstance(ragged_tokens, tf.RaggedTensor):
      ragged_tokens = tf.ragged.constant(ragged_tokens, dtype=tf.int32)

    input_ids = ragged_tokens.to_tensor(default_value=self.pad_token_id)

    # Pad/slice input_ids to exactly max_seq_len
    paddings = [[0, 0], [0, self.max_seq_len]]
    padded = tf.pad(input_ids, paddings, constant_values=self.pad_token_id)
    input_ids_fixed = tf.slice(padded, [0, 0], [-1, self.max_seq_len])

    if text_input.shape[0] is not None:
      input_ids_fixed.set_shape([text_input.shape[0], self.max_seq_len])

    return {
        "input_ids": tf.cast(input_ids_fixed, tf.int32),
    }

  def post_process_fn(
      self, inputs: dict[str, tf.Tensor]
  ) -> dict[str, tf.Tensor]:
    """Detokenizes token ids back to string inside TF graph.

    Args:
      inputs: Dictionary containing "token_ids" integer tensor.

    Returns:
      Dictionary containing "text_output" string tensor.
    """
    token_ids = inputs["token_ids"]

    ndims = token_ids.shape.ndims
    if ndims == 3:
      # Beam search output shape: (batch_size, beam_width, max_new_tokens)
      batch_size = tf.shape(token_ids)[0]
      beam_width = tf.shape(token_ids)[1]
      max_new_tokens = tf.shape(token_ids)[2]

      flat_ids = tf.reshape(token_ids, [-1, max_new_tokens])
      truncated_ids = truncate_at_eos(flat_ids, self.eos_token_id)
      flat_text = self.tokenizer.detokenize(truncated_ids)
      text_output = tf.reshape(flat_text, [batch_size, beam_width])
    else:
      truncated_ids = truncate_at_eos(token_ids, self.eos_token_id)
      text_output = self.tokenizer.detokenize(truncated_ids)

    if not isinstance(text_output, tf.Tensor):
      text_output = tf.constant(text_output)
    return {"text_output": text_output}

  def get_trackable_resources(self) -> list[Any]:
    """Returns trackable resources for Keras Hub tokenizer."""
    resources = []
    t = self.tokenizer
    if hasattr(t, "token_to_id_map"):
      resources.append(t.token_to_id_map)
    if hasattr(t, "id_to_token_map"):
      resources.append(t.id_to_token_map)
    if hasattr(t, "unicode2byte"):
      resources.append(t.unicode2byte)
    if hasattr(t, "byte2unicode"):
      resources.append(t.byte2unicode)
    if hasattr(t, "merge_ranks"):
      resources.append(t.merge_ranks)
    if hasattr(t, "cache"):
      resources.append(t.cache)
    return resources
